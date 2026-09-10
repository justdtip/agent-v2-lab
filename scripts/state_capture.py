"""Run the plan-progress capture pass: one decision position per decision, every layer, width one.

The contract lives in `state_programme/capture.py` and is enforced there. This is the seam that
supplies it a model: it tokenizes the rendered prompt, forwards at batch one, and hands back every
layer's residual at the decision position.

**The rendered prompt is already a complete chat string** — it carries its own `<bos>` and its own
`<start_of_turn>model` — so it is tokenized with `add_special_tokens=False`. A tokenizer that adds a
second beginning-of-sequence token shifts every position by one, and the position this pass exists to
capture is the last one, so the whole capture would be off by one at exactly the place that matters
and nothing downstream would say so. The doubling is checked rather than assumed, per row.

    python scripts/state_capture.py --checkpoint DIR --capture-set FILE --corpus DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--capture-set", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True, help="the rendered corpus directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--entry", required=True, help="the registry entry this capture is of")
    parser.add_argument("--decoding", default="teacher-forced")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many decisions")
    args = parser.parse_args(argv)

    started = time.monotonic()

    def emit(event: str, /, **fields) -> None:
        row = {"event": event, "elapsed_s": round(time.monotonic() - started, 2), **fields}
        print(json.dumps(row, default=str), flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=str) + "\n")
            stream.flush()

    from local_llm_lab import device

    device.pin(seed=0)

    import torch
    from transformers import AutoTokenizer

    from local_llm_lab import hf_text
    from local_llm_lab.pipeline.lens_fitting.upstream import CorpusLensModel, load_upstream
    from local_llm_lab.pipeline.state_programme import capture

    model, report = hf_text.load_text_causal_lm(
        args.checkpoint, dtype="bfloat16", attn_implementation="eager", device="cuda:0"
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    # Native, and it stays native: the capture records the deployed computation, and a promoted
    # capture is `residual_precision_probe`'s measurement and not this one.
    observed = str(next(model.parameters()).dtype)
    if observed != "torch.bfloat16":
        raise SystemExit(f"the model loaded as {observed}; the capture contract is native bf16")

    # The whole digest manifest, never one file's (Codex F2): two checkpoints sharing a config and
    # differing in every weight would otherwise carry the same identity into every captured cell.
    identity = capture.checkpoint_identity(report.get("sha256"))

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    load_upstream()
    import jlens.hf as upstream_hf
    from jlens.hooks import ActivationRecorder

    wrapped = CorpusLensModel(upstream_hf.HFLensModel(model, tokenizer=None))
    n_layers, d_model = int(wrapped.n_layers), int(wrapped.d_model)
    every = list(range(n_layers))
    bos = tokenizer.bos_token_id
    emit("loaded", dtype=observed, n_layers=n_layers, d_model=d_model, entry=args.entry,
         checkpoint_sha256=identity["checkpoint_sha256"],
         config_sha256=identity["config_sha256"], weight_files=identity["weight_files"],
         bos_token_id=bos)

    def prepare(row: dict) -> dict:
        """The exact model input, computed **before** the reuse decision.

        Resume cannot otherwise check that this run's tokenizer produces the ids the existing cells
        record, and two tokenizers give different ids for the same bytes while the checkpoint
        identity covers no tokenizer asset (Codex R2).
        """
        import hashlib

        ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if bos is not None and len(ids) > 1 and ids[0] == bos and ids[1] == bos:
            raise SystemExit(
                "the tokenized prompt begins with two beginning-of-sequence tokens: the rendered "
                "prompt carries its own and the tokenizer added another. Every position is shifted "
                "by one and the decision position is the last one, so the capture would be off by "
                "one exactly where it matters."
            )
        return {
            "token_ids": ids,
            "token_ids_sha256": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest(),
            "token_ids_length": len(ids),
            "rendered_prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
        }

    def forward(row: dict, prepared: dict) -> dict:
        """Run the pass and **attest to the pass that ran**, not to the pass that was intended.

        Every field returned here is read back from the objects that did the work — the tensor's own
        leading dimension, the residual's own dtype, the digest of the bytes handed to the tokenizer
        and of the ids it returned. The writer compares these to the contract and refuses on a gap.
        It used to stamp the contract's constants into the cell itself, which meant a seam that
        expanded a batch or promoted its arithmetic was recorded as width 1, native (Codex C1).
        """
        # The ids `prepare` already produced, so the pass forwards exactly what was compared
        # against the existing cells rather than tokenizing a second time and hoping.
        ids = prepared["token_ids"]
        tensor = torch.as_tensor(ids, dtype=torch.int64, device="cuda:0").reshape(1, -1)
        with torch.no_grad():
            with ActivationRecorder(wrapped.layers, at=every) as recorder:
                wrapped.forward(tensor)
            position = tensor.shape[-1] - 1
            residuals = torch.stack(
                [recorder.activations[i][0, position].detach().clone() for i in every]
            )
        moved = residuals.cpu()
        return {
            "residuals": moved,
            "seq_len": int(tensor.shape[-1]),
            "token_index": int(position),
            "layers": n_layers,
            "d_model": d_model,
            "device": str(residuals.device),
            "dtype": str(moved.dtype),
            # Observed, never declared: the batch this forward actually ran at is the tensor's own
            # leading dimension, and the arithmetic path is native exactly when the residual came
            # back in the model's own dtype rather than a promoted one.
            "forward_batch": int(tensor.shape[0]),
            "anchor_batch": int(tensor.shape[0]),
            "capture_dtype": "native" if str(moved.dtype) == observed else "promoted-float32",
            "rendered_prompt_sha256": prepared["rendered_prompt_sha256"],
            "token_ids_sha256": prepared["token_ids_sha256"],
            "token_ids_length": prepared["token_ids_length"],
        }

    decisions = [json.loads(line) for line in args.capture_set.read_text().splitlines() if line.strip()]
    # The size of the set before any truncation, kept so the run's own record can distinguish a
    # complete pass from a limited one. `capture_decisions` reports completeness against what it was
    # given, which is the right scope for it and the wrong scope for the pass as a whole.
    capture_set_size = len(decisions)
    if args.limit is not None:
        decisions = decisions[: args.limit]
    corpus_rows = []
    for split in ("train", "valid", "test"):
        path = args.corpus / f"{split}.jsonl"
        corpus_rows.extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    corpus = capture.rows_by_decision(corpus_rows)
    # The allocator's configuration is recorded because the run was launched under it. It changes
    # segment strategy and not kernels, so it is not expected to change any number here — but "not
    # expected to" is why it belongs in the manifest rather than in a message: if a later pass
    # disagrees with this one, the first question is what differed, and an environment variable
    # nobody wrote down is the answer nobody finds.
    emit("allocator", pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
         cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
         cwd=str(Path.cwd()))
    emit("loaded_peak", **memory(torch),
         note="weights only; the first shard adds one forward at the longest row seen. This pass "
              "runs the decoder stack alone — `HFLensModel.forward` calls the text module, not the "
              "causal LM — so it has no vocabulary-sized logits allocation and no spike on long "
              "rows, which is the allocation that ended the Chief's 4B capture at 03:22Z")
    emit("inputs", decisions=len(decisions), capture_set_size=capture_set_size,
         limit=args.limit, whole_set=len(decisions) == capture_set_size,
         capture_set=str(args.capture_set), corpus_rows=len(corpus_rows), keyed=len(corpus))

    target = capture.CaptureTarget(
        directory=args.out, entry=args.entry, identity=identity,
        decoding=args.decoding, shard_size=args.shard_size,
    )
    (args.out / "run.json").write_text(json.dumps({
        "entry": args.entry,
        "checkpoint": args.checkpoint,
        **identity,
        "capture_set": str(args.capture_set),
        "capture_set_size": capture_set_size,
        "limit": args.limit,
        "corpus": str(args.corpus),
        "decoding": args.decoding,
        "shard_size": args.shard_size,
        "device": device.describe(),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cwd": str(Path.cwd()),
        "bos_token_id": bos,
        "basis": "measured-here",
    }, indent=2, sort_keys=True, default=str) + "\n")
    summary = capture.capture_decisions(
        decisions=decisions, corpus=corpus, forward=forward, prepare=prepare, target=target,
        progress=emit_shard(emit, torch),
    )
    emit("done", **summary, capture_set_size=capture_set_size,
         whole_set=summary["requested"] == capture_set_size, **memory(torch))
    return 0


def memory(torch) -> dict:
    """What this pass costs the card, in the terms a shared card is planned in.

    `max_memory_allocated` is the allocator's high-water mark for **tensors**, and it is not what
    another seat needs. A process also holds allocator segments it has not handed out and a CUDA
    context on top of that: the Chief's capture measured 29.30 GiB of process memory against 22.12
    allocated, with 6.52 reserved-unallocated, and c3 showed 62.25 process against 59.72 allocated —
    a 2.5 GiB gap that decided an out-of-memory. So the headroom rule is read on **process memory**,
    and the closest in-process proxy is peak *reserved*, with the device's own free/total beside it
    as the figure that needs no proxy at all.
    """
    free, total = torch.cuda.mem_get_info()
    return {
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "device_used_gib": round((total - free) / 2**30, 3),
        "device_free_gib": round(free / 2**30, 3),
    }


def emit_shard(emit, torch):
    """Relay the writer's progress, with what the pass costs the card beside it.

    The figures belong on the **first** shard and not only at the end. Two passes share the card and
    whether the second may start depends on the first's measured cost, so a number that arrives with
    the summary arrives after the decision it informs.
    """
    def progress(row: dict) -> None:
        emit(row.pop("event", "shard"), **memory(torch), **row)
    return progress


if __name__ == "__main__":
    raise SystemExit(main())
