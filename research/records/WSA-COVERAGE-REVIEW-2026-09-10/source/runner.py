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

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    load_upstream()
    import jlens.hf as upstream_hf
    from jlens.hooks import ActivationRecorder

    wrapped = CorpusLensModel(upstream_hf.HFLensModel(model, tokenizer=None))
    n_layers, d_model = int(wrapped.n_layers), int(wrapped.d_model)
    every = list(range(n_layers))
    bos = tokenizer.bos_token_id
    emit("loaded", dtype=observed, n_layers=n_layers, d_model=d_model, entry=args.entry,
         checkpoint_sha256=report.get("sha256", {}).get("config.json"), bos_token_id=bos)

    def forward(row: dict) -> dict:
        ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if bos is not None and len(ids) > 1 and ids[0] == bos and ids[1] == bos:
            raise SystemExit(
                "the tokenized prompt begins with two beginning-of-sequence tokens: the rendered "
                "prompt carries its own and the tokenizer added another. Every position is shifted "
                "by one and the decision position is the last one, so the capture would be off by "
                "one exactly where it matters."
            )
        tensor = torch.as_tensor(ids, dtype=torch.int64, device="cuda:0").reshape(1, -1)
        with torch.no_grad():
            with ActivationRecorder(wrapped.layers, at=every) as recorder:
                wrapped.forward(tensor)
            position = tensor.shape[-1] - 1
            residuals = torch.stack(
                [recorder.activations[i][0, position].detach().clone() for i in every]
            )
        return {
            "residuals": residuals.cpu(), "seq_len": int(tensor.shape[-1]),
            "token_index": int(position), "layers": n_layers, "d_model": d_model,
            "device": str(residuals.device), "dtype": str(residuals.dtype),
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
    emit("inputs", decisions=len(decisions), capture_set_size=capture_set_size,
         limit=args.limit, whole_set=len(decisions) == capture_set_size,
         capture_set=str(args.capture_set), corpus_rows=len(corpus_rows), keyed=len(corpus))

    target = capture.CaptureTarget(
        directory=args.out, entry=args.entry,
        checkpoint_sha256=report.get("sha256", {}).get("config.json") or "",
        decoding=args.decoding, shard_size=args.shard_size,
    )
    summary = capture.capture_decisions(
        decisions=decisions, corpus=corpus, forward=forward, target=target, progress=emit_shard(emit)
    )
    emit("done", **summary, capture_set_size=capture_set_size,
         whole_set=summary["requested"] == capture_set_size,
         peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3))
    return 0


def emit_shard(emit):
    def progress(row: dict) -> None:
        emit(row.pop("event", "shard"), **row)
    return progress


if __name__ == "__main__":
    raise SystemExit(main())
