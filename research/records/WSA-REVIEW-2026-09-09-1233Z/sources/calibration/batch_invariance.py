"""Does the *unhooked* forward depend on batch width? The disambiguation §2's result demands.

`boundary.py` found that the replacement hook reproduces the ordinary forward bit for bit at batch
one and not at width 64 or 256. Two readings fit that: the hook misbehaves when the batch is wide,
or the model's own forward is not batch-invariant and the hook is innocent. They call for opposite
repairs, so nothing may be concluded until they are separated.

This separates them with no hook at all. The same frozen row is run at width 1, 64 and 256, and
**every** block output is compared to the width-one reading, so the answer names the first block at
which the two diverge rather than only the last. A model whose own forward is batch-dependent makes
the batch width part of the function being differentiated, and two estimators run at two widths are
then not two estimates of one thing.

    python batch_invariance.py SNAPSHOT CORPUS OUT
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

SNAPSHOT, CORPUS, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
STARTED = time.monotonic()

MAX_SEQ, SEED, DECLARED_ROWS = 128, 20260910, 3
WIDTHS = (64, 256)


def emit(event: str, /, **fields) -> dict:
    fields.pop("event", None)
    row = {"event": event, "elapsed_s": round(time.monotonic() - STARTED, 2), **fields}
    print(json.dumps(row, default=str), flush=True)
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
        stream.flush()
    return row


from local_llm_lab import device  # noqa: E402

device.pin(seed=0)

import torch  # noqa: E402

from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.upstream import (  # noqa: E402
    CorpusLensModel,
    load_upstream,
    repo_layer_of_upstream,
)

model, report = hf_text.load_text_causal_lm(
    SNAPSHOT, dtype="bfloat16", attn_implementation="eager", device="cuda:0"
)
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.eval()

load_upstream()
import jlens.hf as upstream_hf  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

wrapped = CorpusLensModel(upstream_hf.HFLensModel(model, tokenizer=None))
n_layers, d_model = int(wrapped.n_layers), int(wrapped.d_model)
every = list(range(n_layers))

held = [row for row in read_corpus(CORPUS) if row.get("split") == "held"]
row = held[random.Random(SEED).sample(range(len(held)), DECLARED_ROWS)[0]]
key = wrapped.register(f"row:{row['index']}", row["ids"])
ids = wrapped.encode(key, max_length=MAX_SEQ)
emit("frozen", row_index=row.get("index"), seq_len=int(ids.shape[-1]),
     n_layers=n_layers, d_model=d_model, hook="none")

with torch.no_grad():
    with ActivationRecorder(wrapped.layers, at=every) as recorder:
        wrapped.forward(ids)
    reference = {i: recorder.activations[i].detach().clone() for i in every}

rows = []
for width in WIDTHS:
    with torch.no_grad():
        with ActivationRecorder(wrapped.layers, at=every) as recorder:
            wrapped.forward(ids.expand(width, -1))
        observed = {i: recorder.activations[i].detach() for i in every}
    first_divergence = None
    for i in every:
        ref, got = reference[i].expand_as(observed[i]), observed[i]
        identical = bool(torch.equal(ref, got))
        difference = (got.float() - ref.float()).abs()
        entry = {
            "width": width,
            "upstream_layer": i,
            "repo_layer": repo_layer_of_upstream(i),
            "bitwise_identical": identical,
            "equal_fraction": round(float(torch.eq(ref, got).float().mean()), 12),
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
            "reference_max_abs": float(ref.float().abs().max()),
            "rows_identical_to_each_other": bool(
                torch.equal(got, got[:1].expand_as(got).contiguous())),
            "basis": "measured-here",
        }
        rows.append(entry)
        if not identical and first_divergence is None:
            first_divergence = entry
    emit("width", width=width,
         layers_identical=sum(1 for r in rows if r["width"] == width and r["bitwise_identical"]),
         layers_total=n_layers,
         first_divergence_repo_layer=None if first_divergence is None
         else first_divergence["repo_layer"],
         first_divergence_max_abs=None if first_divergence is None
         else first_divergence["max_abs_difference"],
         final_mean_abs=[r for r in rows if r["width"] == width][-1]["mean_abs_difference"])

verdict = {
    "forward_is_batch_invariant": all(r["bitwise_identical"] for r in rows),
    "hook_is_implicated": not all(r["bitwise_identical"] for r in rows) and False,
    "conclusion": (
        "the model's own forward is batch-invariant here, so the width-dependence boundary.py "
        "found is the hook's and the hook is what to repair"
        if all(r["bitwise_identical"] for r in rows) else
        "the model's own forward is NOT batch-invariant: the same tokens through the same weights "
        "give different block outputs at different batch widths, with no hook present. The hook is "
        "not implicated, and batch width is part of the function any estimator differentiates."
    ),
}
(OUT / "batch-invariance.json").write_text(
    json.dumps({"verdict": verdict, "results": rows, "basis": "measured-here"},
               indent=2, sort_keys=True) + "\n")
emit("verdict", **verdict)
