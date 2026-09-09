"""Why the finite-difference operand does not reproduce the exact one on the real 4B.

The golden run measured a worst relative difference of 1.0245 with a per-layer cosine that climbs
monotonically from **0.015 at layer 1** to 0.825 at layer 33, and halving the step divided the
residual by **1.15** where the fixture gave 3.9. A central difference whose error does not fall as
the step falls is not truncation-limited, and a cosine of 0.015 is not a noisy estimate of a map,
it is noise. So this asks the two questions that separate the candidate causes, at the layer where
the failure is worst and at the layer where it is mildest:

* **the step**: sweep epsilon over four decades. Truncation error falls with the step and roundoff
  rises, so a usable estimator has a minimum. If the residual is flat or monotone in the step there
  is no window in which this estimator works on this model.
* **the arithmetic**: run each step in the model's own bf16 and in promoted float32. The promoted
  path is a *different arithmetic path* and is refused for a golden comparison for that reason —
  but it is kept precisely so it can be measured against, and this is the measurement. If float32
  recovers the exact map and bf16 does not, the cause is cancellation in the difference of two
  nearly-equal forwards, and the estimator is unusable at the precision the programme runs at.

Every row is written as it completes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
STARTED = time.monotonic()


def emit(event: str, /, **fields) -> None:
    fields.pop("event", None)
    row = {"event": event, "elapsed_s": round(time.monotonic() - STARTED, 2), **fields}
    print(json.dumps(row, default=str), flush=True)
    with (OUT / "rows.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
        stream.flush()


from local_llm_lab import device  # noqa: E402

device.pin(seed=0)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.pipeline.lens_fitting import golden  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.finite_difference import (  # noqa: E402
    fit_finite_difference_jacobian,
)
from local_llm_lab.pipeline.lens_fitting.upstream import (  # noqa: E402
    CorpusLensModel,
    load_upstream,
    repo_layer_of_upstream,
)

SNAPSHOT, CORPUS = sys.argv[1], Path(sys.argv[2])
EXACT = np.load(sys.argv[4])
MAX_SEQ, SEED, DIRECTION_BATCH = 128, 20260910, 256
LAYERS = [0, 32]                      # worst and mildest, from the golden run's own per-layer table
SCALES = [1e-2, 1e-3, 1e-4, 1e-5]
PATHS = ["native", "promoted-float32"]

model, report = hf_text.load_text_causal_lm(
    SNAPSHOT, dtype="bfloat16", attn_implementation="eager", device="cuda:0"
)
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.eval()
up = load_upstream()
import jlens.hf as upstream_hf  # noqa: E402

wrapped = CorpusLensModel(upstream_hf.HFLensModel(model, tokenizer=None))
emit("loaded", dtype=report["dtype"], device=report["device"])

import random  # noqa: E402

held = [row for row in read_corpus(CORPUS) if row.get("split") == "held"]
rows = [held[random.Random(SEED).sample(range(len(held)), 3)[0]]]
emit("row", index=rows[0]["index"], n=len(rows))


def two_positions(seq_len: int):
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[8] = True
    mask[seq_len - 1] = True
    return mask


for layer in LAYERS:
    repo = repo_layer_of_upstream(layer)
    reference = np.asarray(EXACT[f"J{repo - 1}"], np.float32)
    emit("reference", upstream_layer=layer, repo_layer=repo,
         frobenius=float(np.linalg.norm(reference.astype(np.float64))))
    for path in PATHS:
        for scale in SCALES:
            t = time.monotonic()
            try:
                fit = fit_finite_difference_jacobian(
                    wrapped, rows, source_layers=[layer], position_selector=two_positions,
                    max_seq_len=MAX_SEQ, epsilon_scale=scale, capture_dtype=path,
                    direction_batch=DIRECTION_BATCH, device="cuda:0", dtype="bfloat16",
                    split="held", upstream=up,
                )
            except Exception as error:  # noqa: BLE001 - the failure is the datum here
                emit("cell", upstream_layer=layer, capture_dtype=path, epsilon_scale=scale,
                     error=f"{type(error).__name__}: {error}", basis="measured-here")
                continue
            candidate = np.asarray(fit.jacobians[layer], np.float32)
            metrics = golden.layer_metrics(reference, candidate)
            emit("cell", upstream_layer=layer, repo_layer=repo, capture_dtype=path,
                 epsilon_scale=scale,
                 epsilon=fit.provenance["epsilon_per_layer"][str(layer)]["mean"],
                 relative_difference=metrics["relative_difference"], cosine=metrics["cosine"],
                 seconds=round(time.monotonic() - t, 1), basis="measured-here")

emit("done", total_minutes=round((time.monotonic() - STARTED) / 60, 2))
