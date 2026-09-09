"""Gate 1, executed for real: an independent exact fit compared against the map the report used.

`golden_float32.py` passed `reproduction=reference` — the exact map itself — so "the exact estimator
reproduces itself at exactly 0.0" compared a thing with itself and 0.0 was its only possible value.
A gate that cannot fail, reported as passing, in the run whose whole point was a comparison done
right. This executes it.

Two comparisons, and the second is the one the report needs:

* **within this process**: two exact fits of the same layer, back to back, which tests that the fit
  is deterministic given one loaded model;
* **against the saved map**: the fresh fit against `exact-maps-L*.npz` from the run whose report is
  being gated, which additionally tests that a fresh process, a fresh load and a fresh allocator
  reproduce it. That is what a repeat is for, and it is the one the report cites.

**The float32 no-op boundary runs first and gates.** The protocol requires the unchanged-residual
check at each new precision, source and schedule, with its pass bound to the records that follow —
the boundary check of §2/§3.1 was run in bf16, and a float32 model is a different arithmetic path
whose hook has never been checked. Nothing below is written if it fails.

    python repeat_gate.py SNAPSHOT CORPUS SAVED_DIR OUT
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

SNAPSHOT, CORPUS, SAVED, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
STARTED = time.monotonic()

MAX_SEQ, SEED, DECLARED_ROWS = 128, 20260910, 3
REPO_LAYERS = (1, 33)
WIDTH = 1


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

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.finite_difference import _replace_output  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.upstream import (  # noqa: E402
    CorpusLensModel, fit_upstream_jacobian, load_upstream, repo_layer_of_upstream,
    upstream_index_of_repo_layer,
)

model, report = hf_text.load_text_causal_lm(
    SNAPSHOT, dtype="bfloat16", attn_implementation="eager", device="cuda:0"
)
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.eval()
model.to(torch.float32)
emit("loaded", dtype=str(next(model.parameters()).dtype),
     checkpoint_sha256=report.get("sha256", {}).get("config.json"),
     tf32=torch.backends.cuda.matmul.allow_tf32,
     float32_matmul_precision=torch.get_float32_matmul_precision())

up = load_upstream()
import jlens.hf as upstream_hf  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

wrapped = CorpusLensModel(upstream_hf.HFLensModel(model, tokenizer=None))
n_layers, d_model = int(wrapped.n_layers), int(wrapped.d_model)
target = n_layers - 1
sources = {repo: upstream_index_of_repo_layer(repo) for repo in REPO_LAYERS}

held = [row for row in read_corpus(CORPUS) if row.get("split") == "held"]
row = held[random.Random(SEED).sample(range(len(held)), DECLARED_ROWS)[0]]
key = wrapped.register(f"row:{row['index']}", row["ids"])
ids = wrapped.encode(key, max_length=MAX_SEQ)


def two_positions(seq_len: int):
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[8] = True
    mask[seq_len - 1] = True
    return mask


# --------------------------------------------- the float32 no-op boundary, first, and it gates
boundary = {}
for repo_layer, layer in sources.items():
    with torch.no_grad():
        with ActivationRecorder(wrapped.layers, at=[layer, target]) as recorder:
            wrapped.forward(ids)
        anchor = recorder.activations[layer].detach().clone()
        reference = recorder.activations[target].detach().clone()
        handle = wrapped.layers[layer].register_forward_hook(
            lambda m, i, o, t=anchor: _replace_output(t, o)
        )
        try:
            with ActivationRecorder(wrapped.layers, at=[target]) as inner:
                wrapped.forward(ids)
                observed = inner.activations[target].detach()
        finally:
            handle.remove()
    holds = bool(torch.equal(reference, observed))
    boundary[repo_layer] = {
        "bitwise_identical": holds,
        "max_abs_difference": float((observed.float() - reference.float()).abs().max()),
        "precision": "float32", "width": WIDTH, "interventions": 1, "basis": "measured-here",
    }
    emit("float32_boundary", repo_layer=repo_layer, **boundary[repo_layer])
    if not holds:
        raise SystemExit(
            f"the unchanged-residual check fails in float32 at repo layer {repo_layer}: the hook "
            "does not reproduce this precision's own forward, so no fit below is interpretable."
        )

# ------------------------------------------------------------------- the repeat, twice over
saved = {}
for repo_layer in REPO_LAYERS:
    path = SAVED / f"exact-maps-L{repo_layer}.npz"
    with np.load(path) as archive:
        saved[repo_layer] = {int(k[1:]): np.asarray(archive[k], np.float64) for k in archive}


def relative(a: np.ndarray, b: np.ndarray) -> dict:
    difference = float(np.linalg.norm(b - a))
    reference_norm = float(np.linalg.norm(a))
    return {
        "relative_frobenius_difference": difference / reference_norm if reference_norm else None,
        "max_abs_difference": float(np.abs(b - a).max()),
        "bitwise_identical": bool(np.array_equal(a, b)),
    }


def fit(repo_layer):
    result = fit_upstream_jacobian(
        wrapped, [row], position_selector=two_positions, max_seq_len=MAX_SEQ, dim_batch=WIDTH,
        device="cuda:0", dtype="float32", split="held", upstream=up,
        source_layers=[sources[repo_layer]], target_layer=target,
    )
    return {repo_layer_of_upstream(i): np.asarray(m, np.float64)
            for i, m in result.jacobians.items()}


rows_out = {}
for repo_layer in REPO_LAYERS:
    t = time.monotonic()
    first, second = fit(repo_layer), fit(repo_layer)
    within = relative(first[repo_layer], second[repo_layer])
    against = relative(saved[repo_layer][repo_layer], first[repo_layer])
    rows_out[repo_layer] = {
        "within_process": within, "against_saved_map": against,
        "seconds": round(time.monotonic() - t, 2),
        "saved_map": str(SAVED / f"exact-maps-L{repo_layer}.npz"),
        "boundary": boundary[repo_layer], "basis": "measured-here",
    }
    emit("repeat", repo_layer=repo_layer,
         within_process_relative=within["relative_frobenius_difference"],
         within_process_bitwise=within["bitwise_identical"],
         against_saved_relative=against["relative_frobenius_difference"],
         against_saved_bitwise=against["bitwise_identical"],
         seconds=rows_out[repo_layer]["seconds"])

verdict = {
    "gate": "exact_reproduces_itself",
    "executed": True,
    "note": "the first run passed reproduction=reference and so never executed this gate; both "
            "comparisons here are between independently produced maps",
    "float32_boundary_holds": all(v["bitwise_identical"] for v in boundary.values()),
    "passes": all(r["against_saved_map"]["relative_frobenius_difference"] == 0.0
                  for r in rows_out.values()),
    # The set the verdict was computed over, per method entry thirty-four. This gate covers the two
    # layers that have maps, not the model: "the exact estimator reproduces itself" is true here of
    # repo layers 1 and 33 on one row, and a reader who takes it for the estimator would be taking
    # more than was measured.
    "coverage": {
        "repo_layers_checked": sorted(REPO_LAYERS),
        "repo_layers_in_model": n_layers,
        "rows": 1,
        "positions_per_row": 2,
        "note": "two layers of the model, one row, the two positions the maps were fitted over; "
                "the claim is about those maps and not about the estimator in general",
    },
    "by_layer": rows_out,
}
(OUT / "repeat-gate.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
emit("verdict", executed=True, passes=verdict["passes"],
     float32_boundary_holds=verdict["float32_boundary_holds"],
     repo_layers_checked=sorted(REPO_LAYERS), repo_layers_in_model=n_layers, rows=1)
