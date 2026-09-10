"""The golden test done right: two float32 maps fitted inside a demonstrated interval, matched.

Everything the first golden run could not have known, now known. Both estimators run at **width one**,
so no schedule term enters and the comparability gate's new `forward_batch`/`anchor_batch` fields
agree by construction rather than by luck. Both run in coherent float32, because the ladder measured
that native bf16 has no useful step interval at any layer tested. And each layer's step is the one
its own ladder minimum names, because the ladder also measured that a single `epsilon_scale` for
every layer is refuted: layer 33's interval sits at k = 6 and layer 1's at k = 10, a factor of 64
apart.

**A deviation from the Chief's ruling, stated here rather than buried.** The ruling named layer 33 at
k = 6 and layer 1 at k = 10, "fitted at the width the anchored-at-width check has cleared (64 is
fine)". Those two k are the **width-one** minima. The width rows, which the ruling was written before
seeing, measured that the intervals move with width: at width 64 layer 33's minimum is k = 8, not 6,
and layer 1 has no demonstrated interval at all, still falling at k = 10 with 4.5e-2. Fitting at
width 64 with the ruled k would put layer 33 sixteen times off its own best and layer 1 outside any
demonstrated interval, which is the one precondition the protocol states for fitting a full map. So
the k are the ruling's and the width is one. If the Chief wants width 64, layer 33 moves to k = 8 and
layer 1 needs its interval found there first.

    python golden_float32.py SNAPSHOT CORPUS OUT
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
#: repo layer -> the ladder's demonstrated minimum, as k in epsilon_scale = 0.01 * 2**-k.
LADDER_MINIMUM = {33: 6, 1: 10}
BASE_EPSILON_SCALE = 0.01
WIDTH = 1


def emit(event: str, /, **fields) -> dict:
    fields.pop("event", None)
    row = {"event": event, "elapsed_s": round(time.monotonic() - STARTED, 2), **fields}
    print(json.dumps(row, default=str), flush=True)
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
        stream.flush()
    return row


def write_json(name: str, payload: dict) -> None:
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    emit("wrote", file=name, bytes=path.stat().st_size)


from local_llm_lab import device  # noqa: E402

device.pin(seed=0)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.pipeline.lens_fitting import golden  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.finite_difference import (  # noqa: E402
    fit_finite_difference_jacobian,
)
from local_llm_lab.pipeline.lens_fitting.upstream import (  # noqa: E402
    ESTIMATOR_FINITE_DIFFERENCE,
    CorpusLensModel,
    declare_nu,
    fit_upstream_jacobian,
    load_upstream,
    repo_layer_of_upstream,
    upstream_index_of_repo_layer,
)

t = time.monotonic()
model, report = hf_text.load_text_causal_lm(
    SNAPSHOT, dtype="bfloat16", attn_implementation="eager", device="cuda:0"
)
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.eval()
# The same stored values cast up, which is exact for bf16 into float32. This is the coherent
# float32 model the protocol's §3 asks for: not autocast, not a promoted residual through bf16
# blocks, but every parameter and buffer in float32.
model.to(torch.float32)
emit("loaded", seconds=round(time.monotonic() - t, 2),
     dtype=str(next(model.parameters()).dtype), attn=report["attn_implementation"],
     checkpoint_sha256=report.get("sha256", {}).get("config.json"),
     tf32=torch.backends.cuda.matmul.allow_tf32,
     float32_matmul_precision=torch.get_float32_matmul_precision())

up = load_upstream()
import jlens.hf as upstream_hf  # noqa: E402

wrapped = CorpusLensModel(upstream_hf.HFLensModel(model, tokenizer=None))
n_layers, d_model = int(wrapped.n_layers), int(wrapped.d_model)
sources = {repo: upstream_index_of_repo_layer(repo) for repo in LADDER_MINIMUM}

held = [row for row in read_corpus(CORPUS) if row.get("split") == "held"]
drawn = random.Random(SEED).sample(range(len(held)), DECLARED_ROWS)
rows = [held[drawn[0]]]


def two_positions(seq_len: int):
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[8] = True
    mask[seq_len - 1] = True
    return mask


corpus_declaration = {"manifest": str(CORPUS), "split": "held",
                      "rows": [int(r["index"]) for r in rows]}
write_json("manifest.json", {
    "schema_version": 1, "seat": "d-cro",
    "stage": "the golden test at a demonstrated interval, coherent float32, width 1",
    "checkpoint": SNAPSHOT, "load_report_sha256": report.get("sha256"),
    "corpus": corpus_declaration,
    "source_layers_repo": sorted(LADDER_MINIMUM),
    "ladder_minimum_k": LADDER_MINIMUM,
    "epsilon_scale_per_layer": {str(k): BASE_EPSILON_SCALE * 2.0 ** -v
                                for k, v in LADDER_MINIMUM.items()},
    "forward_batch": WIDTH, "anchor_batch": WIDTH,
    "deviation": {
        "ruled": "layers 33 at k=6 and 1 at k=10, at width 64",
        "run": "the same k, at width 1",
        "reason": "those k are the width-one ladder minima; at width 64 layer 33's minimum is k=8 "
                  "and layer 1 shows no demonstrated interval, and the protocol forbids a full map "
                  "outside a demonstrated interval",
    },
    "tf32": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "float32_matmul_precision": torch.get_float32_matmul_precision()},
    "device": device.describe(),
})

# ------------------------------------------------------------------ stage 0: the memory smoke row
torch.cuda.reset_peak_memory_stats()
t = time.monotonic()
smoke = fit_upstream_jacobian(
    wrapped, rows, position_selector=two_positions, max_seq_len=MAX_SEQ, dim_batch=WIDTH,
    device="cuda:0", dtype="float32", split="held", upstream=up,
    source_layers=[sources[33]], target_layer=n_layers - 1, max_rows=1,
)
smoke_peak = torch.cuda.max_memory_allocated() / 2**30
emit("smoke", seconds=round(time.monotonic() - t, 2), peak_gib=round(smoke_peak, 3),
     layers=len(smoke.jacobians), dim_batch=WIDTH, precision="float32",
     note="one source layer, width 1; the bf16 exact side peaked at 43.9 GiB at dim_batch 64",
     basis="measured-here")
if smoke_peak > 60.0:
    raise SystemExit(f"smoke peak {smoke_peak:.1f} GiB is too close to the card; reduce the width")

# ------------- stages 1 to 3: per layer, an exact map and a finite-difference map at its own step
#
# One layer per pair, and both sides fitted over that layer alone. An exact map declaring two source
# layers against a finite-difference map declaring one is not a comparison of estimators: the
# endpoint differs, and the gate refuses it — correctly, and it did, which is how this loop came to
# be written this way. Each layer needs its own step anyway, since the ladder's minima are 64 apart.
reports = {}
for repo_layer, k in sorted(LADDER_MINIMUM.items()):
    scale = BASE_EPSILON_SCALE * 2.0 ** -k
    torch.cuda.reset_peak_memory_stats()
    t = time.monotonic()
    exact = fit_upstream_jacobian(
        wrapped, rows, position_selector=two_positions, max_seq_len=MAX_SEQ, dim_batch=WIDTH,
        device="cuda:0", dtype="float32", split="held", upstream=up,
        source_layers=[sources[repo_layer]], target_layer=n_layers - 1,
    )
    exact_nu = declare_nu(exact, num_layers=n_layers, corpus=corpus_declaration)
    emit("exact_fit", repo_layer=repo_layer, seconds=round(time.monotonic() - t, 2),
         layers=len(exact.jacobians),
         peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
         forward_batch=exact_nu["precision"]["forward_batch"],
         anchor_batch=exact_nu["precision"]["anchor_batch"], basis="measured-here")
    exact_maps = {repo_layer_of_upstream(i): np.asarray(m, np.float32)
                  for i, m in exact.jacobians.items()}
    np.savez_compressed(OUT / f"exact-maps-L{repo_layer}.npz",
                        **{f"J{i}": m for i, m in exact_maps.items()})
    write_json(f"nu-exact-L{repo_layer}.json", exact_nu)
    torch.cuda.reset_peak_memory_stats()
    t = time.monotonic()
    fd = fit_finite_difference_jacobian(
        wrapped, rows, source_layers=[sources[repo_layer]], target_layer=n_layers - 1,
        position_selector=two_positions, epsilon_scale=scale, capture_dtype="native",
        direction_batch=WIDTH, anchor_batch=WIDTH, max_seq_len=MAX_SEQ,
        device="cuda:0", dtype="float32", split="held", upstream=up,
    )
    fd_nu = declare_nu(fd, num_layers=n_layers, corpus=corpus_declaration,
                       estimator=ESTIMATOR_FINITE_DIFFERENCE)
    emit("fd_fit", repo_layer=repo_layer, k=k, epsilon_scale=scale,
         seconds=round(time.monotonic() - t, 2),
         peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
         epsilon_per_layer=fd.provenance["epsilon_per_layer"],
         forward_batch=fd_nu["precision"]["forward_batch"],
         anchor_batch=fd_nu["precision"]["anchor_batch"], basis="measured-here")
    fd_maps = {repo_layer_of_upstream(i): np.asarray(m, np.float32)
               for i, m in fd.jacobians.items()}
    np.savez_compressed(OUT / f"fd-maps-L{repo_layer}.npz",
                        **{f"J{i}": m for i, m in fd_maps.items()})
    write_json(f"nu-finite-difference-L{repo_layer}.json", fd_nu)

    # The gate first, then the comparison. It now compares the batch schedule as well, and both
    # sides declare width 1, so it passes on a matched pair rather than on a silence.
    golden.assert_estimator_is_the_only_difference(exact_nu, fd_nu)
    reference = {repo_layer: exact_maps[repo_layer]}
    report_row = golden.golden_report(
        reference=reference, candidate={repo_layer: fd_maps[repo_layer]},
        reference_nu=exact_nu, candidate_nu=fd_nu,
        storage_dtypes=["float32"],
        finite_difference_epsilon=fd.provenance["epsilon_per_layer"],
        reproduction=reference,
        # The transposed control only. The layer-shifted one needs a second layer in the same map
        # to shift to, and there is not one: each layer is fitted against its own step, because the
        # ladder measured the two minima 64 apart, so a two-layer finite-difference map would be a
        # lens no single `epsilon_scale` describes. Recorded as not constructible rather than
        # omitted — a control that quietly disappears is a gate that quietly weakens.
        controls={"transposed": golden.transposed_control(reference)},
        fixture_residual=None,
    )
    report_row["gates"]["layer_shifted_control"] = {
        "available": False,
        "reason": "a single-layer map has no other layer to shift to; the per-layer step this "
                  "ladder implies makes a multi-layer finite-difference fit a different object, "
                  "since one epsilon_scale no longer describes it",
    }
    reports[repo_layer] = report_row
    write_json(f"golden-report-L{repo_layer}.json", report_row)
    emit("golden", repo_layer=repo_layer, k=k,
         worst_relative=report_row["finding"]["worst_relative_difference"],
         exact_reproduces_itself=report_row["gates"]["exact_reproduces_itself"]["measured"],
         controls_separate=all(r["separates"] for r in
                               report_row["gates"]["controls_separate"].values()),
         at_or_below_storage_floor=report_row["finding"]["at_or_below_storage_floor"],
         basis="measured-here")

write_json("summary.json", {
    "basis": "measured-here",
    "worst_relative_difference": {str(L): r["finding"]["worst_relative_difference"]
                                  for L, r in reports.items()},
    "ladder_prediction": {str(L): "the ladder's median relative error at this k on eighteen scalar "
                                  "checks" for L in reports},
    "note": "the scalar ladder and the full map measure different aggregates; agreement between "
            "them is evidence, disagreement is a finding, and neither is assumed",
})
emit("done", layers=sorted(reports))
