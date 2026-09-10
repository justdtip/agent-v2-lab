"""§4 of the resolution protocol at width one: directional checks and the step ladder, matched.

Width one throughout, which is the Chief's ruling and the protocol's "initially use the same forward
batch size". It is also the canonical function: the ordinary forward is what every other reading of
this model means, and the calibration record's §3 measured that the bf16 forward is not
batch-invariant, so a comparison made at any other width carries a schedule term. Here there is none.
Anchor at width one, autograd at width one, each perturbed forward on its own.

**The quantity.** Perturbation at a single source position `p`, target reduced by a sum over the
selected target positions — no averaging over source positions, because a directional check on one
position should not be divided by a count of positions it did not touch. The scalar identity is

    a   = (Jᵀw)ᵀv          from one VJP, no matrix materialised
    d_h = wᵀ{F(x+hv) − F(x−hv)} / (2h)

and the ladder is `h₀ · 2⁻ᵏ` for k = 0, 2, 4, 6, 8, 10, where `h₀` is the estimator's current step,
`epsilon_scale · ‖source residual over the full sequence‖`. A useful interval is several neighbouring
steps whose `d_h` is stable and close to `a`; the point of the ladder is to find out whether one
exists, not to assume it does.

**Both arithmetic paths, separately.** Native bf16 first, then a coherent float32 model — the same
stored weight values cast up, which is exact — with autocast off and TF32 off, the realized settings
stamped rather than assumed. Retaining the native comparison is §3's instruction; if float32 finds a
useful interval and bf16 does not, precision is implicated, and if reducing the step fixes float32,
finite-step error is.

**§5's quantities are kept per cell, before any reduction**, because the point of the whole exercise
is that an aggregate can hide a dead response: requested against realized displacement, the midpoint
shift, the fraction of the perturbed coordinates that did not move, the fraction of target components
that came back exactly equal, the odd response and the even remainder, and `a` recomputed against the
realized direction as well as the requested one.

    python ladder.py SNAPSHOT CORPUS OUT [--precisions native,float32]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("snapshot")
parser.add_argument("corpus", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--precisions", default="native,float32")
parser.add_argument("--layers", default="1,17,33")
parser.add_argument("--ladder-k", default="0,2,4,6,8,10",
                    help="the declared trial ladder, as exponents k in h0 * 2**-k")
parser.add_argument("--width", type=int, default=1,
                    help="the forward width; the anchor is captured at this width too, so the "
                         "anchor and the difference share a function")
args = parser.parse_args()

OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
STARTED = time.monotonic()

MAX_SEQ, SEED, DECLARED_ROWS = 128, 20260910, 3
EPSILON_SCALE = 0.01
LADDER_K = tuple(int(x) for x in args.ladder_k.split(","))
DIRECTION_SEED = 20260911      # pre-registered here, before any number is read
COORDINATE_DIRECTIONS = (0, 137, 1279, 2559)   # fixed, spread across the width
DENSE_DIRECTIONS = 2
COTANGENTS = 3                 # one coordinate, two dense


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
from local_llm_lab.pipeline.lens_fitting.finite_difference import _replace_output  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.upstream import (  # noqa: E402
    CorpusLensModel,
    load_upstream,
    upstream_index_of_repo_layer,
)

# TF32 is off explicitly and the realized setting is stamped below: a float32 comparison run on
# tensor cores is not a float32 comparison, and the label on the tensor does not say which happened.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

model, report = hf_text.load_text_causal_lm(
    args.snapshot, dtype="bfloat16", attn_implementation="eager", device="cuda:0"
)
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.eval()

load_upstream()
import jlens.hf as upstream_hf  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

wrapped = CorpusLensModel(upstream_hf.HFLensModel(model, tokenizer=None))
n_layers, d_model = int(wrapped.n_layers), int(wrapped.d_model)
target = n_layers - 1
repo_layers = tuple(int(x) for x in args.layers.split(","))
sources = {layer: upstream_index_of_repo_layer(layer) for layer in repo_layers}

held = [row for row in read_corpus(args.corpus) if row.get("split") == "held"]
row = held[random.Random(SEED).sample(range(len(held)), DECLARED_ROWS)[0]]
key = wrapped.register(f"row:{row['index']}", row["ids"])
ids = wrapped.encode(key, max_length=MAX_SEQ)
seq_len = int(ids.shape[-1])
mask = torch.zeros(seq_len, dtype=torch.bool, device=ids.device)
mask[8] = True
mask[seq_len - 1] = True
POSITION = 8   # the perturbed source position; one, so the check is directional and not an average
WIDTH = args.width
batched_ids = ids if WIDTH == 1 else ids.expand(WIDTH, -1)

generator = torch.Generator(device="cpu").manual_seed(DIRECTION_SEED)
directions = [("coordinate", i, torch.nn.functional.one_hot(torch.tensor(i), d_model).float())
              for i in COORDINATE_DIRECTIONS]
for j in range(DENSE_DIRECTIONS):
    v = torch.randn(d_model, generator=generator)
    directions.append((f"dense", j, v / torch.linalg.vector_norm(v)))
cotangents = []
w0 = torch.zeros(d_model); w0[7] = 1.0
cotangents.append(("coordinate", 0, w0))
for j in range(COTANGENTS - 1):
    w = torch.randn(d_model, generator=generator)
    cotangents.append(("dense", j, w / torch.linalg.vector_norm(w)))

emit("frozen", row_index=row.get("index"), seq_len=seq_len, d_model=d_model,
     repo_layers=list(repo_layers), position=POSITION, ladder_k=list(LADDER_K),
     directions=[(k, i) for k, i, _ in directions], cotangents=[(k, i) for k, i, _ in cotangents],
     direction_seed=DIRECTION_SEED, width=WIDTH, anchor_batch=WIDTH,
     frozen_token_sha256=__import__("hashlib").sha256(
         json.dumps(row["ids"][:MAX_SEQ]).encode()).hexdigest())

(OUT / "manifest.json").write_text(json.dumps({
    "schema_version": 1, "seat": "d-cro", "stage": "resolution protocol §4, width 1",
    "checkpoint": args.snapshot, "load_report_sha256": report.get("sha256"),
    "corpus": {"manifest": str(args.corpus), "split": "held"},
    "row": {"index": row.get("index"), "seq_len": seq_len},
    "position": POSITION, "repo_layers": list(repo_layers), "target_upstream": target,
    "forward_batch": WIDTH, "anchor_batch": WIDTH,
    "reduction": "perturb one source position; target summed over the selected positions",
    "epsilon_scale": EPSILON_SCALE, "ladder_k": list(LADDER_K),
    "direction_seed": DIRECTION_SEED,
    "tf32": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
             "float32_matmul_precision": torch.get_float32_matmul_precision()},
    "device": device.describe(),
}, indent=2, sort_keys=True, default=str) + "\n")


def replace(base):
    def hook(module, inputs, output):
        return _replace_output(base, output)
    return hook


def unchanged_residual_holds(layer):
    """§3.1's check, at this width, with a capture made at this width. Run before any derivative.

    At zero step the perturbed position is irrelevant — the hook replaces the whole tensor — so this
    is *one* intervention per layer and is counted as one. Six entries reported as six checks would
    be five more than the evidence.
    """
    with torch.no_grad():
        with ActivationRecorder(wrapped.layers, at=[layer, target]) as recorder:
            wrapped.forward(batched_ids)
        anchor = recorder.activations[layer].detach().clone()
        reference = recorder.activations[target].detach().clone()
        handle = wrapped.layers[layer].register_forward_hook(replace(anchor))
        try:
            with ActivationRecorder(wrapped.layers, at=[target]) as inner:
                wrapped.forward(batched_ids)
                observed = inner.activations[target].detach()
        finally:
            handle.remove()
    return bool(torch.equal(reference, observed))


def reduced(activation):
    """The target, reduced as declared: summed over the selected positions, float32.

    Promoted before the sum, never after: summing two nearly equal bf16 numbers in bf16 and casting
    the result is a measurement of bf16's spacing rather than of the model.
    """
    return activation.float()[0][mask].sum(dim=0)  # row 0; every row carries the same prompt


def forward_target(layer, base):
    handle = wrapped.layers[layer].register_forward_hook(replace(base))
    try:
        with ActivationRecorder(wrapped.layers, at=[target]) as inner:
            wrapped.forward(batched_ids)
            return inner.activations[target]
    finally:
        handle.remove()


import hashlib  # noqa: E402

import numpy as np  # noqa: E402

rows_out = []
responses: dict[str, object] = {}
archive_index: list[dict] = []


def _archive(unit: str) -> None:
    """Write the completed unit's arrays, hash them, and index them. Then forget them.

    Per **completed unit**, not once at the end: a run that accumulates its archive in memory and
    writes it after the last cell loses everything it measured if it is interrupted, which is the
    runbook's forbidden shape wearing an archive's clothes — and this record already lost
    twenty-seven paid minutes to exactly that in a logger. Each file is hashed and the hash goes in
    the index beside the cells it holds, so an archive cannot be quietly swapped for another.
    """
    if not responses:
        return
    path = OUT / f"responses-{unit}.npz"
    np.savez_compressed(path, **responses)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    archive_index.append({
        "unit": unit, "file": path.name, "sha256": digest,
        "arrays": sorted(responses), "bytes": path.stat().st_size,
    })
    (OUT / "responses-index.json").write_text(
        json.dumps({"basis": "measured-here", "units": archive_index}, indent=2, sort_keys=True)
        + "\n"
    )
    emit("archived", unit=unit, arrays=len(responses), bytes=path.stat().st_size,
         sha256=digest[:16])
    responses.clear()


for precision in args.precisions.split(","):
    if precision == "float32":
        model.to(torch.float32)
        emit("promoted", dtype=str(next(model.parameters()).dtype),
             note="the same stored values cast up, which is exact; autocast off, TF32 off")
    observed_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")

    with torch.no_grad():
        with ActivationRecorder(wrapped.layers, at=[*sources.values(), target]) as recorder:
            wrapped.forward(batched_ids)
        base = {layer: recorder.activations[layer].detach().clone()
                for layer in (*sources.values(), target)}

    # §3.1 at this width, before any derivative is read. It gates rather than reports: a hook that
    # does not reproduce its own width's forward makes every number below meaningless.
    for repo_layer, layer in sources.items():
        holds = unchanged_residual_holds(layer)
        emit("unchanged_residual", precision=precision, width=WIDTH, repo_layer=repo_layer,
             anchor="width", bitwise_identical=holds, interventions=1, basis="measured-here")
        if not holds:
            raise SystemExit(
                f"the unchanged-residual check fails at width {WIDTH}, repo layer {repo_layer}, "
                "with a capture made at that width: the hook does not reproduce this width's own "
                "forward, so no derivative here is interpretable. Repair the seam first."
            )

    for repo_layer, layer in sources.items():
        source = base[layer]
        # One replica, exactly as `fit_finite_difference_jacobian` does. The capture at width w is
        # `[w, seq, hidden]` and every row is the same prompt, so a norm over the whole tensor is
        # √w times the norm the fitter uses and the step inherits the factor: at width 64 this
        # ladder ran every rung at **8×** the fitter's step, so its rung k was the fitter's k − 3.
        # Found by Codex's width audit, 2026-09-09. Two paths that agreed in intent and differed in
        # arithmetic, with nothing comparing them — this week's shape once more.
        norm = float(torch.linalg.vector_norm(source[:1].float()))
        h0 = EPSILON_SCALE * norm
        coordinate_scale = float(source.float()[0, POSITION].abs().mean())
        token_norm = float(torch.linalg.vector_norm(source.float()[0, POSITION]))

        # -------- the VJP: one backward per cotangent, no matrix materialised
        jt_w = {}
        for kind, index, w in cotangents:
            leaf = source.detach().clone().float().requires_grad_(True)
            handle = wrapped.layers[layer].register_forward_hook(
                replace(leaf.to(source.dtype) if precision == "native" else leaf)
            )
            try:
                with ActivationRecorder(wrapped.layers, at=[target]) as inner:
                    wrapped.forward(batched_ids)
                    scalar = (reduced(inner.activations[target]) * w.to("cuda:0")).sum()
            finally:
                handle.remove()
            grad, = torch.autograd.grad(scalar, leaf)
            jt_w[(kind, index)] = grad[0, POSITION].detach().float().cpu()
            emit("vjp", precision=precision, repo_layer=repo_layer, cotangent=f"{kind}:{index}",
                 grad_norm=float(torch.linalg.vector_norm(jt_w[(kind, index)])),
                 scalar=float(scalar), basis="measured-here")

        with torch.no_grad():
            zero_raw = forward_target(layer, source).float()[0][mask]
            zero = zero_raw.sum(dim=0)
            for dkind, dindex, v in directions:
                v_dev = v.to("cuda:0")
                for k in LADDER_K:
                    h = h0 * (2.0 ** -k)
                    plus_in = source.clone()
                    minus_in = source.clone()
                    # Every row of the batch carries the same prompt and takes the same
                    # perturbation, so the batch reproduces the schedule without turning into a
                    # different experiment: the width is the arithmetic path, not extra directions.
                    plus_in[:, POSITION] += (h * v_dev).to(source.dtype)
                    minus_in[:, POSITION] -= (h * v_dev).to(source.dtype)
                    # §5: what actually landed, not what was asked for.
                    dplus = (plus_in.float() - source.float())[0, POSITION]
                    dminus = (source.float() - minus_in.float())[0, POSITION]
                    midpoint = ((plus_in.float() + minus_in.float()) / 2 - source.float())[0, POSITION]
                    support = v_dev != 0
                    unchanged = float((dplus[support] == 0).float().mean()) if support.any() else None
                    v_actual = ((plus_in.float() - minus_in.float())[0, POSITION] / (2 * h)).cpu()

                    raw_plus = forward_target(layer, plus_in).float()[0][mask]
                    raw_minus = forward_target(layer, minus_in).float()[0][mask]
                    raw_zero = zero_raw
                    # Codex's R2: retain the plus, minus and zero outputs **per target position**
                    # and the requested and realized input vectors, before any summation or
                    # projection. The odd response alone cannot be taken apart afterwards, and the
                    # whole argument of this record is that an aggregate can hide a dead response.
                    cell = f"{precision}|L{repo_layer}|{dkind}:{dindex}|k{k}"
                    responses[f"{cell}|plus"] = raw_plus.cpu().numpy()
                    responses[f"{cell}|minus"] = raw_minus.cpu().numpy()
                    responses[f"{cell}|zero"] = raw_zero.cpu().numpy()
                    responses[f"{cell}|requested"] = (h * v).cpu().numpy()
                    responses[f"{cell}|realized_plus"] = dplus.cpu().numpy()
                    responses[f"{cell}|realized_minus"] = dminus.cpu().numpy()
                    tplus, tminus = raw_plus.sum(dim=0), raw_minus.sum(dim=0)
                    odd = (tplus - tminus)
                    even = (tplus + tminus - 2 * zero)
                    equal_outputs = float(torch.eq(raw_plus, raw_minus).float().mean())
                    per_position_odd = [float(torch.linalg.vector_norm(r))
                                        for r in (raw_plus - raw_minus)]

                    for ckind, cindex, w in cotangents:
                        w_dev = w.to("cuda:0")
                        d_h = float((odd * w_dev).sum() / (2 * h))
                        a = float((jt_w[(ckind, cindex)] * v).sum())
                        a_actual = float((jt_w[(ckind, cindex)] * v_actual).sum())
                        entry = {
                            "precision": precision, "observed_dtype": observed_dtype,
                            "repo_layer": repo_layer, "position": POSITION,
                            "direction": f"{dkind}:{dindex}", "cotangent": f"{ckind}:{cindex}",
                            "k": k, "h": h, "h0": h0,
                            "h_over_token_norm": h / token_norm if token_norm else None,
                            "h_over_coordinate_scale": h / coordinate_scale if coordinate_scale else None,
                            "a_requested": a, "a_realized_direction": a_actual, "d_h": d_h,
                            "absolute_error": abs(d_h - a),
                            "relative_error": abs(d_h - a) / abs(a) if a else None,
                            "relative_error_vs_realized": (abs(d_h - a_actual) / abs(a_actual)
                                                           if a_actual else None),
                            "requested_displacement_norm": h * float(torch.linalg.vector_norm(v)),
                            "realized_plus_norm": float(torch.linalg.vector_norm(dplus)),
                            "realized_minus_norm": float(torch.linalg.vector_norm(dminus)),
                            "midpoint_shift_norm": float(torch.linalg.vector_norm(midpoint)),
                            "unchanged_input_fraction_in_support": unchanged,
                            "equal_output_fraction": equal_outputs,
                            "odd_norm": float(torch.linalg.vector_norm(odd)),
                            "odd_norm_per_target_position": per_position_odd,
                            "even_remainder_norm": float(torch.linalg.vector_norm(even)),
                            "width": WIDTH, "anchor_batch": WIDTH,
                            "basis": "measured-here",
                        }
                        rows_out.append(entry)
                        with (OUT / "ladder.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(entry, default=str) + "\n")
                _archive(f"{precision}-L{repo_layer}-{dkind}{dindex}")
                emit("direction", precision=precision, repo_layer=repo_layer,
                     direction=f"{dkind}:{dindex}", cells=len(LADDER_K) * len(cotangents),
                     best_relative=min(
                         (r["relative_error"] for r in rows_out
                          if r["relative_error"] is not None
                          and r["direction"] == f"{dkind}:{dindex}"
                          and r["repo_layer"] == repo_layer and r["precision"] == precision),
                         default=None))
    emit("precision_done", precision=precision, rows=len(rows_out))

_archive("final")
emit("responses", units=len(archive_index),
     bytes=sum(u["bytes"] for u in archive_index),
     note="plus, minus and zero per selected target position and the requested and realized input "
          "vectors, before any summation, written per completed unit with a hash and an index")

(OUT / "ladder-summary.json").write_text(json.dumps({
    "basis": "measured-here", "cells": len(rows_out),
    "note": "a useful interval is several neighbouring k with stable d_h and reduced error against "
            "a; this file reports the cells and does not declare one",
}, indent=2, sort_keys=True) + "\n")
emit("done", cells=len(rows_out))
