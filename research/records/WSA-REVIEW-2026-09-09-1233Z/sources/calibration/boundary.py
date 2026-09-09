"""§2 of the resolution protocol: verify the intervention before differentiating anything.

The question is narrow and comes before every other question in the protocol. The finite-difference
estimator does not add a perturbation to a block's output; it **replaces** that output wholesale with
a recorded capture, perturbed. If replacing it with the *unperturbed* capture already moves the
target, then every difference the estimator has ever taken has a seam in it, and no step rule, no
precision and no saturation account can be read until that is repaired.

So: an ordinary forward, then the same forward with the replacement hook fed the identical capture,
and the two targets compared **before any reduction** — every target position, every component. The
protocol requires this at each batch width that is actually used, because the fit runs its hooked
forward at the direction batch (256 in the golden run) and the ordinary forward at one, and a
kernel chosen by batch width is a difference the old schedule never proved absent.

Two controls ride along, both from §2:

* **source equals target.** Replacing the target block's own output with its capture must reproduce
  that capture exactly, since nothing runs between them. It is the check that the hook writes what
  it thinks it writes, and it is the only one whose expected answer is known without a model.
* **the sign control.** With a real step, flipping the tangent must negate the odd response
  ``F(x+hv) − F(x−hv)`` and leave the even remainder alone. It separates a genuine directional
  response from an artefact that does not care about direction.

Nothing here differentiates anything and nothing here fits a map. It writes each result as it
completes, because the runbook forbids a tool that reports once at the end on a paid card.

    python boundary.py SNAPSHOT CORPUS OUT
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

MAX_SEQ = 128            # row 39's length, frozen by §1
SEED = 20260910          # the golden run's, so the row is the same row
DECLARED_ROWS = 3
REPO_LAYERS = (1, 17, 33)   # §4's three, in repository indexing
WIDTHS = (1, 64, 256)       # batch one, the exact fit's dim_batch, the FD fit's direction_batch
EPSILON_SCALE = 0.01        # only the sign control uses a real step


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
emit("pinned", determinism=device.describe()["determinism"])

import torch  # noqa: E402

from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.finite_difference import _perturbation  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.upstream import (  # noqa: E402
    CorpusLensModel,
    load_upstream,
    upstream_index_of_repo_layer,
)


def compare(reference: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor) -> dict:
    """Every statistic that distinguishes 'identical' from 'nearly identical', before reduction.

    The readings arrive native; the comparison promotes them afterwards, which changes no arithmetic
    the model did. `equal_fraction` is on the native values, because that is the question — whether
    the bits came back the same — and the norms are on the promoted ones, because a bf16 difference
    of two nearly equal bf16 numbers is not a measurement of anything.
    """
    exact_equal = torch.eq(reference, observed)
    difference = (observed.float() - reference.float())
    selected = difference[:, mask, :]
    return {
        "shape": list(observed.shape),
        "bitwise_identical": bool(torch.equal(reference, observed)),
        "equal_fraction": round(float(exact_equal.float().mean()), 12),
        "equal_fraction_selected": round(float(exact_equal[:, mask, :].float().mean()), 12),
        "max_abs_difference": float(difference.abs().max()),
        "max_abs_difference_selected": float(selected.abs().max()) if selected.numel() else None,
        "mean_abs_difference": float(difference.abs().mean()),
        "reference_max_abs": float(reference.float().abs().max()),
        "rows_identical_to_each_other": bool(
            observed.shape[0] == 1
            or torch.equal(observed, observed[:1].expand_as(observed).contiguous())
        ),
        "basis": "measured-here",
    }


t = time.monotonic()
model, report = hf_text.load_text_causal_lm(
    SNAPSHOT, dtype="bfloat16", attn_implementation="eager", device="cuda:0"
)
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.eval()
emit("loaded", seconds=round(time.monotonic() - t, 2), dtype=report["dtype"],
     device=report["device"], attn=report["attn_implementation"],
     checkpoint_sha256=report.get("sha256", {}).get("config.json"))

up = load_upstream()
import jlens.hf as upstream_hf  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

wrapped = CorpusLensModel(upstream_hf.HFLensModel(model, tokenizer=None))
n_layers, d_model = int(wrapped.n_layers), int(wrapped.d_model)

held = [row for row in read_corpus(CORPUS) if row.get("split") == "held"]
drawn = random.Random(SEED).sample(range(len(held)), DECLARED_ROWS)
row = held[drawn[0]]
# The corpus manifest froze the ids; the adapter never tokenizes text, and `encode` refuses an
# omitted length rather than silently taking upstream's 512. Same two lines the fit uses.
key = wrapped.register(f"row:{row['index']}", row["ids"])
ids = wrapped.encode(key, max_length=MAX_SEQ)
seq_len = int(ids.shape[-1])
mask = torch.zeros(seq_len, dtype=torch.bool)
mask[8] = True
mask[seq_len - 1] = True
positions = [8, seq_len - 1]
target = n_layers - 1
sources = {layer: upstream_index_of_repo_layer(layer) for layer in REPO_LAYERS}

emit("frozen", row_index=row.get("index"), row_id=row.get("id"), seq_len=seq_len,
     n_layers=n_layers, d_model=d_model, positions=positions,
     repo_layers=list(REPO_LAYERS), upstream_layers=sources, target_upstream=target)

write_json("manifest.json", {
    "schema_version": 1,
    "seat": "d-cro",
    "stage": "resolution protocol §2, the intervention boundary",
    "checkpoint": SNAPSHOT,
    "load_report_sha256": report.get("sha256"),
    "corpus": {"manifest": str(CORPUS), "split": "held"},
    "row": {"drawn_indices": drawn, "used": drawn[0], "index": row.get("index"),
            "id": row.get("id"), "seq_len": seq_len, "max_seq_len": MAX_SEQ},
    "positions": positions,
    "repo_layers": list(REPO_LAYERS),
    "upstream_layers": sources,
    "target_upstream": target,
    "widths": list(WIDTHS),
    "epsilon_scale_for_sign_control": EPSILON_SCALE,
    "device": device.describe(),
    "shared_card": False,
    "note": "no differentiation, no map, no fit: this stage only asks whether the replacement hook "
            "fed its own unperturbed capture reproduces the ordinary forward",
})

# ------------------------------------------------------- the ordinary forward, and the captures
with torch.no_grad():
    with ActivationRecorder(wrapped.layers, at=[*sources.values(), target]) as recorder:
        wrapped.forward(ids)
    base = {layer: recorder.activations[layer].detach().clone()
            for layer in (*sources.values(), target)}
reference = base[target]
emit("ordinary_forward", target_shape=list(reference.shape), dtype=str(reference.dtype),
     norm=float(torch.linalg.vector_norm(reference.float())), basis="measured-here")

results = []
failures = []
compute_device = reference.device
basis_columns = torch.eye(d_model, dtype=torch.float32, device=compute_device)

# ------------------------------------------------- 1. the unchanged-residual check, per width
for repo_layer, layer in sources.items():
    for width in WIDTHS:
        columns = basis_columns[:width]
        for position in positions:
            with torch.no_grad():
                handle = wrapped.layers[layer].register_forward_hook(
                    _perturbation(base[layer], position, 0.0, columns)
                )
                try:
                    with ActivationRecorder(wrapped.layers, at=[target]) as inner:
                        wrapped.forward(ids.expand(width, -1))
                        observed = inner.activations[target].detach()
                finally:
                    handle.remove()
            entry = {
                "check": "unchanged_residual",
                "repo_layer": repo_layer,
                "upstream_layer": layer,
                "width": width,
                "position": position,
                **compare(reference.expand_as(observed), observed, mask),
            }
            results.append(entry)
            if not entry["bitwise_identical"]:
                failures.append(entry)
            emit("unchanged_residual", repo_layer=repo_layer, width=width, position=position,
                 bitwise_identical=entry["bitwise_identical"],
                 equal_fraction=entry["equal_fraction"],
                 max_abs_difference=entry["max_abs_difference"],
                 rows_identical=entry["rows_identical_to_each_other"])

# --------------------------------- 2. source equals target: the hook must write what it is given
for width in (1, 64):
    columns = basis_columns[:width]
    with torch.no_grad():
        handle = wrapped.layers[target].register_forward_hook(
            _perturbation(base[target], positions[0], 0.0, columns)
        )
        try:
            with ActivationRecorder(wrapped.layers, at=[target]) as inner:
                wrapped.forward(ids.expand(width, -1))
                observed = inner.activations[target].detach()
        finally:
            handle.remove()
    entry = {"check": "source_equals_target", "repo_layer": None, "upstream_layer": target,
             "width": width, "position": positions[0],
             **compare(reference.expand_as(observed), observed, mask)}
    results.append(entry)
    if not entry["bitwise_identical"]:
        failures.append(entry)
    emit("source_equals_target", width=width, bitwise_identical=entry["bitwise_identical"],
         max_abs_difference=entry["max_abs_difference"])

# ------------------------------------------------------------------- 3. the sign control
sign_rows = []
for repo_layer, layer in sources.items():
    norm = float(torch.linalg.vector_norm(base[layer].float()))
    epsilon = EPSILON_SCALE * norm if norm else EPSILON_SCALE
    columns = basis_columns[:1]
    responses = {}
    for name, step in (("plus", epsilon), ("minus", -epsilon), ("zero", 0.0)):
        with torch.no_grad():
            handle = wrapped.layers[layer].register_forward_hook(
                _perturbation(base[layer], positions[0], step, columns)
            )
            try:
                with ActivationRecorder(wrapped.layers, at=[target]) as inner:
                    wrapped.forward(ids.expand(1, -1))
                    responses[name] = inner.activations[target].detach().float()
            finally:
                handle.remove()
    odd = responses["plus"] - responses["minus"]
    even = responses["plus"] + responses["minus"] - 2.0 * responses["zero"]
    # Flip the tangent: the odd response must negate exactly, the even remainder must not move.
    flipped = {}
    for name, step in (("plus", -epsilon), ("minus", epsilon)):
        with torch.no_grad():
            handle = wrapped.layers[layer].register_forward_hook(
                _perturbation(base[layer], positions[0], step, columns)
            )
            try:
                with ActivationRecorder(wrapped.layers, at=[target]) as inner:
                    wrapped.forward(ids.expand(1, -1))
                    flipped[name] = inner.activations[target].detach().float()
            finally:
                handle.remove()
    odd_flipped = flipped["plus"] - flipped["minus"]
    entry = {
        "check": "sign_control",
        "repo_layer": repo_layer,
        "upstream_layer": layer,
        "epsilon": epsilon,
        "source_norm": norm,
        "odd_norm": float(torch.linalg.vector_norm(odd)),
        "even_remainder_norm": float(torch.linalg.vector_norm(even)),
        "odd_negates_exactly": bool(torch.equal(odd, -odd_flipped)),
        "odd_plus_flipped_norm": float(torch.linalg.vector_norm(odd + odd_flipped)),
        "unchanged_output_fraction": round(
            float(torch.eq(responses["plus"], responses["minus"]).float().mean()), 12),
        "unchanged_output_fraction_selected": round(
            float(torch.eq(responses["plus"], responses["minus"])[:, mask, :].float().mean()), 12),
        "basis": "measured-here",
    }
    sign_rows.append(entry)
    results.append(entry)
    emit("sign_control", repo_layer=repo_layer, epsilon=round(epsilon, 4),
         odd_norm=entry["odd_norm"], even_remainder_norm=entry["even_remainder_norm"],
         odd_negates_exactly=entry["odd_negates_exactly"],
         unchanged_output_fraction=entry["unchanged_output_fraction"])

verdict = {
    "boundary_holds": not failures,
    "unchanged_residual_checks": sum(1 for r in results if r["check"] == "unchanged_residual"),
    "source_equals_target_checks": sum(1 for r in results if r["check"] == "source_equals_target"),
    "failures": failures,
    "conclusion": (
        "the replacement hook fed its own unperturbed capture reproduces the ordinary forward "
        "bit for bit at every tested width and position; the seam is sound and the derivative "
        "comparison may proceed"
        if not failures else
        "the hook fed its own unperturbed capture does NOT reproduce the ordinary forward; this is "
        "a boundary or schedule failure and every finite-difference reading taken through it is "
        "suspect until it is repaired"
    ),
}
write_json("boundary.json", {"verdict": verdict, "results": results, "basis": "measured-here"})
emit("verdict", **{k: v for k, v in verdict.items() if k != "failures"},
     failure_count=len(failures))
raise SystemExit(0 if not failures else 1)
