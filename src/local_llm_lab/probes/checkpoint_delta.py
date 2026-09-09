"""Where did full fine-tuning put its changes? The depth-why quantity over full weight differences.

The reference is `research/records/DEPTH-WHY-2026-09-08/adapter_geometry.py`, not `adapter_delta.py`
-- the per-layer quantity the depth-why record reports does not exist in the probe module, which
emits per-module ratios only. That script's line 7 names the mistake this one must not repeat:

    Per-layer relative perturbation: ||dW_layer||_F / ||W_layer||_F, aggregating numerator and
    denominator separately in quadrature over the layer's adapted modules. NOT the quadrature sum
    of per-module ratios -- that inflates by roughly sqrt(number of modules).

So, per layer `L` over its modules `m`:

    numerator[L]   = sum_m ||dW_m||_F^2
    denominator[L] = sum_m ||W_m||_F^2
    ratio[L]       = sqrt(numerator[L]) / sqrt(denominator[L])

**Two module sets, and the record says which is which.** LoRA moved seven projections per block, so
that restricted set is the only one comparable to a depth-why number; full fine-tuning also moves
four Gemma norms per block, the attention q/k norms, the final norm and the embedding. Reporting
only the restricted set would be dishonest about full fine-tuning and reporting only the full set
would be incomparable, so both are emitted and labelled.

**What has no analogue here.** `adapter_geometry` takes an exact SVD through a QR of the rank-r
factors, never materialising `d x d`. A full weight difference has no rank-r core, so effective rank
here is the SVD of the materialised difference: the same definition, a different computation, and
one that costs `O(d^3)`. It is opt-in for that reason, and the record says which route produced it.

Nothing in the repository reads a delta record, so this writes the same per-layer table shape the
reference script's `cols` dict holds and builds no shared reader (the Chief's ruling A2).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

#: The projections mlx-lm's LoRA adapted, and therefore the only module set whose per-layer number
#: is comparable to a depth-why row.
LORA_COMPARABLE_SUFFIXES: tuple[str, ...] = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)

_LAYER = re.compile(r"layers\.(\d+)\.")


class CheckpointDeltaError(ValueError):
    """A comparison that cannot be made, as opposed to one that comes out zero."""


def layer_of(key: str) -> int | None:
    """The block index a parameter belongs to, or ``None`` for embeddings and the final norm."""
    found = _LAYER.search(key)
    return int(found.group(1)) if found else None


def _shard_map(directory: Path) -> dict[str, str]:
    """Map every tensor name to the file holding it, for sharded and single-file checkpoints."""
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        return dict(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    single = directory / "model.safetensors"
    if not single.is_file():
        raise CheckpointDeltaError(f"{directory} holds neither model.safetensors nor an index")
    from safetensors import safe_open

    with safe_open(str(single), framework="np") as handle:
        # `.keys()` and not iteration: ruff's SIM118 rewrites this to `for key in handle`,
        # which assumes a mapping. A safetensors handle is not one -- it raises
        # `'builtins.safe_open' object is not iterable` -- so the rule is wrong here.
        return {key: single.name for key in handle.keys()}  # noqa: SIM118


def _iter_common_weights(
    finetuned: Path, base: Path, suffixes: tuple[str, ...] | None
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(key, tuned, base)`` one tensor at a time; never holds two checkpoints in memory."""
    import numpy as np
    from safetensors import safe_open

    tuned_map, base_map = _shard_map(finetuned), _shard_map(base)
    shared = sorted(set(tuned_map) & set(base_map))
    if not shared:
        raise CheckpointDeltaError(
            f"{finetuned} and {base} share no tensor names; they are not the same architecture"
        )
    wanted = [
        key for key in shared
        if key.endswith(".weight")
        and (suffixes is None or any(key.endswith(f"{s}.weight") for s in suffixes))
    ]
    if not wanted:
        raise CheckpointDeltaError(
            "no weights matched the requested module set; an empty set must be an error and not "
            "an answer (R56(f))"
        )
    handles: dict[Path, object] = {}
    try:
        for key in wanted:
            arrays = []
            for directory, mapping in ((finetuned, tuned_map), (base, base_map)):
                path = directory / mapping[key]
                if path not in handles:
                    handles[path] = safe_open(str(path), framework="np").__enter__()
                arrays.append(np.asarray(handles[path].get_tensor(key), dtype=np.float64))
            yield key, arrays[0], arrays[1]
    finally:
        for handle in handles.values():
            handle.__exit__(None, None, None)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class LayerPerturbation:
    """One row of the per-layer table, with the aggregates it was formed from."""

    layer: int
    relative_perturbation: float
    delta_norm: float
    base_norm: float
    modules: int


def per_layer_perturbation(
    finetuned: Path | str,
    base: Path | str,
    *,
    suffixes: tuple[str, ...] | None = LORA_COMPARABLE_SUFFIXES,
) -> list[LayerPerturbation]:
    """The depth-why quantity: numerator and denominator aggregated separately, then divided."""
    import numpy as np

    numerator: dict[int, float] = {}
    denominator: dict[int, float] = {}
    counts: dict[int, int] = {}
    for key, tuned, original in _iter_common_weights(Path(finetuned), Path(base), suffixes):
        index = layer_of(key)
        if index is None:
            continue
        delta = float(np.linalg.norm(tuned - original))
        weight = float(np.linalg.norm(original))
        if not np.isfinite(delta) or not np.isfinite(weight):
            raise CheckpointDeltaError(f"{key}: non-finite norm ({delta}, {weight})")
        if weight == 0.0:
            # A zero base weight has no scale to be relative to. Distinct from a zero delta, which
            # is what a frozen layer legitimately produces and must not raise.
            raise CheckpointDeltaError(f"{key}: base weight has zero norm, so a ratio is undefined")
        numerator[index] = numerator.get(index, 0.0) + delta**2
        denominator[index] = denominator.get(index, 0.0) + weight**2
        counts[index] = counts.get(index, 0) + 1

    rows = []
    for index in sorted(numerator):
        num, den = numerator[index] ** 0.5, denominator[index] ** 0.5
        rows.append(
            LayerPerturbation(
                layer=index,
                relative_perturbation=num / den,
                delta_norm=num,
                base_norm=den,
                modules=counts[index],
            )
        )
    return rows


def checkpoint_delta_record(
    finetuned: Path | str, base: Path | str, *, name: str
) -> dict[str, object]:
    """Both module sets, labelled, in the shape the reference script's `cols` dict holds."""
    restricted = per_layer_perturbation(finetuned, base, suffixes=LORA_COMPARABLE_SUFFIXES)
    every = per_layer_perturbation(finetuned, base, suffixes=None)
    return {
        "name": name,
        "finetuned": str(finetuned),
        "base": str(base),
        "quantity": "sqrt(sum_m ||dW_m||_F^2) / sqrt(sum_m ||W_m||_F^2), per layer",
        "reference": "research/records/DEPTH-WHY-2026-09-08/adapter_geometry.py",
        "module_sets": {
            "lora_comparable": {
                "suffixes": list(LORA_COMPARABLE_SUFFIXES),
                "note": "the seven projections LoRA adapted; the only set comparable to a "
                        "depth-why row",
                "cols": {row.layer: row.relative_perturbation for row in restricted},
                "rows": [vars(row) for row in restricted],
            },
            "all_weights": {
                "suffixes": None,
                "note": "every `.weight` full fine-tuning moves, including norms; honest about "
                        "full fine-tuning and not comparable to a LoRA record",
                "cols": {row.layer: row.relative_perturbation for row in every},
                "rows": [vars(row) for row in every],
            },
        },
        "no_analogue": (
            "adapter_geometry's cross-adapter cosine and its effective rank take an exact SVD "
            "through a QR of the rank-r factors. A full weight difference has no rank-r core, so "
            "neither computation carries over; effective rank here would be the SVD of the "
            "materialised difference, a different route to the same definition."
        ),
    }
