"""The depth-why quantity over full weight differences, and the mistake it must not make.

`adapter_geometry.py` line 7 records its own author's first error: summing per-module *ratios* in
quadrature instead of aggregating numerator and denominator separately. It inflates by roughly the
square root of the module count, and both forms produce a plausible per-layer table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from local_llm_lab.probes.checkpoint_delta import (
    LORA_COMPARABLE_SUFFIXES,
    CheckpointDeltaError,
    checkpoint_delta_record,
    per_layer_perturbation,
)

pytest.importorskip("safetensors")

LAYERS, DIM = 3, 4


def _write(directory: Path, tensors: dict[str, np.ndarray]) -> Path:
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.astype(np.float32) for k, v in tensors.items()}, str(directory / "model.safetensors"))
    return directory


def _pair(tmp_path: Path, *, scale: float = 0.1, extra: bool = True):
    """A base and a 'fine-tuned' copy differing by a known multiple of the base in every module."""
    rng = np.random.default_rng(0)
    base, tuned = {}, {}
    for layer in range(LAYERS):
        for suffix in LORA_COMPARABLE_SUFFIXES:
            key = f"model.layers.{layer}.{suffix}.weight"
            weight = rng.normal(size=(DIM, DIM))
            base[key] = weight
            tuned[key] = weight * (1.0 + scale)
        if extra:
            # A norm, which full fine-tuning moves and LoRA never touched.
            key = f"model.layers.{layer}.input_layernorm.weight"
            weight = rng.normal(size=(DIM,)) + 3.0
            base[key] = weight
            tuned[key] = weight * (1.0 + scale)
    return _write(tmp_path / "base", base), _write(tmp_path / "tuned", tuned)


def test_the_ratio_is_the_separately_aggregated_one(tmp_path: Path) -> None:
    """Every module moved by the same relative amount, so the layer ratio is exactly that amount."""
    base, tuned = _pair(tmp_path, scale=0.1)
    rows = per_layer_perturbation(tuned, base)

    assert [row.layer for row in rows] == list(range(LAYERS))
    for row in rows:
        assert row.modules == len(LORA_COMPARABLE_SUFFIXES)
        # Checkpoints store float32, so the recovered ratio carries that rounding and no more:
        # the observed error is 2.7e-8 relative, well inside one float32 ulp.
        assert row.relative_perturbation == pytest.approx(0.1, rel=1e-6)


def test_it_is_not_the_quadrature_sum_of_per_module_ratios(tmp_path: Path) -> None:
    """The named mistake, quantified: it inflates by sqrt(module count)."""
    base, tuned = _pair(tmp_path, scale=0.1)
    correct = per_layer_perturbation(tuned, base)[0].relative_perturbation

    modules = len(LORA_COMPARABLE_SUFFIXES)
    mistaken = (sum(0.1**2 for _ in range(modules))) ** 0.5  # quadrature sum of per-module ratios

    assert correct == pytest.approx(0.1, rel=1e-6)
    assert mistaken == pytest.approx(0.1 * modules**0.5, rel=1e-9)
    assert mistaken > correct * 2.6  # sqrt(7) = 2.646


def test_a_frozen_layer_is_zero_and_not_an_error(tmp_path: Path) -> None:
    """Under per-layer freezing an untouched layer legitimately moves by nothing."""
    base, tuned = _pair(tmp_path, scale=0.0)
    rows = per_layer_perturbation(tuned, base)
    assert all(row.relative_perturbation == 0.0 for row in rows)
    assert all(row.base_norm > 0 for row in rows)


def test_a_zero_base_weight_is_an_error_because_the_ratio_has_no_scale(tmp_path: Path) -> None:
    """The guards sit on opposite sides: zero numerator is an answer, zero denominator is not."""
    key = "model.layers.0.self_attn.q_proj.weight"
    base = _write(tmp_path / "base", {key: np.zeros((DIM, DIM))})
    tuned = _write(tmp_path / "tuned", {key: np.ones((DIM, DIM))})
    with pytest.raises(CheckpointDeltaError, match="zero norm"):
        per_layer_perturbation(tuned, base)


def test_the_two_module_sets_are_both_reported_and_differ(tmp_path: Path) -> None:
    """Restricted is comparable to a LoRA record; the full set is honest about full fine-tuning."""
    base, tuned = _pair(tmp_path, scale=0.1)
    record = checkpoint_delta_record(tuned, base, name="smoke")

    restricted = record["module_sets"]["lora_comparable"]
    every = record["module_sets"]["all_weights"]
    assert restricted["rows"][0]["modules"] == len(LORA_COMPARABLE_SUFFIXES)
    assert every["rows"][0]["modules"] == len(LORA_COMPARABLE_SUFFIXES) + 1
    # Same shape as the reference script's `cols`: {layer: ratio}.
    assert sorted(restricted["cols"]) == list(range(LAYERS))
    assert "no rank-r core" in record["no_analogue"]


def test_a_module_set_that_matches_nothing_is_an_error(tmp_path: Path) -> None:
    """An empty filter must be an error and not an answer (R56(f))."""
    base, tuned = _pair(tmp_path, scale=0.1)
    with pytest.raises(CheckpointDeltaError, match="empty set"):
        per_layer_perturbation(tuned, base, suffixes=("nothing.matches_this",))


def test_checkpoints_of_different_architectures_are_refused(tmp_path: Path) -> None:
    base = _write(tmp_path / "base", {"model.layers.0.self_attn.q_proj.weight": np.ones((DIM, DIM))})
    tuned = _write(tmp_path / "tuned", {"something.else.weight": np.ones((DIM, DIM))})
    with pytest.raises(CheckpointDeltaError, match="share no tensor names"):
        per_layer_perturbation(tuned, base)
