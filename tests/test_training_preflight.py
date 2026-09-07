"""Launch checks exercise the recorded hybrid recipe without importing MLX.

Recipe evidence: configs/agent_v2e_qwen35_4b.yaml and the 2026-09-06 training log.
"""

from __future__ import annotations

import copy

import pytest

from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.live_lens.preflight import training_preflight


def recipe():
    return {
        "gated_delta_mode": "chunkwise",
        "gated_delta_chunk": 256,
        "max_seq_length": 2688,
        "batch_size": 1,
        "grad_accumulation_steps": 4,
        "iters": 1200,
        "iters_unit": "batches",
        "metal_cache_gib": 2.0,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"gated_delta_mode": None}, "chunkwise"),
        ({"gated_delta_mode": "checkpointed"}, "chunkwise"),
        ({"gated_delta_chunk": 64}, "verified envelope"),
        ({"max_seq_length": 4096}, "verified envelope"),
        ({"batch_size": 2}, "verified envelope"),
        ({"iters_unit": "optimizer_steps"}, "batches"),
        ({"iters_unit": None}, "batches"),
        ({"iters": 1201}, "unapplied"),
        ({"metal_cache_gib": float("nan")}, "cache"),
        ({"metal_cache_gib": float("inf")}, "cache"),
        ({"metal_cache_gib": 0}, "cache"),
        ({"metal_cache_gib": 3}, "cache"),
    ],
)
def test_launch_refusals(change, message):
    with pytest.raises(ValueError, match=message):
        training_preflight(recipe() | change, load_model_spec("qwen35-4b"))


def test_preflight_reports_actual_batches_rows_and_optimizer_updates():
    config = recipe()
    original = copy.deepcopy(config)
    result = training_preflight(config, load_model_spec("qwen35-4b"))
    assert result["batches"] == result["rows"] == 1200
    assert result["optimizer_updates"] == 300
    assert result["metal_cache_bytes"] == 2 * 2**30
    assert config == original
    assert training_preflight({}, load_model_spec("qwen25-coder-3b")) is None


def test_invalid_recipe_fails_before_trainer_import_or_output_write(monkeypatch, tmp_path):
    from local_llm_lab.pipeline import cli

    events = []
    monkeypatch.setattr(cli, "_load_training_entry", lambda: events.append("trainer imported"))
    config = {
        "model": "qwen35-4b",
        "train": recipe() | {"gated_delta_mode": None},
        "output": tmp_path / "not_created",
    }
    with pytest.raises(ValueError, match="chunkwise"):
        cli.stage_train(config, iters=None)
    assert not events and not config["output"].exists()
