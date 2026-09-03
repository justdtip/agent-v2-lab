from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_llm_lab.pipeline import cli
from local_llm_lab.pipeline.tasks import FAMILIES, LONG_HORIZON_FAMILIES


def _summary(step: int, split: str) -> dict[str, object]:
    """Two deliberately opposed scores: macro favours step 100, micro favours step 200."""
    default_success = 1 if step == 100 else 0
    long_success = 1 if step == 100 else 3
    by_family = {
        family: {
            "successes": long_success if family in LONG_HORIZON_FAMILIES else default_success,
            "tasks": 3 if family in LONG_HORIZON_FAMILIES else 1,
        }
        for family in FAMILIES
    }
    successes = sum(item["successes"] for item in by_family.values())
    return {
        "split": split,
        "tasks": 24,
        "successes": successes,
        "clean_rate": 0.9,
        "valid_action_rate": 0.95,
        "by_family": by_family,
        "rate_counts": {
            "success": {"numerator": successes, "denominator": 24},
            "clean": {"numerator": 22, "denominator": 24},
            "valid_actions": {"numerator": 46, "denominator": 48},
        },
        "wilson_95": {"success": [0.0, 1.0]},
    }


def _row(
    name: str, macro: float, micro: float, clean: float, valid: float, loss: float | None, step: int
) -> dict[str, object]:
    return {
        "name": name,
        "components": {
            "family_macro_success": macro,
            "micro_success": micro,
            "clean_rate": clean,
            "valid_action_rate": valid,
        },
        "val_loss": loss,
        "step": step,
    }
def test_selection_key_orders_all_tiebreakers_and_missing_loss_last() -> None:
    """Catch any departure from macro, micro, clean, valid, loss, then earlier-step ranking."""
    rows = [
        _row("macro", 0.9, 0.1, 0.1, 0.1, None, 999),
        _row("micro", 0.8, 0.9, 0.1, 0.1, 9.0, 999),
        _row("clean", 0.8, 0.8, 0.9, 0.1, 9.0, 999),
        _row("valid", 0.8, 0.8, 0.8, 0.9, 9.0, 999),
        _row("loss", 0.8, 0.8, 0.8, 0.8, 0.1, 999),
        _row("earlier", 0.8, 0.8, 0.8, 0.8, 0.2, 10),
        _row("later", 0.8, 0.8, 0.8, 0.8, 0.2, 20),
        _row("missing", 0.8, 0.8, 0.8, 0.8, None, 1),
    ]

    assert [row["name"] for row in sorted(rows, key=cli._selection_key, reverse=True)] == [
        "macro", "micro", "clean", "valid", "loss", "earlier", "later", "missing"
    ]


def test_stage_select_aggregates_two_cells_and_writes_deterministic_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch a one-cell screen, micro-only family score, log-loss omission, or unstable winner."""
    output = tmp_path / "run"
    adapters = []
    for step in (100, 200):
        adapter = tmp_path / f"step-{step}"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        adapters.append((step, adapter))
    (output / "train.log").parent.mkdir(parents=True)
    (output / "train.log").write_text(
        "Iter 100: Val loss 0.200, Val took 87.010s\nIter 200: Val loss 0.100, Val took 87.010s\n",
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_evaluation(**kwargs):
        calls.append(kwargs)
        step = int(str(kwargs["label"]).split("-")[1])
        return _summary(step, str(kwargs["split"]))

    monkeypatch.setattr(cli, "checkpoint_dirs", lambda _config: adapters)
    monkeypatch.setattr(cli, "run_evaluation", fake_evaluation)
    config = {
        "output": output,
        "model": "fake-model",
        "seed": 17,
        "keep_last": 2,
        "eval": {"max_steps": 4, "max_tokens": 8},
        "select": {
            "screen": [
                {"split": "valid", "difficulty": 1, "per_family": {"default": 1, "long": 3}},
                {"split": "valid2", "difficulty": 2, "per_family": {"default": 1, "long": 3}},
            ]
        },
    }

    best = cli.stage_select(config, limit=None, quiet=True)
    first = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    cli.stage_select(config, limit=None, quiet=True)
    second = json.loads((output / "selection.json").read_text(encoding="utf-8"))

    assert len(calls) == 8
    assert all(call["limit"] is None for call in calls)
    assert {(call["split"], call["difficulty"]) for call in calls} == {
        ("valid", 1),
        ("valid2", 2),
    }
    assert all(call["family_quotas"] == {"default": 1, "long": 3} for call in calls)
    assert best == output / "best-adapter"
    assert (best / "adapter_config.json").is_file()
    assert first == second
    assert first["selected_step"] == 100
    assert first["behavior_best_step"] == 100
    assert first["loss_best_step"] == 200
    assert first["disagreement"] is True
    assert first["model"] == "fake-model" and first["data_seed"] == 17
    assert [cell["difficulty"] for cell in first["screen"]] == [1, 2]
    assert first["checkpoints"][0]["components"]["tasks"] == 48
    assert first["checkpoints"][0]["val_loss"] == pytest.approx(0.2)
