from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_llm_lab.pipeline import cli
from local_llm_lab.pipeline.cli import load_config, stage_data
from local_llm_lab.pipeline.data import SplitSpec


@pytest.mark.parametrize("name", ("agent_v2.yaml", "agent_v2b.yaml", "agent_v2c.yaml"))
def test_shipped_configs_define_only_the_complete_two_cell_selection_screen(name: str) -> None:
    """Catch a stale legacy split/limit selector in any shipped run configuration."""
    config = load_config(Path(__file__).parents[1] / "configs" / name)

    assert config["select"] == {
        "screen": [
            {"split": "valid", "difficulty": 1, "per_family": {"default": 1, "long": 3}},
            {"split": "valid2", "difficulty": 2, "per_family": {"default": 1, "long": 3}},
        ]
    }


def test_stage_data_normalizes_legacy_and_explicit_splits(monkeypatch, tmp_path) -> None:
    """Catch a data stage that passes raw YAML counts instead of SplitSpec values."""
    received = []
    def capture_dataset(*args, **kwargs):
        received.append((args, kwargs))
        return {"splits": {}}

    monkeypatch.setattr(cli, "write_dataset", capture_dataset)
    monkeypatch.setattr(cli, "write_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_model_spec", lambda name: object())
    base = {
        "model": "fake",
        "output": tmp_path / "out",
        "data": tmp_path / "data",
        "seed": 1,
        "keep_last": 2,
    }
    stage_data({**base, "tasks": {"train": 2, "valid": 3, "test": 4}}, [])
    legacy = received[-1][0][1]
    assert legacy == {
        "train": SplitSpec(2, role="train"),
        "valid": SplitSpec(3, role="valid"),
        "test": SplitSpec(4, role="test"),
    }
    explicit = {"train": {"count": 2, "difficulty": 0, "perturb": True, "role": "train"}}
    stage_data({**base, "splits": explicit}, [])
    assert received[-1][0][1] == {"train": SplitSpec(2, difficulty=0, perturb=True, role="train")}
    with pytest.raises(ValueError, match="exactly one"):
        stage_data({**base, "tasks": {"train": 1}, "splits": explicit}, [])
    with pytest.raises(ValueError, match="exactly one"):
        stage_data(base, [])


def test_stage_data_passes_the_exact_six_run_d_chunks_and_recovery_multipliers(
    monkeypatch, tmp_path
) -> None:
    """Stage the shipped recipe through the fake writer; no YAML dict may leak through."""
    received = []

    def capture_dataset(*args, **kwargs):
        received.append((args, kwargs))
        return {"splits": {}}

    monkeypatch.setattr(cli, "write_dataset", capture_dataset)
    monkeypatch.setattr(cli, "write_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_model_spec", lambda name: object())
    config = load_config(Path(__file__).parents[1] / "configs" / "agent_v2d.yaml")
    config["data"] = tmp_path / "data"
    config["output"] = tmp_path / "output"

    stage_data(config, [])

    assert received[-1][0][1] == {
        "train": SplitSpec(240, difficulty=0, perturb=True, role="train"),
        "train1": SplitSpec(120, difficulty=1, perturb=True, role="train"),
        "valid": SplitSpec(24, difficulty=1, perturb=False, role="valid"),
        "valid2": SplitSpec(24, difficulty=2, perturb=False, role="valid"),
        "test": SplitSpec(180, difficulty=2, perturb=False, role="test"),
        "test3": SplitSpec(60, difficulty=3, perturb=False, role="test"),
    }
    assert received[-1][1]["recovery_repeats"] == {
        "transient": 1,
        "wrong_path": 2,
        "unknown_tool": 2,
        "stale_path": 6,
        "failed_edit": 6,
    }


def test_run_d_configs_are_literal_pairwise_recipes() -> None:
    """Catch a data-recipe drift between D3/D4 or B/B4."""
    root = Path(__file__).parents[1] / "configs"
    d3 = load_config(root / "agent_v2d.yaml")
    d4 = load_config(root / "agent_v2d_qwen35_4b.yaml")
    b = load_config(root / "agent_v2b.yaml")
    b4 = load_config(root / "agent_v2b_qwen35_4b.yaml")
    for left, right in ((d3, d4), (b, b4)):
        for key in ("model", "output"):
            left.pop(key)
            right.pop(key)
        assert left == right
    assert d3["splits"] == {
        "train": {"count": 240, "difficulty": 0, "perturb": True, "role": "train"},
        "train1": {"count": 120, "difficulty": 1, "perturb": True, "role": "train"},
        "valid": {"count": 24, "difficulty": 1, "perturb": False, "role": "valid"},
        "valid2": {"count": 24, "difficulty": 2, "perturb": False, "role": "valid"},
        "test": {"count": 180, "difficulty": 2, "perturb": False, "role": "test"},
        "test3": {"count": 60, "difficulty": 3, "perturb": False, "role": "test"},
    }


def test_stage_select_writes_provenance_after_selection_json(monkeypatch, tmp_path: Path) -> None:
    """Selection provenance must capture the same durable payload without loading a model."""
    output = tmp_path / "output"
    output.mkdir()
    adapter = tmp_path / "step-10"
    adapter.mkdir()
    config = {
        "output": output,
        "model": "fake-model",
        "seed": 17,
        "keep_last": 2,
        "eval": {"max_steps": 2, "max_tokens": 3},
        "select": {
            "screen": [
                {"split": "valid", "difficulty": 1, "per_family": {"default": 1}}
            ]
        },
    }
    spec = object()
    calls = []
    model_lookups = []

    monkeypatch.setattr(cli, "checkpoint_dirs", lambda config: [(10, adapter)])
    monkeypatch.setattr(cli.Transcript, "start_run", lambda path: None)
    def capture_model_spec(model):
        model_lookups.append(model)
        return spec

    monkeypatch.setattr(cli, "load_model_spec", capture_model_spec)
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda **kwargs: {
            "rate_counts": {
                "success": {"numerator": 1, "denominator": 1},
                "clean": {"numerator": 1, "denominator": 1},
                "valid_actions": {"numerator": 1, "denominator": 1},
            },
            "by_family": {"read": {"successes": 1, "tasks": 1}},
        },
    )

    def capture_provenance(run_dir, *, resolved, spec, extra):
        assert (output / "selection.json").is_file()
        calls.append((run_dir, resolved, spec, extra))

    monkeypatch.setattr(cli, "write_provenance", capture_provenance)

    assert cli.stage_select(config, limit=None, quiet=True) == output / "best-adapter"

    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert calls == [(output, None, spec, {"stage": "select", "selection": selection})]
    assert model_lookups == [config["model"]]
