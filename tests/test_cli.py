from __future__ import annotations

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
