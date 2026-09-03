from __future__ import annotations

import pytest


def test_stage_select_refuses_empty_checkpoint_directory(tmp_path) -> None:
    """Selection with no checkpoint weights must stop before any evaluation is attempted."""
    from local_llm_lab.pipeline.cli import stage_select

    output = tmp_path / "run"
    adapters = output / "adapters"
    adapters.mkdir(parents=True)
    (adapters / "adapter_config.json").write_text("{}", encoding="utf-8")
    config = {"output": output, "select": {"limit": 1, "split": "valid"}}

    with pytest.raises(SystemExit, match="no checkpoint directories.*train"):
        stage_select(config, limit=None, quiet=True)
