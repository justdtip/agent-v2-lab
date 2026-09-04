from __future__ import annotations

import sys

import pytest

from local_llm_lab import build_expanded_data
from local_llm_lab.build_expanded_data import DECISION_REPEATS, decision_balanced, target_action
from local_llm_lab.pipeline import data as data_module
from local_llm_lab.pipeline.data import DatasetWriteGuardError, ProtectedDatasetError


def _tool_row(name: str) -> dict:
    return {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": name, "arguments": "{}"}}],
            }
        ]
    }


def test_target_action_distinguishes_tools_from_chat() -> None:
    assert target_action(_tool_row("set_plan")) == "set_plan"
    assert target_action({"messages": [{"role": "assistant", "content": "hello"}]}) is None


def test_decision_balancing_oversamples_high_level_actions() -> None:
    plan = _tool_row("set_plan")
    read = _tool_row("read_file")
    rows = decision_balanced([plan, read])
    assert rows.count(plan) == DECISION_REPEATS["set_plan"]
    assert rows.count(read) == 1


def _temp_project_root(monkeypatch, tmp_path):
    """R10: a temp project root, so this test can never see the real ``data/``."""
    monkeypatch.setattr(data_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(build_expanded_data, "PROJECT_ROOT", tmp_path)
    protected = tmp_path / "data" / "agent_v2c"
    protected.mkdir(parents=True)
    return protected


def test_main_refuses_a_protected_target(monkeypatch, tmp_path) -> None:
    """R21: ``expanded-agent-data`` refuses before it reads or mkdirs anything."""
    protected = _temp_project_root(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["expanded-agent-data", "--output", str(protected)])
    with pytest.raises(ProtectedDatasetError, match="irreplaceable"):
        build_expanded_data.main()
    assert list(protected.iterdir()) == []


def test_main_refuses_an_existing_manifest(monkeypatch, tmp_path) -> None:
    """This stage has no override flag, so an existing manifest refuses outright."""
    _temp_project_root(monkeypatch, tmp_path)
    target = tmp_path / "expanded"
    target.mkdir()
    (target / "manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["expanded-agent-data", "--output", str(target)])
    with pytest.raises(DatasetWriteGuardError, match="no overwrite path"):
        build_expanded_data.main()
    assert (target / "manifest.json").read_text(encoding="utf-8") == "{}\n"
    assert sorted(path.name for path in target.iterdir()) == ["manifest.json"]
