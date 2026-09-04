from __future__ import annotations

import sys

import pytest

from local_llm_lab import chat_replay
from local_llm_lab.chat_replay import make_chat_prompts
from local_llm_lab.pipeline import data as data_module
from local_llm_lab.pipeline.data import DatasetWriteGuardError, ProtectedDatasetError


def test_chat_prompt_splits_are_distinct_and_balanced() -> None:
    train = make_chat_prompts("train", 12)
    test = make_chat_prompts("test", 12)
    assert {item.prompt_id for item in train}.isdisjoint(item.prompt_id for item in test)
    assert {item.messages[-1]["content"] for item in train}.isdisjoint(
        item.messages[-1]["content"] for item in test
    )
    assert len({item.category for item in train}) == 6


def test_follow_up_prompts_have_conversation_history() -> None:
    prompts = make_chat_prompts("train", 6)
    follow_up = next(item for item in prompts if item.category == "follow_up")
    assert [message["role"] for message in follow_up.messages] == [
        "user",
        "assistant",
        "user",
    ]


def _temp_project_root(monkeypatch, tmp_path):
    """Point both the script and the guard at a temp project root (R10: never the real data/)."""
    monkeypatch.setattr(chat_replay, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(data_module, "PROJECT_ROOT", tmp_path)
    protected = tmp_path / "data" / "chat_replay"
    protected.mkdir(parents=True)
    return protected


def _weight_load_tripwire() -> None:
    raise AssertionError("the guard must refuse before any weight is loaded")


def test_main_with_its_default_output_refuses_the_protected_replay_directory(
    monkeypatch, tmp_path
) -> None:
    """R21: ``chat-replay-data`` run with no arguments aims at a PROTECTED_DATASETS member.

    The refusal fires before the teacher weights are loaded, so the installed console script
    can neither overwrite the irreplaceable directory nor spend a generation pass discovering
    that it may not write. Fakes only: the loader is a tripwire that fails if it is reached.
    """
    protected = _temp_project_root(monkeypatch, tmp_path)
    (protected / "train.jsonl").write_text('{"kept": true}\n', encoding="utf-8")

    monkeypatch.setattr(chat_replay, "configure_local_cache", _weight_load_tripwire)
    monkeypatch.setattr(sys, "argv", ["chat-replay-data"])

    with pytest.raises(ProtectedDatasetError, match="irreplaceable"):
        chat_replay.main()

    assert (protected / "train.jsonl").read_text(encoding="utf-8") == '{"kept": true}\n'
    assert not (protected / "manifest.json").exists()


def test_main_refuses_an_unprotected_target_that_already_holds_a_manifest(
    monkeypatch, tmp_path
) -> None:
    """R21: this stage has no override flag, so an existing manifest refuses outright."""
    _temp_project_root(monkeypatch, tmp_path)
    target = tmp_path / "replay-out"
    target.mkdir()
    (target / "manifest.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(chat_replay, "configure_local_cache", _weight_load_tripwire)
    monkeypatch.setattr(sys, "argv", ["chat-replay-data", "--output", str(target)])

    with pytest.raises(DatasetWriteGuardError, match="no overwrite path"):
        chat_replay.main()
    assert (target / "manifest.json").read_text(encoding="utf-8") == "{}\n"
