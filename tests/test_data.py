from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict

import pytest

from local_llm_lab.agent_protocol import Action
from local_llm_lab.models import ChatSpec, LoraSpec, ModelSpec, load_model_spec
from local_llm_lab.pipeline.data import (
    SplitSpec,
    build_rows,
    read_jsonl,
    render_rows,
    write_dataset,
    write_jsonl,
)
from local_llm_lab.pipeline.protocol import assistant_message
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION, make_tasks
from local_llm_lab.tuner_data import RenderedRowsDataset


def _legacy_spec() -> ModelSpec:
    return ModelSpec(
        name="legacy",
        hf_id="legacy",
        family="legacy",
        chat=ChatSpec("unsupported", {}, "<eot>", ()),
        lora=LoraSpec("attention+mlp", 1, 1.0, 0.0),
        train={},
        cache_strategy="none",
        probe_layer_fractions=(1.0,),
        memory_budget_gib=1.0,
        policies={},
        cache_equivalence_verified=None,
    )


def _thinking_spec(mode: str) -> ModelSpec:
    return ModelSpec(
        name=mode,
        hf_id=mode,
        family=mode,
        chat=ChatSpec(mode, {"enable_thinking": True}, "<eot>", ()),  # type: ignore[arg-type]
        lora=LoraSpec("attention+mlp", 1, 1.0, 0.0),
        train={},
        cache_strategy="none",
        probe_layer_fractions=(1.0,),
        memory_budget_gib=1.0,
        policies={},
        cache_equivalence_verified=None,
    )


class _LegacyTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs) -> str:
        assert not tokenize and not kwargs
        rendered = "".join(f"{message['role']}:{message['content']}\n" for message in messages)
        return rendered + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


class _ThinkingTokenizer(_LegacyTokenizer):
    def __init__(self) -> None:
        self.template_kwargs: list[dict[str, object]] = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs) -> str:
        self.template_kwargs.append(kwargs)
        rendered = "".join(f"{message['role']}:{message['content']}\n" for message in messages)
        if not add_generation_prompt:
            return rendered
        suffix = "<|im_start|>assistant\n"
        return rendered + (
            suffix + "<think>\n\n</think>\n\n"
            if kwargs["enable_thinking"] is False
            else suffix
        )


class _Qwen25TemplateTokenizer:
    """Byte-level fake for the registered 3B Qwen2.5 chat-template contract."""

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs) -> str:
        assert not tokenize and not kwargs
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        return rendered + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return list(text.encode("utf-8"))


def _legacy_qwen25_messages(messages: list[dict[str, object]]) -> str:
    return "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    )


def test_render_rows_adds_canonical_fields_without_mutating_messages() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "task"},
            assistant_message("note", Action("finish", {"answer": "done"})),
        ],
        "metadata": {"task_id": "fake"},
    }
    original = copy.deepcopy(row)

    rendered = render_rows([row], _LegacyTokenizer(), spec=_legacy_spec())

    assert row == original
    assert rendered[0]["messages"] == original["messages"]
    assert rendered[0]["metadata"] == {"task_id": "fake"}
    assert rendered[0]["prompt"] == "system:rules\nuser:task\n<|im_start|>assistant\n"
    assert rendered[0]["completion"] == (
        'note\n```json\n{"name": "finish", "arguments": {"answer": "done"}}\n```<eot>\n'
    )


def test_write_jsonl_atomically_replaces_payloads_at_or_above_one_mebibyte(
    monkeypatch, tmp_path
) -> None:
    destination = tmp_path / "rows.jsonl"
    replacements: list[tuple[str, str]] = []
    original_replace = os.replace

    def record_replace(source: str, target: str) -> None:
        replacements.append((source, target))
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", record_replace)
    digest = write_jsonl(destination, [{"payload": "x" * (1024 * 1024)}])

    assert len(replacements) == 1
    assert replacements[0][1] == str(destination)
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_write_jsonl_atomically_replaces_sub_mebibyte_array_payloads(monkeypatch, tmp_path) -> None:
    destination = tmp_path / "rows.jsonl"
    replacements: list[tuple[str, str]] = []
    original_replace = os.replace

    def record_replace(source: str, target: str) -> None:
        replacements.append((source, target))
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", record_replace)

    write_jsonl(destination, [{"messages": [{"role": "user", "content": "small"}]}])

    assert len(replacements) == 1
    assert replacements[0][1] == str(destination)


def test_write_dataset_emits_rendered_rows_when_given_a_tokenizer_and_spec(tmp_path) -> None:
    write_dataset(
        tmp_path,
        {"train": 12},
        tokenizer=_LegacyTokenizer(),
        spec=_legacy_spec(),
    )

    row = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert {"messages", "metadata", "prompt", "completion"} <= set(row)


def test_write_dataset_records_the_effective_inference_training_renderer(tmp_path) -> None:
    manifest = write_dataset(
        tmp_path,
        {"train": 12},
        tokenizer=_ThinkingTokenizer(),
        spec=_thinking_spec("inference"),
    )

    assert manifest["rendering"]["thinking"] == "off"
    assert manifest["rendering"]["template_kwargs"] == {"enable_thinking": False}


def test_write_dataset_records_the_complete_unresolved_spec_without_resolving(
    monkeypatch, tmp_path
) -> None:
    spec = _thinking_spec("inference")

    def forbid_resolve(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("write_dataset must not resolve a model specification")

    monkeypatch.setattr(ModelSpec, "resolve", forbid_resolve)

    manifest = write_dataset(
        tmp_path,
        {"train": 12},
        tokenizer=_ThinkingTokenizer(),
        spec=spec,
    )

    assert manifest["model"] == asdict(spec)
    assert manifest["rendering"]["thinking"] == "off"


def test_current_generator_qwen25_messages_migrate_to_identical_rendered_tokens() -> None:
    assert GENERATOR_VERSION == 4
    tokenizer = _Qwen25TemplateTokenizer()
    spec = load_model_spec("qwen25-coder-3b")
    row = build_rows(make_tasks("train", 12, seed=20260902)[0])[0]
    legacy_render = _legacy_qwen25_messages(row["messages"])
    legacy_tokens = list(legacy_render.encode("utf-8"))

    rendered = render_rows([row], tokenizer, spec=spec)[0]
    dataset = RenderedRowsDataset(
        [rendered], tokenizer, max_seq_length=len(legacy_tokens) + 1
    )
    tokens, offset = dataset[0]

    assert rendered["prompt"] + rendered["completion"] == legacy_render
    assert tokens == legacy_tokens
    assert offset == len(rendered["prompt"].encode("utf-8"))


def test_render_rows_disables_inference_thinking_and_preserves_trained_reasoning() -> None:
    action = Action("finish", {"answer": "done"})
    thought = (
        "<think>reason</think>\n\nnote\n```json\n"
        '{"name": "finish", "arguments": {"answer": "done"}}\n```'
    )
    row = {
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": thought},
        ],
        "metadata": {},
    }
    inference_tokenizer = _ThinkingTokenizer()

    inference = render_rows([row], inference_tokenizer, spec=_thinking_spec("inference"))[0]
    trained = render_rows([row], _ThinkingTokenizer(), spec=_thinking_spec("trained"))[0]

    assert inference_tokenizer.template_kwargs == [{"enable_thinking": False}]
    assert "<think>reason</think>" not in inference["completion"]
    assert trained["completion"] == (
        "<think>reason</think>\n\n"
        + assistant_message("note", action)["content"]
        + "<eot>\n"
    )


def test_recovery_rows_are_marked_and_oversampled_in_train_only(tmp_path) -> None:
    task = next(t for t in make_tasks("train", 144) if t.variant == "failed_edit")
    rows = build_rows(task)
    marked = [row for row in rows if row["metadata"]["recovery"]]
    assert len(marked) == 1, "exactly the step after the deliberate failure is a recovery"

    manifest = write_dataset(tmp_path, {"train": 24, "valid": 12, "test": 12}, recovery_repeats=5)
    train = manifest["splits"]["train"]
    assert train["recovery_rows_after_repeats"] == train["recovery_targets"] * 5
    for split in ("valid", "test"):
        held = manifest["splits"][split]
        assert held["recovery_rows_after_repeats"] == held["recovery_targets"]


def test_split_specs_aggregate_named_chunks_by_role_deterministically(tmp_path) -> None:
    """Catch lost difficulty/perturb flags or accidental cross-role concatenation."""
    splits = {
        "train": SplitSpec(2, difficulty=0, perturb=True, role="train"),
        "train1": SplitSpec(2, difficulty=1, perturb=True, role="train"),
        "valid": SplitSpec(2, difficulty=1, perturb=False, role="valid"),
        "valid2": SplitSpec(2, difficulty=2, perturb=False, role="valid"),
        "test": SplitSpec(2, difficulty=2, perturb=False, role="test"),
        "test3": SplitSpec(2, difficulty=3, perturb=False, role="test"),
    }
    manifest = write_dataset(tmp_path, splits, recovery_repeats={"wrong_path": 2})

    split_metadata = {
        (name, info["difficulty"], info["perturb"], info["role"])
        for name, info in manifest["splits"].items()
    }
    assert split_metadata == {
        ("train", 0, True, "train"),
        ("train1", 1, True, "train"),
        ("valid", 1, False, "valid"),
        ("valid2", 2, False, "valid"),
        ("test", 2, False, "test"),
        ("test3", 3, False, "test"),
    }
    roles = {
        "train": {"train", "train1"},
        "valid": {"valid", "valid2"},
        "test": {"test", "test3"},
    }
    for role, names in roles.items():
        task_ids = {
            row["metadata"]["task_id"].split("-", 1)[0]
            for row in read_jsonl(tmp_path / f"{role}.jsonl")
            if row["metadata"]["source"] == "expert"
        }
        assert task_ids <= names
        assert task_ids == names
        assert manifest["outputs"][role]["rows"] == len(read_jsonl(tmp_path / f"{role}.jsonl"))


def _logical_hash(rows: list[dict[str, object]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_split_specs_preserve_chunk_boundaries_effects_hashes_and_recovery_repeats(
    tmp_path,
) -> None:
    """Independently prove the six logical chunks concatenate in declaration order."""
    splits = {
        "train": SplitSpec(24, difficulty=0, perturb=True, role="train"),
        "train1": SplitSpec(24, difficulty=1, perturb=True, role="train"),
        "valid": SplitSpec(24, difficulty=1, perturb=False, role="valid"),
        "valid2": SplitSpec(24, difficulty=2, perturb=False, role="valid"),
        "test": SplitSpec(24, difficulty=2, perturb=False, role="test"),
        "test3": SplitSpec(24, difficulty=3, perturb=False, role="test"),
    }
    repeats = {"wrong_path": 2, "stale_path": 6, "failed_edit": 6}
    first = write_dataset(tmp_path / "one", splits, recovery_repeats=repeats)
    second = write_dataset(tmp_path / "two", splits, recovery_repeats=repeats)

    for role, chunk_names in {
        "train": ("train", "train1"),
        "valid": ("valid", "valid2"),
        "test": ("test", "test3"),
    }.items():
        first_bytes = (tmp_path / "one" / f"{role}.jsonl").read_bytes()
        assert first_bytes == (tmp_path / "two" / f"{role}.jsonl").read_bytes()
        rows = read_jsonl(tmp_path / "one" / f"{role}.jsonl")
        offset = 0
        for name in chunk_names:
            info = first["splits"][name]
            chunk = rows[offset : offset + info["rows"]]
            offset += info["rows"]
            assert len(chunk) == info["rows"]
            assert {row["metadata"]["task_id"].split("-", 1)[0] for row in chunk} == {name}
            assert _logical_hash(chunk) == info["sha256"]
            assert {row["metadata"]["difficulty"] for row in chunk} == {info["difficulty"]}
            assert {row["metadata"]["perturb"] for row in chunk} == {info["perturb"]}
            if info["perturb"]:
                assert {row["metadata"]["variant"] for row in chunk} != {"clean"}
        assert offset == len(rows)
        assert hashlib.sha256(first_bytes).hexdigest() == first["outputs"][role]["sha256"]

    for name in ("train", "train1"):
        info = first["splits"][name]
        assert info["recovery_targets"] > 0
        assert info["recovery_rows_after_repeats"] > info["recovery_targets"]
    assert first == second
    assert (tmp_path / "one" / "manifest.json").read_bytes() == (
        tmp_path / "two" / "manifest.json"
    ).read_bytes()


def test_split_specs_add_role_replay_once_and_reject_invalid_values(tmp_path) -> None:
    """Catch replay duplication on secondary chunks and malformed split declarations."""
    chat = tmp_path / "chat"
    chat.mkdir()
    for role in ("train", "valid", "test"):
        write_jsonl(chat / f"{role}.jsonl", [{"metadata": {"source": "chat", "role": role}}])
    extra = tmp_path / "extra"
    extra.mkdir()
    write_jsonl(extra / "train.jsonl", [{"metadata": {"source": "extra"}}])
    splits = {
        "train": SplitSpec(1, role="train"), "train1": SplitSpec(1, role="train"),
        "valid": SplitSpec(1, role="valid"), "valid2": SplitSpec(1, role="valid"),
        "test": SplitSpec(1, role="test"), "test3": SplitSpec(1, role="test"),
    }
    manifest = write_dataset(tmp_path / "out", splits, chat_dir=chat, extra_dirs=[extra])
    assert {
        role: manifest["outputs"][role]["chat_rows"]
        for role in ("train", "valid", "test")
    } == {"train": 1, "valid": 1, "test": 1}
    assert manifest["outputs"]["train"]["extra_rows"] == 1
    with pytest.raises(ValueError):
        SplitSpec(0)
    with pytest.raises(ValueError):
        SplitSpec(1, difficulty=-1)
    with pytest.raises(ValueError):
        SplitSpec(1, perturb="yes")  # type: ignore[arg-type]
