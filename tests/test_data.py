from __future__ import annotations

import copy
import hashlib
import json
import os

from local_llm_lab.agent_protocol import Action
from local_llm_lab.models import ChatSpec, LoraSpec, ModelSpec
from local_llm_lab.pipeline.data import build_rows, render_rows, write_dataset, write_jsonl
from local_llm_lab.pipeline.protocol import assistant_message
from local_llm_lab.pipeline.tasks import make_tasks


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
        return rendered + (suffix + "<think>\n\n</think>\n\n" if kwargs["enable_thinking"] is False else suffix)


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
    assert rendered[0]["completion"] == 'note\n```json\n{"name": "finish", "arguments": {"answer": "done"}}\n```<eot>'


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


def test_render_rows_preserves_legacy_prompt_bytes_and_independent_token_ids() -> None:
    tokenizer = _LegacyTokenizer()
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
        assistant_message("note", Action("finish", {"answer": "done"})),
    ]

    row = render_rows([{"messages": messages, "metadata": {}}], tokenizer, spec=_legacy_spec())[0]

    expected_prompt = "system:rules\nuser:task\n<|im_start|>assistant\n"
    expected_completion = 'note\n```json\n{"name": "finish", "arguments": {"answer": "done"}}\n```<eot>'
    assert row["prompt"] == expected_prompt
    assert row["completion"] == expected_completion
    assert tokenizer.encode(row["prompt"]) == [ord(character) for character in expected_prompt]
    assert tokenizer.encode(row["completion"]) == [ord(character) for character in expected_completion]


def test_render_rows_disables_inference_thinking_and_preserves_trained_reasoning() -> None:
    action = Action("finish", {"answer": "done"})
    thought = "<think>reason</think>\n\nnote\n```json\n{\"name\": \"finish\", \"arguments\": {\"answer\": \"done\"}}\n```"
    row = {"messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": thought}], "metadata": {}}
    inference_tokenizer = _ThinkingTokenizer()

    inference = render_rows([row], inference_tokenizer, spec=_thinking_spec("inference"))[0]
    trained = render_rows([row], _ThinkingTokenizer(), spec=_thinking_spec("trained"))[0]

    assert inference_tokenizer.template_kwargs == [{"enable_thinking": False}]
    assert "<think>reason</think>" not in inference["completion"]
    assert trained["completion"] == "<think>reason</think>\n\n" + assistant_message("note", action)["content"] + "<eot>"


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
