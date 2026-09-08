from __future__ import annotations

from typing import Any

import pytest

from local_llm_lab.agent_protocol import TOOL_SPECS, Action, ActionParseError
from local_llm_lab.models import ChatSpec, LoraSpec, ModelSpec
from local_llm_lab.pipeline.protocol import (
    SYSTEM_PROMPT,
    Step,
    assistant_message,
    build_prompt,
    generation_suffix,
    parse_turn,
    render_completion,
    render_tools,
    render_turn,
    strip_thinking,
    system_prompt,
    tool_message,
    turn_is_complete,
    window_messages,
)


def _spec(mode: str) -> ModelSpec:
    return ModelSpec(
        name=f"fake-{mode}",
        hf_id="fake",
        family="fake",
        chat=ChatSpec(
            mode,  # type: ignore[arg-type]
            {"enable_thinking": mode != "off"} if mode != "unsupported" else {},
            "<eot>",
            (),
            generation_prefix="<|im_start|>assistant\n",
        ),
        lora=LoraSpec("attention+mlp", 1, 1.0, 0.0),
        train={},
        cache_strategy="none",
        probe_layer_fractions=(1.0,),
        memory_budget_gib=1.0,
        policies={},
    )


class _RenderingTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.calls.append({"messages": messages, **kwargs})
        base = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if not kwargs["add_generation_prompt"]:
            return base
        suffix = "<|im_start|>assistant\n"
        if kwargs.get("enable_thinking") is False:
            suffix += "<think>\n\n</think>\n\n"
        return base + suffix


@pytest.mark.parametrize("mode", ["unsupported", "off", "inference", "trained"])
def test_build_prompt_uses_spec_template_policy_and_declared_suffix(mode: str) -> None:
    spec = _spec(mode)
    tokenizer = _RenderingTokenizer()

    prompt = build_prompt(tokenizer, [{"role": "user", "content": "hello"}], spec=spec)

    assert prompt.endswith(generation_suffix(spec))
    assert tokenizer.calls[-1] == {
        "messages": [{"role": "user", "content": "hello"}],
        "add_generation_prompt": True,
        "tokenize": False,
        **spec.chat.template_kwargs,
    }


def test_strip_thinking_extracts_only_the_first_closed_block() -> None:
    thinking, remainder = strip_thinking(
        "<think>plan</think>\n\nNote\n```json\n{}\n```<think>literal</think>"
    )

    assert thinking == "plan"
    assert remainder == "Note\n```json\n{}\n```<think>literal</think>"


def test_open_thinking_blocks_completion_and_spec_rendering_declares_end_of_turn() -> None:
    action = Action("finish", {"answer": "done"})
    spec = _spec("inference")

    assert render_completion("note", action, spec=spec) == render_turn("note", action) + "<eot>\n"
    assert parse_turn(render_turn("note", action)).thinking is None
    assert not turn_is_complete("<think>```json\n{}\n```\n" + render_turn("note", action))


def test_build_prompt_rejects_a_wrong_suffix_and_skips_the_check_without_generation() -> None:
    spec = _spec("unsupported")

    class _WrongSuffixTokenizer:
        def apply_chat_template(self, messages, **kwargs) -> str:
            del messages
            return "context" if not kwargs["add_generation_prompt"] else "wrong"

    with pytest.raises(ValueError, match="generation suffix"):
        build_prompt(_WrongSuffixTokenizer(), [], spec=spec)
    assert build_prompt(_WrongSuffixTokenizer(), [], spec=spec, generation=False) == "context"


def test_build_prompt_rejects_a_tools_keyword() -> None:
    """Task 6: the dead ``tools=`` seam is gone; it was accepted and silently discarded."""
    spec = _spec("unsupported")
    tokenizer = _RenderingTokenizer()

    with pytest.raises(TypeError, match="tools"):
        build_prompt(tokenizer, [{"role": "user", "content": "hello"}], tools=TOOL_SPECS, spec=spec)

    assert tokenizer.calls == []


def test_window_hides_only_older_observations() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        assistant_message("note 1", Action("read_file", {"path": "a"})),
        tool_message("read_file", "content a\nline 2"),
        assistant_message("note 2", Action("read_file", {"path": "b"})),
        tool_message("read_file", "content b"),
        assistant_message("note 3", Action("read_file", {"path": "c"})),
        tool_message("read_file", "content c"),
    ]
    windowed = window_messages(messages, keep_last=2)
    assert windowed[3]["content"].startswith("[earlier read_file result hidden: 2 line(s)")
    assert windowed[5]["content"] == "content b"
    assert windowed[7]["content"] == "content c"
    assert [parse_turn(m["content"]).thought for m in windowed if m["role"] == "assistant"] == [
        "note 1",
        "note 2",
        "note 3",
    ]
    assert messages[3]["content"] == "content a\nline 2", "input is not mutated"


def test_assistant_message_is_content_only_with_fenced_call() -> None:
    action = Action("read_file", {"path": "a"})
    message = assistant_message("note", action)
    assert set(message) == {"role", "content"} and message["role"] == "assistant"
    assert "tool_calls" not in message
    assert message["content"] == render_turn("note", action)
    assert (
        message["content"]
        == 'note\n```json\n{"name": "read_file", "arguments": {"path": "a"}}\n```'
    )
    assert (
        render_turn("", action) == '```json\n{"name": "read_file", "arguments": {"path": "a"}}\n```'
    )


def test_parse_turn_round_trips_render_turn() -> None:
    action = Action("replace_text", {"path": "a/b.txt", "old": "x {y}", "new": "```z```"})
    thought = "Collected: a=1, b=2.\nNext: replace then re-read."
    turn = parse_turn(render_turn(thought, action))
    assert turn.thought == thought and turn.action == action
    bare = parse_turn(render_turn("", action))
    assert bare.thought == "" and bare.action == action
    with_end = parse_turn(render_turn(thought, action) + "<|im_end|>")
    assert with_end.thought == thought and with_end.action == action


def test_parse_turn_ignores_trailing_junk_after_first_call() -> None:
    first = Action("read_file", {"path": "x"})
    junk = (
        '\n spep\n{"name": "finish", "arguments": {"answer": "no"}}\n```json\n'
        '{"name": "list_files", "arguments": {"directory": "/"}}\n```'
    )
    turn = parse_turn(render_turn("n", first) + junk)
    assert turn.thought == "n" and turn.action == first
    cut = parse_turn(render_turn("n", first) + "\n``` `` <|im_end|>garbage")
    assert cut.action == first


def test_parse_turn_accepts_legacy_and_raw_forms() -> None:
    native = (
        'Reading the file.\n<tool_call>\n'
        '{"name": "read_file", "arguments": {"path": "x"}}\n</tool_call>'
    )
    turn = parse_turn(native)
    assert turn.thought == "Reading the file." and turn.action == Action("read_file", {"path": "x"})
    unlabelled = parse_turn('note\n```\n{"name": "finish", "arguments": {"answer": "1"}}\n```')
    assert unlabelled.action.name == "finish" and unlabelled.thought == "note"
    raw = parse_turn('{"name": "finish", "arguments": "{\\"answer\\": \\"2\\"}"}')
    assert raw.action.arguments == {"answer": "2"} and raw.thought == ""
    bare_with_note = parse_turn('done {"name": "finish", "arguments": {"answer": "3"}} trailing')
    assert bare_with_note.thought == "done" and bare_with_note.action.arguments == {"answer": "3"}


def test_parse_turn_rejects_missing_or_malformed_calls() -> None:
    with pytest.raises(ActionParseError, match="no tool call"):
        parse_turn("I will just think about it.")
    with pytest.raises(ActionParseError, match="no JSON object"):
        parse_turn("note\n```json\nnothing here\n```")
    with pytest.raises(ActionParseError, match="invalid JSON"):
        parse_turn('note\n```json\n{"name": "finish", "arguments": \n```')
    with pytest.raises(ActionParseError, match="no name"):
        parse_turn('note\n```json\n{"arguments": {"answer": "x"}}\n```')
    with pytest.raises(ActionParseError, match="JSON object"):
        parse_turn('note\n```json\n{"name": "finish", "arguments": [1]}\n```')


def test_turn_is_complete_requires_closed_fence() -> None:
    action = Action("finish", {"answer": "x"})
    rendered = render_turn("note", action)
    assert turn_is_complete(rendered)
    assert turn_is_complete(rendered + "\nJUNK")
    assert not turn_is_complete(rendered[: rendered.rfind("```")])
    assert not turn_is_complete("note\n```json\n")
    assert not turn_is_complete("note ``` ```"), "two fences with no object between them"
    assert turn_is_complete('<tool_call>\n{"name": "finish"}\n</tool_call>')
    assert turn_is_complete("anything<|im_end|>")


def test_system_prompt_lists_tools_and_fenced_format() -> None:
    assert system_prompt(TOOL_SPECS) == SYSTEM_PROMPT
    for spec in TOOL_SPECS:
        assert spec["function"]["name"] in SYSTEM_PROMPT
    assert "```json" in SYSTEM_PROMPT
    assert "<tool_call>" not in SYSTEM_PROMPT
    tools = render_tools(TOOL_SPECS)
    assert tools.startswith("# Tools\n")
    assert (
        "- replace_text(path: string, old: string, new: string): "
        "Replace exact text in one virtual file."
        in tools
    )
    assert "- finish(answer: string): Finish the task with a concise grounded answer." in tools
    custom = render_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "t",
                    "description": "d",
                    "parameters": {
                        "type": "object",
                        "properties": {"opt": {"type": "integer"}, "req": {"type": "string"}},
                        "required": ["req"],
                    },
                },
            }
        ]
    )
    assert "- t(req: string, opt: integer): d" in custom, "required arguments come first"


def test_step_is_supervised_by_default() -> None:
    assert Step("n", Action("finish", {"answer": "x"})).supervise


class _ChatTokenizer:
    """Adds a chat template, and keeps only the last few words so the fake model stays cheap."""

    window = 48

    def __init__(self, vocab_size: int = 40) -> None:
        self.vocab_size = vocab_size

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        assert not tokenize
        rendered = "".join(f"<|{message['role']}|>\n{message['content']}\n" for message in messages)
        return rendered + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        words = text.split()[-self.window :] or ["empty"]
        return [sum(map(ord, word)) % self.vocab_size for word in words]


def test_build_prompt_with_a_wide_window_does_not_re_hide_observations() -> None:
    """``build_probe_dataset`` re-renders rows that ``build_rows`` already windowed; windowing
    twice would re-stub a stub and silently change the text, so the call must be a no-op."""
    from local_llm_lab.pipeline.protocol import build_prompt, hidden_observation

    tokenizer = _ChatTokenizer()
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
        {
            "role": "tool",
            "name": "read_file",
            "content": hidden_observation("read_file", "a\nb\nc"),
        },
        {"role": "assistant", "content": "note"},
        {"role": "tool", "name": "read_file", "content": "x\ny"},
    ]
    wide = build_prompt(tokenizer, messages, keep_last=len(messages))
    assert wide == tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    assert "3 line(s)" in wide  # the original stub, not a stub of the stub
    assert "1 line(s)" not in build_prompt(tokenizer, messages, keep_last=len(messages))
