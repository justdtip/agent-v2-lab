from __future__ import annotations

import pytest

from local_llm_lab.agent_protocol import TOOL_SPECS, Action, ActionParseError
from local_llm_lab.pipeline.protocol import (
    SYSTEM_PROMPT,
    Step,
    assistant_message,
    build_prompt,
    hidden_observation,
    parse_turn,
    render_tools,
    render_turn,
    system_prompt,
    tool_message,
    turn_is_complete,
    window_messages,
)


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
    junk = '\n spep\n{"name": "finish", "arguments": {"answer": "no"}}\n```json\n{"name": "list_files", "arguments": {"directory": "/"}}\n```'
    turn = parse_turn(render_turn("n", first) + junk)
    assert turn.thought == "n" and turn.action == first
    cut = parse_turn(render_turn("n", first) + "\n``` `` <|im_end|>garbage")
    assert cut.action == first


def test_parse_turn_accepts_legacy_and_raw_forms() -> None:
    native = 'Reading the file.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "x"}}\n</tool_call>'
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
        "- replace_text(path: string, old: string, new: string): Replace exact text in one virtual file."
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
        return rendered + ("<|assistant|>\n" if add_generation_prompt else "")

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
