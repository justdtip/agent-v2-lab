import pytest

from local_llm_lab.agent_protocol import Action, ActionParseError, parse_action


def test_parse_qwen_tool_call() -> None:
    text = '<tool_call>\n{"name":"read_file","arguments":{"path":"a.txt"}}\n</tool_call>'
    assert parse_action(text) == Action("read_file", {"path": "a.txt"})


def test_parse_raw_compatibility_action() -> None:
    assert parse_action('{"tool":"finish","args":{"answer":"done"}}') == Action(
        "finish", {"answer": "done"}
    )


def test_parse_string_arguments() -> None:
    text = '{"name":"calculate","arguments":"{\\"expression\\":\\"2+2\\"}"}'
    assert parse_action(text) == Action("calculate", {"expression": "2+2"})


def test_reject_non_action() -> None:
    with pytest.raises(ActionParseError):
        parse_action("I would read the file.")
