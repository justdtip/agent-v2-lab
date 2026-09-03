from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = """You are a careful autonomous tool-using agent.

Rules:
- Use tools to inspect state instead of guessing.
- Emit exactly one tool call at a time.
- Base each next action on the latest tool result.
- Never claim a write succeeded until a tool confirms it.
- Call finish only when the task is complete.
- Keep the finish answer concise and grounded in observed results."""


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files below a virtual directory.",
            "parameters": {
                "type": "object",
                "properties": {"directory": {"type": "string"}},
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one virtual text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Find virtual files whose path or content contains a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression safely.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_text",
            "description": "Replace exact text in one virtual file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the task with a concise grounded answer.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


_TOOL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ActionParseError(ValueError):
    """Raised when model output is not one supported tool action."""


@dataclass(frozen=True)
class Action:
    name: str
    arguments: dict[str, Any]


def parse_action(text: str) -> Action:
    """Parse Qwen tool-call markup or a raw JSON action."""
    block = _TOOL_BLOCK.search(text)
    if block:
        payload_text = block.group(1)
    else:
        raw = _JSON_OBJECT.search(text)
        if not raw:
            raise ActionParseError("response contains no JSON tool call")
        payload_text = raw.group(0)

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ActionParseError(f"invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ActionParseError("tool call must be a JSON object")

    name = payload.get("name", payload.get("tool"))
    arguments = payload.get("arguments", payload.get("args", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ActionParseError("arguments string is not valid JSON") from error
    if not isinstance(name, str) or not name:
        raise ActionParseError("tool call has no name")
    if not isinstance(arguments, dict):
        raise ActionParseError("tool arguments must be a JSON object")
    return Action(name=name, arguments=arguments)


def assistant_tool_message(action: Action) -> dict[str, Any]:
    """Represent an action in the native chat-template function-call format."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": action.name,
                    "arguments": json.dumps(
                        action.arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
        ],
    }


def tool_result_message(name: str, result: str) -> dict[str, str]:
    return {"role": "tool", "name": name, "content": result}
