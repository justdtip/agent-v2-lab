from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from local_llm_lab.agent_protocol import TOOL_SPECS, Action, ActionParseError, parse_action

if TYPE_CHECKING:
    from local_llm_lab.models import ModelSpec


def _compatibility_spec() -> ModelSpec:
    """Return the sole temporary default renderer specification; Task 6 removes this seam."""
    from local_llm_lab.models import load_model_spec

    return load_model_spec("qwen25-coder-3b")

__all__ = [
    "DEFAULT_KEEP_LAST",
    "END_OF_TURN",
    "RULES",
    "SYSTEM_PROMPT",
    "TOOL_FENCE_CLOSE",
    "TOOL_FENCE_OPEN",
    "TOOL_SPECS",
    "Action",
    "ActionParseError",
    "Step",
    "Turn",
    "action_json",
    "assistant_message",
    "build_prompt",
    "generation_suffix",
    "hidden_observation",
    "parse_turn",
    "render_completion",
    "render_tools",
    "render_turn",
    "system_prompt",
    "strip_thinking",
    "tool_message",
    "turn_is_complete",
    "window_messages",
]

TOOL_FENCE_OPEN = "```json"
TOOL_FENCE_CLOSE = "```"
END_OF_TURN = _compatibility_spec().chat.end_of_turn
DEFAULT_KEEP_LAST = 2

# Tool calls are rendered as ordinary text (a fenced JSON block) rather than through the chat
# template's native `tool_calls` field: the `<tool_call>` / `</tool_call>` special tokens are
# untrained in the target checkpoint, so the model cannot learn to emit them reliably.
RULES = """You are a careful autonomous agent operating a small file workspace through tools.

Each turn, write one short progress note, then exactly one tool call.
The progress note is your only memory: earlier tool results are hidden after two turns, so record every fact you still need (paths still to visit, values collected so far, what is done, what comes next).

Rules:
- Inspect state with tools instead of guessing paths, file names, or values.
- Base each action on the latest tool result and your notes.
- If a tool returns an error, read it and recover: list or search to find the right path, use an available tool, or retry a transient failure.
- Never claim a write succeeded until a tool result confirms it; re-read a file after changing it when asked to verify.
- Call finish only when the task is complete, with the answer in exactly the requested form.

Turn format: one short progress note, then a newline, then exactly one tool call as a fenced JSON block of the exact form
```json
{"name": "<tool>", "arguments": {...}}
```
Write nothing after the closing fence."""


def render_tools(tools: list[dict[str, Any]] = TOOL_SPECS) -> str:
    """Deterministic plain-text tool list for the system prompt.

    One line per tool, ``- name(arg: type, ...): description``, with required arguments
    listed first, followed by a short instruction on how to call them.
    """
    lines = ["# Tools"]
    for spec in tools:
        function = spec["function"]
        parameters = function.get("parameters", {})
        properties: dict[str, Any] = parameters.get("properties", {})
        required = [name for name in parameters.get("required", []) if name in properties]
        optional = [name for name in properties if name not in required]
        signature = ", ".join(
            f"{name}: {properties[name].get('type', 'any')}" for name in [*required, *optional]
        )
        lines.append(f"- {function['name']}({signature}): {function['description']}")
    lines.append("")
    lines.append(
        'Call exactly one of these tools per turn using the fenced JSON block above; "arguments" must be a JSON object using the argument names listed.'
    )
    return "\n".join(lines)


def system_prompt(tools: list[dict[str, Any]] = TOOL_SPECS) -> str:
    """Rules, turn format, and the rendered tool list."""
    return f"{RULES}\n\n{render_tools(tools)}"


SYSTEM_PROMPT = system_prompt()


@dataclass(frozen=True)
class Step:
    """One expert decision: a state-carrying note and exactly one action."""

    thought: str
    action: Action
    supervise: bool = True


@dataclass(frozen=True)
class Turn:
    """One parsed model output."""

    thought: str
    action: Action
    raw: str
    thinking: str | None = None


def action_json(action: Action) -> str:
    """Canonical JSON for an action; the same text is used for data, parsing, and preferences."""
    return json.dumps({"name": action.name, "arguments": action.arguments}, ensure_ascii=False)


def render_turn(thought: str, action: Action) -> str:
    """Plain-text assistant turn: the note, then the tool call as a fenced JSON block."""
    block = f"{TOOL_FENCE_OPEN}\n{action_json(action)}\n{TOOL_FENCE_CLOSE}"
    return f"{thought}\n{block}" if thought else block


def render_completion(thought: str, action: Action, *, spec: ModelSpec) -> str:
    """Render the supervised assistant completion using the model's declared turn terminator."""
    return render_turn(thought, action) + spec.chat.end_of_turn + "\n"


def assistant_message(thought: str, action: Action) -> dict[str, Any]:
    """Content-only assistant message; the tool call lives in the text, not in ``tool_calls``."""
    return {"role": "assistant", "content": render_turn(thought, action)}


def tool_message(name: str, result: str) -> dict[str, str]:
    return {"role": "tool", "name": name, "content": result}


def hidden_observation(name: str, result: str) -> str:
    lines = result.count("\n") + 1 if result else 0
    return f"[earlier {name} result hidden: {lines} line(s). Use your notes.]"


def window_messages(
    messages: list[dict[str, Any]], keep_last: int = DEFAULT_KEEP_LAST
) -> list[dict[str, Any]]:
    """Hide all but the most recent tool observations; assistant notes stay intact."""
    tool_positions = [index for index, message in enumerate(messages) if message["role"] == "tool"]
    hide = set(tool_positions[: max(0, len(tool_positions) - keep_last)])
    windowed = []
    for index, message in enumerate(messages):
        if index in hide:
            replaced = dict(message)
            replaced["content"] = hidden_observation(
                message.get("name", "tool"), message["content"]
            )
            windowed.append(replaced)
        else:
            windowed.append(copy.deepcopy(message))
    return windowed


_FENCE_OPEN = re.compile(r"```[A-Za-z]*[ \t]*\r?\n?")
_TOOL_OPEN = re.compile(r"<tool_call>", re.IGNORECASE)
_TOOL_CLOSE = "</tool_call>"
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_DECODER = json.JSONDecoder()


def _locate_call(text: str) -> tuple[int, int]:
    """Return (start of the call block, index of the JSON object's first brace)."""
    fence = _FENCE_OPEN.search(text)
    if fence:
        brace = text.find("{", fence.end())
        if brace < 0:
            raise ActionParseError("fenced block contains no JSON object")
        return fence.start(), brace
    legacy = _TOOL_OPEN.search(text)
    if legacy:
        brace = text.find("{", legacy.end())
        if brace < 0:
            raise ActionParseError("<tool_call> block contains no JSON object")
        return legacy.start(), brace
    brace = text.find("{")
    if brace < 0:
        raise ActionParseError("response contains no tool call")
    return brace, brace


def parse_turn(text: str) -> Turn:
    """Split model output into the note and exactly one action.

    The call is taken from the first fenced block, else a legacy ``<tool_call>`` block, else
    the first ``{``. Only the first JSON object is decoded, so trailing junk is ignored.
    """
    cleaned = text.replace(END_OF_TURN, "").strip()
    block_start, brace = _locate_call(cleaned)
    try:
        _, end = _DECODER.raw_decode(cleaned, brace)
    except json.JSONDecodeError as error:
        raise ActionParseError(f"invalid JSON in tool call: {error.msg}") from error
    action = parse_action(cleaned[brace:end])
    thought = cleaned[:block_start].strip()
    return Turn(thought=thought, action=action, raw=text)


def strip_thinking(text: str) -> tuple[str | None, str]:
    """Extract the first closed reasoning block and leave the model action text intact."""
    match = _THINK_BLOCK.search(text)
    if match is None:
        return None, text
    remainder = (text[: match.start()] + text[match.end() :]).lstrip("\r\n")
    return match.group(1), remainder


def turn_is_complete(text: str) -> bool:
    """True once the text contains a closed tool call (fenced JSON, legacy tag, or end of turn)."""
    thinking_start = text.find("<think>")
    if thinking_start >= 0:
        thinking_end = text.find("</think>", thinking_start)
        if thinking_end < 0:
            return False
        text = text[thinking_end + len("</think>") :]
    if END_OF_TURN in text or _TOOL_CLOSE in text:
        return True
    if text.count(TOOL_FENCE_CLOSE) < 2:
        return False
    first = text.find(TOOL_FENCE_CLOSE)
    second = text.find(TOOL_FENCE_CLOSE, first + len(TOOL_FENCE_CLOSE))
    body = text[first + len(TOOL_FENCE_CLOSE) : second]
    return "{" in body and "}" in body


def build_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    keep_last: int = DEFAULT_KEEP_LAST,
    spec: ModelSpec | None = None,
    generation: bool = True,
) -> str:
    """Render the windowed conversation for generation.

    An omitted ``spec`` is the one remaining Task 6 compatibility seam: it resolves to the
    registered legacy 3B spec and skips the generation-suffix assertion. It stays until the
    caller that omits ``spec`` (``tests/test_probes.py``, an uncommitted slice at the Chief's
    gate) can be updated. The dead ``tools=`` seam, never passed to the template and never
    supplied by any caller, is gone.

    **Precondition on ``messages``, which the template enforces and this function does not.**
    Qwen3.5's chat template scans the list in reverse for a ``user`` turn whose content, once
    trimmed, is not a ``<tool_response>`` wrapper, and raises ``jinja2`` ``TemplateError: No
    user query found in messages.`` when it finds none; it also refuses a system message
    anywhere but index 0, an unknown role, and content that is neither a string, ``None``, nor a
    list of content items. Qwen2.5's template has none of these guards, so a
    caller that was correct on the 3B is not thereby correct on the 4B -- EXP-002 died three
    seconds into a run on ``[system]`` alone, a prefix that renders fine under 2.5. Every
    caller in this repo builds ``[system, user, ...]`` and only ever appends, so the precondition
    holds by construction; the two places that render a *slice* of a conversation rather than
    the whole of it check it explicitly (``pipeline.data.render_rows``, which drops the
    assistant target, and ``probes.state_swap._rendered_boundaries``, which walks every prefix).
    A new caller that slices, filters or reorders messages owes the same check.
    """
    compatibility_mode = spec is None
    resolved_spec = _compatibility_spec() if compatibility_mode else spec
    prompt = tokenizer.apply_chat_template(
        window_messages(messages, keep_last),
        add_generation_prompt=generation,
        tokenize=False,
        **resolved_spec.chat.template_kwargs,
    )
    if generation and not compatibility_mode and not prompt.endswith(generation_suffix(resolved_spec)):
        raise ValueError("chat template generation suffix does not match the model specification")
    return prompt


def generation_suffix(spec: ModelSpec) -> str:
    """Return the template suffix that anchors the first generated assistant token."""
    assistant = "<|im_start|>assistant\n"
    if spec.chat.thinking == "off":
        return assistant + "<think>\n\n</think>\n\n"
    return assistant
