from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from local_llm_lab.agent_protocol import ActionParseError
from local_llm_lab.pipeline.env import Fault, Simulator
from local_llm_lab.pipeline.protocol import (
    DEFAULT_KEEP_LAST,
    SYSTEM_PROMPT,
    TOOL_SPECS,
    assistant_message,
    build_prompt,
    parse_turn,
    tool_message,
    turn_is_complete,
    window_messages,
)
from local_llm_lab.pipeline.tasks import Task
from local_llm_lab.pipeline.transcript import Transcript

# Window sizes for the three repetition rules in :func:`detect_loop`.
LOOP_IDENTICAL_CALLS = 3
LOOP_SAME_TOOL_ERRORS = 6
LOOP_SAME_SHAPE_CALLS = 8


@dataclass
class Trajectory:
    task_id: str
    family: str
    variant: str
    label: str
    prompt: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    verdict: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None
    turns: int = 0
    valid_turns: int = 0
    elapsed_seconds: float = 0.0
    faults: list[int] = field(default_factory=list)
    generated_tokens: int = 0
    loop_detected: bool = False
    exhausted: bool = False

    @property
    def success(self) -> bool:
        return bool(self.verdict.get("success"))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def common_prefix_length(left: list[int], right: list[int]) -> int:
    """How many leading token ids the two sequences share."""
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


class TurnCache:
    """A KV cache reused across the turns of one task via longest-common-prefix matching.

    Consecutive prompts in a trajectory share almost everything: the system message and task
    never change, assistant notes are never rewritten, and an observation is replaced by its
    stub exactly once and then stays fixed. Only the turn falling out of the observation window
    differs, so the shared prefix typically covers most of the prompt (measured: 56-58% of all
    prompt tokens across a trajectory, against 23-26% for caching the fixed prefix alone).

    Each turn we trim the cache back to the longest prefix it shares with the new prompt and
    encode only the remainder. A stale cache would produce silently wrong logits rather than an
    error, so the offset is asserted against the prefix length after every trim, and any cache
    that cannot be trimmed is discarded and rebuilt.
    """

    def __init__(self, model: Any):
        self.model = model
        self.cache: Any = None
        self.tokens: list[int] = []
        self.reused_tokens = 0
        self.encoded_tokens = 0

    def _reset(self) -> None:
        from mlx_lm.models import cache as kv

        self.cache = kv.make_prompt_cache(self.model)
        self.tokens = []

    def prepare(self, token_ids: list[int]) -> list[int]:
        """Trim to the shared prefix; return the suffix of ``token_ids`` still to be encoded."""
        from mlx_lm.models import cache as kv

        if self.cache is None:
            self._reset()
        shared = common_prefix_length(self.tokens, token_ids)
        # The model needs at least one token to run a forward pass, so never consume the whole
        # prompt from cache: keep the final token for the suffix.
        shared = min(shared, len(token_ids) - 1)
        if shared < 0:
            shared = 0
        drop = len(self.tokens) - shared
        if drop > 0 and (
            not kv.can_trim_prompt_cache(self.cache)
            or kv.trim_prompt_cache(self.cache, drop) != drop
        ):
            self._reset()
            shared = 0
        self.tokens = list(token_ids[:shared])
        offset = getattr(self.cache[0], "offset", shared)
        if offset != shared:  # a mismatch here means silently wrong attention, so rebuild
            self._reset()
            shared = 0
            self.tokens = []
        self.reused_tokens += shared
        suffix = list(token_ids[shared:])
        self.encoded_tokens += len(suffix)
        return suffix

    def commit(self, token_ids: list[int], generated: list[int]) -> None:
        """Record everything the cache now holds: the full prompt plus what was generated."""
        self.tokens = list(token_ids) + list(generated)


def generate_turn_with_count(
    model: Any,
    tokenizer: Any,
    prompt: str,
    sampler: Any,
    max_tokens: int,
    turn_cache: TurnCache | None = None,
) -> tuple[str, int]:
    """Generate one assistant turn, stopping as soon as the tool call closes.

    Returns the decoded text and the number of tokens generated. The closing fence is ordinary
    text, so completion is checked by decoding the accumulated ids whenever the newest piece
    could close a fence, a legacy tag, or the turn. When ``turn_cache`` is given, the shared
    prefix with the previous turn is served from its KV cache and only the remainder is encoded.
    """
    import mlx.core as mx
    from mlx_lm import stream_generate

    stop_ids = set()
    with contextlib.suppress(Exception):  # tokenizer wrappers vary
        stop_ids.add(tokenizer.convert_tokens_to_ids("</tool_call>"))

    kwargs: dict[str, Any] = {}
    prompt_input: Any = prompt
    prompt_ids: list[int] = []
    if turn_cache is not None:
        prompt_ids = list(tokenizer.encode(prompt))
        prompt_input = mx.array(turn_cache.prepare(prompt_ids))
        kwargs["prompt_cache"] = turn_cache.cache

    ids: list[int] = []
    for response in stream_generate(
        model, tokenizer, prompt=prompt_input, max_tokens=max_tokens, sampler=sampler, **kwargs
    ):
        ids.append(response.token)
        if response.token in stop_ids:
            break
        piece = response.text or ""
        if any(mark in piece for mark in ("`", "<", "|")) and turn_is_complete(
            tokenizer.decode(ids)
        ):
            break
    if turn_cache is not None:
        turn_cache.commit(prompt_ids, ids)
    return tokenizer.decode(ids), len(ids)


def generate_turn(model: Any, tokenizer: Any, prompt: str, sampler: Any, max_tokens: int) -> str:
    """Text-only view of :func:`generate_turn_with_count`, for callers that ignore token counts."""
    return generate_turn_with_count(model, tokenizer, prompt, sampler, max_tokens)[0]


def detect_loop(steps: list[dict[str, Any]]) -> bool:
    """Pure repetition check over the executed steps of a trajectory.

    Flags a degenerate loop when any of these holds for the most recent executed actions:

    1. the last 3 calls are identical (same tool and the same arguments);
    2. the last 6 calls use the same tool and every observation is an error;
    3. the last 8 calls use the same tool with the same argument keys, none of them errored,
       and none is ``finish`` (the "calculate with an incrementing number" pattern).

    Parse-error steps carry no action and are ignored. The runner only records the flag; it
    never stops a run early, so the evaluator measures the loop instead of rescuing it.
    """
    executed = [step for step in steps if "action" in step]

    def last(size: int) -> list[dict[str, Any]]:
        return executed[-size:] if len(executed) >= size else []

    def same_tool(window: list[dict[str, Any]]) -> bool:
        return len({step["action"]["name"] for step in window}) == 1

    window = last(LOOP_IDENTICAL_CALLS)
    if window and all(step["action"] == window[0]["action"] for step in window):
        return True
    window = last(LOOP_SAME_TOOL_ERRORS)
    if (
        window
        and same_tool(window)
        and all(step["observation"].startswith("ERROR") for step in window)
    ):
        return True
    window = last(LOOP_SAME_SHAPE_CALLS)
    if window and same_tool(window) and window[0]["action"]["name"] != "finish":
        shapes = {tuple(sorted(step["action"]["arguments"])) for step in window}
        if len(shapes) == 1 and not any("ERROR" in step["observation"] for step in window):
            return True
    return False


def run_task(
    model: Any,
    tokenizer: Any,
    task: Task,
    *,
    sampler: Any,
    label: str = "policy",
    max_steps: int = 24,
    max_tokens: int = 200,
    keep_last: int = DEFAULT_KEEP_LAST,
    faults: tuple[Fault, ...] | None = None,
    transcript: Transcript | None = None,
    tools: list[dict[str, Any]] = TOOL_SPECS,
    use_cache: bool = True,
) -> Trajectory:
    """Drive one task end to end, mirroring every step to the transcript.

    ``tools`` is forwarded to :func:`build_prompt` for API compatibility only; the rendered
    prompt takes its tool list from the system message. ``use_cache`` reuses the KV cache across
    the task's turns; it is a pure speed optimisation and must not change any output, which
    ``research/cache_equivalence.py`` checks against a cache-free run.
    """
    started = time.monotonic()
    turn_cache = TurnCache(model) if use_cache else None
    simulator = Simulator.for_task(task, faults=faults)
    trajectory = Trajectory(
        task.task_id,
        task.family,
        task.variant,
        label,
        task.prompt,
        faults=[fault.call_index for fault in simulator.faults],
    )
    if transcript is not None:
        transcript.start(task, label)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    finished = False
    for index in range(max_steps):
        prompt = build_prompt(tokenizer, messages, tools=tools, keep_last=keep_last)
        raw, n_tokens = generate_turn_with_count(
            model, tokenizer, prompt, sampler, max_tokens, turn_cache
        )
        trajectory.turns += 1
        trajectory.generated_tokens += n_tokens
        try:
            turn = parse_turn(raw)
        except ActionParseError as error:
            trajectory.parse_error = str(error)
            trajectory.steps.append({"index": index, "raw": raw, "parse_error": str(error)})
            if transcript is not None:
                transcript.step(index, "", None, None, raw=raw, parse_error=str(error))
            break
        trajectory.valid_turns += 1
        observation = simulator.execute(turn.action)
        trajectory.steps.append(
            {
                "index": index,
                "thought": turn.thought,
                "action": {"name": turn.action.name, "arguments": turn.action.arguments},
                "observation": observation,
                "raw": raw,
            }
        )
        if transcript is not None:
            transcript.step(index, turn.thought, turn.action, observation, raw=raw)
        # Measure repetition but never rescue: the run continues to finish or max_steps.
        if detect_loop(trajectory.steps):
            trajectory.loop_detected = True
        if turn.action.name == "finish":
            finished = True
            break
        messages.append(assistant_message(turn.thought, turn.action))
        messages.append(tool_message(turn.action.name, observation))
    # The budget ran out only if neither finish nor a parse error ended the loop.
    trajectory.exhausted = not finished and trajectory.parse_error is None
    trajectory.verdict = simulator.verdict().as_dict()
    trajectory.elapsed_seconds = round(time.monotonic() - started, 2)
    if transcript is not None:
        transcript.finish(trajectory.verdict, trajectory.elapsed_seconds)
    return trajectory


def trajectory_rows(
    trajectory: Trajectory,
    task: Task,
    *,
    keep_last: int = DEFAULT_KEEP_LAST,
    source: str = "rollout",
) -> list[dict[str, Any]]:
    """Turn a verified on-policy trajectory into supervised rows (self-generated notes).

    Rows carry no ``tools`` key: the tool list is part of the system message.
    """
    faults = set(trajectory.faults)
    context: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    rows = []
    for step in trajectory.steps:
        if "action" not in step:
            break
        action = step["action"]
        observation = step["observation"]
        assistant = assistant_message(step["thought"], _action(action))
        errored = observation.startswith("ERROR") and step["index"] not in faults
        if not errored:
            rows.append(
                {
                    "messages": [*window_messages(context, keep_last), assistant],
                    "metadata": {
                        "task_id": task.task_id,
                        "family": task.family,
                        "variant": task.variant,
                        "step": step["index"],
                        "source": source,
                    },
                }
            )
        context.append(assistant)
        if action["name"] != "finish":
            context.append(tool_message(action["name"], observation))
    return rows


def _action(payload: dict[str, Any]) -> Any:
    from local_llm_lab.agent_protocol import Action

    return Action(payload["name"], dict(payload["arguments"]))
