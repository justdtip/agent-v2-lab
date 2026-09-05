from __future__ import annotations

import contextlib
import copy
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from local_llm_lab.agent_protocol import ActionParseError
from local_llm_lab.pipeline.env import Fault, Simulator
from local_llm_lab.pipeline.protocol import (
    DEFAULT_KEEP_LAST,
    SYSTEM_PROMPT,
    assistant_message,
    build_prompt,
    parse_turn,
    strip_thinking,
    tool_message,
    turn_is_complete,
    window_messages,
)
from local_llm_lab.pipeline.tasks import Task
from local_llm_lab.pipeline.transcript import Transcript

if TYPE_CHECKING:
    from local_llm_lab.arch import ArchitectureView
    from local_llm_lab.models import ModelSpec, ResolvedSpec

# Window sizes for the three repetition rules in :func:`detect_loop`.
LOOP_IDENTICAL_CALLS = 3
LOOP_SAME_TOOL_ERRORS = 4
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
    difficulty: int = -1
    integrity: dict[str, Any] = field(default_factory=dict)
    think_tokens: int = 0
    model: dict[str, Any] = field(default_factory=dict)

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


class TurnCacheBase(Protocol):
    cache: Any
    reused_tokens: int
    encoded_tokens: int

    def prepare(self, token_ids: list[int]) -> list[int]: ...

    def commit(self, token_ids: list[int], generated: list[int]) -> None: ...


class TrimCache:
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


class SnapshotCache:
    """Reuse only a verified immutable-prefix snapshot across turns."""

    def __init__(self, model: Any, view: ArchitectureView, prefix_tokens: int):
        self.model = model
        self.view = view
        self.prefix_tokens = prefix_tokens
        self.cache: Any = None
        self.reused_tokens = 0
        self.encoded_tokens = 0
        self._prefix: list[int] | None = None
        self._states: list[Any] | None = None

    def prepare(self, token_ids: list[int]) -> list[int]:
        if not 0 < self.prefix_tokens < len(token_ids):
            raise ValueError("prefix_tokens must lie strictly within the prompt")
        prefix = list(token_ids[: self.prefix_tokens])
        suffix = list(token_ids[self.prefix_tokens :])
        if self.cache is None:
            import mlx.core as mx

            self.cache = self.view.make_cache()
            self.model(mx.array(prefix)[None, :], cache=self.cache)
            mx.eval(*(entry.state for entry in self.cache))
            self._prefix = prefix
            self._states = [_copy_cache_state(entry.state) for entry in self.cache]
            self.encoded_tokens += len(token_ids)
            return suffix
        if prefix != self._prefix:
            raise ValueError("immutable prefix changed after snapshot creation")
        assert self._states is not None
        for entry, state in zip(self.cache, self._states, strict=True):
            entry.state = _copy_cache_state(state)
        self.reused_tokens += self.prefix_tokens
        self.encoded_tokens += len(suffix)
        return suffix

    def commit(self, token_ids: list[int], generated: list[int]) -> None:
        """The live cache may advance; the saved prefix snapshot remains unchanged."""


def _copy_cache_state(value: Any) -> Any:
    if isinstance(value, list):
        return [_copy_cache_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_cache_state(item) for item in value)
    if isinstance(value, dict):
        return {key: _copy_cache_state(item) for key, item in value.items()}
    return copy.deepcopy(value)


def make_turn_cache(
    model: Any,
    view: ArchitectureView,
    resolved: ResolvedSpec,
    *,
    prefix_tokens: int,
) -> TurnCacheBase | None:
    if resolved.cache_strategy == "trim":
        return TrimCache(model)
    if resolved.cache_strategy == "snapshot":
        return SnapshotCache(model, view, prefix_tokens)
    if resolved.cache_strategy == "none":
        hybrid = any(layer_type == "linear_attention" for layer_type in resolved.layer_types)
        if hybrid and resolved.cache_strategy_reason == "auto:equivalence_unverified":
            print(
                f"WARNING: {resolved.spec.name} cache auto-resolution disabled reuse: "
                "equivalence is unverified",
                file=sys.stderr,
            )
        return None
    raise ValueError(f"unsupported resolved cache strategy: {resolved.cache_strategy}")


@dataclass
class _ThinkingTracker:
    enabled: bool
    max_tokens: int
    started: bool = False
    finished: bool = False
    start_token: int = 0
    tokens: int = 0
    forced_close_at: int | None = None

    def update(self, ids: list[int], decoded: str, tokenizer: Any) -> None:
        if not self.enabled or self.finished:
            return
        thinking_start = decoded.find("<think>")
        if thinking_start < 0:
            return
        if not self.started:
            self.started = True
            self.start_token = next(
                index
                for index in range(len(ids))
                if len(tokenizer.decode(ids[: index + 1])) > thinking_start
            )
        self.tokens = len(ids) - self.start_token
        if decoded.find("</think>", thinking_start) >= 0:
            self.finished = True
        elif self.tokens >= self.max_tokens:
            self.forced_close_at = len(ids)
            self.finished = True

    def decoded_text(self, ids: list[int], tokenizer: Any) -> str:
        if self.forced_close_at is None:
            return tokenizer.decode(ids)
        return (
            tokenizer.decode(ids[: self.forced_close_at])
            + "\n</think>\n\n"
            + tokenizer.decode(ids[self.forced_close_at :])
        )


def generate_turn_with_count(
    model: Any,
    tokenizer: Any,
    prompt: str,
    sampler: Any,
    max_tokens: int,
    turn_cache: TurnCacheBase | None = None,
    *,
    spec: ModelSpec,
) -> tuple[str, int, int]:
    """Generate one assistant turn, stopping as soon as the tool call closes.

    Returns decoded text, total generated tokens, and tokens spent in the first think block.
    When a thinking mode exhausts its budget, a closing tag is inserted into the raw output and
    generation continues until the visible note and tool call are complete.
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
    thinking = _ThinkingTracker(
        enabled=spec.chat.thinking in {"inference", "trained"},
        max_tokens=spec.chat.max_think_tokens,
    )

    for response in stream_generate(
        model, tokenizer, prompt=prompt_input, max_tokens=max_tokens, sampler=sampler, **kwargs
    ):
        ids.append(response.token)
        decoded = tokenizer.decode(ids)
        thinking.update(ids, decoded, tokenizer)
        if response.token in stop_ids:
            break
        piece = response.text or ""
        if any(mark in piece for mark in ("`", "<", "|")) and turn_is_complete(
            thinking.decoded_text(ids, tokenizer)
        ):
            break
    if turn_cache is not None:
        turn_cache.commit(prompt_ids, ids)
    think_tokens = thinking.tokens if thinking.started else 0
    return thinking.decoded_text(ids, tokenizer), len(ids), think_tokens


def generate_turn(model: Any, tokenizer: Any, prompt: str, sampler: Any, max_tokens: int) -> str:
    """Text-only view of :func:`generate_turn_with_count`, for callers that ignore token counts."""
    spec, _, _ = _compatibility_runner_inputs(None, None, None)
    return generate_turn_with_count(
        model, tokenizer, prompt, sampler, max_tokens, spec=spec
    )[0]


def detect_loop(steps: list[dict[str, Any]]) -> bool:
    """Pure repetition check over the executed steps of a trajectory.

    Flags a degenerate loop when any of these holds for the most recent executed actions:

    1. the last 3 calls are identical (same tool and the same arguments);
    2. the last 4 calls use the same tool and every observation is an error;
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
    spec: ModelSpec | None = None,
    view: ArchitectureView | None = None,
    resolved: ResolvedSpec | None = None,
    label: str = "policy",
    max_steps: int = 24,
    max_tokens: int = 200,
    keep_last: int = DEFAULT_KEEP_LAST,
    faults: tuple[Fault, ...] | None = None,
    transcript: Transcript | None = None,
    use_cache: bool = True,
) -> Trajectory:
    """Drive one task end to end, mirroring every step to the transcript.

    Omitted model metadata is the temporary legacy 3B compatibility path. Callers that provide
    a view and resolved spec always use the resolved safe cache strategy.
    """
    started = time.monotonic()
    spec, view, resolved = _compatibility_runner_inputs(spec, view, resolved)
    simulator = Simulator.for_task(task, faults=faults)
    trajectory = Trajectory(
        task.task_id,
        task.family,
        task.variant,
        label,
        task.prompt,
        faults=[fault.call_index for fault in simulator.faults],
        model={} if resolved is None else resolved.as_dict(),
    )
    if transcript is not None:
        transcript.start(task, label)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    # ``[:2]`` is the whole list, built two lines above, and it is written as a slice only to
    # say that the cache prefix is the system-and-user opening. It is safe because that opening
    # contains a user turn: a chat template may refuse a conversation that carries none --
    # Qwen3.5's raises ``No user query found in messages.`` -- so a prefix slice that dropped the
    # user message would die inside the template. Anything appended below keeps it.
    prefix_prompt = build_prompt(
        tokenizer,
        messages[:2],
        spec=spec,
        keep_last=keep_last,
        generation=False,
    )
    prefix_tokens = len(tokenizer.encode(prefix_prompt))
    turn_cache = (
        make_turn_cache(model, view, resolved, prefix_tokens=prefix_tokens)
        if use_cache and view is not None and resolved is not None
        else TrimCache(model)
        if use_cache
        else None
    )
    finished = False
    for index in range(max_steps):
        prompt = build_prompt(
            tokenizer,
            messages,
            spec=spec,
            keep_last=keep_last,
            generation=True,
        )
        raw, n_tokens, think_tokens = generate_turn_with_count(
            model, tokenizer, prompt, sampler, max_tokens, turn_cache, spec=spec
        )
        trajectory.turns += 1
        trajectory.generated_tokens += n_tokens
        trajectory.think_tokens += think_tokens
        thinking, action_text = strip_thinking(raw)
        try:
            turn = parse_turn(action_text)
        except ActionParseError as error:
            trajectory.parse_error = str(error)
            trajectory.steps.append(
                {
                    "index": index,
                    "thinking": thinking,
                    "think_tokens": think_tokens,
                    "raw": raw,
                    "parse_error": str(error),
                }
            )
            if transcript is not None:
                transcript.step(
                    index,
                    "",
                    None,
                    None,
                    thinking=thinking,
                    think_tokens=think_tokens,
                    raw=raw,
                    parse_error=str(error),
                )
            break
        trajectory.valid_turns += 1
        observation = simulator.execute(turn.action)
        trajectory.steps.append(
            {
                "index": index,
                "thinking": thinking,
                "think_tokens": think_tokens,
                "thought": turn.thought,
                "action": {"name": turn.action.name, "arguments": turn.action.arguments},
                "observation": observation,
                "raw": raw,
            }
        )
        if transcript is not None:
            transcript.step(
                index,
                turn.thought,
                turn.action,
                observation,
                thinking=thinking,
                think_tokens=think_tokens,
                raw=raw,
            )
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


def _compatibility_runner_inputs(
    spec: ModelSpec | None,
    view: ArchitectureView | None,
    resolved: ResolvedSpec | None,
) -> tuple[ModelSpec, ArchitectureView | None, ResolvedSpec | None]:
    if spec is None:
        from local_llm_lab.models import load_model_spec

        spec = load_model_spec("qwen25-coder-3b")
    return spec, view, resolved


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
