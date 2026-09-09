from __future__ import annotations

import contextlib
import copy
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from local_llm_lab.agent_protocol import ActionParseError
from local_llm_lab.forward import ForwardLedger, ForwardPass, encode_prompt, prefill_passes
from local_llm_lab.models import NATIVE_PREFILL_STEP_SIZE
from local_llm_lab.pipeline.env import Fault, Simulator
from local_llm_lab.pipeline.protocol import (
    DEFAULT_KEEP_LAST,
    SYSTEM_PROMPT,
    assistant_message,
    build_prompt,
    parse_turn,
    strip_thinking,
    system_prompt,
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
    truncated: bool = False
    repetition: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    horizon: int = (
        -1
    )  # the task's expert step count, set by the evaluator; -1 when unknown (older artifacts)

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
            self._states = snapshot_cache(self.cache)
            self.encoded_tokens += len(token_ids)
            return suffix
        if prefix != self._prefix:
            raise ValueError("immutable prefix changed after snapshot creation")
        assert self._states is not None
        restore_cache(self.cache, self._states)
        self.reused_tokens += self.prefix_tokens
        self.encoded_tokens += len(suffix)
        return suffix

    def commit(self, token_ids: list[int], generated: list[int]) -> None:
        """The live cache may advance; the saved prefix snapshot remains unchanged."""


def snapshot_cache(entries: list[Any]) -> list[tuple[Any, Any]]:
    """Save everything that defines a cache entry, which is contents *and* position.

    Saving ``state`` alone is a defect, and a silent one. ``mlx_lm``'s two cache kinds split
    the job differently: ``KVCache.state``'s setter recovers the offset from the restored
    array's own length, while ``RotatingKVCache.state``'s setter assigns keys and values and
    nothing else -- its ``offset`` and ``_idx`` live in ``meta_state`` (``models/cache.py``,
    the ``state`` and ``meta_state`` properties of each class).

    So a snapshot of ``state`` alone round-trips correctly on a full-attention model and
    restores a rotating layer to the right contents at the wrong position. Gemma 3 4B runs a
    rotating cache on 29 of its 34 blocks, which is where this stops being theoretical. The
    failure mode is wrong attention rather than an exception.
    """
    saved = []
    for entry in entries:
        meta = getattr(entry, "meta_state", None)
        saved.append((_copy_cache_state(entry.state), meta))
    return saved


def restore_cache(entries: list[Any], saved: list[tuple[Any, Any]]) -> None:
    """Restore contents then position, in that order and for that reason.

    ``KVCache.state``'s setter derives the offset from the restored length, so ``state`` must
    land first; ``meta_state`` then puts back the true offset and write index for the entries
    that keep them there. An empty ``meta_state`` is the base class's "no metadata" value and
    assigning it back would raise, so it is skipped.
    """
    for entry, (state, meta) in zip(entries, saved, strict=True):
        entry.state = _copy_cache_state(state)
        if meta:
            entry.meta_state = meta


def _copy_cache_state(value: Any) -> Any:
    if isinstance(value, list):
        return [_copy_cache_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_cache_state(item) for item in value)
    if isinstance(value, dict):
        return {key: _copy_cache_state(item) for key, item in value.items()}
    return copy.deepcopy(value)


def clone_prompt_cache(cache: list[Any]) -> list[Any]:
    """Clone complete native objects, including offsets and offset-less mask metadata."""
    import mlx.core as mx

    def evaluate(entries):
        mx.eval([entry.state for entry in entries])
        for entry in entries:
            for name in ("lengths", "left_padding"):
                value = getattr(entry, name, None)
                if value is not None:
                    mx.eval(value)

    evaluate(cache)
    cloned = copy.deepcopy(cache)
    evaluate(cloned)
    return cloned


@dataclass(frozen=True)
class _HistorySnapshot:
    passes: tuple[ForwardPass, ...]
    cache: list[Any]
    nbytes: int

    @property
    def offset(self) -> int:
        last = self.passes[-1]
        return last.offset + len(last.input_ids)


class _HistoryModel:
    """Delegate native arithmetic; observe actual inputs, never generated text."""

    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner.model, name)

    def __call__(self, ids, *, cache, **kwargs):
        owner = self.owner
        if cache is not owner.cache or owner.ledger is None:
            raise ValueError("history forward must use its prepared cache")
        if ids.ndim != 2 or ids.shape[0] != 1 or kwargs.get("input_embeddings") is not None:
            raise ValueError("history cache supports single-sequence token inputs only")
        tokens = ids[0].tolist()
        offset = owner.ledger.offset
        owner.ledger.validate(offset, tokens)
        owner._check_offsets(offset)
        try:
            logits = owner.model(ids, cache=cache, **kwargs)
            owner._check_offsets(offset + len(tokens))
            owner.ledger.record(offset, tokens)
        except BaseException:
            # A partly advanced hybrid state cannot be rolled back by adjusting an offset.
            owner.cache = None
            owner.ledger = None
            raise
        owner.encoded_tokens += len(tokens)
        if owner.ledger.offset < len(owner.ledger.prompt_ids):
            owner._save()
        return logits


class HistoryCache:
    """Bounded, episode-local hybrid snapshots at existing native forward boundaries.

    Exact token AND partition prefixes are required for reuse. A partial prefill or a
    token-by-token generated suffix cannot silently substitute for a differently shaped
    prefill. Rewritten observations invalidate all dependent state. This strategy is
    explicit-only until real-checkpoint acceptance; no registry is updated by this class.

    Model weights must remain immutable for this object's lifetime (one run_task).
    """

    def __init__(
        self,
        model: Any,
        view: ArchitectureView,
        *,
        prefill_step_size: int = NATIVE_PREFILL_STEP_SIZE,
        max_checkpoints: int = 4,
        max_bytes: int = 512 * 1024 * 1024,
    ):
        if any(
            type(value) is not int or value < 1
            for value in (prefill_step_size, max_checkpoints, max_bytes)
        ):
            raise ValueError("cache cadence and budgets must be positive integers")
        if model is not view.model:
            raise ValueError("history cache view must belong to its model")
        self.model, self.view = model, view
        self.prefill_step_size = prefill_step_size
        self.max_checkpoints, self.max_bytes = max_checkpoints, max_bytes
        self.cache: Any = None
        self.ledger: ForwardLedger | None = None
        self.reused_tokens = 0
        self.encoded_tokens = 0
        self._snapshots: list[_HistorySnapshot] = []
        self.generation_model = _HistoryModel(self)

    @property
    def snapshot_bytes(self) -> int:
        return sum(snapshot.nbytes for snapshot in self._snapshots)

    def _check_offsets(self, expected: int) -> None:
        for entry in self.cache:
            if hasattr(entry, "offset") and entry.offset != expected:
                raise ValueError(
                    f"native cache offset {entry.offset} disagrees with ledger {expected}"
                )

    def _save(self) -> None:
        assert self.ledger is not None
        nbytes = sum(entry.nbytes for entry in self.cache)
        nbytes += sum(
            getattr(getattr(entry, name, None), "nbytes", 0)
            for entry in self.cache
            for name in ("lengths", "left_padding")
        )
        if nbytes > self.max_bytes:
            return
        self._snapshots = [s for s in self._snapshots if s.offset != self.ledger.offset]
        # Evict before allocation so adding a new snapshot cannot double the stored budget.
        while self._snapshots and (
            len(self._snapshots) >= self.max_checkpoints
            or self.snapshot_bytes + nbytes > self.max_bytes
        ):
            # Keep the earliest surviving boundary as a rollback anchor when possible;
            # the remaining slots rotate through recent history.
            self._snapshots.pop(1 if len(self._snapshots) > 1 else 0)
        self._snapshots.append(
            _HistorySnapshot(tuple(self.ledger.passes), clone_prompt_cache(self.cache), nbytes)
        )

    def prepare(self, token_ids: list[int]) -> list[int]:
        # Validate before touching existing state, including the context bound.
        ledger = ForwardLedger(token_ids)
        planned = prefill_passes(token_ids, self.prefill_step_size)
        self._snapshots = [
            s
            for s in self._snapshots
            if s.offset < len(token_ids)
            and all(
                tuple(token_ids[p.offset : p.offset + len(p.input_ids)]) == p.input_ids
                for p in s.passes
            )
        ]
        eligible = [s for s in self._snapshots if s.passes == tuple(planned[: len(s.passes)])]
        if eligible:
            chosen = max(eligible, key=lambda s: s.offset)
            self.cache = clone_prompt_cache(chosen.cache)
            ledger = ForwardLedger(token_ids, restored_passes=chosen.passes)
        else:
            self.cache = self.view.make_cache()
        self.ledger = ledger
        self._check_offsets(ledger.offset)
        self.reused_tokens += ledger.offset
        return list(token_ids[ledger.offset :])

    def emitted(self, token_id: int) -> None:
        if self.ledger is None:
            raise ValueError("emission requires a prepared history cache")
        self.ledger.emitted(token_id)

    def commit(self, token_ids: list[int], generated: list[int]) -> None:
        # Display text and synthetic thinking delimiters are never a source of cache IDs.
        del generated
        if self.ledger is None or self.ledger.prompt_ids != list(token_ids):
            raise ValueError("history commit disagrees with the prepared prompt")
        if self.ledger.offset < len(token_ids):
            raise ValueError("history commit before the prompt was fully forwarded")
        self._check_offsets(self.ledger.offset)
        self._save()


def is_torch_model(model: Any) -> bool:
    """Whether ``model`` is a torch module, discovered structurally rather than by name.

    The same principle the architecture view already follows: never consult a model-type
    string. A missing torch is not an error here, it is simply an MLX process.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised only where torch is absent
        return False
    return isinstance(model, torch.nn.Module)


def make_turn_cache(
    model: Any,
    view: ArchitectureView,
    resolved: ResolvedSpec,
    *,
    prefix_tokens: int,
) -> TurnCacheBase | None:
    if is_torch_model(model) and resolved.cache_strategy != "none":
        # The three reuse strategies are deferred on torch, and the golden records never
        # exercised them. Refusing loudly is the point: a silent fallback to no reuse would
        # be a speed regression that no test fails on, and an MLX cache handed to a torch
        # model would attend to the wrong keys rather than raise.
        raise NotImplementedError(
            f"cache strategy {resolved.cache_strategy!r} is not ported to torch; "
            "stage two ran under 'none' and only 'none' is implemented (WS-B)"
        )
    if resolved.cache_strategy == "trim":
        return TrimCache(model)
    if resolved.cache_strategy == "snapshot":
        return SnapshotCache(model, view, prefix_tokens)
    if resolved.cache_strategy == "history":
        return HistoryCache(model, view)
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


#: Cheap gate before the expensive completeness test: a closing token always adds one of
#: these characters, so a piece without them cannot have closed the call.
_COMPLETION_MARKS = ("`", "<", "|")

#: Why a turn stopped. ``token_cap`` is the one that means the turn was cut off rather than
#: finished, and a turn that ends there without a parseable action is a truncation, not a
#: wrong answer.
STOP_TOKEN = "stop_token"
STOP_TURN_COMPLETE = "turn_complete"
STOP_TOKEN_CAP = "token_cap"


def _consume_stream(
    stream: Any,
    *,
    ids: list[int],
    thinking: _ThinkingTracker,
    tokenizer: Any,
    stop_ids: set[int],
    capture: Any | None,
    turn_cache: TurnCacheBase | None,
) -> str:
    """Apply the turn's stop rule to a stream of ``(token_id, text_piece)`` pairs.

    Both backends feed this one function, so the stop semantics cannot drift between them.
    That matters more than it looks: the golden records were produced under exactly this rule,
    and two copies of it that agree today are two copies that can disagree later.
    """
    for token, piece in stream:
        ids.append(token)
        if capture is not None:
            capture.emitted(token)
        if isinstance(turn_cache, HistoryCache):
            turn_cache.emitted(token)
        decoded = tokenizer.decode(ids)
        thinking.update(ids, decoded, tokenizer)
        if token in stop_ids:
            return STOP_TOKEN
        if any(mark in piece for mark in _COMPLETION_MARKS) and turn_is_complete(
            thinking.decoded_text(ids, tokenizer)
        ):
            return STOP_TURN_COMPLETE
    return STOP_TOKEN_CAP


def config_eos_ids(model: Any) -> frozenset[int]:
    """Terminators from the model config's own id set, never from the tokenizer.

    Gemma 3 declares ``eos_token_id: [1, 106]`` -- ``<eos>`` and ``<end_of_turn>`` -- and MLX
    stops on that set. The tokenizer's ``eos_token_id`` is a single id and is not that set, so
    taking it leaves ``<end_of_turn>`` unrecognised and every turn runs to the token cap.

    This is measured rather than argued: in the golden records, token 106 is emitted on six of
    the ninety-four turns and is the last token of each of them.
    """
    config = getattr(model, "config", None)
    declared = getattr(config, "eos_token_id", None)
    if declared is None:
        raise ValueError(
            "the model config declares no eos_token_id; refusing to fall back to the "
            "tokenizer's single id, which would leave <end_of_turn> unrecognised and run "
            "every turn to the token cap"
        )
    if isinstance(declared, int):
        return frozenset({declared})
    return frozenset(int(value) for value in declared)


_DETERMINISM_PINNED = False


def pin_torch_determinism() -> dict[str, Any] | None:
    """Pin determinism once per process, through ``device.pin``, which owns the policy.

    This function holds no settings of its own. It exists so that a loop entered without the
    acceptance kit having run cannot take a number under unpinned settings, and it is a
    once-per-process guard because ``pin`` reseeds and a per-turn reseed would be noise in the
    record for no benefit under greedy decoding.

    ``CUBLAS_WORKSPACE_CONFIG`` is read by cuBLAS at first use, so ``pin`` refuses rather than
    reports success if CUDA is already initialised without it. That refusal is worth more than a
    late best effort, and it is the kit's job to call ``pin`` first.
    """
    global _DETERMINISM_PINNED
    if _DETERMINISM_PINNED:
        return None
    from local_llm_lab import device

    reading = device.pin()
    _DETERMINISM_PINNED = True
    return reading


def _forward_logits(model: Any, view: ArchitectureView, token_ids, cache: Any) -> Any:
    """One forward through the capture wrapper, returning the model's own logits.

    The seam WS-A fixed: the wrapper takes the cache as a keyword and returns the HF output,
    whose ``.logits`` are the native ones. Hidden states do not come back here at all; they
    reach the readout gate through the capture's sink, which is what keeps the readout a thing
    compared against the model's head rather than a substitute for it.

    ``view._ids`` is the view's own token-to-tensor conversion and carries the device and dtype
    with it. It is private, and it is nonetheless the right call: WS-A's own reference loop uses
    it, and building the tensor here instead would leave it on the CPU while the model sat on a
    GPU. The fallback exists for a stub view that has no such helper.
    """
    import torch

    ids = view._ids(list(token_ids)) if hasattr(view, "_ids") else torch.tensor([list(token_ids)])
    result = model(ids, cache=cache)
    return result if torch.is_tensor(result) else result.logits


def torch_greedy_stream(
    model: Any,
    view: ArchitectureView,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    max_tokens: int,
    *,
    eos_ids: Iterable[int],
) -> Iterator[tuple[int, str]]:
    """Greedy decode on torch, yielding ``(token_id, text_piece)`` like the MLX stream.

    **The model's own head produces the token; the readout is never the producer.** The
    capture wrapper hands back the model's logits and the readout gate then recomputes
    ``native_readout(h)`` at layer 34 and *compares* it to them, recording the difference on
    every forward. Generating from the readout instead would leave the gate comparing the
    model's head against the thing that generated the token, which is the wrong way round and
    would quietly change what the golden records mean.

    **The prefill partition is the native one**: chunks of ``NATIVE_PREFILL_STEP_SIZE`` with
    the final prompt token always separate. ``ForwardLedger.validate`` asserts it on every
    forward, and a single-chunk prefill would produce different forward rows under an
    identical trajectory.

    **The lookahead is MLX's.** The forward for a token runs before that token is yielded, so
    a turn of *n* emissions leaves *n* single-token forwards behind it and the last one's
    argmax is never used. That is what makes the recorded forward at offset *p* the prediction
    of position *p + 1*.

    **EOS terminates and is kept.** ``eos_ids`` comes from the model config's own set, and the
    token is yielded before the stream ends because the recorded emissions contain it.

    ``cache_strategy: none`` refers to reuse *across turns*. Within a turn the cache is still
    needed or decoding is quadratic, so the loop makes its own and drops it at the end.
    """
    import torch

    pin_torch_determinism()
    cache = view.make_cache()
    generated: list[int] = []
    previous_text = ""
    terminators = frozenset(eos_ids)
    # `no_grad` rather than `inference_mode` to match WS-A's own reference loop. Residuals
    # reach the gate through the sink and an inference tensor is awkward to use later; there is
    # no speed argument here worth diverging from the reference for.
    with torch.no_grad():
        logits = None
        for prefill in prefill_passes(list(prompt_ids)):
            logits = _forward_logits(model, view, prefill.input_ids, cache)
        while len(generated) < max_tokens:
            # Cast before the argmax: bfloat16 ties against a vocabulary this large are common
            # and the reference resolves them in float32.
            token = int(logits[0, -1].float().argmax().item())
            generated.append(token)
            text = tokenizer.decode(generated)
            piece = text[len(previous_text) :]
            previous_text = text
            logits = _forward_logits(model, view, (token,), cache)
            yield token, piece
            if token in terminators:
                return


def generate_turn_tokens(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    max_tokens: int,
    *,
    view: ArchitectureView,
    spec: ModelSpec,
) -> tuple[list[int], str]:
    """Torch generation from token ids, returning the tokens and why the turn stopped.

    The acceptance harness needs both, and it starts from recorded ``prompt_ids`` rather than
    from a prompt string, so it cannot go through the string-shaped entry point.
    """
    stop_ids = _stop_ids(tokenizer)
    ids: list[int] = []
    thinking = _ThinkingTracker(
        enabled=spec.chat.thinking in {"inference", "trained"},
        max_tokens=spec.chat.max_think_tokens,
    )
    reason = _consume_stream(
        torch_greedy_stream(
            model, view, tokenizer, prompt_ids, max_tokens, eos_ids=config_eos_ids(model)
        ),
        ids=ids,
        thinking=thinking,
        tokenizer=tokenizer,
        stop_ids=stop_ids,
        capture=None,
        turn_cache=None,
    )
    return ids, reason


def _stop_ids(tokenizer: Any) -> set[int]:
    stop_ids: set[int] = set()
    with contextlib.suppress(Exception):  # tokenizer wrappers vary
        stop_ids.add(tokenizer.convert_tokens_to_ids("</tool_call>"))
    return stop_ids


def generate_turn_with_count(
    model: Any,
    tokenizer: Any,
    prompt: str,
    sampler: Any,
    max_tokens: int,
    turn_cache: TurnCacheBase | None = None,
    *,
    spec: ModelSpec,
    capture: Any | None = None,
    view: ArchitectureView | None = None,
) -> tuple[str, int, int]:
    """Generate one assistant turn, stopping as soon as the tool call closes.

    Returns decoded text, total generated tokens, and tokens spent in the first think block.
    When a thinking mode exhausts its budget, a closing tag is inserted into the raw output and
    generation continues until the visible note and tool call are complete.
    """
    stop_ids = _stop_ids(tokenizer)
    if is_torch_model(model):
        return _generate_turn_torch(
            model,
            tokenizer,
            prompt,
            max_tokens,
            turn_cache=turn_cache,
            spec=spec,
            capture=capture,
            view=view,
            sampler=sampler,
            stop_ids=stop_ids,
        )

    import mlx.core as mx
    from mlx_lm import stream_generate

    kwargs: dict[str, Any] = {}
    prompt_input: Any = prompt
    prompt_ids: list[int] = []
    if capture is not None and turn_cache is not None:
        raise ValueError("capture requires the resolved no-reuse cache strategy")
    if turn_cache is not None:
        prompt_ids = (
            encode_prompt(tokenizer, prompt)
            if isinstance(turn_cache, HistoryCache)
            else list(tokenizer.encode(prompt))
        )
        prompt_input = mx.array(turn_cache.prepare(prompt_ids))
        kwargs["prompt_cache"] = turn_cache.cache
    if isinstance(turn_cache, HistoryCache):
        if turn_cache.model is not model:
            raise ValueError("history cache belongs to a different model")
        kwargs["prefill_step_size"] = turn_cache.prefill_step_size

    ids: list[int] = []
    thinking = _ThinkingTracker(
        enabled=spec.chat.thinking in {"inference", "trained"},
        max_tokens=spec.chat.max_think_tokens,
    )

    context = (
        capture.generation(model, tokenizer, prompt, turn_cache=turn_cache)
        if capture is not None
        else contextlib.nullcontext(
            turn_cache.generation_model if isinstance(turn_cache, HistoryCache) else model
        )
    )
    with context as generation_model:
        stream = stream_generate(
            generation_model,
            tokenizer,
            prompt=prompt_input,
            max_tokens=max_tokens,
            sampler=sampler,
            **kwargs,
        )
        try:
            _consume_stream(
                ((response.token, response.text or "") for response in stream),
                ids=ids,
                thinking=thinking,
                tokenizer=tokenizer,
                stop_ids=stop_ids,
                capture=capture,
                turn_cache=turn_cache,
            )
        finally:
            if (capture is not None or isinstance(turn_cache, HistoryCache)) and callable(
                getattr(stream, "close", None)
            ):
                stream.close()
    if turn_cache is not None:
        turn_cache.commit(prompt_ids, ids)
    think_tokens = thinking.tokens if thinking.started else 0
    return thinking.decoded_text(ids, tokenizer), len(ids), think_tokens


def _generate_turn_torch(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    *,
    turn_cache: TurnCacheBase | None,
    spec: ModelSpec,
    capture: Any | None,
    view: ArchitectureView | None,
    sampler: Any,
    stop_ids: set[int],
) -> tuple[str, int, int]:
    """Torch branch of :func:`generate_turn_with_count`.

    Same thinking tracker, same stop rule, a hand-rolled greedy loop in place of
    ``mlx_lm.stream_generate``. Only ``cache_strategy: none`` is implemented, which is what
    stage two ran under and what the golden records exercise.
    """
    if view is None:
        raise ValueError(
            "torch generation needs the architecture view: the loop reads through "
            "view.native_readout and builds its within-turn cache with view.make_cache"
        )
    if turn_cache is not None:
        raise NotImplementedError(
            "torch generation implements cache_strategy 'none' only; the reuse strategies "
            "are deferred and the golden records never exercised them (WS-B)"
        )
    temperature = getattr(sampler, "sampling_temperature", None)
    if temperature not in (None, 0.0):
        # Silently ignoring a temperature would produce a plausible trajectory that no test
        # fails on and that does not match the sampler the caller asked for.
        raise NotImplementedError(
            f"torch generation is greedy; the caller asked for temperature {temperature}"
        )

    prompt_ids = encode_prompt(tokenizer, prompt)
    ids: list[int] = []
    thinking = _ThinkingTracker(
        enabled=spec.chat.thinking in {"inference", "trained"},
        max_tokens=spec.chat.max_think_tokens,
    )
    context = (
        capture.generation(model, tokenizer, prompt, turn_cache=None)
        if capture is not None
        else contextlib.nullcontext(model)
    )
    # The eos set comes from the model, not from the capture wrapper around it: a wrapper is
    # not required to forward `.config`, and a missing set would silently become an empty one.
    terminators = config_eos_ids(model)
    with context as generation_model:
        _consume_stream(
            torch_greedy_stream(
                generation_model, view, tokenizer, prompt_ids, max_tokens, eos_ids=terminators
            ),
            ids=ids,
            thinking=thinking,
            tokenizer=tokenizer,
            stop_ids=stop_ids,
            capture=capture,
            turn_cache=None,
        )
    think_tokens = thinking.tokens if thinking.started else 0
    return thinking.decoded_text(ids, tokenizer), len(ids), think_tokens


def generate_turn(model: Any, tokenizer: Any, prompt: str, sampler: Any, max_tokens: int) -> str:
    """Text-only view of :func:`generate_turn_with_count`, for callers that ignore token counts."""
    spec, _, _ = _compatibility_runner_inputs(None, None, None)
    return generate_turn_with_count(model, tokenizer, prompt, sampler, max_tokens, spec=spec)[0]


def call_signatures(steps: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Each executed step as (tool name, its arguments), skipping unparsed turns.

    Same convention as :func:`detect_loop`: a step carries its call under ``action``, and a
    parse-error step carries none and is not a call.
    """
    signatures = []
    for step in steps:
        action = step.get("action")
        if not action:
            continue
        arguments = action.get("arguments") or {}
        signatures.append((str(action.get("name")), repr(sorted(arguments.items()))))
    return signatures


def longest_identical_run(signatures: list[tuple[str, str]]) -> int:
    """The original column, with its original arithmetic, so old tables stay readable.

    One call is a run of one and no calls is a run of none, which is why this is not simply
    ``longest_period_run(signatures, 1)``: that function reports 0 when nothing repeats, and
    changing the old column's values would silently rewrite every table that quoted it.

    One difference from the records script this is lifted from: that version turned an
    unparsed turn into the signature ``null`` rather than skipping it. It cannot change any
    real trajectory, because a parse error ends the run, so at most one such step exists and it
    is last. It would differ on a synthetic step list, and that is worth knowing rather than
    discovering.
    """
    longest = 1 if signatures else 0
    run = 1
    for earlier, later in zip(signatures, signatures[1:], strict=False):
        run = run + 1 if earlier == later else 1
        longest = max(longest, run)
    return longest


def longest_period_run(signatures: list[tuple[str, str]], period: int) -> int:
    """Length of the longest stretch that repeats with the given period.

    A stretch counts only if it holds at least two full cycles, so a period is never read off
    a sequence too short to show it.
    """
    if period < 1 or len(signatures) < 2 * period:
        return 0
    best = 0
    start = 0
    for index in range(period, len(signatures)):
        if signatures[index] != signatures[index - period]:
            start = index - period + 1
        length = index - start + 1
        if length >= 2 * period:
            best = max(best, length)
    return best


def repetition_metrics(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Both repetition columns: the old identical-run one and a cycle-aware one.

    ``longest_identical_run`` counts consecutive identical calls, which a model alternating two
    failing calls defeats completely: it reads 1 while the trajectory loops. That column was
    quoted across a day of comparison tables before the two-cycle was noticed, so it is kept
    rather than replaced and the cycle-aware figure is reported beside it. A reader of an old
    table can still find the number that table used.

    ``cycle_period`` is the shortest period achieving ``longest_period_run``; a 1 means the two
    columns describe the same repetition and anything larger means they do not.
    """
    signatures = call_signatures(steps)
    identical = longest_identical_run(signatures)
    best_length = longest_period_run(signatures, 1)
    best_period = 1 if best_length else 0
    for period in range(2, len(signatures) // 2 + 1):
        length = longest_period_run(signatures, period)
        if length > best_length:
            best_length = length
            best_period = period
    return {
        "longest_identical_run": identical,
        "longest_period_run": best_length,
        "cycle_period": best_period if best_length else 0,
        "distinct_calls": len(set(signatures)),
        "executed_calls": len(signatures),
    }


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
    capture: Any | None = None,
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
        {"role": "system", "content": system_prompt(spec=spec)},
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
        capture_kwargs = {}
        if capture is not None:
            capture.set_context(
                task_id=task.task_id,
                step=index,
                keep_last=keep_last,
                messages=window_messages(messages, keep_last=keep_last),
            )
            capture_kwargs["capture"] = capture
        raw, n_tokens, think_tokens = generate_turn_with_count(
            model, tokenizer, prompt, sampler, max_tokens, turn_cache, spec=spec, **capture_kwargs
        )
        trajectory.turns += 1
        trajectory.generated_tokens += n_tokens
        trajectory.think_tokens += think_tokens
        thinking, action_text = strip_thinking(raw)
        try:
            turn = parse_turn(action_text)
        except ActionParseError as error:
            # A turn that ran out of budget and produced nothing parseable was cut off; it is
            # not a model that answered wrongly. Scoring the two as one outcome charges the
            # model for a cap we chose.
            truncated = n_tokens >= max_tokens
            trajectory.parse_error = str(error)
            trajectory.truncated = truncated
            trajectory.steps.append(
                {
                    "index": index,
                    "thinking": thinking,
                    "think_tokens": think_tokens,
                    "raw": raw,
                    "parse_error": str(error),
                    "truncated": truncated,
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
    trajectory.repetition = repetition_metrics(trajectory.steps)
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
