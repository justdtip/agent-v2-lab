"""Cache reuse on torch, against the real ``DynamicCache`` rather than an idea of one.

Every rule here is a property of ``transformers.cache_utils`` and not of this codebase, which
is why the tests build a real four-layer Gemma 3 cache instead of a stub. A stub would only
prove the rules were typed in correctly.

The three that matter, all measured (transformers 5.16.1, torch 2.14.0, CPU):

**The offset comes from the API, never from a tensor shape.** ``get_seq_length()`` is the
absolute position on both layer types. A sliding layer stores ``sliding_window - 1`` entries
while its absolute position runs far ahead, so ``keys.shape[-2]`` disagrees with the offset on
five of every six Gemma 3 layers. On a full-attention layer the two happen to coincide, which
is exactly why reading the shape looks correct until it is run on the layers that rotate.

**Rewinding takes a negative count.** ``crop(3)`` raises once the window has been reached;
``crop(-3)`` removes three. The sign is not a convention this module gets to choose.

**Rollback must be armed at construction, and arming late is worse than not arming at all.**
``activate_past_recording()`` on every sliding layer, before a single token is written. Never
arming makes ``crop`` raise once the window is reached, which is a safe failure. Arming *after*
the window has filled does **not** raise: it rewinds to the right offset holding the wrong
contents.

Measured, on a four-layer fixture with each token's key set to its own index, rewinding from
absolute 12 back to absolute 6:

    armed at construction   offset 6, stored keys [0, 1, 2, 3, 4, 5]
    armed after 12 tokens   offset 6, stored keys [5]

Both report an offset of 6. One of them has the six tokens; the other has one and would attend
over a five-token hole with no error anywhere. So :func:`enable_rollback` refuses a cache that
has already advanced, rather than arming it and returning a number that looks like success.

**What decides whether a strategy is used is fidelity, not memory.** Arming rollback costs
2.15x on the KV cache at 2,749 positions (see :func:`enable_rollback`), and that is 2.15x of a
small base: read off the loaded config, this checkpoint is 4 KiB per layer per token, 136 KiB
per token across 34 layers, so 178 MB bounded against 383 MB armed. Inside the laptop's cap and
nothing on an 80 GB card. ``head_dim`` is the class default of 256 because the checkpoint's
config leaves it null, which is worth saying because deriving it from hidden size over heads
gives 320 and figures that are 25% too large.

So the gate is behavioural. Each strategy is an arm of the same acceptance gate as ``none`` and
must reproduce the ``none`` trajectories **byte for byte within one backend**. A strategy that
changes a single token is a defect and not a speed setting, because a rewindable sliding cache
is exactly where the stored keys and the attention mask can part company. That gate cannot run
until the tolerance runner has a ``none`` baseline against the real view, which is why
``make_turn_cache`` refuses all three meanwhile.

``history`` is not implemented here. It is Qwen 3.5's strategy, it carries snapshot files and a
generation-model proxy, and none of that is exercised by the golden records. It is named in
this docstring so its absence is a decision rather than an oversight.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TorchSnapshotCache",
    "TorchTrimCache",
    "cache_offset",
    "enable_rollback",
    "rewind",
]


def cache_offset(cache: Any) -> int:
    """The cache's absolute position, read from the API and checked for agreement.

    Every layer of one cache should stand at the same absolute position; they are all fed the
    same tokens. Disagreement means a partial update, and a strategy that trimmed against one
    layer's idea of the offset would leave the rest attending elsewhere. So this raises rather
    than picking a layer.
    """
    layers = list(getattr(cache, "layers", []) or [])
    if not layers:
        return int(cache.get_seq_length())
    offsets = {int(layer.get_seq_length()) for layer in layers}
    if len(offsets) > 1:
        raise ValueError(
            f"cache layers disagree about the absolute position: {sorted(offsets)}; "
            "trimming against any one of them would leave the others attending elsewhere"
        )
    return offsets.pop()


def enable_rollback(cache: Any) -> int:
    """Arm ``crop`` on every sliding layer. Must run before the window fills.

    Returns how many layers were armed, so a caller can record it rather than assume it. Zero
    is a legitimate answer on a model with no sliding layers and is not an error.

    Refuses a cache that has already advanced. Arming late succeeds and then rewinds to the
    right offset with the wrong contents, which no later check would catch; refusing here turns
    that into an error at the one point where the caller can still fix it.
    """
    if cache_offset(cache) > 0:
        raise ValueError(
            "rollback must be armed before the first token: a cache armed after its window "
            "has filled rewinds to the right offset holding the wrong contents, with no error"
        )
    armed = 0
    for layer in getattr(cache, "layers", []) or []:
        if getattr(layer, "is_sliding", False):
            layer.activate_past_recording()
            armed += 1
    return armed


def rewind(cache: Any, tokens_to_remove: int) -> int:
    """Remove ``tokens_to_remove`` tokens from the end of the cache. Returns the new offset.

    ``crop`` takes a negative count: a positive one raises once the window has been reached.
    Passing the caller's positive count straight through would work on short prompts and raise
    on long ones, so the negation happens here, once.
    """
    if tokens_to_remove < 0:
        raise ValueError(
            f"rewind takes a positive number of tokens to remove, not {tokens_to_remove}; "
            "the negation crop requires is applied here"
        )
    if tokens_to_remove:
        cache.crop(-tokens_to_remove)
    return cache_offset(cache)


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


class TorchTrimCache:
    """Reuse across turns by rewinding to the longest prefix the next prompt shares.

    The same strategy as the MLX ``TrimCache`` and the same safety property: a stale cache
    produces silently wrong logits rather than an error, so the offset is read back after every
    rewind and any disagreement rebuilds rather than continues.
    """

    def __init__(self, make_cache: Any) -> None:
        self._make_cache = make_cache
        self.cache: Any = None
        self.tokens: list[int] = []
        self.reused_tokens = 0
        self.encoded_tokens = 0
        self.rebuilds = 0
        self.sliding_layers_armed = 0

    def _reset(self) -> None:
        self.cache = self._make_cache()
        # Armed at construction, before a single token is written, because a layer that has
        # already filled its window has thrown away the past that crop would roll back into.
        self.sliding_layers_armed = enable_rollback(self.cache)
        self.tokens = []
        self.rebuilds += 1

    def prepare(self, token_ids: list[int]) -> list[int]:
        """Rewind to the shared prefix; return the suffix still to be forwarded."""
        if self.cache is None:
            self._reset()
        # Never consume the whole prompt: the model needs at least one token to forward.
        shared = min(_common_prefix_length(self.tokens, token_ids), len(token_ids) - 1)
        shared = max(shared, 0)
        drop = cache_offset(self.cache) - shared
        if drop > 0:
            try:
                rewind(self.cache, drop)
            except RuntimeError:
                # A layer that cannot roll back is not a cache we can trim; rebuilding is
                # correct and slow, where continuing would be fast and wrong.
                self._reset()
                shared = 0
        if cache_offset(self.cache) != shared:
            self._reset()
            shared = 0
        self.tokens = list(token_ids[:shared])
        self.reused_tokens += shared
        suffix = list(token_ids[shared:])
        self.encoded_tokens += len(suffix)
        return suffix

    def commit(self, token_ids: list[int], generated: list[int]) -> None:
        """Record what the cache now holds: the whole prompt plus what was generated."""
        self.tokens = list(token_ids) + list(generated)


class TorchSnapshotCache:
    """Reuse only a fixed, verified prefix, by rewinding to it at the start of every turn.

    The MLX version saved and restored the cache tensors. Here the prefix is by definition
    unchanged, so rewinding to its length reaches the same state without copying anything, and
    without the save-and-restore round trip that lost the position on rotating layers.
    """

    def __init__(self, make_cache: Any, prefix_tokens: int) -> None:
        self._make_cache = make_cache
        self.prefix_tokens = prefix_tokens
        self.cache: Any = None
        self.tokens: list[int] = []
        self.reused_tokens = 0
        self.encoded_tokens = 0
        self.sliding_layers_armed = 0
        self._prefix: list[int] | None = None

    def prepare(self, token_ids: list[int]) -> list[int]:
        if not 0 < self.prefix_tokens < len(token_ids):
            raise ValueError("prefix_tokens must lie strictly within the prompt")
        prefix = list(token_ids[: self.prefix_tokens])
        suffix = list(token_ids[self.prefix_tokens :])
        if self.cache is None:
            self.cache = self._make_cache()
            self.sliding_layers_armed = enable_rollback(self.cache)
            self._prefix = prefix
            self.encoded_tokens += len(token_ids)
            return list(token_ids)
        if prefix != self._prefix:
            raise ValueError("immutable prefix changed after snapshot creation")
        offset = cache_offset(self.cache)
        if offset < self.prefix_tokens:
            raise ValueError(
                f"cache holds {offset} tokens, fewer than the {self.prefix_tokens}-token "
                "prefix it is meant to have kept"
            )
        rewind(self.cache, offset - self.prefix_tokens)
        self.reused_tokens += self.prefix_tokens
        self.encoded_tokens += len(suffix)
        return suffix

    def commit(self, token_ids: list[int], generated: list[int]) -> None:
        """The live cache advances; the prefix it rewinds to does not move."""
