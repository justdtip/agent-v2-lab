"""Cache reuse held to the real ``DynamicCache``, not to a stub of one.

Every rule the strategies obey is a property of ``transformers.cache_utils``, so a stub would
only prove they were typed in correctly. The fixture here is the one Research pinned in
``test_gemma3_cache.py``: a four-layer Gemma 3 alternating sliding and full attention, small
enough to need no weights and real enough to have the trap in it.

The trap being: on a sliding layer past the window the stored tensor is ``sliding_window - 1``
long while the absolute position runs far ahead, and on a full layer the two coincide. Anything
that reads the shape is correct on one layer type and wrong on the other.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig  # noqa: E402

from local_llm_lab.pipeline.torch_cache import (  # noqa: E402
    TorchSnapshotCache,
    TorchTrimCache,
    cache_offset,
    enable_rollback,
    rewind,
)

SLIDING_WINDOW = 8
HEAD_DIM = 8
LAYERS = 4


def _config(pattern: int = 2) -> Gemma3TextConfig:
    return Gemma3TextConfig(
        num_hidden_layers=LAYERS,
        sliding_window=SLIDING_WINDOW,
        sliding_window_pattern=pattern,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        intermediate_size=64,
        vocab_size=128,
    )


def _new_cache() -> DynamicCache:
    return DynamicCache(config=_config())


def _advance(cache: DynamicCache, tokens: int) -> None:
    """Feed ``tokens`` one at a time, as decoding does."""
    for _ in range(tokens):
        keys = torch.zeros(1, 1, 1, HEAD_DIM)
        values = torch.zeros(1, 1, 1, HEAD_DIM)
        for layer_idx in range(LAYERS):
            cache.update(keys, values, layer_idx)


def test_the_offset_comes_from_the_api_and_not_from_the_stored_shape() -> None:
    # Unarmed, because arming changes what a sliding layer stores; that cost is its own test.
    cache = _new_cache()
    _advance(cache, 12)

    assert cache_offset(cache) == 12
    sliding = [layer for layer in cache.layers if layer.is_sliding]
    full = [layer for layer in cache.layers if not layer.is_sliding]
    assert sliding and full, "the fixture must exercise both layer types"
    for layer in sliding:
        assert layer.keys.shape[-2] == SLIDING_WINDOW - 1 != cache_offset(cache), (
            "reading the shape here gives 7 where the absolute position is 12"
        )
    for layer in full:
        assert layer.keys.shape[-2] == cache_offset(cache), (
            "and gives the right answer here, which is why the bug hides"
        )


def test_arming_rollback_makes_a_sliding_layer_unbounded() -> None:
    """The cost of a rewindable cache, measured rather than assumed.

    An unarmed sliding layer stores ``sliding_window - 1`` entries however long the context
    runs. An armed one keeps the past it would need to roll back into, so it stores everything.
    On Gemma 3 4B that is 29 of 34 layers, and the ratio at the map's longest episode, 2,749
    positions against a 1,024 window, is 2.15x on the whole cache. Below the window it costs
    nothing, which is why a short test would have missed it.

    This is the measured need the deferral of the reuse strategies was waiting on: turning one
    on is not free, and the price is paid only on long contexts.
    """
    for tokens in (12, 40, 200):
        unarmed = _new_cache()
        _advance(unarmed, tokens)
        armed = _new_cache()
        enable_rollback(armed)
        _advance(armed, tokens)

        unarmed_layer = next(layer for layer in unarmed.layers if layer.is_sliding)
        armed_layer = next(layer for layer in armed.layers if layer.is_sliding)

        assert unarmed_layer.keys.shape[-2] == SLIDING_WINDOW - 1, (
            "an unarmed sliding layer is bounded by its window at any length"
        )
        assert armed_layer.keys.shape[-2] == tokens, (
            "an armed one grows with the context, because rolling back needs the past"
        )
        assert cache_offset(unarmed) == cache_offset(armed) == tokens, (
            "both report the same absolute position; only the storage differs"
        )


def test_rewind_takes_a_positive_count_and_negates_it_once() -> None:
    cache = _new_cache()
    enable_rollback(cache)
    _advance(cache, 12)

    assert rewind(cache, 3) == 9
    assert rewind(cache, 0) == 9, "removing nothing is not an error"
    with pytest.raises(ValueError, match="positive number"):
        rewind(cache, -3)


def test_crop_refuses_a_positive_count_which_is_why_rewind_owns_the_sign() -> None:
    """The dependency's rule, pinned here because the strategies depend on it."""
    cache = _new_cache()
    enable_rollback(cache)
    _advance(cache, 12)
    with pytest.raises(RuntimeError, match="negative int"):
        cache.crop(3)


def test_never_arming_rollback_raises_which_is_the_safe_failure() -> None:
    cache = _new_cache()
    _advance(cache, 12)
    with pytest.raises(RuntimeError, match="past states|activate_past_recording"):
        cache.crop(-3)


def test_arming_late_is_refused_because_it_would_be_silently_wrong() -> None:
    """The finding: arming after the window fills does not raise, it corrupts.

    Rewinding a late-armed layer reports the right offset and holds the wrong contents, so
    nothing downstream can catch it. `enable_rollback` refuses instead.
    """
    late = _new_cache()
    _advance(late, 12)
    with pytest.raises(ValueError, match="armed before the first token"):
        enable_rollback(late)


def test_what_arming_late_would_have_done_measured_by_contents() -> None:
    """The refused case, measured, so the refusal is evidence rather than caution."""

    def build(arm_first: bool) -> DynamicCache:
        cache = _new_cache()
        if arm_first:
            enable_rollback(cache)
        for token in range(12):
            keys = torch.full((1, 1, 1, HEAD_DIM), float(token))
            for layer_idx in range(LAYERS):
                cache.update(keys, keys, layer_idx)
        if not arm_first:
            for layer in cache.layers:
                if layer.is_sliding:
                    layer.activate_past_recording()
        return cache

    def sliding_keys(cache: DynamicCache) -> list[float]:
        layer = next(layer for layer in cache.layers if layer.is_sliding)
        return [float(value) for value in layer.keys[0, 0, :, 0]]

    early, late = build(True), build(False)
    early.crop(-6)
    late.crop(-6)

    assert cache_offset(early) == cache_offset(late) == 6, "both report the same offset"
    assert sliding_keys(early) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert sliding_keys(late) == [5.0], (
        "the late-armed layer holds one token where it should hold six, and reports the same "
        "offset as the correct one; that is why enable_rollback refuses rather than warns"
    )


def test_arming_at_construction_is_what_makes_a_rewind_correct() -> None:
    early = _new_cache()
    armed = enable_rollback(early)
    _advance(early, 12)
    assert armed == 2, "two of the four layers slide under this pattern"
    assert rewind(early, 3) == 9


def test_a_model_with_no_sliding_layers_arms_nothing_and_that_is_not_an_error() -> None:
    # `sliding_window_pattern` is the period at which a full-attention layer appears, so 1
    # makes every layer full. Setting it large does the opposite and makes them all slide,
    # which is how the first draft of this test got it backwards.
    cache = DynamicCache(config=_config(pattern=1))
    assert not any(layer.is_sliding for layer in cache.layers)
    assert enable_rollback(cache) == 0
    _advance(cache, 12)
    assert rewind(cache, 3) == 9


def test_the_trim_strategy_reuses_the_shared_prefix() -> None:
    cache_holder = TorchTrimCache(_new_cache)
    first = list(range(10))
    assert cache_holder.prepare(first) == first, "nothing is cached yet, so everything is new"
    _advance(cache_holder.cache, 10)
    cache_holder.commit(first, [])

    second = list(range(10)) + [100, 101]
    assert cache_holder.prepare(second) == [100, 101], "only the new tail is forwarded"
    assert cache_holder.reused_tokens == 10
    assert cache_offset(cache_holder.cache) == 10


def test_the_trim_strategy_rewinds_when_the_prompt_diverges() -> None:
    holder = TorchTrimCache(_new_cache)
    first = list(range(12))
    holder.prepare(first)
    _advance(holder.cache, 12)
    holder.commit(first, [])
    assert cache_offset(holder.cache) == 12

    diverged = list(range(6)) + [900, 901]
    suffix = holder.prepare(diverged)
    assert suffix == [900, 901]
    assert cache_offset(holder.cache) == 6, (
        "the cache is rewound to the shared prefix, using the API's absolute offset"
    )
    assert holder.rebuilds == 1, "a rewind is not a rebuild"


def test_the_trim_strategy_never_consumes_the_whole_prompt() -> None:
    """The model needs at least one token to forward, so a full match still leaves one."""
    holder = TorchTrimCache(_new_cache)
    prompt = list(range(10))
    holder.prepare(prompt)
    _advance(holder.cache, 10)
    holder.commit(prompt, [])

    suffix = holder.prepare(prompt)
    assert suffix == [9], "one token is always left to forward"
    assert cache_offset(holder.cache) == 9


def test_the_trim_strategy_rebuilds_rather_than_continuing_when_it_cannot_rewind() -> None:
    """Unarmed rollback is the case here.

    Rebuilding is slow and correct; continuing would be fast and wrong.
    """
    holder = TorchTrimCache(_new_cache)
    holder.prepare(list(range(12)))
    # Disarm what the strategy armed, then fill the window, so crop can no longer roll back.
    holder.cache = _new_cache()
    _advance(holder.cache, 12)
    holder.tokens = list(range(12))

    suffix = holder.prepare(list(range(6)) + [900, 901])
    assert holder.rebuilds == 2, "the failed rewind forced a rebuild"
    assert suffix == list(range(6)) + [900, 901], "everything is forwarded again"
    assert cache_offset(holder.cache) == 0


def test_the_snapshot_strategy_rewinds_to_its_fixed_prefix_every_turn() -> None:
    holder = TorchSnapshotCache(_new_cache, prefix_tokens=6)
    first = list(range(10))
    assert holder.prepare(first) == first
    _advance(holder.cache, 10)

    second = list(range(6)) + [200, 201, 202]
    assert holder.prepare(second) == [200, 201, 202]
    assert cache_offset(holder.cache) == 6
    assert holder.reused_tokens == 6


def test_the_snapshot_strategy_refuses_a_prefix_that_moved() -> None:
    holder = TorchSnapshotCache(_new_cache, prefix_tokens=6)
    holder.prepare(list(range(10)))
    _advance(holder.cache, 10)
    with pytest.raises(ValueError, match="immutable prefix changed"):
        holder.prepare([99] + list(range(1, 10)))


def test_the_snapshot_strategy_refuses_a_prefix_outside_the_prompt() -> None:
    holder = TorchSnapshotCache(_new_cache, prefix_tokens=6)
    with pytest.raises(ValueError, match="strictly within"):
        holder.prepare(list(range(4)))


def test_layers_that_disagree_about_the_position_raise_rather_than_pick_one() -> None:
    """A partial update; trimming against one layer would leave the others attending elsewhere."""
    cache = _new_cache()
    enable_rollback(cache)
    _advance(cache, 8)
    keys = torch.zeros(1, 1, 1, HEAD_DIM)
    cache.update(keys, keys, 0)  # advance one layer only
    with pytest.raises(ValueError, match="disagree about the absolute position"):
        cache_offset(cache)
