from __future__ import annotations

import inspect

import mlx.core as mx
import numpy as np
import pytest

from local_llm_lab.probes import capture


class _Block:
    def __call__(self, value, *, mask=None, cache=None):
        del mask, cache
        return (value + 1.0).astype(mx.float32)


class _View:
    num_layers = 2

    def __init__(self) -> None:
        self.blocks = [_Block(), _Block()]
        self.calls = 0

    def residuals(self, ids, layers):
        self.calls += 1
        values = mx.array(ids).astype(mx.float32)[None, :, None]
        return {layer: values + float(layer) for layer in layers}

    def run_block(self, index, value, masks, cache_i=None):
        del masks
        return self.blocks[index](value, cache=cache_i)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]


def test_capture_residuals_uses_view_once_and_preserves_position_convention() -> None:
    view = _View()
    captured = capture.capture_residuals(view, [2, 4, 8], [0, 2], positions="all")

    assert view.calls == 1
    assert set(captured) == {0, 2}
    np.testing.assert_allclose(np.asarray(captured[2]), [[4.0], [6.0], [10.0]])


def test_response_mean_activations_returns_metadata_for_view_callers(monkeypatch) -> None:
    view = _View()
    monkeypatch.setattr(
        capture,
        "capture_residuals",
        lambda _view, _ids, _layers, *, positions: {1: mx.array([[1.0], [3.0]], dtype=mx.float32)},
    )

    means, metadata = capture.response_mean_activations(view, _Tokenizer(), "a", "b", [1])

    np.testing.assert_allclose(np.asarray(means[1]), [3.0])
    assert metadata == {"sequences": 1}


def test_response_mean_activations_keeps_mapping_for_legacy_model_callers(monkeypatch) -> None:
    legacy_model = object()
    monkeypatch.setattr(
        capture,
        "capture_residuals",
        lambda _view, _ids, _layers, *, positions: {1: mx.array([[1.0], [3.0]], dtype=mx.float32)},
    )

    means = capture.response_mean_activations(legacy_model, _Tokenizer(), "a", "b", [1])

    assert isinstance(means, dict)
    np.testing.assert_allclose(np.asarray(means[1]), [3.0])


def test_injection_tracks_arrays_cache_position_and_restores_block() -> None:
    class ArraysCache:
        state = [None, None]

    view = _View()
    original = view.blocks[0]
    cache = ArraysCache()
    with capture.InjectionHook(view, 0, mx.array([5.0]), alpha=2.0, at_positions=[2]) as hook:
        first = view.run_block(0, mx.zeros((1, 2, 1)), {}, cache)
        second = view.run_block(0, mx.zeros((1, 2, 1)), {}, cache)

    np.testing.assert_allclose(np.asarray(first), [[[1.0], [1.0]]])
    np.testing.assert_allclose(np.asarray(second), [[[11.0], [1.0]]])
    assert hook.injected == 1
    assert view.blocks[0] is original


def test_injection_uses_kv_cache_offset_for_absolute_position() -> None:
    class KVCache:
        offset = 4

    view = _View()
    with capture.InjectionHook(view, 0, mx.array([2.0]), at_positions=[5]) as hook:
        shifted = view.run_block(0, mx.zeros((1, 3, 1)), {}, KVCache())

    np.testing.assert_allclose(np.asarray(shifted), [[[1.0], [3.0], [1.0]]])
    assert hook.injected == 1


def test_injection_uses_kv_offset_before_block_updates_it() -> None:
    class KVCache:
        offset = 4

    class UpdatingView(_View):
        def run_block(self, index, value, masks, cache_i=None):
            result = super().run_block(index, value, masks, cache_i)
            cache_i.offset += value.shape[1]
            return result

    view = UpdatingView()
    cache = KVCache()
    with capture.InjectionHook(view, 0, mx.array([2.0]), at_positions=[5]):
        shifted = view.run_block(0, mx.zeros((1, 3, 1)), {}, cache)

    np.testing.assert_allclose(np.asarray(shifted), [[[1.0], [3.0], [1.0]]])
    assert cache.offset == 7


def test_injection_requires_registered_offset_for_prefilled_arrays_cache() -> None:
    class ArraysCache:
        # Recurrent cache state width is bounded and does not encode history length.
        state = [mx.zeros((1, 4, 8)), mx.zeros((1, 4, 8))]

    view = _View()
    cache = ArraysCache()
    with capture.InjectionHook(view, 0, mx.array([2.0]), at_positions=[7]) as hook:
        with pytest.raises(ValueError, match="register.*offset"):
            view.run_block(0, mx.zeros((1, 3, 1)), {}, cache)
        hook._register_offsetless_cache(cache, 6)
        shifted = view.run_block(0, mx.zeros((1, 3, 1)), {}, cache)

    np.testing.assert_allclose(np.asarray(shifted), [[[1.0], [3.0], [1.0]]])


def test_injection_restores_shared_view_run_block_after_an_error() -> None:
    view = _View()
    original = view.run_block

    with pytest.raises(RuntimeError), capture.InjectionHook(
        view, 0, mx.array([1.0]), positions=("at", 0)
    ):
        raise RuntimeError("boom")

    restored = view.run_block
    assert restored.__self__ is original.__self__
    assert restored.__func__ is original.__func__

    # A surviving wrapper would inject at position zero and turn this baseline 1 into 2.
    after_error = view.run_block(0, mx.zeros((1, 1, 1)), {}, None)
    np.testing.assert_allclose(np.asarray(after_error), [[[1.0]]])


def test_replacement_overwrites_selected_rows_without_addition() -> None:
    view = _View()

    with capture.InjectionHook(
        view,
        0,
        mx.array([7.0]),
        at_positions=[1],
        replace=True,
    ) as hook:
        result = view.run_block(0, mx.zeros((1, 3, 1)), {}, None)

    np.testing.assert_allclose(np.asarray(result), [[[1.0], [7.0], [1.0]]])
    assert result.dtype == mx.float32
    assert hook.calls == 1 and hook.injected == 1


def test_replacement_maps_ordered_rows_across_cached_calls() -> None:
    class Cache:
        offset = 0

    class UpdatingView(_View):
        def run_block(self, index, value, masks, cache_i=None):
            result = super().run_block(index, value, masks, cache_i)
            cache_i.offset += value.shape[1]
            return result

    view = UpdatingView()
    cache = Cache()
    with capture.InjectionHook(
        view,
        0,
        mx.array([[7.0], [9.0]]),
        at_positions=[1, 3],
        replace=True,
    ) as hook:
        first = view.run_block(0, mx.zeros((1, 2, 1)), {}, cache)
        second = view.run_block(0, mx.zeros((1, 2, 1)), {}, cache)

    np.testing.assert_allclose(np.asarray(first), [[[1.0], [7.0]]])
    np.testing.assert_allclose(np.asarray(second), [[[1.0], [9.0]]])
    assert hook.calls == 2 and hook.injected == 2


@pytest.mark.parametrize(
    ("vector", "at_positions", "message"),
    [
        (mx.array([1.0, 2.0]), [0], "hidden width"),
        (mx.array([[1.0], [2.0]]), [0], "row count"),
    ],
)
def test_replacement_validates_its_shape(vector, at_positions, message) -> None:
    view = _View()
    if message == "row count":
        with pytest.raises(ValueError, match=message):
            capture.InjectionHook(view, 0, vector, at_positions=at_positions, replace=True)
    else:
        with (
            capture.InjectionHook(view, 0, vector, at_positions=at_positions, replace=True),
            pytest.raises(ValueError, match=message),
        ):
            view.run_block(0, mx.zeros((1, 1, 1)), {}, None)


def test_replacement_rejects_nondefault_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        capture.InjectionHook(_View(), 0, mx.array([1.0]), replace=True, alpha=0.5)


def test_lora_block_mask_uses_view_owned_blocks_and_restores_after_error() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.lora_a = mx.ones((1, 1))
            self.lora_b = mx.ones((1, 1))

    class Block:
        def __init__(self) -> None:
            self.adapter = Adapter()

        def named_modules(self):
            return [("adapter", self.adapter)]

    class AdapterView(_View):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = [Block(), Block()]

    view = AdapterView()
    saved = [block.adapter.lora_b for block in view.blocks]
    with pytest.raises(RuntimeError), capture.lora_block_mask(view, keep_layers={1}) as masked:
        assert masked == 1
        assert float(mx.sum(view.blocks[0].adapter.lora_b).item()) == 0.0
        assert float(mx.sum(view.blocks[1].adapter.lora_b).item()) == 1.0
        raise RuntimeError("boom")

    for block, original in zip(view.blocks, saved, strict=True):
        assert bool(mx.array_equal(block.adapter.lora_b, original).item())


def test_capture_residuals_declares_binding_positions_annotation() -> None:
    parameter = inspect.signature(capture.capture_residuals).parameters["positions"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert "Literal['last', 'all'] | Sequence[int]" in str(parameter.annotation)


# ------------------------------------------------------- R18b capture dtype and the note span


class _NativeView(_View):
    """A view whose blocks emit bfloat16, so the R18b dtype switch is observable."""

    def residuals(self, ids, layers):
        self.calls += 1
        values = mx.array(ids).astype(mx.float32)[None, :, None] / 3.0
        return {layer: (values + float(layer)).astype(mx.bfloat16) for layer in layers}


def test_capture_residuals_native_dtype_returns_the_blocks_own_precision() -> None:
    """R18b: ``native`` hands back the block output untouched; ``float32`` casts as before."""
    view = _NativeView()

    native = capture.capture_residuals(view, [2, 4, 8], [1], positions="all", dtype="native")[1]
    upcast = capture.capture_residuals(view, [2, 4, 8], [1], positions="all")[1]

    assert native.dtype == mx.bfloat16
    assert upcast.dtype == mx.float32
    # Widening bfloat16 to float32 is exact, so the stored last-token view is identical either
    # way; the two paths separate only once something is pooled, as the note mean pools.
    assert bool(mx.all(native.astype(mx.float32) == upcast).item())
    assert mx.mean(native, axis=0).dtype == mx.bfloat16
    assert float(mx.mean(native, axis=0).item()) != float(mx.mean(upcast, axis=0).item())


def test_capture_residuals_rejects_an_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        capture.capture_residuals(_View(), [1, 2], [1], dtype="float16")


def test_capture_residuals_dtype_is_keyword_only_and_defaults_to_float32() -> None:
    parameter = inspect.signature(capture.capture_residuals).parameters["dtype"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == "float32"


def test_response_token_span_counts_the_prompt_prefix() -> None:
    tokenizer = _Tokenizer()

    joint, start, repaired = capture.response_token_span(tokenizer, "abc", "de")

    assert joint == tokenizer.encode("abcde")
    assert start == 3
    assert repaired is False


class _MergingTokenizer(_Tokenizer):
    """Characters merge pairwise, so an odd-length prompt is not a token prefix of the join."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        pairs = [text[index : index + 2] for index in range(0, len(text), 2)]
        return [sum(map(ord, pair)) for pair in pairs]


def test_response_token_span_repairs_a_merged_boundary_by_rebuilding() -> None:
    """P1's historical behaviour, kept byte for byte: the joint sequence is rebuilt."""
    tokenizer = _MergingTokenizer()

    joint, start, repaired = capture.response_token_span(tokenizer, "odd", "note")

    assert repaired is True
    assert start == len(tokenizer.encode("odd"))
    assert joint == [*tokenizer.encode("odd"), *tokenizer.encode("note")]


def test_response_token_span_rejects_an_empty_response() -> None:
    with pytest.raises(ValueError, match="empty"):
        capture.response_token_span(_Tokenizer(), "prompt", "")


# ------------------------------------------------------------------------- note_token_span


def test_note_token_span_counts_the_prompt_prefix_on_a_clean_boundary() -> None:
    tokenizer = _Tokenizer()

    joint, start, repairs = capture.note_token_span(tokenizer, "abc", "de")

    assert joint == tokenizer.encode("abcde")
    assert start == 3
    assert repairs == 0


def test_note_token_span_widens_over_a_merged_boundary_token() -> None:
    """The natural joint tokenisation is kept; the span widens back over the merged token."""
    tokenizer = _MergingTokenizer()
    prompt, note = "odd", "note"

    joint, start, repairs = capture.note_token_span(tokenizer, prompt, note)

    natural = tokenizer.encode(prompt + note)
    assert joint == natural, "the sequence the model sees must not be rebuilt"
    assert joint != [*tokenizer.encode(prompt), *tokenizer.encode(note)]
    counted = len(tokenizer.encode(prompt))
    assert repairs == 1
    assert start == counted - 1, "widened by exactly one token, never more"
    # That boundary token really is the merged one: it carries the prompt's last character.
    assert joint[start] == ord(prompt[-1]) + ord(note[0])
    assert joint[:start] == tokenizer.encode(prompt)[:start]


class _UnstableTokenizer(_Tokenizer):
    """Not prefix-stable at all: appending anything re-tokenises from character zero."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        offset = 1 if len(text) > 4 else 0
        return [ord(character) + offset for character in text]


def test_note_token_span_refuses_a_boundary_that_would_move_more_than_one_token() -> None:
    with pytest.raises(ValueError, match="more than one token"):
        capture.note_token_span(_UnstableTokenizer(), "abcd", "ef")


def test_note_token_span_rejects_an_empty_note() -> None:
    with pytest.raises(ValueError, match="empty"):
        capture.note_token_span(_Tokenizer(), "prompt", "")
