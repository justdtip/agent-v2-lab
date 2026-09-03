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

    assert view.run_block is original


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
