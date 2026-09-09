"""Three facts about the Gemma 3 cache that every seat's offset handling depends on.

Measured on CPU, torch 2.14.0, transformers 5.16.1, on the machine that wrote this file.
**Not verified on CUDA.** These are pinned as tests because they are defaults of a dependency,
not of this codebase, and a torch or transformers upgrade can change them silently.

The three facts (CUDA-MIGRATION-RESEARCH-BRIEF-ANSWERS-2026-09-09 §3):

1. ``get_seq_length()`` is the **absolute** position on both layer types, not the window-clipped
   length. The clipping applies to the stored tensors only.
2. So ``keys.shape[-2]`` is **not** the offset: on a sliding layer past the window it disagrees
   with ``get_seq_length()``. Deriving an offset from the tensor shape is wrong on five of every
   six Gemma 3 layers.
3. ``crop()`` on a sliding layer past the window raises unless ``activate_past_recording()`` was
   called before the window filled, and then accepts only a negative argument.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig  # noqa: E402

SLIDING_WINDOW = 8
ABSOLUTE_TOKENS = 12
HEAD_DIM = 8


def _config() -> Gemma3TextConfig:
    """A four-layer Gemma 3 alternating sliding and full attention. No weights are needed."""
    return Gemma3TextConfig(
        num_hidden_layers=4,
        sliding_window=SLIDING_WINDOW,
        sliding_window_pattern=2,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        intermediate_size=64,
        vocab_size=128,
    )


def _filled_cache(*, record_past: bool = False) -> DynamicCache:
    """A cache advanced to ``ABSOLUTE_TOKENS``, one token at a time after a six-token prefill."""
    config = _config()
    cache = DynamicCache(config=config)

    def kv(n: int):
        return torch.zeros(1, 1, n, HEAD_DIM), torch.zeros(1, 1, n, HEAD_DIM)

    keys, values = kv(1)
    for layer_idx in range(4):
        cache.update(keys, values, layer_idx)
    if record_past:
        # Must be activated before the window fills, or `crop` can never roll back.
        for layer in cache.layers:
            if getattr(layer, "is_sliding", False):
                layer.activate_past_recording()
    for _ in range(ABSOLUTE_TOKENS - 1):
        keys, values = kv(1)
        for layer_idx in range(4):
            cache.update(keys, values, layer_idx)
    return cache


def test_get_seq_length_is_absolute_on_both_layer_types() -> None:
    cache = _filled_cache()
    for layer_idx, layer in enumerate(cache.layers):
        assert layer.get_seq_length() == ABSOLUTE_TOKENS, (
            f"layer {layer_idx} (is_sliding={layer.is_sliding}) reported "
            f"{layer.get_seq_length()} for an absolute position of {ABSOLUTE_TOKENS}"
        )
        assert cache.get_seq_length(layer_idx) == ABSOLUTE_TOKENS


def test_stored_shape_is_not_the_offset_on_sliding_layers() -> None:
    """The trap. A sliding layer stores `sliding_window - 1` while absolute is far ahead."""
    cache = _filled_cache()
    sliding = [layer for layer in cache.layers if layer.is_sliding]
    full = [layer for layer in cache.layers if not layer.is_sliding]
    assert sliding and full, "the fixture must exercise both layer types"

    for layer in sliding:
        assert layer.keys.shape[-2] == SLIDING_WINDOW - 1
        assert layer.keys.shape[-2] != layer.get_seq_length()
        # The window's start in absolute coordinates is available, and is not the shape.
        _kv_length, kv_offset = layer.get_mask_sizes(1)
        assert kv_offset == ABSOLUTE_TOKENS - SLIDING_WINDOW + 1

    for layer in full:
        # On a full layer the shape happens to equal the offset, which is why the bug hides.
        assert layer.keys.shape[-2] == layer.get_seq_length() == ABSOLUTE_TOKENS
        assert layer.get_mask_sizes(1)[1] == 0


def test_crop_on_a_sliding_layer_requires_recording_and_a_negative_argument() -> None:
    without = _filled_cache(record_past=False)
    sliding = next(layer for layer in without.layers if layer.is_sliding)
    assert sliding.get_seq_length() >= SLIDING_WINDOW
    with pytest.raises(RuntimeError, match="activate_past_recording"):
        sliding.crop(-2)
    with pytest.raises(RuntimeError, match="activate_past_recording"):
        sliding.crop(2)

    with_recording = _filled_cache(record_past=True)
    sliding = next(layer for layer in with_recording.layers if layer.is_sliding)
    with pytest.raises(RuntimeError, match="negative int"):
        sliding.crop(2)
    sliding.crop(-3)
    assert sliding.get_seq_length() == ABSOLUTE_TOKENS - 3
    assert sliding.keys.shape[-2] == SLIDING_WINDOW - 1
