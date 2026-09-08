"""The observing forward, checked against stand-ins shaped like the two real decoders.

**No MLX and no model.** `ArchitectureView._observe_forward` manipulates the module's own objects
and computes nothing, so what it does can be shown exactly here: hand a module a known input, read
what its first block received, and read the mask each block was given. The stand-ins reproduce the
two dispatch shapes from the installed library, and each one is the case the other would get wrong.

The real decoders are exercised by `test_arch.py`, which loads the model library. This file is the
part that can be proved on any machine, and it is where a regression in the mechanism shows up
first.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_llm_lab.arch import ArchitectureView

#: Gemma 3 4B's own value, from `gemma3_text.py:190`: `h *= sqrt(hidden_size)`, and 2560 ** 0.5.
GEMMA_ENTRY_SCALE = 50.596442562694

#: The two shapes, from the installed library rather than from memory.
GEMMA_PATTERN = 6  # `sliding_window_pattern`; block i is global when i % pattern == pattern - 1
QWEN_INTERVAL = 4  # `full_attention_interval`; block i is attention when (i + 1) % interval == 0


class _Block:
    """A decoder block that computes nothing and carries the flag its family dispatches on."""

    def __init__(self, index: int, *, is_linear: bool | None = None):
        self.index = index
        self.calls: list[tuple[object, object]] = []
        if is_linear is not None:
            self.is_linear = is_linear

    def __call__(self, x, mask=None, cache=None):
        self.calls.append((x, mask))
        return f"block-{self.index}-output"


class _GemmaShaped:
    """Gemma's forward: an unconditional entry scale, two masks, dispatch on the loop index."""

    def __init__(self, blocks: int = 12):
        self.layers = [_Block(index) for index in range(blocks)]
        self.entry_calls: list[object] = []

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else f"embed({inputs})"
        # The scale sits AFTER the branch and applies to both, which is the footgun the design
        # names: a mid-network residual handed in as `input_embeddings` comes back scaled.
        h = f"{h}*{GEMMA_ENTRY_SCALE}"
        self.entry_calls.append(h)
        global_mask = f"global(cache={None if cache is None else cache[GEMMA_PATTERN - 1]})"
        sliding_mask = f"sliding(cache={None if cache is None else cache[0]})"
        for index, block in enumerate(self.layers):
            is_global = index % GEMMA_PATTERN == GEMMA_PATTERN - 1
            h = block(h, global_mask if is_global else sliding_mask, None)
        return h


class _QwenShaped:
    """Qwen's forward: no entry transform, one mask per kind, dispatch on `layer.is_linear`."""

    def __init__(self, blocks: int = 8):
        self.layers = [
            _Block(index, is_linear=(index + 1) % QWEN_INTERVAL != 0) for index in range(blocks)
        ]

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else f"embed({inputs})"
        for block in self.layers:
            # Reading the block's own attribute is the reason the proxy must forward attributes.
            h = block(h, "ssm" if block.is_linear else "attention", None)
        return h


def _view(module) -> SimpleNamespace:
    """Enough of a view to call the observer on: it reads `text_module` and `num_layers`."""
    return SimpleNamespace(text_module=module, num_layers=len(module.layers))


def _observe(module, ids="ids", **kwargs):
    return ArchitectureView._observe_forward(_view(module), ids, **kwargs)


def test_the_entry_transform_is_inherited_rather_than_reconstructed() -> None:
    """The defect this replaces: `embed_tokens(ids)` is the residual only where entry is identity.

    Gemma multiplies by about 50.6 before the first block. Reconstructing the layer-zero residual
    as the raw embedding was out by that factor on every layer of every reading, and a family that
    added or normalised instead would have been wrong differently with nothing to say so.
    """
    gemma = _GemmaShaped()
    entry, _ = _observe(gemma)

    assert entry == f"embed(ids)*{GEMMA_ENTRY_SCALE}", "what block 0 got, not what embed returned"
    assert entry != "embed(ids)"

    # And on a family whose entry is the identity, the same call returns the embedding unchanged,
    # so the observation is not a Gemma special case.
    qwen = _QwenShaped()
    assert _observe(qwen)[0] == "embed(ids)"


def test_masks_come_back_per_block_because_two_blocks_of_one_kind_can_differ() -> None:
    """Gemma hands a global mask to every sixth block and a windowed mask to the rest.

    A kind-keyed mapping cannot express that: every Gemma block is an attention module, so all
    twelve would have taken one entry and eleven of them would have been wrong.
    """
    gemma = _GemmaShaped(blocks=12)
    _, masks = _observe(gemma)

    assert set(masks) == set(range(12))
    globals_seen = {index for index, mask in masks.items() if mask.startswith("global")}
    assert globals_seen == {5, 11}, "block i is global when i % 6 == 5"
    assert all(masks[index].startswith("sliding") for index in set(range(12)) - globals_seen)


def test_the_models_own_cache_selection_is_inherited_not_reimplemented() -> None:
    """Gemma builds the global mask from `cache[pattern - 1]` and the sliding one from `cache[0]`.

    One cache of each kind, because the two carry different offsets once a window has rotated.
    Selecting by kind cannot tell Gemma's two attention caches apart, which was the fourth defect
    and the one no description would have caught.
    """
    gemma = _GemmaShaped(blocks=12)
    cache = [f"cache-{index}" for index in range(12)]
    _, masks = _observe(gemma, cache=cache)

    assert masks[5] == "global(cache=cache-5)"
    assert masks[0] == "sliding(cache=cache-0)"


def test_the_proxy_forwards_attribute_access_or_it_changes_what_it_observes() -> None:
    """Qwen dispatches on `layer.is_linear`; Gemma dispatches on the index.

    A proxy that answered neither would raise or mis-dispatch, and the masks it recorded would be
    a property of the proxy rather than of the model.
    """
    qwen = _QwenShaped(blocks=8)
    _, masks = _observe(qwen)

    attention = {index for index, mask in masks.items() if mask == "attention"}
    assert attention == {3, 7}, "block i is attention when (i + 1) % 4 == 0"
    assert all(masks[index] == "ssm" for index in set(range(8)) - attention)


def test_the_module_is_left_exactly_as_it_was_found() -> None:
    """Including when the forward raises, or one observation would poison every later one."""

    class _Angry(_GemmaShaped):
        def __call__(self, *args, **kwargs):
            raise RuntimeError("no")

    module = _Angry()
    before = list(module.layers)
    with pytest.raises(RuntimeError):
        _observe(module)
    assert module.layers == before and all(isinstance(b, _Block) for b in module.layers)


def test_a_decoder_that_does_not_iterate_its_own_layers_is_refused() -> None:
    """The observation is only evidence if every block was actually reached."""

    class _Partial(_GemmaShaped):
        def __call__(self, inputs, cache=None, input_embeddings=None):
            h = "embed"
            for block in self.layers[:2]:
                h = block(h, "m", None)
            return h

    with pytest.raises(ValueError, match="does not iterate its own layer list"):
        _observe(_Partial(blocks=12))
