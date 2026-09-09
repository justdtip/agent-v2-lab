"""The chunked loss must be the unchunked loss, in value and in gradient.

The whole point of chunking is that it changes peak memory and nothing else. A test that only
compared losses would miss the failure that matters -- a gradient that is wrong by the number of
chunks, which trains a model perfectly happily.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from local_llm_lab.training.chunked_ce import (  # noqa: E402
    IGNORE_INDEX,
    chunked_linear_cross_entropy,
    shift_for_causal_lm,
)

VOCAB, HIDDEN, TOKENS = 97, 16, 40


def _inputs(seed: int = 0, *, masked: bool = True):
    torch.manual_seed(seed)
    hidden = torch.randn(TOKENS, HIDDEN, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, VOCAB, (TOKENS,))
    if masked:
        labels[:7] = IGNORE_INDEX  # a prompt span, as `mask_prompt` produces
    return hidden, weight, labels


def _reference(hidden, weight, labels):
    import torch.nn.functional as F

    logits = F.linear(hidden, weight)
    return F.cross_entropy(logits.float(), labels, ignore_index=IGNORE_INDEX, reduction="sum")


@pytest.mark.parametrize("chunk_size", [TOKENS, 4096])
def test_a_single_chunk_is_bit_for_bit_the_unchunked_loss(chunk_size: int) -> None:
    """With one chunk the additions happen in the same order, so nothing may move at all."""
    hidden_a, weight_a, labels = _inputs()
    expected = _reference(hidden_a, weight_a, labels)
    expected.backward()

    hidden_b, weight_b, _ = _inputs()
    got, count = chunked_linear_cross_entropy(hidden_b, weight_b, labels, chunk_size=chunk_size)
    got.backward()

    assert count.item() == TOKENS - 7
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden_b.grad, hidden_a.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight_b.grad, weight_a.grad, rtol=0, atol=0)


@pytest.mark.parametrize("chunk_size", [1, 7, 13])
def test_many_chunks_agree_to_float32_summation_order(chunk_size: int) -> None:
    """Splitting the sum reorders it, and reordering a float32 sum moves the last bits.

    The tolerance is set to what reassociating a few dozen float32 additions costs -- the observed
    worst case here is 2e-6 relative on one weight element in 1,552 -- and is far tighter than any
    real defect. A gradient that missed a chunk, or counted one twice, is wrong by a factor, not by
    an ulp. The exactness that *is* required is asserted in the single-chunk test above.
    """
    hidden_a, weight_a, labels = _inputs()
    expected = _reference(hidden_a, weight_a, labels)
    expected.backward()

    hidden_b, weight_b, _ = _inputs()
    got, count = chunked_linear_cross_entropy(hidden_b, weight_b, labels, chunk_size=chunk_size)
    got.backward()

    assert count.item() == TOKENS - 7
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hidden_b.grad, hidden_a.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weight_b.grad, weight_a.grad, rtol=1e-5, atol=1e-5)


def test_a_fully_masked_row_contributes_nothing_and_counts_nothing() -> None:
    """`reduction="sum"` over an all-ignored chunk is zero, not NaN, so it can be accumulated."""
    hidden, weight, labels = _inputs(masked=False)
    labels[:] = IGNORE_INDEX
    total, count = chunked_linear_cross_entropy(hidden, weight, labels, chunk_size=8)
    assert count.item() == 0
    assert total.item() == 0.0


def test_the_shift_matches_the_model_the_loss_is_for() -> None:
    """Our shift plus our loss reproduces `Gemma3ForCausalLM`'s own reported loss.

    This is the check that the off-by-one is right. It compares against the model's loss rather
    than against a hand-written shift, so it stays honest if upstream changes its convention.
    """
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=8,
        rms_norm_eps=1e-6, use_cache=False,
    )
    model = Gemma3ForCausalLM(config).eval()
    ids = torch.randint(0, 64, (2, 12))
    labels = ids.clone()
    labels[:, :3] = IGNORE_INDEX

    out = model(input_ids=ids, labels=labels, use_cache=False)
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state

    flat_hidden, flat_labels = shift_for_causal_lm(hidden, labels)
    total, count = chunked_linear_cross_entropy(
        flat_hidden, model.lm_head.weight, flat_labels, chunk_size=5
    )
    # `Trainer` normalises a summed loss by the token count; the model reports the mean.
    torch.testing.assert_close(total / count, out.loss, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("hidden_shape", "label_shape", "match"),
    [((2, 3, 4), (6,), "must be"), ((6, 4), (2, 3), "must be")],
)
def test_shapes_are_refused_rather_than_broadcast(hidden_shape, label_shape, match) -> None:
    hidden = torch.zeros(hidden_shape)
    labels = torch.zeros(label_shape, dtype=torch.long)
    weight = torch.zeros(VOCAB, hidden_shape[-1])
    with pytest.raises(ValueError, match=match):
        chunked_linear_cross_entropy(hidden, weight, labels, chunk_size=2)
