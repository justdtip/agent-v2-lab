"""What the injection hook does inside a training step, established by running it.

The bench's `Injection` is a forward hook that mutates a decoder block's output in place and tracks
absolute position in a mutable counter, because a decode step's slice is one token wide and the
hook cannot tell where it is from the slice alone. Both of those are safe in inference. Only one of
them is safe in training, and the unsafe one fails silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

IDS = [[1, 5, 9, 12, 30, 7, 3, 8]]


def _tiny():
    from transformers import AutoModelForCausalLM, LlamaConfig
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=True)).train()


def _loss_and_grads(model, hook):
    handle = model.model.layers[2].register_forward_hook(hook)
    try:
        ids = torch.tensor(IDS)
        loss = model(input_ids=ids, labels=ids).loss
        model.zero_grad(set_to_none=True)
        loss.backward()
        return float(loss), {n: (p.grad.norm().item() if p.grad is not None else None)
                             for n, p in model.named_parameters()}
    finally:
        handle.remove()


def test_an_in_place_add_in_a_forward_hook_does_not_break_the_backward():
    """It does not. The gradients match the out-of-place form exactly."""
    delta = torch.randn(16) * 0.1

    def inplace(_m, _a, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, 0:, :].add_(delta.to(h.dtype))
        return out

    def outofplace(_m, _a, out):
        h = out[0] if isinstance(out, tuple) else out
        h2 = h + delta.to(h.dtype)
        return (h2, *out[1:]) if isinstance(out, tuple) else h2

    loss_a, grads_a = _loss_and_grads(_tiny(), inplace)
    loss_b, grads_b = _loss_and_grads(_tiny(), outofplace)
    assert loss_a == pytest.approx(loss_b, abs=1e-9)
    for name, value in grads_b.items():
        assert grads_a[name] == pytest.approx(value, abs=1e-6), name


@pytest.mark.parametrize("reentrant, fires", [(True, 2), (False, 1)])
def test_reentrant_checkpointing_runs_the_hook_twice(reentrant, fires):
    """The one that matters.

    Under reentrant checkpointing the block is recomputed during backward and every forward hook
    fires again. A hook carrying a position counter -- which `Injection` does, as `_seen` -- then
    believes the recomputation starts where the forward ended, so the delta lands somewhere else
    during the pass that produces the gradients. The loss looks fine and the model trains against
    an intervention that never happened.
    """
    model = _tiny()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    model.config.use_cache = False
    assert model.model.gradient_checkpointing is True, "checkpointing did not switch on"

    seen = []

    def hook(_m, _a, out):
        h = out[0] if isinstance(out, tuple) else out
        seen.append(h.shape[1])
        return out

    handle = model.model.layers[2].register_forward_hook(hook)
    ids = torch.tensor(IDS)
    loss = model(input_ids=ids, labels=ids).loss
    assert len(seen) == 1, "the forward itself should fire the hook once"
    loss.backward()
    handle.remove()
    assert len(seen) == fires


def test_a_stateless_hook_is_correct_under_either_checkpointing_mode():
    """The fix: in training every forward sees the whole sequence, so position needs no counter."""
    delta = torch.randn(16) * 0.1
    from_position = 3

    def stateless(_m, _a, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, from_position:, :].add_(delta.to(h.dtype))
        return out

    losses = []
    for reentrant in (True, False):
        model = _tiny()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": reentrant})
        model.config.use_cache = False
        losses.append(_loss_and_grads(model, stateless)[0])
    plain = _loss_and_grads(_tiny(), stateless)[0]
    assert losses[0] == pytest.approx(plain, abs=1e-9)
    assert losses[1] == pytest.approx(plain, abs=1e-9)
