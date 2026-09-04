"""R32(a) stage 1: the chunked gated-delta recurrence must be the reference loop, exactly.

R31: every test here drives the library's own functions — ``gated_delta_ops``,
``_gated_delta_step_ops`` and the real ``GatedDeltaNet`` module — on random tensors and
randomly initialised layers. Nothing loads a checkpoint and nothing runs a training stage.
"""

from __future__ import annotations

import importlib

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.gated_delta import gated_delta_ops
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs

from local_llm_lab.training import gated_delta_chunked as gdc

# Shapes small enough that a 130-token row is cheap, and Hv > Hk so the reference's repeat of
# the key-side tensors is exercised by default.
_BATCH, _HEADS_K, _HEADS_V, _DIM_K, _DIM_V = 2, 1, 2, 4, 3


def _random_inputs(
    tokens: int,
    *,
    seed: int,
    vectorised: bool,
    masked: bool,
    heads_k: int = _HEADS_K,
    heads_v: int = _HEADS_V,
) -> dict[str, mx.array | None]:
    """Random q/k/v/g/beta (and mask) in the shapes ``gated_delta_ops`` documents."""
    mx.random.seed(seed)
    gate_shape = (
        (_BATCH, tokens, heads_v, _DIM_K) if vectorised else (_BATCH, tokens, heads_v)
    )
    return {
        "q": mx.random.normal((_BATCH, tokens, heads_k, _DIM_K)),
        "k": mx.random.normal((_BATCH, tokens, heads_k, _DIM_K)),
        "v": mx.random.normal((_BATCH, tokens, heads_v, _DIM_V)),
        # Gating is a decay in (0, 1), as ``compute_g`` produces it.
        "g": mx.random.uniform(shape=gate_shape),
        "beta": mx.random.uniform(shape=(_BATCH, tokens, heads_v)),
        "mask": (
            mx.random.uniform(shape=(_BATCH, tokens)) > 0.25 if masked else None
        ),
    }


@pytest.mark.parametrize("tokens", [1, 5, 64, 65, 130])
@pytest.mark.parametrize("chunk", [1, 7, 64, 128])
@pytest.mark.parametrize("vectorised", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_chunked_forward_and_final_state_are_bit_exact(
    tokens: int, chunk: int, vectorised: bool, masked: bool
) -> None:
    """Same ops in the same order, so nothing may differ by a single bit (R32(a))."""
    inputs = _random_inputs(tokens, seed=11, vectorised=vectorised, masked=masked)
    reference, reference_state = gated_delta_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, inputs["mask"]
    )
    chunked, chunked_state = gdc.gated_delta_chunked_ops(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        None,
        inputs["mask"],
        chunk=chunk,
    )

    assert chunked.shape == reference.shape
    assert mx.array_equal(chunked, reference).item()
    assert mx.array_equal(chunked_state, reference_state).item()


@pytest.mark.parametrize(("heads_k", "heads_v"), [(1, 1), (2, 2), (1, 4), (2, 4)])
def test_the_key_side_repeat_matches_the_reference_for_every_head_ratio(
    heads_k: int, heads_v: int
) -> None:
    """Hv > Hk repeats q and k inside the loop; a repeat done differently would drift."""
    inputs = _random_inputs(
        13, seed=3, vectorised=False, masked=False, heads_k=heads_k, heads_v=heads_v
    )
    reference, reference_state = gated_delta_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, None
    )
    chunked, chunked_state = gdc.gated_delta_chunked_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, None, chunk=5
    )

    assert mx.array_equal(chunked, reference).item()
    assert mx.array_equal(chunked_state, reference_state).item()


def test_a_carried_in_state_is_honoured_and_stays_float32() -> None:
    """The recurrence may start from a non-zero state; R32 keeps that state float32."""
    inputs = _random_inputs(9, seed=5, vectorised=False, masked=False)
    start = mx.random.normal((_BATCH, _HEADS_V, _DIM_V, _DIM_K)).astype(mx.float32)
    reference, reference_state = gated_delta_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], start, None
    )
    chunked, chunked_state = gdc.gated_delta_chunked_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], start, None, chunk=4
    )

    assert mx.array_equal(chunked, reference).item()
    assert mx.array_equal(chunked_state, reference_state).item()
    assert chunked_state.dtype == mx.float32


def test_a_chunk_below_one_token_is_rejected() -> None:
    inputs = _random_inputs(4, seed=5, vectorised=False, masked=False)
    with pytest.raises(ValueError, match="at least one token"):
        gdc.gated_delta_chunked_ops(
            inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], chunk=0
        )


def _scalar_loss(function):
    """A scalar of both outputs, so the gradient covers the outputs and the final state."""

    def loss(q, k, v, g, beta):
        y, state = function(q, k, v, g, beta)
        return y.sum() + state.sum()

    return loss


@pytest.mark.parametrize("chunk", [1, 3, 7])
@pytest.mark.parametrize("vectorised", [False, True])
def test_gradients_match_the_reference_loop(chunk: int, vectorised: bool) -> None:
    """Recomputation replays the identical ops, so the gradients are exact to float32 noise.

    The tolerance is 0 by construction; ``mx.allclose`` is used at 1e-6/1e-6 so that a
    future MLX release reordering the recomputed backward reports a near miss as a near
    miss rather than as a silent bit flip.
    """
    inputs = _random_inputs(11, seed=7, vectorised=vectorised, masked=False)
    arguments = (inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"])
    reference = mx.grad(
        _scalar_loss(lambda *a: gated_delta_ops(*a)), argnums=(0, 1, 2, 3, 4)
    )(*arguments)
    chunked = mx.grad(
        _scalar_loss(lambda *a: gdc.gated_delta_chunked_ops(*a, chunk=chunk)),
        argnums=(0, 1, 2, 3, 4),
    )(*arguments)

    for expected, actual in zip(reference, chunked):
        assert expected.shape == actual.shape
        assert mx.allclose(actual, expected, rtol=1e-6, atol=1e-6).item()


def test_the_chunked_path_steps_once_per_token_and_recomputes_at_most_once(monkeypatch) -> None:
    """The memory claim in structural form: one forward pass of steps, one recompute.

    Peak memory needs a device to observe; what is checkable here is that the checkpointed
    region really is re-entered on the backward pass (so the forward graph did not retain
    it) and that it is entered exactly once more, not once per chunk boundary.
    """
    tokens, chunk = 12, 5
    inputs = _random_inputs(tokens, seed=13, vectorised=False, masked=False)
    calls = []
    step = gdc._gated_delta_step_ops

    def counting_step(*args, **kwargs):
        calls.append(1)
        return step(*args, **kwargs)

    monkeypatch.setattr(gdc, "_gated_delta_step_ops", counting_step)

    y, state = gdc.gated_delta_chunked_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, None, chunk=chunk
    )
    mx.eval(y, state)
    forward = len(calls)

    gradients = mx.grad(
        _scalar_loss(lambda *a: gdc.gated_delta_chunked_ops(*a, chunk=chunk)),
        argnums=(0, 1, 2, 3, 4),
    )(inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"])
    mx.eval(gradients)
    recompute = len(calls) - 2 * forward

    assert forward == tokens
    assert 0 < recompute <= tokens

    # Control: the library's own loop retains the whole graph, so differentiating it enters
    # the step function once per token and never again. The extra pass above is the
    # recomputation, not an artefact of how the counter is placed.
    library = importlib.import_module("mlx_lm.models.gated_delta")
    monkeypatch.setattr(library, "_gated_delta_step_ops", counting_step)
    calls.clear()
    reference_gradients = mx.grad(
        _scalar_loss(lambda *a: gated_delta_ops(*a)), argnums=(0, 1, 2, 3, 4)
    )(inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"])
    mx.eval(reference_gradients)

    assert len(calls) == tokens


# ------------------------------------------------------------------ the installer, on the real module


def _small_gated_delta_net(seed: int) -> tuple[GatedDeltaNet, mx.array]:
    """A real ``GatedDeltaNet`` with randomly initialised weights, and one random input."""
    mx.random.seed(seed)
    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    net = GatedDeltaNet(args)
    net.train()
    return net, mx.random.normal((1, 9, args.hidden_size))


def test_the_real_gated_delta_net_takes_the_chunked_path_and_is_unchanged_by_it(
    monkeypatch,
) -> None:
    """R31: the installer is judged by driving the library's own module in training mode."""
    net, inputs = _small_gated_delta_net(seed=17)
    assert net.training is True
    baseline = net(inputs)
    mx.eval(baseline)

    seen = []
    chunked_ops = gdc.gated_delta_chunked_ops

    def spy(*args, **kwargs):
        seen.append(kwargs.get("chunk"))
        return chunked_ops(*args, **kwargs)

    monkeypatch.setattr(gdc, "gated_delta_chunked_ops", spy)
    with gdc.install_chunked_gated_delta(4):
        installed = net(inputs)
        mx.eval(installed)

    # One gated-delta layer, one forward pass, one call — carrying the configured chunk.
    assert seen == [4]
    assert mx.array_equal(installed, baseline).item()


def test_the_real_gated_delta_net_backward_is_unchanged_by_the_installer() -> None:
    """The path the trainer takes: a gradient through the real module, chunked and not.

    This is the closest a weightless test gets to the training step the lane probe measures —
    randomly initialised library layers, a real backward, and the chunked recurrence swapped
    in underneath. R32(a)'s claim is that only the memory changes.
    """
    net, inputs = _small_gated_delta_net(seed=19)
    loss_and_grad = nn.value_and_grad(net, lambda model, x: model(x).sum())

    baseline_loss, baseline_grads = loss_and_grad(net, inputs)
    mx.eval(baseline_loss, baseline_grads)
    with gdc.install_chunked_gated_delta(4):
        chunked_loss, chunked_grads = loss_and_grad(net, inputs)
        mx.eval(chunked_loss, chunked_grads)

    assert mx.array_equal(chunked_loss, baseline_loss).item()
    expected = dict(tree_flatten(baseline_grads))
    actual = dict(tree_flatten(chunked_grads))
    assert actual.keys() == expected.keys() and actual
    for name, value in actual.items():
        assert mx.allclose(value, expected[name], rtol=1e-6, atol=1e-6).item(), name


def test_the_installer_rebinds_only_the_training_path_and_restores_it() -> None:
    """Inference keeps the Metal kernel; the library's own binding comes back on exit."""
    module = importlib.import_module("mlx_lm.models.gated_delta")
    original = module.gated_delta_ops
    original_kernel = module.gated_delta_kernel

    with gdc.install_chunked_gated_delta(8) as installed:
        assert module.gated_delta_ops is installed
        assert module.gated_delta_ops is not original
        assert module.gated_delta_kernel is original_kernel

    assert module.gated_delta_ops is original


def test_the_installer_restores_the_binding_when_the_body_raises() -> None:
    module = importlib.import_module("mlx_lm.models.gated_delta")
    original = module.gated_delta_ops

    with pytest.raises(RuntimeError, match="training blew up"):
        with gdc.install_chunked_gated_delta(8):
            raise RuntimeError("training blew up")

    assert module.gated_delta_ops is original


# ------------------------------------------------------------------------------ footprint helper


def test_training_state_bytes_counts_one_state_per_retained_step() -> None:
    """Hand-computed: one state is 2 x 3 x 4 x 5 x 4 bytes = 480 bytes."""
    state = 2 * 3 * 4 * 5 * 4
    assert state == 480
    shape = {"batch": 2, "heads_v": 3, "dim_v": 4, "dim_k": 5}

    # Unrolled: every step's state stays in the graph.
    assert gdc.training_state_bytes(**shape, tokens=100, chunk=None) == 100 * state
    # Chunked and divisible: 100 / 10 boundaries + one 10-token recompute window.
    assert gdc.training_state_bytes(**shape, tokens=100, chunk=10) == (10 + 10) * state
    # Chunked and not divisible: ceil(100 / 8) = 13 boundaries + an 8-token window.
    assert gdc.training_state_bytes(**shape, tokens=100, chunk=8) == (13 + 8) * state
    # A chunk longer than the row: one boundary, and the window charged at full length.
    assert gdc.training_state_bytes(**shape, tokens=6, chunk=64) == (1 + 64) * state
    assert gdc.training_state_bytes(**shape, tokens=1, chunk=1) == 2 * state


def test_training_state_bytes_shows_the_chunking_saving_and_rejects_empty_shapes() -> None:
    """The estimate the preflight gate reads must fall by orders of magnitude at C = 64."""
    shape = {"batch": 1, "heads_v": 32, "dim_v": 128, "dim_k": 128, "tokens": 997}
    unrolled = gdc.training_state_bytes(**shape, chunk=None)
    chunked = gdc.training_state_bytes(**shape, chunk=64)

    assert unrolled == 997 * 32 * 128 * 128 * 4
    assert chunked == (16 + 64) * 32 * 128 * 128 * 4
    assert chunked * 12 < unrolled

    for bad in ({"batch": 0}, {"tokens": 0}, {"dim_k": -1}):
        with pytest.raises(ValueError, match="at least one"):
            gdc.training_state_bytes(**{**shape, "chunk": 64, **bad})
    with pytest.raises(ValueError, match="at least one token"):
        gdc.training_state_bytes(**shape, chunk=0)
