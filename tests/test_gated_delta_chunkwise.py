"""R32 stage 2: the chunkwise-parallel gated delta rule must be the reference loop, to tolerance.

Stage 1 (``test_gated_delta_chunked.py``) runs the library's own step function in the library's
own order, so it is bit-exact. Stage 2 replaces the recurrence with matrix products inside each
chunk and a state hand-off between chunks: exact in exact arithmetic, so what is testable is
agreement to a tolerance against ``gated_delta_ops`` — outputs, final state and gradients.

R31: every test here drives the library's own functions — ``gated_delta_ops`` and the real
``GatedDeltaNet`` module — on random tensors and randomly initialised layers. Nothing loads a
checkpoint and nothing runs a training stage.
"""

from __future__ import annotations

import contextlib
import importlib

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.gated_delta import gated_delta_ops
from mlx_lm.models.qwen3_5 import GatedDeltaNet, Model, ModelArgs, TextModelArgs

from local_llm_lab.training import gated_delta_chunked as gdc
from local_llm_lab.training import gated_delta_chunkwise as gdw

# Small enough that a 257-token row is cheap, and Hv > Hk so the reference's repeat of the
# key-side tensors is exercised by default.
_BATCH, _HEADS_K, _HEADS_V, _DIM_K, _DIM_V = 2, 1, 2, 8, 6

# The tolerance R32 stage 2 sets for the float32 forward and final state. The chunkwise form
# reassociates the same sums, so the two differ by float32 rounding only.
_RTOL, _ATOL = 1e-4, 1e-5


def _random_inputs(
    tokens: int,
    *,
    seed: int,
    vectorised: bool = False,
    masked: bool = False,
    heads_k: int = _HEADS_K,
    heads_v: int = _HEADS_V,
    normalise_keys: bool = True,
) -> dict[str, mx.array | None]:
    """Random q/k/v/g/beta (and mask) in the shapes ``gated_delta_ops`` documents.

    ``normalise_keys`` reproduces what the model hands the recurrence: ``GatedDeltaNet``
    rms-norms k and scales it by ``Dk ** -0.5`` (``mlx_lm/models/qwen3_5.py:180``), so every
    key reaching the recurrence has unit norm and ``I - beta k kT`` is a contraction. The
    within-chunk triangular system is then well conditioned, which is the regime the tolerance
    is quoted for; ``normalise_keys=False`` is the deliberately harsher control below.
    """
    mx.random.seed(seed)
    gate_shape = (
        (_BATCH, tokens, heads_v, _DIM_K) if vectorised else (_BATCH, tokens, heads_v)
    )
    key = mx.random.normal((_BATCH, tokens, heads_k, _DIM_K))
    if normalise_keys:
        key = key / mx.sqrt((key * key).sum(axis=-1, keepdims=True))
    inputs = {
        "q": mx.random.normal((_BATCH, tokens, heads_k, _DIM_K)),
        "k": key,
        "v": mx.random.normal((_BATCH, tokens, heads_v, _DIM_V)),
        # Gating is a decay in (0, 1), as ``compute_g`` produces it.
        "g": mx.random.uniform(shape=gate_shape),
        "beta": mx.random.uniform(shape=(_BATCH, tokens, heads_v)),
        "mask": (
            mx.random.uniform(shape=(_BATCH, tokens)) > 0.25 if masked else None
        ),
    }
    # Materialise before use: both implementations then read one fixed set of numbers.
    mx.eval([value for value in inputs.values() if value is not None])
    return inputs


def _worst(actual: mx.array, expected: mx.array) -> tuple[float, float]:
    """Largest absolute error, and largest error relative to the reference's own scale."""
    difference = mx.abs(actual - expected)
    scale = mx.maximum(mx.abs(expected).max(), mx.array(1.0, dtype=mx.float32))
    return float(difference.max()), float(difference.max() / scale)


def _close(actual: mx.array, expected: mx.array, *, rtol: float = _RTOL, atol: float = _ATOL):
    return mx.allclose(actual, expected, rtol=rtol, atol=atol).item()


# ------------------------------------------------------------------- exactness to a tolerance


@pytest.mark.parametrize("tokens", [1, 5, 64, 65, 130, 257])
@pytest.mark.parametrize("chunk", [16, 64, 128])
@pytest.mark.parametrize("carried", [False, True])
def test_chunkwise_forward_and_final_state_match_the_reference(
    tokens: int, chunk: int, carried: bool
) -> None:
    """The stage 2 claim: the same recurrence, computed with matrix products (R32 stage 2)."""
    inputs = _random_inputs(tokens, seed=11)
    start = (
        mx.random.normal((_BATCH, _HEADS_V, _DIM_V, _DIM_K)).astype(mx.float32)
        if carried
        else None
    )
    reference, reference_state = gated_delta_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], start, None
    )
    actual, actual_state = gdw.gated_delta_chunkwise_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], start, None,
        chunk=chunk,
    )

    assert actual.shape == reference.shape
    assert actual_state.dtype == mx.float32
    assert _close(actual, reference), _worst(actual, reference)
    assert _close(actual_state, reference_state), _worst(actual_state, reference_state)
    assert gdw.fallback_counts() == {}


@pytest.mark.parametrize(("heads_k", "heads_v"), [(1, 1), (2, 2), (1, 4), (2, 4)])
def test_the_key_side_repeat_matches_the_reference_for_every_head_ratio(
    heads_k: int, heads_v: int
) -> None:
    """Hv > Hk repeats q and k before the recurrence; a repeat done differently would drift."""
    inputs = _random_inputs(130, seed=3, heads_k=heads_k, heads_v=heads_v)
    reference, reference_state = gated_delta_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, None
    )
    actual, actual_state = gdw.gated_delta_chunkwise_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, None, chunk=64
    )

    assert _close(actual, reference), _worst(actual, reference)
    assert _close(actual_state, reference_state), _worst(actual_state, reference_state)


def test_bfloat16_inputs_match_the_reference_within_a_looser_tolerance() -> None:
    """Inputs arrive in the model's dtype; the chunk math is float32 and y comes back in q's.

    The reference is itself float32 inside the step (the state is float32, so every product
    with it promotes), and returns ``y.astype(q.dtype)``: rounding y to bfloat16 costs about
    2^-8 relative, which is the tolerance quoted here and the reason it is not 1e-4.
    """
    inputs = _random_inputs(130, seed=23)
    cast = {
        name: (value if value is None else value.astype(mx.bfloat16))
        for name, value in inputs.items()
    }
    reference, reference_state = gated_delta_ops(
        cast["q"], cast["k"], cast["v"], cast["g"], cast["beta"], None, None
    )
    actual, actual_state = gdw.gated_delta_chunkwise_ops(
        cast["q"], cast["k"], cast["v"], cast["g"], cast["beta"], None, None, chunk=64
    )

    assert actual.dtype == mx.bfloat16 == reference.dtype
    assert actual_state.dtype == mx.float32
    assert _close(actual, reference, rtol=8e-3, atol=8e-3), _worst(actual, reference)
    assert _close(actual_state, reference_state, rtol=8e-3, atol=8e-3), _worst(
        actual_state, reference_state
    )


def test_un_normalised_keys_still_agree_at_a_stated_looser_tolerance() -> None:
    """The control for the fixture's key normalisation.

    With raw normal keys ``I - beta k kT`` is not a contraction and the within-chunk
    triangular system is worse conditioned than anything the model produces, so the
    chunkwise form's reassociation costs more digits. It is recorded here rather than
    hidden: if this ever fails the algorithm is wrong, not merely reassociated.
    """
    inputs = _random_inputs(130, seed=29, normalise_keys=False)
    reference, reference_state = gated_delta_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, None
    )
    actual, actual_state = gdw.gated_delta_chunkwise_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, None, chunk=64
    )

    assert _close(actual, reference, rtol=1e-2, atol=1e-3), _worst(actual, reference)
    assert _close(actual_state, reference_state, rtol=1e-2, atol=1e-3), _worst(
        actual_state, reference_state
    )


def test_the_same_inputs_give_identical_outputs_twice() -> None:
    """Determinism: two evaluations of the same graph on the same numbers must agree bitwise."""
    inputs = _random_inputs(257, seed=31)
    arguments = (inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"])
    first, first_state = gdw.gated_delta_chunkwise_ops(*arguments, None, None, chunk=64)
    mx.eval(first, first_state)
    second, second_state = gdw.gated_delta_chunkwise_ops(*arguments, None, None, chunk=64)
    mx.eval(second, second_state)

    assert mx.array_equal(first, second).item()
    assert mx.array_equal(first_state, second_state).item()


# ------------------------------------------------------------------------------ the fallbacks


@pytest.mark.parametrize("chunk", [16, 64])
def test_vectorised_gating_falls_back_to_the_checkpointed_recurrence(chunk: int) -> None:
    """Qwen3.5 gates per head, not per key dimension; the vectorised shape takes stage 1."""
    inputs = _random_inputs(65, seed=37, vectorised=True)
    gdw.reset_fallbacks()
    actual, actual_state = gdw.gated_delta_chunkwise_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, None,
        chunk=chunk,
    )
    expected, expected_state = gdc.gated_delta_chunked_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, None,
        chunk=chunk,
    )

    assert gdw.fallback_counts() == {gdw.FALLBACK_VECTORISED_GATING: 1}
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(actual_state, expected_state).item()


def test_a_mask_falls_back_to_the_checkpointed_recurrence() -> None:
    """A masked step freezes the state but still reads an updated one; stage 1 keeps that."""
    inputs = _random_inputs(65, seed=41, masked=True)
    gdw.reset_fallbacks()
    actual, actual_state = gdw.gated_delta_chunkwise_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None,
        inputs["mask"], chunk=64,
    )
    expected, expected_state = gdc.gated_delta_chunked_ops(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None,
        inputs["mask"], chunk=64,
    )

    assert gdw.fallback_counts() == {gdw.FALLBACK_MASK: 1}
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(actual_state, expected_state).item()


def test_fallback_counts_accumulate_until_they_are_reset() -> None:
    """The installer reads this to report that a run did not take the chunkwise path."""
    inputs = _random_inputs(9, seed=43, masked=True)
    gdw.reset_fallbacks()
    for _ in range(2):
        gdw.gated_delta_chunkwise_ops(
            inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None,
            inputs["mask"], chunk=16,
        )
    assert gdw.fallback_counts() == {gdw.FALLBACK_MASK: 2}
    gdw.reset_fallbacks()
    assert gdw.fallback_counts() == {}


@pytest.mark.parametrize("chunk", [0, -4, 1, 7, 65])
def test_a_chunk_that_is_not_a_positive_even_token_count_is_rejected(chunk: int) -> None:
    """An odd chunk is refused, not silently used: see the module docstring's MLX defect."""
    inputs = _random_inputs(9, seed=47)
    with pytest.raises(ValueError, match="positive even"):
        gdw.gated_delta_chunkwise_ops(
            inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], chunk=chunk
        )


def test_a_row_shorter_than_a_chunk_is_padded_to_the_full_even_chunk() -> None:
    """The tail is padded so every chunk the matmuls see is exactly ``chunk`` tokens long.

    Padding with a unit gate, a zero key/value and a zero beta is an identity step, so the
    padded columns change neither the outputs nor the final state; the guard is that a row
    whose length is not a multiple of the chunk still matches the reference exactly.
    """
    for tokens in (1, 9, 65, 129):
        inputs = _random_inputs(tokens, seed=53)
        reference, reference_state = gated_delta_ops(
            inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, None
        )
        actual, actual_state = gdw.gated_delta_chunkwise_ops(
            inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, None,
            chunk=64,
        )
        assert actual.shape[1] == tokens
        assert _close(actual, reference), (tokens, _worst(actual, reference))
        assert _close(actual_state, reference_state), (
            tokens,
            _worst(actual_state, reference_state),
        )


# --------------------------------------------------------------------------------- gradients


def _scalar_loss(function):
    """A scalar of both outputs, so the gradient covers the outputs and the final state."""

    def loss(q, k, v, g, beta):
        y, state = function(q, k, v, g, beta)
        return y.sum() + state.sum()

    return loss


@pytest.mark.parametrize("chunk", [16, 64])
def test_gradients_match_the_reference_loop_within_tolerance(chunk: int) -> None:
    """Backward is the point of the slice, so it is compared directly, not inferred."""
    inputs = _random_inputs(65, seed=59)
    arguments = (inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"])
    reference = mx.grad(
        _scalar_loss(lambda *a: gated_delta_ops(*a)), argnums=(0, 1, 2, 3, 4)
    )(*arguments)
    actual = mx.grad(
        _scalar_loss(lambda *a: gdw.gated_delta_chunkwise_ops(*a, chunk=chunk)),
        argnums=(0, 1, 2, 3, 4),
    )(*arguments)

    for expected, got in zip(reference, actual):
        assert expected.shape == got.shape
        assert _close(got, expected, rtol=1e-3, atol=1e-4), _worst(got, expected)


# ------------------------------------------------------ the installer, on the real module


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


def test_qwen35_gates_the_recurrence_per_head_so_the_chunkwise_path_is_the_one_taken() -> None:
    """The shape claim the implementation rests on, read off the real module, not assumed.

    ``compute_g(A_log, a, dt_bias)`` (``mlx_lm/models/gated_delta.py:9-10``) broadcasts
    ``A_log`` and ``dt_bias``, both ``(num_v_heads,)``, against ``a = in_proj_a(inputs)``,
    which is ``[B, S, num_v_heads]`` (``qwen3_5.py:119``). So ``g`` is three-dimensional —
    scalar gating — and never the ``[B, S, Hv, Dk]`` vectorised shape.
    """
    net, inputs = _small_gated_delta_net(seed=61)
    seen: list[tuple[int, ...]] = []
    library = importlib.import_module("mlx_lm.models.gated_delta")
    original = library.gated_delta_ops

    def spy(q, k, v, g, beta, state=None, mask=None):
        seen.append(g.shape)
        return original(q, k, v, g, beta, state, mask)

    library.gated_delta_ops = spy
    try:
        mx.eval(net(inputs))
    finally:
        library.gated_delta_ops = original

    assert seen and all(len(shape) == 3 for shape in seen), seen
    assert all(shape[-1] == net.num_v_heads for shape in seen), seen


def test_the_real_gated_delta_net_takes_the_chunkwise_path_and_agrees_with_the_library() -> None:
    """R31: the installer is judged by driving the library's own module in training mode."""
    net, inputs = _small_gated_delta_net(seed=67)
    assert net.training is True
    baseline = net(inputs)
    mx.eval(baseline)

    gdw.reset_fallbacks()
    with gdw.install_chunkwise_gated_delta(16):
        installed = net(inputs)
        mx.eval(installed)

    assert gdw.fallback_counts() == {}
    assert _close(installed, baseline, rtol=1e-3, atol=1e-4), _worst(installed, baseline)


def test_the_real_gated_delta_net_backward_agrees_through_the_installer() -> None:
    """The path the trainer takes: a gradient through the real module, chunkwise and not."""
    net, inputs = _small_gated_delta_net(seed=71)
    loss_and_grad = nn.value_and_grad(net, lambda model, x: model(x).sum())

    baseline_loss, baseline_grads = loss_and_grad(net, inputs)
    mx.eval(baseline_loss, baseline_grads)
    with gdw.install_chunkwise_gated_delta(16):
        actual_loss, actual_grads = loss_and_grad(net, inputs)
        mx.eval(actual_loss, actual_grads)

    assert _close(actual_loss, baseline_loss, rtol=1e-3, atol=1e-4)
    expected = dict(tree_flatten(baseline_grads))
    actual = dict(tree_flatten(actual_grads))
    assert actual.keys() == expected.keys() and actual
    for name, value in actual.items():
        assert _close(value, expected[name], rtol=1e-2, atol=1e-3), (
            name,
            _worst(value, expected[name]),
        )


# ------------------------------------------------- the model's own forward, not just the layer

# A hand-made Qwen3.5 text config, small enough that a whole model is cheap to build and run.
# ``full_attention_interval`` with four layers gives the real interleave — three linear layers
# and one full-attention layer — so both branches of ``DecoderLayer`` are exercised, and
# ``Qwen3_5TextModel`` indexes ``cache[self.fa_idx]`` at ``full_attention_interval - 1``, which
# needs at least that many layers. R31: no checkpoint is loaded; the weights are random.
_TINY_TEXT_CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "vocab_size": 32,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
}


def _tiny_model(seed: int) -> tuple[Model, mx.array]:
    """A real ``qwen3_5.Model`` on random weights, in training mode, and one short input row."""
    mx.random.seed(seed)
    model = Model(ModelArgs(model_type="qwen3_5", text_config=dict(_TINY_TEXT_CONFIG)))
    model.train()
    return model, mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])


def test_the_model_forward_in_training_mode_never_masks_the_linear_layers() -> None:
    """K6(c): the mask fallback is dormant in training, proved through the model's own forward.

    Every other test here calls ``GatedDeltaNet`` directly and passes ``mask=None`` itself, so
    none of them can see what the *model* hands its linear layers. The mechanism, read off the
    library: ``create_ssm_mask`` (``mlx_lm/models/base.py:58-61``) returns ``None`` unless the
    cache has a ``make_mask``; ``Qwen3_5TextModel.__call__`` calls it as
    ``create_ssm_mask(hidden_states, cache[self.ssm_idx])``, which is safe on a bare ``None``
    only because the two lines above turn ``cache=None`` — what training passes — into
    ``[None] * len(self.layers)``. So in training the linear layers receive ``mask=None`` and
    the chunkwise form is what runs.

    The second half is the control: it makes ``create_ssm_mask`` return a real mask and shows
    the same forward then falls back once per linear layer. Without it this test would pass on
    a build where the fallback fires for some other reason and the counter was never wired.
    """
    model, tokens = _tiny_model(seed=73)
    assert model.training is True

    gdw.reset_fallbacks()
    with gdw.install_chunkwise_gated_delta(4):
        mx.eval(model(tokens))
    assert gdw.fallback_counts() == {}

    # The control. ``qwen3_5`` imports ``create_ssm_mask`` by value, so its own module global
    # is the patch point — the same reasoning the installer's docstring uses for
    # ``gated_delta_ops``. One count per linear layer proves the assertion above has teeth.
    library = importlib.import_module("mlx_lm.models.qwen3_5")
    original = library.create_ssm_mask
    linear_layers = sum(1 for layer in model.language_model.layers if layer.is_linear)
    library.create_ssm_mask = lambda h, cache=None: mx.ones(h.shape[:2], dtype=mx.bool_)
    try:
        gdw.reset_fallbacks()
        with gdw.install_chunkwise_gated_delta(4):
            mx.eval(model(tokens))
    finally:
        library.create_ssm_mask = original

    assert linear_layers > 0
    assert gdw.fallback_counts() == {gdw.FALLBACK_MASK: linear_layers}


def test_the_installer_rebinds_only_the_training_path_and_restores_it() -> None:
    """Inference keeps the Metal kernel; the library's own binding comes back on exit."""
    library = importlib.import_module("mlx_lm.models.gated_delta")
    original = library.gated_delta_ops
    original_kernel = library.gated_delta_kernel

    with gdw.install_chunkwise_gated_delta(64) as installed:
        assert library.gated_delta_ops is installed
        assert library.gated_delta_ops is not original
        assert library.gated_delta_kernel is original_kernel

    assert library.gated_delta_ops is original


def test_the_installer_restores_the_binding_when_the_body_raises() -> None:
    library = importlib.import_module("mlx_lm.models.gated_delta")
    original = library.gated_delta_ops

    with pytest.raises(RuntimeError, match="training blew up"):
        with gdw.install_chunkwise_gated_delta(64):
            raise RuntimeError("training blew up")

    assert library.gated_delta_ops is original


# ----------------------------------------------------------------- the mode the CLI selects


def _recording_installer(name: str, events: list[tuple[str, int]]):
    @contextlib.contextmanager
    def installer(chunk: int):
        events.append((name, chunk))
        yield

    return installer


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(None, "checkpointed"), ("checkpointed", "checkpointed"), ("chunkwise", "chunkwise")],
)
def test_the_training_backbone_enters_the_installer_named_by_the_mode(
    monkeypatch, mode: str | None, expected: str
) -> None:
    """R32 stage 2: the arm config chooses the recurrence form, and the default is stage 1."""
    cli = importlib.import_module("local_llm_lab.pipeline.cli")
    training = importlib.import_module("local_llm_lab.training")
    events: list[tuple[str, int]] = []
    monkeypatch.setattr(
        training, "install_chunked_gated_delta", _recording_installer("checkpointed", events)
    )
    monkeypatch.setattr(
        training, "install_chunkwise_gated_delta", _recording_installer("chunkwise", events)
    )

    kwargs = {} if mode is None else {"mode": mode}
    with cli._training_backbone(64, **kwargs):
        pass

    assert events == [(expected, 64)]


def test_the_training_backbone_installs_nothing_for_a_dense_arm(monkeypatch) -> None:
    cli = importlib.import_module("local_llm_lab.pipeline.cli")
    training = importlib.import_module("local_llm_lab.training")
    for name in ("install_chunked_gated_delta", "install_chunkwise_gated_delta"):
        monkeypatch.setattr(
            training,
            name,
            lambda chunk: pytest.fail(f"a dense arm installed a recurrence patch ({chunk})"),
        )

    with cli._training_backbone(None, mode="chunkwise"):
        pass


def test_an_unknown_recurrence_mode_is_rejected() -> None:
    cli = importlib.import_module("local_llm_lab.pipeline.cli")
    with pytest.raises(ValueError, match="gated_delta_mode"):
        with cli._training_backbone(64, mode="quadratic"):
            pass
