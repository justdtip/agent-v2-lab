"""Chunkwise-parallel gated delta rule for training (wiring map R32, stage 2).

Stage 1 (``gated_delta_chunked``) keeps the library's per-token loop and checkpoints it, so it
is bit-exact but still sequential: on the lane, one training step at 997 tokens costs 11.85 GB
and 46 s, against a 5.36 GB / 7.9 s floor with the recurrence's backward removed. The
recurrence's backward is therefore ~6.5 GB and ~85% of the step, and both R32 gates fail.

This module replaces the *differentiated* recurrence with the chunkwise-parallel form every
DeltaNet trainer uses (R32 stage 2): matrix products within a chunk, states passed between
chunks. It is exact in exact arithmetic and tested to a tolerance, not bit for bit.

The algorithm
-------------
``_gated_delta_step_ops`` (``mlx_lm/models/gated_delta.py:127-168``) is, per (batch, head), with
state ``S`` of shape ``[Dv, Dk]``, scalar gate ``g_t`` and scalar ``beta_t``::

    S_t = g_t S_{t-1}                                       (decay first)
    S_t = S_t + beta_t (v_t - S_t k_t) k_t^T                 (then the delta update)
    y_t = S_t q_t                                           (then the read, post-update)

Substituting the second line into itself gives a linear recurrence in ``S``::

    S_t = g_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T

Because the gate is a scalar it commutes, so with ``G_t = prod_{s<=t} g_s`` the normalised state
``S~_t = S_t / G_t`` obeys ``S~_t = S~_{t-1} (I - beta_t k_t k_t^T) + (beta_t / G_t) v_t k_t^T``,
i.e. a rank-one update per token::

    S~_t = S~_{t-1} + u~_t k_t^T / G_t,
    u~_t = beta_t ( v_t - G_t S_in k_t - sum_{s<t} (G_t / G_s) (k_s . k_t) u~_s )

Every coefficient there is a gate *ratio* ``G_t / G_s`` with ``s <= t``, which is at most one:
no ``1 / G`` ever appears, so a chunk of strongly decaying gates cannot overflow. Collecting the
chunk's ``C`` tokens into matrices ``Q, K [C, Dk]``, ``V [C, Dv]`` and the incoming state
``S_in [Dv, Dk]``, and writing ``D_ij = G_i / G_j`` on the causal triangle (zero above it):

* ``A   = tril(beta_i * D * (K K^T), -1)``      the strictly lower "UT transform" matrix
* ``U~  = (I + A)^{-1} ( beta * V - (beta * G) * (K S_in^T) )``   the chunk's pseudo-values
* ``Y   = G * (Q S_in^T) + (D * (Q K^T)) U~``   the read, causal-inclusive so token t sees its
  own update, exactly as the reference reads *after* updating
* ``S_out = G_C S_in + ((G_C / G) * U~)^T K``   the state handed to the next chunk

The two terms of ``U~``'s right-hand side are the WY pair: ``U~ = U_v - W S_in^T`` with
``U_v = (I + A)^{-1} (beta * V)`` and ``W = (I + A)^{-1} (beta * G * K)``. They are formed by
one solve against the combined right-hand side here, since the chunk loop already has ``S_in``.

``(I + A)^{-1}`` is the forward substitution over the ``C x C`` triangle, done blockwise so it
is matmuls rather than ``C`` sequential steps. ``Inv`` starts as the identity — the inverse of
each 1x1 diagonal block of a unit lower triangular matrix — and each level merges neighbouring
blocks of size ``s`` into blocks of size ``2s`` using the exact block formula
``[[L11, 0], [L21, L22]]^{-1} = [[L11^-1, 0], [-L22^-1 L21 L11^-1, L22^-1]]``, which in full
matrices is ``Inv <- Inv - mask_s * (Inv L Inv)``: the ``(2, 1)`` sub-block of ``Inv L Inv`` is
exactly ``Inv22 L21 Inv11`` while ``Inv`` is block diagonal. That is ``ceil(log2 C)`` levels of
two ``C x C`` matmuls, and it is the same arithmetic forward substitution performs, not a
truncated series.

Gates enter in exactly three places, all as ratios or cumulative products of the same
``cumsum(log g)``: ``D`` inside the chunk, ``G`` on the terms that touch the incoming state, and
``G_C / G`` on the terms that leave for the next chunk.

An MLX defect this works around
-------------------------------
On MLX 0.32.2's Metal backend this op graph evaluates *incorrectly* — sporadically per process,
on identical materialised inputs — whenever a chunk's token count is **odd**; the same graph on
``mx.cpu`` is always right, and forcing intermediate evaluation makes the error go away, so it
is an evaluation defect and not this algorithm. Measured over eight processes on the test
shapes: chunk lengths {7, 33, 65} failed 6/8 and produced errors up to 2.7e+01 against the
reference, while {2, 4, 6, 8, 16, 34, 48, 64, 66, 128} failed 0/8. Two consequences, both
enforced below: ``chunk`` must be a positive even number, and a row that is not a multiple of
``chunk`` is **padded** to one so that every chunk the matmuls see is exactly ``chunk`` tokens.
The padding is an identity step (unit gate, zero key, value and beta), so it changes neither the
outputs nor the final state; the padded outputs are sliced off.

Scope and fallbacks
-------------------
Qwen3.5 gates the recurrence **per head**: ``compute_g`` broadcasts ``A_log`` and ``dt_bias``,
both ``(num_v_heads,)``, against ``a = in_proj_a(inputs)`` of shape ``[B, S, num_v_heads]``
(``mlx_lm/models/qwen3_5.py:119-123``), so ``g`` is ``[B, T, Hv]``. That is the shape this
module implements. Two inputs fall back to stage 1's checkpointed loop, and the fallback is
counted so the installer can report it:

* **vectorised gating** ``g: [B, T, Hv, Dk]`` — the gate is then a diagonal matrix on the key
  axis and no longer commutes with the rank-one updates, so the derivation above does not hold.
  Qwen3.5 never produces it.
* **a mask** — ``_gated_delta_step_ops`` computes ``y`` from the *updated* state and then
  reverts the state for a masked token (``gated_delta.py:163-167``), so a masked token's output
  and its state contribution disagree; that is not a linear recurrence and the chunkwise form
  cannot express it. mlx-lm's own trainer passes no mask into the recurrence.

The state stays float32 throughout and the chunk math is float32 (R32 efficiency note 4); a
bfloat16 input is cast once on entry and ``y`` is cast back to ``q``'s dtype, as the reference
does. The reference is itself float32 inside the step, because its state is.

No footprint estimate lives here
-------------------------------
This module carried an analytic per-layer state-bytes model until R37; it is superseded by the
fitted empirical envelope in ``pipeline/preflight.py``, which is what gates a training arm
(wiring map §8). Do not re-add one. The estimator it replaced predicted a 2.12x spread of
retained state across chunk lengths where the measurement is 1.07x, and an analytic memory
model that nothing checks against measurements is the defect this work repaired.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable, Iterator
from typing import Any, Optional

import functools

import mlx.core as mx

from local_llm_lab.training.gated_delta_chunked import gated_delta_chunked_ops

__all__ = [
    "FALLBACK_MASK",
    "FALLBACK_VECTORISED_GATING",
    "fallback_counts",
    "gated_delta_chunkwise_ops",
    "install_chunkwise_gated_delta",
    "reset_fallbacks",
]

# Reasons a call took stage 1's loop instead of the chunkwise form; see the module docstring.
FALLBACK_MASK = "mask"
FALLBACK_VECTORISED_GATING = "vectorised_gating"

_FALLBACKS: dict[str, int] = {}


def fallback_counts() -> dict[str, int]:
    """How many calls have taken stage 1's loop, by reason, since the last reset."""
    return dict(_FALLBACKS)


def reset_fallbacks() -> None:
    """Forget the recorded fallbacks; the installer calls this on entry."""
    _FALLBACKS.clear()


def _validated_chunk(chunk: int) -> int:
    """Reject a chunk length MLX's Metal backend evaluates incorrectly (module docstring)."""
    if chunk < 2 or chunk % 2:
        raise ValueError(
            "gated-delta chunkwise chunk must be a positive even number of tokens "
            f"(MLX 0.32.2 evaluates an odd chunk incorrectly on Metal); got {chunk}"
        )
    return chunk


@functools.lru_cache(maxsize=None)
def _causal_masks(chunk: int) -> tuple[mx.array, mx.array]:
    """``i >= j`` and ``i > j`` over a chunk, as constants of the chunk length alone.

    Cached and materialised once per chunk length: these are constants, and rebuilding them
    for every chunk of every layer of every micro-step allocated thousands of small Metal
    buffers per step. On 2026-09-06 the unified run died at iteration 51 with
    ``[metal::malloc] Resource limit (499000) exceeded``, Metal's cap on the number of live
    buffers rather than on bytes, raised inside ``_doubling_masks`` on a long row.
    """
    index = mx.arange(chunk)
    masks = (index[:, None] >= index[None, :], index[:, None] > index[None, :])
    mx.eval(*masks)
    return masks


@functools.lru_cache(maxsize=None)
def _identity(chunk: int, dtype: mx.Dtype) -> mx.array:
    """``mx.eye(chunk, dtype)`` as a cached constant (same reason as ``_causal_masks``)."""
    identity = mx.eye(chunk, dtype=dtype)
    mx.eval(identity)
    return identity


@functools.lru_cache(maxsize=None)
def _doubling_masks(chunk: int) -> tuple[mx.array, ...]:
    """One mask per level of the blockwise forward substitution (module docstring).

    Level ``s`` selects, inside each block of ``2s`` rows, the lower-left ``s x s`` sub-block:
    the only part of ``Inv L Inv`` the merge of two ``s``-blocks writes. A chunk length that is
    not a power of two leaves a short final block, which the same expression handles because it
    merges whatever the last block contains with the block before it.
    """
    index = mx.arange(chunk)
    masks = []
    size = 1
    while size < chunk:
        pair = 2 * size
        masks.append(
            (index[:, None] // pair == index[None, :] // pair)
            & (index[:, None] // size == index[None, :] // size + 1)
        )
        size = pair
    mx.eval(*masks)
    return tuple(masks)


def _unit_lower_inverse(strict_lower: mx.array, chunk: int) -> mx.array:
    """Invert ``I + strict_lower`` by blockwise forward substitution; ``ceil(log2 C)`` levels."""
    identity = _identity(chunk, strict_lower.dtype)
    lower = strict_lower + identity
    inverse = mx.broadcast_to(identity, strict_lower.shape)
    for mask in _doubling_masks(chunk):
        inverse = inverse - mx.where(mask, inverse @ lower @ inverse, 0.0)
    return inverse


def _chunk_outputs(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """One chunk of the chunkwise gated delta rule; see the module docstring for the algebra.

    Shapes: q, k ``[B, C, Hv, Dk]`` (already repeated to the value heads), v ``[B, C, Hv, Dv]``,
    g, beta ``[B, C, Hv]``, state ``[B, Hv, Dv, Dk]``. Returns ``y [B, C, Hv, Dv]`` and the
    state after the chunk. Every differentiated tensor is an explicit argument, because
    ``mx.checkpoint`` recomputes with respect to the callable's inputs.
    """
    chunk = q.shape[1]
    inclusive, strict = _causal_masks(chunk)

    # [B, Hv, C, D]: the head axis becomes a batch axis for every matrix product below.
    queries = q.transpose(0, 2, 1, 3)
    keys = k.transpose(0, 2, 1, 3)
    values = v.transpose(0, 2, 1, 3)
    gates = g.transpose(0, 2, 1)
    betas = beta.transpose(0, 2, 1)[..., None]

    # Cumulative log gate. The floor keeps a gate that underflowed to zero from turning into
    # -inf and then into a nan on the diagonal, where the difference of two logs is taken.
    smallest = mx.finfo(mx.float32).smallest_normal
    cumulative = mx.cumsum(mx.log(mx.maximum(gates, smallest)), axis=-1)
    exponents = cumulative[..., :, None] - cumulative[..., None, :]
    # D_ij = G_i / G_j on the causal triangle (<= 1 there), exactly zero above it.
    decay = mx.exp(mx.where(inclusive, exponents, -float("inf")))
    entering = mx.exp(cumulative)[..., None]  # G_i, on the terms that read the incoming state
    leaving = mx.exp(cumulative[..., -1:] - cumulative)[..., None]  # G_C / G_i, <= 1

    ut_transform = betas * mx.where(strict, decay * (keys @ keys.swapaxes(-1, -2)), 0.0)
    inverse = _unit_lower_inverse(ut_transform, chunk)

    transposed_state = state.swapaxes(-1, -2)  # [B, Hv, Dk, Dv]
    pseudo_values = inverse @ (
        betas * (values - entering * (keys @ transposed_state))
    )
    y = entering * (queries @ transposed_state) + (
        decay * (queries @ keys.swapaxes(-1, -2))
    ) @ pseudo_values
    final = mx.exp(cumulative[..., -1])[..., None, None] * state + (
        pseudo_values * leaving
    ).swapaxes(-1, -2) @ keys
    return y.transpose(0, 2, 1, 3), final


_checkpointed_chunk = mx.checkpoint(_chunk_outputs)


def _pad_to_chunk(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    *,
    pad: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Extend the row with identity steps: unit gate, zero key, value, query and beta.

    A step with ``beta = 0`` leaves ``S_t = g_t S_{t-1}``, and ``g_t = 1`` makes that the
    identity, so the padded tokens change neither the final state nor any real token's output;
    their own outputs are zero and are sliced off by the caller.
    """
    batch, _, heads_v, dim_k = q.shape
    dim_v = v.shape[-1]

    def zeros(*shape: int, like: mx.array) -> mx.array:
        return mx.zeros(shape, dtype=like.dtype)

    return (
        mx.concatenate([q, zeros(batch, pad, heads_v, dim_k, like=q)], axis=1),
        mx.concatenate([k, zeros(batch, pad, heads_v, dim_k, like=k)], axis=1),
        mx.concatenate([v, zeros(batch, pad, heads_v, dim_v, like=v)], axis=1),
        mx.concatenate([g, mx.ones((batch, pad, heads_v), dtype=g.dtype)], axis=1),
        mx.concatenate([beta, zeros(batch, pad, heads_v, like=beta)], axis=1),
    )


def gated_delta_chunkwise_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    *,
    chunk: int,
) -> tuple[mx.array, mx.array]:
    """``mlx_lm.models.gated_delta.gated_delta_ops`` in its chunkwise-parallel form.

    Same signature, same shapes and the same recurrence as the reference loop:

      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar gating; the vectorised shape falls back to stage 1)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk], float32, zeros when omitted
      - mask: [B, T] or None (a mask falls back to stage 1)
    Returns ``(y, state)`` with y: [B, T, Hv, Dv] in ``q``'s dtype and the final state in
    float32.

    ``chunk`` is the number of tokens whose recurrence is solved with matrix products; it must
    be a positive even number and the row is padded up to a multiple of it. Agreement with the
    reference is to a tolerance, not bit for bit: the same sums are taken in a different order.
    """
    _validated_chunk(chunk)
    if mask is not None:
        _FALLBACKS[FALLBACK_MASK] = _FALLBACKS.get(FALLBACK_MASK, 0) + 1
        return gated_delta_chunked_ops(q, k, v, g, beta, state, mask, chunk=chunk)
    if g.ndim == 4:
        _FALLBACKS[FALLBACK_VECTORISED_GATING] = (
            _FALLBACKS.get(FALLBACK_VECTORISED_GATING, 0) + 1
        )
        return gated_delta_chunked_ops(q, k, v, g, beta, state, mask, chunk=chunk)

    batch, tokens, heads_k, dim_k = q.shape
    heads_v, dim_v = v.shape[-2:]
    if state is None:
        state = mx.zeros((batch, heads_v, dim_v, dim_k), dtype=mx.float32)
    else:
        state = state.astype(mx.float32)

    # The reference repeats the key-side tensors before the loop when there are more value
    # heads than key heads (``gated_delta.py:242-244``); doing it here keeps the repeat inside
    # the differentiated region, so gradients reach the un-repeated q and k the same way.
    if (repeat_factor := heads_v // heads_k) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    out_dtype = q.dtype
    q, k, v, g, beta = (t.astype(mx.float32) for t in (q, k, v, g, beta))
    if (pad := -tokens % chunk):
        q, k, v, g, beta = _pad_to_chunk(q, k, v, g, beta, pad=pad)

    ys = []
    for start in range(0, tokens + pad, chunk):
        window = slice(start, start + chunk)
        y, state = _checkpointed_chunk(
            q[:, window], k[:, window], v[:, window], g[:, window], beta[:, window], state
        )
        ys.append(y)
    y = ys[0] if len(ys) == 1 else mx.concatenate(ys, axis=1)
    return y[:, :tokens].astype(out_dtype), state


@contextlib.contextmanager
def install_chunkwise_gated_delta(chunk: int) -> Iterator[Callable[..., tuple[Any, Any]]]:
    """Make training-mode gated-delta calls take the chunkwise form, for this block only.

    The patch point is stage 1's: ``mlx_lm.models.gated_delta.gated_delta_ops``. ``qwen3_5``
    imports ``gated_delta_update`` by value (``qwen3_5.py:16``), but ``gated_delta_update``
    resolves ``gated_delta_ops`` from its own module's globals when it runs
    (``gated_delta.py:282``), and only on the ``use_kernel=False`` branch — the branch a model
    in training mode takes. So rebinding that one name reaches every training-mode call without
    touching the inference path, which returns from ``gated_delta_kernel`` on the next line.

    Fallback counts are reset on entry, so ``fallback_counts()`` after the block reports what
    this training run actually took; a non-empty count means some calls ran stage 1's loop and
    the step time and memory of those calls are stage 1's.

    Yields the installed callable and restores the library's own on exit, exception included.
    """
    _validated_chunk(chunk)
    module = importlib.import_module("mlx_lm.models.gated_delta")
    original = module.gated_delta_ops
    reset_fallbacks()

    def chunkwise(q, k, v, g, beta, state=None, mask=None):
        # Resolved as a module global on each call, so a test can substitute a spy.
        return gated_delta_chunkwise_ops(q, k, v, g, beta, state, mask, chunk=chunk)

    module.gated_delta_ops = chunkwise
    try:
        yield chunkwise
    finally:
        module.gated_delta_ops = original
