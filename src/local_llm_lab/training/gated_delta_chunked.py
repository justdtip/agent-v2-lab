"""Chunked, checkpointed gated-delta recurrence for training (wiring map R32(a), stage 1).

The pinned ``mlx_lm`` routes a training-mode ``GatedDeltaNet`` to the reference loop:
``qwen3_5.GatedDeltaNet.__call__`` calls ``gated_delta_update(..., use_kernel=not
self.training)`` (``mlx_lm/models/qwen3_5.py:183-193``), and with ``use_kernel=False`` that
dispatches to ``gated_delta_ops`` (``mlx_lm/models/gated_delta.py:282``), a Python
``for t in range(T)`` over ``_gated_delta_step_ops`` (``gated_delta.py:247-257``). Every step
produces a new ``(B, Hv, Dv, Dk)`` float32 state, and the autograd graph retains all of them,
so training memory is linear in the row's token count per layer. The Metal kernel
(``gated_delta.py:171``) has no VJP, so training cannot take it.

``gated_delta_chunked_ops`` runs the same steps in the same order, in chunks of ``chunk``
tokens, with each chunk's step loop wrapped in ``mx.checkpoint``. The graph then retains only
the chunk-boundary states and recomputes a chunk's steps during the backward pass. Because
the ops and their order are unchanged, the forward outputs and the final state are bit-exact
against ``gated_delta_ops``; only the retained-state footprint changes.

The state stays float32 (R32 efficiency note 4: a narrower state would silently change
numerics and is not permitted).
"""

from __future__ import annotations

import contextlib
import importlib
import math
from collections.abc import Callable, Iterator
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.gated_delta import _gated_delta_step_ops

__all__ = [
    "gated_delta_chunked_ops",
    "install_chunked_gated_delta",
    "training_state_bytes",
]

# The recurrence state is float32 (``gated_delta_ops`` allocates it as such at
# ``gated_delta.py:240``), so one element costs four bytes.
_STATE_BYTES_PER_ELEMENT = 4


def _chunk_steps_masked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array,
) -> tuple[mx.array, mx.array]:
    """One chunk of the reference step loop, masked variant (see ``_chunk_steps``)."""
    ys = []
    for t in range(q.shape[1]):
        y, state = _gated_delta_step_ops(
            q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state, mask[:, t]
        )
        ys.append(y)
    return mx.stack(ys, axis=1), state


def _chunk_steps(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """One chunk of the reference step loop.

    Every differentiated tensor is an explicit argument: ``mx.checkpoint`` recomputes with
    respect to the callable's *inputs*, so a q/k/v/g/beta closed over from the enclosing
    scope would be outside the recomputed region. ``_gated_delta_step_ops`` is resolved as a
    module global on each call, which is what lets a test count the steps by substituting a
    counting wrapper for it.
    """
    ys = []
    for t in range(q.shape[1]):
        y, state = _gated_delta_step_ops(
            q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state, None
        )
        ys.append(y)
    return mx.stack(ys, axis=1), state


_checkpointed_chunk = mx.checkpoint(_chunk_steps)
_checkpointed_chunk_masked = mx.checkpoint(_chunk_steps_masked)


def gated_delta_chunked_ops(
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
    """``mlx_lm.models.gated_delta.gated_delta_ops`` with a checkpointed chunk boundary.

    Same signature, same shapes and the same semantics as the reference loop:

      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar gating) or [B, T, Hv, Dk] (vectorised gating)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk], float32, zeros when omitted
      - mask: [B, T] or None
    Returns ``(y, state)`` with y: [B, T, Hv, Dv] and the final state.

    ``chunk`` is the number of tokens per checkpointed segment. A trailing partial chunk is
    handled, and ``chunk >= T`` degenerates to a single checkpointed segment covering the
    whole row. T must be at least one, as for the reference loop.
    """
    if chunk < 1:
        raise ValueError(f"gated-delta chunk must be at least one token; got {chunk}")
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    # The reference repeats the key-side tensors before the loop when there are more value
    # heads than key heads (``gated_delta.py:242-244``); doing it here keeps the repeat inside
    # the differentiated region, so gradients reach the un-repeated q and k the same way.
    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    for start in range(0, T, chunk):
        stop = min(start + chunk, T)
        window = slice(start, stop)
        if mask is None:
            y, state = _checkpointed_chunk(
                q[:, window], k[:, window], v[:, window], g[:, window], beta[:, window], state
            )
        else:
            y, state = _checkpointed_chunk_masked(
                q[:, window],
                k[:, window],
                v[:, window],
                g[:, window],
                beta[:, window],
                state,
                mask[:, window],
            )
        ys.append(y)
    y = ys[0] if len(ys) == 1 else mx.concatenate(ys, axis=1)
    return y, state


@contextlib.contextmanager
def install_chunked_gated_delta(chunk: int) -> Iterator[Callable[..., tuple[Any, Any]]]:
    """Make training-mode gated-delta calls take the chunked recurrence, for this block only.

    The patch point is ``mlx_lm.models.gated_delta.gated_delta_ops``. ``qwen3_5`` imports
    ``gated_delta_update`` by value (``qwen3_5.py:16``), but ``gated_delta_update`` resolves
    ``gated_delta_ops`` from its own module's globals when it runs (``gated_delta.py:282``),
    and only on the ``use_kernel=False`` branch — the branch a model in training mode takes.
    So rebinding that one name reaches every training-mode call without touching the
    inference path, which returns from ``gated_delta_kernel`` on the next line
    (``gated_delta.py:283``). Patching ``qwen3_5.gated_delta_update`` instead would sit in
    front of both branches and would have to re-derive ``g`` and ``beta``.

    Yields the installed callable and restores the library's own on exit, exception included.
    """
    module = importlib.import_module("mlx_lm.models.gated_delta")
    original = module.gated_delta_ops

    def chunked(q, k, v, g, beta, state=None, mask=None):
        # Resolved as a module global on each call, so a test can substitute a spy.
        return gated_delta_chunked_ops(q, k, v, g, beta, state, mask, chunk=chunk)

    module.gated_delta_ops = chunked
    try:
        yield chunked
    finally:
        module.gated_delta_ops = original


def training_state_bytes(
    *,
    batch: int,
    heads_v: int,
    dim_v: int,
    dim_k: int,
    tokens: int,
    chunk: int | None,
) -> int:
    """Bytes of recurrence state one layer's backward pass retains, for a preflight estimate.

    One state is ``batch * heads_v * dim_v * dim_k`` float32 elements, four bytes each.

    * ``chunk is None`` — the library's unrolled loop: every one of the ``tokens`` steps
      leaves its state in the graph, so the estimate is ``tokens * state``.
    * otherwise — the chunked recurrence: ``ceil(tokens / chunk)`` boundary states stay in
      the graph, plus one chunk's worth of states live at a time inside the window being
      recomputed, so the estimate is ``(ceil(tokens / chunk) + chunk) * state``.

    The recompute window is charged at the configured chunk length rather than
    ``min(chunk, tokens)``, so a row shorter than one chunk is over-estimated; the number
    gates a memory check, where erring high is the safe direction.
    """
    sizes = {
        "batch": batch,
        "heads_v": heads_v,
        "dim_v": dim_v,
        "dim_k": dim_k,
        "tokens": tokens,
    }
    for name, value in sizes.items():
        if value < 1:
            raise ValueError(f"{name} must be at least one; got {value}")
    if chunk is not None and chunk < 1:
        raise ValueError(f"gated-delta chunk must be at least one token; got {chunk}")
    state = batch * heads_v * dim_v * dim_k * _STATE_BYTES_PER_ELEMENT
    if chunk is None:
        return tokens * state
    return (math.ceil(tokens / chunk) + chunk) * state
