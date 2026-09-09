"""Cross-entropy over a large vocabulary without ever holding the whole logit tensor.

Gemma 3's vocabulary is 262,208 tokens. At the 2,688-token row cap the arm-A recipe uses, one
sequence's logits are 1.41 GB in bfloat16 and 2.82 GB once cross-entropy upcasts them to float32 --
both live at once, and the backward pass wants a gradient of the same shape again. That is the
largest single allocation in a full fine-tune of this model and it is entirely avoidable: the loss
is a sum over positions, so the projection and the loss can be taken a slice at a time.

Each chunk runs under :func:`torch.utils.checkpoint.checkpoint`, so its logits are freed after the
forward and recomputed during the backward. Peak logit memory becomes one chunk rather than one
sequence, at the cost of recomputing the projection once. The arithmetic, at hidden 2,560 and
vocabulary 262,208:

    chunk   float32 logits held
      256   0.27 GB
      512   0.54 GB
    2,688   2.82 GB   (what not chunking costs)

The function returns the **sum** of the per-token losses and the number of tokens that entered it,
never a mean. A mean over a micro-batch cannot be averaged with another micro-batch's mean unless
both held the same number of supervised tokens, and under length-sorted batching they never do.
Returning the pair lets the caller normalise by the token count across a whole accumulation cycle,
which is what `transformers` does with `num_items_in_batch` and what mlx-lm's unweighted average of
per-batch means does not. That difference is a departure from the MLX records and is recorded in
the run manifest rather than absorbed here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-closure discipline: torch stays out of module scope
    import torch

#: Positions the loss ignores, matching `torch.nn.functional.cross_entropy`'s own default.
IGNORE_INDEX = -100


def _chunk_loss(
    hidden: "torch.Tensor",
    weight: "torch.Tensor",
    bias: "torch.Tensor | None",
    labels: "torch.Tensor",
) -> "torch.Tensor":
    import torch.nn.functional as F

    logits = F.linear(hidden, weight, bias)
    # float32 for the softmax regardless of the parameter dtype: a bfloat16 log-sum-exp over
    # 262,208 logits loses accuracy where it matters most, in the tail this loss is made of.
    return F.cross_entropy(
        logits.float(),
        labels,
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


def chunked_linear_cross_entropy(
    hidden_states: "torch.Tensor",
    weight: "torch.Tensor",
    labels: "torch.Tensor",
    *,
    chunk_size: int = 512,
    bias: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Return ``(summed_loss, supervised_token_count)`` for already-shifted inputs.

    ``hidden_states`` is ``(N, H)`` and ``labels`` is ``(N,)``: the caller does the shift, so that
    the off-by-one between "the forward at position p predicts position p+1" and this function's
    arithmetic is visible where it is decided rather than hidden in here.

    The result is exactly what ``F.cross_entropy(hidden @ weight.T, labels, reduction="sum")``
    returns, to floating-point associativity, and the gradients likewise; ``chunk_size`` trades
    peak memory against one recomputation of the projection and changes no value.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    if hidden_states.dim() != 2:
        raise ValueError(f"hidden_states must be (N, H); got {tuple(hidden_states.shape)}")
    if labels.dim() != 1 or labels.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"labels must be (N,) matching hidden_states; got {tuple(labels.shape)} "
            f"against {tuple(hidden_states.shape)}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")

    count = (labels != IGNORE_INDEX).sum()
    total = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden_states.shape[0], chunk_size):
        stop = start + chunk_size
        hidden_chunk = hidden_states[start:stop]
        label_chunk = labels[start:stop]
        if hidden_states.requires_grad or weight.requires_grad:
            # use_reentrant=False is the supported form and is the one that works when only some
            # inputs require grad, which is exactly the case while the trunk is frozen.
            chunk = checkpoint(
                _chunk_loss, hidden_chunk, weight, bias, label_chunk, use_reentrant=False
            )
        else:
            chunk = _chunk_loss(hidden_chunk, weight, bias, label_chunk)
        total = total + chunk
    return total, count


def shift_for_causal_lm(
    hidden_states: "torch.Tensor", labels: "torch.Tensor"
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Drop the last position's hidden state and the first label, then flatten.

    The forward at position ``p`` predicts position ``p + 1``. Written once, here, because this
    programme has already paid for one join across two position conventions.
    """
    hidden = hidden_states[:, :-1, :].contiguous()
    targets = labels[:, 1:].contiguous()
    return hidden.view(-1, hidden.shape[-1]), targets.view(-1)
