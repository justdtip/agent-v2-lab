"""Add a direction to the residual stream during a training forward, one plan per micro-batch.

The bench's `Injection` (``scripts/inject_repl.py``) cannot be reused here, and the reason is worth
stating because it is not obvious. That hook tracks absolute position in a mutable counter, because
during generation a slice is one token wide and the hook cannot tell where it is from the slice
alone. In a teacher-forced training forward there is exactly one call per block and the whole
sequence arrives at once, so the counter has no job -- and under PyTorch's reentrant gradient
checkpointing the block is recomputed during backward, the hook fires again, and that counter
advances a second time. Measured on an eight-token input: 8 after the forward, 16 after the
backward. The delta would land somewhere else on the pass that produces the gradients, the loss
would look fine, and the model would train against an intervention it never saw.

So: stateless by construction. Position comes from a mask built by the caller, never from anything
the hook remembers.

It is a forward PRE-hook, which matters for three reasons. Gemma 4's decoder layer returns a bare
tensor and its last operation is an in-place multiply by ``layer_scalar``, so a post-hook writing
into that tensor is correct only while that buffer stays non-trainable. A pre-hook adds to a
checkpoint-boundary input and depends on nothing. And the residual index this repository uses --
layer L is the residual after block L-1 -- is exactly the input to block L, so extraction and
injection agree by construction rather than by coincidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def decoder_blocks(model: Any) -> Any:
    """The list of decoder blocks, found without importing the architecture view.

    ``arch_torch`` runs an upstream import at module scope, which the training path on the card
    cannot afford; this needs one attribute walk and nothing else.
    """
    for path in (("model", "layers"), ("model", "language_model", "layers"), ("layers",)):
        node = model
        for name in path:
            node = getattr(node, name, None)
            if node is None:
                break
        if node is not None and len(node):
            return node
    raise AttributeError("no decoder block list found on this model")


@dataclass
class PatchPlan:
    """What one micro-batch's injection is, per row, before it becomes masks.

    `site` is an absolute token index per row, and it is per row because the collator right-pads:
    a scalar site is wrong the moment two rows have different lengths.

    `span` optionally widens a row's injection from that one token to a half-open range. One token
    is not the only sensible scope and has never been the paper's: Lindsey injects from a chosen
    position and continues through the response, and this project's own base-model grid injected at
    every prompt token while its training and evaluation injected at one. A scope difference is a
    different experiment, so it is carried in the plan rather than assumed.
    """

    layer: list[int]
    site: list[int]
    scale: list[float]
    vector: list[torch.Tensor | None]
    live: list[bool] = field(default_factory=list)
    #: per row, (start, end) half-open. None means the single token at `site`.
    span: list[tuple[int, int] | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.layer)
        if not self.live:
            self.live = [v is not None for v in self.vector]
        if not self.span:
            self.span = [None] * n
        for name in ("site", "scale", "vector", "live", "span"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"PatchPlan.{name} has {len(getattr(self, name))} entries, "
                                 f"not {n}")

    def layers_used(self) -> list[int]:
        return sorted({l for l, on in zip(self.layer, self.live) if on})


def build_masks(plan: PatchPlan, *, width: int, hidden: int, device, dtype,
                lengths: list[int] | None = None) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """One (mask, delta) pair per layer the plan touches. A row appears in exactly one layer's mask.

    The mask is zero for every row that layer does not own, so all the layers' hooks can be
    registered together and each still applies only to its own rows.
    """
    rows = len(plan.layer)
    out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in plan.layers_used():
        mask = torch.zeros(rows, width, 1, device=device, dtype=dtype)
        delta = torch.zeros(rows, hidden, device=device, dtype=dtype)
        for row in range(rows):
            if not plan.live[row] or plan.layer[row] != layer:
                continue
            limit = width if lengths is None else lengths[row]
            span = plan.span[row]
            if span is None:
                site = plan.site[row]
                if not 0 <= site < limit:
                    raise IndexError(f"row {row}: site {site} outside its {limit} real tokens")
                mask[row, site, 0] = 1.0
            else:
                if not (isinstance(span, (tuple, list)) and len(span) == 2
                        and all(isinstance(x, int) and not isinstance(x, bool) for x in span)):
                    raise IndexError(f"row {row}: span {span!r} is not a pair of ints")
                start, end = span
                if not 0 <= start < end <= limit:
                    raise IndexError(f"row {row}: span {span} outside its {limit} real tokens")
                mask[row, start:end, 0] = 1.0
            vector = plan.vector[row]
            delta[row] = (vector.to(device=device, dtype=torch.float32)
                          * float(plan.scale[row])).to(dtype)
        out[layer] = (mask, delta)
    return out


class PlannedPatch:
    """One block's injection for one micro-batch. Holds no position state of any kind."""

    def __init__(self, block: Any, *, mask: torch.Tensor, delta: torch.Tensor, layer: int):
        self.block, self.mask, self.delta, self.layer = block, mask, delta, layer
        self.fires = 0
        self._arity: int | None = None
        self.fingerprints: list[tuple[float, float]] = []
        self._handle = None

    def __enter__(self) -> "PlannedPatch":
        self._handle = self.block.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *_: object) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def _hook(self, _module: Any, args: tuple):
        hidden = args[0]
        if hidden.shape[:2] != self.mask.shape[:2]:
            raise AssertionError(
                f"L{self.layer}: the block saw {tuple(hidden.shape[:2])} but the plan is for "
                f"{tuple(self.mask.shape[:2])} -- a decode-shaped slice, a live cache, or the "
                "collator and the plan disagreeing about the batch")
        # The defect worth guarding is a pre-hook that returns a BARE TENSOR: nn.Module wraps
        # that as a one-tuple, so every later positional argument silently becomes None. This hook
        # returns the whole tuple, so what has to hold is that its arity never changes -- which is
        # checkable without knowing what those arguments are.
        #
        # It is NOT "args[1] is None means something is wrong". On a checkpoint with the per-layer
        # embedding disabled -- which this one has, hidden_size_per_layer_input is 0 -- None is
        # the correct value, and asserting otherwise stops a run that is working.
        if self._arity is None:
            self._arity = len(args)
        elif len(args) != self._arity:
            raise AssertionError(
                f"L{self.layer}: the block was called with {len(args)} positional arguments, "
                f"having been called with {self._arity} before. Something upstream is collapsing "
                "them, which drops whatever rode in the later ones.")
        self.fires += 1
        self.fingerprints.append((float(self.mask.sum()), float(self.delta.abs().sum())))
        # Out of place, and every tuple element preserved. `nn.Module` wraps a bare return value
        # as a one-tuple, so returning just the tensor would set every later argument to None.
        out = (hidden + self.mask * self.delta.unsqueeze(1), *args[1:])
        if len(out) != len(args):
            raise AssertionError(f"L{self.layer}: the hook changed the argument count")
        return out

    def fingerprint(self) -> tuple[float, float] | None:
        """What every firing of this hook must have carried, if they all carried the same."""
        return self.fingerprints[0] if self.fingerprints else None

    def consistent(self) -> bool:
        return len(set(self.fingerprints)) <= 1
