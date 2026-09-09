"""A torch finite-difference Jacobian, so the golden comparison has a second operand on the device.

**Why this exists and why it is small.** The golden number is the difference between two estimators
on the same rows. The exact one is upstream's; the finite-difference one was
`lens_fitting/jacobian.py`, which is MLX and cannot run beside it. The records hold no
finite-difference Gemma lens and no artefact declaring an estimator at all, and fitting one under
MLX now is the long laptop run the Director has forbidden. So the second operand is this: the same
estimator definition as `jacobian.py`'s, in torch, over **the same model object upstream fits**, so
that backend, rows and precision are identical by construction rather than by discipline, and the
residual is the estimator difference and nothing else.

It exists to be run once and retired. That is what the ledger's deletion of the finite-difference
machinery meant, and this docstring is where it is written down rather than assumed.

**The estimator, stated so it can be checked against upstream's.** For source block ``i`` and target
block ``t``, with ``V`` the valid positions (upstream's own mask, called not reimplemented):

    J[i][a][b] = mean over p in V of  sum over p' in V of  d target[p'][a] / d source[p][b]

which is upstream's rule exactly: a one-hot cotangent at **every selected target position at once**
(hence the sum), then the **mean** over selected source positions. Both reductions take the same
mask, tied, because upstream ties them. Causality makes the p' < p terms zero, but the sum is still
taken over ``V`` and not over every position: upstream's default mask drops the last position, and
summing past it would be a different estimator that no ν field would record.

The derivative is central: ``(f(h + eps e_b) - f(h - eps e_b)) / 2 eps``, with

    eps = epsilon_scale * ||source residual over the full sequence||

for unit basis directions — `jacobian.py`'s own rule, restated here rather than reinvented, so the
two implementations are one estimator.

**How the perturbation is applied, and why not a tail.** A forward hook replaces the block's output,
and the model's own forward runs. `TorchArchitectureView.tail` was the obvious mechanism and is the
wrong one for two reasons: it applies `final_norm`, while upstream's target is the block output
**before** the norm, so the two estimators would be differentiating different functions; and it runs
to the last block rather than to a chosen target. A hook needs neither, uses no knowledge of block
call signatures or masks, and works on the fixture's decoder and on a real HF model without a branch
between them.

**The cost, stated because it decides how this is run.** Full basis, central differences, one full
forward per direction batch:

    forwards = n_rows * |V| * 2 * ceil(d_model / direction_batch)

At the fixture's six dimensions that is nothing. At Gemma's 2560, with 111 valid positions of a
128-token row and a direction batch of 64, it is **8,880 forwards per row** — the reason the golden
comparison at model scale is run on a declared subset of rows and positions, with the exact side
re-run on the *same* subset so that `golden.assert_estimator_is_the_only_difference` has nothing to
refuse. No autograd graph is built, so the memory is one batch of activations rather than a retained
graph; that is this estimator's only advantage over the exact one and it is not the point of it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from local_llm_lab.pipeline.lens_fitting.upstream import (
    ESTIMATOR_FINITE_DIFFERENCE,
    CorpusLensModel,
    CotangentSelectionError,
    UpstreamJacobianFit,
    _observed_precision,
    _require_frozen,
    default_position_selector,
    load_upstream,
    same_device,
    selector_descriptor,
)

#: `jacobian.py`'s rule: the step is a fixed fraction of the source residual's norm over the full
#: sequence, so it scales with the model rather than with an absolute guess.
DEFAULT_EPSILON_SCALE = 0.01

#: The arithmetic path a fit ran, declared because two fits of one checkpoint can share a weight
#: dtype and not share this. `native` writes the perturbation back in the block's own dtype, which
#: is the path upstream's exact estimator runs; `promoted-float32` writes float32 back, so every
#: block above the source runs promoted. They are different arithmetic, and on CUDA at 1,400 tokens
#: they differ by 69.4% (WS-A, 2026-09-11). A golden comparison across them is not a measurement of
#: the estimator, which is why `golden.COMPARABLE_KEYS` refuses it.
CAPTURE_DTYPES = ("native", "promoted-float32")

#: Human-readable form of the same rule, carried into ν beside the residual. A residual without its
#: epsilon cannot be reproduced or compared with another run's.
EPSILON_RULE = "epsilon_scale * ||source residual, full sequence|| / ||direction|| (unit basis)"


def _replace_output(tensor: Any, output: Any) -> Any:
    """Put ``tensor`` back in the shape the block returned, tuple or not.

    Upstream's recorder reads `output if torch.is_tensor(output) else output[0]`; this is the same
    convention in the write direction, so a block whose output upstream can read is one this can
    replace.
    """
    import torch

    if torch.is_tensor(output):
        return tensor
    return (tensor, *tuple(output)[1:])


def _valid_positions(up: Any, seq_len: int, skip: int, selector: Callable[[int], Any] | None) -> Any:
    """Upstream's own mask, or the caller's selector, as a boolean tensor of length ``seq_len``."""
    import torch

    mask = selector(int(seq_len)) if selector is not None else up.valid_position_mask(
        int(seq_len), skip_first=int(skip)
    )
    mask = torch.as_tensor(mask).reshape(-1).to(torch.bool)
    if mask.numel() != seq_len:
        raise CotangentSelectionError(
            f"position mask has {mask.numel()} entries for a sequence of {seq_len}"
        )
    return mask


def fit_finite_difference_jacobian(
    model: Any,
    rows: Iterable[dict],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    position_selector: Callable[[int], Any] | None = None,
    skip_first: int | None = None,
    epsilon_scale: float = DEFAULT_EPSILON_SCALE,
    capture_dtype: str = "native",
    direction_batch: int = 64,
    max_seq_len: int,
    device: str = "cpu",
    dtype: str = "float32",
    split: str | None = "fit",
    max_rows: int | None = None,
    upstream: Any = None,
    progress: Callable[[dict], None] | None = None,
) -> UpstreamJacobianFit:
    """The finite-difference operand, in :class:`UpstreamJacobianFit` so ν and the writer are shared.

    Returning upstream's own result type is deliberate: the two fits then differ in exactly one ν
    field, ``estimator``, which is the condition
    :func:`local_llm_lab.pipeline.lens_fitting.golden.assert_estimator_is_the_only_difference`
    enforces. A separate result type would have made that check pass on a shape rather than on the
    thing it is checking.

    ``max_seq_len`` is required for the reason it is required upstream: upstream fits at 128 and
    reads out at 512 by default and neither signature says so, and a lens read outside its fitted
    length raises nowhere.
    """
    import torch

    up = upstream or load_upstream()
    from jlens.hooks import ActivationRecorder  # the clone is on sys.path once upstream loaded

    started = time.monotonic()
    wrapped = model if isinstance(model, CorpusLensModel) else CorpusLensModel(model)
    _require_frozen(wrapped)
    observed = _observed_precision(wrapped)
    if not same_device(device, observed["device"]) or observed["dtype"] != dtype:
        raise ValueError(
            f"this fit declares {dtype} on {device} and the model is "
            f"{observed['dtype']} on {observed['device']}. The declaration must be a measurement "
            "of what actually ran, not an intention."
        )
    if capture_dtype not in CAPTURE_DTYPES:
        raise ValueError(f"capture_dtype must be one of {CAPTURE_DTYPES}; got {capture_dtype!r}")
    if not isinstance(direction_batch, int) or direction_batch < 1:
        raise ValueError(f"direction_batch must be a positive integer; got {direction_batch!r}")
    if not np.isfinite(epsilon_scale) or epsilon_scale <= 0:
        raise ValueError(f"epsilon_scale must be positive and finite; got {epsilon_scale!r}")

    n_layers, d_model = int(wrapped.n_layers), int(wrapped.d_model)
    target = n_layers - 1 if target_layer is None else int(target_layer)
    if target < 0:
        target += n_layers
    sources = (
        list(range(target)) if source_layers is None else sorted({int(s) for s in source_layers})
    )
    if not sources or sources[0] < 0 or sources[-1] >= target:
        raise ValueError(
            f"source layers {sources} must be nonempty and strictly below target {target}"
        )
    skip = up.skip_first_default if skip_first is None else int(skip_first)

    running = {layer: np.zeros((d_model, d_model), dtype=np.float64) for layer in sources}
    per_prompt: list[dict] = []
    skipped: list[dict] = []
    epsilons: dict[int, list[float]] = {layer: [] for layer in sources}
    n_done = 0
    # On the compute device, not on the CPU the fixture happens to use. The basis, the position
    # mask and the perturbation all meet the model's activations, and a CPU tensor meeting a CUDA
    # one raises at the first real fit rather than at any test on a CPU-only box (found on the
    # card, 2026-09-11). The accumulator stays on the CPU in float64 deliberately: it is the one
    # tensor whose precision decides the result and the one that is cheap to keep off the device.
    compute_device = torch.device(observed["device"])
    basis = torch.eye(d_model, dtype=torch.float32, device=compute_device)

    for row in rows:
        if max_rows is not None and n_done >= max_rows:
            break
        if split is not None and row.get("split") != split:
            continue
        key = f"row:{row['index']}"
        wrapped.register(key, row["ids"])
        ids = wrapped.encode(key, max_length=max_seq_len)
        seq_len = int(ids.shape[1])
        mask = _valid_positions(up, seq_len, skip, position_selector).to(compute_device)
        positions = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
        if not positions:
            # Counted, never absorbed into a smaller n that still returns a lens. Upstream refuses
            # a prompt shorter than its mask needs; this is the same refusal in this estimator.
            skipped.append({"index": int(row["index"]), "reason": "no valid position"})
            continue

        with torch.no_grad():
            with ActivationRecorder(wrapped.layers, at=[*sources, target]) as recorder:
                wrapped.forward(ids)
                # `native` keeps each block's own dtype, so the perturbed forward runs the
                # arithmetic the exact estimator runs; `promoted-float32` is the older path, kept
                # because it is the one a caller may want to *measure* against native rather than
                # inherit by accident.
                base = {
                    layer: (
                        recorder.activations[layer].detach().clone()
                        if capture_dtype == "native"
                        else recorder.activations[layer].detach().float()
                    )
                    for layer in (*sources, target)
                }

            for layer in sources:
                norm = float(torch.linalg.vector_norm(base[layer].float()))
                epsilon = epsilon_scale * norm if norm else epsilon_scale
                epsilons[layer].append(epsilon)
                accumulated = torch.zeros(d_model, d_model, dtype=torch.float64)
                for position in positions:
                    for start in range(0, d_model, direction_batch):
                        columns = basis[start : start + direction_batch]
                        width = int(columns.shape[0])
                        both = []
                        for sign in (1.0, -1.0):
                            handle = wrapped.layers[layer].register_forward_hook(
                                _perturbation(base[layer], position, sign * epsilon, columns)
                            )
                            try:
                                with ActivationRecorder(wrapped.layers, at=[target]) as inner:
                                    wrapped.forward(ids.expand(width, -1))
                                    # The *difference* is always taken in float32: promoting the
                                    # two readings after the forward changes no arithmetic the
                                    # model did, and differencing two bf16 tensors of nearly equal
                                    # value in bf16 would lose the signal to rounding.
                                    both.append(inner.activations[target].detach().float())
                            finally:
                                handle.remove()
                        # Sum over the *selected* target positions, not over every position:
                        # upstream's default mask drops the last one and summing past it would be
                        # a different estimator that no ν field records.
                        delta = ((both[0] - both[1])[:, mask, :]).sum(dim=1) / (2.0 * epsilon)
                        accumulated[:, start : start + width] += delta.T.to(
                            device="cpu", dtype=torch.float64
                        )
                running[layer] += (accumulated / len(positions)).numpy()
                # Per layer, not per row. At Gemma width one row is half an hour of forwards, and
                # a callback that fires only at the end of it is the runbook's forbidden shape:
                # a tool that reports once at the end, on a paid card, cannot say what failed or
                # where. Learned by losing twenty-seven minutes of exactly that (2026-09-11).
                if progress is not None:
                    progress({
                        "event": "layer",
                        "estimator": ESTIMATOR_FINITE_DIFFERENCE,
                        "row_index": int(row["index"]),
                        "upstream_layer": layer,
                        "layers_done": sources.index(layer) + 1,
                        "layers_total": len(sources),
                        "epsilon": epsilon,
                        "elapsed_s": round(time.monotonic() - started, 1),
                    })

        n_done += 1
        record = {
            "index": int(row["index"]),
            "seq_len": seq_len,
            "n_valid_positions": len(positions),
            "split": row.get("split"),
        }
        per_prompt.append(record)
        if progress is not None:
            progress({"event": "row", "estimator": ESTIMATOR_FINITE_DIFFERENCE, **record})

    if n_done == 0:
        raise ValueError(
            f"no usable row: {len(skipped)} skipped. A lens is not returned over a corpus that "
            "contributed nothing."
        )

    precision = dict(observed)
    precision["capture_dtype"] = capture_dtype
    precision["backward_accumulation_dtype"] = "float64"
    precision["note"] = (
        "no autograd graph is built: this estimator differences forward passes, so the "
        "accumulation dtype describes the running sum rather than a backward pass"
    )
    return UpstreamJacobianFit(
        jacobians={layer: (running[layer] / n_done).astype(np.float32) for layer in sources},
        target_layer=target,
        n_prompts=n_done,
        d_model=d_model,
        per_prompt=per_prompt,
        skipped=skipped,
        elapsed_s=round(time.monotonic() - started, 3),
        provenance={
            **up.provenance(),
            "estimator_implementation": __name__,
            "estimator": ESTIMATOR_FINITE_DIFFERENCE,
            "epsilon_rule": EPSILON_RULE,
            "epsilon_scale": epsilon_scale,
            "epsilon_per_layer": {
                str(layer): {
                    "min": min(values), "max": max(values), "mean": float(np.mean(values))
                }
                for layer, values in epsilons.items()
                if values
            },
            "capture_dtype": capture_dtype,
            "direction_basis": "full standard basis",
            "difference": "central",
            "retired_when": (
                "this estimator exists to produce the golden comparison's second operand once; "
                "it is not a fitting path the programme keeps"
            ),
        },
        precision=precision,
        # The same descriptor the exact fit builds, field for field, with `direction_batch` where
        # it records `dim_batch`. Both are knobs of their own estimator and neither changes which
        # positions were chosen; what makes the two fits comparable is the fingerprint above them,
        # and it is computed here exactly as it is there so the two can be compared at all.
        selector=selector_descriptor(
            position_selector
            if position_selector is not None
            else default_position_selector(skip_first=skip, upstream=up),
            probe_lengths=sorted({record["seq_len"] for record in per_prompt}),
        )
        | {
            "rule": (
                "upstream valid_position_mask" if position_selector is None
                else "explicit selector"
            ),
            "skip_first": skip if position_selector is None else None,
            "upstream_default_path": position_selector is None,
            "source_and_target_tied": True,
            "max_seq_len": int(max_seq_len),
            "direction_batch": int(direction_batch),
        },
    )


def _perturbation(base: Any, position: int, step: float, columns: Any):
    """A forward hook that replaces the block's output with the base residual, perturbed.

    The base is written in whole rather than added to, so the batch dimension carries one direction
    per row and every row is otherwise the unperturbed sequence. Adding to the block's own output
    would work only while that output is bit-identical to the recorded base, which is true today and
    is not a property this estimator should depend on.
    """

    def hook(module, inputs, output):
        width = int(columns.shape[0])
        perturbed = base.expand(width, -1, -1).clone()
        perturbed[:, position, :] += (step * columns).to(perturbed.dtype)
        return _replace_output(perturbed, output)

    return hook


__all__ = [
    "CAPTURE_DTYPES",
    "DEFAULT_EPSILON_SCALE",
    "EPSILON_RULE",
    "fit_finite_difference_jacobian",
]
