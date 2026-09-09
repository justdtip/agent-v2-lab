"""Upstream's target-summed, source-averaged Jacobian with one unreplicated forward.

The recorder, layer convention and default position rule are upstream's. Only the backward
schedule changes: cotangents have a leading batch axis, while the forward batch stays one.
This saves the forward tape once; transient batched gradients still scale with ``dim_batch``.
Actual device memory savings are unmeasured until the device gate runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from local_llm_lab.upstream_ref import load_upstream


def jacobian_for_prompt_vjp(
    model: Any,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int | None = None,
    position_selector: Callable[[int], Any] | None = None,
) -> tuple[dict[int, Any], int, int]:
    """Return upstream-compatible ``(fp32 CPU maps, sequence length, selected count)``.

    Map keys are upstream block-output indices, with output dimensions as rows and source
    dimensions as columns. A cotangent places one at every selected target position; gradients
    are averaged over the same selected source positions. There is no target-count division.

    ``position_selector(seq_len)`` follows WS-D's existing contract: one nonempty Boolean mask
    replaces the entire position rule for both reductions. It does not change upstream globals.
    Invalid selectors raise RuntimeError, so they cannot be mistaken for short-prompt skips.
    ``None`` delegates to upstream's valid_position_mask and its current skip-first default.

    The caller provides a LensModel and owns placement, precision, determinism and freezing.
    No weights are loaded, moved or frozen here. Upstream's recorder roots the graph at the
    earliest source; frozen model parameters allow the prefix graph to be omitted as upstream
    intends. Only residual gradients are requested; parameter .grad fields are not accumulated.
    """
    import torch

    upstream = load_upstream()
    for name, value in (("dim_batch", dim_batch), ("max_seq_len", max_seq_len)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    skip = upstream.skip_first_default if skip_first is None else skip_first
    if type(skip) is not int or skip < 0:
        raise ValueError("skip_first must be a nonnegative integer")
    sources, target = upstream.fitting._check_layer_indices(
        source_layers, target_layer, model.n_layers
    )
    if not sources:
        raise ValueError("at least one source layer below the target is required")
    width = model.d_model
    if type(width) is not int or width < 1:
        raise ValueError("model.d_model must be a positive integer")

    ids = model.encode(prompt, max_length=max_seq_len)
    if not torch.is_tensor(ids) or ids.ndim != 2 or ids.shape[0] != 1:
        raise RuntimeError("model.encode must return one unpadded sequence with shape [1, S]")
    seq_len = int(ids.shape[1])
    if position_selector is None:
        mask = upstream.valid_position_mask(seq_len, skip_first=skip)
    else:
        try:
            mask = position_selector(seq_len)
            if (
                not torch.is_tensor(mask)
                or mask.dtype != torch.bool
                or tuple(mask.shape) != (seq_len,)
                or not bool(mask.any())
            ):
                raise RuntimeError("expected a nonempty Boolean tensor of shape [sequence length]")
        except Exception as error:
            raise RuntimeError(f"invalid position_selector: {error}") from error
    positions = mask.nonzero(as_tuple=True)[0]
    count = int(positions.numel())
    maps = {
        layer: torch.zeros(width, width, dtype=torch.float32, device="cpu") for layer in sources
    }
    recorder = upstream.fitting.ActivationRecorder(
        model.layers, at=[*sources, target], start_graph_at=min(sources)
    )
    output = inputs = gradients = cotangents = gradient = values = None
    try:
        with recorder, torch.enable_grad():
            model.forward(ids)
            output = recorder.activations[target]
            inputs = [recorder.activations[layer] for layer in sources]
            expected_shape = (1, seq_len, width)
            if any(tuple(value.shape) != expected_shape for value in (output, *inputs)):
                raise RuntimeError("recorded residual shapes differ from the LensModel contract")
            target_positions = positions.to(output.device)
            source_positions = [positions.to(value.device) for value in inputs]
            for start in range(0, width, dim_batch):
                rows = min(dim_batch, width - start)
                cotangents = torch.zeros(
                    (rows, *output.shape), dtype=output.dtype, device=output.device
                )
                row_indices = torch.arange(rows, device=output.device)[:, None]
                cotangents[
                    row_indices, 0, target_positions[None, :], start + row_indices
                ] = 1.0
                gradients = torch.autograd.grad(
                    outputs=output,
                    inputs=inputs,
                    grad_outputs=cotangents,
                    is_grads_batched=True,
                    retain_graph=start + rows < width,
                )
                for layer, gradient, chosen in zip(
                    sources, gradients, source_positions, strict=True
                ):
                    values = gradient[:, 0, chosen, :].float().mean(dim=1)
                    if not bool(torch.isfinite(values).all()):
                        raise RuntimeError(f"non-finite Jacobian at source block {layer}")
                    maps[layer][start : start + rows] = values.detach().cpu()
                gradients = cotangents = gradient = values = None
    except BaseException:
        maps.clear()
        raise
    finally:
        # ActivationRecorder removes hooks but deliberately retains its activation dictionary.
        # Clear our references even when a caller retains this frame through an exception.
        # Deeper framework traceback frames may retain their own tensors on failure.
        recorder.activations.clear()
        output = inputs = gradients = cotangents = gradient = values = None
        ids = mask = positions = target_positions = source_positions = row_indices = chosen = None
    return maps, seq_len, count
