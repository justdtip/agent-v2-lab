"""Native torch capture extending Anthropic's Apache-2.0 ActivationRecorder.

Upstream owns block hook installation and graph-preserving activation recording. This
adapter adds our residual numbering, native input observation and replacement interventions.
"""

from __future__ import annotations

import inspect
import math

import torch
from jlens.hooks import ActivationRecorder


def _arguments(module, args, kwargs):
    """Bind exactly what the native caller supplied; invent no mask or rotary values."""
    signature = inspect.signature(module.forward)
    values = dict(signature.bind_partial(*args, **kwargs).arguments)
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            values.update(values.pop(name, {}))
    return values


class TorchCapture(ActivationRecorder):
    """Sink-compatible native capture, preserving tensors' dtype and autograd graph.

    Layer zero is the input of block zero; layer L is block L-1's output before the
    final normalization. ``intervene(L, p, fn)`` gives ``fn`` a cloned hidden vector
    at absolute position p, and replaces that vector with its same-shape/dtype/device
    result. Multiple interventions compose in registration order. They apply once per
    fresh-prefill sequence; attempting an uncaptured past position fails closed.

    ``layer_inputs`` retains native block keyword bundles without hidden states, so
    observing kwargs does not itself retain every residual. Native outputs remain in
    upstream's ``activations`` only at requested/intervened blocks until the next
    forward or context exit. Attention/head capture is not ported.
    """

    def __init__(
        self, view, sink, *, layers, attention_blocks=(), injection=None, head_vectors=True
    ):
        self.view, self.model, self.sink = view, view.model, sink
        self.layers = set(layers)
        if not self.layers or any(
            type(i) is not int or not 0 <= i <= view.num_layers for i in self.layers
        ):
            raise ValueError("capture layers outside model")
        if tuple(attention_blocks):
            raise NotImplementedError("torch attention/head capture is not ported")
        self.attention_blocks, self.head_vectors = set(), head_vectors
        self.injection = injection
        self.layer_inputs = {}
        self.entry_residual = None
        self._active = False
        self._forwarding = False
        self._offset = 0
        self._ids = None
        self._interventions = []
        self._applied = set()
        super().__init__(view.layers, (layer - 1 for layer in self.layers if layer))
        if injection is not None:
            layer, position, delta = injection
            if (
                type(layer) is not int
                or not 1 <= layer <= view.num_layers
                or len(delta) != view.hidden_size
                or any(not math.isfinite(float(value)) for value in delta)
            ):
                raise ValueError("invalid injection layer, source or direction")

            def add(h):
                direction = torch.as_tensor(delta, device=h.device, dtype=h.dtype)
                return h if not bool(torch.any(direction != 0)) else h + direction

            self.intervene(layer, position, add)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def intervene(self, layer, position, fn):
        """Register a position replacement; add and interchange are ordinary functions."""
        if (
            type(layer) is not int
            or not 0 <= layer <= self.view.num_layers
            or type(position) is not int
            or position < 0
            or not callable(fn)
        ):
            raise ValueError("invalid intervention layer, position or function")
        if self._forwarding:
            raise RuntimeError("cannot register interventions during a forward")
        if layer and layer - 1 not in self._indices:
            if self._active:
                self._handles.append(
                    self._blocks[layer - 1].register_forward_hook(self._make_hook(layer - 1))
                )
            self._indices = sorted({*self._indices, layer - 1})
        self._interventions.append((layer, position, fn))
        return self

    def _replace(self, layer, h):
        for number, (target, position, fn) in enumerate(self._interventions):
            local = position - self._offset
            if target != layer or number in self._applied or not 0 <= local < h.shape[1]:
                continue
            replacement = fn(h[0, local].clone())
            if (
                not torch.is_tensor(replacement)
                or replacement.shape != h[0, local].shape
                or replacement.dtype != h.dtype
                or replacement.device != h.device
            ):
                raise ValueError("intervention replacement must preserve shape, dtype and device")
            result = h.clone()
            result[0, local] = replacement
            h = result
            self._applied.add(number)
        return h

    def _before_model(self, module, args, kwargs):
        if self._forwarding:
            raise RuntimeError("capture cannot observe recursive model forwards")
        supplied = _arguments(module, args, kwargs)
        ids = supplied.get("input_ids", args[0] if args else None)
        if not torch.is_tensor(ids) or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("capture supports an unpadded batch of one")
        cache = supplied.get("past_key_values")
        self._offset = int(cache.get_seq_length()) if cache is not None else 0
        self._ids = ids
        self.activations.clear()
        self.layer_inputs.clear()
        self.entry_residual = None
        if self._offset == 0:
            self._applied.clear()
        if any(
            position < self._offset and i not in self._applied
            for i, (_, position, _) in enumerate(self._interventions)
        ):
            raise ValueError("intervention source is already cached; rebuild from a fresh prefill")
        self._forwarding = True

    def _before_block(self, index):
        def hook(module, args, kwargs):
            values = _arguments(module, args, kwargs)
            hidden_key = next(iter(inspect.signature(module.forward).parameters))
            h = values.pop(hidden_key)
            self.layer_inputs[index] = values
            if index != 0:
                return None
            replacement = self._replace(0, h)
            if 0 in self.layers:
                self.entry_residual = replacement
                self.sink.residual(0, self._offset, replacement)
            if replacement is h:
                return None
            if args:
                return (replacement, *args[1:]), kwargs
            return args, {**kwargs, hidden_key: replacement}

        return hook

    def _make_hook(self, index):
        record = super()._make_hook(index)

        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            replacement = self._replace(index + 1, h)
            updated = output
            if replacement is not h:
                if torch.is_tensor(output):
                    updated = replacement
                elif isinstance(output, tuple):
                    updated = (replacement, *output[1:])
                elif isinstance(output, list):
                    updated = [replacement, *output[1:]]
                else:
                    raise TypeError("intervention requires tensor, tuple or list block output")
            record(module, inputs, updated)
            if index + 1 in self.layers:
                self.sink.residual(index + 1, self._offset, self.activations[index])
            return updated if updated is not output else None

        return hook

    def _after_model(self, module, args, kwargs, output):
        self._forwarding = False
        if output is not None:
            logits = (
                output
                if torch.is_tensor(output)
                else (output.logits if hasattr(output, "logits") else output[0])
            )
            self.sink.output(self._offset, self._ids[0].tolist(), logits)

    def __call__(self, ids, *args, **kwargs):
        if not self._active:
            raise RuntimeError("native capture must be entered before forwarding")
        if args or "cache" in kwargs:
            if len(args) > 1 or (args and "cache" in kwargs) or "past_key_values" in kwargs:
                raise TypeError("pass only one cache")
            cache = args[0] if args else kwargs.pop("cache")
            kwargs["past_key_values"] = self.view._unwrap_cache(cache)
            kwargs.setdefault("use_cache", cache is not None)
        kwargs.setdefault("use_cache", kwargs.get("past_key_values") is not None)
        return self.model(input_ids=ids, **kwargs)

    def __enter__(self):
        if self._active:
            raise RuntimeError("native capture cannot be nested")
        self._active = True
        try:
            super().__enter__()
            for index, block in enumerate(self._blocks):
                self._handles.append(
                    block.register_forward_pre_hook(self._before_block(index), with_kwargs=True)
                )
            self._handles.append(
                self.model.register_forward_pre_hook(self._before_model, with_kwargs=True)
            )
            self._handles.append(
                self.model.register_forward_hook(
                    self._after_model, with_kwargs=True, always_call=True
                )
            )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        try:
            super().__exit__(*exc)
        finally:
            self._active = self._forwarding = False
            self.activations.clear()
            self.layer_inputs.clear()
            self.entry_residual = self._ids = None
