"""Torch implementation of the existing residual interface, observed through HF hooks.

Discovery and native readout delegate to jacobian-lens. Model loading, parameter freezing,
attention-kernel selection and precision remain the caller's decisions. Importing this module
never imports MLX or selects CUDA/MPS.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any

import torch
from torch.utils._pytree import tree_leaves
from transformers.cache_utils import DynamicCache

from local_llm_lab.arch_base import (
    _TEXT_MODULE_MEMBERS,
    _TEXT_MODULE_PATHS,
    ArchitectureViewBase,
    _validate_scored_position,
)
from local_llm_lab.torch_capture import TorchCapture
from local_llm_lab.upstream_ref import load_upstream

# Resolve and validate the selected clone before importing any additional upstream module.
load_upstream()
hf = import_module("jlens.hf")

__all__ = ["TorchArchitectureView", "TorchCapture"]


def _promote_bundle(value, dtype):
    """Promote additive/rotary values with diagnostic arithmetic; preserve Boolean predicates."""
    if torch.is_tensor(value):
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _promote_bundle(item, dtype) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_promote_bundle(item, dtype) for item in value)
    return value


def _call_promoted(module, h, kwargs):
    """MLX promotes a mixed-precision matmul; torch requires explicit matching operands.

    Use torch's stateless call with only this module's promoted parameters/buffers. Native
    weights are never converted in place; the temporary diagnostic workspace is per block,
    not a second full model. Gradients through the hidden input remain available.
    """
    replacements = {
        name: value.to(h.dtype)
        for name, value in (*module.named_parameters(), *module.named_buffers())
        if value.is_floating_point() and value.dtype != h.dtype
    }
    if replacements:
        return torch.func.functional_call(module, replacements, (h,), kwargs)
    return module(h, **kwargs)


class _CacheEntry:
    """One layer's view of a shared HF cache; no snapshot/reuse state is promised."""

    def __init__(self, cache, index):
        self.cache, self.index = cache, index

    @property
    def offset(self):
        return int(self.cache.get_seq_length(self.index))

    def is_trimmable(self):
        # A sliding cache discards old keys; crop's existence cannot recover them.
        return callable(getattr(self.cache, "crop", None)) and not any(
            getattr(layer, "is_sliding", False) for layer in self.cache.layers
        )

    def crop(self, tokens_to_remove):
        if not self.is_trimmable():
            raise ValueError("sliding cache cannot guarantee prefix trimming")
        self.cache.crop(tokens_to_remove)


class TorchArchitectureView(ArchitectureViewBase):
    """The ArchitectureView contract on an existing HF model, without loading or mutating it."""

    def __init__(self, model: Any, text_module: Any) -> None:
        self.model, self.text_module = model, text_module
        try:
            layout = hf._find_layout(model)
        except ValueError:
            if not all(hasattr(text_module, name) for name in _TEXT_MODULE_MEMBERS):
                raise
            layout = hf.Layout("")
        else:
            if hf._resolve_attr_path(model, layout.path) is not text_module:
                raise ValueError("explicit text module disagrees with upstream discovery")
        self.layout = layout
        self.blocks = list(getattr(text_module, layout.layers))
        self.layers = self.blocks
        if not self.blocks:
            raise ValueError("text module must contain at least one decoder block")
        self.num_layers = len(self.blocks)
        self._embed_tokens = getattr(text_module, layout.embed)
        self._final_norm = getattr(text_module, layout.norm)
        self.vocab_size, self.hidden_size = self._embed_tokens.weight.shape
        self._lm_head = getattr(model, layout.lm_head, None)
        if self._lm_head is None:
            self._lm_head = getattr(getattr(model, "language_model", None), layout.lm_head, None)
        if self._lm_head is None:
            raise ValueError("could not locate the native unembedding")
        if tuple(self._lm_head.weight.shape) != (self.vocab_size, self.hidden_size):
            raise ValueError("lm_head dimensions must be (vocab_size, hidden_size)")
        self._unembed_module = self._lm_head
        self.tie_word_embeddings = self._lm_head.weight is self._embed_tokens.weight
        config = getattr(text_module, "config", model.config)
        self.config = config.get_text_config() if hasattr(config, "get_text_config") else config
        if any(kind == "linear_attention" for kind in getattr(self.config, "layer_types", ())):
            raise NotImplementedError("recurrent architectures remain on the MLX path")
        if any(bool(getattr(block, "is_linear", False)) for block in self.blocks):
            raise NotImplementedError("recurrent architectures remain on the MLX path")
        self._logit_softcap = getattr(self.config, "final_logit_softcapping", None)
        self._observing = False

    @classmethod
    def from_model(cls, model: Any) -> TorchArchitectureView:
        try:
            layout = hf._find_layout(model)
        except ValueError:
            # The repository's prior structural convention is a fallback, never an alternative
            # selected by model name. Upstream already supports more layouts than it does.
            for path in _TEXT_MODULE_PATHS:
                candidate = model
                for member in path:
                    candidate = getattr(candidate, member, None)
                if candidate is not None and all(
                    hasattr(candidate, member) for member in _TEXT_MODULE_MEMBERS
                ):
                    return cls(model, candidate)
            raise
        return cls(model, hf._resolve_attr_path(model, layout.path))

    @staticmethod
    def _linear_types():
        return (torch.nn.Linear,)

    @staticmethod
    def _linear_dimensions(module):
        shape = tuple(module.weight.shape)
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"linear weight must have positive rank-2 shape; got {shape}")
        return shape

    def layer_kind(self, index: int) -> str:
        self._validate_block_index(index)
        return "attention"

    def attention_span(self, index: int) -> str | None:
        index = self._validate_block_index(index)
        kinds = getattr(self.config, "layer_types", None)
        if kinds is None:
            return None
        if len(kinds) != self.num_layers:
            raise ValueError("config.layer_types must contain one entry per block")
        return {"sliding_attention": "sliding", "full_attention": "global"}.get(kinds[index])

    def _ids(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self._embed_tokens.weight.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2:
            raise ValueError(f"token ids must have one or two dimensions; got {ids.ndim}")
        return ids

    def _unwrap_cache(self, cache):
        if cache is None:
            return None
        if isinstance(cache, DynamicCache):
            return cache
        entries = self._cache_entries(cache)
        if all(entry is None for entry in entries):
            return None
        if not all(isinstance(entry, _CacheEntry) for entry in entries):
            raise ValueError("expected per-layer entries from make_cache")
        shared = entries[0].cache
        if any(entry.cache is not shared or entry.index != i for i, entry in enumerate(entries)):
            raise ValueError("cache entries must belong to one native cache in layer order")
        return shared

    def make_cache(self) -> list[Any]:
        cache = DynamicCache(config=self.config)
        return [_CacheEntry(cache, index) for index in range(self.num_layers)]

    @property
    def cache_trimmable(self) -> bool:
        return all(entry.is_trimmable() for entry in self.make_cache())

    def _observe_forward(self, ids: Any, cache: Any = None):
        """Capture the entry and opaque arguments actually dispatched to each native block.

        This is a real text forward with pre-hooks, not a reconstruction of masks or RoPE.
        Observation runs without autograd. A supplied cache is copied because the observing
        forward must not advance the caller's cache before the hand-run forward uses it.
        That extra forward and copy are explicit costs; no cache-reuse strategy is ported here.
        """
        if self._observing:
            raise RuntimeError("nested architecture observation is unsupported")
        self._observing = True
        handles, entry, seen = [], [], {}
        try:
            for index, block in enumerate(self.blocks):
                signature = inspect.signature(block.forward)

                def observe(module, args, kwargs, *, index=index, signature=signature):
                    bound = signature.bind(*args, **kwargs).arguments
                    # Preserve extra native kwargs rather than invent a family-specific list.
                    values = dict(bound)
                    for key, parameter in signature.parameters.items():
                        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                            values.update(values.pop(key, {}))
                    hidden_key = next(iter(signature.parameters))
                    hidden = values.pop(hidden_key)
                    if index == 0:
                        entry.append(hidden)
                    values.pop("past_key_values", None)
                    values.pop("past_key_value", None)
                    seen[index] = values

                handles.append(block.register_forward_pre_hook(observe, with_kwargs=True))
            native_cache = self._unwrap_cache(cache)
            observed_cache = None
            if native_cache is not None:
                # Deep-copy native cache structure, using detached clones for tensor leaves.
                # torch forbids deepcopy(nonleaf_tensor); observation needs values, not its graph.
                leaves = tree_leaves([vars(native_cache), *[vars(x) for x in native_cache.layers]])
                memo = {id(x): x.detach().clone() for x in leaves if torch.is_tensor(x)}
                observed_cache = copy.deepcopy(native_cache, memo)
            with torch.no_grad():
                self.text_module(
                    input_ids=self._ids(ids),
                    past_key_values=observed_cache,
                    use_cache=observed_cache is not None,
                )
        finally:
            for handle in handles:
                handle.remove()
            self._observing = False
        if len(entry) != 1 or len(seen) != self.num_layers:
            raise ValueError("native forward did not visit the complete discovered decoder")
        return entry[0], seen

    def embed(self, ids: Any) -> Any:
        return self._observe_forward(self._ids(ids))[0].float()

    def masks(
        self,
        h: Any,
        cache: list[Any] | None,
        *,
        hidden_spans: Sequence[tuple[int, int]] | None = None,
        record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if hidden_spans is not None:
            raise ValueError("hidden_spans is deferred on a dense backbone")
        stand_in = torch.zeros(
            h.shape[:2], dtype=torch.long, device=self._embed_tokens.weight.device
        )
        _, observed = self._observe_forward(stand_in, cache)
        if record is not None:
            native_cache = self._unwrap_cache(cache)
            mask = observed[0].get("attention_mask")
            record.update(
                attention_mask_route="model helper, unmodified",
                hidden_spans=(),
                query_tokens=int(h.shape[1]),
                cache_offset=0 if native_cache is None else int(native_cache.get_seq_length()),
                attention_mask_shape=None if mask is None else tuple(mask.shape),
            )
        return observed

    def _block(self, index, h, masks, cache_i):
        index = self._validate_block_index(index)
        if index not in masks:
            raise ValueError(f"missing mask for block {index}")
        kwargs = dict(masks[index])
        signature = inspect.signature(self.blocks[index].forward)
        cache_key = (
            "past_key_value" if "past_key_value" in signature.parameters else "past_key_values"
        )
        kwargs[cache_key] = cache_i.cache if cache_i is not None else None
        kwargs = _promote_bundle(kwargs, h.dtype)
        result = _call_promoted(self.blocks[index], h, kwargs)
        return result if torch.is_tensor(result) else result[0]

    def run_block(self, index: int, h: Any, masks: dict[str, Any], cache_i: Any | None) -> Any:
        return self._block(index, h, masks, cache_i).float()

    def final_norm(self, h: Any) -> Any:
        return _call_promoted(self._final_norm, h, {}).float()

    def native_readout(self, h: Any) -> Any:
        # Reuse upstream's norm/device/dtype/softcap implementation without its constructor,
        # whose parameter-freezing and tokenizer mutations are not a view's authority.
        return hf.HFLensModel.unembed(self, h)

    def unembed(self, h: Any) -> Any:
        logits = _call_promoted(self._lm_head, h, {})
        if self._logit_softcap is not None:
            logits = self._logit_softcap * torch.tanh(logits / self._logit_softcap)
        return logits.float()

    def _wanted(self, layers):
        wanted = sorted({int(layer) for layer in layers})
        if not wanted or wanted[0] < 0 or wanted[-1] > self.num_layers:
            raise ValueError(f"layers must be a non-empty sequence within [0, {self.num_layers}]")
        return wanted

    def residuals(self, ids: Any, layers: Sequence[int]) -> dict[int, Any]:
        wanted = self._wanted(layers)
        h, observed = self._observe_forward(self._ids(ids))
        h = h.float()
        captured = {0: h} if 0 in wanted else {}
        for index in range(wanted[-1]):
            h = self.run_block(index, h, observed, None)
            if index + 1 in wanted:
                captured[index + 1] = h
        return captured

    def native_residuals(self, ids: Any, layers: Sequence[int]) -> dict[int, Any]:
        wanted = self._wanted(layers)
        captured = {}

        class Collect:
            def residual(self, layer, offset, h):
                captured[layer] = h

            def output(self, offset, ids, logits):
                pass

        try:
            with TorchCapture(self, Collect(), layers=wanted) as wrapped:
                wrapped(self._ids(ids))
            if set(captured) != set(wanted):
                raise ValueError("native capture did not emit every requested residual")
            return dict(captured)
        finally:
            captured.clear()

    def residual_source_agreement(self, ids: Any, layers: Sequence[int]) -> dict[int, float]:
        loop, native = self.residuals(ids, layers), self.native_residuals(ids, layers)
        return {
            layer: float((loop[layer].float() - native[layer].float()).abs().max())
            for layer in sorted(loop)
        }

    def diagnostic_native_final_residual(self, ids: Any) -> Any:
        h, observed = self._observe_forward(self._ids(ids))
        for index in range(self.num_layers):
            h = self._block(index, h, observed, None)
        return self._final_norm(h)

    def cached_logits(
        self,
        ids: Any,
        cache: list[Any] | None,
        *,
        hidden_spans: Sequence[tuple[int, int]] | None = None,
        position: int = -1,
        record: dict[str, Any] | None = None,
    ) -> Any:
        entries = self._cache_entries(cache)
        if hidden_spans is not None:
            raise ValueError("hidden_spans is deferred on a dense backbone")
        h, masks = self._observe_forward(self._ids(ids), entries)
        h = h.float()
        index = _validate_scored_position(position, int(h.shape[1]))
        if record is not None:
            mask = masks[0].get("attention_mask")
            native_cache = self._unwrap_cache(entries)
            record.update(
                attention_mask_route="model helper, unmodified",
                hidden_spans=(),
                query_tokens=int(h.shape[1]),
                cache_offset=0 if native_cache is None else int(native_cache.get_seq_length()),
                attention_mask_shape=None if mask is None else tuple(mask.shape),
            )
        for block_index in range(self.num_layers):
            h = self.run_block(block_index, h, masks, entries[block_index])
        return self.unembed(self.final_norm(h)[:, index, :])

    def tail(self, layer: int) -> Callable[[Any], Any]:
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or not 0 <= layer <= self.num_layers
        ):
            raise ValueError(f"layer must lie in [0, {self.num_layers}]; got {layer!r}")

        def apply_tail(h):
            hidden = h.float()
            masks = self.masks(hidden, None)
            for index in range(layer, self.num_layers):
                hidden = self.run_block(index, hidden, masks, None)
            return self.final_norm(hidden)

        return apply_tail
