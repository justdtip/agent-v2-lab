"""Structural access to decoder-only language-model backbones.

The view is the single compatibility boundary for hand-running decoder blocks.  In particular,
it always asks the model's mask helpers for both attention kinds; passing ``None`` as a mask
shortcut would make multi-token probe activations disagree with inference.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any, Literal

__all__ = ["ArchitectureView", "LORA_POLICIES"]

LORA_POLICIES = frozenset({"auto", "attention+mlp", "all-linear"})

_TEXT_MODULE_PATHS = ((), ("model",), ("language_model", "model"))
_TEXT_MODULE_MEMBERS = ("embed_tokens", "layers", "norm")
_ATTENTION_KIND = "attention"
_LINEAR_ATTENTION_KIND = "linear_attention"
_DENSE_LORA_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_LINEAR_ATTENTION_LORA_SUFFIXES = (
    "in_proj_qkvz",
    "in_proj_ba",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


class ArchitectureView:
    """A model-type-independent view of one decoder's embeddings, blocks, and readout."""

    def __init__(self, model: Any, text_module: Any) -> None:
        self.model = model
        self.text_module = text_module
        self.blocks = list(text_module.layers)
        if not self.blocks:
            raise ValueError("text module must contain at least one decoder block")
        self.num_layers = len(self.blocks)
        self.vocab_size, self.hidden_size = _embedding_dimensions(text_module.embed_tokens)
        self._unembed_module = _find_unembedding(model, text_module)
        self.tie_word_embeddings = self._unembed_module is None
        if self.tie_word_embeddings:
            if not callable(getattr(text_module.embed_tokens, "as_linear", None)):
                raise ValueError("tied embed_tokens must expose as_linear")
        else:
            output_size, input_size = _linear_dimensions(self._unembed_module)
            if (output_size, input_size) != (self.vocab_size, self.hidden_size):
                raise ValueError(
                    "lm_head dimensions must be (vocab_size, hidden_size); "
                    f"got {(output_size, input_size)}"
                )
        self._attention_mask = _find_mask_helper(model, text_module, "create_attention_mask")
        self._ssm_mask = _find_mask_helper(model, text_module, "create_ssm_mask")

    @classmethod
    def from_model(cls, model: Any) -> ArchitectureView:
        """Discover the decoder structurally, without consulting a model-type string."""
        for path in _TEXT_MODULE_PATHS:
            candidate = model
            for member in path:
                candidate = getattr(candidate, member, None)
                if candidate is None:
                    break
            if candidate is not None and all(
                hasattr(candidate, member) for member in _TEXT_MODULE_MEMBERS
            ):
                return cls(model, candidate)
        members = ", ".join(_TEXT_MODULE_MEMBERS)
        raise ValueError(
            "could not find a text module owning "
            f"{members} at model, model.model, or model.language_model.model"
        )

    def layer_kind(self, index: int) -> Literal["attention", "linear_attention"]:
        """Return the block's structurally declared attention kind."""
        block = self.blocks[self._validate_block_index(index)]
        if bool(getattr(block, "is_linear", False)):
            return _LINEAR_ATTENTION_KIND
        return _ATTENTION_KIND

    def embed(self, ids: Any) -> Any:
        """Return the layer-zero residual as a batched float32 activation."""
        import mlx.core as mx

        token_ids = mx.array(ids).astype(mx.int32)
        if token_ids.ndim == 1:
            token_ids = token_ids[None, :]
        if token_ids.ndim != 2:
            raise ValueError(f"token ids must have one or two dimensions; got {token_ids.ndim}")
        return self.text_module.embed_tokens(token_ids).astype(mx.float32)

    def masks(self, h: Any, cache: list[Any] | None) -> dict[str, Any]:
        """Build the attention and SSM masks through the model's own helpers."""
        entries: list[Any | None]
        if cache is None:
            entries = [None] * self.num_layers
        else:
            entries = list(cache)
            if len(entries) != self.num_layers:
                raise ValueError(
                    "cache must contain one entry per block; "
                    f"got {len(entries)} for {self.num_layers}"
                )
        attention_cache = self._first_cache(entries, _ATTENTION_KIND)
        linear_cache = self._first_cache(entries, _LINEAR_ATTENTION_KIND)
        return {
            _ATTENTION_KIND: self._attention_mask(h, attention_cache),
            _LINEAR_ATTENTION_KIND: self._ssm_mask(h, linear_cache),
        }

    def run_block(
        self,
        index: int,
        h: Any,
        masks: dict[str, Any],
        cache_i: Any | None,
    ) -> Any:
        """Run one block with the mask selected for that block's attention kind."""
        import mlx.core as mx

        block_index = self._validate_block_index(index)
        kind = self.layer_kind(block_index)
        if kind not in masks:
            raise ValueError(f"missing {kind!r} mask for block {block_index}")
        return self.blocks[block_index](h, mask=masks[kind], cache=cache_i).astype(mx.float32)

    def final_norm(self, h: Any) -> Any:
        """Apply the decoder's final norm and return a float32 activation."""
        import mlx.core as mx

        return self.text_module.norm(h).astype(mx.float32)

    def unembed(self, h: Any) -> Any:
        """Project hidden states to vocabulary logits through the tied or untied readout."""
        import mlx.core as mx

        if self.tie_word_embeddings:
            logits = self.text_module.embed_tokens.as_linear(h)
        else:
            logits = self._unembed_module(h)
        return logits.astype(mx.float32)

    def make_cache(self) -> list[Any]:
        """Create the model-native per-block cache and validate its cardinality."""
        factory = getattr(self.model, "make_cache", None)
        if callable(factory):
            cache = list(factory())
        else:
            from mlx_lm.models.cache import make_prompt_cache

            cache = list(make_prompt_cache(self.model))
        if len(cache) != self.num_layers:
            raise ValueError(
                f"cache factory returned {len(cache)} entries for {self.num_layers} blocks"
            )
        return cache

    @property
    def cache_trimmable(self) -> bool:
        """Whether every native per-block cache supports prefix trimming."""
        cache = self.make_cache()
        for entry in cache:
            check = getattr(entry, "is_trimmable", None)
            if not callable(check):
                raise ValueError(f"cache {type(entry).__name__} does not expose is_trimmable()")
            if not bool(check()):
                return False
        return True

    def lora_targets(self, policy: str | tuple[str, ...]) -> tuple[str, ...]:
        """Resolve a policy to explicit, existing linear-module paths."""
        modules = self._linear_modules()
        available = {path for _, path, _ in modules}
        if isinstance(policy, tuple):
            if not policy:
                raise ValueError("explicit LoRA targets must not be empty")
            if len(set(policy)) != len(policy):
                raise ValueError("explicit LoRA targets must not contain duplicates")
            unknown = tuple(path for path in policy if path not in available)
            if unknown:
                raise ValueError(f"unknown LoRA target(s): {', '.join(unknown)}")
            return policy
        if policy not in LORA_POLICIES:
            raise ValueError(
                f"unknown LoRA policy {policy!r}; expected one of {sorted(LORA_POLICIES)}"
            )
        if policy == "auto":
            hybrid = any(
                self.layer_kind(index) == _LINEAR_ATTENTION_KIND
                for index in range(self.num_layers)
            )
            policy = "all-linear" if hybrid else "attention+mlp"
        suffixes = list(_DENSE_LORA_SUFFIXES)
        if policy == "all-linear":
            suffixes.extend(_LINEAR_ATTENTION_LORA_SUFFIXES)
        selected: list[str] = []
        for suffix in suffixes:
            for block_index, path, _ in modules:
                if path.rsplit(".", 1)[-1] != suffix:
                    continue
                if suffix in _LINEAR_ATTENTION_LORA_SUFFIXES and self.layer_kind(
                    block_index
                ) != _LINEAR_ATTENTION_KIND:
                    continue
                if path not in selected:
                    selected.append(path)
        if not selected:
            raise ValueError(f"LoRA policy {policy!r} matched no linear modules")
        return tuple(selected)

    def lora_parameter_count(self, keys: tuple[str, ...], rank: int) -> int:
        """Count implied LoRA A/B parameters without changing or dequantizing weights."""
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"LoRA rank must be a positive integer; got {rank!r}")
        resolved = set(self.lora_targets(tuple(keys)))
        total = 0
        for _, path, module in self._linear_modules():
            if path in resolved:
                output_size, input_size = _linear_dimensions(module)
                total += rank * (input_size + output_size)
        return total

    def residuals(self, ids: Any, layers: Sequence[int]) -> dict[int, Any]:
        """Capture requested residual indices in one pass, stopping at the deepest index."""
        wanted = sorted({int(layer) for layer in layers})
        if not wanted or wanted[0] < 0 or wanted[-1] > self.num_layers:
            raise ValueError(
                f"layers must be a non-empty sequence within [0, {self.num_layers}]; got {wanted}"
            )
        h = self.embed(ids)
        masks = self.masks(h, None)
        captured: dict[int, Any] = {}
        if 0 in wanted:
            captured[0] = h
        for index in range(wanted[-1]):
            h = self.run_block(index, h, masks, None)
            if index + 1 in wanted:
                captured[index + 1] = h
        return captured

    def tail(self, layer: int) -> Callable[[Any], Any]:
        """Return ``blocks[layer:] + final_norm`` with model-native per-kind masks."""
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or not 0 <= layer <= self.num_layers
        ):
            raise ValueError(f"layer must lie in [0, {self.num_layers}]; got {layer!r}")

        def apply_tail(h: Any) -> Any:
            import mlx.core as mx

            hidden = h.astype(mx.float32)
            masks = self.masks(hidden, None)
            for index in range(layer, self.num_layers):
                hidden = self.run_block(index, hidden, masks, None)
            return self.final_norm(hidden)

        return apply_tail

    def _first_cache(self, cache: list[Any | None], kind: str) -> Any | None:
        for index, entry in enumerate(cache):
            if self.layer_kind(index) == kind:
                return entry
        return None

    def _linear_modules(self) -> list[tuple[int, str, Any]]:
        import mlx.nn as nn

        linear_types = (nn.Linear, nn.QuantizedLinear)
        found: list[tuple[int, str, Any]] = []
        for index, block in enumerate(self.blocks):
            if not callable(getattr(block, "named_modules", None)):
                continue
            found.extend(
                (index, path, module)
                for path, module in block.named_modules()
                if path and isinstance(module, linear_types)
            )
        return found

    def _validate_block_index(self, index: int) -> int:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.num_layers
        ):
            raise ValueError(f"block index must lie in [0, {self.num_layers - 1}]; got {index!r}")
        return index


def _embedding_dimensions(embedding: Any) -> tuple[int, int]:
    quantized_vocab = getattr(embedding, "num_embeddings", None)
    quantized_hidden = getattr(embedding, "dims", None)
    if isinstance(quantized_vocab, int) and isinstance(quantized_hidden, int):
        if quantized_vocab <= 0 or quantized_hidden <= 0:
            raise ValueError("embed_tokens dimensions must be positive")
        return quantized_vocab, quantized_hidden
    weight = getattr(embedding, "weight", None)
    shape = getattr(weight, "shape", ())
    if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
        raise ValueError(f"embed_tokens.weight must have positive rank-2 shape; got {shape}")
    return int(shape[0]), int(shape[1])


def _find_unembedding(model: Any, text_module: Any) -> Any | None:
    owners = (model, getattr(model, "language_model", None), text_module)
    for owner in owners:
        head = getattr(owner, "lm_head", None) if owner is not None else None
        if head is not None:
            return head
    return None


def _find_mask_helper(model: Any, text_module: Any, name: str) -> Callable[..., Any]:
    owners = (text_module, model, getattr(model, "language_model", None))
    for owner in owners:
        helper = getattr(owner, name, None) if owner is not None else None
        if callable(helper):
            return helper
    for owner in owners:
        module = sys.modules.get(type(owner).__module__) if owner is not None else None
        helper = getattr(module, name, None) if module is not None else None
        if callable(helper):
            return helper
    from mlx_lm.models import base

    helper = getattr(base, name, None)
    if not callable(helper):
        raise ValueError(f"model does not provide required mask helper {name}")
    return helper


def _linear_dimensions(module: Any) -> tuple[int, int]:
    import mlx.nn as nn

    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", ())
    if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
        raise ValueError(f"linear weight must have positive rank-2 shape; got {shape}")
    output_size, input_size = int(shape[0]), int(shape[1])
    if isinstance(module, nn.QuantizedLinear):
        bits = getattr(module, "bits", None)
        if isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0 or 32 % bits:
            raise ValueError(f"quantized linear has unsupported bits value {bits!r}")
        input_size = input_size * 32 // bits
    return output_size, input_size
