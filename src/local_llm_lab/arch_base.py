"""Backend-neutral architecture contracts, shared without importing an array runtime.

Residual layer L is the output of block L-1; layer zero is the model's observed
entry residual and the final residual is before the installed final norm.
Backend views supply their own linear types and physical weight dimensions.
"""

from __future__ import annotations

from typing import Any

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



class ArchitectureViewBase:
    """Index, cache-cardinality and LoRA contracts shared by backend views.

    Subclasses own ``blocks``, ``num_layers`` and ``layer_kind`` and provide
    ``_linear_types()`` and ``_linear_dimensions(module)``. Traversal reports
    adapter wrappers under the same paths as their underlying projections.
    """

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
                output_size, input_size = self._linear_dimensions(module)
                total += rank * (input_size + output_size)
        return total

    def _cache_entries(self, cache: list[Any] | None) -> list[Any | None]:
        """One cache entry per block, or one ``None`` per block when running uncached."""
        if cache is None:
            return [None] * self.num_layers
        entries = list(cache)
        if len(entries) != self.num_layers:
            raise ValueError(
                f"cache must contain one entry per block; got {len(entries)} "
                f"for {self.num_layers}"
            )
        return entries

    def _first_cache(self, cache: list[Any | None], kind: str) -> Any | None:
        for index, entry in enumerate(cache):
            if self.layer_kind(index) == kind:
                return entry
        return None

    def _linear_modules(self) -> list[tuple[int, str, Any]]:
        """Every linear projection per block, seen through adapter wrappers.

        Loading a policy with an adapter (``mlx_lm.load(..., adapter_path=...)``) replaces
        each projection with a ``LoRALinear``: a plain ``nn.Module`` that keeps the original at
        ``.linear``. The wrapper is not a linear type, and its child sits at ``<path>.linear``,
        whose suffix matches no LoRA target. So a wrapper is reported under *its own* path with
        the *base* module, and the base is not reported a second time. Paths therefore stay
        identical with or without an adapter, which adapter comparison and provenance rely on,
        and dimension readers receive the module that owns ``weight`` and ``bits``.
        """
        linear_types = self._linear_types()
        found: list[tuple[int, str, Any]] = []
        for index, block in enumerate(self.blocks):
            if not callable(getattr(block, "named_modules", None)):
                continue
            entries = [(path, module) for path, module in block.named_modules() if path]
            wrapped: dict[str, Any] = {}
            for path, module in entries:
                if isinstance(module, linear_types):
                    continue
                base = getattr(module, "linear", None)
                if isinstance(base, linear_types):
                    wrapped[path] = base
            for path, module in entries:
                if path in wrapped:
                    found.append((index, path, wrapped[path]))
                    continue
                if any(path.startswith(prefix + ".") for prefix in wrapped):
                    continue
                if isinstance(module, linear_types):
                    found.append((index, path, module))
        return found

    def _validate_block_index(self, index: int) -> int:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.num_layers
        ):
            raise ValueError(f"block index must lie in [0, {self.num_layers - 1}]; got {index!r}")
        return index


def _validate_scored_position(position: Any, length: int) -> int:
    """Resolve the scored row, allowing Python's negative indexing from the end."""
    if isinstance(position, bool) or not isinstance(position, int):
        raise ValueError(f"scored position must be an integer; got {position!r}")
    if not -length <= position < length:
        raise ValueError(
            f"scored position must lie in [{-length}, {length - 1}]; got {position}"
        )
    return position


def _validate_hidden_span(span: Any, total: int) -> tuple[int, int]:
    """Check one half-open ``(start, end)`` span against the key range it must lie in.

    An empty or inverted span is rejected rather than ignored: it would hide nothing while the
    arm still reported itself as masked, which is the failure mode that reads as a null.
    """
    try:
        pair = tuple(span)
    except TypeError as error:
        raise ValueError(f"hidden span must be a (start, end) pair; got {span!r}") from error
    if len(pair) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in pair
    ):
        raise ValueError(f"hidden span must be a pair of integers; got {span!r}")
    start, end = pair
    if not 0 <= start < end <= total:
        raise ValueError(
            f"hidden span must satisfy 0 <= start < end <= {total}; got {span!r}"
        )
    return start, end
