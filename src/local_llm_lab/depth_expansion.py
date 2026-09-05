from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: Recipe versions this module knows how to apply. A ``version`` field exists to say that a
#: recipe is NOT one of these, so it is compared rather than merely required (#75): reading it
#: and never checking it makes it a comment with a type annotation.
SUPPORTED_VERSIONS: tuple[int, ...] = (1,)


@dataclass(frozen=True)
class ExpansionSpec:
    """Serializable interleaved block-expansion recipe."""

    insertion_after: tuple[int, ...]
    initialization: str = "copied_identity"
    version: int = 1

    def __post_init__(self) -> None:
        if not self.insertion_after:
            raise ValueError("insertion_after must contain at least one layer index")
        if tuple(sorted(set(self.insertion_after))) != self.insertion_after:
            raise ValueError("insertion_after must be sorted and unique")
        if self.insertion_after[0] < 0:
            raise ValueError("layer indices must be non-negative")
        if self.initialization != "copied_identity":
            raise ValueError(f"unsupported initialization: {self.initialization}")
        # Validated here rather than in ``load`` so a spec constructed directly cannot bypass
        # the check that a file cannot; ``load`` builds through this constructor, so it is
        # covered by construction. An older version is refused rather than migrated: a
        # migration that does not exist must not be implied by silence (#75).
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"unsupported expansion recipe version {self.version!r}; "
                f"this code applies {list(SUPPORTED_VERSIONS)}"
            )

    @classmethod
    def evenly_spaced(cls, base_layers: int, added_layers: int) -> ExpansionSpec:
        if base_layers < 1 or added_layers < 1 or added_layers > base_layers:
            raise ValueError("require 1 <= added_layers <= base_layers")
        positions = tuple(
            min(base_layers - 1, ((index + 1) * base_layers // added_layers) - 1)
            for index in range(added_layers)
        )
        return cls(tuple(sorted(set(positions))))

    @classmethod
    def load(cls, path: Path) -> ExpansionSpec:
        """Read back what :meth:`save` wrote, requiring all three fields it always writes.

        R38 (issue #70): ``initialization`` and ``version`` used to be read with the dataclass
        defaults. ``save`` serialises through ``asdict``, so it has emitted all three since the
        file's first commit and no spec lacking them has ever been written -- the fallbacks had
        no subjects. They were not harmless either: the defaults are the only values these
        fields have ever taken, so a key one level from where ``save`` puts it read back as
        exactly the right answer, and ``version`` is the field whose whole job is to say that a
        recipe is NOT the one this code knows how to apply.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = {"insertion_after", "initialization", "version"} - payload.keys()
        if missing:
            raise ValueError(f"{path}: expansion spec is missing {sorted(missing)}")
        return cls(
            insertion_after=tuple(payload["insertion_after"]),
            initialization=payload["initialization"],
            version=payload["version"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["insertion_after"] = list(self.insertion_after)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _effective_linear(module: Any) -> Any:
    """Return a floating-point Linear containing any loaded LoRA contribution."""
    import mlx.core as mx
    import mlx.nn as nn

    if hasattr(module, "fuse"):
        fused = module.fuse(dequantize=True)
        if not isinstance(fused, nn.Linear):
            raise TypeError(f"expected fused Linear, got {type(fused).__name__}")
        return fused
    if isinstance(module, nn.QuantizedLinear):
        weight = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
        )
        output_dims, input_dims = weight.shape
        linear = nn.Linear(input_dims, output_dims, bias="bias" in module)
        linear.weight = weight
        if "bias" in module:
            linear.bias = module.bias
        return linear
    if isinstance(module, nn.Linear):
        return module
    raise TypeError(f"cannot copy linear module {type(module).__name__}")


def _copy_linear(target: Any, source: Any, *, zero: bool = False) -> None:
    import mlx.core as mx

    effective = _effective_linear(source)
    if target.weight.shape != effective.weight.shape:
        raise ValueError(
            f"linear shape mismatch: {target.weight.shape} != {effective.weight.shape}"
        )
    target.weight = mx.zeros_like(effective.weight) if zero else mx.array(effective.weight)
    if "bias" in target:
        target.bias = mx.zeros_like(effective.bias) if zero else mx.array(effective.bias)


def _copy_norm(target: Any, source: Any) -> None:
    import mlx.core as mx

    target.weight = mx.array(source.weight)


def make_identity_block(source: Any, model_args: Any) -> Any:
    """Copy a Qwen2 block, zeroing residual outputs to make an exact identity."""
    from mlx_lm.models.qwen2 import TransformerBlock

    block = TransformerBlock(model_args)
    _copy_norm(block.input_layernorm, source.input_layernorm)
    _copy_norm(block.post_attention_layernorm, source.post_attention_layernorm)
    _copy_linear(block.self_attn.q_proj, source.self_attn.q_proj)
    _copy_linear(block.self_attn.k_proj, source.self_attn.k_proj)
    _copy_linear(block.self_attn.v_proj, source.self_attn.v_proj)
    _copy_linear(block.self_attn.o_proj, source.self_attn.o_proj, zero=True)
    _copy_linear(block.mlp.gate_proj, source.mlp.gate_proj)
    _copy_linear(block.mlp.up_proj, source.mlp.up_proj)
    _copy_linear(block.mlp.down_proj, source.mlp.down_proj, zero=True)
    return block


def expand_model(model: Any, spec: ExpansionSpec) -> tuple[int, ...]:
    """Insert trainable identity blocks and freeze all inherited parameters."""
    original_layers = list(model.layers)
    if spec.insertion_after[-1] >= len(original_layers):
        raise ValueError(
            f"expansion references layer {spec.insertion_after[-1]}, "
            f"but model has {len(original_layers)} layers"
        )

    model.freeze()
    expanded_layers: list[Any] = []
    added_indices = []
    insertion_set = set(spec.insertion_after)
    for original_index, layer in enumerate(original_layers):
        expanded_layers.append(layer)
        if original_index in insertion_set:
            block = make_identity_block(layer, model.args)
            block.unfreeze()
            expanded_layers.append(block)
            added_indices.append(len(expanded_layers) - 1)

    model.model.layers = expanded_layers
    model.model.num_hidden_layers = len(expanded_layers)
    return tuple(added_indices)


def load_expanded_model(
    model_path: str,
    base_adapter: Path,
    spec: ExpansionSpec,
    extension_weights: Path | None = None,
) -> tuple[Any, Any, tuple[int, ...]]:
    """Load the quantized base+LoRA policy, rebuild the extension, and load its weights."""
    from mlx_lm import load

    model, tokenizer = load(model_path, adapter_path=str(base_adapter.resolve()))
    added_indices = expand_model(model, spec)
    if extension_weights is not None:
        model.load_weights(str(extension_weights.resolve()), strict=False)
    return model, tokenizer, added_indices


def trainable_parameter_count(model: Any) -> int:
    from mlx.utils import tree_flatten

    return sum(value.size for _, value in tree_flatten(model.trainable_parameters()))
