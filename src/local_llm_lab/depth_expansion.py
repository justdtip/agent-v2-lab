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


# --- torch ---------------------------------------------------------------------------------
#
# The MLX path above builds each inserted block by constructing a fresh ``TransformerBlock`` and
# copying named submodules into it. That spelling is architecture-specific and fails silently on
# an architecture with more parts than the one it was written for: Gemma 3's block carries
# ``pre_feedforward_layernorm`` and ``post_feedforward_layernorm`` in addition to Qwen 2's two
# norms, and a field-by-field copy leaves them default-initialised while still running and still
# training. The torch path therefore deep-copies the source block whole and zeroes only the two
# residual outputs, which is both shorter and correct for any block whose residual stream is
# ``x + o_proj(...)`` and ``x + down_proj(...)``.
#
# Gemma 3 also selects each layer's attention type and rotary embedding by *position*, through
# ``config.layer_types[i]`` read in the model's forward. Inserting a block shifts every later
# layer's index, so an insertion that does not repair the config silently moves which layers
# attend globally. ``expand_model_torch`` repairs it, giving each inserted block the attention
# type of the block it copies, so every original layer keeps its own.
#
# Evidence: research/records/CUDA-WS-C-2026-09-09 checks the property this construction exists to
# have -- that the expanded model is function-identical to its source at initialisation -- at
# every insertion point of a tiny Gemma 3, and finds a maximum logit difference of exactly zero.

#: The two projections whose outputs are added back into the residual stream. Zeroing both makes
#: an inserted block an exact identity; stage 1 of training warms them away from it.
RESIDUAL_OUTPUT_PROJECTIONS: tuple[tuple[str, str], ...] = (
    ("self_attn", "o_proj"),
    ("mlp", "down_proj"),
)


def _zero_residual_outputs(block: Any) -> None:
    import torch

    for parent_name, child_name in RESIDUAL_OUTPUT_PROJECTIONS:
        parent = getattr(block, parent_name, None)
        projection = getattr(parent, child_name, None) if parent is not None else None
        if projection is None:
            raise TypeError(
                f"block {type(block).__name__} has no {parent_name}.{child_name}; "
                "an identity block cannot be made without knowing which outputs to zero"
            )
        with torch.no_grad():
            projection.weight.zero_()
            if getattr(projection, "bias", None) is not None:
                projection.bias.zero_()


def _rebind_layer_positions(layers: Any, layer_types: list[str] | None) -> None:
    """Point every layer at the index it now occupies.

    ``layer_idx`` addresses the KV cache and ``layer_type`` selects the attention mask. The
    forward pass reads the mask from the config rather than from the module, so a stale
    ``layer_idx`` does not disturb training and does disturb generation -- which is the reason to
    repair it here rather than where the failure would appear.
    """
    for index, layer in enumerate(layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = index
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            if hasattr(attention, "layer_idx"):
                attention.layer_idx = index
            if layer_types is not None and hasattr(attention, "layer_type"):
                attention.layer_type = layer_types[index]


def expand_model_torch(model: Any, spec: ExpansionSpec) -> tuple[int, ...]:
    """Insert identity blocks into a torch causal LM and freeze everything inherited.

    Mirrors :func:`expand_model`: the returned indices are positions in the *expanded* stack, all
    inherited parameters come back frozen, and the added blocks come back trainable.
    """
    import copy as _copy

    import torch

    inner = model.model
    original = list(inner.layers)
    if spec.insertion_after[-1] >= len(original):
        raise ValueError(
            f"expansion references layer {spec.insertion_after[-1]}, "
            f"but model has {len(original)} layers"
        )

    config = model.config
    original_types = getattr(config, "layer_types", None)
    if original_types is not None and len(original_types) != len(original):
        raise ValueError(
            f"config.layer_types has {len(original_types)} entries for {len(original)} layers; "
            "the config and the stack disagree before anything was inserted"
        )
    types: list[str] | None = list(original_types) if original_types is not None else None

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    expanded: list[Any] = []
    added_indices: list[int] = []
    insertion_set = set(spec.insertion_after)
    for original_index, layer in enumerate(original):
        expanded.append(layer)
        if original_index in insertion_set:
            block = _copy.deepcopy(layer)
            _zero_residual_outputs(block)
            for parameter in block.parameters():
                parameter.requires_grad_(True)
            expanded.append(block)
            added_indices.append(len(expanded) - 1)
            if types is not None:
                # The copy is the source block, so it inherits the source's attention type and
                # every original layer keeps its own. Indexed into the *original* list: `types`
                # has already shifted under earlier insertions, and reading the shifted list here
                # gives the inserted block a neighbour's attention type. Nothing downstream
                # notices -- a zeroed block outputs zero whichever way it attends, so the identity
                # check passes either way and the recipe is wrong from the first warmed step.
                types.insert(len(expanded) - 1, original_types[original_index])

    inner.layers = torch.nn.ModuleList(expanded)
    config.num_hidden_layers = len(expanded)
    if types is not None:
        config.layer_types = types
    _rebind_layer_positions(inner.layers, types)
    return tuple(added_indices)


def trainable_parameter_count_torch(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_output_warmup_torch(model: Any, added_indices: tuple[int, ...]) -> int:
    """Stage 1: train only the zeroed residual outputs of the added blocks.

    The copied feature projections stay fixed, so the expanded model departs from exact identity
    only through the two projections that were zeroed to create it.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for index in added_indices:
        block = model.model.layers[index]
        for parent_name, child_name in RESIDUAL_OUTPUT_PROJECTIONS:
            for parameter in getattr(getattr(block, parent_name), child_name).parameters():
                parameter.requires_grad_(True)
    return trainable_parameter_count_torch(model)


def set_full_extension_training_torch(model: Any, added_indices: tuple[int, ...]) -> int:
    """Stage 2: open every parameter in the added blocks, inherited weights still frozen."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for index in added_indices:
        for parameter in model.model.layers[index].parameters():
            parameter.requires_grad_(True)
    return trainable_parameter_count_torch(model)
