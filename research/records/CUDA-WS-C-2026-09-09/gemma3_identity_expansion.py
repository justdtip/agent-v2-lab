"""Does the depth-expansion mechanism still produce an exact identity on Gemma 3?

`depth_expansion.make_identity_block` was written against MLX's Qwen2 block, which carries two
norms. Gemma 3 carries four (`pre_feedforward_layernorm` and `post_feedforward_layernorm` in
addition), and indexes each layer's attention type by position through `config.layer_types`. Both
differences are silent: a block copied without the extra norms, or inserted without repairing
`layer_types`, still runs and still trains, and the model it trains is not the one the recipe
describes.

This probe asserts the property the mechanism exists to have -- that the expanded model is
function-identical to its source at initialisation -- on a randomly initialised tiny Gemma 3, so
it needs no checkpoint and no accelerator. Run under `--strict` it exits non-zero unless the
difference is exactly zero.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

# The insertion rule under test. The copy inherits the type of the block it copies, so every
# original layer keeps its own attention type and only its position moves.
def insert_identity_block(model: Gemma3ForCausalLM, after: int) -> int:
    inner = model.model
    block = copy.deepcopy(inner.layers[after])
    with torch.no_grad():
        block.self_attn.o_proj.weight.zero_()
        block.mlp.down_proj.weight.zero_()
        if block.self_attn.o_proj.bias is not None:
            block.self_attn.o_proj.bias.zero_()
        if block.mlp.down_proj.bias is not None:
            block.mlp.down_proj.bias.zero_()

    layers = list(inner.layers)
    layers.insert(after + 1, block)
    inner.layers = torch.nn.ModuleList(layers)

    types = list(model.config.layer_types)
    types.insert(after + 1, types[after])
    model.config.layer_types = types
    model.config.num_hidden_layers = len(layers)

    # `layer_idx` addresses the KV cache and is stale for every layer after the seam; the forward
    # pass reads masks from the config rather than the module, so training does not notice and
    # generation does.
    for index, layer in enumerate(inner.layers):
        layer.layer_idx = index
        layer.self_attn.layer_idx = index
        layer.self_attn.layer_type = types[index]
    return after + 1


def build_tiny(seed: int) -> tuple[Gemma3ForCausalLM, Gemma3TextConfig]:
    torch.manual_seed(seed)
    config = Gemma3TextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=12,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=8,
        rms_norm_eps=1e-6, use_cache=False,
    )
    model = Gemma3ForCausalLM(config).eval()
    # Norm weights initialise to zero, and Gemma's RMSNorm scales by (1 + weight); left at zero
    # every norm is the identity and the test cannot tell a copied norm from a missing one.
    for parameter in model.parameters():
        if parameter.dim() == 1:
            torch.nn.init.normal_(parameter, std=0.3)
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    model, config = build_tiny(args.seed)
    ids = torch.randint(0, config.vocab_size, (2, 24))
    with torch.no_grad():
        baseline = model(input_ids=ids, use_cache=False).logits.clone()

    rows = []
    for after in range(config.num_hidden_layers):
        expanded = copy.deepcopy(model)
        added = insert_identity_block(expanded, after)
        with torch.no_grad():
            got = expanded(input_ids=ids, use_cache=False).logits
        rows.append({
            "insert_after": after,
            "added_index": added,
            "source_type": config.layer_types[after],
            "max_abs_delta_logits": (got - baseline).abs().max().item(),
            "global_layers_before": [i for i, t in enumerate(config.layer_types) if t == "full_attention"],
            "global_layers_after": [i for i, t in enumerate(expanded.config.layer_types) if t == "full_attention"],
        })

    worst = max(row["max_abs_delta_logits"] for row in rows)
    payload = {
        "torch": torch.__version__,
        "base_layer_types": list(config.layer_types),
        "worst_max_abs_delta_logits": worst,
        "exact_identity_everywhere": worst == 0.0,
        "rows": rows,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.strict and worst != 0.0:
        raise SystemExit(f"identity broken: worst max|delta| = {worst}")


if __name__ == "__main__":
    main()
