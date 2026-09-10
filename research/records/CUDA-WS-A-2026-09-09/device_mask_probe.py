"""Read the attention masks the native model hands each block at 1,400 tokens on this device.

Written on the card after the first device run of the WS-A gates: the mask-dispatch control at
1,400 tokens fell inside the precision floor, so either the sliding and global masks the model
observed are the same tensor here, or the control never reached the kernel. This reads them.
"""
import json, sys
import torch
from local_llm_lab import device
from local_llm_lab.hf_text import load_text_causal_lm
from local_llm_lab.arch_torch import TorchArchitectureView

device.pin(seed=0, attention="eager")
ids = json.load(open("research/records/CUDA-WS-A-2026-09-09/calibration-token-ids.json"))
rows = ids["1400"] if isinstance(ids, dict) and "1400" in ids else ids
if isinstance(rows, dict): rows = rows.get("token_ids", rows)
tokens = torch.tensor([rows[:1400] if isinstance(rows[0], int) else rows[0][:1400]])
print("tokens:", tuple(tokens.shape))
model, report = load_text_causal_lm(sys.argv[1], dtype="bfloat16", attn_implementation="eager", device="cuda:0")
view = TorchArchitectureView.from_model(model)
with torch.no_grad():
    entry, bundles = view._observe_forward(view._ids(tokens.to("cuda:0")))
kinds = {}
for index in range(view.num_layers):
    b = bundles[index]; m = b.get("attention_mask")
    span = view.attention_span(index)
    if m is None:
        desc = "None"
    else:
        mm = m
        if mm.dtype != torch.bool:
            allowed = (mm == 0) if mm.min() < 0 else (mm != 0)
        else:
            allowed = mm
        allowed = allowed.reshape(-1, allowed.shape[-2], allowed.shape[-1])[0]
        last = int(allowed[-1].sum()); mid = int(allowed[1100].sum()); first = int(allowed[0].sum())
        desc = f"shape={tuple(m.shape)} dtype={m.dtype} allowed@row0={first} @row1100={mid} @row1399={last} id={id(m)}"
    kinds.setdefault(span, []).append(index)
    if index in (0, 1, 4, 5, 6, 11, 12) or span == "global":
        print(f"block {index:2d} {span:8s} mask {desc}")
print("spans:", {k: (len(v), v[:3]) for k, v in kinds.items()})
print("other bundle keys:", sorted(k for k in bundles[0] if k != "attention_mask"))
print("config sliding_window:", getattr(model.config, "sliding_window", None), "| layer_types sample:", getattr(model.config, "layer_types", ["?"])[:8])
