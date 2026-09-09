"""Is the derivative's sensitivity to its anchor the arithmetic's, or the function's?

The same-anchor control found that at repo layer 1 a **0.19%** move of the anchor changes the native
bf16 directional derivative read at it by **50%**. Two readings fit that, and they call for opposite
conclusions: bfloat16's arithmetic is that fragile, or the function is genuinely that ill-conditioned
there and any precision would show it.

This separates them by transplanting the displacement. Let

    δ = x₆₄ − x₁     the anchor move the width change produced, in the native path

and read the derivative in **coherent float32**, at width one, at its own anchor and at that anchor
plus δ. If the float32 derivative moves by the order of the displacement, the sensitivity is
bfloat16's. If it moves by tens of per cent, the function is ill-conditioned there and the earlier
result is a property of the model rather than of the arithmetic.

Three further readings ride along, because each is one more forward and each rules something out:
the same displacement applied in **bfloat16 at fixed width one**, which is the anchor term measured
without changing the schedule at all; a **random** displacement of identical norm in float32, which
says whether any perturbation of that size does this or only this one; and the float32 derivative at
the transplanted bf16 anchor, which moves the base point as well as the step.

    python displacement_control.py SNAPSHOT CORPUS OUT [--wide 64]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("snapshot")
parser.add_argument("corpus", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--wide", type=int, default=64)
parser.add_argument("--layers", default="1,17,33")
args = parser.parse_args()

OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
STARTED = time.monotonic()
MAX_SEQ, SEED, DECLARED_ROWS, POSITION, DIRECTION_SEED = 128, 20260910, 3, 8, 20260911


def emit(event: str, /, **fields) -> dict:
    fields.pop("event", None)
    row = {"event": event, "elapsed_s": round(time.monotonic() - STARTED, 2), **fields}
    print(json.dumps(row, default=str), flush=True)
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
        stream.flush()
    return row


from local_llm_lab import device  # noqa: E402

device.pin(seed=0)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.finite_difference import _replace_output  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.upstream import (  # noqa: E402
    CorpusLensModel, load_upstream, upstream_index_of_repo_layer,
)

model, report = hf_text.load_text_causal_lm(
    args.snapshot, dtype="bfloat16", attn_implementation="eager", device="cuda:0"
)
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.eval()

load_upstream()
import jlens.hf as upstream_hf  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

wrapped = CorpusLensModel(upstream_hf.HFLensModel(model, tokenizer=None))
n_layers, d_model = int(wrapped.n_layers), int(wrapped.d_model)
target = n_layers - 1
repo_layers = tuple(int(x) for x in args.layers.split(","))
sources = {repo: upstream_index_of_repo_layer(repo) for repo in repo_layers}

held = [row for row in read_corpus(args.corpus) if row.get("split") == "held"]
row = held[random.Random(SEED).sample(range(len(held)), DECLARED_ROWS)[0]]
key = wrapped.register(f"row:{row['index']}", row["ids"])
ids = wrapped.encode(key, max_length=MAX_SEQ)
seq_len = int(ids.shape[-1])
mask = torch.zeros(seq_len, dtype=torch.bool, device=ids.device)
mask[8] = True
mask[seq_len - 1] = True

generator = torch.Generator(device="cpu").manual_seed(DIRECTION_SEED)
directions = [("coordinate", i, torch.nn.functional.one_hot(torch.tensor(i), d_model).float())
              for i in (0, 137, 1279, d_model - 1)]
for j in range(2):
    v = torch.randn(d_model, generator=generator)
    directions.append(("dense", j, v / v.norm()))
cotangents = []
w0 = torch.zeros(d_model)
w0[7] = 1.0
cotangents.append(("coordinate", 0, w0))
for j in range(2):
    w = torch.randn(d_model, generator=generator)
    cotangents.append(("dense", j, w / w.norm()))


def anchors_at(width):
    batched = ids if width == 1 else ids.expand(width, -1)
    with torch.no_grad():
        with ActivationRecorder(wrapped.layers, at=[*sources.values()]) as recorder:
            wrapped.forward(batched)
        return {layer: recorder.activations[layer][:1].detach().clone()
                for layer in sources.values()}


def jt_w(layer, anchor):
    """(Jᵀw) at the source position, width one, reading whatever precision the model is in now."""
    out = {}
    for kind, index, w in cotangents:
        leaf = anchor.detach().clone().float().requires_grad_(True)
        handle = wrapped.layers[layer].register_forward_hook(
            lambda m, i, o, t=leaf: _replace_output(t.to(anchor.dtype), o)
        )
        try:
            with ActivationRecorder(wrapped.layers, at=[target]) as inner:
                wrapped.forward(ids)
                scalar = (inner.activations[target].float()[0][mask].sum(dim=0)
                          * w.to("cuda:0")).sum()
        finally:
            handle.remove()
        grad, = torch.autograd.grad(scalar, leaf)
        out[(kind, index)] = grad[0, POSITION].detach().float().cpu()
    return out


# δ comes from the native path: it is the anchor move the width change actually produced.
native_1, native_wide = anchors_at(1), anchors_at(args.wide)
delta = {layer: (native_wide[layer].float() - native_1[layer].float())
         for layer in sources.values()}
random_delta = {}
for layer, d in delta.items():
    noise = torch.randn(d.shape, generator=generator).to(d.device)
    scale = torch.linalg.vector_norm(d) / torch.linalg.vector_norm(noise)
    random_delta[layer] = noise * scale

readings: dict[tuple, dict] = {}
for phase in ("native", "float32"):
    if phase == "float32":
        model.to(torch.float32)
        emit("promoted", dtype=str(next(model.parameters()).dtype))
    own = anchors_at(1)
    for repo_layer, layer in sources.items():
        base = own[layer]
        moved = (base.float() + delta[layer]).to(base.dtype)
        moved_random = (base.float() + random_delta[layer]).to(base.dtype)
        readings[(phase, repo_layer, "base")] = jt_w(layer, base)
        readings[(phase, repo_layer, "plus_delta")] = jt_w(layer, moved)
        readings[(phase, repo_layer, "plus_random")] = jt_w(layer, moved_random)
        if phase == "float32":
            readings[(phase, repo_layer, "native_anchor")] = jt_w(layer, native_1[layer].float())
        rel = float(torch.linalg.vector_norm(delta[layer][0, POSITION])
                    / torch.linalg.vector_norm(base.float()[0, POSITION]))
        emit("read", phase=phase, repo_layer=repo_layer, relative_displacement=rel,
             basis="measured-here")

rows_out = []
for phase in ("native", "float32"):
    for repo_layer in repo_layers:
        for dkind, dindex, v in directions:
            for ckind, cindex, _ in cotangents:
                def a(which):
                    entry = readings.get((phase, repo_layer, which))
                    return None if entry is None else float((entry[(ckind, cindex)] * v).sum())
                base = a("base")
                rows_out.append({
                    "phase": phase, "repo_layer": repo_layer,
                    "direction": f"{dkind}:{dindex}", "cotangent": f"{ckind}:{cindex}",
                    "a_base": base, "a_plus_delta": a("plus_delta"),
                    "a_plus_random": a("plus_random"),
                    "a_at_native_anchor": a("native_anchor"),
                    "delta_relative": (abs(a("plus_delta") - base) / abs(base)) if base else None,
                    "random_relative": (abs(a("plus_random") - base) / abs(base)) if base else None,
                    "basis": "measured-here",
                })

with (OUT / "displacement.jsonl").open("w", encoding="utf-8") as stream:
    for r in rows_out:
        stream.write(json.dumps(r) + "\n")
emit("done", cells=len(rows_out))
