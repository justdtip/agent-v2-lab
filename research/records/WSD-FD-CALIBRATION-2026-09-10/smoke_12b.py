"""The 12B smoke row in float32 at width one, and the same-anchor control on the larger model.

Two numbers everything after this is planned on: whether a coherent float32 Gemma 3 12B fits on the
card with an autograd graph beside it, and what an exact fit costs there. The weights alone are about
48 GB in float32 against 95 GiB of card, so `dim_batch` starts at one and the peak is measured rather
than projected — every memory figure in this registry is re-measured per device and this is no
exception.

The same-anchor control from the 4B is repeated here because its answer was depth-dependent there —
almost all anchor at deep sources, arithmetic dominating at the shallowest — and a model with 48
blocks rather than 34 is the obvious place to ask whether that pattern is about depth.

    python smoke_12b.py SNAPSHOT CORPUS OUT [--wide W]
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
parser.add_argument("--wide", type=int, default=8, help="the second width for the anchor control")
parser.add_argument("--layers", default="1,24,47", help="repo layers to probe")
parser.add_argument("--precision", default="float32", choices=("float32", "native"),
                    help="native keeps the loaded bf16; the same-anchor control in float32 "
                         "cannot answer a question about bf16, because in float32 the two "
                         "anchors barely differ and both terms vanish by construction")
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
    CorpusLensModel, declare_nu, fit_upstream_jacobian, load_upstream,
    upstream_index_of_repo_layer,
)

t = time.monotonic()
model, report = hf_text.load_text_causal_lm(
    args.snapshot, dtype="bfloat16", attn_implementation="eager", device="cuda:0"
)
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.eval()
emit("loaded_bf16", seconds=round(time.monotonic() - t, 2),
     allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 3),
     checkpoint_sha256=report.get("sha256", {}).get("config.json"))

t = time.monotonic()
if args.precision == "float32":
    model.to(torch.float32)
torch.cuda.reset_peak_memory_stats()
emit("promoted_float32" if args.precision == "float32" else "kept_native",
     seconds=round(time.monotonic() - t, 2),
     allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 3),
     free_gib=round(torch.cuda.mem_get_info()[0] / 2**30, 3),
     dtype=str(next(model.parameters()).dtype), basis="measured-here")

up = load_upstream()
import jlens.hf as upstream_hf  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

wrapped = CorpusLensModel(upstream_hf.HFLensModel(model, tokenizer=None))
n_layers, d_model = int(wrapped.n_layers), int(wrapped.d_model)
target = n_layers - 1
repo_layers = tuple(int(x) for x in args.layers.split(","))
sources = {repo: upstream_index_of_repo_layer(repo) for repo in repo_layers}
emit("wrapped", n_layers=n_layers, d_model=d_model, repo_layers=list(repo_layers),
     upstream_layers=sources, target_upstream=target)

held = [row for row in read_corpus(args.corpus) if row.get("split") == "held"]
row = held[random.Random(SEED).sample(range(len(held)), DECLARED_ROWS)[0]]
key = wrapped.register(f"row:{row['index']}", row["ids"])
ids = wrapped.encode(key, max_length=MAX_SEQ)
seq_len = int(ids.shape[-1])
mask = torch.zeros(seq_len, dtype=torch.bool, device=ids.device)
mask[8] = True
mask[seq_len - 1] = True


def two_positions(n):
    m = torch.zeros(n, dtype=torch.bool)
    m[8] = True
    m[n - 1] = True
    return m


# ------------------------------------------------------------------------ the smoke row itself
torch.cuda.reset_peak_memory_stats()
t = time.monotonic()
smoke = fit_upstream_jacobian(
    wrapped, [row], position_selector=two_positions, max_seq_len=MAX_SEQ, dim_batch=1,
    device="cuda:0", dtype=args.precision if args.precision != "native" else "bfloat16",
    split="held", upstream=up,
    source_layers=[sources[repo_layers[-1]]], target_layer=target, max_rows=1,
)
nu = declare_nu(smoke, num_layers=n_layers, corpus={"manifest": str(args.corpus), "split": "held"})
emit("smoke", seconds=round(time.monotonic() - t, 2),
     peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
     free_after_gib=round(torch.cuda.mem_get_info()[0] / 2**30, 3),
     repo_layer=repo_layers[-1], dim_batch=1, d_model=d_model,
     forward_batch=nu["precision"]["forward_batch"],
     anchor_batch=nu["precision"]["anchor_batch"], basis="measured-here")

# ----------------------------------------------------------------- the same-anchor control, W2
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


def jt_w_at(layer, anchor, width):
    batched = ids if width == 1 else ids.expand(width, -1)
    out = {}
    for kind, index, w in cotangents:
        leaf = anchor.detach().clone().float().requires_grad_(True)
        wide = leaf if width == 1 else leaf.expand(width, -1, -1)
        handle = wrapped.layers[layer].register_forward_hook(
            lambda m, i, o, tensor=wide.to(anchor.dtype): _replace_output(tensor, o)
        )
        try:
            with ActivationRecorder(wrapped.layers, at=[target]) as inner:
                wrapped.forward(batched)
                scalar = (inner.activations[target].float()[0][mask].sum(dim=0)
                          * w.to("cuda:0")).sum()
        finally:
            handle.remove()
        grad, = torch.autograd.grad(scalar, leaf)
        out[(kind, index)] = grad[0, POSITION].detach().float().cpu()
    return out


anchor_1, anchor_wide = anchors_at(1), anchors_at(args.wide)
rows_out = []
for repo_layer, layer in sources.items():
    displacement = float(torch.linalg.vector_norm(
        (anchor_wide[layer].float() - anchor_1[layer].float())[0, POSITION]))
    scale = float(torch.linalg.vector_norm(anchor_1[layer].float()[0, POSITION]))
    emit("anchors", repo_layer=repo_layer, anchor_displacement_norm=displacement,
         anchor_norm=scale, relative_displacement=displacement / scale if scale else None,
         width=args.wide, basis="measured-here")
    g = {("w1", "x1"): jt_w_at(layer, anchor_1[layer], 1),
         ("wW", "x1"): jt_w_at(layer, anchor_1[layer], args.wide),
         ("wW", "xW"): jt_w_at(layer, anchor_wide[layer], args.wide)}
    for dkind, dindex, v in directions:
        for ckind, cindex, _ in cotangents:
            a = {k: float((g[k][(ckind, cindex)] * v).sum()) for k in g}
            base = abs(a[("w1", "x1")])
            rows_out.append({
                "repo_layer": repo_layer, "width": args.wide,
                "direction": f"{dkind}:{dindex}", "cotangent": f"{ckind}:{cindex}",
                "observed_shift": a[("wW", "xW")] - a[("w1", "x1")],
                "anchor_term": a[("wW", "xW")] - a[("wW", "x1")],
                "arithmetic_term": a[("wW", "x1")] - a[("w1", "x1")],
                "observed_relative": (a[("wW", "xW")] - a[("w1", "x1")]) / base if base else None,
                "anchor_relative": (a[("wW", "xW")] - a[("wW", "x1")]) / base if base else None,
                "arithmetic_relative": (a[("wW", "x1")] - a[("w1", "x1")]) / base if base else None,
                "identity_residual": abs((a[("wW", "xW")] - a[("w1", "x1")])
                                         - ((a[("wW", "xW")] - a[("wW", "x1")])
                                            + (a[("wW", "x1")] - a[("w1", "x1")]))),
                "basis": "measured-here",
            })
    emit("layer_done", repo_layer=repo_layer, cells=len(rows_out),
         peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3))

with (OUT / f"same-anchor-12b-{args.precision}.jsonl").open("w", encoding="utf-8") as stream:
    for r in rows_out:
        stream.write(json.dumps(r) + "\n")
emit("done", cells=len(rows_out),
     max_identity_residual=max(r["identity_residual"] for r in rows_out),
     peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3))
