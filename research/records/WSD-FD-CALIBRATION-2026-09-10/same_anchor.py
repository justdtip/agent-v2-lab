"""Does the batch width change the derivative, or does it change the point it is read at?

The width rows measured that the native-path autograd value moves by a median 76% at repo layer 1
when the forward width goes from 1 to 64. That is a measurement of `a₆₄(x₆₄) − a₁(x₁)`, and it moves
two things at once: the arithmetic of the forward, and the anchor, because the width-64 forward
produces a different residual at the source. A smooth function can move its derivative by 76% between
two anchors 0.076% apart, so the observed shift does not attribute itself.

The protocol's same-anchor control separates them into two brackets that sum to the observed shift:

    a₆₄(x₆₄) − a₁(x₁)  =  [a₆₄(x₆₄) − a₆₄(x₁)]  +  [a₆₄(x₁) − a₁(x₁)]
                            the anchor term          the arithmetic term
                          same width, two anchors    same anchor, two widths

Per projection, both vectors kept. Nothing is differenced against a reference that moved.

    python same_anchor.py SNAPSHOT CORPUS OUT
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

SNAPSHOT, CORPUS, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
STARTED = time.monotonic()

MAX_SEQ, SEED, DECLARED_ROWS = 128, 20260910, 3
REPO_LAYERS = (1, 17, 33)
WIDE = 64
POSITION = 8
DIRECTION_SEED = 20260911


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
    SNAPSHOT, dtype="bfloat16", attn_implementation="eager", device="cuda:0"
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
sources = {repo: upstream_index_of_repo_layer(repo) for repo in REPO_LAYERS}

held = [row for row in read_corpus(CORPUS) if row.get("split") == "held"]
row = held[random.Random(SEED).sample(range(len(held)), DECLARED_ROWS)[0]]
key = wrapped.register(f"row:{row['index']}", row["ids"])
ids = wrapped.encode(key, max_length=MAX_SEQ)
seq_len = int(ids.shape[-1])
mask = torch.zeros(seq_len, dtype=torch.bool, device=ids.device)
mask[8] = True
mask[seq_len - 1] = True

generator = torch.Generator(device="cpu").manual_seed(DIRECTION_SEED)
directions = [("coordinate", i, torch.nn.functional.one_hot(torch.tensor(i), d_model).float())
              for i in (0, 137, 1279, 2559)]
for j in range(2):
    v = torch.randn(d_model, generator=generator); directions.append(("dense", j, v / v.norm()))
cotangents = []
w0 = torch.zeros(d_model); w0[7] = 1.0
cotangents.append(("coordinate", 0, w0))
for j in range(2):
    w = torch.randn(d_model, generator=generator); cotangents.append(("dense", j, w / w.norm()))

emit("frozen", row_index=row.get("index"), seq_len=seq_len, widths=[1, WIDE],
     repo_layers=list(REPO_LAYERS), position=POSITION, direction_seed=DIRECTION_SEED)


def anchors_at(width):
    batched = ids if width == 1 else ids.expand(width, -1)
    with torch.no_grad():
        with ActivationRecorder(wrapped.layers, at=[*sources.values()]) as recorder:
            wrapped.forward(batched)
        return {layer: recorder.activations[layer][:1].detach().clone()
                for layer in sources.values()}


def jt_w_at(layer, anchor_one_row, width):
    """(Jᵀw) at the source position, reading the width-`width` forward at the given anchor.

    The anchor is one row and is broadcast to the width, so the *point* is held fixed while the
    schedule changes. That is the whole construction: without it, changing the width also changes
    the residual the width's own forward would have produced there.
    """
    batched = ids if width == 1 else ids.expand(width, -1)
    out = {}
    for kind, index, w in cotangents:
        leaf = anchor_one_row.detach().clone().float().requires_grad_(True)
        wide = leaf if width == 1 else leaf.expand(width, -1, -1)
        handle = wrapped.layers[layer].register_forward_hook(
            lambda m, i, o, t=wide.to(anchor_one_row.dtype): _replace_output(t, o)
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


anchor_1, anchor_wide = anchors_at(1), anchors_at(WIDE)
rows_out, vectors = [], {}
for repo_layer, layer in sources.items():
    displacement = float(torch.linalg.vector_norm(
        (anchor_wide[layer].float() - anchor_1[layer].float())[0, POSITION]))
    scale = float(torch.linalg.vector_norm(anchor_1[layer].float()[0, POSITION]))
    emit("anchors", repo_layer=repo_layer, anchor_displacement_norm=displacement,
         anchor_norm=scale, relative_displacement=displacement / scale if scale else None,
         basis="measured-here")

    g = {
        ("w1", "x1"): jt_w_at(layer, anchor_1[layer], 1),
        ("w64", "x1"): jt_w_at(layer, anchor_1[layer], WIDE),
        ("w64", "x64"): jt_w_at(layer, anchor_wide[layer], WIDE),
    }
    for dkind, dindex, v in directions:
        for ckind, cindex, _ in cotangents:
            a = {k: float((g[k][(ckind, cindex)] * v).sum()) for k in g}
            observed = a[("w64", "x64")] - a[("w1", "x1")]
            anchor_term = a[("w64", "x64")] - a[("w64", "x1")]
            arithmetic_term = a[("w64", "x1")] - a[("w1", "x1")]
            base = abs(a[("w1", "x1")])
            rows_out.append({
                "repo_layer": repo_layer, "direction": f"{dkind}:{dindex}",
                "cotangent": f"{ckind}:{cindex}",
                "a_w1_x1": a[("w1", "x1")], "a_w64_x1": a[("w64", "x1")],
                "a_w64_x64": a[("w64", "x64")],
                "observed_shift": observed,
                "anchor_term": anchor_term, "arithmetic_term": arithmetic_term,
                "observed_relative": observed / base if base else None,
                "anchor_relative": anchor_term / base if base else None,
                "arithmetic_relative": arithmetic_term / base if base else None,
                "identity_residual": abs(observed - (anchor_term + arithmetic_term)),
                "basis": "measured-here",
            })
    for k, cots in g.items():
        for ck, vec in cots.items():
            vectors[f"L{repo_layer}|{k[0]}|{k[1]}|{ck[0]}:{ck[1]}"] = vec.numpy()
    emit("layer_done", repo_layer=repo_layer, cells=len(rows_out))

np.savez_compressed(OUT / "jt_w.npz", **vectors)
with (OUT / "same-anchor.jsonl").open("w", encoding="utf-8") as stream:
    for r in rows_out:
        stream.write(json.dumps(r) + "\n")
emit("done", cells=len(rows_out), vectors=len(vectors),
     max_identity_residual=max(r["identity_residual"] for r in rows_out))
