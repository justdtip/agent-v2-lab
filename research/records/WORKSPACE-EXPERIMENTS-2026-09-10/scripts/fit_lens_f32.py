"""Chief's overnight lens fit: the exact (autograd) Jacobian lens at every layer, coherent float32,
over the corpus's fit split, at a declared dim_batch. Writes exact-maps.npz (J{repo_layer}), nu.json
(declare_nu) and manifest.json under OUT, with per-row progress and a projected finish.

usage: fit_lens_f32.py SNAPSHOT CORPUS_MANIFEST OUT --dim-batch W [--max-rows N] [--label NAME]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("snapshot"); ap.add_argument("corpus"); ap.add_argument("out")
ap.add_argument("--dim-batch", type=int, required=True)
ap.add_argument("--max-rows", type=int, default=None)
ap.add_argument("--rows-from", type=int, default=0)
ap.add_argument("--rows-to", type=int, default=None)
ap.add_argument("--label", default="chief-overnight")
ap.add_argument("--max-seq-len", type=int, default=128)
ap.add_argument("--peak-abort-gib", type=float, default=88.0)
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
STARTED = time.monotonic()

def emit(event, /, **fields):
    row = {"event": event, "elapsed_s": round(time.monotonic() - STARTED, 2), **fields}
    print(json.dumps(row, default=str), flush=True)
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row

def write_json(name, payload):
    p = OUT / name
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    emit("wrote", file=name, bytes=p.stat().st_size)

from local_llm_lab import device  # noqa: E402
device.pin(seed=0)
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.upstream import (  # noqa: E402
    CorpusLensModel, declare_nu, fit_upstream_jacobian, load_upstream, repo_layer_of_upstream,
)

t = time.monotonic()
model, report = hf_text.load_text_causal_lm(
    args.snapshot, dtype="bfloat16", attn_implementation="eager", device="cuda:0")
for p in model.parameters():
    p.requires_grad_(False)
model.eval()
model.to(torch.float32)  # the same stored values cast up: the coherent float32 model
emit("loaded", seconds=round(time.monotonic() - t, 2), dtype=str(next(model.parameters()).dtype),
     attn=report["attn_implementation"], checkpoint_sha256=report.get("sha256", {}).get("config.json"),
     tf32=torch.backends.cuda.matmul.allow_tf32, matmul_precision=torch.get_float32_matmul_precision(),
     weights_gib=round(torch.cuda.memory_allocated() / 2**30, 3))

up = load_upstream()
import jlens.hf as upstream_hf  # noqa: E402
wrapped = CorpusLensModel(upstream_hf.HFLensModel(model, tokenizer=None))
n_layers, d_model = int(wrapped.n_layers), int(wrapped.d_model)

rows = [r for r in read_corpus(Path(args.corpus)) if r.get("split") == "fit"]
rows = rows[args.rows_from: args.rows_to]
if args.max_rows:
    rows = rows[: args.max_rows]
corpus_declaration = {"manifest": str(args.corpus), "split": "fit",
                      "rows": [int(r["index"]) for r in rows]}
write_json("manifest.json", {
    "schema_version": 1, "seat": "chief", "label": args.label,
    "stage": "exact autograd lens at every layer, coherent float32, over the fit split",
    "checkpoint": args.snapshot, "load_report_sha256": report.get("sha256"),
    "corpus": corpus_declaration, "n_rows": len(rows), "rows_from": args.rows_from, "rows_to": args.rows_to, "max_seq_len": args.max_seq_len,
    "dim_batch": args.dim_batch, "forward_batch": args.dim_batch, "anchor_batch": args.dim_batch,
    "precision": "float32", "n_layers": n_layers, "d_model": d_model,
    "tf32": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "float32_matmul_precision": torch.get_float32_matmul_precision()},
    "device": device.describe(),
    "basis": "chief overnight fit for the J-space heat map; the D-CRO's canonical refit follows their queue",
})

state = {"n": 0, "t0": time.monotonic()}
def progress(rec):
    if rec.get("event") != "prompt":
        return
    state["n"] = rec.get("n_done", state["n"] + 1)
    el = time.monotonic() - state["t0"]
    per = el / max(state["n"], 1)
    remaining = per * (len(rows) - state["n"])
    peak = torch.cuda.max_memory_allocated() / 2**30
    emit("row", n_done=state["n"], of=len(rows), s_per_row=round(per, 1),
         eta_min=round(remaining / 60, 1), peak_gib=round(peak, 2))
    if peak > args.peak_abort_gib:
        raise SystemExit(f"peak {peak:.1f} GiB exceeds the abort line {args.peak_abort_gib}")

torch.cuda.reset_peak_memory_stats()
t = time.monotonic()
fit = fit_upstream_jacobian(
    wrapped, rows, position_selector=None, max_seq_len=args.max_seq_len, dim_batch=args.dim_batch,
    device="cuda:0", dtype="float32", split="fit", upstream=up,
    source_layers=None, target_layer=n_layers - 1, max_rows=None, progress=progress,
)
emit("fit_done", seconds=round(time.monotonic() - t, 1), n_prompts=fit.n_prompts,
     layers=len(fit.jacobians), skipped=len(fit.skipped),
     peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3), basis="measured-here")
nu = declare_nu(fit, num_layers=n_layers, corpus=corpus_declaration)
maps = {repo_layer_of_upstream(i): np.asarray(m, np.float32) for i, m in fit.jacobians.items()}
np.savez_compressed(OUT / "exact-maps.npz", **{f"J{i}": m for i, m in maps.items()})
emit("wrote", file="exact-maps.npz", layers=sorted(maps), bytes=(OUT / "exact-maps.npz").stat().st_size)
write_json("nu.json", nu)
write_json("per_prompt.json", {"per_prompt": fit.per_prompt, "skipped": fit.skipped})
emit("done", total_minutes=round((time.monotonic() - STARTED) / 60, 2))
