"""Workspace experiments, set 1 — the capture pass (WORKSPACE-EXPERIMENTS-2026-09-10.md §0).

Per distinct decision of the rendered agent corpus: the residual at every layer at P_note (the last
prompt token) and P_act (the token before the tool-name token in the completion's JSON), float32,
into memmaps; the model's own six-tool distribution and top-5 at both positions; the lens and
logit-lens six-tool distributions per layer at both positions (reduced). For the stratified sample:
attention from both positions at every layer/head onto the prompt, with span tags; and the lens
six-tool readout at every note token.

usage: workspace_capture.py SNAPSHOT CORPUS_JSONL MAPS_NPZ OUT_DIR --model 4b|12b [--device cuda:0]
       [--max-rows N] [--sample-per-family 25] [--sample-max-tokens 1500] [--seed 20260910]
"""
from __future__ import annotations
import argparse, json, hashlib, re, time, random
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("snapshot"); ap.add_argument("corpus"); ap.add_argument("maps"); ap.add_argument("out")
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max-rows", type=int, default=None)
ap.add_argument("--sample-per-family", type=int, default=25)
ap.add_argument("--sample-max-tokens", type=int, default=1500)
ap.add_argument("--seed", type=int, default=20260910)
ap.add_argument("--skip-sample", action="store_true")
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True); T0 = time.monotonic()
TOOLS = ["read_file", "finish", "replace_text", "calculate", "search_files", "list_files"]

def emit(event, /, **f):
    row = {"event": event, "elapsed_s": round(time.monotonic() - T0, 1), **f}
    print(json.dumps(row, default=str), flush=True)
    with (OUT / "progress.jsonl").open("a") as fh: fh.write(json.dumps(row, default=str) + "\n")

from local_llm_lab import device  # noqa: E402
device.pin(seed=0)
import torch  # noqa: E402
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
from transformers import AutoTokenizer  # noqa: E402
from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.upstream import upstream_index_of_repo_layer  # noqa: E402
import jlens.hf as upstream_hf  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

# ---- corpus: distinct decisions in file order (train, valid, test), keyed (task_id, step)
rows, seen = [], set()
for line in open(args.corpus):
    r = json.loads(line); m = r.get("metadata", {})
    if "task_id" not in m: continue
    k = (m["task_id"], m["step"])
    if k in seen: continue
    seen.add(k); rows.append(r)
if args.max_rows: rows = rows[: args.max_rows]
emit("corpus", decisions=len(rows), corpus_sha256=hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest())

model, report = hf_text.load_text_causal_lm(args.snapshot, dtype="bfloat16", attn_implementation="eager", device=args.device)
for p in model.parameters(): p.requires_grad_(False)
model.eval(); model.to(torch.float32)
tok = AutoTokenizer.from_pretrained(args.snapshot)
lens_model = upstream_hf.HFLensModel(model, tokenizer=tok)
n_layers, d_model = int(lens_model.n_layers), int(lens_model.d_model)
z = np.load(args.maps); repo_layers = sorted(int(k[1:]) for k in z.files)
maps = {r: torch.from_numpy(z[f"J{r}"]).to(args.device) for r in repo_layers}
record_at = sorted(set(upstream_index_of_repo_layer(r) for r in repo_layers) | {n_layers - 1})
tool_first = {t: tok(t, add_special_tokens=False).input_ids[0] for t in TOOLS}
_cfg = getattr(model.config, "text_config", model.config)
LAYER_TYPES = list(getattr(_cfg, "layer_types", []) or [])
IS_GLOBAL = [not bool(lay.self_attn.is_sliding) for lay in lens_model.layers]
if LAYER_TYPES:
    assert [t == "full_attention" for t in LAYER_TYPES] == IS_GLOBAL, "config layer_types disagree with the modules' is_sliding"
SLIDING_WINDOW = int(getattr(_cfg, "sliding_window", 0) or 0)
emit("loaded", dtype=str(next(model.parameters()).dtype), attn=report["attn_implementation"], layers=n_layers, d_model=d_model, global_layers=[i + 1 for i, g in enumerate(IS_GLOBAL) if g], sliding_window=SLIDING_WINDOW,
     maps_sha256=hashlib.sha256(Path(args.maps).read_bytes()).hexdigest(), tool_first_tokens=tool_first,
     tf32=torch.backends.cuda.matmul.allow_tf32, matmul=torch.get_float32_matmul_precision())

N = len(rows); L = len(repo_layers)
res_note = np.lib.format.open_memmap(OUT / "residual_note.npy", mode="w+", dtype=np.float32, shape=(N, L, d_model))
res_act = np.lib.format.open_memmap(OUT / "residual_act.npy", mode="w+", dtype=np.float32, shape=(N, L, d_model))
index, name_pat = [], re.compile(r'"name":\s*"([a-z_]+)"')
tool_ids = torch.tensor([tool_first[t] for t in TOOLS], device=args.device)

def six(logits):  # logits [..., vocab] -> probs over the six tool-first tokens, renormalised, plus raw mass
    p = torch.softmax(logits.float(), dim=-1); q = p[..., tool_ids]; return q / q.sum(-1, keepdim=True).clamp_min(1e-30), q.sum(-1)

def readouts(h_by_layer, pos):  # dict upstream-index -> [seq, d]; returns per-layer six-tool dists for lens and logit lens at pos
    out = {"lens": {}, "logit": {}}
    for r in repo_layers:
        h = h_by_layer[upstream_index_of_repo_layer(r)][pos]
        lp, lm = six(lens_model.unembed(h @ maps[r].T)); gp, gm = six(lens_model.unembed(h))
        out["lens"][str(r)] = {"six": lp.tolist(), "mass": float(lm)}; out["logit"][str(r)] = {"six": gp.tolist(), "mass": float(gm)}
    return out

def spans(prompt):  # character spans by kind, every turn tagged
    tags = []
    m = re.search(r"<start_of_turn>user\n", prompt); first_model = prompt.find("<start_of_turn>model")
    if m: tags.append(("task", m.end(), first_model if first_model > 0 else len(prompt)))
    results = []
    for mm in re.finditer(r"<tool_response[^>]*>\n(.*?)\n</tool_response>", prompt, re.S):
        body = mm.group(1)
        if body.startswith("[earlier"): tags.append(("hidden_result", mm.start(1), mm.end(1)))
        else: results.append((mm.start(1), mm.end(1)))
    for k, (a, b) in enumerate(results): tags.append(("latest_result" if k == len(results) - 1 else "older_result", a, b))
    turns = list(re.finditer(r"<start_of_turn>model\n(.*?)```json\n(.*?)```", prompt, re.S))
    for k, mm in enumerate(turns):
        last = k == len(turns) - 1
        tags.append(("previous_note" if last else "older_note", mm.start(1), mm.end(1)))
        tags.append(("previous_call" if last else "older_call", mm.start(2), mm.end(2)))
    return tags

# ---- the sample, by rule: per family, sorted (task_id, step), decisions with prompt <= max tokens, every k-th to get n
fam = {}
for i, r in enumerate(rows): fam.setdefault(r["metadata"]["family"], []).append(i)
sample = set()
if not args.skip_sample:
    for f, idxs in sorted(fam.items()):
        idxs = sorted(idxs, key=lambda i: (rows[i]["metadata"]["task_id"], rows[i]["metadata"]["step"]))
        short = [i for i in idxs if len(tok(rows[i]["prompt"], add_special_tokens=False).input_ids) <= args.sample_max_tokens]
        step = max(1, len(short) // args.sample_per_family); sample.update(short[::step][: args.sample_per_family])
emit("sample", n=len(sample), per_family=args.sample_per_family, max_tokens=args.sample_max_tokens)
sample_out = (OUT / "sample.jsonl").open("w")

with torch.no_grad():
    for i, r in enumerate(rows):
        m = r["metadata"]; prompt, comp = r["prompt"], r["completion"]
        enc_p = tok(prompt, add_special_tokens=False, return_offsets_mapping=True); p_ids = enc_p.input_ids  # the rendered prompt carries its own <bos>
        assert p_ids[0] == tok.bos_token_id and p_ids[1] != tok.bos_token_id, "the prompt must begin with exactly one <bos>"
        mm = name_pat.search(comp); tool = mm.group(1)
        enc_c = tok(comp, add_special_tokens=False, return_offsets_mapping=True)
        name_char = mm.start(1)
        t_idx = next(j for j, (a, b) in enumerate(enc_c.offset_mapping) if a <= name_char < b)  # the tool-name token in the completion
        c_ids = enc_c.input_ids[: t_idx + 1]
        ids = torch.tensor([p_ids + c_ids], device=args.device)
        P_note = len(p_ids) - 1; P_act = len(p_ids) + t_idx - 1
        in_sample = i in sample
        with ActivationRecorder(lens_model.layers, at=record_at) as rec:
            out = model(input_ids=ids, output_attentions=in_sample)
            acts = {k: rec.activations[k].detach()[0].float() for k in record_at}
        for j, rr in enumerate(repo_layers):
            u = upstream_index_of_repo_layer(rr); res_note[i, j] = acts[u][P_note].cpu().numpy(); res_act[i, j] = acts[u][P_act].cpu().numpy()
        logits = out.logits[0].float()
        s_note, m_note = six(logits[P_note]); s_act, m_act = six(logits[P_act])
        top5 = lambda v: [tok.decode([int(t)]) for t in torch.topk(v, 5).indices.tolist()]
        rec_row = {"i": i, "task_id": m["task_id"], "step": m["step"], "family": m["family"], "variant": m["variant"], "recovery": m.get("recovery"),
                   "tool": tool, "tool_idx": TOOLS.index(tool), "P_note": P_note, "P_act": P_act, "n_prompt_tokens": len(p_ids), "n_note_tokens": t_idx,
                   "model_six_note": s_note.tolist(), "model_six_act": s_act.tolist(), "model_mass_note": float(m_note), "model_mass_act": float(m_act),
                   "model_top5_note": top5(logits[P_note]), "model_top5_act": top5(logits[P_act]),
                   "lens_note": readouts(acts, P_note), "lens_act": readouts(acts, P_act), "in_sample": in_sample}
        index.append(rec_row)
        if in_sample:
            # attention from both positions, every layer/head, onto every key position; spans by kind
            att = [a[0, :, [P_note, P_act], :].float().cpu().numpy() for a in out.attentions]  # per layer [heads, 2, seq]
            offs = enc_p.offset_mapping; tags = spans(prompt); kind = ["format"] * len(ids[0])
            for k, (a, b) in enumerate(offs):
                for name, s0, s1 in tags:
                    if a >= s0 and b <= s1: kind[k] = name
            for k in range(len(p_ids), len(ids[0])): kind[k] = "note"
            note_traj = {str(rr): [] for rr in repo_layers}; note_mass = {str(rr): [] for rr in repo_layers}
            for pos in range(len(p_ids) - 1, len(p_ids) + t_idx):  # every note token's readout (position predicting the next token)
                ro = readouts(acts, pos)
                for rr in repo_layers: note_traj[str(rr)].append(ro["lens"][str(rr)]["six"]); note_mass[str(rr)].append(ro["lens"][str(rr)]["mass"])
            np.save(OUT / f"attn_{i}.npy", np.stack(att).astype(np.float16))  # [layers, heads, 2, seq]
            sample_out.write(json.dumps({"i": i, "kinds": kind, "note_trajectory_lens_six": note_traj, "note_trajectory_lens_mass": note_mass,
                                         "is_global_layer": IS_GLOBAL}) + "\n"); sample_out.flush()
        if i % 100 == 0 or i == N - 1:
            emit("row", n_done=i + 1, of=N, s_per_row=round((time.monotonic() - T0) / (i + 1), 2), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2) if args.device.startswith("cuda") else None)
res_note.flush(); res_act.flush(); sample_out.close()
(OUT / "index.jsonl").write_text("\n".join(json.dumps(x) for x in index) + "\n")
(OUT / "manifest.json").write_text(json.dumps({"schema_version": 1, "seat": "chief", "model": args.model, "checkpoint": args.snapshot, "load_report_sha256": report.get("sha256"),
    "corpus": args.corpus, "decisions": N, "positions": ["P_note: last prompt token", "P_act: token before the tool-name token"], "precision": "float32", "width": 1,
    "device": args.device, "maps": args.maps, "repo_layers": repo_layers, "tools": TOOLS, "tool_first_tokens": tool_first,
    "sample": sorted(sample), "sample_rule": f"per family, sorted (task_id, step), prompts <= {args.sample_max_tokens} tokens, every k-th to {args.sample_per_family}",
    "tf32": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "float32_matmul_precision": torch.get_float32_matmul_precision()},
    "design": "WORKSPACE-EXPERIMENTS-2026-09-10.md", "layer_types": LAYER_TYPES, "is_global_layer": IS_GLOBAL, "sliding_window": SLIDING_WINDOW}, indent=2) + "\n")
emit("done", decisions=N, sample=len(sample), minutes=round((time.monotonic() - T0) / 60, 1))
