"""W-3b: the carrier masked in the attention. For each sampled decision: the ordinary forward, then
forwards with the carrier spans masked as KEYS for every query at or after the current turn's start
(task statement never masked), one arm per kind and one for all kinds, plus a random-span mask of
equal token count. Reads at P_note and P_act: the model's six-tool distribution (with mass) and the
lens readout per layer. Implemented as a forward pre-hook on every attention module that adds the
extra masking to the layer's own attention_mask, so the sliding window is preserved; the gate checks
that attention on masked keys is zero and that local layers still place zero mass beyond the window.

usage: workspace_w3b.py SNAPSHOT CORPUS_JSONL MAPS_NPZ CAPTURE_DIR OUT_DIR [--device cuda:0] [--max-sample N] [--seed 20260910]
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("snapshot"); ap.add_argument("corpus"); ap.add_argument("maps"); ap.add_argument("capture"); ap.add_argument("out")
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--max-sample", type=int, default=None); ap.add_argument("--seed", type=int, default=20260910)
args = ap.parse_args(); OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True); T0 = time.monotonic()
TOOLS = ["read_file", "finish", "replace_text", "calculate", "search_files", "list_files"]
KINDS = ["older_note", "previous_note", "older_call", "previous_call", "older_result", "latest_result", "hidden_result"]
def emit(event, /, **f):
    row = {"event": event, "elapsed_s": round(time.monotonic() - T0, 1), **f}; print(json.dumps(row, default=str), flush=True)
    with (OUT / "progress.jsonl").open("a") as fh: fh.write(json.dumps(row, default=str) + "\n")

from local_llm_lab import device  # noqa: E402
device.pin(seed=0)
import torch  # noqa: E402
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
from transformers import AutoTokenizer  # noqa: E402
from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.upstream import upstream_index_of_repo_layer  # noqa: E402
import jlens.hf as upstream_hf  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
import importlib.util, sys as _sys
spec = importlib.util.spec_from_file_location("wcap", str(Path(__file__).with_name("workspace_capture.py")))
# reuse the span tagger by reading its source rather than importing the script (which runs a capture on import)
_src = Path(__file__).with_name("workspace_capture.py").read_text()
_ns = {"re": re}; exec(_src[_src.index("def spans(prompt):"): _src.index("# ---- the sample, by rule")], _ns); spans = _ns["spans"]

man = json.loads((Path(args.capture) / "manifest.json").read_text()); sample_ids = man["sample"]
if args.max_sample: sample_ids = sample_ids[: args.max_sample]
idx = {r["i"]: r for r in (json.loads(l) for l in (Path(args.capture) / "index.jsonl").read_text().splitlines() if l.strip())}
rows_all, seen = [], set()
for line in open(args.corpus):
    r = json.loads(line); m = r.get("metadata", {})
    if "task_id" not in m: continue
    k = (m["task_id"], m["step"])
    if k in seen: continue
    seen.add(k); rows_all.append(r)

model, report = hf_text.load_text_causal_lm(args.snapshot, dtype="bfloat16", attn_implementation="eager", device=args.device)
for p in model.parameters(): p.requires_grad_(False)
model.eval(); model.to(torch.float32)
tok = AutoTokenizer.from_pretrained(args.snapshot); lens_model = upstream_hf.HFLensModel(model, tokenizer=tok)
n_layers = int(lens_model.n_layers); z = np.load(args.maps); repo_layers = sorted(int(k[1:]) for k in z.files)
maps = {r: torch.from_numpy(z[f"J{r}"]).to(args.device) for r in repo_layers}
record_at = sorted(set(upstream_index_of_repo_layer(r) for r in repo_layers) | {n_layers - 1})
tool_ids = torch.tensor([tok(t, add_special_tokens=False).input_ids[0] for t in TOOLS], device=args.device)
IS_GLOBAL = [not bool(lay.self_attn.is_sliding) for lay in lens_model.layers]; W = int(getattr(getattr(model.config, "text_config", model.config), "sliding_window", 0) or 0)
rng = np.random.default_rng(args.seed)

# ---- the extra mask, applied by pre-hook on every attention module
STATE = {"extra": None}  # [1, 1, q, k] boolean: True = block
def pre_hook(module, args_, kwargs):
    extra = STATE["extra"]
    if extra is None: return None
    am = kwargs.get("attention_mask")
    if am is None: return None
    if am.dtype == torch.bool:
        kwargs["attention_mask"] = am & ~extra.to(am.device)
    else:
        neg = torch.finfo(am.dtype).min
        kwargs["attention_mask"] = torch.where(extra.to(am.device), torch.full_like(am, neg), am)
    return args_, kwargs
handles = [lay.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True) for lay in lens_model.layers]

def six(logits):
    p = torch.softmax(logits.float(), dim=-1); q = p[..., tool_ids]; return (q / q.sum(-1, keepdim=True).clamp_min(1e-30)).tolist(), float(q.sum(-1))
def readouts(acts, pos):
    out = {}
    for r in repo_layers:
        h = acts[upstream_index_of_repo_layer(r)][pos]; s, m = six(lens_model.unembed(h @ maps[r].T)); out[str(r)] = {"six": s, "mass": m}
    return out

def run(ids, extra, want_attn):
    STATE["extra"] = extra
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=record_at) as rec:
        out = model(input_ids=ids, output_attentions=want_attn)
        acts = {k: rec.activations[k].detach()[0].float() for k in record_at}
    STATE["extra"] = None
    return out, acts

results = []
LEAKY_CHECKED = False
for n, i in enumerate(sample_ids):
    r = rows_all[i]; m = r["metadata"]; rec_row = idx[i]; prompt, comp = r["prompt"], r["completion"]
    enc_p = tok(prompt, add_special_tokens=False, return_offsets_mapping=True); p_ids = enc_p.input_ids  # the rendered prompt carries its own <bos>
    assert p_ids[0] == tok.bos_token_id and p_ids[1] != tok.bos_token_id, "the prompt must begin with exactly one <bos>"
    enc_c = tok(comp, add_special_tokens=False, return_offsets_mapping=True); t_idx = rec_row["P_act"] - len(p_ids) + 1
    ids = torch.tensor([p_ids + enc_c.input_ids[: t_idx + 1]], device=args.device); S = ids.shape[1]
    P_note, P_act = rec_row["P_note"], rec_row["P_act"]
    kinds = ["format"] * S
    for k, (a, b) in enumerate(enc_p.offset_mapping):
        for name, s0, s1 in spans(prompt):
            if a >= s0 and b <= s1: kinds[k] = name
    for k in range(len(p_ids), S): kinds[k] = "note"
    turn_start = P_note  # queries from the decision position onward
    arms = {"unmasked": None}
    for kind in KINDS + ["all_carriers"]:
        keys = [k for k, kk in enumerate(kinds) if (kk == kind if kind != "all_carriers" else kk in KINDS)]
        if not keys: continue
        e = torch.zeros(1, 1, S, S, dtype=torch.bool); e[0, 0, turn_start:, keys] = True; arms[kind] = e
    # the current note, masked as keys for queries after it (the action position reads without what it just wrote)
    note_keys = [k for k, kk in enumerate(kinds) if kk == "note"]
    if note_keys:
        e = torch.zeros(1, 1, S, S, dtype=torch.bool); e[0, 0, note_keys[-1] + 1:, note_keys] = True; arms["current_note"] = e
        if "all_carriers" in arms:
            e2 = arms["all_carriers"].clone(); e2[0, 0, note_keys[-1] + 1:, note_keys] = True; arms["all_carriers_and_current_note"] = e2
    n_all = int(arms["all_carriers"][0, 0, turn_start].sum()) if "all_carriers" in arms else 0
    if n_all:
        candidates = [k for k, kk in enumerate(kinds) if kk not in ("task",) and k < turn_start]
        pick = rng.choice(candidates, size=min(n_all, len(candidates)), replace=False)
        e = torch.zeros(1, 1, S, S, dtype=torch.bool); e[0, 0, turn_start:, pick] = True; arms["random_equal_count"] = e
    row = {"i": i, "task_id": m["task_id"], "step": m["step"], "family": m["family"], "tool": rec_row["tool"], "tool_idx": rec_row["tool_idx"],
           "kind_token_counts": {k: kinds.count(k) for k in set(kinds)}, "arms": {}}
    if not LEAKY_CHECKED and "all_carriers" in arms:  # the cut verified with a deliberately leaky mask: one carrier key left open must receive attention
        leaky = arms["all_carriers"].clone(); keys = torch.nonzero(leaky[0, 0, P_act]).flatten().tolist(); leaky[0, 0, :, keys[0]] = False
        out_l, _ = run(ids, leaky, True); a = torch.stack([A[0, :, P_act, :].float().mean(0) for A in out_l.attentions]).sum(0)
        open_mass = float(a[keys[0]]); masked_mass = float(a[keys[1:]].sum()) if len(keys) > 1 else 0.0
        emit("leaky_mask_check", open_key_attention_sum_over_layers=open_mass, masked_keys_attention=masked_mass, passes=open_mass > 0.0 and masked_mass == 0.0)
        LEAKY_CHECKED = True
    for arm, extra in arms.items():
        want = True
        out, acts = run(ids, extra, want)
        lg = out.logits[0].float()
        def margin(v):  # taken tool's logit minus the best other tool's logit, in logits
            t = v[tool_ids]; own = float(t[rec_row["tool_idx"]]); others = torch.cat([t[:rec_row["tool_idx"]], t[rec_row["tool_idx"] + 1:]]); return own - float(others.max())
        rec_arm = {"model_six_note": six(lg[P_note]), "model_six_act": six(lg[P_act]), "lens_note": readouts(acts, P_note), "lens_act": readouts(acts, P_act),
                   "margin_act_logits": margin(lg[P_act]), "margin_note_logits": margin(lg[P_note])}
        if want:  # the gate: masked keys receive zero attention; local layers place zero mass beyond the window
            masked_mass, far_mass = 0.0, 0.0
            for li, A in enumerate(out.attentions):
                a = A[0, :, P_act, :].float().mean(0)
                if extra is not None: masked_mass += float(a[extra[0, 0, P_act].to(a.device)].sum())
                if not IS_GLOBAL[li] and W and P_act - W + 1 > 0: far_mass += float(a[: P_act - W + 1].sum())
            rec_arm["gate"] = {"attention_on_masked_keys": masked_mass, "local_mass_beyond_window": far_mass}
        row["arms"][arm] = rec_arm
    results.append(row)
    emit("row", n_done=n + 1, of=len(sample_ids), arms=list(arms), gate=row["arms"].get("all_carriers", {}).get("gate"))
for h in handles: h.remove()
(OUT / "w3b.jsonl").write_text("\n".join(json.dumps(x) for x in results) + "\n")
(OUT / "manifest.json").write_text(json.dumps({"schema_version": 1, "seat": "chief", "checkpoint": args.snapshot, "maps": args.maps, "capture": args.capture, "sample": sample_ids,
    "kinds": KINDS, "queries_masked_from": "P_note (the decision position) onward", "control": "random tokens of equal count (a token-count control only); the same-kind arms (older vs previous note, older vs previous call) are the role- and contiguity-matched comparisons; survival under a mask shows non-necessity of the masked edges under this intervention and does not date the decision; the two-hop relay through unmasked earlier positions is an open route", "precision": "float32", "width": 1,
    "device": args.device, "is_global_layer": IS_GLOBAL, "sliding_window": W, "design": "WORKSPACE-EXPERIMENTS-2026-09-10.md W-3b"}, indent=2) + "\n")
emit("done", n=len(results), minutes=round((time.monotonic() - T0) / 60, 1))
