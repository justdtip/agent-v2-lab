"""A3: move one SAE feature and leave the dictionary's unexplained error exactly where it was.

For a cell (layer L, position P_act) the dictionary writes h = b + sum_i z_i d_i + e. Removing feature k
means h' = h - z_k d_k, which changes the reconstruction and leaves e untouched by construction: the new
activation's distance from the dictionary's span is the same vector e it always was. The model is then run
from that patched residual and the WHOLE tool call is generated greedily, so the outcome is a parsed call,
not a next token.

Arms, all at the same patch site, all the same norm ||z_k d_k|| unless stated:
  baseline        unpatched
  same_state      h written back, must change nothing
  knockout        h - z_k d_k, the feature removed
  random          h + r, r Gaussian of the removed norm
  angle_matched   h + a, same norm and same signed projection on h as (-z_k d_k)
  sham            h - z_j d_j for a different active feature j, chosen as the active feature whose
                  contribution to the model's chosen tool is closest to zero while its norm is closest to
                  the removed one -- a feature the decomposition says is NOT carrying this decision
  amplify         h + z_k d_k, the feature doubled rather than removed

Which feature is k: the active feature whose contribution most separates the model's own chosen tool from
its own runner-up, read from the atlas decomposition recomputed here from the capture, not from the bundle.

usage: feature_intervention.py CONFIG LAYER N_CELLS OUT_JSON
"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/workspace/chief/lab-a2/src")
from transformers import AutoTokenizer
from local_llm_lab import hf_text
from local_llm_lab.probes import device_bridge as DB, sae_bridge as B
from local_llm_lab.probes.device_lens_admission import load_admitted_device_lens
from local_llm_lab.pipeline.lens_fitting.upstream import load_upstream, upstream_index_of_repo_layer
import jlens.hf as upstream_hf

conf = json.loads(Path(sys.argv[1]).read_text()); LAYER = int(sys.argv[2]); N = int(sys.argv[3]); out = Path(sys.argv[4])
if out.exists(): sys.exit(f"refused: {out} exists")
S4 = Path("/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")
SITE = os.environ.get("DICT_SITE", "resid_post_all"); SUFFIX = os.environ.get("DICT_SUFFIX", "width_16k_l0_small")
HUB = Path("/workspace/.hf_home/hub/models--google--gemma-scope-2-4b-it/snapshots/3e94b68be95290aada5b7525cf431d3040f81bb1") / SITE
REC = Path("/workspace/.hf_home/dictionaries/google/gemma-scope-2-4b-it") / SITE
TOOLS = ["read_file", "finish", "replace_text", "calculate", "search_files", "list_files"]
name_pat = re.compile(r'"name":\s*"([a-z_]+)"'); path_pat = re.compile(r'"path":\s*"([^"]+)"')
t0 = time.time(); rng = np.random.default_rng(20260911); K = int(os.environ.get("K_CONTROLS", "2")); dev = "cuda:0"; MAX_NEW = 48
lens, metadata = load_admitted_device_lens(Path("/workspace/chief/out/lens4b-f32/admitted-maps.npz"), checkpoint=S4, model_base=conf["model_base"], expected_sha256=conf["lens_sha256"])
prepared, prov = DB._capture_cells(Path("/workspace/chief/captures/4b"), Path("/workspace/chief/captures/4b/positions-sample.json"), Path("/workspace/chief/all-splits.jsonl"), S4, metadata, lens, conf, [LAYER])
STATE = {"patch": None, "applied": 0}
def hook(module, inputs, output):
    h = output[0] if isinstance(output, tuple) else output
    if STATE["patch"] is not None:
        pos, vec = STATE["patch"]
        if h.shape[1] > pos:
            h = h.clone(); h[0, pos, :] = vec.to(h.dtype); STATE["applied"] += 1
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return None
def parse(text):
    js = '{"name": "' + text.split("```")[0]; mm = name_pat.search(js); pm = path_pat.search(js)
    try: obj = json.loads(js.strip()); valid = isinstance(obj, dict) and "name" in obj and "arguments" in obj
    except Exception: valid = False
    return {"tool": mm.group(1) if mm else None, "path": pm.group(1) if pm else None, "valid_json": valid}
def generate(ids, pos, vec):
    STATE["patch"] = None if vec is None else (pos, vec); STATE["applied"] = 0
    hk = blocks[u].register_forward_hook(hook)
    try:
        with torch.no_grad(): o = model.generate(input_ids=ids, max_new_tokens=MAX_NEW, do_sample=False, num_beams=1, use_cache=True, output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
    finally: hk.remove(); STATE["patch"] = None
    new = o.sequences[0, ids.shape[1]:].tolist(); text = tok.decode(new); first = o.scores[0][0].float(); t = first[tool_ids]; srt = torch.sort(first, descending=True).values
    return {"text": text, "first_tool": TOOLS[int(torch.argmax(t))], "first_argmax": int(torch.argmax(first)), "first_margin": float(srt[0] - srt[1]), "patch_applied": STATE["applied"], **parse(text)}
def against(g, base): return {"tool_same": g["tool"] == base["tool"], "path_same": g["path"] == base["path"], "text_same": g["text"] == base["text"], "first_token_same": g["first_argmax"] == base["first_argmax"]}
act = [p for p in prepared if p["cell"]["position"] == "P_act"]
MULTIPLES = tuple(float(x) for x in os.environ.get("MULTIPLES", "1").split(","))
SELECT = os.environ.get("SELECT", "first")  # first | low_margin
model, report = hf_text.load_text_causal_lm(S4, dtype="bfloat16", attn_implementation="eager", device=dev); model.eval(); model.to(torch.float32)
for p in model.parameters(): p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained(S4); load_upstream(); lm = upstream_hf.HFLensModel(model, tokenizer=tok); blocks = lm.layers
u = upstream_index_of_repo_layer(LAYER)
folder = f"layer_{LAYER - 1}_{SUFFIX}"; receipt = json.loads((REC / folder / "DIGEST.json").read_text())
dictionary = B.load_dictionary(HUB / folder / "params.safetensors", HUB / folder / "config.json", source=f"{receipt['repo']}/{receipt['folder']}")
assert B.layer_for_hook(dictionary.hook_point) == LAYER
unemb = B.load_unembedding(S4); J = B.lens_map_for_layer(lens, LAYER)
tool_ids = torch.tensor([tok(t, add_special_tokens=False).input_ids[0] for t in TOOLS], device=dev)
Lscore = lambda x: ((x @ J.T) * unemb.gain) @ unemb.weight.T
if SELECT == "low_margin":
    # one unpatched generation per cell to rank by the first-token margin; the margin does not depend on L
    scored = []
    for c in act:
        P = c["cell"]["token_index"]; ids = torch.tensor([c["input_ids"][:P + 1]], device=dev)
        g = generate(ids, P, None); scored.append((g["first_margin"], c))
    scored.sort(key=lambda x: x[0]); cells = [c for _, c in scored[:N]]
    print(json.dumps({"selection": "lowest first-token margin", "n": len(cells), "margin_max_selected": scored[min(N, len(scored)) - 1][0]}), flush=True)
else:
    cells = act[:N]
print(json.dumps({"loaded_s": round(time.time() - t0, 1), "cells": len(cells), "layer": LAYER, "dictionary": f"{SITE}/{folder}", "multiples": MULTIPLES, "select": SELECT}), flush=True)
rows = []
for c in cells:
    P = c["cell"]["token_index"]; ids = torch.tensor([c["input_ids"][:P + 1]], device=dev); h = np.asarray(c["residuals"][LAYER], np.float32)
    z = B.encode(dictionary, h); act = np.flatnonzero(z > 0)
    if act.size < 2: continue
    base = generate(ids, P, None)
    if base["tool"] not in TOOLS: continue
    # the model's own pick and runner-up among the six tool first-tokens, from the baseline generation
    w = TOOLS.index(base["first_tool"]); scores = None
    lw = unemb.weight[int(tool_ids[w])].astype(np.float64) * unemb.gain; lw = lw @ J
    per_tool = {}
    for ti, t in enumerate(TOOLS):
        lv = unemb.weight[int(tool_ids[ti])] * unemb.gain; lv = lv if J is None else lv @ J
        per_tool[t] = z[act] * (dictionary.w_dec[act] @ lv)
    sc = {t: float(v.sum()) for t, v in per_tool.items()}
    runner = max((t for t in TOOLS if t != base["first_tool"]), key=lambda t: sc[t])
    sep = per_tool[base["first_tool"]] - per_tool[runner]
    k_local = int(np.argmax(sep)); k = int(act[k_local])
    delta = (z[k] * dictionary.w_dec[k]).astype(np.float32); nd = float(np.linalg.norm(delta)); hn = h / np.linalg.norm(h); proj = float((-delta) @ hn)
    # sham: the active feature whose separation is nearest zero, with the closest norm to the removed one
    norms = np.array([float(np.linalg.norm(z[j] * dictionary.w_dec[j])) for j in act])
    cand = np.argsort(np.abs(sep) / (np.abs(sep).max() + 1e-9) + np.abs(norms - nd) / (nd + 1e-9))
    j_local = int(next(i for i in cand if int(act[i]) != k)); j = int(act[j_local])
    sham = (z[j] * dictionary.w_dec[j]).astype(np.float32)
    ht = torch.as_tensor(h, device=dev); dt = torch.as_tensor(delta, device=dev); st = torch.as_tensor(sham, device=dev)
    row = {"row": c["cell"]["row"], "task_id": c["prompt_identity"]["task_id"], "family": c["prompt_identity"].get("family"), "layer": LAYER,
           "active_features": int(act.size), "feature": k, "feature_activation": float(z[k]), "feature_separation": float(sep[k_local]),
           "feature_separation_share": float(sep[k_local] / sep.sum()) if sep.sum() else None, "removed_norm": nd, "residual_norm": float(np.linalg.norm(h)),
           "sham_feature": j, "sham_separation": float(sep[j_local]), "sham_norm": float(norms[j_local]),
           "baseline": {**base, "expert_tool": None}, "arms": {}}
    row["arms"]["same_state"] = {**(g := generate(ids, P, ht)), **against(g, base)}
    for mult in MULTIPLES:
        tag = f"x{mult:g}"
        row["arms"][f"knockout_{tag}"] = {**(g := generate(ids, P, ht - mult * dt)), **against(g, base), "moved_norm": mult * nd, "moved_share_of_h": mult * nd / float(np.linalg.norm(h))}
        row["arms"][f"amplify_{tag}"] = {**(g := generate(ids, P, ht + mult * dt)), **against(g, base), "moved_norm": mult * nd}
        row["arms"][f"sham_{tag}"] = {**(g := generate(ids, P, ht - mult * st)), **against(g, base), "moved_norm": mult * float(norms[j_local])}
        row["arms"][f"random_{tag}"] = []; row["arms"][f"angle_matched_{tag}"] = []
        for _ in range(K):
            r = rng.standard_normal(h.size).astype(np.float32); r *= mult * nd / np.linalg.norm(r)
            row["arms"][f"random_{tag}"].append({**(g := generate(ids, P, ht + torch.as_tensor(r, device=dev))), **against(g, base), "moved_norm": mult * nd})
            n = rng.standard_normal(h.size).astype(np.float32); n -= (n @ hn) * hn
            pr = mult * proj; tang = np.sqrt(max((mult * nd) ** 2 - pr ** 2, 0.0)); n *= tang / np.linalg.norm(n)
            a = (pr * hn + n).astype(np.float32)
            assert abs(float(np.linalg.norm(a)) - mult * nd) < 1e-3 * mult * nd
            row["arms"][f"angle_matched_{tag}"].append({**(g := generate(ids, P, ht + torch.as_tensor(a, device=dev))), **against(g, base), "moved_norm": mult * nd})
    rows.append(row)
    ko = {f"x{m:g}": row["arms"][f"knockout_x{m:g}"]["tool"] for m in MULTIPLES}
    print(json.dumps({"row": row["row"], "base": base["tool"], "margin": round(base["first_margin"], 1), "feat": k, "share": round(row["feature_separation_share"] or 0, 2), "knockout_by_multiple": ko, "sham_x_top": row["arms"][f"sham_x{MULTIPLES[-1]:g}"]["tool"], "rand_changed_top": sum(not g["tool_same"] for g in row["arms"][f"random_x{MULTIPLES[-1]:g}"])}), flush=True)
def tally(arm):
    gs = [g for r in rows for g in (r["arms"][arm] if isinstance(r["arms"][arm], list) else [r["arms"][arm]])]
    return {"n": len(gs), "tool_changed": sum(not g["tool_same"] for g in gs), "path_changed": sum(not g["path_same"] for g in gs), "text_changed": sum(not g["text_same"] for g in gs), "invalid": sum(not g["valid_json"] for g in gs)}
arms = ["same_state"] + [f"{b}_x{m:g}" for m in MULTIPLES for b in ("knockout", "amplify", "sham", "random", "angle_matched")]
summary = {"cells": len(rows), "layer": LAYER, "dictionary": f"{SITE}/{folder}", "multiples": list(MULTIPLES), "select": SELECT, **{a: tally(a) for a in arms}}
out.write_text(json.dumps({"basis": __doc__, "conventions": {"dtype": str(next(model.parameters()).dtype), "attn": report["attn_implementation"], "decoding": "greedy", "max_new_tokens": MAX_NEW, "K": K}, "summary": summary, "rows": rows, "elapsed_s": round(time.time() - t0, 1)}, indent=1) + "\n")
print(json.dumps({"event": "done", "elapsed_s": round(time.time() - t0, 1), "summary": summary}), flush=True)
