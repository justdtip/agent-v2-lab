"""The complete-call test (second review, 2026-09-11): at the action position, generate the whole tool call greedily
with the residual at layer L replaced by (a) nothing, (b) itself, (c) the dictionary reconstruction, (d) four random
errors of the same norm, (e) four angle-matched errors (same norm, same signed projection on h as delta = h_hat - h),
and parse the call: tool, path, JSON validity.  Outcomes are paired against the model's own baseline call, and
against the expert's call from the corpus.  Two samples of 32 action cells taken from the substitution test's
selection file (the first 32, and the 32 lowest-margin among all 300).  The patch applies on the prefill only, as
in the steering pilot; decode steps carry one new position.  Float32 eager, greedy, 48 new tokens.

usage: call_test.py CONFIG LAYERS(comma) SELECTION_JSON OUT_JSON
"""
import json, re, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/workspace/chief/lab-a2/src")
from transformers import AutoTokenizer
from local_llm_lab import hf_text
from local_llm_lab.probes import device_bridge as DB, sae_bridge as B
from local_llm_lab.probes.device_lens_admission import load_admitted_device_lens
from local_llm_lab.pipeline.lens_fitting.upstream import load_upstream, upstream_index_of_repo_layer
import jlens.hf as upstream_hf

conf = json.loads(Path(sys.argv[1]).read_text()); layers = [int(x) for x in sys.argv[2].split(",")]; sel = json.loads(Path(sys.argv[3]).read_text()); out = Path(sys.argv[4])
if out.exists(): sys.exit(f"refused: {out} exists")
S4 = Path("/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")
HUB = Path("/workspace/.hf_home/hub/models--google--gemma-scope-2-4b-it/snapshots/3e94b68be95290aada5b7525cf431d3040f81bb1/resid_post_all"); REC = Path("/workspace/.hf_home/dictionaries/google/gemma-scope-2-4b-it/resid_post_all")
TOOLS = ["read_file", "finish", "replace_text", "calculate", "search_files", "list_files"]; name_pat = re.compile(r'"name":\s*"([a-z_]+)"'); path_pat = re.compile(r'"path":\s*"([^"]+)"')
t0 = time.time(); rng = np.random.default_rng(20260911); K = 4; dev = "cuda:0"; MAX_NEW = 48
lens, metadata = load_admitted_device_lens(Path("/workspace/chief/out/lens4b-f32/admitted-maps.npz"), checkpoint=S4, model_base=conf["model_base"], expected_sha256=conf["lens_sha256"])
prepared, prov = DB._capture_cells(Path("/workspace/chief/captures/4b"), Path("/workspace/chief/captures/4b/positions-sample.json"), Path("/workspace/chief/all-splits.jsonl"), S4, metadata, lens, conf, layers)
act_cells = [p for p in prepared if p["cell"]["position"] == "P_act"]
# the expert call, from the corpus completion (first occurrence of (task_id, step), as the runner dedups)
corpus, seen = {}, set()
for line in Path("/workspace/chief/all-splits.jsonl").read_text().splitlines():
    if not line.strip(): continue
    r = json.loads(line); m = r.get("metadata", {})
    if "task_id" not in m: continue
    key = (m["task_id"], m["step"])
    if key in seen: continue
    seen.add(key); corpus[key] = r
model, report = hf_text.load_text_causal_lm(S4, dtype="bfloat16", attn_implementation="eager", device=dev); model.eval(); model.to(torch.float32)
for p in model.parameters(): p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained(S4); load_upstream(); lm = upstream_hf.HFLensModel(model, tokenizer=tok); blocks = lm.layers
tool_ids = torch.tensor([tok(t, add_special_tokens=False).input_ids[0] for t in TOOLS], device=dev)
STATE = {"patch": None, "applied": 0}
def make_hook():
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if STATE["patch"] is not None:
            pos, vec = STATE["patch"]
            if h.shape[1] > pos:
                h = h.clone(); h[0, pos, :] = vec.to(h.dtype); STATE["applied"] += 1
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return None
    return hook
def generate(ids, pos, u, vec):
    STATE["patch"] = None if vec is None else (pos, vec); STATE["applied"] = 0
    hk = blocks[u].register_forward_hook(make_hook())
    try:
        with torch.no_grad(): o = model.generate(input_ids=ids, max_new_tokens=MAX_NEW, do_sample=False, num_beams=1, use_cache=True, output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
    finally: hk.remove(); STATE["patch"] = None
    new = o.sequences[0, ids.shape[1]:].tolist(); text = tok.decode(new); first = o.scores[0][0].float(); t = first[tool_ids]; srt = torch.sort(first, descending=True).values
    return {"text": text, "first_top_tool": TOOLS[int(torch.argmax(t))], "first_argmax": int(torch.argmax(first)), "first_margin": float(srt[0] - srt[1]), "patch_applied": STATE["applied"], **parse(text)}
def parse(text):
    js = '{"name": "' + text.split("```")[0]; mm = name_pat.search(js); pm = path_pat.search(js)
    try: obj = json.loads(js.strip()); valid = isinstance(obj, dict) and "name" in obj and "arguments" in obj
    except Exception: valid = False
    return {"tool": mm.group(1) if mm else None, "path": pm.group(1) if pm else None, "valid_json": valid}
def against(g, base):
    return {"tool_same_as_baseline": g["tool"] == base["tool"], "path_same_as_baseline": g["path"] == base["path"], "call_text_same_as_baseline": g["text"] == base["text"], "first_token_same_as_baseline": g["first_argmax"] == base["first_argmax"]}
rows = []
for L in layers:
    folder = f"layer_{L - 1}_width_16k_l0_small"; receipt = json.loads((REC / folder / "DIGEST.json").read_text())
    dictionary = B.load_dictionary(HUB / folder / "params.safetensors", HUB / folder / "config.json", source=f"{receipt['repo']}/{receipt['folder']}"); u = upstream_index_of_repo_layer(L)
    for sample, idxs in sel["samples"].items():
        for i in idxs:
            c = act_cells[i]; P = c["cell"]["token_index"]; ids = torch.tensor([c["input_ids"][:P + 1]], device=dev); h = np.asarray(c["residuals"][L], np.float32)
            row_meta = c["prompt_identity"]; exp = corpus[(row_meta["task_id"], row_meta["step"])]; expert = parse('"name": "'.join(exp["completion"].split('"name": "')[1:]) if '"name": "' in exp["completion"] else exp["completion"])
            z = B.encode(dictionary, h); hhat = B.decode(dictionary, z); delta = hhat - h; nd = float(np.linalg.norm(delta)); hn = h / np.linalg.norm(h); proj = float(delta @ hn)
            base = generate(ids, P, u, None); same = generate(ids, P, u, torch.as_tensor(h, device=dev)); rec = generate(ids, P, u, torch.as_tensor(hhat, device=dev))
            assert same["patch_applied"] == 1 and rec["patch_applied"] == 1
            row = {"sample": sample, "index": i, "row": c["cell"]["row"], "layer": L, "task_id": row_meta["task_id"], "step": row_meta["step"], "family": row_meta.get("family"), "expert": {"tool": expert["tool"], "path": expert["path"]},
                   "baseline": {**base, "tool_is_expert": base["tool"] == expert["tool"], "path_is_expert": base["path"] == expert["path"]}, "same_state": {**same, **against(same, base)}, "reconstruction": {**rec, **against(rec, base), "tool_is_expert": rec["tool"] == expert["tool"]},
                   "norm_delta": nd, "raw_share": nd / float(np.linalg.norm(h)), "random": [], "angle_matched": []}
            for _ in range(K):
                r = rng.standard_normal(h.size).astype(np.float32); r *= nd / np.linalg.norm(r); g = generate(ids, P, u, torch.as_tensor(h + r, device=dev)); row["random"].append({**g, **against(g, base)})
                n = rng.standard_normal(h.size).astype(np.float32); n -= (n @ hn) * hn; n *= np.sqrt(max(nd**2 - proj**2, 0.0)) / np.linalg.norm(n); a = (proj * hn + n).astype(np.float32)
                assert abs(float(np.linalg.norm(a)) - nd) < 1e-3 * nd and abs(float(a @ hn) - proj) < 1e-3 * max(abs(proj), 1e-6)
                g = generate(ids, P, u, torch.as_tensor(h + a, device=dev)); row["angle_matched"].append({**g, **against(g, base)})
            rows.append(row); print(json.dumps({"L": L, "sample": sample, "row": row["row"], "base_tool": base["tool"], "expert_tool": expert["tool"], "rec_tool": rec["tool"], "rec_same_call": rec["text"] == base["text"], "same_state_same_call": same["text"] == base["text"], "rand_tool_changes": sum(not g["tool_same_as_baseline"] for g in row["random"])}), flush=True)
summary = {}
for L in layers:
    for sample in sel["samples"]:
        rs = [r for r in rows if r["layer"] == L and r["sample"] == sample]; n = len(rs)
        def arm_counts(arm):
            gs = [g for r in rs for g in (r[arm] if isinstance(r[arm], list) else [r[arm]])]
            return {"n": len(gs), "tool_changed": sum(not g["tool_same_as_baseline"] for g in gs), "path_changed": sum(not g["path_same_as_baseline"] for g in gs), "call_text_changed": sum(not g["call_text_same_as_baseline"] for g in gs), "first_token_changed": sum(not g["first_token_same_as_baseline"] for g in gs), "invalid_json": sum(not g["valid_json"] for g in gs)}
        summary[f"L{L}/{sample}"] = {"cells": n, "baseline": {"tool_is_expert": sum(r["baseline"]["tool_is_expert"] for r in rs), "path_is_expert": sum(r["baseline"]["path_is_expert"] for r in rs), "valid_json": sum(r["baseline"]["valid_json"] for r in rs), "first_margin_median": float(np.median([r["baseline"]["first_margin"] for r in rs]))},
                                      "same_state": arm_counts("same_state"), "reconstruction": {**arm_counts("reconstruction"), "tool_is_expert": sum(r["reconstruction"]["tool_is_expert"] for r in rs)}, "random": arm_counts("random"), "angle_matched": arm_counts("angle_matched")}
out.write_text(json.dumps({"basis": __doc__, "conventions": {"dtype": str(next(model.parameters()).dtype), "attn": report["attn_implementation"], "decoding": "greedy", "max_new_tokens": MAX_NEW, "K": K}, "summary": summary, "rows": rows, "elapsed_s": round(time.time() - t0, 1)}, indent=1) + "\n")
print(json.dumps({"event": "done", "elapsed_s": round(time.time() - t0, 1), "summary": summary}), flush=True)
