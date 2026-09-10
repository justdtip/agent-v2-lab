"""Steering pilot, experiment 1 in its minimal form (Codex's proposal §6, on the existing corpus): does the
residual at the action position carry the target, and can a donor's residual redirect the executed call?

Per recipient decision (aggregate_report, clean, steps 2–4, the expert reads metric-(step-1).txt in the
episode's directory), at each layer L of a sweep: the residual stream at block L's output, position P_act
(the token before the tool-name token; the expert's note is teacher-forced), is replaced by a donor's
residual at the donor's own P_act, then the complete tool call is generated greedily and parsed.
Arms: baseline (no patch); same_state (the recipient's own residual, must reproduce the baseline
token for token — the no-op control); opposite_target (another episode, same step and metric index,
different directory: prediction under a target representation, the path's directory switches);
pending_file (the same episode's next step, the next metric: the path's file switches, directory kept);
operation (the same episode's later calculate step: the tool switches); unrelated_family (a read_file
decision from another family: whether a foreign residual also switches the call — the rival, 'the whole
plan moves').

Scored on the generated text: the tool, the path, JSON validity; the path classified against the
recipient's expert path, the donor's expert path and the recipient's own baseline path. The first
generated token's logit margin (the recipient's expert tool's first token against the best other tool's)
is recorded as a graded readout. A whole-residual swap is a localisation test: a donor-consistent
switch is a counterfactual effect on the generated call at that layer, not identification of a target
component, not task success (Codex 712f78c, proposal §6). The base instruction-tuned 4B; float32, eager,
determinism pinned; greedy decoding; nothing sampled.

usage: steer_pilot.py SNAPSHOT CORPUS_JSONL OUT_DIR [--device cuda:0] [--n-recipients 24] [--layers 4,8,...]
       [--max-new-tokens 48] [--seed 20260910]
"""
from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("snapshot"); ap.add_argument("corpus"); ap.add_argument("out")
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n-recipients", type=int, default=24)
ap.add_argument("--layers", default="4,8,12,16,20,24,28,32"); ap.add_argument("--max-new-tokens", type=int, default=48); ap.add_argument("--seed", type=int, default=20260910)
args = ap.parse_args(); OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True); T0 = time.monotonic()
LAYERS = [int(x) for x in args.layers.split(",")]
TOOLS = ["read_file", "finish", "replace_text", "calculate", "search_files", "list_files"]
FENCE = "```json"; SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
name_pat = re.compile(r'"name":\s*"([a-z_]+)"'); path_pat = re.compile(r'"path":\s*"([^"]+)"')
def emit(event, /, **f):
    row = {"event": event, "elapsed_s": round(time.monotonic() - T0, 1), **f}; print(json.dumps(row, default=str), flush=True)
    with (OUT / "progress.jsonl").open("a") as fh: fh.write(json.dumps(row, default=str) + "\n")

# ---- the corpus, and the pairs
rows, seen = [], set()
for line in open(args.corpus):
    r = json.loads(line); m = r.get("metadata", {})
    if "task_id" not in m: continue
    k = (m["task_id"], m["step"])
    if k in seen: continue
    seen.add(k); rows.append(r)
corpus_sha = hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()
def info(i):
    r = rows[i]; m = r["metadata"]; mm = name_pat.search(r["completion"]); pm = path_pat.search(r["completion"])
    return {"i": i, "task_id": m["task_id"], "base": m["task_id"].rsplit("-", 1)[0], "family": m["family"], "variant": m["variant"], "step": m["step"],
            "tool": mm.group(1) if mm else None, "path": pm.group(1) if pm else None}
INFO = {i: info(i) for i in range(len(rows))}
by_key = {(v["base"], v["step"]): i for i, v in INFO.items()}
rng = np.random.default_rng(args.seed)
cands = [i for i, v in INFO.items() if v["family"] == "aggregate_report" and v["variant"] == "clean" and v["tool"] == "read_file" and v["path"] and "metric-" in v["path"] and v["step"] in (2, 3, 4)]
def opposite_for(v):
    return [j for j, w in INFO.items() if w["family"] == "aggregate_report" and w["variant"] == "clean" and w["step"] == v["step"] and w["tool"] == "read_file" and w["base"] != v["base"]
            and w["path"] and w["path"].rsplit("/", 1)[1] == v["path"].rsplit("/", 1)[1] and w["path"] != v["path"]]
def operation_for(v):
    return [by_key[(v["base"], s)] for s in range(v["step"] + 1, v["step"] + 6) if (v["base"], s) in by_key and INFO[by_key[(v["base"], s)]]["tool"] == "calculate"]
unrelated_pool = [j for j, w in INFO.items() if w["family"] in ("search", "read", "pointer_chain") and w["tool"] == "read_file" and w["path"]]
pairs = []
for i in cands:
    v = INFO[i]; nxt = by_key.get((v["base"], v["step"] + 1))
    if nxt is None or INFO[nxt]["tool"] != "read_file" or not INFO[nxt]["path"] or INFO[nxt]["path"] == v["path"]: continue
    opp = opposite_for(v); op = operation_for(v)
    if not opp: continue
    pairs.append({"recipient": i, "opposite_target": int(rng.choice(opp)), "pending_file": nxt, "operation": (op[0] if op else None), "unrelated_family": int(rng.choice(unrelated_pool))})
# balanced over steps 2, 3, 4
per_step = {s: [p for p in pairs if INFO[p["recipient"]]["step"] == s] for s in (2, 3, 4)}
for s in (2, 3, 4): rng.shuffle(per_step[s])
chosen, k = [], 0
while len(chosen) < args.n_recipients and any(per_step[s] for s in (2, 3, 4)):  # round-robin over steps, balanced for any N
    st = (2, 3, 4)[k % 3]; k += 1
    if per_step[st]: chosen.append(per_step[st].pop())
emit("pairs", candidates=len(cands), pairs_available=len(pairs), chosen=len(chosen), per_step={s: sum(1 for p in chosen if INFO[p["recipient"]]["step"] == s) for s in (2, 3, 4)},
     with_operation_donor=sum(1 for p in chosen if p["operation"] is not None), corpus_sha256=corpus_sha)

# ---- the model
from local_llm_lab import device  # noqa: E402
device.pin(seed=0)
import torch  # noqa: E402
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
from transformers import AutoTokenizer  # noqa: E402
from local_llm_lab import hf_text  # noqa: E402
import jlens.hf as upstream_hf  # noqa: E402
model, report = hf_text.load_text_causal_lm(args.snapshot, dtype="bfloat16", attn_implementation="eager", device=args.device)
for p in model.parameters(): p.requires_grad_(False)
model.eval(); model.to(torch.float32)
tok = AutoTokenizer.from_pretrained(args.snapshot); lens_model = upstream_hf.HFLensModel(model, tokenizer=tok)
n_layers = int(lens_model.n_layers); assert max(LAYERS) <= n_layers, f"layer sweep beyond {n_layers}"
tool_first = {t: tok(t, add_special_tokens=False).input_ids[0] for t in TOOLS}; assert len(set(tool_first.values())) == len(TOOLS)
tool_ids = torch.tensor([tool_first[t] for t in TOOLS], device=args.device)
emit("loaded", layers=n_layers, sweep=LAYERS, dtype=str(next(model.parameters()).dtype), attn=report["attn_implementation"], load_report_sha256=report.get("sha256"))

# ---- hooks: record residuals at a position (donor capture) and patch a position (recipient generation), on block L's output
STATE = {"record": None, "patch": None}  # record: (pos, dict layer->vec); patch: (pos, {layer: vec})
def make_hook(L):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if STATE["record"] is not None:
            pos, store = STATE["record"]
            if h.shape[1] > pos: store[L] = h[0, pos, :].detach().clone()
        if STATE["patch"] is not None:
            pos, vecs = STATE["patch"]
            if L in vecs and h.shape[1] > pos:  # the prefill only; decode steps carry one new position
                h = h.clone(); h[0, pos, :] = vecs[L].to(h.dtype); STATE["patch_applied"] = STATE.get("patch_applied", 0) + 1
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return None
    return hook
handles = [lens_model.layers[L - 1].register_forward_hook(make_hook(L)) for L in LAYERS]

def prefix_ids(i):
    r = rows[i]; prompt, comp = r["prompt"], r["completion"]
    p_ids = tok(prompt, add_special_tokens=False).input_ids; assert p_ids[0] == tok.bos_token_id and p_ids[1] != tok.bos_token_id
    enc_c = tok(comp, add_special_tokens=False, return_offsets_mapping=True); mm = name_pat.search(comp); name_char = mm.start(1)
    t_idx = next(j for j, (a, b) in enumerate(enc_c.offset_mapping) if a <= name_char < b)
    ids = p_ids + enc_c.input_ids[:t_idx]  # through P_act; the tool-name token is what the model generates next
    return torch.tensor([ids], device=args.device), len(ids) - 1  # P_act = last position

def residuals_at(i):
    ids, pos = prefix_ids(i); store = {}
    STATE["record"] = (pos, store); STATE["patch"] = None
    with torch.no_grad(): model(input_ids=ids, logits_to_keep=1)
    STATE["record"] = None
    assert set(store) == set(LAYERS), f"recorded {sorted(store)} of {LAYERS}"
    return store, pos

def generate(ids, pos, vecs):
    STATE["record"] = None; STATE["patch"] = (pos, vecs) if vecs else None; STATE["patch_applied"] = 0
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1, use_cache=True, output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
    STATE["patch"] = None
    new = out.sequences[0, ids.shape[1]:].tolist(); text = tok.decode(new)
    first = out.scores[0][0].float(); t = first[tool_ids]; top_tool = TOOLS[int(torch.argmax(t))]
    return {"ids": new, "text": text, "first_top_tool": top_tool, "first_tool_logits": {TOOLS[j]: float(t[j]) for j in range(6)}, "patch_applied": STATE.get("patch_applied", 0)}

def parse(text, expert_tool_prefix=True):
    cut = text.split("```")[0]
    js = '{"name": "' + cut  # the model generates from the tool name onward
    tool = None; path = None; valid = False
    mm = name_pat.search(js); tool = mm.group(1) if mm else None
    pm = path_pat.search(js); path = pm.group(1) if pm else None
    try:
        obj = json.loads(js.strip()); valid = isinstance(obj, dict) and "name" in obj and "arguments" in obj
    except Exception:
        valid = False
    return {"tool": tool, "path": path, "valid_json": valid}

def classify(parsed, rec, donor, baseline_path):
    p = parsed["path"]
    cls = "recipient" if p is not None and p == rec["path"] else "donor" if (donor is not None and p is not None and p == donor["path"]) else "none" if p is None else "other"
    return {"path_class": cls, "path_equals_baseline": (p == baseline_path) if p is not None else None, "tool_preserved": parsed["tool"] == rec["tool"], "tool_equals_donor": (donor is not None and parsed["tool"] == donor["tool"]),
            "directory_switched": (p is not None and donor is not None and p.rsplit("/", 1)[0] == donor["path"].rsplit("/", 1)[0] and p.rsplit("/", 1)[0] != rec["path"].rsplit("/", 1)[0]),
            "file_switched_same_directory": (p is not None and donor is not None and p.rsplit("/", 1)[0] == rec["path"].rsplit("/", 1)[0] and p.rsplit("/", 1)[1] == donor["path"].rsplit("/", 1)[1] and p != rec["path"])}

results_fh = (OUT / "pilot.jsonl").open("w"); n_rows = 0
(OUT / "run.json").write_text(json.dumps({"schema_version": 1, "seat": "chief", "script_sha256": SCRIPT_SHA, "checkpoint": args.snapshot, "load_report_sha256": report.get("sha256"), "corpus": args.corpus, "corpus_sha256": corpus_sha,
    "design": "steering pilot, experiment 1 minimal form: whole-residual donor replacement at P_act, greedy complete-call generation", "layers": LAYERS, "arms": ["baseline", "same_state", "opposite_target", "pending_file", "operation", "unrelated_family"],
    "pairs": chosen, "seed": args.seed, "max_new_tokens": args.max_new_tokens, "precision": "float32", "attn": "eager", "decoding": "greedy", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2, default=str) + "\n")
for n, pair in enumerate(chosen):
    ri = pair["recipient"]; rec = INFO[ri]; ids, pos = prefix_ids(ri)
    own, own_pos = residuals_at(ri); assert own_pos == pos
    donors = {}
    for arm in ("opposite_target", "pending_file", "operation", "unrelated_family"):
        j = pair.get(arm)
        if j is None: continue
        vecs, _ = residuals_at(j); donors[arm] = (j, vecs)
    base = generate(ids, pos, None); base_p = parse(base["text"]); baseline_path = base_p["path"]
    row = {"n": n, "recipient": rec, "P_act": pos, "S": int(ids.shape[1]), "donors": {arm: INFO[j] for arm, (j, _) in donors.items()},
           "baseline": {**base, **base_p, **classify(base_p, rec, None, baseline_path), "expert_path_reproduced": base_p["path"] == rec["path"], "expert_tool_reproduced": base_p["tool"] == rec["tool"]}, "layers": {}}
    for L in LAYERS:
        lay = {}
        ss = generate(ids, pos, {L: own[L]}); ss_p = parse(ss["text"])
        lay["same_state"] = {**ss, **ss_p, **classify(ss_p, rec, None, baseline_path), "identical_to_baseline": ss["ids"] == base["ids"]}
        assert ss["patch_applied"] == 1, f"same-state patch applied {ss['patch_applied']} times at layer {L}"
        for arm, (j, vecs) in donors.items():
            g = generate(ids, pos, {L: vecs[L]}); gp = parse(g["text"]); lay[arm] = {**g, **gp, **classify(gp, rec, INFO[j], baseline_path)}
            assert g["patch_applied"] == 1, f"{arm} patch applied {g['patch_applied']} times at layer {L}"
        row["layers"][str(L)] = lay
    results_fh.write(json.dumps(row, default=str) + "\n"); results_fh.flush(); n_rows += 1
    emit("recipient", n_done=n + 1, of=len(chosen), task=rec["task_id"], step=rec["step"], baseline_path_class=row["baseline"]["path_class"], baseline_tool=base_p["tool"],
         same_state_identical_all_layers=all(row["layers"][str(L)]["same_state"]["identical_to_baseline"] for L in LAYERS),
         switches={arm: [L for L in LAYERS if row["layers"][str(L)].get(arm, {}).get("path_class") == "donor" or (arm == "operation" and row["layers"][str(L)].get(arm, {}).get("tool_equals_donor"))] for arm in donors},
         peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2) if args.device.startswith("cuda") else None)
for h in handles: h.remove()
results_fh.close()
(OUT / "manifest.json").write_text(json.dumps({"schema_version": 1, "rows_written": n_rows, "script_sha256": SCRIPT_SHA, "layers": LAYERS, "recipients": len(chosen),
    "claim_ceiling": "a whole-residual swap is a localisation test: a donor-consistent switch is a counterfactual effect on the generated call at that layer, not identification of a target component, not task success; the expert's note is teacher-forced; the base model's own baseline call is scored beside the expert's",
    "same_state_control": "the recipient's own residual re-injected at the same layer must reproduce the baseline token for token", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n")
emit("done", recipients=n_rows, minutes=round((time.monotonic() - T0) / 60, 1))
