"""Rebuild index.jsonl for a workspace capture that did not reach its end-of-run index write.

The capture writes residual_note.npy / residual_act.npy incrementally (memmap) and sample.jsonl per
sample row, but index.jsonl only at the very end (workspace_capture.py line 168). If the pass dies —
an OOM under a shared card, say — the residuals for every completed decision are on disk but
uninterpretable. This rebuilds the per-row index from the corpus deterministically: the fields the
capture derives without a forward (positions, tool, token counts). The model's own six-tool
distribution needs a forward and is left null; the lens and logit-lens readouts are a pure function
of the saved residual, the maps and the unembed and are recomputed later by a GPU readout pass
(recompute_readouts, no forward). A row is accepted only if its residual is non-zero, so the count
matches what actually reached disk. Read-only: it never writes into the capture dir.

usage: reconstruct_index.py CAPTURE_DIR CORPUS_JSONL SNAPSHOT OUT_JSONL [--check]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

TOOLS = ["read_file", "finish", "replace_text", "calculate", "search_files", "list_files"]
ap = argparse.ArgumentParser()
ap.add_argument("capture"); ap.add_argument("corpus"); ap.add_argument("snapshot"); ap.add_argument("out")
ap.add_argument("--check", action="store_true", help="verify against sample.jsonl and the tool-token rule; write nothing")
a = ap.parse_args(); CAP = Path(a.capture)
man = json.loads((CAP / "manifest.json").read_text()) if (CAP / "manifest.json").exists() else {}

rows, seen = [], set()
for line in open(a.corpus):
    r = json.loads(line); m = r.get("metadata", {})
    if "task_id" not in m: continue
    k = (m["task_id"], m["step"])
    if k in seen: continue
    seen.add(k); rows.append(r)

res_note = np.load(CAP / "residual_note.npy", mmap_mode="r")
res_act = np.load(CAP / "residual_act.npy", mmap_mode="r")
N_alloc = res_note.shape[0]
assert N_alloc == len(rows) or N_alloc == man.get("decisions", N_alloc), f"memmap rows {N_alloc} vs corpus {len(rows)}"
tok = AutoTokenizer.from_pretrained(a.snapshot)
tool_first = {t: tok(t, add_special_tokens=False).input_ids[0] for t in TOOLS}
name_pat = re.compile(r'"name":\s*"([a-z_]+)"')
# the sample rows: from the manifest when the pass finished, else from sample.jsonl on disk (the manifest is also end-only)
sample_ids = set(man.get("sample", []))
if not sample_ids and (CAP / "sample.jsonl").exists():
    sample_ids = {json.loads(l)["i"] for l in (CAP / "sample.jsonl").read_text().splitlines() if l.strip()}

# a completed row has a non-zero residual; find how many rows actually reached disk
def done_count():
    lo, hi = 0, N_alloc
    # the memmap is zero-initialised; completed rows are (almost surely) non-zero at layer 0.
    nz = np.any(res_note[:, 0, :] != 0.0, axis=1)  # [N_alloc] bool, one read of the first layer plane
    idx = np.nonzero(nz)[0]
    return (int(idx[-1]) + 1) if len(idx) else 0, nz

n_done, nz = done_count()
# contiguity: every row below n_done must be non-zero (a hole would mean a row wrote zeros, unexpected)
holes = [i for i in range(n_done) if not nz[i]]
print(json.dumps({"event": "scan", "allocated_rows": N_alloc, "n_done": n_done, "holes": holes[:10], "n_holes": len(holes)}))

built, bad_tool = [], []
for i in range(n_done):
    if not nz[i]: continue
    r = rows[i]; m = r["metadata"]; prompt, comp = r["prompt"], r["completion"]
    enc_p = tok(prompt, add_special_tokens=False); p_ids = enc_p.input_ids
    assert p_ids[0] == tok.bos_token_id and p_ids[1] != tok.bos_token_id, f"row {i}: BOS"
    mm = name_pat.search(comp); tool = mm.group(1); name_char = mm.start(1)
    enc_c = tok(comp, add_special_tokens=False, return_offsets_mapping=True)
    t_idx = next(j for j, (x, y) in enumerate(enc_c.offset_mapping) if x <= name_char < y)
    P_note = len(p_ids) - 1; P_act = len(p_ids) + t_idx - 1
    if enc_c.input_ids[t_idx] != tool_first[tool]: bad_tool.append(i)
    built.append({"i": i, "task_id": m["task_id"], "step": m["step"], "family": m["family"], "variant": m.get("variant"), "recovery": m.get("recovery"),
                  "tool": tool, "tool_idx": TOOLS.index(tool), "P_note": P_note, "P_act": P_act, "n_prompt_tokens": len(p_ids), "n_note_tokens": t_idx,
                  "model_six_note": None, "model_six_act": None, "model_mass_note": None, "model_mass_act": None,
                  "model_top5_note": None, "model_top5_act": None, "lens_note": None, "lens_act": None,
                  "in_sample": (i in sample_ids), "reconstructed": True})
print(json.dumps({"event": "built", "n": len(built), "bad_tool_token_rows": bad_tool[:10], "n_bad": len(bad_tool)}))

if a.check:
    # cross-check the sample rows against sample.jsonl: same set below n_done, kinds length == S
    sj = {}
    sp = CAP / "sample.jsonl"
    if sp.exists():
        for l in sp.read_text().splitlines():
            if l.strip():
                s = json.loads(l); sj[s["i"]] = s
    sample_below = sorted(i for i in sample_ids if i < n_done)
    in_sj = sorted(sj)
    missing = [i for i in sample_below if i not in sj]
    len_ok = all(len(sj[b["i"]]["kinds"]) == b["P_act"] + 2 for b in built if b["i"] in sj) if sj else None
    print(json.dumps({"event": "check", "sample_below_n_done": len(sample_below), "sample_rows_on_disk": len(in_sj),
                      "sample_missing_from_disk": missing[:10], "kinds_len_equals_P_act_plus_2": len_ok,
                      "bad_tool_token_rows": len(bad_tool), "verdict": "ok" if (not missing and not bad_tool and not holes and len_ok in (True, None)) else "SEE ABOVE"}))
    sys.exit(0)

Path(a.out).write_text("\n".join(json.dumps(x) for x in built) + "\n")
print(json.dumps({"event": "wrote", "path": a.out, "n": len(built), "note": "model_* null (needs a forward); lens_*/logit null — recompute from residuals+maps+unembed with recompute_readouts"}))
