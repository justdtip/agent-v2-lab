"""Diagnostic for the A2 refusal at repository layer 18 on the 4B: is the score-decomposition identity
L h = L b + sum_i z_i (L d_i) + L e violated by orientation, or by float32 accumulation over 16,384 terms?

For every cell of the positions file, computes the runner's own float32 gap (its exact arithmetic path)
and the same identity in float64 from the same float32 inputs (h, and z as the module encodes it), with
the magnitude of the terms being summed. Nothing here changes the runner or any threshold; it reads.

usage: diag_identity.py CONFIG LAYER OUT_JSON
"""
import json, sys, time
from pathlib import Path
import numpy as np
from local_llm_lab.probes import device_bridge as DB, sae_bridge as B
from local_llm_lab.probes.device_lens_admission import load_admitted_device_lens
from local_llm_lab import hf_text

conf = json.loads(Path(sys.argv[1]).read_text()); layer = int(sys.argv[2]); out = Path(sys.argv[3])
if out.exists(): sys.exit(f"refused: {out} exists")
S4 = Path("/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")
folder = conf["dictionary_folder"]
D4 = Path("/workspace/.hf_home/hub/models--google--gemma-scope-2-4b-it/snapshots/3e94b68be95290aada5b7525cf431d3040f81bb1") / folder
R4 = Path("/workspace/.hf_home/dictionaries/google/gemma-scope-2-4b-it") / folder / "DIGEST.json"
t0 = time.time()
lens, metadata = load_admitted_device_lens(Path("/workspace/chief/out/lens4b-f32/admitted-maps.npz"), checkpoint=S4, model_base=conf["model_base"], expected_sha256=conf["lens_sha256"])
dmeta, receipt = DB._dictionary(D4, R4, conf, lens.hidden_size)
assert B.hook_alignment(dmeta, lens, base=conf["model_base"], nu=metadata["nu"]) == layer
residuals, readings, prov = DB._capture(Path("/workspace/chief/captures/4b"), Path("/workspace/chief/captures/4b/positions-sample.json"), Path("/workspace/chief/all-splits.jsonl"), S4, metadata, lens, conf, dmeta, Path("/workspace/chief/captures/4b/pairings-sample.json"))
dictionary = B.load_dictionary(D4 / "params.safetensors", D4 / "config.json", source=f"{receipt['repo']}/{receipt['folder']}")
unemb = B.load_unembedding(S4)
J = B.lens_map_for_layer(lens, layer)
print(json.dumps({"loaded_s": round(time.time() - t0, 1), "cells": len(readings), "J": None if J is None else list(J.shape)}), flush=True)
J64 = None if J is None else J.astype(np.float64); g64 = unemb.gain.astype(np.float64)
wdec64, bdec64 = dictionary.w_dec.astype(np.float64), dictionary.b_dec.astype(np.float64)
rows = []
for cell, h in zip(readings, residuals, strict=True):
    v = int(cell["token_id"]); h = np.asarray(h, dtype=np.float32).reshape(-1)
    z = B.encode(dictionary, h); e = h - B.decode(dictionary, z)
    # the runner's float32 path, verbatim
    def score32(x): return B.lens_scores(unemb, J, x, gain=True)[0]
    lh, lb, le = score32(h), score32(dictionary.b_dec), score32(e)
    lv = unemb.weight[v] * unemb.gain; lv = lv if J is None else lv @ J
    c32 = z * (dictionary.w_dec @ lv)
    gap32 = float(abs(lh[v] - (lb[v] + c32.sum() + le[v]))); scale32 = float(abs(lh[v])) + 1e-6
    # float64 from the same float32 h and z
    h64, z64 = h.astype(np.float64), z.astype(np.float64)
    e64 = h64 - (z64 @ wdec64 + bdec64)
    lv64 = (unemb.weight[v].astype(np.float64) * g64); lv64 = lv64 if J64 is None else lv64 @ J64
    def s64(x): return float(((x @ J64.T if J64 is not None else x) * g64) @ unemb.weight[v].astype(np.float64))
    c64 = z64 * (wdec64 @ lv64)
    gap64 = abs(s64(h64) - (s64(bdec64) + c64.sum() + s64(e64)))
    rows.append({"row": cell["row"], "position": cell["position"], "token_id": v, "score32": float(lh[v]), "gap32": gap32,
                 "runner_refuses": gap32 > conf["a2"]["identity_tolerance"] * max(1.0, scale32), "gap64": float(gap64),
                 "sum_abs_terms": float(np.abs(c32).sum()), "max_abs_term": float(np.abs(c32).max()), "active": int((z > 0).sum()),
                 "raw_share": float(np.linalg.norm(e) / np.linalg.norm(h)),
                 "score_share": float(np.linalg.norm(le) / np.linalg.norm(lh))})
g32 = np.array([r["gap32"] for r in rows]); g64 = np.array([r["gap64"] for r in rows]); sat = np.array([r["sum_abs_terms"] for r in rows])
summary = {"layer": layer, "cells": len(rows), "runner_would_refuse": int(sum(r["runner_refuses"] for r in rows)),
           "gap32": {"median": float(np.median(g32)), "max": float(g32.max())}, "gap64": {"median": float(np.median(g64)), "max": float(g64.max())},
           "gap32_over_sum_abs_terms": {"median": float(np.median(g32 / sat)), "max": float((g32 / sat).max())},
           "sum_abs_terms": {"median": float(np.median(sat)), "max": float(sat.max())},
           "raw_share": {"median": float(np.median([r["raw_share"] for r in rows])), "max": float(max(r["raw_share"] for r in rows))},
           "score_share": {"median": float(np.median([r["score_share"] for r in rows])), "max": float(max(r["score_share"] for r in rows))},
           "active": {"median": float(np.median([r["active"] for r in rows]))}, "elapsed_s": round(time.time() - t0, 1)}
out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
print(json.dumps(summary), flush=True)
