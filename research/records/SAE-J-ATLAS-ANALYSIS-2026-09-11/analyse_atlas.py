"""First analysis of the SAE/J atlas with the model's own prediction beside the expert's and the lens's.

Three readings, all on the 16 released cells (4 episodes x {P_note, P_act} x layers {18, 24}):
 (1) agreement — expert token, the model's six-tool argmax, the lens's, with each side's raw six-tool mass
     against W-5's 0.001 floor, so an argmax below the floor is reported as such rather than counted;
 (2) the tool margin decomposed — for the model's own pick against its runner-up, what share of the lens
     score gap the feature sum supplies and what share the dictionary's error supplies;
 (3) concentration and tool specificity — which feature supplies most of the separation, and whether it is
     active in the cells that choose a different tool.

All of this is algebra on the exported decomposition, which is exact to 1e-12. Reading a feature as "the
read_file feature" is an interpretation of that algebra and is not established by it.

usage: analyse_atlas.py ATLAS_JSON OUT_JSON
"""
import json, sys
from pathlib import Path
import numpy as np

d = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2])
tools = {t["token_id"]: t["name"] for t in d["tools"]}
ex = d["exported_token_ids"]; idx = {t: i for i, t in enumerate(ex)}
text = {x["id"]: x["text"] for x in d["tokenizer"]["tokens"]}
FLOOR = 1e-3
rows = []
for c in d["cells"]:
    mp = c["model_prediction"]["six_tool"]; lp = c["lens_prediction"]; a2 = c["a2_status"]
    probs = np.array(mp["probabilities"]); order = np.argsort(-probs)
    w_tok, r_tok = mp["winner_token_id"], mp["token_ids"][int(order[1])]
    A = np.array(c["contributions"]); S = np.array(c["scores"]); F = np.array(c["feature_sum"]); E = np.array(c["error"])
    w, r = idx[w_tok], idx[r_tok]; gap = float(S[w] - S[r]); sep = A[:, w] - A[:, r]
    k = int(np.argmax(np.abs(sep))); srt = np.sort(sep)[::-1]
    rows.append({
        "id": c["id"], "episode": c["episode_id"], "layer": c["layer"], "position": c["position"],
        "expert_token": text.get(c["emitted_token_id"], str(c["emitted_token_id"])), "expert_is_tool": tools.get(c["emitted_token_id"]),
        "model_pick": tools[w_tok], "model_runner_up": tools[r_tok], "model_raw_mass": mp.get("raw_mass"),
        "model_resolved": bool(mp.get("raw_mass", 0) >= FLOOR),
        "lens_pick": tools.get(lp.get("winner_token_id")) if lp.get("status") == "measured" else None,
        "lens_raw_mass": lp.get("raw_mass"), "lens_resolved": bool((lp.get("raw_mass") or 0) >= FLOOR),
        "lens_matches_model": (tools.get(lp.get("winner_token_id")) == tools[w_tok]) if lp.get("status") == "measured" else None,
        "model_matches_expert": tools.get(c["emitted_token_id"]) == tools[w_tok] if tools.get(c["emitted_token_id"]) else None,
        "lens_gap_model_pick_over_runner_up": gap,
        "feature_share_of_gap": float((F[w] - F[r]) / gap) if gap else None,
        "error_share_of_gap": float((E[w] - E[r]) / gap) if gap else None,
        "top_feature": int(c["feature_ids"][k]), "top_feature_share_of_gap": float(sep[k] / gap) if gap else None,
        "top1_share_of_feature_separation": float(srt[0] / srt.sum()) if srt.sum() else None,
        "top3_share_of_feature_separation": float(srt[:3].sum() / srt.sum()) if srt.sum() else None,
        "active_features": len(c["feature_ids"]),
        "raw_reconstruction_share": a2["raw_reconstruction_share"], "lens_score_error_share": a2["lens_score_error_share"],
        "a2_ranked_under_budget": bool(a2["ranked"]),
    })
# tool specificity of each cell's top feature, at the licensed site
site = [r for r in rows if r["layer"] == 24 and r["position"] == "P_act"]
spec = {}
for f in sorted({r["top_feature"] for r in site}):
    entries = []
    for c in d["cells"]:
        if c["layer"] != 24 or c["position"] != "P_act": continue
        mp = c["model_prediction"]["six_tool"]; ids = c["feature_ids"]
        if f not in ids:
            entries.append({"episode": c["episode_id"], "model_pick": tools[mp["winner_token_id"]], "active": False}); continue
        i = ids.index(f); A = np.array(c["contributions"])[i]
        per = {tools[t]: float(A[idx[t]]) for t in tools if t in idx}
        entries.append({"episode": c["episode_id"], "model_pick": tools[mp["winner_token_id"]], "active": True,
                        "activation": float(c["activity"][i]), "pushes_most_toward": max(per, key=per.get), "per_tool": per})
    spec[str(f)] = entries
summary = {}
for L in (18, 24):
    for p in ("P_note", "P_act"):
        g = [r for r in rows if r["layer"] == L and r["position"] == p]
        summary[f"L{L}/{p}"] = {
            "cells": len(g),
            "lens_argmax_matches_model": sum(bool(r["lens_matches_model"]) for r in g),
            "lens_resolved_above_floor": sum(r["lens_resolved"] for r in g),
            "model_resolved_above_floor": sum(r["model_resolved"] for r in g),
            "model_matches_expert": sum(1 for r in g if r["model_matches_expert"]),
            "expert_token_is_a_tool": sum(1 for r in g if r["expert_is_tool"]),
            "lens_gap_positive": sum(1 for r in g if r["lens_gap_model_pick_over_runner_up"] > 0),
            "median_feature_share_of_gap": float(np.median([r["feature_share_of_gap"] for r in g])),
            "median_error_share_of_gap": float(np.median([r["error_share_of_gap"] for r in g])),
            "median_raw_reconstruction_share": float(np.median([r["raw_reconstruction_share"] for r in g])),
            "a2_ranked_under_budget": sum(r["a2_ranked_under_budget"] for r in g),
        }
out.write_text(json.dumps({"basis": __doc__, "bundle_id": d["bundle_id"], "mass_floor": FLOOR,
                           "interpretation_limits": d["interpretation_limits"], "labels": d["labels"], "labels_reason": d["labels_reason"],
                           "summary": summary, "tool_specificity_at_L24_P_act": spec, "cells": rows}, indent=1) + "\n")
print(json.dumps(summary, indent=1))
