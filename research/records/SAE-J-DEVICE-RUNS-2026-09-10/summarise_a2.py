"""Summarise a device_sae_bridge result: A1's four numbers and, per position, A2's error shares, the
ratio of the lens-score share to the raw share (how much the lens amplifies the dictionary's residual),
the identity gaps, the active-feature counts and the ranking verdicts. Counts and quantiles only.

usage: summarise_a2.py RESULT_JSON OUT_JSON
"""
import json, sys
from collections import Counter
import numpy as np
r = json.loads(open(sys.argv[1]).read()); out = {"status": r["status"], "stage": r["stage"], "layer": r["provenance"]["bridge"]["dictionary"].get("layer") if isinstance(r["provenance"]["bridge"].get("dictionary"), dict) else None}
def q(xs): xs = np.asarray([x for x in xs if x is not None], dtype=float); return {"n": int(xs.size), "min": float(xs.min()), "median": float(np.median(xs)), "p90": float(np.quantile(xs, 0.9)), "max": float(xs.max())} if xs.size else {"n": 0}
if "a1" in r:
    a = r["a1"]; ts = np.asarray(a["top_scores"]); top1 = Counter(np.asarray(a["top_tokens"])[:, 0].tolist())
    out["a1"] = {"convention": a["convention_check"], "two_products": a["two_products"], "negative_control": a["negative_control"],
                 "top_scores": q(ts.ravel()), "distinct_top1_tokens": len(top1), "most_common_top1_share": top1.most_common(1)[0][1] / ts.shape[0]}
if "a2" in r:
    out["a2"] = {}
    for pos in sorted({c["position"] for c in r["a2"]}):
        cells = [c for c in r["a2"] if c["position"] == pos]
        raw = [c["raw_reconstruction_share"] for c in cells]; sc = [c["lens_score_error_share"] for c in cells]
        ratio = [s / w for s, w in zip(sc, raw) if s is not None and w]
        out["a2"][pos] = {"cells": len(cells), "raw_reconstruction_share": q(raw), "lens_score_error_share": q(sc), "score_share_over_raw_share": q(ratio),
                          "identity_gap": q([c["identity_gap"] for c in cells]), "identity_gap_float32": q([c.get("identity_gap_float32") for c in cells]),
                          "active_features": q([c["active_features"] for c in cells]), "score": q([c["score"] for c in cells]),
                          "raw_admissible": sum(bool(c["raw_admissible"]) for c in cells), "ranked": sum(bool(c["ranked"]) for c in cells),
                          "reasons": dict(Counter(c.get("reason", "ranked") for c in cells))}
    out["a2"]["error_budget"] = r["error_budget"]
open(sys.argv[2], "w").write(json.dumps(out, indent=1) + "\n"); print(json.dumps({k: (v if k != "a2" else {p: (x["ranked"], x["raw_admissible"], round(x["lens_score_error_share"]["median"], 3)) for p, x in v.items() if p != "error_budget"}) for k, v in out.items() if k in ("status", "a2")}))
