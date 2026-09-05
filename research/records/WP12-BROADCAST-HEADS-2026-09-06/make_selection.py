"""Post-process the null-distribution stage-1 run into out/selection_nulldist.json and re-score the predictions."""
import json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / "out"
d = json.load(open(OUT / "broadcast_heads_nulldist.json")); rows = d["rows"]
att = [r for r in rows if r["channel"] == "attention" and r["layer_written"] != 32]; rec = [r for r in rows if r["channel"] == "recurrent"]
sel = sorted([r for r in att if r["broadcasts_J"]], key=lambda r: -r["mrr_margin"])
by_layer = {L: sum(1 for r in sel if r["layer_written"] == L) for L in sorted({r["layer_written"] for r in att})}
inband = [r for r in sel if r["layer_written"] in (16, 20, 24, 28)]
first_half = sum(1 for r in inband if r["layer_written"] in (16, 20)); second_half = len(inband) - first_half
retr = [(19, 2), (23, 9), (19, 12), (27, 1), (3, 15), (19, 15)]
status = {}
for b, h in retr:
    r = next(x for x in att if x["block"] == b and x["head"] == h)
    status[f"b{b}_h{h}"] = {"selected": r["broadcasts_J"], "J_mrr": r["J_mrr"], "rot_med": r["J_rot_mrr"], "rot_max": r["J_rot_mrr_max"], "mlp_med": r["MLP_rows_mrr"], "mlp_max": r["MLP_rows_mrr_max"], "margin": r["mrr_margin"], "rank": next(i + 1 for i, x in enumerate(sorted(att, key=lambda x: -x["mrr_margin"])) if x["block"] == b and x["head"] == h)}
rec_sel = [r for r in rec if r["broadcasts_J"]]
out = {"rule": "J label preservation beyond every one of 20 rotated-J draws and 20 MLP-row draws, and at least double their medians; layer 32 excluded; gain reported beside; ordered by preservation margin",
       "selected_attention": [{k: r[k] for k in ("layer_written", "block", "head", "J_mrr", "J_rot_mrr", "J_rot_mrr_max", "MLP_rows_mrr", "MLP_rows_mrr_max", "J_gain", "J_rot_gain", "MLP_rows_gain", "mrr_margin", "gain_margin")} for r in sel],
       "n_of_N": [len(sel), len(att)], "by_layer": by_layer, "in_band": len(inband), "first_half_of_band": first_half, "second_half_of_band": second_half,
       "prediction_1_first_half": "refuted" if second_half > first_half else ("confirmed" if first_half > second_half else "tied"),
       "retrieval_heads": status, "recurrent": {"selected": len(rec_sel), "of": len(rec), "median_J_mrr": float(np.median([r["J_mrr"] for r in rec])), "median_MLP_control": float(np.median([r["MLP_rows_mrr"] for r in rec])), "strongest_J_mrr": float(max(r["J_mrr"] for r in rec)), "attention_median_J_mrr": float(np.median([r["J_mrr"] for r in att])), "recurrent_at_or_above_attention_median": int(sum(r["J_mrr"] >= np.median([x["J_mrr"] for x in att]) for r in rec))},
       "gain_among_selected": {"median_J": float(np.median([r["J_gain"] for r in sel])) if sel else None, "fraction_above_both_controls": float(np.mean([r["gain_margin"] > 0 for r in sel])) if sel else None},
       "copy_maps": {L: float(np.median([r["J_rot_mrr"] for r in att if r["layer_written"] == L])) for L in sorted({r["layer_written"] for r in att})}}
json.dump(out, open(OUT / "selection_nulldist.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "selected_attention"}, indent=1))
print("selected:", [(r["layer_written"], r["block"], r["head"], round(r["J_mrr"], 3), round(max(r["J_rot_mrr_max"], r["MLP_rows_mrr_max"]), 3)) for r in sel])
