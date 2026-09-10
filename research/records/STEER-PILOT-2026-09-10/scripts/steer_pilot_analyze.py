"""Aggregate the steering pilot: per layer and arm, what the generated call did — the tool kept or switched,
the path classified against the recipient's expert path, the donor's, and the recipient's own baseline;
the same-state control's identity; the baseline's agreement with the expert; first-token logit margins.
Counts, not bounds: 24 recipients is a pilot. usage: steer_pilot_analyze.py PILOT_DIR OUT_JSON"""
import json, sys, statistics
from pathlib import Path
from collections import defaultdict, Counter
D = Path(sys.argv[1]); out_path = Path(sys.argv[2])
rows = [json.loads(l) for l in (D / "pilot.jsonl").read_text().splitlines() if l.strip()]
run = json.loads((D / "run.json").read_text()); man = json.loads((D / "manifest.json").read_text()) if (D / "manifest.json").exists() else None
layers = [str(L) for L in run["layers"]]
def share(k, n): return {"share": (k / n) if n else None, "k": k, "n": n}
def med(xs): xs = [x for x in xs if x is not None]; return {"n": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)} if xs else {"n": 0}
def margin(x, tool):  # the named tool's first-token logit minus the best other tool's
    lg = x["first_tool_logits"]; return lg[tool] - max(v for t, v in lg.items() if t != tool)
base = {"n": len(rows), "expert_path_reproduced": share(sum(r["baseline"]["expert_path_reproduced"] for r in rows), len(rows)),
        "expert_tool_reproduced": share(sum(r["baseline"]["expert_tool_reproduced"] for r in rows), len(rows)),
        "valid_json": share(sum(r["baseline"]["valid_json"] for r in rows), len(rows)),
        "path_class": dict(Counter(r["baseline"]["path_class"] for r in rows))}
per = {}
for L in layers:
    per[L] = {}
    for arm in ("same_state", "opposite_target", "pending_file", "operation", "unrelated_family"):
        xs = [(r, r["layers"][L][arm]) for r in rows if arm in r["layers"][L]]
        if not xs: per[L][arm] = {"n": 0}; continue
        n = len(xs); d = {"n": n, "tool_preserved": share(sum(x["tool_preserved"] for _, x in xs), n), "valid_json": share(sum(x["valid_json"] for _, x in xs), n),
                          "path_class": dict(Counter(x["path_class"] for _, x in xs)), "path_equals_baseline": share(sum(1 for _, x in xs if x["path_equals_baseline"]), sum(1 for _, x in xs if x["path_equals_baseline"] is not None)),
                          "directory_switched_to_donor": share(sum(x["directory_switched"] for _, x in xs), n), "file_switched_same_directory": share(sum(x["file_switched_same_directory"] for _, x in xs), n),
                          "tool_equals_donor": share(sum(x["tool_equals_donor"] for _, x in xs), n),
                          "delta_margin_expert_tool_vs_baseline": med([margin(x, r["recipient"]["tool"]) - margin(r["baseline"], r["recipient"]["tool"]) for r, x in xs])}
        if arm == "same_state": d["identical_to_baseline"] = share(sum(x["identical_to_baseline"] for _, x in xs), n)
        if arm == "operation": d["delta_margin_donor_tool_vs_baseline"] = med([margin(x, r["donors"]["operation"]["tool"]) - margin(r["baseline"], r["donors"]["operation"]["tool"]) for r, x in xs])
        per[L][arm] = d
# per recipient: at which layers each arm switched (path to the donor's, or the tool to the donor's for operation)
switch_layers = {arm: Counter() for arm in ("opposite_target", "pending_file", "operation", "unrelated_family")}
for r in rows:
    for arm in switch_layers:
        for L in layers:
            x = r["layers"][L].get(arm)
            if x and (x["path_class"] == "donor" or (arm == "operation" and x["tool_equals_donor"])): switch_layers[arm][L] += 1
out = {"basis": f"{D}; base 4B float32 eager greedy; whole-residual donor replacement at P_act at block L's output; the expert note teacher-forced; unit = recipient decision; counts, not bounds",
       "recipients": len(rows), "rows_written": (man or {}).get("rows_written"), "layers": layers, "baseline": base, "per_layer_arm": per,
       "switch_counts_by_layer": {arm: {L: c.get(L, 0) for L in layers} for arm, c in switch_layers.items()},
       "same_state_identical_everywhere": all(r["layers"][L]["same_state"]["identical_to_baseline"] for r in rows for L in layers)}
out_path.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"event": "pilot_analyzed", "recipients": len(rows), "same_state_identical_everywhere": out["same_state_identical_everywhere"], "baseline_expert_path": base["expert_path_reproduced"], "switch_counts": out["switch_counts_by_layer"]}))
