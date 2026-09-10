"""Aggregate the steering pilot (Codex 59e6b11 applied): admission first, then paired outcomes.

Admission: the manifest exists and its rows_written equals the rows on disk; the recipients on disk are
exactly the requested ones, unique, one episode each; every row carries every layer of the sweep and every
prescribed arm (operation only where the pair had a donor); every patched arm applied exactly once; the
prefill-vs-direct residual check is within 1e-4. A (row, layer) whose same-state arm did not reproduce the
baseline is VOID: kept as diagnostic, excluded from every effect count. An empty or incomplete store is
refused (exit 1), never summarised as vacuously clean.

Outcomes per layer and arm are paired states from the model's own baseline call to the arm's call, on
jointly complete calls: path — unchanged / switched_to_donor / changed_to_other, tool — the same; with the
counts of not-jointly-valid pairs (invalid baseline, invalid arm) beside, so the full cohort stays the
denominator. Donor agreement (the arm's path equals the donor's) is reported as a descriptive label and is
not a switch. Counts, not bounds: 24 recipients is a pilot.

usage: steer_pilot_analyze.py PILOT_DIR OUT_JSON [--diagnostic]
"""
import json, sys, statistics, argparse
from pathlib import Path
from collections import defaultdict, Counter
ap = argparse.ArgumentParser(); ap.add_argument("pilot"); ap.add_argument("out"); ap.add_argument("--diagnostic", action="store_true")
a = ap.parse_args(); D = Path(a.pilot)
rows = [json.loads(l) for l in (D / "pilot.jsonl").read_text().splitlines() if l.strip()] if (D / "pilot.jsonl").exists() else []
run = json.loads((D / "run.json").read_text()); man = json.loads((D / "manifest.json").read_text()) if (D / "manifest.json").exists() else None
layers = [str(L) for L in run["layers"]]; ARMS = ("same_state", "opposite_target", "pending_file", "operation", "unrelated_family")
requested = [p["recipient"] for p in run["pairs"]]; optional_missing = {p["recipient"] for p in run["pairs"] if p.get("operation") is None}
def share(k, n): return {"share": (k / n) if n else None, "k": k, "n": n}
def med(xs): xs = [x for x in xs if x is not None]; return {"n": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)} if xs else {"n": 0}
def margin(x, tool): lg = x["first_tool_logits"]; return lg[tool] - max(v for t, v in lg.items() if t != tool)

# ---- admission
problems = []; ids = [r["recipient"]["i"] for r in rows]
if man is None: problems.append("no manifest: the pass did not finish")
elif man.get("rows_written") != len(rows): problems.append(f"manifest rows_written {man.get('rows_written')} != rows on disk {len(rows)}")
if not rows: problems.append("no rows")
if sorted(ids) != sorted(requested): problems.append(f"recipients on disk {sorted(ids)[:6]}... != requested {sorted(requested)[:6]}...")
if len(set(ids)) != len(ids): problems.append("duplicate recipient rows")
bases = [r["recipient"]["base"] for r in rows]
if len(set(bases)) != len(bases): problems.append("a recipient episode repeats")
void = defaultdict(set); row_problems = defaultdict(list)
for r in rows:
    i = r["recipient"]["i"]
    if max(r.get("prefill_vs_direct_residual_max_abs_diff", {"x": 1.0}).values()) > 1e-4: row_problems[i].append("prefill residuals differ from the direct forward")
    for L in layers:
        lay = r["layers"].get(L)
        if lay is None: row_problems[i].append(f"layer {L} absent"); continue
        for arm in ARMS:
            if arm == "operation" and i in optional_missing: continue
            x = lay.get(arm)
            if x is None: row_problems[i].append(f"layer {L}: arm {arm} absent"); continue
            if x.get("patch_applied") != 1: row_problems[i].append(f"layer {L}: arm {arm} applied {x.get('patch_applied')} times")
        if lay.get("same_state", {}).get("identical_to_baseline") is not True: void[i].add(L)
eligible_rows = [r for r in rows if r["recipient"]["i"] not in row_problems]
verdict = "admitted" if not problems and not row_problems else "refused"

# ---- outcomes on eligible rows, per layer and arm, excluding void (row, layer) cells
per = {}; switch_counts = {arm: {L: 0 for L in layers} for arm in ARMS[1:]}
for L in layers:
    per[L] = {"void_rows": sum(1 for r in eligible_rows if L in void[r["recipient"]["i"]])}
    for arm in ARMS[1:]:
        xs = [(r, r["layers"][L][arm]) for r in eligible_rows if L not in void[r["recipient"]["i"]] and arm in r["layers"][L]]
        n = len(xs)
        if not n: per[L][arm] = {"n": 0}; continue
        ps = Counter(x["path_state"] for _, x in xs); ts = Counter(x["tool_state"] for _, x in xs)
        jv = sum(1 for _, x in xs if x["jointly_valid"])
        d = {"n": n, "jointly_valid": share(jv, n), "invalid_baseline": sum(1 for r, _ in xs if not r["baseline"]["valid_json"]), "invalid_arm": sum(1 for _, x in xs if not x["valid_json"]),
             "path_state": {k: ps.get(k, 0) for k in ("unchanged", "switched_to_donor", "changed_to_other")}, "tool_state": {k: ts.get(k, 0) for k in ("unchanged", "switched_to_donor", "changed_to_other")},
             "switched_to_donor_path_share_of_jointly_valid": share(ps.get("switched_to_donor", 0), jv), "switched_to_donor_tool_share_of_jointly_valid": share(ts.get("switched_to_donor", 0), jv),
             "donor_path_agreement_descriptive": share(sum(1 for _, x in xs if x["path_class"] == "donor"), n),
             "directory_switched_to_donor": share(sum(1 for _, x in xs if x["directory_switched_to_donor"]), sum(1 for _, x in xs if x["directory_switched_to_donor"] is not None)),
             "file_switched_same_directory": share(sum(1 for _, x in xs if x["file_switched_same_directory"]), sum(1 for _, x in xs if x["file_switched_same_directory"] is not None)),
             "delta_margin_expert_tool_vs_baseline": med([margin(x, r["recipient"]["tool"]) - margin(r["baseline"], r["recipient"]["tool"]) for r, x in xs])}
        if arm == "operation": d["delta_margin_donor_tool_vs_baseline"] = med([margin(x, r["donors"]["operation"]["tool"]) - margin(r["baseline"], r["donors"]["operation"]["tool"]) for r, x in xs])
        per[L][arm] = d
        switch_counts[arm][L] = ps.get("switched_to_donor", 0) + (ts.get("switched_to_donor", 0) if arm == "operation" else 0)
base = {"n": len(eligible_rows), "valid_json": share(sum(r["baseline"]["valid_json"] for r in eligible_rows), len(eligible_rows)),
        "expert_path_reproduced": share(sum(r["baseline"]["expert_path_reproduced"] for r in eligible_rows), len(eligible_rows)),
        "expert_tool_reproduced": share(sum(r["baseline"]["expert_tool_reproduced"] for r in eligible_rows), len(eligible_rows))}
out = {"basis": f"{D}; base 4B float32 eager greedy; whole-residual donor replacement at P_act at block L's output; the expert note teacher-forced; outcomes paired from the model's own baseline on jointly complete calls; unit = recipient decision, one per episode; counts, not bounds",
       "verdict": verdict, "strict": not a.diagnostic, "admission": {"rows": len(rows), "requested": len(requested), "eligible_rows": len(eligible_rows), "problems": problems, "row_problems": dict(row_problems),
                                                                    "void_row_layers": {str(i): sorted(v) for i, v in void.items()}, "unique_episodes": len(set(bases)), "manifest_present": man is not None},
       "layers": layers, "baseline": base, "per_layer_arm": per, "switch_counts_by_layer": switch_counts,
       "switch_definition": "path: the arm's complete call names the donor's expert path and the baseline's complete call did not; tool (operation arm): the arm's complete call names the donor's tool and the baseline's did not"}
Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"event": "pilot_analyzed", "verdict": verdict, "rows": len(rows), "eligible": len(eligible_rows), "problems": problems, "void": {str(i): sorted(v) for i, v in void.items()}, "switch_counts": switch_counts}))
if verdict == "refused" and not a.diagnostic: sys.exit(1)
