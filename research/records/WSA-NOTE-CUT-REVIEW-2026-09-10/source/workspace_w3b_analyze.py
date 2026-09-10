"""Aggregate W-3b: per arm, the change in the model's own p(taken tool) at P_act and P_note and in the
lens's last-layer p(taken) at P_act, against the unmasked forward and the random control; the gate,
with the blocked-edge counts and the current-note arm's skip count (schema 2, after Codex 712f78c).
usage: workspace_w3b_analyze.py W3B_DIR OUT_JSON [--mass-floor 1e-3]"""
import json, sys, statistics, argparse
from pathlib import Path
from collections import defaultdict
ap = argparse.ArgumentParser(); ap.add_argument("w3b"); ap.add_argument("out"); ap.add_argument("--mass-floor", type=float, default=1e-3); a = ap.parse_args()
rows = [json.loads(l) for l in (Path(a.w3b) / "w3b.jsonl").read_text().splitlines() if l.strip()]
man = json.loads((Path(a.w3b) / "manifest.json").read_text()); FLOOR = a.mass_floor
def tail(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs: return {"n": 0}
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"n": len(xs), "median": statistics.median(xs), "p10": q(0.1), "p90": q(0.9), "min": xs[0], "max": xs[-1]}
per_arm = defaultdict(lambda: defaultdict(list)); gates = defaultdict(list); skipped = defaultdict(int); arm_rows = defaultdict(int)
p_note_identical = []
for r in rows:
    base = r["arms"]["unmasked"]; ti = r["tool_idx"]; last = max(base["lens_act"], key=int)
    b_act = base["model_six_act"][0][ti] if base["model_six_act"][1] >= FLOOR else None
    b_lens = base["lens_act"][last]["six"][ti] if base["lens_act"][last]["mass"] >= FLOOR else None
    for arm_name, why in r.get("skipped_arms", {}).items(): skipped[arm_name] += 1
    for arm, x in r["arms"].items():
        if arm == "unmasked": continue
        arm_rows[arm] += 1
        p_act = x["model_six_act"][0][ti] if x["model_six_act"][1] >= FLOOR else None
        p_lens = x["lens_act"][last]["six"][ti] if x["lens_act"][last]["mass"] >= FLOOR else None
        per_arm[arm]["delta_model_p_taken_at_act"].append(None if (p_act is None or b_act is None) else p_act - b_act)
        per_arm[arm]["taken_still_top_at_act"].append(None if x["model_six_act"][1] < FLOOR else int(max(range(6), key=lambda j: x["model_six_act"][0][j]) == ti))
        per_arm[arm]["delta_lens_last_p_taken_at_act"].append(None if (p_lens is None or b_lens is None) else p_lens - b_lens)
        per_arm[arm]["unresolved_at_act"].append(int(x["model_six_act"][1] < FLOOR))
        if "margin_act_logits" in x and "margin_act_logits" in base:
            per_arm[arm]["delta_margin_act_logits"].append(x["margin_act_logits"] - base["margin_act_logits"])
            per_arm[arm]["delta_margin_note_logits"].append(x["margin_note_logits"] - base["margin_note_logits"])
        if "gate" in x: gates[arm].append(x["gate"])
        if arm == "current_note" and "p_note_identical_to_unmasked" in x: p_note_identical.append(int(x["p_note_identical_to_unmasked"]))
def gate_summary(gs):
    if not gs: return {"n": 0}
    out = {"n": len(gs), "attention_on_masked_keys_max": max(g["attention_on_masked_keys"] for g in gs), "local_mass_beyond_window_max": max(g["local_mass_beyond_window"] for g in gs)}
    if "attention_on_masked_edges_all_queries" in gs[0]:
        out["attention_on_masked_edges_all_queries_max"] = max(g["attention_on_masked_edges_all_queries"] for g in gs)
        out["n_blocked_edges_at_P_act"] = tail([g["n_blocked_edges_at_P_act"] for g in gs]); out["rows_with_zero_blocked_at_P_act"] = sum(1 for g in gs if g["n_blocked_edges_at_P_act"] == 0)
    return out
all_gates = [g for gs in gates.values() for g in gs]
out = {"basis": f"{a.w3b}; queries masked from {man['queries_masked_from']}; control: {man['control']}; unit = decision (the sample has one decision per episode by rule only if so sampled); mass floor {FLOOR}",
       "schema_version": man.get("schema_version", 1), "n": len(rows), "rows_written": man.get("rows_written"),
       "gate": gate_summary(all_gates), "gate_per_arm": {arm: gate_summary(gs) for arm, gs in gates.items()},
       "arm_row_counts": dict(arm_rows), "skipped_arm_counts": dict(skipped),
       "current_note_p_note_identical_to_unmasked": {"n": len(p_note_identical), "share": (sum(p_note_identical) / len(p_note_identical)) if p_note_identical else None},
       "note_prose_tokens": tail([r.get("note_prose_tokens") for r in rows]), "note_syntax_tokens": tail([r.get("note_syntax_tokens") for r in rows]),
       "arms": {arm: {k: (tail(v) if k != "taken_still_top_at_act" and k != "unresolved_at_act" else {"share": (sum(x for x in v if x is not None) / max(1, sum(1 for x in v if x is not None))), "n": sum(1 for x in v if x is not None)}) for k, v in d.items()} for arm, d in per_arm.items()}}
Path(a.out).write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps({"event": "w3b_analyzed", "n": len(rows), "arms": list(per_arm), "skipped": dict(skipped)}))
