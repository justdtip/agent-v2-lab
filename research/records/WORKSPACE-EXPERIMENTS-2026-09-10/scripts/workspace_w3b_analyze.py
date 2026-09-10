"""Aggregate W-3b (schema 2/3), admitting only verified rows and arms (Codex 63d0549, F2/F3).

Admission, before any estimate: the manifest (or run.json when the pass did not finish) names the
requested sample; the rows on disk must be exactly that set, unique; every present arm must carry a
complete gate with finite values, zero attention on every masked edge, zero local mass beyond the
window, at least one blocked edge at P_act; the current-note arm must report P_note identical to the
unmasked forward; a v3 row's receipts must show no missing or unexpected edge and the placement oracle
passed. A row that fails is *ineligible* and is counted with its reasons, never averaged. In strict mode
(default) any ineligible row or an incomplete set makes the exit code 1 and the output carries
"verdict": "refused"; --diagnostic writes the summary anyway, labelled ineligible.

Per arm, on rows where the unmasked baseline and the arm are both resolved (six-tool mass >= floor) and
untied: the paired transition of the EXPERT tool's argmax status — retained / lost / gained / neither —
with unresolved-baseline, unresolved-arm and tie counts kept separately; the old "still top" is kept as
masked_expert_top_agreement with a null share when its denominator is empty. Deltas are tails over
jointly resolved rows; the comparison with the random control is paired on rows where both are
eligible. Unit = decision; unique episodes are counted and reported, not used as the unit.

usage: workspace_w3b_analyze.py W3B_DIR OUT_JSON [--mass-floor 1e-3] [--diagnostic]
"""
import json, sys, math, statistics, argparse
from pathlib import Path
from collections import defaultdict, Counter
ap = argparse.ArgumentParser(); ap.add_argument("w3b"); ap.add_argument("out"); ap.add_argument("--mass-floor", type=float, default=1e-3)
ap.add_argument("--diagnostic", action="store_true", help="write the summary even when rows are ineligible (labelled), exit 0")
a = ap.parse_args(); D = Path(a.w3b); FLOOR = a.mass_floor
rows = [json.loads(l) for l in (D / "w3b.jsonl").read_text().splitlines() if l.strip()]
man = json.loads((D / "manifest.json").read_text()) if (D / "manifest.json").exists() else None
run = json.loads((D / "run.json").read_text()) if (D / "run.json").exists() else None
requested = (man or run or {}).get("sample")
GATE_FIELDS = ("attention_on_masked_keys", "attention_on_masked_edges_all_queries", "local_mass_beyond_window", "n_blocked_edges_at_P_act", "n_blocked_edges_total")
def finite(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
def tail(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs: return {"n": 0}
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"n": len(xs), "median": statistics.median(xs), "p10": q(0.1), "p90": q(0.9), "min": xs[0], "max": xs[-1]}
def share(k, n): return {"share": (k / n) if n else None, "k": k, "n": n}
def top(six, mass):  # resolved argmax, or None when below the floor or tied
    if mass is None or mass < FLOOR: return None
    s = sorted(range(6), key=lambda j: -six[j])
    return None if six[s[0]] == six[s[1]] else s[0]

# ---- admission
problems = defaultdict(list); ids = [r["i"] for r in rows]; dup = [i for i, c in Counter(ids).items() if c > 1]
set_problems = []
if requested is None: set_problems.append("no manifest or run.json names the requested sample")
else:
    missing = sorted(set(requested) - set(ids)); extra = sorted(set(ids) - set(requested))
    if missing: set_problems.append(f"{len(missing)} requested rows absent (first {missing[:5]})")
    if extra: set_problems.append(f"{len(extra)} rows not requested (first {extra[:5]})")
if dup: set_problems.append(f"duplicate row ids {dup[:5]}")
if man and man.get("rows_written") is not None and man["rows_written"] != len(rows): set_problems.append(f"manifest rows_written {man['rows_written']} != rows on disk {len(rows)} (a stale manifest beside a rerun?)")
schema = (man or run or {}).get("schema_version")
for r in rows:
    why = []
    if "unmasked" not in r.get("arms", {}): why.append("no unmasked arm")
    for arm, x in r.get("arms", {}).items():
        if arm == "unmasked": continue
        g = x.get("gate")
        if not isinstance(g, dict): why.append(f"{arm}: gate absent"); continue
        for f in GATE_FIELDS:
            if f not in g or not finite(g[f]): why.append(f"{arm}: gate field {f} absent or not finite")
        if all(f in g and finite(g[f]) for f in GATE_FIELDS):
            if g["attention_on_masked_keys"] != 0.0 or g["attention_on_masked_edges_all_queries"] != 0.0: why.append(f"{arm}: attention on masked edges")
            if g["local_mass_beyond_window"] != 0.0: why.append(f"{arm}: local mass beyond the window")
            if g["n_blocked_edges_at_P_act"] <= 0: why.append(f"{arm}: no blocked edge at P_act")
        if arm in ("current_note", "random_equal_count_note") and x.get("p_note_identical_to_unmasked") is not True: why.append(f"{arm}: P_note not identical to unmasked")
        rc = x.get("receipt")
        if rc is not None:  # v3 rows carry receipts; absence is recorded, not failed (pre-v3 producer)
            if rc.get("missing_edges") or rc.get("unexpected_edges"): why.append(f"{arm}: receipt mismatch (missing {rc.get('missing_edges')}, unexpected {rc.get('unexpected_edges')})")
            if rc.get("causality_preserved") is False or rc.get("pre_cut_queries_still_attend") is False: why.append(f"{arm}: effective-mask check failed")
    if r.get("placement_oracle") is False: why.append("placement oracle failed")
    if why: problems[r["i"]] = why
eligible = [r for r in rows if r["i"] not in problems and r["i"] not in dup]
receipts_present = sum(1 for r in rows for x in r.get("arms", {}).values() if isinstance(x, dict) and x.get("receipt") is not None)
complete = not set_problems and not problems
verdict = "admitted" if complete else "refused"

# ---- estimates on eligible rows only
per_arm = defaultdict(lambda: defaultdict(list)); trans = defaultdict(Counter); agree = defaultdict(lambda: [0, 0]); paired_random = defaultdict(list); ctrl_used = {}
script_version = (man or run or {}).get("script_version")
CONTROL_STATUS = ("v3.2 pool: positions 1..P_note-1 in no carrier span (task statement and format tokens), <bos> never; random_equal_count (all-carriers count, queries from P_note) pairs the carrier arms, random_equal_count_note (the note's prose count, queries from the fence) pairs the current-note arm"
                  + ("; all_carriers_matched (v3.3) is the carrier arm count-matched to random_equal_count in every row" if script_version == "3.3" else "; NOTE: in rows where the carriers outnumber the pool the control removed fewer keys than all_carriers (v3.2 has no matched carrier arm)")
                  if script_version in ("3.2", "3.3") else
                  "v3.1 pool: the non-task tokens before P_note — the carrier spans plus the format tokens, so the draw was mostly the all-carriers mask itself (4B pass: three quarters of its keys carrier tokens, <bos> masked in 152 of 187 rows); DEGENERATE — paired_vs_random is not a control comparison in this output, and the current-note arm has no count-matched control")
skipped = Counter(); arm_rows = Counter(); gates = defaultdict(list)
for r in eligible:
    base = r["arms"]["unmasked"]; ti = r["tool_idx"]; last = max(base["lens_act"], key=int)
    b_top = top(base["model_six_act"][0], base["model_six_act"][1]); b_mass = base["model_six_act"][1]
    b_p = base["model_six_act"][0][ti] if b_mass >= FLOOR else None
    b_lens = base["lens_act"][last]["six"][ti] if base["lens_act"][last]["mass"] >= FLOOR else None
    for arm_name in r.get("skipped_arms", {}): skipped[arm_name] += 1
    deltas_here = {}
    for arm, x in r["arms"].items():
        if arm == "unmasked": continue
        arm_rows[arm] += 1; gates[arm].append(x["gate"])
        a_top = top(x["model_six_act"][0], x["model_six_act"][1]); a_mass = x["model_six_act"][1]
        if b_mass < FLOOR: trans[arm]["baseline_unresolved"] += 1
        elif a_mass < FLOOR: trans[arm]["arm_unresolved"] += 1
        elif b_top is None or a_top is None: trans[arm]["tie"] += 1
        else:
            was, now = b_top == ti, a_top == ti
            trans[arm]["retained" if was and now else "lost" if was and not now else "gained" if now and not was else "neither"] += 1
        if a_mass >= FLOOR and a_top is not None: agree[arm][0] += int(a_top == ti); agree[arm][1] += 1
        p_a = x["model_six_act"][0][ti] if a_mass >= FLOOR else None
        d_p = None if (p_a is None or b_p is None) else p_a - b_p
        per_arm[arm]["delta_model_p_taken_at_act"].append(d_p)
        p_l = x["lens_act"][last]["six"][ti] if x["lens_act"][last]["mass"] >= FLOOR else None
        per_arm[arm]["delta_lens_last_p_taken_at_act"].append(None if (p_l is None or b_lens is None) else p_l - b_lens)
        per_arm[arm]["unresolved_at_act"].append(int(a_mass < FLOOR))
        d_m = x["margin_act_logits"] - base["margin_act_logits"]
        per_arm[arm]["delta_margin_act_logits"].append(d_m); per_arm[arm]["delta_margin_note_logits"].append(x["margin_note_logits"] - base["margin_note_logits"])
        deltas_here[arm] = (d_p, d_m)
    for arm, (dp, dm) in deltas_here.items():
        if arm.startswith("random_equal_count"): continue
        ctrl = "random_equal_count_note" if arm == "current_note" else "random_equal_count"
        if ctrl not in deltas_here: continue
        rp, rm = deltas_here[ctrl]; ctrl_used[arm] = ctrl
        paired_random[arm].append({"d_margin_minus_random": dm - rm, "d_p_minus_random": None if (dp is None or rp is None) else dp - rp})
def gate_summary(gs):
    if not gs: return {"n": 0}
    fields = set().union(*(set(g) for g in gs))
    out = {"n": len(gs), "field_sets_identical": all(set(g) == fields for g in gs)}
    for f in sorted(fields):
        vals = [g[f] for g in gs if f in g and finite(g[f])]
        out[f + "_max"] = max(vals) if vals else None; out[f + "_min"] = min(vals) if vals else None
    return out
out = {"basis": f"{a.w3b}; schema {schema}; queries masked from {(man or run or {}).get('queries_masked_from')}; control: {(man or run or {}).get('control')}; mass floor {FLOOR}; unit = decision",
       "verdict": verdict, "strict": not a.diagnostic, "script_version": script_version, "control_status": CONTROL_STATUS,
       "carrier_control_short_rows": sum(1 for r in eligible if (rc := (r.get("random_control") or {}).get("random_equal_count")) and rc["picked"] < rc["requested"]),
       "all_carriers_matched_subsampled_rows": sum(1 for r in eligible if ((r.get("random_control") or {}).get("all_carriers_matched") or {}).get("subsampled")),
       "admission": {"rows_on_disk": len(rows), "requested": (len(requested) if requested is not None else None), "eligible": len(eligible), "ineligible": len(problems), "set_problems": set_problems,
                     "ineligible_rows": {str(i): w for i, w in list(problems.items())[:50]}, "receipts_present_arms": receipts_present, "receipts_note": "receipts and the placement oracle exist from producer v3; rows without them are admitted on the schema-2 gates only and say so here",
                     "unique_episodes_eligible": len({r["task_id"] for r in eligible}), "manifest_present": man is not None, "run_json_present": run is not None},
       "arm_row_counts": dict(arm_rows), "skipped_arm_counts": dict(skipped), "gate_per_arm": {arm: gate_summary(gs) for arm, gs in gates.items()},
       "note_prose_tokens": tail([r.get("note_prose_tokens") for r in eligible]), "note_syntax_tokens": tail([r.get("note_syntax_tokens") for r in eligible]),
       "arms": {arm: {"expert_top_transition_paired": dict(trans[arm]),
                      "retained_share_of_jointly_resolved": share(trans[arm]["retained"], trans[arm]["retained"] + trans[arm]["lost"]),
                      "masked_expert_top_agreement": share(agree[arm][0], agree[arm][1]),
                      **{k: (tail(v) if k != "unresolved_at_act" else share(sum(v), len(v))) for k, v in d.items()},
                      "paired_vs_random": {"control_arm": ctrl_used.get(arm), "control_status": CONTROL_STATUS, "n": len(paired_random[arm]), "d_margin_minus_random": tail([q["d_margin_minus_random"] for q in paired_random[arm]]), "d_p_minus_random": tail([q["d_p_minus_random"] for q in paired_random[arm]])} if not arm.startswith("random_equal_count") else None}
                for arm, d in per_arm.items()}}
Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"event": "w3b_analyzed", "verdict": verdict, "rows": len(rows), "eligible": len(eligible), "ineligible": len(problems), "set_problems": set_problems, "arms": list(per_arm)}))
if verdict == "refused" and not a.diagnostic: sys.exit(1)
