"""Workspace experiments, set 1 — the analyses (W-1, W-3, W-4 here; W-2 in workspace_w2.py).
Reads a capture directory (index.jsonl, sample.jsonl, attn_*.npy, manifest.json) and writes
w1.json, w3.json, w4.json with every figure carrying its basis, its unit, and its tail.

usage: workspace_analyze.py CAPTURE_DIR OUT_DIR [--mass-floor 1e-3] [--corpus JSONL --snapshot DIR]
With --corpus and --snapshot the completion prefix is split at the ```json fence into the prose note and
the syntax after it (the repair rule of workspace_w3b.py, Codex 712f78c), for W-3's kinds and W-4's trajectory.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("capture"); ap.add_argument("out"); ap.add_argument("--mass-floor", type=float, default=1e-3)
ap.add_argument("--corpus", default=None); ap.add_argument("--snapshot", default=None)
args = ap.parse_args(); CAP = Path(args.capture); OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
man = json.loads((CAP / "manifest.json").read_text()); TOOLS = man["tools"]; L = man["repo_layers"]; nL = len(L)
idx = [json.loads(l) for l in (CAP / "index.jsonl").read_text().splitlines() if l.strip()]
BASIS = f"computed from the capture at {CAP} (model {man['model']}, float32, width 1); unit of any bound: the episode"
FLOOR = args.mass_floor

def tail(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return {"n": 0}
    xs = sorted(xs); q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"n": len(xs), "median": statistics.median(xs), "min": xs[0], "p10": q(0.1), "p90": q(0.9), "max": xs[-1]}

prior = Counter(r["tool"] for r in idx); Nd = len(idx)
PROSE = {}  # sample row i -> completion tokens that start before the ```json fence (the prose note, its trailing newline included)
if args.corpus and args.snapshot:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.snapshot); rows_all, seen = [], set()
    for line in open(args.corpus):
        r = json.loads(line); m = r.get("metadata", {})
        if "task_id" not in m: continue
        k = (m["task_id"], m["step"])
        if k in seen: continue
        seen.add(k); rows_all.append(r)
    assert len(rows_all) >= Nd, f"the corpus has {len(rows_all)} distinct decisions, the capture {Nd}"  # a --max-rows test capture holds a prefix; every row is identity-checked below
    for r in idx: assert (rows_all[r["i"]]["metadata"]["task_id"], rows_all[r["i"]]["metadata"]["step"]) == (r["task_id"], r["step"]), f"row {r['i']}: corpus/capture identity mismatch"
    for i in man["sample"]:
        comp = rows_all[i]["completion"]; note_end = max(comp.find("```json"), 0); enc = tok(comp, add_special_tokens=False, return_offsets_mapping=True)
        PROSE[i] = sum(1 for (a, b) in enc.offset_mapping if a < note_end)
prior_p = {t: prior[t] / Nd for t in TOOLS}
episodes = {r["task_id"] for r in idx}

def profile(r, where, which):  # per layer: (p of the expert action among the six, top tool, mass) with the floor
    out = []
    for l in L:
        e = r[where][which][str(l)]; six, mass = e["six"], e["mass"]
        if mass < FLOOR: out.append((None, None, mass)); continue
        out.append((six[r["tool_idx"]], TOOLS[int(np.argmax(six))], mass))
    return out

def log_odds(r, where, which):  # per layer: log p(expert) - log max p(other) among the six, and log full-vocabulary p(expert) = log(six*mass)
    out = []
    for l in L:
        e = r[where][which][str(l)]; six, mass = e["six"], e["mass"]; ti = r["tool_idx"]
        p_t = max(six[ti], 1e-30); p_o = max(max(v for j, v in enumerate(six) if j != ti), 1e-30)
        out.append({"layer": l, "log_odds_vs_best_other": float(np.log(p_t) - np.log(p_o)), "log_p_full": float(np.log(max(p_t * mass, 1e-300))), "resolved": mass >= FLOOR})
    return out

# ---------------- W-1: ignition and competition
def w1_row(r, where, which):
    prof = profile(r, where, which); ps = [p for p, _, _ in prof]
    resolved = sum(1 for p in ps if p is not None)
    ign_depth = next((L[i] / L[-1] for i, p in enumerate(ps) if p is not None and p >= 0.5), None)
    first01 = next((i for i, p in enumerate(ps) if p is not None and p >= 0.1), None)
    first09 = next((i for i, p in enumerate(ps) if p is not None and p >= 0.9), None)
    width = (first09 - first01) if (first01 is not None and first09 is not None) else None
    comp = None
    for i, (p, top, mass) in enumerate(prof[:-1]):
        if p is not None and top != r["tool"]:
            comp = {"layer": L[i], "competitor": top, "is_prior_favourite": top == max(prior_p, key=prior_p.get)}
    # the LAST layer before the end at which a different tool leads (the late competitor)
    return {"first_crossing_depth": ign_depth, "crossing_width_layers": width, "late_readout_competitor": comp, "resolved_layers": resolved,
            "final_p": ps[-1], "never": ign_depth is None, "log_odds_profile": log_odds(r, where, which)}

w1 = {"basis": BASIS + "; quantities: first confidence crossing on the six-tool conditional mass (full-vocabulary p = six*mass stored beside), readout competitor; the label is the EXPERT action; frozen rules: first crossing in depth order, ties miss, unresolved layers skipped and counted; an observational screen, not an ignition test", "mass_floor": FLOOR, "decisions": Nd, "episodes": len(episodes), "tool_prior": prior_p, "cells": {}}
for where in ("lens_act", "lens_note"):
    for which in ("lens", "logit"):
        rows = [w1_row(r, where, which) for r in idx]
        by_fam = defaultdict(list); by_var = defaultdict(list)
        for r, x in zip(idx, rows): by_fam[r["family"]].append(x); by_var["perturbed" if r["variant"] != "clean" else "clean"].append(x)
        def summ(xs):
            comps = [x["late_readout_competitor"] for x in xs if x["late_readout_competitor"]]
            return {"n": len(xs), "never_rank1_share": sum(x["never"] for x in xs) / len(xs),
                    "first_crossing_depth": tail([x["first_crossing_depth"] for x in xs]), "crossing_width_layers": tail([x["crossing_width_layers"] for x in xs]), "never_count": sum(x["never"] for x in xs),
                    "late_readout_competitor_share": len(comps) / len(xs), "competitor_names": dict(Counter(c["competitor"] for c in comps)),
                    "competitor_is_prior_favourite_share": (sum(c["is_prior_favourite"] for c in comps) / len(comps)) if comps else None,
                    "competitor_layer": tail([c["layer"] for c in comps]), "unresolved_layers_mean": nL - statistics.mean(x["resolved_layers"] for x in xs)}
        w1["cells"][f"{where}/{which}"] = {"all": summ(rows), "by_family": {f: summ(xs) for f, xs in by_fam.items()}, "by_variant": {v: summ(xs) for v, xs in by_var.items()}}
(OUT / "w1.json").write_text(json.dumps(w1, indent=1, default=float) + "\n")

# ---------------- W-4: report or computation
def top_six(six, mass): return None if mass < FLOOR else TOOLS[int(np.argmax(six))]
w4 = {"basis": BASIS + "; early versus late linear decodability of the EXPERT action is W-4's primary (workspace_w2.py at P_note vs P_act); this file holds the secondary readouts, mostly unresolved by the floor", "mass_floor": FLOOR, "decisions": Nd, "tool_prior": prior_p, "model_readout": {}, "lens_last_layer": {}, "note_trajectory": None}
for where, key_six, key_mass in (("P_note", "model_six_note", "model_mass_note"), ("P_act", "model_six_act", "model_mass_act")):
    tops = [(r["tool"], top_six(r[key_six], r[key_mass])) for r in idx]
    resolved = [(a, b) for a, b in tops if b is not None]
    w4["model_readout"][where] = {"resolved": len(resolved), "unresolved": Nd - len(resolved),
        "taken_is_top_share": (sum(a == b for a, b in resolved) / len(resolved)) if resolved else None,
        "prior_favourite_share": prior_p[max(prior_p, key=prior_p.get)], "mass": tail([r[key_mass] for r in idx])}
    by_fam = defaultdict(list)
    for r, (a, b) in zip(idx, tops):
        if b is not None: by_fam[r["family"]].append(a == b)
    w4["model_readout"][where]["by_family"] = {f: {"n": len(v), "taken_is_top_share": sum(v) / len(v)} for f, v in by_fam.items()}
for where in ("lens_note", "lens_act"):
    e = [r[where]["lens"][str(L[-1])] for r in idx]
    tops = [(r["tool"], top_six(x["six"], x["mass"])) for r, x in zip(idx, e)]
    resolved = [(a, b) for a, b in tops if b is not None]
    w4["lens_last_layer"][where] = {"resolved": len(resolved), "taken_is_top_share": (sum(a == b for a, b in resolved) / len(resolved)) if resolved else None}
sample_path = CAP / "sample.jsonl"
if sample_path.exists() and sample_path.read_text().strip():
    traj = []
    for l in sample_path.read_text().splitlines():
        s = json.loads(l); r = idx[s["i"]]; last = str(L[-1]); seq = s["note_trajectory_lens_six"][last]
        masses = s.get("note_trajectory_lens_mass", {}).get(last)
        tops = [TOOLS[int(np.argmax(x))] if (masses is None or masses[k] >= FLOOR) else None for k, x in enumerate(seq)]; n = len(tops)
        first = next((k for k, t in enumerate(tops) if t == r["tool"]), None)
        stays = all(t == r["tool"] for t in tops[first:] if t is not None) if first is not None else False
        unresolved = sum(1 for t in tops if t is None)
        n_prose = PROSE.get(s["i"]); extra = {}
        if n_prose is not None:  # index k sits at position P_note + k: k <= n_prose sits on the prompt's last token or a prose token, k > n_prose on the fence/JSON syntax, k = n at P_act
            pr, sy = tops[: n_prose + 1], tops[n_prose + 1:]
            first_pr = next((k for k, t in enumerate(pr) if t == r["tool"]), None); first_sy = next((k for k, t in enumerate(sy) if t == r["tool"]), None)
            extra = {"n_prose_tokens": n_prose, "n_syntax_positions": len(sy), "first_top_frac_prose": (first_pr / max(n_prose, 1)) if first_pr is not None else None,
                     "top_at_prose_end": pr[-1] if pr else None, "first_top_syntax_index": first_sy, "crossing_only_in_syntax": first_pr is None and first_sy is not None}
        traj.append({"i": s["i"], "tool": r["tool"], "note_tokens": n, "first_top_frac": (first / max(n - 1, 1)) if first is not None else None,
                     "top_at_note_start": tops[0], "top_at_note_end": tops[-1], "settles_and_stays": stays, "unresolved_tokens": unresolved, "n_changes": sum(1 for a, b in zip(tops, tops[1:]) if a != b and a is not None and b is not None), **extra})
    w4["note_trajectory"] = {"n": len(traj), "basis": "the 300-sample, lens at the last layer, six-tool top per note token, the mass floor applied per token (unresolved tokens counted)",
        "taken_is_top_at_note_start_share": sum(t["top_at_note_start"] == t["tool"] for t in traj) / len(traj),
        "taken_is_top_at_note_end_share": sum(t["top_at_note_end"] == t["tool"] for t in traj) / len(traj),
        "first_top_frac": tail([t["first_top_frac"] for t in traj]), "settles_and_stays_share": sum(t["settles_and_stays"] for t in traj) / len(traj),
        "changes_per_note": tail([t["n_changes"] for t in traj]), "unresolved_tokens_per_note": tail([t["unresolved_tokens"] for t in traj]),
        "prose_syntax_split": ({"basis": "positions split at the ```json fence by token offsets (the W-3b repair rule); a prose position sits on the prompt's last token or a prose token, a syntax position on the fence or the JSON prefix, the last at P_act",
            "n_prose_tokens": tail([t["n_prose_tokens"] for t in traj]), "taken_is_top_at_prose_end_share": sum(t["top_at_prose_end"] == t["tool"] for t in traj) / len(traj),
            "first_top_frac_prose": tail([t["first_top_frac_prose"] for t in traj]), "crossing_only_in_syntax_count": sum(t["crossing_only_in_syntax"] for t in traj),
            "never_top_count": sum(1 for t in traj if t["first_top_frac_prose"] is None and t["first_top_syntax_index"] is None)} if PROSE else "not computed: pass --corpus and --snapshot"), "rows": traj}
(OUT / "w4.json").write_text(json.dumps(w4, indent=1, default=float) + "\n")

# ---------------- W-3: broadcast and the carrier (the sample)
w3 = {"basis": BASIS + "; DIRECT attention to tagged spans from P_note and P_act, eager kernel, the 300-sample; attention weight is a routing weight, not a causal share; global layers are the only single-edge routes beyond the window, not the only end-to-end routes", "sliding_window": man.get("sliding_window"), "global_layers": [i + 1 for i, g in enumerate(man.get("is_global_layer", [])) if g], "gate": None, "by_layer": [], "by_kind": {}}
if sample_path.exists():
    is_global = man.get("is_global_layer"); W = int(man.get("sliding_window") or 0)
    acc = defaultdict(lambda: defaultdict(float)); cnt = 0; gate_viol = 0.0; gate_cells = 0
    far_mass = {"local": [], "global": []}
    for l in sample_path.read_text().splitlines():
        s = json.loads(l); A = np.load(CAP / f"attn_{s['i']}.npy").astype(np.float32)  # [layers, heads, 2, seq]
        kinds = list(s["kinds"]); r = idx[s["i"]]; q_pos = {"P_note": r["P_note"], "P_act": r["P_act"]}
        if s["i"] in PROSE:  # the capture labelled every completion token "note"; split it as the repair does
            for j in range(r["n_prompt_tokens"] + PROSE[s["i"]], len(kinds)): kinds[j] = "note_syntax"
            kinds[-1] = "tool_name"
        for li in range(A.shape[0]):
            for qi, qname in enumerate(("P_note", "P_act")):
                a = A[li, :, qi, :].mean(0)  # mean over heads [seq]
                qp = q_pos[qname]; far = a[: max(0, qp - W + 1)].sum() if W else 0.0
                (far_mass["global"] if is_global[li] else far_mass["local"]).append(float(far))
                if not is_global[li]: gate_viol += float(far); gate_cells += 1
                for k in set(kinds): acc[(li, qname, k)]["mass"] += float(a[[j for j, kk in enumerate(kinds) if kk == k][:len(a)]].sum()) if k in kinds else 0.0
        cnt += 1
    if cnt == 0:
        w3["sample_n"] = 0; w3["note"] = "no attention sample in this capture"
    w3["gate"] = {"local_layer_mass_beyond_window_mean": (gate_viol / gate_cells) if gate_cells else None, "cells": gate_cells, "passes_if_zero": (gate_viol / gate_cells) < 1e-6 if gate_cells else None}
    w3["far_mass"] = {k: tail(v) for k, v in far_mass.items()}
    kinds_all = sorted({k for (_, _, k) in acc})
    for li in (range(len(is_global)) if cnt else []):
        row = {"layer": li + 1, "global": bool(is_global[li])}
        for qname in ("P_note", "P_act"):
            row[qname] = {k: acc[(li, qname, k)]["mass"] / cnt for k in kinds_all}
        w3["by_layer"].append(row)
    for k in (kinds_all if cnt else []):
        w3["by_kind"][k] = {"local_mean": statistics.mean(acc[(li, "P_note", k)]["mass"] / cnt for li in range(len(is_global)) if not is_global[li]),
                            "global_mean": statistics.mean(acc[(li, "P_note", k)]["mass"] / cnt for li in range(len(is_global)) if is_global[li]) if any(is_global) else None}
    if cnt: w3["sample_n"] = cnt
(OUT / "w3.json").write_text(json.dumps(w3, indent=1, default=float) + "\n")

# ---------------- W-5: lens validity by position band and source layer (an operational readout diagnostic, not a derivative check)
def band(n):
    return "<=256" if n <= 256 else "<=1024" if n <= 1024 else "<=2048" if n <= 2048 else ">2048"
w5 = {"basis": BASIS + "; per source layer and position band: does the six-tool argmax through that source layer's map at P_act agree with the model's own six-tool argmax at P_act; readouts below the mass floor are unresolved and counted; agreement with the model's output does not identify a derivative", "mass_floor": FLOOR, "cells": {}}
for l in L:
    for which in ("lens", "logit"):
        agg = defaultdict(lambda: {"n": 0, "agree": 0, "unresolved": 0})
        for r in idx:
            b = band(r["n_prompt_tokens"]); e = r["lens_act"][which][str(l)]
            if r["model_mass_act"] < FLOOR or e["mass"] < FLOOR: agg[b]["unresolved"] += 1; agg[b]["n"] += 1; continue
            agg[b]["n"] += 1; agg[b]["agree"] += int(int(np.argmax(e["six"])) == int(np.argmax(r["model_six_act"])))
        w5["cells"][f"{which}/L{l}"] = {b: {**v, "agree_share_of_resolved": (v["agree"] / max(1, v["n"] - v["unresolved"]))} for b, v in agg.items()}
(OUT / "w5.json").write_text(json.dumps(w5, indent=1, default=float) + "\n")
print(json.dumps({"event": "analyzed", "decisions": Nd, "episodes": len(episodes), "w1_cells": list(w1["cells"]), "w4_note_trajectory_n": (w4["note_trajectory"] or {}).get("n"), "w3_sample": w3.get("sample_n")}))
