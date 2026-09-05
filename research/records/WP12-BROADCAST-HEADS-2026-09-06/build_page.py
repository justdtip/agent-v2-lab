"""Build the Broadcast Heads page from out/broadcast_heads.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "out" / "broadcast_heads.json")); rows = d["rows"]; S = d["summary"]
SEL = json.load(open(HERE / "out" / "selection_nulldist.json"))
ABL = None; ABL_IN = None   # the four-layer-band runs (ablation.json, ablation_inband.json) are superseded by the five-layer run below and not shown
def _load(name): return json.load(open(HERE / "out" / name)) if (HERE / "out" / name).exists() else None
ABL_F = _load("ablation_inband_floors.json"); ABL_K1 = _load("ablation_k1.json"); SING = _load("singles.json"); COUP = _load("coupling.json"); SPEC = _load("lens_spectra.json"); SPECC = _load("lens_spectra_concepts.json"); INJ = _load("inject_events_L12.json"); DIAG = _load("layer28_diagnostic.json"); KVC = _load("kvgroup_clustering.json"); KVG = _load("kvgroup.json")
SEL_ORTH = json.load(open(HERE / "out" / "selection_orth.json")) if (HERE / "out" / "selection_orth.json").exists() else None
COMP = json.load(open(HERE / "out" / "composition_summary.json")) if (HERE / "out" / "composition_summary.json").exists() else None
SEP = json.load(open(HERE / "out" / "separation.json")) if (HERE / "out" / "separation.json").exists() else None
selset = {(r["block"], r["head"]) for r in SEL["selected_attention"]}
for r in rows: r["selected"] = (r["channel"] == "attention" and (r["block"], r["head"]) in selset)
att = [r for r in rows if r["channel"] == "attention" and r["layer_written"] != 32]; rec = [r for r in rows if r["channel"] == "recurrent"]
def fmt(x, dd=2): return "n/a" if x is None else f"{x:,.{dd}f}"
def layer_series(rs, key):
    out = {}
    for r in rs: out.setdefault(r["layer_written"], []).append(r[key])
    return [[L, float(np.median(v)), float(np.percentile(v, 90)), len(v)] for L, v in sorted(out.items())]
series = {"att_mrr": layer_series(att, "J_mrr"), "att_mrr_rot": layer_series(att, "J_rot_mrr"), "rec_mrr": layer_series(rec, "J_mrr"), "rec_mrr_rot": layer_series(rec, "J_rot_mrr"),
          "att_count": [[int(L), c, 16] for L, c in SEL["by_layer"].items()], "rec_count": [[int(L), sum(1 for r in rec if r["layer_written"] == int(L) and r.get("broadcasts_J")), 32] for L in S["recurrent"]["by_layer"]]}
abl_series = [] if ABL is None else [[r["k"], r["stratified"]["recall_downstream_mean"], r["control_recall_downstream_mean"], r["change_rate"], r["control_change_rate_mean"], r["specificity_holds"]] for r in ABL["results"]]
abl_in_series = [] if ABL_IN is None else [[r["k"], r["stratified"]["recall_downstream_mean"], r["control_recall_downstream_mean"], r["change_rate"], r["control_change_rate_mean"], r["specificity_holds"]] for r in ABL_IN["results"]]
orth_line = "" if SEL_ORTH is None else f" Re-selected on the lens directions with the token-identity component removed at every layer: {SEL_ORTH['n_of_N'][0]} of {SEL_ORTH['n_of_N'][1]}, by layer {', '.join(f'{L}: {c}' for L, c in SEL_ORTH['by_layer'].items())}; first half of the band {SEL_ORTH['first_half_of_band']} against second half {SEL_ORTH['second_half_of_band']}, so the prediction that the relays sit in the first half is {SEL_ORTH['prediction_1_first_half']} on that population too."
retr = {(19, 2): "b19 h2", (23, 9): "b23 h9", (19, 12): "b19 h12", (27, 1): "b27 h1", (3, 15): "b3 h15", (19, 15): "b19 h15"}
top_att = sorted(att, key=lambda r: -r['mrr_margin'])[:12]
def head_row(r):
    tag = retr.get((r["block"], r["head"]), "")
    return f"<tr><td>layer {r['layer_written']}, block {r['block']}, head {r['head']}{' · retrieval head' if tag else ''}</td><td>{fmt(r['J_gain'])} / {fmt(r['J_rot_gain'])} / {fmt(r['MLP_rows_gain'])}</td><td>{fmt(r['J_mrr'],3)} / {fmt(r['J_rot_mrr'],3)} / {fmt(r['MLP_rows_mrr'],3)}</td><td>{'yes' if r['selected'] else 'no'}</td></tr>"
ranks = S["retrieval_heads_rank_among_128_attention"]
# ---- stage 2 on the five-layer band (R41e attention members 12, 16, 20, 24, 28), arms across floors
ATT5 = (12, 16, 20, 24, 28); FIRST5, SECOND5 = (12, 16, 20), (24, 28)
abl_f = []
if ABL_F:
    ordered = ABL_F.get("parameters", {}).get("ordered"); n_all = len(ordered) if ordered else None
    rep = ABL_F["reproducibility"]
    abl_f.append({"k": 0, "label": "unablated: recall is measured against this readout, so 1.0 by construction; bit-identical run to run" if rep["bit_identical"] else "unablated", "recall": 1.0, "ctl": 1.0, "ctl_min": 1.0, "ctl_max": 1.0, "ctl_sd": 0.0, "change": 0.0, "ctl_change": 0.0, "spec": True, "overlap": 0, "deterministic": False, "det_layers": [], "drop": 0.0, "upstream_empty": False, "by_layer": {}})
    for r in ABL_F["results"]:
        k = r["k"]
        if ordered: lo = ordered[k - 1][3]; nxt = ordered[k][3] if k < n_all else -1.0; label = "no floor, all " + str(n_all) if k == n_all else ", ".join(f"floor {f}" for f in (0.2, 0.1, 0.05) if lo >= f > nxt)
        else: label = "partial dump, labels after the run"
        cd = [c["stratified"]["recall_downstream_mean"] for c in r["controls"] if c["stratified"]["recall_downstream_mean"] is not None]
        ov = max(c["overlap_with_selection"] for c in r["controls"])
        det_layers = [b + 1 for b in sorted({bb for c in r["controls"] for bb, h in c["heads"]}) if len({frozenset(h for bb, h in c["heads"] if bb == b) for c in r["controls"]}) == 1 and sum(1 for bb, h in r["heads"] if bb == b) > 1]
        det = bool(det_layers)
        abl_f.append({"k": k, "label": label, "recall": r["stratified"]["recall_downstream_mean"], "ctl": r["control_recall_downstream_mean"], "ctl_min": min(cd), "ctl_max": max(cd), "ctl_sd": 0.0 if det else float(np.std(cd)), "change": r["change_rate"], "ctl_change": r["control_change_rate_mean"], "spec": r["specificity_holds"], "overlap": ov, "deterministic": det, "det_layers": det_layers, "drop": r["downstream_recall_drop_vs_control"], "upstream_empty": not r["stratified"].get("recall_upstream_control"), "by_layer": {str(L): [r["recall_by_layer"][str(L)], r["control_recall_by_layer_mean"][str(L)]] for L in r["recall_by_layer"]}})
import math
def equivalence(arms):
    """for each selected arm, the number of random heads at which the random curve reaches the same overlap, interpolated linearly in log2(k + 1) between measured arms; beyond the last arm the value is reported as beyond it."""
    pts = [(a["k"], a["ctl"]) for a in arms]; out = {}
    for a in arms:
        if a["k"] == 0: continue
        v = a["recall"]; ke = None
        for (k0, c0), (k1, c1) in zip(pts, pts[1:]):
            if (c0 >= v >= c1) or (c0 <= v <= c1):
                t = 0.0 if c0 == c1 else (c0 - v) / (c0 - c1); x = math.log2(k0 + 1) + t * (math.log2(k1 + 1) - math.log2(k0 + 1)); ke = 2 ** x - 1; br = (k0, k1); break
        out[a["k"]] = {"k_equiv": ke, "beyond": ke is None and v < min(c for _, c in pts), "factor": None if ke is None else ke / a["k"], "bracket": None if ke is None else br}
    return out
def eq_text(k, e, kmax):
    if e["beyond"]: return f"beyond the {kmax} measured (random at {kmax} heads is still above it)"
    return f"{fmt(e['k_equiv'], 1)} random heads ({fmt(e['factor'], 1)} times), between the measured arms at {e['bracket'][0]} and {e['bracket'][1]}"
EQ = equivalence(abl_f) if abl_f else {}
K1 = None
if ABL_K1:
    r1 = next((r for r in ABL_K1["results"] if r["k"] == 1), None)
    if r1:
        up = r1["stratified"].get("recall_upstream_control") or {}
        K1 = {"recall": r1["stratified"]["recall_downstream_mean"], "ctl": r1["control_recall_downstream_mean"], "ctl_range": (min(c["stratified"]["recall_downstream_mean"] for c in r1["controls"]), max(c["stratified"]["recall_downstream_mean"] for c in r1["controls"])), "change": r1["change_rate"], "ctl_change": r1["control_change_rate_mean"], "upstream": up, "gate_passed": bool(up) and all(v == 1.0 for v in up.values()), "head": r1["heads"][0]}
    rep26 = next((r for r in ABL_K1["results"] if r["k"] == len(ABL_K1.get("parameters", {}).get("ordered", [])) and r["k"] > 1), None)
    full26 = next((r for r in ABL_F["results"] if r["k"] == 26), None) if ABL_F else None
    REP26 = None if not (rep26 and full26) else {"first": full26["stratified"]["recall_downstream_mean"], "second": rep26["stratified"]["recall_downstream_mean"], "ctl_first": full26["control_recall_downstream_mean"], "ctl_second": rep26["control_recall_downstream_mean"]}
K1_SEC = "" if not K1 else f"""<p><strong>The hard upstream gate, restored on a single-head arm.</strong> Block {K1['head'][0]} head {K1['head'][1]} alone, writing layer {K1['head'][0] + 1}: the readout at every band layer below it ({', '.join(K1['upstream'].keys())}) must be bit-identical to the unablated readout, and it {'is, all exactly 1.0' if K1['gate_passed'] else 'is not: ' + str(K1['upstream'])}. The intervention does not reach where it cannot reach. Downstream of it (layers 24, 27, 28) the overlap is {fmt(K1['recall'], 3)} against {fmt(K1['ctl'], 3)} for a random layer-24 head ({fmt(K1['ctl_range'][0], 3)} to {fmt(K1['ctl_range'][1], 3)} over five seeds), change rate {fmt(K1['change'], 3)} against {fmt(K1['ctl_change'], 3)}. This arm reads out over three layers where the others read over ten, so it is not placed on the curve.{'' if not REP26 else ' The run also replicated the 26-head arm under the same seeds: ' + fmt(REP26['first'], 3) + ' then ' + fmt(REP26['second'], 3) + ' for the relays, ' + fmt(REP26['ctl_first'], 3) + ' then ' + fmt(REP26['ctl_second'], 3) + ' for the random sets' + (', identical' if abs(REP26['first'] - REP26['second']) < 1e-9 and abs(REP26['ctl_first'] - REP26['ctl_second']) < 1e-9 else ', not identical') + '.'}</p>"""
CLOSE = """<div class="callout" style="margin:12px 0 4px"><strong>Where the thread stands, with the weight of each leg.</strong> Established: stage 1 finds a real and reproducible population of relays by weight arithmetic, and single-head ablation shows no member of it is individually special (a proper within-layer null). Open: at set level the relays damage the readout more than random sets and do so redundantly, and that is consistent with both readings, specific carried content (heads that damage the same content carry the same content, which is what broadcast looks like from outside) and general influence; the injection arm built to separate them did not achieve sensitivity at its pre-registered readout. Generated, not tested: a reversal at the output-aligned readout, p = 0.10. The paper's causal claim is therefore not supported at single-head resolution and undecided at set level; the contrast-direction injection at layer 20, pre-registered, is the experiment that closes it.</div>"""
STAGE2 = "" if not abl_f else f"""<h2>Stage 2: ablating the relays, graded, on the five-layer band</h2>
{CLOSE}
<div class="callout"><strong>The metric.</strong> Each arm is scored by the overlap of the lens readout's top-25 tokens with the unablated readout's own top-25, so the unablated row is 1.0 by construction and the number is damage to the model's prior output, not recall of anything external. The paper's recall at 25 scored the lens's recovery of an externally known target and starts at 0.86; it is not the same quantity and the two are not compared. The paper's question is whether a small specific set does the work of a much larger arbitrary one, so each selected arm is also expressed as the number of random heads at which the random curve reaches the same overlap, interpolated linearly in log2(k + 1) between the two measured random arms that bracket it, which are shown beside every figure because the transform does real work.</div>
{K1_SEC}
<div class="tiles">{"".join(f"<div class='tile att'><div class='n'>{'&gt; ' + str(max(a['k'] for a in abl_f)) if e['beyond'] else fmt(e['k_equiv'], 1)}</div><div class='l'>random heads for the damage of the top {k}{'' if e['beyond'] else ' (' + fmt(e['factor'], 1) + ' times)'}{'' if e['beyond'] else '; between the arms at ' + str(e['bracket'][0]) + ' and ' + str(e['bracket'][1])}</div></div>" for k, e in EQ.items() if k <= 16)}</div>
<p><strong>What the set-level curve does and does not show, after the single-head sweep below.</strong> The set-level effect stands as measured: the top relays zeroed together damage the readout more than layer-matched random sets, and the sets are strongly redundant (measured overlap above the product of the single-head overlaps at every k). But it is not lens-specific transport by individually special heads: relays are level with the unselected heads at every layer when ablated alone, the small-k advantage is one strong layer-12 head counted at all ten readouts, and the concentration factor is arithmetic of the layer mix. Random ablation of any two heads already costs ten points and the selection adds six. The cumulative factor falling with k is not a validated ranking: per head, ranks three to eight add half of what ranks nine to sixteen add, and the sweep found no rank correlation between preservation and single-head effect. Whether redundancy means the relays carry the same workspace content, or that they are not special, is what the injection test decides.{" The sixteen strongest relays do more damage than " + str(max(a['k'] for a in abl_f)) + " arbitrary heads at the same layers, stated as a bound." if EQ.get(16, {}).get("beyond") else ""} Specificity: at k = 21 the selected set changes the next token at {fmt(next((a['change'] for a in abl_f if a['k'] == 21), None), 3)} of positions against the random sets' {fmt(next((a['ctl_change'] for a in abl_f if a['k'] == 21), None), 3)}; breaking the model would move the output far more than a control that damages the readout less.</p>
<p>The in-band relays under the orthogonalised rule, {n_all or "the in-band"} heads writing layers 12, 16, 20, 24 and 28, are zeroed in order of absolute lens preservation, k at a time, and the lens is read at all ten band layers on eight 512-token documents at 64 positions each. Beside each arm, five layer-matched random sets of the same size drawn from the unselected heads at the same layers. The arms k = 2, 4, 8 and 16 are the same at every floor; the arms {", ".join(str(a["k"]) for a in abl_f if a["k"] > 16) or "past 16"} are the sets left by floors of 0.2, 0.1, 0.05 and none on absolute preservation, so the floor's sensitivity is three extra arms of one run. Reproducibility: {"bit-identical readout run to run" if rep["bit_identical"] else "recall floor " + fmt(rep["upstream_floor"]["recall"], 4)}; the exact-equality upstream gate has no layer to fire on once a layer-12 head is in the set, which happens at k = 2. The selected arm has no seed variance by construction, the heads and documents being fixed, so the random sets' range is the whole variance estimate and a deterministic row is not a converged one.</p>
<div class="charts">
  <div class="chart"><h3>Top-25 overlap with the unablated readout, downstream of the ablated heads</h3><p class="sub">Share of the unablated readout's top-25 tokens that survive at readout layers with an ablated head at or below them; the random sets' mean, with their range over five seeds shaded. 1.0 at k = 0 by construction.</p><div id="ablf"></div>
    <div class="legend"><span><i class="sw" style="background:var(--att)"></i>relays, by absolute preservation</span><span><i class="sw" style="background:var(--ctl)"></i>layer-matched random, mean and range</span></div></div>
  <div class="chart"><h3>Next-token change rate</h3><p class="sub">Share of positions whose top-1 prediction changes. A recall drop is read only where this stays within twice the random sets'; every arm is shown either way.</p><div id="chgf"></div>
    <div class="legend"><span><i class="sw" style="background:var(--att)"></i>relays</span><span><i class="sw" style="background:var(--ctl)"></i>layer-matched random</span></div></div>
  <div class="chart"><h3>Overlap by readout layer</h3><p class="sub">The same overlap at each of the ten band layers, for the arms k = 8 and k = {max(a['k'] for a in abl_f)}. The gap grows with depth, but so does the number of ablated heads upstream of each readout; per upstream ablated head the gap at k = 16 is about 0.01 at every readout from layer 16 to 28 and zero at 12 and 13, so the gradient is arithmetic and the per-head excess is flat with depth.</p><div id="abll"></div>
    <div class="legend"><span><i class="sw" style="background:var(--att)"></i>relays, k = {max(a['k'] for a in abl_f)}</span><span><i class="sw" style="background:var(--ctl)"></i>random, k = {max(a['k'] for a in abl_f)}</span><span><i class="sw" style="background:var(--att);opacity:.45"></i>relays, k = 8 (dashed)</span><span><i class="sw" style="background:var(--ctl);opacity:.45"></i>random, k = 8 (dashed)</span></div></div>
</div>
<div class="wrap"><table><tr><th>k</th><th>arm</th><th>overlap downstream: relays / random (range)</th><th>drop vs random</th><th>random heads for the same damage</th><th>change rate: relays / random</th><th>specificity holds</th><th>control</th></tr>{"".join(f"<tr><td>{a['k']}</td><td>{a['label']}</td><td>{fmt(a['recall'],3)} / {fmt(a['ctl'],3)} ({fmt(a['ctl_min'],3)} to {fmt(a['ctl_max'],3)})</td><td>{fmt(a['drop'],3)}</td><td>{'' if a['k'] == 0 else ('beyond ' + str(max(x['k'] for x in abl_f)) if EQ[a['k']]['beyond'] else fmt(EQ[a['k']]['k_equiv'],1) + ' (' + fmt(EQ[a['k']]['factor'],1) + '×; arms ' + str(EQ[a['k']]['bracket'][0]) + ' to ' + str(EQ[a['k']]['bracket'][1]) + ')')}</td><td>{fmt(a['change'],3)} / {fmt(a['ctl_change'],3)}</td><td>{'yes' if a['spec'] else 'no'}</td><td>{('layer ' + ', '.join(map(str, a['det_layers'])) + ': one possible set, variance zero there') if a['deterministic'] else ('forced overlap ' + str(a['overlap']) + ' at layer 28' if a['overlap'] else ('' if a['k'] == 0 else 'exact'))}</td></tr>" for a in abl_f)}</table></div>"""
# ---- layer 28's nulls
DIAG_SEC = "" if not DIAG else f"""<h2>Layer 28 is saturated, and its nulls are not low</h2>
<p>Nine of sixteen heads writing layer 28 pass the orthogonalised rule. The Head's diagnostic asked whether the null draws collapse there, which would make the margin rule easier to pass. Medians over the sixteen heads writing each layer, on the orthogonalised population: real preservation, the two null families, the per-head ratio of real to the better null, and passes with and without an absolute floor of 0.1.</p>
<div class="wrap"><table><tr><th>layer written</th><th>real</th><th>rotated null</th><th>MLP-row null</th><th>ratio</th><th>pass</th><th>pass, floor 0.1</th></tr>{"".join(f"<tr><td>{L}</td><td>{fmt(v['real_median'],3)}</td><td>{fmt(v['rotated_null_median'],4)}</td><td>{fmt(v['mlp_null_median'],4)}</td><td>{fmt(v['ratio_real_over_best_null_median'],2)}</td><td>{v['pass']}</td><td>{v['pass_with_floor_0.1']}</td></tr>" for L, v in DIAG['per_layer_orthogonalised_population'].items())}</table></div>
<p>Layer 28's nulls are not lower than layer 20's and its ratio is the highest in the model: the saturation is the heads' property. Layer 8 is the mirror, the highest real median and no passes, because random MLP rows preserve that population almost as well. The rule admitted tiny-number passes (block 27 head 13 at 0.002 over 0.001), which the floor removes; the floor was chosen after seeing the data and happens to make the layer-matched control exact, so the k curve is shown at every floor above rather than at one.</p>"""
# ---- prediction 1, every scoring
def rate_rows():
    rows_ = []
    if S.get("first_half_of_band") is not None: rows_.append(("gain and preservation, the paper's rule, lens population", S["first_half_of_band"], S["second_half_of_band"], "16, 20 vs 24, 28"))
    for name, X in (("preservation beyond 40 draws, lens population", SEL), ("same rule, orthogonalised population", SEL_ORTH)):
        if X and X.get("first_half_of_band") is not None: rows_.append((name, X["first_half_of_band"], X["second_half_of_band"], "16, 20 vs 24, 28"))
    if DIAG:
        for f, v in DIAG["absolute_floor_scorings_of_prediction_1"].items(): rows_.append((f"orthogonalised, floor {f}", v["first_half"], v["second_half"], "16, 20 vs 24, 28"))
    if SEL_ORTH:
        sa = SEL_ORTH["selected_attention"]
        for f in (0.0, 0.05, 0.1, 0.2):
            a = sum(1 for r in sa if r["layer_written"] in FIRST5 and r["J_mrr"] >= f); b = sum(1 for r in sa if r["layer_written"] in SECOND5 and r["J_mrr"] >= f)
            rows_.append((f"orthogonalised, five-layer band, floor {f}: rates per head", f"{a} of 48 ({a/48:.2f})", f"{b} of 32 ({b/32:.2f})", "12, 16, 20 vs 24, 28"))
    return rows_
def verdict(a, b):
    fa = float(str(a).split("(")[-1].rstrip(")")) if "(" in str(a) else float(a); fb = float(str(b).split("(")[-1].rstrip(")")) if "(" in str(b) else float(b)
    return "confirmed" if fa > fb else "refuted" if fb > fa else "tied"
PRED_SEC = f"""<h2>Prediction 1, scored every time the rule changed</h2>
<p>The paper's claim that broadcast heads concentrate in the first half of the workspace band. Every selection rule used tonight scores it (R38(h)); the five-layer band's halves are uneven, 48 heads against 32, so those rows are rates.</p>
<div class="wrap"><table><tr><th>selection</th><th>first half</th><th>second half</th><th>halves</th><th>verdict</th></tr>{"".join(f"<tr><td>{n}</td><td>{a}</td><td>{b}</td><td>{h}</td><td>{verdict(a, b)}</td></tr>" for n, a, b, h in rate_rows())}</table></div>"""
# ---- key-value groups
KV_SEC = ""
if KVC:
    kv_rows = "".join(f"<tr><td>layer {int(k.split('_')[0][1:]) + 1}, block {k.split('_')[0][1:]}</td><td>{k.split('floor')[1]}</td><td>{v['k']}</td><td>{', '.join(map(str, v['by_group']))}</td><td>{fmt(v['p_dispersion'],3)}</td></tr>" for k, v in KVC.items() if k.startswith("b") and int(k.split("_")[0][1:]) >= 11)
    KV_SEC = f"""<h2>Are the relays organised by key-value group?</h2>
<p>Sixteen query heads share four value projections. At layer 28 all four heads of group 0 are non-relays and all four of group 3 are relays, which suggested the property sits in the shared projection. Membership alone cannot show it: exact enumeration over subsets of the same size gives p = 0.055 at layers 20 and 28 and 0.33 at 24 (Fisher {fmt(KVC.get('fisher_19_23_27_floor0.0'), 3)}), and under the floor of 0.1 the combination is {fmt(KVC.get('fisher_19_23_27_floor0.1'), 3)}: no evidence either way.</p>
<div class="wrap"><table><tr><th>block</th><th>floor</th><th>relays</th><th>by group 0, 1, 2, 3</th><th>p, dispersion</th></tr>{kv_rows}</table></div>"""
    if KVG:
        pr = {b: e["prereg"] for b, e in KVG["blocks"].items()}
        n_pass = sum(1 for v in pr.values() if v["beyond_all_draws"])
        KV_SEC += f"""<h3 style="margin-top:22px">The pre-registered weight test</h3>
<p>Per block, over the sixteen real heads, the ratio of between-group to within-group variance of lens preservation, against the same ratio on each of forty null populations under the same heads. The property lives in the shared value projection where F on the lens exceeds every draw: <strong>{n_pass} of {len(pr)} blocks</strong>.</p>
<div class="chart"><div id="kvf"></div><div class="legend"><span><i class="sw" style="background:var(--att)"></i>F on the lens population</span><span><i class="sw" style="background:var(--ctl)"></i>null draws: median and maximum</span></div></div>
<div class="wrap"><table><tr><th>block</th><th>F, lens</th><th>null median</th><th>null max</th><th>draws at or above</th><th>beyond all draws</th><th>group means of preservation</th><th>lens energy share by group, isotropic 0.10</th><th>relays beating the pairing null (chance)</th><th>non-relays beating it</th></tr>{"".join(f"<tr><td>layer {e['layer_written']}, block {b}</td><td>{fmt(e['prereg']['F_J'],2)}</td><td>{fmt(e['prereg']['F_null_median'],2)}</td><td>{fmt(e['prereg']['F_null_max'],2)}</td><td>{e['prereg']['draws_at_or_above_F_J']} of 40</td><td>{'yes' if e['prereg']['beyond_all_draws'] else 'no'}</td><td>{', '.join(fmt(x,3) for x in e['prereg']['per_group_mean_J_mrr'])}</td><td>{', '.join(fmt(e['energy'][g]['share']['J'], 3) for g in sorted(e['energy']))}</td><td>{e['pairing_summary']['relays_beating']} of {e['pairing_summary']['relays']} ({fmt(e['pairing_summary']['relays'] / 21, 1)})</td><td>{sum(1 for h, v in e['pairing'].items() if v['beats_pairing_null'] and not e['cross'][f"h{h}_g{v['own_group']}"]['passes'])} of {16 - e['pairing_summary']['relays']}</td></tr>" for b, e in KVG['blocks'].items())}</table></div>
<p>The cross table pairs every output slice with every value projection; only its diagonal, the real heads, ever passes, and the 48 off-diagonal cells per block are maps the model never computes. The pairing control, each head's own slice against twenty scrambled correspondences with its own group's projection, is beaten by 25 of 26 relays and by 36 of 54 non-relays alike: trained heads work with their own partners, which is true of any network and singles out no relay. The scale-free energy share says the value projections do not carry the lens directions differentially by group except at layer 20's copy-map group, so the group structure of preservation at layer 28 has no mechanism at the level of value-projection alignment with the lens.</p>"""
# ---- the single-head sweep
SING_SEC = ""; SING_ROWS = []; SING_LAYERS = []
if SING and "summary" in SING:
    srows = SING["rows"]; ten = lambda r: 1 - float(np.mean([r["overlap_by_layer"][str(L)] for L in SING["summary"]["readout_layers"]]))
    for r in srows: r["damage_ten"] = ten(r); r["damage_28"] = 1 - r["overlap_at_28"]
    rank = {tuple(x[1:3]): i + 1 for i, x in enumerate(ABL_F["parameters"]["ordered"])} if ABL_F else {}
    for r in srows: r["rank"] = rank.get((r["block"], r["head"]))
    sel_rows = sorted([r for r in srows if r["selected"]], key=lambda r: r["rank"] or 99)
    SING_ROWS = [[r["layer_written"], r["block"], r["head"], r["preservation"], r["damage_28"], r["damage_ten"], r["change_rate"], r["selected"], r["rank"]] for r in srows]
    for L in (12, 16, 20, 24, 28):
        s_ = [r["damage_28"] for r in srows if r["layer_written"] == L and r["selected"]]; u_ = [r["damage_28"] for r in srows if r["layer_written"] == L and not r["selected"]]
        SING_LAYERS.append([L, float(np.median(s_)) if s_ else None, float(np.median(u_)), len(s_), len(u_)])
    sp_all = SING["summary"]["spearman_preservation_vs_damage_at_28_all_80"]; sp_sel = SING["summary"]["spearman_within_selected"]
    red = []
    if ABL_F:
        arm = {r["k"]: 1 - float(np.mean([r["recall_by_layer"][str(L)] for L in SING["summary"]["readout_layers"]])) for r in ABL_F["results"]}
        by_rank = {r["rank"]: r for r in srows if r["rank"]}
        for k in (2, 4, 8, 16):
            if k in arm and all(i in by_rank for i in range(1, k + 1)):
                prod = float(np.prod([1 - by_rank[i]["damage_ten"] for i in range(1, k + 1)])); meas = 1 - arm[k]
                red.append([k, prod, meas, "redundant" if meas > prod else "synergistic"])
    from scipy.stats import spearmanr
    uns = [r for r in srows if not r["selected"]]; sp_uns = spearmanr([r["preservation"] for r in uns], [r["damage_28"] for r in uns])
    SING_SEC = f"""<h2>Every in-band head ablated alone</h2>
<p>Eighty single-head ablations on the same documents and positions, the lens read at all ten band layers. Damage is one minus the overlap at layer 28, the one readout downstream of every head; the ten-layer mean is used for the sums. Spearman of preservation against damage at 28 over the eighty heads: <strong>{fmt(sp_all[0], 2)}</strong> (p = {sp_all[1]:.2g}); within the {len(sel_rows)} relays: <strong>{fmt(sp_sel[0], 2)}</strong> (p = {sp_sel[1]:.2g}); within the {len(uns)} unselected heads: <strong>{fmt(float(sp_uns.correlation), 2)}</strong> (p = {float(sp_uns.pvalue):.2g}). Over the eighty the correlation is carried partly by the selection separating high-damage from low-damage heads; the two within-set values say whether preservation orders heads inside each. Upstream readouts were exactly 1.0 for {'every head' if all(r['upstream_exact'] for r in srows) else str(sum(1 for r in srows if r['upstream_exact'])) + ' of 80 heads'}.</p>
<div class="charts">
  <div class="chart"><h3>Damage at layer 28 against preservation</h3><p class="sub">One point per head; relays filled, the rest hollow. If preservation ordered heads by effect the points would climb to the right.</p><div id="sing"></div>
    <div class="legend"><span><i class="sw" style="background:var(--att)"></i>relays (the rule's selection)</span><span><i class="sw" style="background:var(--ctl)"></i>unselected in-band heads</span></div></div>
  <div class="chart"><h3>Relays against the rest, by layer</h3><p class="sub">Median single-head damage at layer 28 for the relays and for the unselected heads writing each layer.</p><div id="singl"></div>
    <div class="legend"><span><i class="sw" style="background:var(--att)"></i>relays</span><span><i class="sw" style="background:var(--ctl)"></i>unselected</span></div></div>
</div>
{"" if not red else "<div class='wrap'><table><tr><th>k</th><th>independence null: product of the top-k single-head overlaps (ten layers)</th><th>measured k-arm overlap (ten layers)</th><th>reading, pre-registered</th></tr>" + "".join(f"<tr><td>{k}</td><td>{fmt(a, 3)}</td><td>{fmt(b, 3)}</td><td>{v}</td></tr>" for k, a, b, v in red) + "</table></div>"}
<div class="wrap"><table><tr><th>rank</th><th>head</th><th>preservation</th><th>damage at 28, alone</th><th>damage, ten-layer mean</th><th>change rate</th></tr>{"".join(f"<tr><td>{r['rank']}</td><td>layer {r['layer_written']}, block {r['block']}, head {r['head']}</td><td>{fmt(r['preservation'], 3)}</td><td>{fmt(r['damage_28'], 3)}</td><td>{fmt(r['damage_ten'], 3)}</td><td>{fmt(r['change_rate'], 3)}</td></tr>" for r in sel_rows)}</table></div>"""
SPEC_SEC = ""
if SPEC:
    SPEC_SEC = f"""<h2>The lens is a narrow filter at the start of the band (F11)</h2>
<p>From the lens file alone, no model: the singular spectrum of each band lens, its participation ratio, the number of singular values at or above a hundredth of the largest, and the share of a random direction that lies in that reachable subspace. The lens at layer 12 reaches under half of the residual's directions at any usable gain; by layer 20 it reaches nearly all. This is why injection is feasible at some layers and not others, and it sits beside the cosine ramp as the second thing learned about the instrument by using it.</p>
<div class="wrap"><table><tr><th>layer</th><th>largest singular value</th><th>condition number</th><th>participation ratio</th><th>singular values at or above 1% of the largest</th><th>reachable share of a random direction</th><th>reachable share of the sixteen injected concepts' rows (median, min)</th></tr>{"".join(f"<tr><td>{L}</td><td>{fmt(v['sigma_max'], 1)}</td><td>{v['condition_number']:,.0f}</td><td>{fmt(v['participation_ratio'], 0)}</td><td>{v['n_sv_ge_1pct']:,}</td><td>{fmt(v['median_share_of_u_c_in_top_1pct_subspace'], 2)}</td><td>{'' if not SPECC else fmt(SPECC['by_layer'][str(L)]['concepts_median'], 2) + ', ' + fmt(SPECC['by_layer'][str(L)]['concepts_min'], 2)}</td></tr>" for L, v in SPEC['layers'].items())}</table></div>"""
COUP_SEC = ""
if COUP and "summary" in COUP:
    cs = COUP["summary"]
    COUP_SEC = f"""<h2>The layer-4 relays do not couple to the band (the Director's hypothesis, causal test)</h2>
<p>Each of the sixteen heads writing layer 4 zeroed alone; per head, the change it causes in the 26 in-band relays' own attention outputs and in the 54 unselected in-band heads'. The three layer-4 relays change the relays' outputs by a median of {fmt(cs['relay_output_change']['layer4_relays_median'], 3)} against {fmt(cs['relay_output_change']['others_median'], 3)} for the other thirteen heads (one-sided p = {fmt(cs['relay_output_change']['mannwhitney_p_greater'], 2)}), and their selectivity, the change caused in the relays over the change caused in the unselected heads, is {', '.join(fmt(x, 2) for x in cs['selectivity_relays_vs_unselected_targets']['layer4_relays'])} against {fmt(min(cs['selectivity_relays_vs_unselected_targets']['others']), 2)} to {fmt(max(cs['selectivity_relays_vs_unselected_targets']['others']), 2)} for the rest: every layer-4 head perturbs the unselected band heads more than the relays. With the weight-composition null of stage 1b, the coupling hypothesis is unsupported by weights and by intervention.</p>"""
INJ_SEC = ""
if INJ and "arms" in INJ:
    ua = INJ["unablated_means"]["same"]; prim = next((a for a in INJ["arms"] if a["k"] == 16), None)
    def cell(t): return f"{fmt(t['relays'], 3)} against {fmt(min(t['control_means']), 3)} to {fmt(max(t['control_means']), 3)} (mean {fmt(float(np.mean(t['control_means'])), 3)}); rank {t['relay_rank_low_first']:g} of {t['n_arms']}"
    INJ_SEC = f"""<h2>The injection test: does downstream recall of an injected concept depend on the relays as a set?</h2>
<p>A lens direction for a concept token is added to the residual stream at layer 12, after block 11, so the relays at blocks 15, 19, 23 and 27 are downstream; the lens reads the concept downstream at the same position and at later positions. The instrument had to be built three times: the gradient direction read at 0.27 one block downstream, the exact pre-image at zero (the layer-12 lens's condition number is 636,586 and the pre-image lives in its near-null space), and the regularised pre-image, lambda {INJ.get('lambda_rel')} of the largest singular value squared at {INJ.get('alpha')} times the residual norm, reads the concept at {fmt(ua.get('16', ua.get(16)), 2)} at layer 16, {fmt(ua.get('20', ua.get(20)), 2)} at 20, {fmt(ua.get('24', ua.get(24)), 2)} at 24 and {fmt(ua.get('28', ua.get(28)), 2)} at 28 with every relay intact. Primary comparison, pre-registered: k = 16, same position, layer 24, the relay arm's rank among twenty arms (nineteen layer-matched random sets), the hypothesis surviving redundancy only if the relays are the lowest; layer 28 beside it, two thirds an output readout by the cosine ramp.</p>
<div class="wrap"><table><tr><th>k</th><th>recall at layer 24: relays against random (range, mean); rank</th><th>recall at layer 28</th><th>recall four positions later at 28</th></tr>{"".join(f"<tr><td>{a['k']}{' (primary)' if a['k'] == 16 else ''}</td><td>{cell(a['tests']['same_24'])}</td><td>{cell(a['tests']['same_28'])}</td><td>{cell(a['tests']['later_28+4'])}</td></tr>" for a in INJ['arms'])}</table></div>
{"" if not prim else "<div class='callout'><strong>Verdict, pre-registered rule.</strong> At k = 16 the relays' recall of the injected concept at layer 24 is " + fmt(prim['tests']['same_24']['relays'], 3) + " against all nineteen random sets at " + fmt(min(prim['tests']['same_24']['control_means']), 3) + ", the unablated value to the event: rank " + f"{prim['tests']['same_24']['relay_rank_low_first']:g}" + " of " + str(prim['tests']['same_24']['n_arms']) + " where survival required rank 1. On 64 events that is 60 against nineteen sets at 59: a rank among ties separated by one event measures the tie-breaking, not an effect. <strong>The primary comparison was uninformative</strong>: nothing moved the metric at 24, so it cannot distinguish the hypothesis from its negation, the hypothesis is untested at layer 24 by this instrument, and the one-event difference is reported and not interpreted. At layer 28, where there is range, the relays' ablation raised recall from " + fmt(INJ['unablated_means']['same'].get('28', INJ['unablated_means']['same'].get(28)), 3) + " to " + fmt(prim['tests']['same_28']['relays'], 3) + " while random sets scattered around the unablated value, rank " + f"{prim['tests']['same_28']['relay_rank_low_first']:g}" + " of " + str(prim['tests']['same_28']['n_arms']) + ": four events of 64 in the direction opposite to the hypothesis, one-sided p = 0.10, the pre-registered third outcome, a hypothesis this run generated rather than one it tested (a secondary readout read after the primary failed), written as a suppression hypothesis under the magnitude caveat (the injected vector is twice the residual at its position), with a contrast-direction injection at layer 20 as its test. At every later position, on every arm, the injected concept is never read: zero broadcast across tokens for this instrument, stated no wider.</div>"}"""
KVF = [] if not KVG else [[int(b), e["prereg"]["F_J"], e["prereg"]["F_null_median"], e["prereg"]["F_null_max"]] for b, e in KVG["blocks"].items()]
html = f"""<title>Broadcast Heads</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{ --ground:#f6f4ef; --panel:#fff; --ink:#1c2128; --muted:#5c6570; --rule:#d8d3c8; --att:#c8591b; --rec:#2a6f97; --ctl:#8a8f96; --grid:#e7e2d8; --soft:#f3ece2; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --ground:#14171b; --panel:#1c2026; --ink:#e8e6e1; --muted:#a2a8b1; --rule:#343a42; --att:#f0955a; --rec:#7cc1ea; --ctl:#a0a6ae; --grid:#2a3038; --soft:#2a241c; }} }}
:root[data-theme="dark"] {{ --ground:#14171b; --panel:#1c2026; --ink:#e8e6e1; --muted:#a2a8b1; --rule:#343a42; --att:#f0955a; --rec:#7cc1ea; --ctl:#a0a6ae; --grid:#2a3038; --soft:#2a241c; }}
body {{ background:var(--ground); color:var(--ink); font-family:"IBM Plex Sans", system-ui, sans-serif; font-size:16px; line-height:1.5; margin:0; }}
main {{ max-width:1040px; margin:0 auto; padding:40px 24px 64px; }}
h1 {{ font-family:"Fraunces", Georgia, serif; font-weight:700; font-size:2.4rem; line-height:1.1; margin:0 0 8px; text-wrap:balance; }}
h2 {{ font-family:"Fraunces", Georgia, serif; font-weight:500; font-size:1.35rem; margin:36px 0 10px; }}
.lede {{ font-size:1.1rem; max-width:64ch; color:var(--muted); margin:0 0 28px; }}
.eyebrow {{ font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-bottom:10px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:14px; margin:20px 0 8px; }}
.tile {{ background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:14px 16px; }}
.tile .n {{ font-family:"IBM Plex Mono", monospace; font-size:1.7rem; font-variant-numeric:tabular-nums; }} .tile .l {{ font-size:.85rem; color:var(--muted); }}
.tile.att .n {{ color:var(--att); }} .tile.rec .n {{ color:var(--rec); }}
.charts {{ display:grid; grid-template-columns:1fr; gap:18px; }} @media (min-width: 880px) {{ .charts {{ grid-template-columns:1fr 1fr; }} }}
.chart {{ background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:12px 12px 6px; }} .chart h3 {{ margin:0 0 4px; font-size:1rem; font-weight:600; }} .chart .sub {{ font-size:.85rem; color:var(--muted); margin:0 0 6px; }}
svg {{ width:100%; height:auto; display:block; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:.85rem; color:var(--muted); margin:6px 4px 4px; }} .sw {{ display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:6px; border-radius:2px; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:.92rem; }} th, td {{ text-align:right; padding:6px 10px; border-bottom:1px solid var(--rule); }} th:first-child, td:first-child {{ text-align:left; }}
th {{ font-weight:600; color:var(--muted); font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; }} .wrap {{ overflow-x:auto; }} p {{ max-width:70ch; }}
.callout {{ background:var(--soft); border-left:4px solid var(--att); padding:12px 16px; border-radius:4px; max-width:70ch; }} code {{ font-family:"IBM Plex Mono", monospace; font-size:.9em; }}
</style>
<main>
<div class="eyebrow">Qwen3.5-4B · 4-bit · hosted J-lens · weights only · 2026-09-06</div>
<h1>Broadcast Heads</h1>
<p class="lede">Which attention mediates entry to the workspace? Following the paper's section 4.3.2, every head's output map is scored on how strongly and how faithfully it relays the lens's own directions, against the same directions rotated and against MLP output rows. A head that relays the lens selectively scores high on both and low on the controls.</p>
<div class="tiles">
  <div class="tile att"><div class="n">{len(SEL['selected_attention'])} / 112</div><div class="l">attention heads whose label preservation on the lens is at least double the best control (beyond every one of twenty rotated and twenty MLP-row draws and double their medians; layer 32 excluded, the lens being the identity there)</div></div>
  <div class="tile att"><div class="n">{SEL['in_band']} / 64</div><div class="l">of those, in the band (layers 16 to 28)</div></div>
  <div class="tile att"><div class="n">{fmt(S['attention']['J_mrr_median'],3)} / {fmt(S['attention']['J_rot_mrr_median'],3)}</div><div class="l">attention: label preservation on the lens directions / on the same directions rotated</div></div>
  <div class="tile rec"><div class="n">{fmt(S['recurrent']['J_mrr_median'],4)} / {fmt(S['recurrent']['random_mrr_median'],4)}</div><div class="l">recurrent value paths: label preservation on the lens / on random directions</div></div>
</div>
<div class="charts">
  <div class="chart"><h3>Where the relays are</h3><p class="sub">Heads whose label preservation on the lens directions exceeds every one of forty null draws and doubles their medians, by the layer the block writes, as a share of the heads there. Gain did not discriminate between populations here, so the set is selected on preservation and gain is reported beside it.</p><div id="count"></div>
    <div class="legend"><span><i class="sw" style="background:var(--att)"></i>attention (16 heads per layer)</span><span><i class="sw" style="background:var(--rec)"></i>recurrent value paths (32 per layer)</span></div></div>
  <div class="chart"><h3>How faithfully the lens directions survive each map</h3><p class="sub">Median label preservation (mean reciprocal rank) by layer: solid on the lens directions, dotted on the same directions rotated. Attention preserves them; the recurrent weight-only paths do not.</p><div id="mrr"></div>
    <div class="legend"><span><i class="sw" style="background:var(--att)"></i>attention</span><span><i class="sw" style="background:var(--rec)"></i>recurrent</span><span><i class="sw" style="background:var(--ctl)"></i>rotated control</span></div></div>
</div>
<div class="callout">Entry and retrieval share heads without being one operation, said of six heads and not as a rate. Of the six strongest retrieval heads from the retrieval run, block 19 head 12 and block 23 head 9 are selective lens relays and block 19 head 15 enters at the margin; block 19 head 2 preserves every direction, a copy head, and is not selective; block 27 head 1 and block 3 head 15 are not relays. The strongest relay, block 19 head 0, was not a strong retrieval head. The relays sit at layers 20, 24 and 28, with the largest count at 28: the paper's finding that they concentrate in the first half of the workspace does not reproduce here, a prediction written before the run and refuted.{orth_line} The recurrent value paths are not relays by magnitude: their median loses to its own control and the strongest of 768 sits below the median attention head.</div>
{STAGE2}
{DIAG_SEC}
{PRED_SEC}
{KV_SEC}
{SING_SEC}
{SPEC_SEC}
{COUP_SEC}
{INJ_SEC}
{"" if COMP is None else f'''<h2>Stage 1b: do the layer-4 relays compose with the band?</h2>
<p>For the three layer-4 relays against all 64 band heads, {COMP["n_pairs"]} pairs: pairs whose composition exceeds the 99th percentile of the population of every layer-4 and layer-8 head paired with the band: Q {COMP["pairs_above_p99_of_pair_population"]["Q"]}, K {COMP["pairs_above_p99_of_pair_population"]["K"]}, V {COMP["pairs_above_p99_of_pair_population"]["V"]} (chance about {COMP["n_pairs"] * 0.01:.1f} each).</p>
<div class="wrap"><table><tr><th>layer-4 relay</th><th>preservation: lens</th><th>token identity</th><th>lens minus identity component</th><th>random</th><th>mean |cos| lens vs identity</th></tr>{"".join(f"<tr><td>block {x['block']} head {x['head']}{' (relay)' if x['relay'] else ''}</td><td>{fmt(x['mrr_lens'],3)}</td><td>{fmt(x['mrr_token_identity'],3)}</td><td>{fmt(x['mrr_lens_orthogonal_to_identity'],3)}</td><td>{fmt(x['mrr_random'],3)}</td><td>{fmt(x['mean_abs_cos_lens_identity'],3)}</td></tr>" for x in (SEP or []) if x['layer'] == 4 and x['relay'])}</table></div>'''}
<h2>The strongest attention heads, ordered by preservation margin</h2>
<div class="wrap"><table><tr><th>head</th><th>gain: lens / rotated / MLP rows</th><th>label preservation: lens / rotated / MLP rows</th><th>relays the lens</th></tr>{''.join(head_row(r) for r in top_att)}</table></div>
<h2>Method</h2>
<p>Weight arithmetic only, no forward pass, 26 seconds. For each attention head the map is its output projection composed with its value projection. For each recurrent value path it is the output projection composed with the value slice of the input projection, which omits the causal convolution, the state and the gated normalisation between them, all input-dependent; the two channels are therefore different objects and are shown apart. Populations at each block's input layer: 2,000 lens directions from the hosted Jacobian lens and the tied embedding, the same directions under one fixed random rotation, 2,000 MLP output rows of the preceding block, and random unit directions. Gain is the mean image norm divided by the median over random directions; label preservation is the mean reciprocal rank of each direction's own image among all. No percentile selection: a head relays the lens when its label preservation on the lens directions is at least double the best control; the paper's gain criterion did not discriminate between populations on this model (41 heads passed preservation and failed only gain against 17 passing both), so gain is reported beside the selection rather than gating it. Layer 32 is excluded because the lens is the identity there. The count is stated as n of N and the whole ordered list is in the record.</p>
</main>
<script>
const SER = {json.dumps(series)}; const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function bars() {{
  const W=520, H=330, m={{l:52, r:14, t:12, b:40}};
  const X = L => m.l + ((L - 0.5) / 32) * (W - m.l - m.r), Y = v => H - m.b - v * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="share of heads relaying the lens by layer">`;
  for (const v of [0, 0.25, 0.5]) s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(v)}}" y2="${{Y(v)}}" stroke="${{css('--grid')}}"/><text x="${{m.l-8}}" y="${{Y(v)+4}}" text-anchor="end" font-size="11" fill="${{css('--muted')}}">${{v}}</text>`;
  for (const L of [4,8,12,16,20,24,28,32]) s += `<text x="${{X(L)}}" y="${{H-m.b+16}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{L}}</text>`;
  s += `<text x="${{(m.l+W-m.r)/2}}" y="${{H-6}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">layer written by the block</text><text transform="rotate(-90)" x="${{-(H)/2}}" y="14" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">share of heads relaying the lens</text>`;
  const bw = (W - m.l - m.r) / 32 * 0.8;
  for (const [L, c, n] of SER.rec_count) s += `<rect x="${{X(L) - bw/2}}" y="${{Y(c/n)}}" width="${{bw}}" height="${{Y(0) - Y(c/n)}}" fill="${{css('--rec')}}" opacity=".85"/>`;
  for (const [L, c, n] of SER.att_count) s += `<rect x="${{X(L) - bw/2}}" y="${{Y(c/n)}}" width="${{bw}}" height="${{Y(0) - Y(c/n)}}" fill="${{css('--att')}}"/>`;
  document.getElementById('count').innerHTML = s + '</svg>';
}}
function mrr() {{
  const W=520, H=330, m={{l:52, r:14, t:12, b:40}};
  const X = L => m.l + ((L - 0.5) / 32) * (W - m.l - m.r), Y = v => H - m.b - (Math.log10(Math.max(v, 1e-3)) + 3) / 3 * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="label preservation by layer">`;
  for (const v of [0.001, 0.01, 0.1, 1]) s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(v)}}" y2="${{Y(v)}}" stroke="${{css('--grid')}}"/><text x="${{m.l-8}}" y="${{Y(v)+4}}" text-anchor="end" font-size="11" fill="${{css('--muted')}}">${{v}}</text>`;
  for (const L of [4,8,12,16,20,24,28,32]) s += `<text x="${{X(L)}}" y="${{H-m.b+16}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{L}}</text>`;
  s += `<text x="${{(m.l+W-m.r)/2}}" y="${{H-6}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">layer written by the block</text><text transform="rotate(-90)" x="${{-(H)/2}}" y="14" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">median label preservation (log)</text>`;
  const line = (pts, c, dash) => `<path d="${{pts.map((p,i)=>`${{i?'L':'M'}}${{X(p[0])}} ${{Y(p[1])}}`).join(' ')}}" fill="none" stroke="${{css(c)}}" stroke-width="${{dash?1.5:2.5}}" ${{dash?'stroke-dasharray="3 4"':''}}/>` + (dash ? '' : pts.map(p=>`<circle cx="${{X(p[0])}}" cy="${{Y(p[1])}}" r="3.5" fill="${{css(c)}}"/>`).join(''));
  s += line(SER.rec_mrr_rot, '--ctl', true) + line(SER.att_mrr_rot, '--ctl', true) + line(SER.rec_mrr, '--rec', false) + line(SER.att_mrr, '--att', false);
  document.getElementById('mrr').innerHTML = s + '</svg>';
}}
bars(); mrr();
const ABLS = {json.dumps(abl_series)}; const ABLI = {json.dumps(abl_in_series)};
function line2(id, yi, ci, label, ymax) {{
  if (!ABLS.length) return;
  const W=520, H=330, m={{l:52, r:14, t:12, b:40}}; const ks = ABLS.map(r => r[0]); const kmax = Math.max(...ks);
  const X = k => m.l + (Math.log2(k) / Math.log2(kmax)) * (W - m.l - m.r), Y = v => H - m.b - v / ymax * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{label}}">`;
  for (const v of [0, 0.25, 0.5, 0.75, 1].filter(v => v <= ymax)) s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(v)}}" y2="${{Y(v)}}" stroke="${{css('--grid')}}"/><text x="${{m.l-8}}" y="${{Y(v)+4}}" text-anchor="end" font-size="11" fill="${{css('--muted')}}">${{v}}</text>`;
  for (const k of ks) s += `<text x="${{X(k)}}" y="${{H-m.b+16}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{k}}</text>`;
  s += `<text x="${{(m.l+W-m.r)/2}}" y="${{H-6}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">heads ablated, k (log)</text><text transform="rotate(-90)" x="${{-(H)/2}}" y="14" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{label}}</text>`;
  for (const [idx, c] of [[ci, '--ctl'], [yi, '--att']]) {{ s += `<path d="${{ABLS.map((r,i)=>`${{i?'L':'M'}}${{X(r[0])}} ${{Y(r[idx])}}`).join(' ')}}" fill="none" stroke="${{css(c)}}" stroke-width="2.5"/>` + ABLS.map(r=>`<circle cx="${{X(r[0])}}" cy="${{Y(r[idx])}}" r="3.5" fill="${{css(c)}}"/>`).join(''); }}
  if (ABLI.length) s += `<path d="${{ABLI.map((r,i)=>`${{i?'L':'M'}}${{X(r[0])}} ${{Y(r[yi])}}`).join(' ')}}" fill="none" stroke="${{css('--rec')}}" stroke-width="2" stroke-dasharray="5 4"/>` + ABLI.map(r=>`<circle cx="${{X(r[0])}}" cy="${{Y(r[yi])}}" r="3" fill="${{css('--rec')}}"/>`).join('');
  document.getElementById(id).innerHTML = s + '</svg>';
}}
const ABLF = {json.dumps(abl_f)}; const KVF = {json.dumps(KVF)};
function curve(id, yk, ck, lo, hi, label, ymax) {{
  if (!ABLF.length) return;
  const W=520, H=330, m={{l:52, r:14, t:12, b:52}}; const ks = ABLF.map(a => a.k); const kmax = Math.max(...ks);
  const X = k => m.l + (Math.log2(k + 1) / Math.log2(kmax + 1)) * (W - m.l - m.r), Y = v => H - m.b - v / ymax * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{label}}">`;
  for (const v of [0, 0.25, 0.5, 0.75, 1].filter(v => v <= ymax)) s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(v)}}" y2="${{Y(v)}}" stroke="${{css('--grid')}}"/><text x="${{m.l-8}}" y="${{Y(v)+4}}" text-anchor="end" font-size="11" fill="${{css('--muted')}}">${{v}}</text>`;
  for (const a of ABLF) {{ s += `<text x="${{X(a.k)}}" y="${{H-m.b+16}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{a.k}}</text>`; if (a.k > 16) s += `<text x="${{X(a.k)}}" y="${{H-m.b+30}}" text-anchor="middle" font-size="9" fill="${{css('--muted')}}">${{a.label.replace('no floor, all ', 'all ').replace('floor ', '')}}</text>`; }}
  s += `<text x="${{(m.l+W-m.r)/2}}" y="${{H-6}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">heads ablated, k (log scale of k + 1); floors under the tail arms</text><text transform="rotate(-90)" x="${{-(H)/2}}" y="14" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{label}}</text>`;
  if (lo) s += `<path d="${{ABLF.map((a,i)=>`${{i?'L':'M'}}${{X(a.k)}} ${{Y(a[lo])}}`).join(' ')}} ${{ABLF.slice().reverse().map(a=>`L${{X(a.k)}} ${{Y(a[hi])}}`).join(' ')}} Z" fill="${{css('--ctl')}}" opacity=".18"/>`;
  for (const [key, c] of [[ck, '--ctl'], [yk, '--att']]) s += `<path d="${{ABLF.map((a,i)=>`${{i?'L':'M'}}${{X(a.k)}} ${{Y(a[key])}}`).join(' ')}}" fill="none" stroke="${{css(c)}}" stroke-width="2.5"/>` + ABLF.map(a=>`<circle cx="${{X(a.k)}}" cy="${{Y(a[key])}}" r="3.5" fill="${{css(c)}}"/>`).join('');
  document.getElementById(id).innerHTML = s + '</svg>';
}}
curve('ablf', 'recall', 'ctl', 'ctl_min', 'ctl_max', 'top-25 overlap with the unablated readout', 1.0);
curve('chgf', 'change', 'ctl_change', null, null, 'next-token change rate', Math.max(0.1, ...ABLF.map(a => Math.max(a.change, a.ctl_change))) * 1.15);
function kvf() {{
  if (!KVF.length || !document.getElementById('kvf')) return;
  const W=520, H=260, m={{l:52, r:14, t:12, b:40}}; const ymax = Math.max(1, ...KVF.map(r => Math.max(r[1], r[3]))) * 1.15;
  const X = i => m.l + (i + 0.5) / KVF.length * (W - m.l - m.r), Y = v => H - m.b - v / ymax * (H - m.t - m.b); const bw = (W - m.l - m.r) / KVF.length * 0.5;
  let s = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="between/within group variance ratio by block">`;
  const step = ymax > 10 ? 5 : ymax > 4 ? 2 : 1; for (let v = 0; v <= ymax; v += step) s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(v)}}" y2="${{Y(v)}}" stroke="${{css('--grid')}}"/><text x="${{m.l-8}}" y="${{Y(v)+4}}" text-anchor="end" font-size="11" fill="${{css('--muted')}}">${{v}}</text>`;
  KVF.forEach((r, i) => {{ s += `<rect x="${{X(i)-bw/2}}" y="${{Y(r[1])}}" width="${{bw}}" height="${{Y(0)-Y(r[1])}}" fill="${{css('--att')}}"/><line x1="${{X(i)-bw/2-6}}" x2="${{X(i)+bw/2+6}}" y1="${{Y(r[3])}}" y2="${{Y(r[3])}}" stroke="${{css('--ctl')}}" stroke-width="2"/><line x1="${{X(i)-bw/2-6}}" x2="${{X(i)+bw/2+6}}" y1="${{Y(r[2])}}" y2="${{Y(r[2])}}" stroke="${{css('--ctl')}}" stroke-width="2" stroke-dasharray="3 3"/><text x="${{X(i)}}" y="${{H-m.b+16}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">layer ${{r[0]+1}}</text>`; }});
  s += `<text transform="rotate(-90)" x="${{-(H)/2}}" y="14" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">F, between over within group</text>`;
  document.getElementById('kvf').innerHTML = s + '</svg>';
}}
kvf();
const SING = {json.dumps(SING_ROWS)}; const SINGL = {json.dumps(SING_LAYERS)};
function singScatter() {{
  if (!SING.length || !document.getElementById('sing')) return;
  const W=520, H=330, m={{l:52, r:14, t:12, b:40}}; const ymax = Math.max(...SING.map(r => r[4])) * 1.1;
  const X = v => m.l + v * (W - m.l - m.r), Y = v => H - m.b - v / ymax * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="single-head damage against preservation">`;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) s += `<text x="${{X(v)}}" y="${{H-m.b+16}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{v}}</text>`;
  const step = ymax > 0.2 ? 0.1 : 0.05; for (let v = 0; v <= ymax; v += step) s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(v)}}" y2="${{Y(v)}}" stroke="${{css('--grid')}}"/><text x="${{m.l-8}}" y="${{Y(v)+4}}" text-anchor="end" font-size="11" fill="${{css('--muted')}}">${{v.toFixed(2)}}</text>`;
  s += `<text x="${{(m.l+W-m.r)/2}}" y="${{H-6}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">lens preservation (stage 1, orthogonalised population)</text><text transform="rotate(-90)" x="${{-(H)/2}}" y="14" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">damage at layer 28, head ablated alone</text>`;
  for (const r of SING) s += r[7] ? `<circle cx="${{X(r[3])}}" cy="${{Y(r[4])}}" r="4" fill="${{css('--att')}}"><title>layer ${{r[0]}} block ${{r[1]}} head ${{r[2]}}, rank ${{r[8]}}</title></circle>` : `<circle cx="${{X(r[3])}}" cy="${{Y(r[4])}}" r="3.5" fill="none" stroke="${{css('--ctl')}}" stroke-width="1.5"><title>layer ${{r[0]}} block ${{r[1]}} head ${{r[2]}}</title></circle>`;
  document.getElementById('sing').innerHTML = s + '</svg>';
}}
function singLayers() {{
  if (!SINGL.length || !document.getElementById('singl')) return;
  const W=520, H=330, m={{l:52, r:14, t:12, b:40}}; const ymax = Math.max(...SINGL.flatMap(r => [r[1] || 0, r[2]])) * 1.2;
  const X = L => m.l + ((L - 10) / 20) * (W - m.l - m.r), Y = v => H - m.b - v / ymax * (H - m.t - m.b); const bw = (W - m.l - m.r) / 20 * 0.7;
  let s = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="median single-head damage by layer">`;
  const step = ymax > 0.2 ? 0.1 : 0.05; for (let v = 0; v <= ymax; v += step) s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(v)}}" y2="${{Y(v)}}" stroke="${{css('--grid')}}"/><text x="${{m.l-8}}" y="${{Y(v)+4}}" text-anchor="end" font-size="11" fill="${{css('--muted')}}">${{v.toFixed(2)}}</text>`;
  for (const r of SINGL) {{ s += `<text x="${{X(r[0])}}" y="${{H-m.b+16}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{r[0]}}</text>`; s += `<rect x="${{X(r[0]) - bw}}" y="${{Y(r[2])}}" width="${{bw}}" height="${{Y(0) - Y(r[2])}}" fill="${{css('--ctl')}}" opacity=".8"/>`; if (r[1] !== null) s += `<rect x="${{X(r[0])}}" y="${{Y(r[1])}}" width="${{bw}}" height="${{Y(0) - Y(r[1])}}" fill="${{css('--att')}}"/>`; }}
  s += `<text x="${{(m.l+W-m.r)/2}}" y="${{H-6}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">layer written</text><text transform="rotate(-90)" x="${{-(H)/2}}" y="14" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">median damage at layer 28, alone</text>`;
  document.getElementById('singl').innerHTML = s + '</svg>';
}}
singScatter(); singLayers();
function layersChart() {{
  if (!ABLF.length || !document.getElementById('abll')) return;
  const big = ABLF[ABLF.length - 1], mid = ABLF.find(a => a.k === 8) || ABLF[1]; const Ls = Object.keys(big.by_layer).map(Number);
  const W=520, H=330, m={{l:52, r:14, t:12, b:40}}; const X = L => m.l + ((L - 11) / 18) * (W - m.l - m.r), Y = v => H - m.b - v * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="overlap by readout layer">`;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(v)}}" y2="${{Y(v)}}" stroke="${{css('--grid')}}"/><text x="${{m.l-8}}" y="${{Y(v)+4}}" text-anchor="end" font-size="11" fill="${{css('--muted')}}">${{v}}</text>`;
  for (const L of Ls) s += `<text x="${{X(L)}}" y="${{H-m.b+16}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{L}}</text>`;
  s += `<text x="${{(m.l+W-m.r)/2}}" y="${{H-6}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">readout layer</text><text transform="rotate(-90)" x="${{-(H)/2}}" y="14" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">top-25 overlap with the unablated readout</text>`;
  const line = (arm, idx, c, dash) => `<path d="${{Ls.map((L,i)=>`${{i?'L':'M'}}${{X(L)}} ${{Y(arm.by_layer[L][idx])}}`).join(' ')}}" fill="none" stroke="${{css(c)}}" stroke-width="${{dash?1.5:2.5}}" ${{dash?'stroke-dasharray="4 4" opacity=".6"':''}}/>` + (dash ? '' : Ls.map(L=>`<circle cx="${{X(L)}}" cy="${{Y(arm.by_layer[L][idx])}}" r="3.5" fill="${{css(c)}}"/>`).join(''));
  s += line(mid, 1, '--ctl', true) + line(mid, 0, '--att', true) + line(big, 1, '--ctl', false) + line(big, 0, '--att', false);
  document.getElementById('abll').innerHTML = s + '</svg>';
}}
layersChart();
line2('abl\', 1, 2, 'recall at 25, downstream readout layers', 1.0); line2('chg', 3, 4, 'next-token change rate', Math.max(0.1, ...ABLS.map(r => Math.max(r[3], r[4])) ) * 1.15);
</script>
"""
(HERE / "broadcast-heads.html").write_text(html, encoding="utf-8"); print("wrote", HERE / "broadcast-heads.html")
