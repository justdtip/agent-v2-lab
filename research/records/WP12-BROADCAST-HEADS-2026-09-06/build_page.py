"""Build the Broadcast Heads page from out/broadcast_heads.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "out" / "broadcast_heads.json")); rows = d["rows"]; S = d["summary"]
SEL = json.load(open(HERE / "out" / "selection_nulldist.json"))
ABL = json.load(open(HERE / "out" / "ablation.json")) if (HERE / "out" / "ablation.json").exists() else None
ABL_IN = json.load(open(HERE / "out" / "ablation_inband.json")) if (HERE / "out" / "ablation_inband.json").exists() else None
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
{"" if ABL is None else f'''<h2>Stage 2: ablating the relays, graded</h2>
<div class="charts">
  <div class="chart"><h3>Lens recall downstream of the ablated heads</h3><p class="sub">The share of the unablated readout's top-25 tokens that survive, at readout layers with an ablated head at or below them, as the k strongest relays are zeroed; the layer-matched random set of the same size beside it. Reproducibility floor: {"bit-identical" if ABL["reproducibility"]["bit_identical"] else "measured, " + fmt(ABL["reproducibility"]["upstream_floor"]["recall"], 4)}.</p><div id="abl"></div>
    <div class="legend"><span><i class="sw" style="background:var(--att)"></i>all 29 relays</span><span><i class="sw" style="background:var(--rec)"></i>in-band 24 only</span><span><i class="sw" style="background:var(--ctl)"></i>layer-matched random</span></div></div>
  <div class="chart"><h3>Next-token change rate</h3><p class="sub">The share of positions whose top-1 prediction changes under the same ablations. A recall drop is read only where this stays within twice the control's; rows are shown either way.</p><div id="chg"></div>
    <div class="legend"><span><i class="sw" style="background:var(--att)"></i>relays</span><span><i class="sw" style="background:var(--ctl)"></i>layer-matched random</span></div></div>
</div>
<div class="wrap"><table><tr><th>set</th><th>k</th><th>recall downstream, relays / random</th><th>change rate, relays / random</th><th>specificity holds</th></tr>{"".join(f"<tr><td>all 29</td><td>{r[0]}</td><td>{fmt(r[1],3)} / {fmt(r[2],3)}</td><td>{fmt(r[3],3)} / {fmt(r[4],3)}</td><td>{'yes' if r[5] else 'no'}</td></tr>" for r in abl_series)}{"".join(f"<tr><td>in-band 24</td><td>{r[0]}</td><td>{fmt(r[1],3)} / {fmt(r[2],3)}</td><td>{fmt(r[3],3)} / {fmt(r[4],3)}</td><td>{'yes' if r[5] else 'no'}</td></tr>" for r in abl_in_series)}</table></div>'''}
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
line2('abl', 1, 2, 'recall at 25, downstream readout layers', 1.0); line2('chg', 3, 4, 'next-token change rate', Math.max(0.1, ...ABLS.map(r => Math.max(r[3], r[4])) ) * 1.15);
</script>
"""
(HERE / "broadcast-heads.html").write_text(html, encoding="utf-8"); print("wrote", HERE / "broadcast-heads.html")
