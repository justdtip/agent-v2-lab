"""Build the Retrieval Channels page from out-full/retrieval_probe.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / (sys.argv[1] if len(sys.argv) > 1 else "out-full") / "retrieval_probe.json"
OUT = HERE / "retrieval-channels.html"
d = json.load(open(SRC)); S = d["summary"]; R = d["records"]; items = d["items"]; C = d["query_cosine"]
ATT_LAYER = {b: 4 + 4 * b for b in range(8)}          # attention block b writes layer 4, 8, ..., 32
REC_LAYER = {}
li = 0
for L in range(1, 33):
    if L % 4 != 0: REC_LAYER[li] = L; li += 1        # recurrent blocks in order write the layers that are not multiples of 4

def pct(v, p): return float(np.percentile(v, p)) if len(v) else None
def fmt(x, dd=2): return "n/a" if x is None else f"{x:,.{dd}f}"

kinds = sorted({i["kind"] for i in items})
rows_kind = []
for k in kinds:
    it = [i for i in items if i["kind"] == k and not i["control"]]
    rr = [r for r in R if r["kind"] == k and not r["control"] and r["channel"] == "recurrent"]
    ra = [r for r in R if r["kind"] == k and not r["control"] and r["channel"] == "attention"]
    rows_kind.append(f"<tr><td>{k}</td><td>{sum(i['correct'] for i in it)}/{len(it)}</td>"
                     f"<td>{fmt(pct([r['ratio_null'] for r in rr],50))} / {fmt(pct([r['ratio_null'] for r in rr],99))}</td>"
                     f"<td>{fmt(pct([r['ratio_null'] for r in ra],50))} / {fmt(pct([r['ratio_null'] for r in ra],99))}</td>"
                     f"<td>{fmt(np.mean([r['hit_at_k'] for r in rr]) if rr else None)} vs {fmt(np.mean([r['null_hit_rate'] for r in rr]) if rr else None)}</td>"
                     f"<td>{fmt(np.mean([r['hit_at_k'] for r in ra]) if ra else None)} vs {fmt(np.mean([r['null_hit_rate'] for r in ra]) if ra else None)}</td></tr>")

# per-layer series: median and p90 of ratio_null per block, both channels, retrieval items only
def per_layer(ch, layer_map):
    out = []
    for b, L in layer_map.items():
        v = [r["ratio_null"] for r in R if r["channel"] == ch and r["block"] == b and not r["control"]]
        if v: out.append([L, pct(v, 50), pct(v, 90), pct(v, 99), len(v)])
    return out
series = {"recurrent": per_layer("recurrent", REC_LAYER), "attention": per_layer("attention", ATT_LAYER)}
# query cosine per channel: within retrieval vs within control (median over heads)
qc = {ch: {m: pct([r[m] for r in C if r["channel"] == ch and r[m] is not None], 50) for m in ("within_context_retrieval_median", "within_context_control_median", "across_context_median")} for ch in ("recurrent", "attention")}
rec = S["by_channel"]["recurrent"]; att = S["by_channel"]["attention"]
top_att = S["top_heads"]["attention"][:8]; top_rec = S["top_heads"]["recurrent"][:8]
def head_rows(lst, layer_map):
    return "".join(f"<tr><td>L{layer_map[h['block']]} b{h['block']} h{h['head']}</td><td>{h['kind']}</td><td>{fmt(h['span_share'],3)}</td><td>{fmt(h['null_share_mean'],3)}</td><td>{fmt(h['ratio_null'],1)}</td><td>{'yes' if h['hit_at_k'] else 'no'}</td><td>{'yes' if h['correct'] else 'no'}</td></tr>" for h in lst)

html = f"""<title>Retrieval Channels</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{ --ground:#f6f4ef; --panel:#fff; --ink:#1c2128; --muted:#5c6570; --rule:#d8d3c8; --rec:#2a6f97; --att:#c8591b; --null:#8a8f96; --grid:#e7e2d8; --soft:#e7eef3; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --ground:#14171b; --panel:#1c2026; --ink:#e8e6e1; --muted:#a2a8b1; --rule:#343a42; --rec:#7cc1ea; --att:#f0955a; --null:#a0a6ae; --grid:#2a3038; --soft:#1e2c36; }} }}
:root[data-theme="dark"] {{ --ground:#14171b; --panel:#1c2026; --ink:#e8e6e1; --muted:#a2a8b1; --rule:#343a42; --rec:#7cc1ea; --att:#f0955a; --null:#a0a6ae; --grid:#2a3038; --soft:#1e2c36; }}
body {{ background:var(--ground); color:var(--ink); font-family:"IBM Plex Sans", system-ui, sans-serif; font-size:16px; line-height:1.5; margin:0; }}
main {{ max-width:1040px; margin:0 auto; padding:40px 24px 64px; }}
h1 {{ font-family:"Fraunces", Georgia, serif; font-weight:700; font-size:2.4rem; line-height:1.1; margin:0 0 8px; text-wrap:balance; }}
h2 {{ font-family:"Fraunces", Georgia, serif; font-weight:500; font-size:1.35rem; margin:36px 0 10px; }}
.lede {{ font-size:1.1rem; max-width:64ch; color:var(--muted); margin:0 0 28px; }}
.eyebrow {{ font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-bottom:10px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:14px; margin:20px 0 8px; }}
.tile {{ background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:14px 16px; }}
.tile .n {{ font-family:"IBM Plex Mono", monospace; font-size:1.7rem; font-variant-numeric:tabular-nums; }}
.tile .l {{ font-size:.85rem; color:var(--muted); }}
.tile.rec .n {{ color:var(--rec); }} .tile.att .n {{ color:var(--att); }}
.charts {{ display:grid; grid-template-columns:1fr; gap:18px; }}
@media (min-width: 880px) {{ .charts {{ grid-template-columns:1fr 1fr; }} }}
.chart {{ background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:12px 12px 6px; }}
.chart h3 {{ margin:0 0 4px; font-size:1rem; font-weight:600; }} .chart .sub {{ font-size:.85rem; color:var(--muted); margin:0 0 6px; }}
svg {{ width:100%; height:auto; display:block; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:.85rem; color:var(--muted); margin:6px 4px 4px; }}
.sw {{ display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:6px; border-radius:2px; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:.92rem; }}
th, td {{ text-align:right; padding:6px 10px; border-bottom:1px solid var(--rule); }} th:first-child, td:first-child {{ text-align:left; }}
th {{ font-weight:600; color:var(--muted); font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; }}
.wrap {{ overflow-x:auto; }} p {{ max-width:70ch; }}
.callout {{ background:var(--soft); border-left:4px solid var(--rec); padding:12px 16px; border-radius:4px; max-width:70ch; }}
code {{ font-family:"IBM Plex Mono", monospace; font-size:.9em; }}
</style>
<main>
<div class="eyebrow">Qwen3.5-4B · 4-bit · retrieval-heavy text · {S['items']} questions · 2026-09-06</div>
<h1>Retrieval Channels</h1>
<p class="lede">When the model has to look something up, which channel reads the place where the answer sits: the recurrent heads that carry a running summary, or the attention heads? Every read weight is tested against the answer's known token span, under one statistic for both channels.</p>
<div class="tiles">
  <div class="tile rec"><div class="n">{fmt(rec['retrieval'].get('ratio_null_percentiles', {}).get('50'))}</div><div class="l">recurrent heads: median mass on the answer span, as a multiple of the random-query null</div></div>
  <div class="tile att"><div class="n">{fmt(att['retrieval'].get('ratio_null_percentiles', {}).get('50'))}</div><div class="l">attention heads: the same statistic</div></div>
  <div class="tile rec"><div class="n">{fmt(rec['retrieval'].get('hit_at_k_rate'))} / {fmt(rec['retrieval'].get('null_hit_rate_mean'))}</div><div class="l">recurrent: answer span in the top-10 far sources, real / null</div></div>
  <div class="tile att"><div class="n">{fmt(att['retrieval'].get('hit_at_k_rate'))} / {fmt(att['retrieval'].get('null_hit_rate_mean'))}</div><div class="l">attention: answer span in the top-10 far sources, real / null</div></div>
  <div class="tile"><div class="n">{fmt(S['accuracy_retrieval'])}</div><div class="l">the model's accuracy on the retrieval questions, whole answer under teacher forcing</div></div>
  <div class="tile"><div class="n">{fmt(S['attention_reconstruction']['worst_relative_error'],3)}</div><div class="l">worst attention reconstruction error, share of the block's output scale, across {S['attention_reconstruction']['blocks_checked']} blocks</div></div>
</div>
<div class="charts">
  <div class="chart"><h3>Mass on the answer span, by layer</h3><p class="sub">Median (solid) and 99th percentile (thin) across heads and questions of the span mass as a multiple of the random-query null. One means no more than a random query would put there.</p><div id="layers"></div>
    <div class="legend"><span><i class="sw" style="background:var(--rec)"></i>recurrent blocks</span><span><i class="sw" style="background:var(--att)"></i>attention blocks</span><span><i class="sw" style="background:var(--null)"></i>null = 1</span></div></div>
  <div class="chart"><h3>Does the query move with the question?</h3><p class="sub">Median cosine between a head's read queries on the same context: two retrieval questions, two no-lookup controls, and across contexts. A query that moves more for retrieval than for wording is dynamic addressing.</p><div id="cos"></div>
    <div class="legend"><span><i class="sw" style="background:var(--rec)"></i>recurrent</span><span><i class="sw" style="background:var(--att)"></i>attention</span></div></div>
</div>
<h2>By context type</h2>
<div class="wrap"><table><tr><th>type</th><th>model correct</th><th>recurrent ratio, median / p99</th><th>attention ratio, median / p99</th><th>recurrent hit@10 vs null</th><th>attention hit@10 vs null</th></tr>{''.join(rows_kind)}</table></div>
<h2>Correct against incorrect, and controls</h2>
<div class="wrap"><table><tr><th>slice</th><th>records</th><th>recurrent ratio median / p99</th><th>attention ratio median / p99</th><th>attention hit@10 vs null</th></tr>
<tr><td>retrieval, model correct</td><td>{rec['retrieval_correct'].get('n',0)} / {att['retrieval_correct'].get('n',0)}</td><td>{fmt(rec['retrieval_correct'].get('ratio_null_percentiles', {}).get('50'))} / {fmt(rec['retrieval_correct'].get('ratio_null_percentiles', {}).get('99'))}</td><td>{fmt(att['retrieval_correct'].get('ratio_null_percentiles', {}).get('50'))} / {fmt(att['retrieval_correct'].get('ratio_null_percentiles', {}).get('99'))}</td><td>{fmt(att['retrieval_correct'].get('hit_at_k_rate'))} vs {fmt(att['retrieval_correct'].get('null_hit_rate_mean'))}</td></tr>
<tr><td>retrieval, model incorrect</td><td>{rec['retrieval_incorrect'].get('n',0)} / {att['retrieval_incorrect'].get('n',0)}</td><td>{fmt(rec['retrieval_incorrect'].get('ratio_null_percentiles', {}).get('50'))} / {fmt(rec['retrieval_incorrect'].get('ratio_null_percentiles', {}).get('99'))}</td><td>{fmt(att['retrieval_incorrect'].get('ratio_null_percentiles', {}).get('50'))} / {fmt(att['retrieval_incorrect'].get('ratio_null_percentiles', {}).get('99'))}</td><td>{fmt(att['retrieval_incorrect'].get('hit_at_k_rate'))} vs {fmt(att['retrieval_incorrect'].get('null_hit_rate_mean'))}</td></tr>
<tr><td>controls, placebo span</td><td>{rec['control'].get('n',0)} / {att['control'].get('n',0)}</td><td>{fmt(rec['control'].get('ratio_null_percentiles', {}).get('50'))} / {fmt(rec['control'].get('ratio_null_percentiles', {}).get('99'))}</td><td>{fmt(att['control'].get('ratio_null_percentiles', {}).get('50'))} / {fmt(att['control'].get('ratio_null_percentiles', {}).get('99'))}</td><td>{fmt(att['control'].get('hit_at_k_rate'))} vs {fmt(att['control'].get('null_hit_rate_mean'))}</td></tr>
</table></div>
<h2>The strongest heads</h2>
<div class="charts">
<div class="chart"><h3>Attention</h3><div class="wrap"><table><tr><th>head</th><th>type</th><th>span share</th><th>null share</th><th>ratio</th><th>hit@10</th><th>correct</th></tr>{head_rows(top_att, ATT_LAYER)}</table></div></div>
<div class="chart"><h3>Recurrent</h3><div class="wrap"><table><tr><th>head</th><th>type</th><th>span share</th><th>null share</th><th>ratio</th><th>hit@10</th><th>correct</th></tr>{head_rows(top_rec, REC_LAYER)}</table></div></div>
</div>
<h2>Method</h2>
<p>Eight seeded contexts of four kinds (sixteen key-value needles, reference-back prose with a two-hop form, code constants, in-context learning of a symbol-to-digit mapping), facts early and questions at the end with at least 300 tokens between, six retrieval questions and three no-lookup controls each, every question its own forward through the 4-bit checkpoint. Recurrent read weights are |alpha| by the backward scan at the read position for heads whose gate constant on the context exceeds {S['parameters']['TAU_MIN']:.0f} tokens; attention weights are the softmax recomputed from the post-rotary queries and keys and checked by reconstructing each block's own output (gate {S['attention_reconstruction']['gate_relative']} of the output scale, proven to fail on a wrong head pairing). Both channels go through one statistic: the share of far-source (gap at least {S['parameters']['GAP']}) mass on the answer's token span, against the uniform share and against {S['parameters']['NDRAW']} random unit queries at the real norm. Controls use a placebo span from the context's own facts. The model's answers are scored under teacher forcing over the whole answer.</p>
</main>
<script>
const SER = {json.dumps(series)}; const QC = {json.dumps(qc)};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function layers() {{
  const W=520, H=330, m={{l:52, r:14, t:12, b:40}};
  const all = [...SER.recurrent, ...SER.attention]; const ymax = Math.max(2, ...all.map(s => s[3])) * 1.08;
  const X = L => m.l + ((L - 1) / 32) * (W - m.l - m.r), Y = v => H - m.b - (Math.log10(Math.max(v, 0.05)) - Math.log10(0.05)) / (Math.log10(ymax) - Math.log10(0.05)) * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="mass on the answer span by layer">`;
  for (const v of [0.1, 0.3, 1, 3, 10, 30, 100]) if (v < ymax) s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(v)}}" y2="${{Y(v)}}" stroke="${{v===1?css('--null'):css('--grid')}}" stroke-width="${{v===1?1.5:1}}"/><text x="${{m.l-8}}" y="${{Y(v)+4}}" text-anchor="end" font-size="11" fill="${{css('--muted')}}">${{v}}</text>`;
  for (const L of [4,8,12,16,20,24,28,32]) s += `<text x="${{X(L)}}" y="${{H-m.b+16}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{L}}</text>`;
  s += `<text x="${{(m.l+W-m.r)/2}}" y="${{H-6}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">layer written by the block</text>`;
  s += `<text transform="rotate(-90)" x="${{-(H)/2}}" y="14" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">span mass / null (log)</text>`;
  for (const [ch, c] of [["recurrent","--rec"],["attention","--att"]]) {{
    const pts = SER[ch];
    s += `<path d="${{pts.map((p,i)=>`${{i?'L':'M'}}${{X(p[0])}} ${{Y(p[3])}}`).join(' ')}}" fill="none" stroke="${{css(c)}}" stroke-width="1" opacity=".7"/>`;
    s += `<path d="${{pts.map((p,i)=>`${{i?'L':'M'}}${{X(p[0])}} ${{Y(p[1])}}`).join(' ')}}" fill="none" stroke="${{css(c)}}" stroke-width="2.5"/>`;
    for (const p of pts) s += `<circle cx="${{X(p[0])}}" cy="${{Y(p[1])}}" r="3.5" fill="${{css(c)}}"/>`;
  }}
  document.getElementById('layers').innerHTML = s + '</svg>';
}}
function cosine() {{
  const W=520, H=330, m={{l:52, r:14, t:12, b:60}};
  const cats = [["within_context_retrieval_median","two retrieval questions"],["within_context_control_median","two control questions"],["across_context_median","across contexts"]];
  const X = (i, k) => m.l + (i + 0.2 + k * 0.3) * ((W - m.l - m.r) / cats.length), Y = v => H - m.b - v * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="query cosine">`;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(v)}}" y2="${{Y(v)}}" stroke="${{css('--grid')}}"/><text x="${{m.l-8}}" y="${{Y(v)+4}}" text-anchor="end" font-size="11" fill="${{css('--muted')}}">${{v}}</text>`;
  s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(0.83)}}" y2="${{Y(0.83)}}" stroke="${{css('--null')}}" stroke-dasharray="6 4"/><text x="${{W-m.r-4}}" y="${{Y(0.83)-5}}" text-anchor="end" font-size="11" fill="${{css('--null')}}">ledger run 0.83</text>`;
  cats.forEach(([k, label], i) => {{
    s += `<text x="${{m.l + (i + 0.5) * ((W - m.l - m.r) / cats.length)}}" y="${{H-m.b+18}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{label}}</text>`;
    [["recurrent","--rec"],["attention","--att"]].forEach(([ch, c], j) => {{ const v = QC[ch][k]; if (v == null) return;
      const bw = (W - m.l - m.r) / cats.length * 0.28; s += `<rect x="${{X(i, j) - bw/2}}" y="${{Y(v)}}" width="${{bw}}" height="${{Y(0) - Y(v)}}" fill="${{css(c)}}"/><text x="${{X(i, j)}}" y="${{Y(v)-5}}" text-anchor="middle" font-size="11" fill="${{css('--ink')}}">${{v.toFixed(2)}}</text>`; }});
  }});
  s += `<text transform="rotate(-90)" x="${{-(H)/2}}" y="14" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">median query cosine</text>`;
  document.getElementById('cos').innerHTML = s + '</svg>';
}}
layers(); cosine();
</script>
"""
OUT.write_text(html, encoding="utf-8")
print("wrote", OUT)
