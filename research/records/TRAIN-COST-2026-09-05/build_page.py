"""Build the Training Row Cost page from rows.jsonl: fits, ceilings, decomposition, charts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.jsonl"
OUT = HERE / "training-row-cost.html"
WS = 17.76
CAP = 0.85
GIB = 1024**3
VOCAB, HIDDEN, LAYERS, HEADS = 248320, 2560, 32, 16
FLOOR_SLOPE = (9.90 - 5.36) / (2874 - 997)            # preflight floor, GiB per token
CHUNKWISE_INTERCEPT = 10.12 - FLOOR_SLOPE * 2874        # preflight's single chunkwise point


def load():
    rows = []
    for name in ("rows-probe1.jsonl", "rows-probe2.jsonl", "rows-probe3.jsonl", "rows-probe4.jsonl", "rows-probe5.jsonl"):
        f = HERE / name
        if f.exists():
            rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    ok = {v: sorted([r for r in rows if r.get("ok") and r["variant"] == v], key=lambda r: r["tokens"]) for v in "ABCDEFGJKLMN"}
    stops = {r["variant"]: r for r in rows if not r.get("ok")}
    return ok, stops


def fit(points):
    """Least squares peak = a + b * tokens over rows >= 1024."""
    pts = [(r["tokens"], r["peak_gib"]) for r in points if r["tokens"] >= 1024]
    if len(pts) < 2:
        pts = [(r["tokens"], r["peak_gib"]) for r in points]
    if len(pts) < 2:
        return None
    n = len(pts); sx = sum(t for t, _ in pts); sy = sum(p for _, p in pts)
    sxx = sum(t * t for t, _ in pts); sxy = sum(t * p for t, p in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx); a = (sy - b * sx) / n
    return a, b


def main():
    ok, stops = load()
    fits = {v: fit(ok[v]) for v in "ABCDEFGJKLMN"}
    summary = {}
    for v in "ABCDEFGJKLMN":
        f = fits[v]
        if f:
            a, b = f
            summary[v] = {
                "slope_mib_per_token": b * 1024,
                "ceiling_tokens_ws": (WS - a) / b,
                "ceiling_tokens_cap": (CAP * WS - a) / b,
                "intercept_gib": a,
                "max_measured_tokens": ok[v][-1]["tokens"],
                "max_measured_peak": ok[v][-1]["peak_gib"],
            }
    vocab_slope_mib = VOCAB * 8 / 1024**2   # bf16 logits + f32 log-softmax + bf16 gradient
    if "A" in summary and "B" in summary:
        summary["vocab_measured_mib"] = summary["A"]["slope_mib_per_token"] - summary["B"]["slope_mib_per_token"]
    summary["vocab_analytic_mib"] = vocab_slope_mib
    summary["preflight_floor_mib"] = FLOOR_SLOPE * 1024
    data = {"rows": ok, "stops": stops, "fits": {v: (list(f) if f else None) for v, f in fits.items()},
            "summary": summary, "ws": WS, "cap": CAP, "floor_slope": FLOOR_SLOPE, "chunkwise_intercept": CHUNKWISE_INTERCEPT}
    (HERE / "summary.json").write_text(json.dumps(data, indent=2))

    def fmt(x, d=1):
        return "n/a" if x is None else f"{x:,.{d}f}"

    A = summary.get("A", {}); B = summary.get("B", {}); C = summary.get("C", {}); Dv = summary.get("D", {})
    E = summary.get("E", {}); F = summary.get("F", {}); J = summary.get("J", {}); K = summary.get("K", {}); M = summary.get("M", {}); N = summary.get("N", {})
    have_b = bool(B)
    lever_line = (
        f"Removing the full logits cuts the slope from {fmt(A.get('slope_mib_per_token'),2)} to {fmt(B.get('slope_mib_per_token'),2)} MiB per token and moves the working-set ceiling from about {fmt(A.get('ceiling_tokens_ws'),0)} to about {fmt(B.get('ceiling_tokens_ws'),0)} tokens."
        if have_b else "Variant B is still running; the comparison fills in when it lands."
    )
    table_rows = []
    for v in "ABCDEFGJKLMN":
        for r in ok[v]:
            table_rows.append(f"<tr><td>{v}</td><td>{r['tokens']:,}</td><td>{r['peak_gib']:.2f}</td><td>{r['peak_ws_share']:.2f}</td><td>{r['step_s']:.1f}</td><td>{r['step_tokens_per_s']:.0f}</td><td>{r['loss_step2']:.2f}</td></tr>")
        s = stops.get(v)
        if s:
            reason = s.get("skipped") or s.get("error") or s.get("killed") or "stopped"
            proj = s.get("projected_peak_gib")
            table_rows.append(f"<tr class='stop'><td>{v}</td><td>{s['tokens']:,}</td><td colspan='5'>not run: {reason}{'' if proj is None else f' (projected {proj:.1f} GiB)'}</td></tr>")

    html = f"""<title>Training Row Cost</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:#f7f5f0; --panel:#ffffff; --ink:#1d2229; --muted:#5b6470; --rule:#d9d4ca;
  --a:#0f6e73; --b:#c2601a; --c:#5b4b9e; --d:#7a8a1f; --ws:#9b2c2c; --pre:#7a7f87; --grid:#e6e1d7; --accent-soft:#e2f0ef;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --ground:#14171b; --panel:#1c2026; --ink:#e8e6e1; --muted:#a2a8b1; --rule:#343a42;
  --a:#5fc3c8; --b:#f0954f; --c:#b3a3f0; --d:#c6d65a; --ws:#f28b8b; --pre:#9aa0a8; --grid:#2a3038; --accent-soft:#1f3436;
}} }}
:root[data-theme="dark"] {{
  --ground:#14171b; --panel:#1c2026; --ink:#e8e6e1; --muted:#a2a8b1; --rule:#343a42;
  --a:#5fc3c8; --b:#f0954f; --c:#b3a3f0; --d:#c6d65a; --ws:#f28b8b; --pre:#9aa0a8; --grid:#2a3038; --accent-soft:#1f3436;
}}
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
.charts {{ display:grid; grid-template-columns:1fr; gap:18px; }}
@media (min-width: 880px) {{ .charts {{ grid-template-columns:1fr 1fr; }} }}
.chart {{ background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:12px 12px 6px; }}
.chart h3 {{ margin:0 0 4px; font-size:1rem; font-weight:600; }}
.chart .sub {{ font-size:.85rem; color:var(--muted); margin:0 0 6px; }}
svg {{ width:100%; height:auto; display:block; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:.85rem; color:var(--muted); margin:6px 4px 4px; }}
.sw {{ display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:6px; border-radius:2px; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:.92rem; }}
th, td {{ text-align:right; padding:6px 10px; border-bottom:1px solid var(--rule); }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
th {{ font-weight:600; color:var(--muted); font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; }}
tr.stop td {{ color:var(--muted); font-style:italic; }}
.wrap {{ overflow-x:auto; }}
p {{ max-width:70ch; }}
.callout {{ background:var(--accent-soft); border-left:4px solid var(--a); padding:12px 16px; border-radius:4px; max-width:70ch; }}
code {{ font-family:"IBM Plex Mono", monospace; font-size:.9em; }}
</style>
<main>
<div class="eyebrow">Qwen3.5-4B · 4-bit · one training step per row length · 2026-09-05</div>
<h1>Training Row Cost</h1>
<p class="lede">How much memory and time one training step of the v2b recipe costs as the row gets longer, measured on the trainer's own path, and what changes when the loss stops holding every vocabulary logit at once.</p>

<div class="tiles">
  <div class="tile"><div class="n">{fmt(A.get('slope_mib_per_token'),2)}</div><div class="l">MiB per token of row, trainer as it stands (variant A)</div></div>
  <div class="tile"><div class="n">{fmt(B.get('slope_mib_per_token'),2)}</div><div class="l">MiB per token with chunked cross-entropy (variant B)</div></div>
  <div class="tile"><div class="n">{fmt(A.get('ceiling_tokens_ws'),0)}</div><div class="l">row ceiling at the 17.76 GiB working set, variant A</div></div>
  <div class="tile"><div class="n">{fmt(B.get('ceiling_tokens_ws'),0)}</div><div class="l">row ceiling at the working set, variant B</div></div>
  <div class="tile"><div class="n">{fmt(C.get('slope_mib_per_token'),2)}</div><div class="l">MiB per token with gradient checkpointing off (variant C)</div></div>
  <div class="tile"><div class="n">{fmt(Dv.get('slope_mib_per_token'),2)}</div><div class="l">MiB per token with the recurrence's backward removed (variant D)</div></div>
</div>

<div class="charts">
  <div class="chart"><h3>Peak memory per step</h3><p class="sub">GiB against row length. Dashed: the Metal working set and the 85 percent launch cap. Dotted: preflight's chunkwise envelope, a single-point fit.</p><div id="mem"></div>
    <div class="legend"><span><i class="sw" style="background:var(--a)"></i>A · trainer as it stands</span><span><i class="sw" style="background:var(--b)"></i>B · chunked cross-entropy</span><span><i class="sw" style="background:var(--c)"></i>J · forward only, no gradient</span><span><i class="sw" style="background:var(--d)"></i>M · adapters on the top 8 layers</span><span><i class="sw" style="background:var(--ws)"></i>working set</span><span><i class="sw" style="background:var(--pre)"></i>preflight envelope</span></div></div>
  <div class="chart"><h3>Step throughput</h3><p class="sub">Tokens per second for one optimiser step, the second of two at each length so kernel compilation is excluded.</p><div id="tps"></div>
    <div class="legend"><span><i class="sw" style="background:var(--a)"></i>A · all 32 layers</span><span><i class="sw" style="background:var(--b)"></i>D · recurrence backward removed</span><span><i class="sw" style="background:var(--c)"></i>N · top 16 layers</span><span><i class="sw" style="background:var(--d)"></i>M · top 8 layers</span></div></div>
</div>

<h2>What the slope is made of</h2>
<p>The vocabulary is 248,320 tokens wide, so by arithmetic the bf16 logits, their fp32 log-softmax and the logit gradient would cost {vocab_slope_mib:.2f} MiB per token, three-quarters of the {FLOOR_SLOPE*1024:.2f} MiB per token that preflight's calibration measured for the no-recurrence floor. That arithmetic was the hypothesis behind variant B, and the measurement refuted it. Four further interventions then each removed one candidate, and each moved the slope by a tenth or less:</p>
<div class="wrap"><table>
<tr><th>variant</th><th>what changed</th><th>slope, MiB per token</th><th>against A</th><th>step at 2,688 tokens</th></tr>
<tr><td>A</td><td>the trainer as it stands</td><td>{fmt(A.get('slope_mib_per_token'),2)}</td><td></td><td>21.4 s</td></tr>
<tr><td>B</td><td>full logits never resident</td><td>{fmt(B.get('slope_mib_per_token'),2)}</td><td>{fmt(A.get('slope_mib_per_token',0)-B.get('slope_mib_per_token',0),2)}</td><td>21.0 s</td></tr>
<tr><td>C</td><td>gradient checkpointing off</td><td>{fmt(C.get('slope_mib_per_token'),2)}</td><td>{fmt(A.get('slope_mib_per_token',0)-C.get('slope_mib_per_token',0),2)}</td><td>faster by a quarter</td></tr>
<tr><td>D</td><td>recurrence's backward removed</td><td>{fmt(Dv.get('slope_mib_per_token'),2)}</td><td>{fmt(A.get('slope_mib_per_token',0)-Dv.get('slope_mib_per_token',0),2)}</td><td>15.8 s</td></tr>
<tr><td>E</td><td>no LoRA on the MLP projections</td><td>{fmt(E.get('slope_mib_per_token'),2)}</td><td>{fmt(A.get('slope_mib_per_token',0)-E.get('slope_mib_per_token',0),2)}</td><td>20.9 s</td></tr>
<tr><td>F</td><td>adapters in bfloat16</td><td>{fmt(F.get('slope_mib_per_token'),2)}</td><td>{fmt(A.get('slope_mib_per_token',0)-F.get('slope_mib_per_token',0),2)}</td><td>20.5 s</td></tr>
<tr><td>G</td><td>staged gradient evaluation, last layer first</td><td>higher than A at every length</td><td></td><td>order matters</td></tr>
<tr><td>J</td><td>forward only, training mode, no gradient</td><td>{fmt(J.get('slope_mib_per_token'),2)}</td><td></td><td>4.1 s at 2,048</td></tr>
<tr><td>K</td><td>forward only, inference mode, no gradient</td><td>{fmt(K.get('slope_mib_per_token'),2)}</td><td></td><td>3.9 s at 2,048</td></tr>
<tr><td>L</td><td>sixteen of thirty-two layers checkpointed</td><td>about 7.2 (linearity check only)</td><td></td><td></td></tr>
<tr><td>N</td><td>adapters on the top 16 layers only</td><td>{fmt(N.get('slope_mib_per_token'),2)}</td><td></td><td>9.9 s at 2,048</td></tr>
<tr><td>M</td><td>adapters on the top 8 layers only</td><td>{fmt(M.get('slope_mib_per_token'),2)}</td><td></td><td>7.4 s at 2,048</td></tr>
</table></div>
<div class="callout">Checkpointing works: switched off, the step costs 13.9 MiB per token, so the hook discards about 11.5 of it. The recurrence's backward is a quarter of the step time and none of the memory. The logits, the MLP adapters and the adapter dtype are each worth a tenth. Then two cuts locate what is left. The forward pass alone, with no gradient in the graph, carries about 1 MiB per token in either mode, so roughly half the slope exists before any backward begins; and the backward's extra 1 MiB per token is the same whether eight, sixteen or thirty-two layers are in the chain, so it does not come from the layers. What a chunked loss saves is length-dependent, 1.3 GiB at 2,048 tokens and 0.7 at 4,096, which says the peak moment moves with row length: the loss at short rows, one attention block's quadratic scores under recompute at long ones. Evaluation order matters too: forcing the last layer's gradients first raises the peak by half. The engineering levers that survive: a loss that never holds the full logits in forward or backward, and, for speed rather than memory, adapters on fewer layers, since the top eight train at 276 tokens a second against 126 for all thirty-two at nearly the same memory.</div>

<h2>Every row</h2>
<div class="wrap"><table>
<tr><th>variant</th><th>tokens</th><th>peak GiB</th><th>share of working set</th><th>step s</th><th>tokens per s</th><th>loss after step</th></tr>
{''.join(table_rows)}
</table></div>

<h2>Method</h2>
<p>Each point is its own process: the registry spec resolved through <code>evaluate.load_policy</code>, which takes the model-run lock; LoRA rank 16, scale 32 on the resolved keys over every layer; the chunkwise gated-delta recurrence at chunk 64; gradient checkpointing on; AdamW; <code>mlx_lm</code>'s <code>default_loss</code> on a synthetic row, batch 1. Two steps run at each length and the second is reported. Variant B runs the backbone once and applies the vocabulary projection and cross-entropy per 1,024-position chunk under <code>mx.checkpoint</code>, so the full logits are never resident; it reproduces variant A's loss to three decimals. Lengths stop where the linear projection from the last two points exceeds 85 percent of the working set, and a guard terminates a step on two consecutive critical kernel-pressure samples. Efficiency probes stop at 64k context by the Director's ruling (R48); this one never approached it.</p>
</main>
<script>
const D = {json.dumps(data)};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function chart(id, series, opts) {{
  const W=520, H=330, m={{l:52, r:14, t:12, b:40}};
  const xs = series.flatMap(s => s.pts.map(p => p[0])), ys = series.flatMap(s => s.pts.map(p => p[1])).concat(opts.hlines.map(h => h.y));
  const xmax = opts.xmax || Math.max(...xs) * 1.08, ymax = (opts.ymax || Math.max(...ys)) * 1.08;
  const X = v => m.l + (v / xmax) * (W - m.l - m.r), Y = v => H - m.b - (v / ymax) * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{opts.label}}">`;
  const yt = opts.yticks; for (const v of yt) s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(v)}}" y2="${{Y(v)}}" stroke="${{css('--grid')}}"/><text x="${{m.l-8}}" y="${{Y(v)+4}}" text-anchor="end" font-size="11" fill="${{css('--muted')}}">${{v}}</text>`;
  for (const v of opts.xticks) s += `<text x="${{X(v)}}" y="${{H-m.b+16}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{v>=1000? (v/1024).toFixed(0)+'k': v}}</text>`;
  s += `<text x="${{(m.l+W-m.r)/2}}" y="${{H-6}}" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">row length, tokens</text>`;
  s += `<text transform="rotate(-90)" x="${{-(H)/2}}" y="14" text-anchor="middle" font-size="11" fill="${{css('--muted')}}">${{opts.ylabel}}</text>`;
  for (const h of opts.hlines) s += `<line x1="${{m.l}}" x2="${{W-m.r}}" y1="${{Y(h.y)}}" y2="${{Y(h.y)}}" stroke="${{css(h.c)}}" stroke-dasharray="${{h.d}}" stroke-width="1.5"/><text x="${{W-m.r-4}}" y="${{Y(h.y)-5}}" text-anchor="end" font-size="11" fill="${{css(h.c)}}">${{h.t}}</text>`;
  for (const l of (opts.lines||[])) {{ const x0=0, x1=xmax; s += `<line x1="${{X(x0)}}" y1="${{Y(l.a)}}" x2="${{X(x1)}}" y2="${{Y(l.a+l.b*x1)}}" stroke="${{css(l.c)}}" stroke-dasharray="2 4" stroke-width="1.5"/>`; }}
  for (const se of series) {{ if (!se.pts.length) continue; const d = se.pts.map((p,i)=>`${{i?'L':'M'}}${{X(p[0])}} ${{Y(p[1])}}`).join(' ');
    s += `<path d="${{d}}" fill="none" stroke="${{css(se.c)}}" stroke-width="2.5"/>`;
    for (const p of se.pts) s += `<circle cx="${{X(p[0])}}" cy="${{Y(p[1])}}" r="4" fill="${{css(se.c)}}"/>`; }}
  document.getElementById(id).innerHTML = s + '</svg>';
}}
const A = D.rows.A, B = D.rows.B, C = D.rows.C, Dd = D.rows.D, J = D.rows.J, K = D.rows.K, M = D.rows.M, N = D.rows.N;
chart('mem', [{{pts:A.map(r=>[r.tokens,r.peak_gib]), c:'--a'}}, {{pts:B.map(r=>[r.tokens,r.peak_gib]), c:'--b'}}, {{pts:J.map(r=>[r.tokens,r.peak_gib]), c:'--c'}}, {{pts:M.map(r=>[r.tokens,r.peak_gib]), c:'--d'}}],
  {{label:'peak memory per step', ylabel:'peak GiB', ymax: 20, xmax: 9000, yticks:[0,5,10,15,20], xticks:[0,2048,4096,6144,8192],
   hlines:[{{y:D.ws, c:'--ws', d:'6 4', t:'working set 17.76 GiB'}}, {{y:D.ws*D.cap, c:'--ws', d:'2 3', t:'85% launch cap'}}],
   lines:[{{a:D.chunkwise_intercept, b:D.floor_slope, c:'--pre'}}]}});
chart('tps', [{{pts:A.map(r=>[r.tokens,r.step_tokens_per_s]), c:'--a'}}, {{pts:Dd.map(r=>[r.tokens,r.step_tokens_per_s]), c:'--b'}}, {{pts:N.map(r=>[r.tokens,r.step_tokens_per_s]), c:'--c'}}, {{pts:M.map(r=>[r.tokens,r.step_tokens_per_s]), c:'--d'}}],
  {{label:'step throughput', ylabel:'tokens per second', ymax: 320, yticks:[0,100,200,300], xticks:[0,2048,4096,6144,8192], hlines:[]}});
// xmax: 9000, yticks:[0,50,100,150], xticks:[0,2048,4096,6144,8192], hlines:[]}});
</script>
"""
    OUT.write_text(html, encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
