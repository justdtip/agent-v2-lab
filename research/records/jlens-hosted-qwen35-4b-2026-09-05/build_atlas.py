import json
data = open("dashboard_data.json").read()
html = r'''<title>Qwen3.5-4B Workspace Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#F5F7F8; --bg-2:#EDF1F3; --ink:#16202A; --ink-2:#3C4A56; --muted:#5C6B78; --line:#D4DBE0;
  --lens:#0E8C8F; --lens-soft:rgba(14,140,143,.10); --logit:#C77A17; --null:#9AA6AE; --attn:#6B4FBB;
  --base:#16202A; --good:#2E7D4F; --warn:#B4541A; --card:#FFFFFF;
  --serif:"Newsreader",Georgia,"Times New Roman",serif; --sans:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif; --mono:"IBM Plex Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0F1418; --bg-2:#151C22; --ink:#E6ECF0; --ink-2:#C3CDD4; --muted:#93A3AF; --line:#2A353D;
    --lens:#3FC1C4; --lens-soft:rgba(63,193,196,.12); --logit:#E9A94A; --null:#6F7E89; --attn:#A08BE0;
    --base:#E6ECF0; --good:#5FBF86; --warn:#E58A52; --card:#151C22;
  }
}
:root[data-theme="dark"]{
  --bg:#0F1418; --bg-2:#151C22; --ink:#E6ECF0; --ink-2:#C3CDD4; --muted:#93A3AF; --line:#2A353D;
  --lens:#3FC1C4; --lens-soft:rgba(63,193,196,.12); --logit:#E9A94A; --null:#6F7E89; --attn:#A08BE0;
  --base:#E6ECF0; --good:#5FBF86; --warn:#E58A52; --card:#151C22;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.55}
a{color:var(--lens)}
.shell{display:grid;grid-template-columns:200px minmax(0,1fr);gap:40px;max-width:1240px;margin:0 auto;padding:32px 28px 80px}
@media (max-width:860px){.shell{grid-template-columns:1fr;gap:20px}.rail{position:static}}
.rail{position:sticky;top:24px;align-self:start;font-size:13px}
.rail .name{font-family:var(--serif);font-size:20px;font-weight:600;line-height:1.15;margin:0 0 4px}
.rail .date{color:var(--muted);font-family:var(--mono);font-size:12px;margin-bottom:18px}
.rail nav a{display:block;color:var(--ink-2);text-decoration:none;padding:5px 0;border-top:1px solid var(--line)}
.rail nav a:hover,.rail nav a:focus-visible{color:var(--lens);outline:none}
.rail .legend{margin-top:22px;padding-top:12px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
.rail .legend div{display:flex;align-items:center;gap:8px;margin:4px 0}
.sw{width:18px;height:0;border-top:3px solid var(--lens);display:inline-block}
.sw.logit{border-top-color:var(--logit)} .sw.null{border-top:2px dashed var(--null)} .sw.base{border-top:3px dotted var(--ink)} .sw.attn{border-top-color:var(--attn);width:10px}
main{min-width:0}
h1{font-family:var(--serif);font-weight:400;font-size:40px;line-height:1.1;margin:0 0 10px;text-wrap:balance;letter-spacing:-.01em}
h1 em{font-style:italic;color:var(--lens)}
.lede{max-width:64ch;color:var(--ink-2);font-size:16px;margin:0 0 14px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 36px}
.chip{font-family:var(--mono);font-size:12px;padding:4px 9px;border:1px solid var(--line);border-radius:3px;color:var(--ink-2);background:var(--bg-2)}
section{margin:0 0 56px;scroll-margin-top:24px}
h2{font-family:var(--serif);font-weight:600;font-size:26px;margin:0 0 6px;text-wrap:balance}
.sub{color:var(--muted);max-width:70ch;margin:0 0 18px}
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:0 0 8px}
.finding{padding:14px 16px;border:1px solid var(--line);background:var(--card);border-radius:4px;display:flex;flex-direction:column;gap:6px}
.finding .k{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.finding .v{font-family:var(--serif);font-size:30px;line-height:1;font-variant-numeric:tabular-nums}
.finding .v small{font-size:15px;color:var(--muted);font-family:var(--sans)}
.finding .d{color:var(--ink-2);font-size:13px;line-height:1.45}
.toggle{display:inline-flex;border:1px solid var(--line);border-radius:4px;overflow:hidden;margin:0 0 14px;font-size:13px}
.toggle button{background:var(--bg-2);color:var(--ink-2);border:0;padding:6px 12px;cursor:pointer;font:inherit}
.toggle button[aria-pressed="true"]{background:var(--lens);color:#fff}
.toggle button:focus-visible{outline:2px solid var(--attn);outline-offset:-2px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}
@media (max-width:520px){.charts{grid-template-columns:1fr}}
.chart{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:12px 12px 6px}
.chart h3{font-family:var(--sans);font-weight:600;font-size:13px;margin:0 0 2px}
.chart p{margin:0 0 6px;color:var(--muted);font-size:12px;line-height:1.4}
.chart svg{width:100%;height:auto;display:block;font-family:var(--mono);font-size:10px}
.chart .axis text{fill:var(--muted)} .chart .axis line,.chart .axis path{stroke:var(--line)}
.chart .band{fill:var(--lens-soft)} .chart .attn-tick{stroke:var(--attn);stroke-width:2}
.chart .lens{stroke:var(--lens);fill:none;stroke-width:2} .chart .logit{stroke:var(--logit);fill:none;stroke-width:2}
.chart .nul{stroke:var(--null);fill:none;stroke-width:1.5;stroke-dasharray:4 4} .chart .basel{stroke-dasharray:2 4}
.chart .dot{fill:var(--card);stroke-width:1.5}
.chart .ylab{fill:var(--muted)}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font-family:var(--mono);font-size:11px;padding:6px 8px;border-radius:3px;opacity:0;transition:opacity .1s;z-index:9;max-width:320px;white-space:pre-line}
.heat{display:grid;grid-template-columns:minmax(0,1fr) 220px;gap:18px;align-items:start}
@media (max-width:760px){.heat{grid-template-columns:1fr}}
.heat canvas{width:100%;height:auto;display:block;border:1px solid var(--line);background:var(--card);image-rendering:pixelated}
.heat .read{font-family:var(--mono);font-size:12px;color:var(--ink-2);min-height:80px}
.scale{height:10px;background:linear-gradient(90deg,var(--bg-2),var(--lens));border:1px solid var(--line);margin:8px 0 4px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:4px;background:var(--card)}
th{font-family:var(--mono);font-weight:500;font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:8px 12px;border-bottom:1px solid var(--line);font-family:var(--mono);white-space:nowrap}
tr:last-child td{border-bottom:0}
td.r{color:var(--ink);font-family:var(--sans);font-weight:500}
td.sig{color:var(--lens);font-weight:500} td.strong{color:var(--good);font-weight:500}
.hosted td:first-child{border-left:3px solid var(--lens)} .logitrow td:first-child{border-left:3px solid var(--logit)}
.grid-tokens{display:grid;grid-template-columns:70px minmax(0,1fr);gap:6px 10px;align-items:center;font-size:12px}
.grid-tokens .L{font-family:var(--mono);color:var(--muted);text-align:right}
.grid-tokens .L.attn{color:var(--attn)}
.toks{display:flex;flex-wrap:wrap;gap:4px}
.tok{font-family:var(--mono);font-size:12px;padding:2px 6px;border:1px solid var(--line);border-radius:3px;background:var(--bg-2);color:var(--ink)}
.tok.hit{background:var(--lens);color:#fff;border-color:var(--lens)}
.tok .sp{color:var(--muted)}
select{font:inherit;font-size:13px;padding:5px 8px;border:1px solid var(--line);border-radius:4px;background:var(--card);color:var(--ink);margin:0 0 12px}
.scrub{display:flex;align-items:center;gap:10px;margin:6px 0 10px;font-size:13px;color:var(--muted)}
.scrub input{flex:1;accent-color:var(--lens)}
.scrub b{font-family:var(--mono);color:var(--ink);min-width:36px}
.dots{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.note{border-left:3px solid var(--lens);padding:2px 0 2px 14px;color:var(--ink-2);max-width:68ch;margin:14px 0 0}
.method dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;font-size:13px;max-width:80ch}
.method dt{font-family:var(--mono);color:var(--muted)} .method dd{margin:0}
footer{color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:12px;margin-top:40px}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style>

<div class="shell">
<aside class="rail">
  <p class="name">Qwen3.5-4B Workspace Atlas</p>
  <div class="date">2026-09-05 · hosted J-lens n1000</div>
  <nav>
    <a href="#glance">At a glance</a>
    <a href="#band">Band signatures</a>
    <a href="#cka">Layer alignment</a>
    <a href="#cases">The 42 probe points</a>
    <a href="#decision">Decision-point workspace</a>
    <a href="#sanity">Rank trajectories</a>
    <a href="#contrast">Base vs post-trained</a>
    <a href="#transport">Recurrent memory by head</a>
    <a href="#method">Method and files</a>
  </nav>
  <div class="legend">
    <div><span class="sw"></span> hosted J-lens</div>
    <div><span class="sw logit"></span> logit lens (J = I)</div>
    <div><span class="sw null"></span> shuffled null</div>
    <div><span class="sw base"></span> Base checkpoint</div>
    <div><span class="sw attn"></span> attention layer</div>
  </div>
</aside>
<main>
  <h1>What the workspace of Qwen3.5-4B holds, <em>layer by layer</em></h1>
  <p class="lede">Every reading here comes from the reference-recipe Jacobian lens fitted on Qwen/Qwen3.5-4B, applied to the cached 4-bit checkpoint and to a locally quantised Base, through the repository's own capture path. Layer L is the residual after block L−1; layer 32 is the pre-norm final residual. Hover any chart for the numbers.</p>
  <div class="chips"><span class="chip">post-trained: mlx-community/Qwen3.5-4B-MLX-4bit</span><span class="chip">Base: Qwen3.5-4B-Base → 4-bit affine g64</span><span class="chip">lens: neuronpedia/jacobian-lens · qwen3.5-4b · n1000</span><span class="chip">band by persistence: L13–29</span></div>

  <section id="glance">
    <h2>At a glance</h2>
    <p class="sub">Four numbers that carry the day's conclusions. Each is expanded in its own section below.</p>
    <div class="findings">
      <div class="finding"><span class="k">workspace band</span><span class="v">L13–29</span><span class="d">Top-1 readouts persist across positions at ten times the shuffled null; the first twelve layers sit at the null and the last three are motor.</span></div>
      <div class="finding"><span class="k">recovery lead</span><span class="v">13 <small>layers</small></span><span class="d">“Paris” enters the hosted readout by layer 9–12; the plain unembedding needs layer 24.</span></div>
      <div class="finding"><span class="k">EXP-001 copy signal, L20</span><span class="v">11/42</span><span class="d">Hosted lens equals the logit lens (paired p 0.004); the hidden suffix never moves with context. World A stands.</span></div>
      <div class="finding"><span class="k">reactions on user tokens</span><span class="v">16% <small>Base</small> · 13% <small>post</small></span><span class="d">The Base already carries the Assistant's reactions; only the response-start position separates the two models.</span></div>
    </div>
  </section>

  <section id="band">
    <h2>Band signatures</h2>
    <p class="sub">The paper's four structural signatures, computed on 16 contexts of about 135 tokens with the first 16 positions skipped. The teal wash marks layers 13–29; violet ticks mark the eight attention layers, every other block being Gated DeltaNet.</p>
    <div class="toggle" role="group" aria-label="model"><button data-m="post" aria-pressed="true">post-trained</button><button data-m="base" aria-pressed="false">Base</button><button data-m="both" aria-pressed="false">both</button></div>
    <div class="charts">
      <div class="chart" id="c-hit"><h3>Next-token top-10 hit</h3><p>Fraction of positions where the model's own next token is in the readout's top 10.</p></div>
      <div class="chart" id="c-auto"><h3>Persistence of the top-1 token</h3><p>Probability the top-1 readout token is the same at the next position, against a within-context shuffle.</p></div>
      <div class="chart" id="c-ent"><h3>Readout entropy and top-1 mass</h3><p>Bits over the 248k vocabulary (solid, left axis) and the top token's probability (thin, right axis).</p></div>
      <div class="chart" id="c-dim"><h3>Effective dimensionality of the J-lens vectors</h3><p>Fraction of residual dimensions carrying 90% of the variance of W<sub>U</sub>J<sub>L</sub>; thin line is the participation ratio.</p></div>
    </div>
    <p class="note">Persistence is the cleanest signature: at the null through layer 12, above it from 13, peaking at 17–20, and back at the null by 31. The geometry fans out later, between 18 and 25. The early top-10 hit that the logit lens lacks is a degenerate low-rank readout of generic tokens, which the entropy and top-1 mass panels make visible.</p>
  </section>

  <section id="cka">
    <h2>Layer alignment</h2>
    <p class="sub">Linear CKA between the Gram matrices of J-lens vectors at each pair of layers, over a 4096-token subsample. Blocks are where the workspace geometry is stable; the paper's three-block picture appears here as a tight early block, a slow middle drift, and a distinct final layer.</p>
    <div class="toggle" role="group" aria-label="model for alignment"><button data-cm="post" aria-pressed="true">post-trained</button><button data-cm="base" aria-pressed="false">Base</button></div>
    <div class="heat"><canvas id="cka-canvas" width="640" height="640" aria-label="CKA heatmap"></canvas><div><div class="read" id="cka-read">Hover a cell.</div><div class="scale"></div><div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;color:var(--muted)"><span>0.2</span><span>1.0</span></div></div></div>
  </section>

  <section id="cases">
    <h2>The 42 probe points</h2>
    <p class="sub">EXP-001's decision positions, regenerated from the recorded seed. Each row asks how often the hidden suffix outranks the previously-seen one in its own context and in another task's context, then the ratified paired test and the decomposition: does each candidate's own probability move with its context?</p>
    <div class="toggle" role="group" aria-label="model for probe points"><button data-tm="post" aria-pressed="true">post-trained</button><button data-tm="base" aria-pressed="false">Base</button></div>
    <div class="tablewrap"><table id="cases-table"></table></div>
    <p class="note">Rows with a teal edge are the hosted lens, amber the logit lens. The hidden suffix's column never reaches significance at any layer; the previously-seen suffix's does at 20, 21 and, after post-training, 27 to 32. Our earlier finite-difference estimate showed nothing at 20 and 21.</p>
  </section>

  <section id="decision">
    <h2>Decision-point workspace</h2>
    <p class="sub">The hosted readout's top tokens at the position where the model must name the hidden file, by layer, for three probe points on the post-trained checkpoint. Digits and number words in every script, and never a filename: the workspace holds the type of the answer, not its value.</p>
    <select id="case-sel" aria-label="probe point"></select>
    <div class="grid-tokens" id="tokgrid"></div>
  </section>

  <section id="sanity">
    <h2>Rank trajectories</h2>
    <p class="sub">Rank of a target token in the readout at the last prompt position, by layer, on a log scale. Drag the scrubber to read the top tokens at any layer.</p>
    <div class="toggle" role="group" aria-label="prompt"><button data-p="france" aria-pressed="true">capital of France</button><button data-p="spider" aria-pressed="false">animal that spins webs</button><button data-p="arith" aria-pressed="false">(1 + 2) × 3 − 2 (Base)</button></div>
    <div class="chart" id="c-rank"><h3 id="rank-title"></h3><p id="rank-sub"></p></div>
    <div class="scrub"><span>layer</span><input type="range" id="scrub" min="1" max="32" value="24" aria-label="layer"><b id="scrub-L">L24</b></div>
    <div class="grid-tokens" id="rank-toks"></div>
  </section>

  <section id="contrast">
    <h2>Base against post-trained</h2>
    <p class="sub">The paper's Section 6 statistic: the best rank any reaction token reaches anywhere in the user's turn over band layers 11–29, per prompt, on a log scale. Filled marks are the post-trained model, hollow the Base; the bar is the median.</p>
    <div class="dots" id="dots"></div>
    <p class="note">On the paper's Claude pair the base model almost never carried these reactions while reading. Here the Base does, on every suite. The one separation is at the response-start position on hazardous prompts: median rank 34 post-trained against 160 Base, with the matched safe controls at 504 and 904.</p>
  </section>

  <section id="transport">
    <h2>Recurrent memory by head</h2>
    <p class="sub">The 24 Gated DeltaNet blocks carry a per-head gate that multiplies the state at every token. Its time constant, −1/E[log g] over a 1,550-token agent transcript, says how many tokens a head remembers on that input; the gate is input-dependent, and at neutral input every head is fast. Read weights were recomputed exactly from the forward pass and reconstruct the blocks' outputs to within a millionth over 10,752 cells.</p>
    <div class="heat"><canvas id="tau-canvas" width="768" height="576" aria-label="gate time constant per block and head"></canvas><div><div class="read" id="tau-read">Hover a cell. Rows are recurrent blocks 0–23 in forward order; columns are the 32 value heads.</div><div class="scale" style="background:linear-gradient(90deg,var(--bg-2),var(--lens),var(--attn))"></div><div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;color:var(--muted)"><span>1 token</span><span>100</span><span>10,000</span></div><p id="tau-summary" style="font-size:13px;color:var(--ink-2);margin-top:12px"></p></div></div>
    <div class="charts" style="margin-top:18px">
      <div class="chart" id="c-mass"><h3>Share of a head's read that comes from each distance</h3><p>Median (solid) and 90th percentile (thin) of the mass share per gap bin, against the uniform null (dashed). Pooled over all blocks and 14 positions.</p></div>
      <div class="chart" id="c-ret"><h3>Retention relative to the nearest four tokens</h3><p>Median (solid) and 90th percentile (thin); the 90th percentile alone would mislead, which is why the joint criterion exists.</p></div>
    </div>
    <p class="note" id="transport-note"></p>
  </section>

  <section id="method" class="method">
    <h2>Method and files</h2>
    <dl>
      <dt>readout</dt><dd>softmax(unembed(final_norm(h @ Jᵀ))) through ArchitectureView; file index l = layer l + 1; layer 32 is the identity.</dd>
      <dt>lens files</dt><dd>neuronpedia/jacobian-lens · qwen3.5-4b/jlens/Salesforce-wikitext · n1000 (1000 prompts) and default (417 prompts); wikitext-103, 128 tokens, final-layer target; loaded with torch.load(weights_only=True).</dd>
      <dt>probe points</dt><dd>make_jspace_tasks("jsweep", 720, 20260902), select_cases(probe_step=3); all 42 triples match the EXP-001 artifact.</dd>
      <dt>corpus</dt><dd>build_corpus(size=16, length=128), first 16 positions skipped, k=10; persistence null = 5 within-context shuffles.</dd>
      <dt>timings</dt><dd>post-trained: cases 154 s, corpus 152 s, sanity 2.3 s; Base: 126 s, 131 s, 1.4 s. Reading cost is one 2560² product plus the unembedding per (layer, position).</dd>
      <dt>artifact</dt><dd>outputs/probes/jlens-hosted-qwen35-4b-2026-09-05/ (scripts, raw JSON, logs, provenance with SHA-256s).</dd>
      <dt>memos</dt><dd>design_specifications/under_review/HOSTED-JLENS-QWEN35-4B-RESULTS-2026-09-05.md · CHIEF-JSPACE-PAPER-READING-2026-09-05.md · pending/03-REVISED-PLAN-2026-09-05.md</dd>
    </dl>
  </section>
  <footer>Chief AI Research Scientist · agent-v2-lab · every number on this page is reproducible from the artifact directory named above.</footer>
</main>
</div>
<div class="tip" id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
const D = JSON.parse(document.getElementById('data').textContent);
const tip = document.getElementById('tip');
function showTip(e, text){ tip.textContent = text; tip.style.opacity = 1; tip.style.left = Math.min(window.innerWidth-330, e.clientX+14)+'px'; tip.style.top = (e.clientY+14)+'px'; }
function hideTip(){ tip.style.opacity = 0; }
const ATTN = [4,8,12,16,20,24,28,32];
const BAND = [13,29];
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent){ const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; }
function fmt(v, d){ return (v===null||v===undefined||isNaN(v)) ? '–' : Number(v).toFixed(d===undefined?2:d); }

// ---------- generic line chart over layers ----------
function lineChart(container, spec){
  const old = container.querySelector('svg'); if (old) old.remove();
  const W = 520, H = 250, m = {l:44, r: spec.right ? 44 : 14, t:12, b:30};
  const svg = el('svg', {viewBox:`0 0 ${W} ${H}`, role:'img', 'aria-label': spec.aria||''}, container);
  const xs = spec.layers; const x = L => m.l + (L-1)/(32-1)*(W-m.l-m.r);
  const logY = spec.log;
  const ymin = spec.ymin!==undefined ? spec.ymin : 0, ymax = spec.ymax;
  const y = v => logY ? m.t + (Math.log10(ymax)-Math.log10(Math.max(v,ymin)))/(Math.log10(ymax)-Math.log10(ymin))*(H-m.t-m.b)
                      : m.t + (ymax - v)/(ymax-ymin)*(H-m.t-m.b);
  const y2 = spec.right ? (v => m.t + (spec.right.max - v)/(spec.right.max-spec.right.min)*(H-m.t-m.b)) : null;
  // band
  el('rect', {x:x(BAND[0]), y:m.t, width:x(BAND[1])-x(BAND[0]), height:H-m.t-m.b, class:'band'}, svg);
  // axes
  const ax = el('g', {class:'axis'}, svg);
  el('line', {x1:m.l, x2:W-m.r, y1:H-m.b, y2:H-m.b}, ax);
  for (const L of [1,4,8,12,16,20,24,28,32]){
    el('line', {x1:x(L), x2:x(L), y1:H-m.b, y2:H-m.b+4}, ax);
    const t = el('text', {x:x(L), y:H-m.b+15, 'text-anchor':'middle'}, ax); t.textContent = L;
  }
  for (const L of ATTN){ el('line', {x1:x(L), x2:x(L), y1:H-m.b-6, y2:H-m.b, class:'attn-tick'}, svg); }
  const ticks = spec.ticks || (logY ? [1,10,100,1000,10000,100000] : [0,0.25,0.5,0.75,1]);
  for (const v of ticks){ if (logY && (v<ymin||v>ymax)) continue; if(!logY && (v<ymin||v>ymax)) continue;
    el('line', {x1:m.l-4, x2:m.l, y1:y(v), y2:y(v)}, ax);
    el('line', {x1:m.l, x2:W-m.r, y1:y(v), y2:y(v), 'stroke-opacity':.5}, ax);
    const t = el('text', {x:m.l-6, y:y(v)+3, 'text-anchor':'end'}, ax); t.textContent = logY ? (v>=1000? (v/1000)+'k' : v) : v; }
  if (spec.right){ for (const v of spec.right.ticks){ const t = el('text', {x:W-m.r+6, y:y2(v)+3, 'text-anchor':'start', class:'ylab'}, ax); t.textContent = v; } }
  const yl = el('text', {x:m.l, y:8, class:'ylab'}, svg); yl.textContent = spec.ylabel || '';
  if (spec.right){ const yr = el('text', {x:W-m.r, y:8, class:'ylab','text-anchor':'end'}, svg); yr.textContent = spec.right.label; }
  // series
  for (const s of spec.series){
    const yy = s.axis==='right' ? y2 : y;
    const pts = xs.map((L,i)=> [x(L), yy(s.values[i])]);
    const d = pts.map((p,i)=> (i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' ');
    const cls = s.cls + (s.base ? ' basel' : '');
    const path = el('path', {d, class:cls}, svg);
    if (s.thin) path.setAttribute('stroke-width','1');
    // hover targets
    xs.forEach((L,i)=>{ const c = el('circle', {cx:x(L), cy:yy(s.values[i]), r:3.2, class:'dot', stroke: s.color}, svg);
      c.addEventListener('mousemove', e=> showTip(e, `${s.name}\nlayer ${L} (${ATTN.includes(L)?'attention':'DeltaNet'})\n${fmt(s.values[i], s.digits)}`));
      c.addEventListener('mouseleave', hideTip); });
  }
}
const cs = getComputedStyle(document.documentElement);
const col = n => cs.getPropertyValue(n).trim();

function drawBand(mode){
  const models = mode==='both' ? ['post','base'] : [mode];
  const S = (name, key, cls, opts) => models.map(mn => Object.assign({name: name + (mn==='base'?' · Base':' · post-trained'), values: D.band[mn][key], cls, base: mn==='base', color: cls==='lens'?col('--lens'):cls==='logit'?col('--logit'):col('--null')}, opts||{}));
  const layers = D.band.post.layers;
  lineChart(document.getElementById('c-hit'), {layers, ymax:1, ylabel:'fraction of positions', aria:'top-10 hit by layer',
    series: [...S('hosted lens top-10 hit','topk_hit','lens'), ...S('logit lens top-10 hit','topk_hit_ll','logit')]});
  lineChart(document.getElementById('c-auto'), {layers, ymax:0.3, ticks:[0,0.1,0.2,0.3], ylabel:'P(same top-1 at t+1)', aria:'persistence by layer',
    series: [...S('hosted persistence','autocorr','lens'), ...S('shuffled null','autocorr_null','nul'), ...S('logit-lens persistence','autocorr_ll','logit')]});
  lineChart(document.getElementById('c-ent'), {layers, ymax:17, ticks:[0,4,8,12,16], ylabel:'entropy, bits', aria:'entropy by layer', right:{min:0,max:1,ticks:[0,0.5,1],label:'top-1 mass'},
    series: [...S('hosted entropy (bits)','entropy','lens'), ...S('logit-lens entropy (bits)','entropy_ll','logit'), ...S('hosted top-1 mass','top1_mass','lens',{axis:'right', thin:true, digits:3})]});
  lineChart(document.getElementById('c-dim'), {layers, ymax:1, ylabel:'fraction of 2560 dims', aria:'dimensionality by layer',
    series: [...S('dims for 90% variance','dims90','lens'), ...S('participation ratio','part_ratio','lens',{thin:true, digits:3})]});
}
drawBand('post');
document.querySelectorAll('[data-m]').forEach(b=> b.addEventListener('click', ()=>{ document.querySelectorAll('[data-m]').forEach(x=>x.setAttribute('aria-pressed', x===b)); drawBand(b.dataset.m); }));

// ---------- CKA heatmap ----------
const cv = document.getElementById('cka-canvas'), ctx = cv.getContext('2d');
function hexToRgb(h){ h=h.replace('#',''); return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)]; }
function drawCKA(mn){
  const M = D.band[mn].cka, Ls = D.band[mn].cka_layers, n = Ls.length, cell = cv.width/n;
  const c0 = hexToRgb(col('--bg-2')), c1 = hexToRgb(col('--lens'));
  for (let i=0;i<n;i++) for (let j=0;j<n;j++){ const v = Math.max(0, Math.min(1, (M[i][j]-0.2)/0.8)); const c = c0.map((a,k)=> Math.round(a+(c1[k]-a)*v)); ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.fillRect(j*cell, i*cell, cell+0.5, cell+0.5); }
  ctx.strokeStyle = col('--attn'); ctx.lineWidth = 2;
  for (const L of ATTN){ const k = Ls.indexOf(L); if (k<0) continue; ctx.beginPath(); ctx.moveTo(k*cell, cv.height-6); ctx.lineTo(k*cell+cell, cv.height-6); ctx.stroke(); }
  cv.onmousemove = e => { const r = cv.getBoundingClientRect(); const j = Math.floor((e.clientX-r.left)/r.width*n), i = Math.floor((e.clientY-r.top)/r.height*n); if(i<0||j<0||i>=n||j>=n) return; document.getElementById('cka-read').textContent = `layer ${Ls[i]} × layer ${Ls[j]}\nCKA ${M[i][j].toFixed(2)}\n(${mn==='base'?'Base':'post-trained'})`; };
}
drawCKA('post');
document.querySelectorAll('[data-cm]').forEach(b=> b.addEventListener('click', ()=>{ document.querySelectorAll('[data-cm]').forEach(x=>x.setAttribute('aria-pressed', x===b)); drawCKA(b.dataset.cm); }));

// ---------- cases table ----------
function pfmt(p){ return p<0.001 ? p.toExponential(1) : p.toFixed(3); }
function drawCases(mn){
  const rows = D.cases[mn]; const t = document.getElementById('cases-table');
  const art = D.cases.artifact;
  let h = '<thead><tr><th>readout</th><th>hidden > seen, matched</th><th>other context</th><th>p vs coin</th><th>paired w/l</th><th>paired p</th><th>P(hidden) moves</th><th>P(seen) moves</th><th>our JVP lens, matched</th></tr></thead><tbody>';
  for (const r of rows){
    const L = (r.readout.match(/L(\d+)/)||[])[1];
    const isHosted = r.readout.startsWith('hosted'), isLogit = r.readout.startsWith('logit');
    const label = r.readout==='model_output' ? 'model output' : (isHosted?'hosted L'+L:'logit lens L'+L);
    const jv = mn==='post' ? (isHosted ? (art['jlens_L'+L+'_all']? art['jlens_L'+L+'_all'].matched_wins+'/42' : (art['jlens_L'+L+'_self']? art['jlens_L'+L+'_self'].matched_wins+'/42':'')) : isLogit ? (art['logit_lens_L'+L]? art['logit_lens_L'+L].matched_wins+'/42':'') : art.model_output.matched_wins+'/42') : '';
    h += `<tr class="${isHosted?'hosted':isLogit?'logitrow':''}"><td class="r">${label}</td><td>${r.matched}/${r.n}</td><td>${r.mismatched}/${r.n}</td><td class="${r.matched_p<0.05?'sig':''}">${pfmt(r.matched_p)}</td><td>${r.paired[0]}/${r.paired[1]}</td><td class="${r.paired_p<0.05?'sig':''}">${pfmt(r.paired_p)}</td><td>${pfmt(r.p_true)}</td><td class="${r.p_seen<0.05?'strong':''}">${pfmt(r.p_seen)}</td><td>${jv}</td></tr>`;
  }
  t.innerHTML = h + '</tbody>';
}
drawCases('post');
document.querySelectorAll('[data-tm]').forEach(b=> b.addEventListener('click', ()=>{ document.querySelectorAll('[data-tm]').forEach(x=>x.setAttribute('aria-pressed', x===b)); drawCases(b.dataset.tm); }));

// ---------- decision-point tokens ----------
function tokHtml(t, hit){ const s = t.replace(/^ /, '␣'); return `<span class="tok${hit?' hit':''}">${s.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</span>`; }
const sel = document.getElementById('case-sel');
Object.keys(D.decision_topk).forEach(k=>{ const o = document.createElement('option'); o.value = k; o.textContent = k; sel.appendChild(o); });
function drawTok(){ const g = document.getElementById('tokgrid'); const rows = D.decision_topk[sel.value]; g.innerHTML = Object.keys(rows).map(L=>{ const n = +L.slice(1); return `<div class="L ${ATTN.includes(n)?'attn':''}">${L}</div><div class="toks">${rows[L].map(([t,p])=> `<span title="p=${p}">${tokHtml(t, /\d|eighty|ninety|seventy|百/.test(t))}</span>`).join('')}</div>`; }).join(''); }
sel.addEventListener('change', drawTok); drawTok();

// ---------- rank trajectories ----------
const prompts = {france:{mn:'post', targets:[' Paris'], title:'“The capital of France is” — rank of “ Paris”', sub:'Post-trained checkpoint. Hosted lens in teal, logit lens in amber; the wash is the band.'},
                 spider:{mn:'post', targets:[' spider',' 8'], title:'“The number of legs on the animal that spins webs is” — “ spider” and “ 8”', sub:'Post-trained checkpoint. The unspoken intermediate stays deep in the vocabulary; the answer digit rises.'},
                 arith:{mn:'base', targets:['3','9','7'], title:'“calc: ( 1 + 2 ) * 3 - 2 =” — the digits 3, 9 and 7', sub:'Base checkpoint. Intermediates 3 then 9 then answer 7; Qwen splits numbers into digits, so only single digits are single tokens.'}};
let curP = 'france';
function drawRank(){
  const P = prompts[curP], s = D.sanity[P.mn][curP];
  document.getElementById('rank-title').textContent = P.title; document.getElementById('rank-sub').textContent = P.sub;
  const series = []; const shades = [col('--lens'), col('--attn'), col('--good')];
  P.targets.forEach((t,i)=>{ if (s.targets[t]) series.push({name:'hosted: '+JSON.stringify(t), values:s.targets[t], cls:'lens', color:shades[i%shades.length], digits:0, thin:i>0});
                              if (s.targets_ll[t]) series.push({name:'logit lens: '+JSON.stringify(t), values:s.targets_ll[t], cls:'logit', color:col('--logit'), digits:0, thin:i>0}); });
  lineChart(document.getElementById('c-rank'), {layers:s.layers, log:true, ymin:1, ymax:250000, ylabel:'rank in readout (log)', aria:'rank by layer', series});
  // recolor per-target hosted paths
  const paths = document.querySelectorAll('#c-rank path.lens'); let k=0; P.targets.forEach((t,i)=>{ if (s.targets[t]){ if (paths[k]) paths[k].style.stroke = shades[i%shades.length]; k++; } });
  drawRankToks();
}
function drawRankToks(){ const L = +document.getElementById('scrub').value; document.getElementById('scrub-L').textContent = 'L'+L; const P = prompts[curP], s = D.sanity[P.mn][curP]; const g = document.getElementById('rank-toks');
  const toks = s.top8[String(L)] || []; const ranks = P.targets.map(t=> s.targets[t] ? `${JSON.stringify(t)} rank ${s.targets[t][s.layers.indexOf(L)]}` : '').filter(Boolean).join(' · ');
  g.innerHTML = `<div class="L ${ATTN.includes(L)?'attn':''}">L${L}</div><div class="toks">${toks.map(t=> tokHtml(t, P.targets.some(x=> x.trim()===t.trim()))).join('')}<span style="font-family:var(--mono);font-size:12px;color:var(--muted);align-self:center;margin-left:6px">${ranks}</span></div>`; }
document.getElementById('scrub').addEventListener('input', drawRankToks);
document.querySelectorAll('[data-p]').forEach(b=> b.addEventListener('click', ()=>{ document.querySelectorAll('[data-p]').forEach(x=>x.setAttribute('aria-pressed', x===b)); curP = b.dataset.p; drawRank(); }));
drawRank();

// ---------- contrast dot plots ----------
function dotPlot(container, suite){
  const post = D.contrast.post[suite], base = D.contrast.base[suite];
  const W = 300, H = 170, m = {l:36, r:10, t:22, b:24};
  const wrap = document.createElement('div'); wrap.className = 'chart'; wrap.innerHTML = `<h3>${suite.replace('_',' ')}</h3><p>n = ${post.best_user.length} prompts · best rank in the user turn</p>`; container.appendChild(wrap);
  const svg = el('svg', {viewBox:`0 0 ${W} ${H}`, role:'img'}, wrap);
  const y = v => m.t + (Math.log10(1000)-Math.log10(Math.max(1,Math.min(1000,v))))/3*(H-m.t-m.b);
  const ax = el('g', {class:'axis'}, svg);
  for (const v of [1,10,100,1000]){ el('line',{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v),'stroke-opacity':.6},ax); const t=el('text',{x:m.l-5,y:y(v)+3,'text-anchor':'end'},ax); t.textContent=v; }
  const cols = [{k:'post', x:m.l+ (W-m.l-m.r)*0.3, data:post, fill:true},{k:'base', x:m.l+(W-m.l-m.r)*0.7, data:base, fill:false}];
  for (const c of cols){
    const t = el('text',{x:c.x,y:H-8,'text-anchor':'middle'},ax); t.textContent = c.k==='post'?'post-trained':'Base';
    c.data.best_user.forEach((v,i)=>{ const jitter = ((i*37)%11-5)*2.2; const d = el('circle',{cx:c.x+jitter, cy:y(v), r:4.2, fill:c.fill?col('--lens'):col('--card'), stroke:col('--lens'),'stroke-width':1.6},svg);
      d.addEventListener('mousemove', e=> showTip(e, `${c.k==='post'?'post-trained':'Base'} · best rank ${v}\nfraction of user tokens with a reaction in top-10: ${c.data.frac[i]}\nresponse start: ${c.data.start[i]}\n“${c.data.prompts[i].slice(0,90)}…”`)); d.addEventListener('mouseleave', hideTip); });
    const med = c.data.summary.median_best_rank_user_turn; el('line',{x1:c.x-26,x2:c.x+26,y1:y(med),y2:y(med),stroke:col('--ink'),'stroke-width':2},svg);
    const ml = el('text',{x:c.x+30,y:y(med)+3,'text-anchor':'start'},ax); ml.textContent='med '+med;
  }
}
for (const suite of ['bereavement','danger','danger_control','help']) dotPlot(document.getElementById('dots'), suite);

// ---------- recurrent memory by head ----------
(function(){
  const Tp = D.transport; if (!Tp) return;
  const cv2 = document.getElementById('tau-canvas'), c2 = cv2.getContext('2d');
  const rows = Tp.tau_log10.length, cols = Tp.tau_log10[0].length, cw = cv2.width/cols, ch = cv2.height/rows;
  const c0 = hexToRgb(col('--bg-2')), c1 = hexToRgb(col('--lens')), c2c = hexToRgb(col('--attn'));
  function colr(v){ const t = Math.max(0, Math.min(1, v/4)); const a = t<0.5 ? c0 : c1, b = t<0.5 ? c1 : c2c, u = t<0.5 ? t*2 : (t-0.5)*2; return `rgb(${a.map((x,i)=>Math.round(x+(b[i]-x)*u)).join(',')})`; }
  for (let i=0;i<rows;i++) for (let j=0;j<cols;j++){ c2.fillStyle = colr(Tp.tau_log10[i][j]); c2.fillRect(j*cw, i*ch, cw+0.5, ch+0.5); }
  cv2.onmousemove = e => { const r = cv2.getBoundingClientRect(); const j = Math.floor((e.clientX-r.left)/r.width*cols), i = Math.floor((e.clientY-r.top)/r.height*rows); if(i<0||j<0||i>=rows||j>=cols) return; const v = Math.pow(10, Tp.tau_log10[i][j]); document.getElementById('tau-read').textContent = `block ${i} · value head ${j}\ngate time constant ≈ ${v>=100? Math.round(v).toLocaleString() : v.toFixed(1)} tokens`; };
  const pc = Tp.tau_percentiles;
  document.getElementById('tau-summary').textContent = `Median ${Math.round(pc['50'])} tokens; 90th percentile ${Math.round(pc['90']).toLocaleString()}; maximum ${Math.round(Math.pow(10, Math.max(...Tp.tau_log10.flat()))).toLocaleString()}. ${Tp.heads_over_1000} of ${Tp.heads_n} head entries exceed 1,000 tokens, in every block.`;
  const labels = Tp.bins;
  function binChart(container, spec){
    const W = 520, H = 250, m = {l:44, r:14, t:12, b:30}; const svg = el('svg', {viewBox:`0 0 ${W} ${H}`, role:'img'}, container);
    const x = i => m.l + (i+0.5)/labels.length*(W-m.l-m.r);
    const y = v => m.t + (Math.log10(spec.ymax)-Math.log10(Math.max(v, spec.ymin)))/(Math.log10(spec.ymax)-Math.log10(spec.ymin))*(H-m.t-m.b);
    const ax = el('g', {class:'axis'}, svg); el('line',{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b},ax);
    labels.forEach((L,i)=>{ const t = el('text',{x:x(i),y:H-m.b+15,'text-anchor':'middle'},ax); t.textContent = L; });
    for (const v of spec.ticks){ el('line',{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v),'stroke-opacity':.5},ax); const t = el('text',{x:m.l-6,y:y(v)+3,'text-anchor':'end'},ax); t.textContent = v>=1? v : v.toExponential(0); }
    const yl = el('text',{x:m.l,y:8,class:'ylab'},svg); yl.textContent = spec.ylabel;
    for (const s of spec.series){ const d = s.values.map((v,i)=> (i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join(' '); const p = el('path',{d, class:s.cls},svg); if (s.thin) p.setAttribute('stroke-width','1');
      s.values.forEach((v,i)=>{ const c = el('circle',{cx:x(i),cy:y(v),r:3.2,class:'dot',stroke:s.color},svg); c.addEventListener('mousemove', e=> showTip(e, `${s.name}\ngap ${labels[i]} tokens\n${v.toExponential(2)}`)); c.addEventListener('mouseleave', hideTip); }); }
  }
  binChart(document.getElementById('c-mass'), {ymin:1e-9, ymax:1, ticks:[1e-8,1e-6,1e-4,1e-2,1], ylabel:'mass share (log)', series:[
    {name:'median mass share', values:Tp.mass_p50, cls:'lens', color:col('--lens')},
    {name:'90th percentile mass share', values:Tp.mass_p90, cls:'lens', color:col('--lens'), thin:true},
    {name:'uniform null', values:Tp.mass_null, cls:'nul', color:col('--null')}]});
  binChart(document.getElementById('c-ret'), {ymin:1e-9, ymax:10, ticks:[1e-8,1e-6,1e-4,1e-2,1,10], ylabel:'retention to gap 1–4 (log)', series:[
    {name:'median retention', values:Tp.ret_p50, cls:'lens', color:col('--lens')},
    {name:'90th percentile retention', values:Tp.ret_p90, cls:'lens', color:col('--lens'), thin:true}]});
  const j = Tp.joint, n = Tp.cells;
  document.getElementById('transport-note').textContent = `Reconstruction: worst error ${Tp.recon.worst.toExponential(1)}, median ${Tp.recon.median.toExponential(1)} over ${Tp.recon.cells.toLocaleString()} cells, gate 1e-5. Cells meeting the joint criterion (retention above 0.05 and mass share above the uniform null): ${j[4]} of ${n[4]} at gaps 257–1024 and ${j[5]} of ${n[5]} at 1025–2688. Those cells belong to the long-gate heads on the map. On these agent transcripts most recurrent heads remember tens of tokens and 115 of 768 keep the gate open across the whole context; the same heads are slow in both transcripts, but at neutral input none is, so the text sets the scale and the head parameters set only the order. Their far-source sets overlap across read positions spread over the context by 1.8 of 10 against 1.3 for a null that keeps their write strengths and keys and randomises the query. That null is a write-strength-weighted running summary written down as a model, and the population sits within about a third of it: these heads accumulate such a summary along a comparatively stable key direction (key cosine 0.58 against 0.32) and read it from anywhere, with far-source sets only mildly more stable than the model predicts. No head shows a strong retrieval-like excess; three key-head units (block 1 key head 7, block 6 key head 0, block 15 key head 3) show a small one in both transcripts, about five times what chance would place there, and are named for inspection, block 6 the strongest. Long-range addressable memory in this model is therefore carried by attention. Whether the running summary is workspace content is the next measurement. The earlier claim that only attention carries content across a long context is withdrawn.`;
})();
})();
</script>
'''
open("atlas.html", "w").write(html.replace("__DATA__", data))
import os; print("atlas.html bytes:", os.path.getsize("atlas.html"))
