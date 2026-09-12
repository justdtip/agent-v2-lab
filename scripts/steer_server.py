"""The steering desk as a page, so the intervention can be looked at rather than read about.

The same resident model and the same hook as `steer_chat.py`; this adds a browser in front of it.
Nothing is exposed to the network: it binds to localhost on the card and you reach it through an ssh
tunnel, so the page is private to your machine and the instance needs no reconfiguration.

    # on your machine
    ssh -N -L 8765:localhost:8765 rtx6000 &
    ssh rtx6000 'cd /workspace/wsd/ws-d && PYTHONPATH=src \\
        /workspace/agent-v2-lab/.venv/bin/python scripts/steer_server.py \\
        --model /workspace/.hf_home/hub/models--google--gemma-3-12b-it/snapshots/<id> --port 8765'
    # then open http://localhost:8765

WHAT THE PAGE SHOWS THAT A TRANSCRIPT CANNOT. Every steered reply is replayed through the unsteered
model with its own tokens forced, so both runs visit an identical context at every position and the
only difference between them is the intervention. Each token is then painted by how far the clean
model was from choosing it: pale where the clean model agreed, saturated where the steered token was
one it would essentially never have emitted. Hovering a token says what the clean model wanted
instead and where the steered token ranked. On a real run "dissoci" came out at rank 37,237 with the
clean model's first choice being "Hi" — one forced token, and the rest of the word was the model's
own completion of it. That is the thing worth seeing, and it is invisible in the reply itself.

Only one request touches the model at a time: a single card, a single lock, a queue behind it.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE))

from steer_chat import THOUGHT_CLOSE, Chat, opens_thought, split_thought  # noqa: E402

PAGE = r"""<!doctype html><meta charset=utf-8><title>steering bench</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
/* One token set, defined light on :root and redefined dark, so no rule below carries a literal
   colour. `color-scheme` makes the native range thumb, select popup and scrollbars follow the page
   instead of rendering light chrome on a dark ground.
   One meaning per hue, reserved globally: --hot is failure and nothing else, --cool is the
   reasoning channel, --warm the answer channel, --steer the intervention. Intensity always means
   magnitude of effect. */
:root{
  color-scheme:light dark;
  --bg:#f5f6f8; --panel:#ffffff; --line:#d7dbe3;
  --ink:#14161a; --dim:#5a6374;
  --hot:#c0392b; --cool:#0b6ea8; --warm:#9a5b00; --good:#1b7f43; --steer:#7c3aed;
  --heat-rgb:124,58,237; --heat-ink:#ffffff; --focus:#0b6ea8;
  --bench:340px;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#14161a; --panel:#1c1f26; --line:#2b2f3a;
    --ink:#e6e8ee; --dim:#8b93a7;
    --hot:#ff6b5a; --cool:#5ac8fa; --warm:#ffab5c; --good:#4ade80; --steer:#c084fc;
    --heat-rgb:192,132,252; --heat-ink:#14161a; --focus:#5ac8fa;
  }
}
*{box-sizing:border-box}
body{margin:0;height:100dvh;overflow:hidden;background:var(--bg);color:var(--ink);
  font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  display:grid;grid-template-columns:minmax(0,1fr) var(--bench);
  grid-template-rows:34px minmax(0,1fr);grid-template-areas:"bar bar" "stage bench"}
body[data-bench=hidden]{--bench:0px}
:focus-visible{outline:2px solid var(--focus);outline-offset:1px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}

#bar{grid-area:bar;display:flex;align-items:center;gap:14px;padding:0 14px;
  border-bottom:1px solid var(--line);font:12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--dim);overflow:hidden;white-space:nowrap}
#bar label{margin:0;display:flex;align-items:center;gap:5px;font-size:12px}
#bar .bad{color:var(--hot)}
#activity{margin-left:auto;display:flex;align-items:center;gap:7px}
#activity .dot{width:7px;height:7px;border-radius:50%;background:var(--dim)}
body[data-busy] #activity .dot{background:var(--steer);animation:pulse 1s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.25}50%{opacity:1}}

/* Only #log and #rack scroll. The desk, the maker and the session row stay on screen however many
   vectors are in the rack -- they used to be pushed off by a scrolling right column. */
#stage{grid-area:stage;min-width:0;min-height:0;display:grid;
  grid-template-rows:auto minmax(0,1fr) auto auto}
#stagehead{display:flex;align-items:center;gap:14px;padding:6px 24px;flex-wrap:wrap;
  border-bottom:1px solid var(--line);font-size:11px;color:var(--dim)}
#log{overflow-y:auto;padding:20px 24px;scrollbar-gutter:stable}
#composer{display:grid;gap:8px 10px;padding:12px 24px;border-top:1px solid var(--line);
  grid-template-columns:minmax(0,1fr) auto auto;grid-template-areas:"chip chip chip" "text send ab"}
#chip{grid-area:chip}#msg{grid-area:text}#send{grid-area:send}#ab{grid-area:ab;margin-left:6px}

#bench{grid-area:bench;min-height:0;overflow:hidden;background:var(--panel);
  border-left:1px solid var(--line);display:grid;grid-template-rows:auto minmax(0,1fr) auto auto}
body[data-bench=hidden] #bench{display:none}
#desk{padding:14px;border-bottom:1px solid var(--line)}
#rack{overflow-y:auto;min-height:0;padding:10px 14px}
#make{border-top:1px solid var(--line);padding:12px 14px}
#session{border-top:1px solid var(--line);padding:10px 14px;display:flex;gap:8px;
  align-items:center;font-size:11px;color:var(--dim)}
#session .grow{flex:1}

h2{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);
  margin:0 0 9px;font-weight:600}
label{display:block;font-size:12px;color:var(--dim);margin:10px 0 3px}
label:first-of-type{margin-top:0}
select,input[type=number],input[type=text],textarea{width:100%;background:var(--bg);
  border:1px solid var(--line);border-radius:6px;color:var(--ink);padding:7px;font:inherit}
textarea{resize:none;max-height:30vh;line-height:1.5}
input[type=range]{width:100%;padding:0}
button{background:var(--bg);border:1px solid var(--line);border-radius:7px;color:var(--ink);
  padding:8px 13px;font:inherit;cursor:pointer}
#composer button{padding:11px 18px}
button:hover:not(:disabled){border-color:var(--dim)}
button:disabled{opacity:.4;cursor:default}
button[aria-pressed=true]{background:var(--panel);border-color:var(--focus);color:var(--ink)}
.hint{font-size:11px;color:var(--dim);margin-top:5px;line-height:1.45}
.row{display:flex;gap:8px}.row>*{flex:1}
.seg{display:flex;gap:0}
.seg button{border-radius:0;flex:1;padding:6px 8px;font-size:12px;border-left-width:0}
.seg button:first-child{border-radius:6px 0 0 6px;border-left-width:1px}
.seg button:last-child{border-radius:0 6px 6px 0}

/* the desk */
#armed{font-size:12px;line-height:1.5;padding:8px 10px;border-radius:7px;border:1px solid var(--line);
  background:var(--bg);margin-bottom:11px}
#armed.clean{color:var(--dim)}
#armed.live{border-color:var(--steer)}
#armed b{font-family:ui-monospace,Menlo,monospace;font-weight:600}
.reading{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:24px;text-align:center;
  margin:4px 0 2px}
.reading.hotband{color:var(--hot)}
#fader{background:linear-gradient(to right,
  transparent 0 10%, rgba(27,127,67,.10) 10% 30%, transparent 30% 60%, rgba(192,57,43,.12) 60% 100%)}
#faderband{text-align:center}

/* the rack */
.slotrow{display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:start;
  padding:7px 0;border-bottom:1px solid var(--line)}
.slotrow.armed{background:rgba(124,58,237,.07);border-radius:6px;padding:7px 6px;
  box-shadow:inset 2px 0 0 var(--steer)}
@media (prefers-color-scheme:dark){.slotrow.armed{background:rgba(192,132,252,.10)}}
.slotrow .name{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.slotrow .prov{font-size:11px;color:var(--dim);line-height:1.4;word-break:break-word}
button.arm{padding:3px 9px;font-size:11px}
button.x{padding:2px 8px;font-size:14px;line-height:1;color:var(--dim);border-color:transparent}

/* the transcript */
.turn{margin-bottom:20px;max-width:78ch}
.who{font-size:11px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:5px;color:var(--dim)}
.you .who{color:var(--ink);opacity:.7}
.you .body{max-width:74ch;white-space:pre-wrap;word-wrap:break-word}
.turn.receipt{max-width:none;font:11px/1.6 ui-monospace,Menlo,monospace;color:var(--dim);
  border-left:2px solid var(--line);padding-left:10px;margin:10px 0}
.turn.receipt.bad{color:var(--hot);border-left-color:var(--hot)}
.orient{color:var(--dim);font-size:12px;line-height:1.7;max-width:72ch}
.orient div{margin-bottom:6px}

/* channel blocks: the reply IS the painted text, printed once */
.channel{max-width:76ch;margin:8px 0;padding:9px 12px;border-left:2px solid var(--line);
  border-radius:0 8px 8px 0;background:rgba(127,127,127,.04)}
.channel.thought{border-left-color:var(--cool)}
.channel.answer{border-left-color:var(--warm)}
.channel .head{display:flex;gap:10px;align-items:center;font-size:10px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--dim);margin-bottom:6px}
.channel .head button{padding:1px 7px;font-size:10px;letter-spacing:.06em}
.channel .count{margin-left:auto;text-transform:none;letter-spacing:0;font-family:ui-monospace,Menlo,monospace}
.channel[data-folded] .painted{display:none}
.painted{white-space:pre-wrap;word-break:break-word;
  font:13px/1.95 ui-monospace,SFMono-Regular,Menlo,monospace}
body[data-paint=prose] .painted{font:14px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif}
body[data-paint=prose] .tok{background:none!important;border-bottom:0!important;color:inherit!important;
  box-shadow:none!important}
body[data-paint=prose] .tok::before{content:none!important}
.tok{white-space:pre-wrap;border-radius:3px;border-bottom:0 solid var(--dim);cursor:pointer}
.tok:hover{outline:1px solid var(--dim)}
.tok.pinned{outline:1px solid var(--focus);outline-offset:1px}
.tok[data-ws=nl]::before{content:'\21b5';color:var(--dim);opacity:.55}
.tok[data-ws=sp]{box-shadow:inset 0 0 0 1px rgba(var(--heat-rgb),.45)}
body[data-only-changed] .tok[data-rank="0"]{opacity:.3}
.swatch{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-3px;
  border:1px solid var(--line)}
.legend{display:flex;align-items:center;gap:5px}

/* the three meta lines */
.meta{margin-top:9px;display:grid;grid-template-columns:auto 1fr;gap:2px 10px;
  font:11px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--dim)}
.meta dt{opacity:.8}
.meta dd{margin:0;word-break:break-word}
.meta .bad{color:var(--hot)}

/* A/B as one card */
.ab{max-width:none;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:20px}
.ab .cap{padding:7px 12px;background:rgba(127,127,127,.06);border-bottom:1px solid var(--line);
  font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.ab .arms{display:grid;grid-template-columns:1fr 1fr}
.ab .arm{padding:12px;min-width:0}
.ab .arm+.arm{border-left:1px solid var(--line)}
.ab .arm.clean{box-shadow:inset 3px 0 0 var(--good)}
.ab .arm.steered{box-shadow:inset 3px 0 0 var(--steer)}
.ab .arm h3{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);
  margin:0 0 8px;font-weight:600}
.ab .foot{padding:10px 12px;border-top:1px solid var(--line)}
.ab .verdict{font-size:13px;margin-bottom:6px}
@media (max-width:900px){.ab .arms{grid-template-columns:1fr}.ab .arm+.arm{border-left:0;border-top:1px solid var(--line)}}

/* the inspector */
#inspector{max-height:0;overflow:hidden;padding:0 24px;background:var(--panel);
  border-top:1px solid var(--line);
  font:12px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;
  transition:max-height .12s ease,padding .12s ease}
#inspector[data-open]{max-height:104px;padding:10px 24px}
#inspector .nav{float:right;display:flex;gap:6px}
#inspector button{padding:2px 9px;font-size:11px}
#chip{display:flex;align-items:center;gap:9px;font:11px/1 ui-monospace,Menlo,monospace;
  color:var(--dim)}
#chip .tag{padding:3px 9px;border-radius:99px;border:1px solid var(--line)}
#chip .tag.live{border-color:var(--steer);color:var(--ink)}
</style>
<div id=bar>
  <span id=runfacts>loading…</span>
  <label><input type=checkbox id=think style="width:auto"> reasoning</label>
  <span id=thinwhy></span>
  <span id=logfacts></span>
  <span id=activity><span class=dot></span><span id=activitytext>idle</span></span>
</div>

<div id=stage>
  <div id=stagehead>
    <span class=seg role=group aria-label="how tokens are painted">
      <button id=paintprose type=button aria-pressed=false>prose</button>
      <button id=paintrank type=button aria-pressed=true>rank</button>
    </span>
    <span class=legend id=legend></span>
    <label style="margin:0;display:flex;gap:5px;align-items:center">
      <input type=checkbox id=onlychanged style="width:auto"> only changed</label>
    <button id=benchtoggle type=button style="margin-left:auto;padding:3px 9px;font-size:11px"
      title="collapse the bench (Ctrl+\)">hide bench</button>
  </div>

  <div id=log>
    <div class=orient id=orient>
      <div>The fader is <b>per cent of the residual norm measured at the injection site</b>, not alpha.</div>
      <div>Every steered reply is replayed through the clean model with its own tokens forced, so the colour is the intervention and nothing else.</div>
      <div><b>turn</b> = over your message &middot; <b>reply</b> = while generating &middot; <b>both</b> = the paper's protocol.</div>
      <div>Measured on this model: 25&ndash;75% is the working band; above ~150% it stops producing steered text and produces repetition loops.</div>
    </div>
  </div>

  <div id=inspector class=mono></div>

  <form id=composer autocomplete=off>
    <div id=chip><span class=tag id=chiptag>running clean</span><span id=chipwhy></span></div>
    <textarea id=msg rows=1 placeholder="say something to the model — Enter sends, Shift+Enter for a new line" autofocus></textarea>
    <button id=send>send</button>
    <button id=ab type=button title="re-run the last message clean and steered on identical history">A/B last</button>
  </form>
</div>

<div id=bench>
  <div id=desk>
    <h2>desk</h2>
    <div id=armed class=clean>running clean &mdash; arm a vector in the rack</div>
    <label for=layer>inject at layer <span id=layerof class=mono></span></label>
    <div class=row><input type=number id=layer min=1 value=1><button id=layerdefault type=button
      title="two-thirds of depth">&frac23;</button></div>
    <label for=fader>fader &mdash; per cent of the residual norm at the site</label>
    <div class=reading id=readingwrap><span id=pct>0</span>%</div>
    <input type=range id=fader min=0 max=250 step=1 value=0>
    <div class="hint" id=faderband>0 is clean &mdash; the vector stays in the rack</div>
    <label for=scope>what the injection covers</label>
    <select id=scope>
      <option value=turn>your message, before the model starts writing</option>
      <option value=reply>the reply, while it is being written</option>
      <option value=both>both &mdash; the paper's protocol</option>
      <option value=all>the whole context, every token</option>
    </select>
  </div>

  <div id=rack>
    <h2>vectors</h2>
    <div id=racklist></div>
    <h2 style="margin-top:18px">combine</h2>
    <div class=row><select id=cl></select><select id=cop>
      <option value="-">&minus; difference</option>
      <option value="+">+ sum</option>
      <option value="x">&times; scale</option>
    </select></div>
    <div class=row style="margin-top:6px"><select id=cr></select>
      <input type=number id=cfactor value=2 step=0.5 hidden></div>
    <div class=row style="margin-top:6px"><input type=text id=cinto placeholder="name for the result (required)"><button id=cgo type=button disabled>make</button></div>
    <div class=hint id=chint>A difference of two states the model produced is a direction in the model's own units &mdash; no contrastive frame, no magnitude convention. Try &ldquo;I am cheerful&rdquo; minus &ldquo;I am despairing&rdquo;.</div>
  </div>

  <div id=make>
    <h2>put a vector in the rack</h2>
    <span class=seg role=group aria-label="where the vector comes from">
      <button id=srcword type=button aria-pressed=true>word</button>
      <button id=srcprompt type=button aria-pressed=false>prompt</button>
      <button id=srchere type=button aria-pressed=false>this chat</button>
    </span>
    <div id=srcfields>
      <div data-src=word>
        <label for=word>concept</label>
        <input type=text id=word placeholder="a word or phrase">
        <div class=hint>&ldquo;Tell me about {word}.&rdquo; minus the mean over 24 random words.</div>
      </div>
      <div data-src=prompt hidden>
        <label for=xprompt>prompt to read the residual from</label>
        <input type=text id=xprompt placeholder="any prompt">
        <label for=xpos>position</label>
        <input type=number id=xpos placeholder="blank = the last token">
      </div>
      <div data-src=here hidden>
        <div class=hint id=herehint>Reads the live conversation's own last position &mdash; the move a single-shot console cannot make.</div>
      </div>
    </div>
    <label for=slotname>slot name</label>
    <div class=row><input type=text id=slotname placeholder="required"><button id=mk type=button disabled>make</button></div>
    <div class=hint id=squeezed></div>
  </div>

  <div id=session>
    <span class=grow id=sessionfacts>0 messages</span>
    <button id=undo type=button disabled>undo</button>
    <button id=clear type=button>clear</button>
  </div>
</div>
<script>
const $=id=>document.getElementById(id), log=$('log');
let busy=false, S=null, lastSent='', src='word', pinned=null, spanFrom=null, allRows=[];
const post=(path,body)=>fetch(path,{method:'POST',headers:{'content-type':'application/json'},
  body:JSON.stringify(body||{})}).then(r=>r.json());
const squeeze=s=>[...String(s).toLowerCase()].filter(c=>/[a-z0-9]/.test(c)).join('').slice(0,12);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

/* ---- painting -------------------------------------------------------------------------------
   One ramp, one hue. Alpha is |delta logprob|, which is how much the intervention cost in nats;
   the bottom border is the rank band, which is a different phenomenon (a coin flip the injection
   won versus a confident model overruled) and worth seeing at the same time. The channel is the
   block the tokens sit in, not the colour, so one legend describes the whole reply and heat is
   comparable across the boundary -- which is the comparison the channel split exists to make. */
function heat(d){ return Math.min(0.78, Math.abs(d||0)/6); }
function bandpx(rank){ return rank===0?0 : rank<=10?1 : rank<=1000?2 : 3; }

function painted(rows, from, to){
  const wrap=document.createElement('div'); wrap.className='painted'; wrap.tabIndex=0;
  for(let i=from;i<to;i++){
    const r=rows[i], s=document.createElement('span');
    s.className='tok'; s.textContent=r.tok; s.dataset.rank=r.rank; s.dataset.i=i;
    if(/^\n+$/.test(r.tok)) s.dataset.ws='nl'; else if(/^ +$/.test(r.tok)) s.dataset.ws='sp';
    if(r.rank>0){
      const a=heat(r.d);
      s.style.background=`rgba(var(--heat-rgb),${a})`;
      if(a>=0.5) s.style.color='var(--heat-ink)';
      s.style.borderBottomWidth=bandpx(r.rank)+'px';
    }
    s.title = r.rank===0 ? 'the clean model wanted this too'
      : `pos ${i} · ${JSON.stringify(r.tok)} · rank ${r.rank.toLocaleString()} · Δ ${r.d.toFixed(2)} · clean wanted ${JSON.stringify(r.top)}`;
    s.onclick=e=>{ if(e.shiftKey&&pinned!==null) showSpan(rows,Math.min(pinned,i),Math.max(pinned,i));
                   else pin(rows,i); };
    wrap.appendChild(s);
  }
  return wrap;
}

function channel(kind, rows, from, to, folded){
  const box=document.createElement('div'); box.className='channel '+kind;
  const changed=rows.slice(from,to).filter(r=>r.rank>0).length;
  const head=document.createElement('div'); head.className='head';
  head.innerHTML=`<span>${kind==='thought'?'reasoning':'answer'}</span>`;
  if(kind==='thought'){
    const b=document.createElement('button'); b.type='button';
    b.setAttribute('aria-expanded', folded?'false':'true');
    b.textContent=folded?'unfold':'fold';
    b.onclick=()=>{const f=box.hasAttribute('data-folded');
      if(f) box.removeAttribute('data-folded'); else box.setAttribute('data-folded','');
      b.textContent=f?'fold':'unfold'; b.setAttribute('aria-expanded',String(f));};
    head.appendChild(b);
  }
  const c=document.createElement('span'); c.className='count';
  c.textContent = to>from ? `${changed}/${to-from} changed` : 'nothing generated';
  head.appendChild(c);
  box.appendChild(head);
  box.appendChild(painted(rows,from,to));
  if(folded) box.setAttribute('data-folded','');
  return box;
}

/* Rows carry a channel tag; find the boundary so the two blocks split at the right token. */
function split(rows){
  let b=rows.length;
  for(let i=0;i<rows.length;i++) if(rows[i].phase!=='thought'){ b=i; break; }
  return b;
}

/* ---- the three meta lines -------------------------------------------------------------------- */
function meta(r){
  const d=r.desk||{}, h=r.hook||null, dl=document.createElement('dl'); dl.className='meta';
  const add=(k,v,bad)=>{const dt=document.createElement('dt');dt.textContent=k;
    const dd=document.createElement('dd'); dd.innerHTML=v; if(bad) dd.className='bad';
    dl.appendChild(dt); dl.appendChild(dd);};
  if(!d.live){
    add('desk','clean — no vector armed');
  }else{
    let line=`<b>${esc(d.slot)}</b> ${d.percent}% at L${d.layer} · scope ${esc(d.scope)}`;
    if(d.scope==='all') line+=' — the fader was calibrated at the last prompt token and applied at every position';
    add('desk', line);
    if(h) add('hook', h.applications===0
      ? 'hook never fired — the intervention did not reach the model'
      : `fired ${h.applications}× over ${h.positions_touched} position(s), from token ${h.from_position}, ${h.sustain?'sustained through decoding':'prefill only'}`,
      h.applications===0);
  }
  const cap=S&&S.max_tokens?` · cap ${S.max_tokens}`:'';
  add('run', `${r.emitted??'?'} tokens${cap} · ${(r.seconds??0).toFixed(1)} s · prompt ${r.prompt_tokens??'?'} tokens`);
  const rows=r.tokens||[];
  if(!rows.length){ add('divergence', d.live?'no replay — the reply was empty':'nothing to compare — the desk was clean'); }
  else{
    const b=split(rows), ch=rows.filter(x=>x.rank>0).length;
    const worst=rows.reduce((a,x,i)=>Math.abs(x.d)>Math.abs(rows[a].d)?i:a,0);
    const th=rows.slice(0,b).filter(x=>x.rank>0).length;
    let line=`${ch} of ${rows.length} tokens changed (${Math.round(100*ch/rows.length)}%)`;
    if(b>0&&b<rows.length) line+=` · reasoning ${th}/${b} · answer ${ch-th}/${rows.length-b}`;
    line+=` · max |Δ| ${Math.abs(rows[worst].d).toFixed(2)} at pos ${worst}`;
    add('divergence', line);
  }
  return dl;
}

/* ---- transcript ------------------------------------------------------------------------------ */
function clearOrient(){ const o=$('orient'); if(o) o.remove(); }
/* The browser has not laid the new turn out yet when it is appended, so setting scrollTop in the
   same frame scrolls to the old height and the reply appears above the fold. */
function toBottom(){ requestAnimationFrame(()=>{ log.scrollTop=log.scrollHeight; }); }

function turn(who, text, cls){
  clearOrient();
  const w=document.createElement('div'); w.className='turn '+(cls||'');
  if(cls&&cls.indexOf('receipt')>=0){ w.textContent=text; }
  else{
    w.innerHTML=`<div class=who>${esc(who)}</div><div class=body></div>`;
    w.querySelector('.body').textContent=text;
  }
  log.appendChild(w); toBottom(); return w;
}

function modelTurn(r){
  clearOrient();
  const w=document.createElement('div'); w.className='turn';
  w.innerHTML='<div class=who>model</div>';
  const rows=r.tokens||[];
  if(rows.length){
    const b=split(rows);
    if(b>0) w.appendChild(channel('thought',rows,0,b,false));
    if(b<rows.length) w.appendChild(channel('answer',rows,b,rows.length,false));
  }else{
    if(r.thought){ const t=document.createElement('div'); t.className='channel thought';
      t.innerHTML='<div class=head><span>reasoning</span></div>';
      const p=document.createElement('div'); p.className='painted'; p.textContent=r.thought;
      t.appendChild(p); w.appendChild(t); }
    const a=document.createElement('div'); a.className='channel answer';
    a.innerHTML='<div class=head><span>answer</span></div>';
    const p=document.createElement('div'); p.className='painted';
    p.textContent = r.reply || (r.truncated
      ? `the ${S&&S.max_tokens?S.max_tokens:'token'}-token cap ran out inside the reasoning channel — turn reasoning off, or restart the server with --max-tokens ${S&&S.max_tokens?S.max_tokens*2:1024}`
      : '(nothing)');
    if(!r.reply) p.style.color='var(--hot)';
    a.appendChild(p); w.appendChild(a);
  }
  w.appendChild(meta(r));
  allRows=rows;
  log.appendChild(w); toBottom();
}

function abCard(r){
  clearOrient();
  const w=document.createElement('div'); w.className='turn ab';
  const cap=document.createElement('div'); cap.className='cap';
  cap.textContent=`A/B — a probe on ${r.history} message(s) of history, nothing was added to the conversation`;
  w.appendChild(cap);
  const arms=document.createElement('div'); arms.className='arms';
  const arm=(cls,title,thought,body,rows,cut)=>{
    const a=document.createElement('div'); a.className='arm '+cls;
    a.innerHTML=`<h3>${title}</h3>`;
    if(thought){ const t=document.createElement('div'); t.className='channel thought';
      t.setAttribute('data-folded','');
      t.innerHTML='<div class=head><span>reasoning</span></div>';
      const b=document.createElement('button'); b.type='button'; b.textContent='unfold';
      b.setAttribute('aria-expanded','false');
      b.onclick=()=>{const f=t.hasAttribute('data-folded');
        if(f)t.removeAttribute('data-folded'); else t.setAttribute('data-folded','');
        b.textContent=f?'fold':'unfold'; b.setAttribute('aria-expanded',String(f));};
      t.querySelector('.head').appendChild(b);
      const p=document.createElement('div'); p.className='painted'; p.textContent=thought;
      t.appendChild(p); a.appendChild(t); }
    const c=document.createElement('div'); c.className='channel answer';
    c.innerHTML='<div class=head><span>answer</span></div>';
    if(rows&&rows.length){ const b=split(rows); c.appendChild(painted(rows,b,rows.length)); }
    else{ const p=document.createElement('div'); p.className='painted';
          p.textContent=body||'(nothing)';
          if(cut) p.style.color='var(--hot)';
          c.appendChild(p); }
    a.appendChild(c); return a;
  };
  const cut=`the ${r.cap||'token'}-token cap ran out inside the reasoning channel — `
    +`restart the server with a larger --max-tokens, or turn reasoning off`;
  arms.appendChild(arm('clean','A — clean',r.clean_thought,r.clean_cut?cut:r.clean,null,r.clean_cut));
  arms.appendChild(arm('steered','B — steered',r.steered_thought,r.steered_cut?cut:r.steered,r.tokens,r.steered_cut));
  w.appendChild(arms);
  const foot=document.createElement('div'); foot.className='foot';
  const v=document.createElement('div'); v.className='verdict';
  const live=r.desk&&r.desk.live;
  v.textContent = (r.clean_cut&&r.steered_cut)
      ? 'both arms spent the whole token budget reasoning and never reached an answer — there is nothing here to compare'
    : !r.same ? 'they differ — neither reply was added to the conversation'
    : live ? 'identical output — neither reply was added to the conversation'
    : 'the desk was clean in both arms — raise the fader and run A/B again';
  if(r.clean_cut&&r.steered_cut) v.style.color='var(--hot)';
  if(r.same&&!live) v.style.color='var(--hot)';
  foot.appendChild(v); foot.appendChild(meta(r));
  w.appendChild(foot);
  allRows=r.tokens||[];
  log.appendChild(w); toBottom();
}

/* ---- inspector ------------------------------------------------------------------------------- */
function pin(rows,i){
  pinned=i; spanFrom=null;
  for(const el of document.querySelectorAll('.tok.pinned')) el.classList.remove('pinned');
  const el=[...document.querySelectorAll('.tok')].find(t=>+t.dataset.i===i&&t.closest('.turn')===log.lastElementChild);
  if(el){ el.classList.add('pinned'); el.scrollIntoView({block:'nearest'}); }
  const r=rows[i], b=split(rows);
  const line = r.rank===0
    ? `pos ${i} · ${JSON.stringify(r.tok)} · the clean model wanted this too · channel ${i<b?'reasoning':'answer'}`
    : `pos ${i} · steered ${JSON.stringify(r.tok)} · clean wanted ${JSON.stringify(r.top)} · rank ${r.rank.toLocaleString()} · Δ ${r.d.toFixed(2)} nats · channel ${i<b?'reasoning':'answer'}`;
  $('inspector').innerHTML=
    `<span class=nav><button type=button id=iprev>‹ prev</button><button type=button id=inext>next ›</button></span>`
    + esc(line) + `<br><span style="color:var(--dim)">← / → step to the next token the clean model did not want · Esc closes</span>`;
  $('inspector').setAttribute('data-open','');
  $('iprev').onclick=()=>step(rows,-1); $('inext').onclick=()=>step(rows,1);
}
function step(rows,dir){
  let i=(pinned===null?(dir>0?-1:rows.length):pinned)+dir;
  while(i>=0&&i<rows.length&&rows[i].rank===0) i+=dir;
  if(i>=0&&i<rows.length) pin(rows,i);
}
function showSpan(rows,a,b){
  const seg=rows.slice(a,b+1), ch=seg.filter(x=>x.rank>0);
  const mean=ch.length?ch.reduce((s,x)=>s+Math.abs(x.d),0)/ch.length:0;
  const worst=seg.reduce((m,x,i)=>Math.abs(x.d)>Math.abs(seg[m].d)?i:m,0);
  $('inspector').innerHTML=esc(
    `span ${a}–${b} · ${ch.length} of ${seg.length} changed (${Math.round(100*ch.length/seg.length)}%)`
    + ` · mean |Δ| ${mean.toFixed(2)} · max ${Math.abs(seg[worst].d).toFixed(2)} at pos ${a+worst}`);
  $('inspector').setAttribute('data-open','');
}
function closeInspector(){ $('inspector').removeAttribute('data-open'); pinned=null;
  for(const el of document.querySelectorAll('.tok.pinned')) el.classList.remove('pinned');
  $('msg').focus(); }

/* ---- busy ------------------------------------------------------------------------------------ */
const LOCKED=['send','ab','mk','cgo','undo','clear','layerdefault'];
function lock(on,what){
  busy=on;
  for(const b of LOCKED){ const e=$(b); if(e) e.disabled=on; }
  if(on) document.body.setAttribute('data-busy',''); else document.body.removeAttribute('data-busy');
  $('activitytext').textContent = on ? (what||'working…') : 'idle';
  if(!on) refreshButtons();
}

/* ---- state ----------------------------------------------------------------------------------- */
function armedText(){
  const d=S.desk, box=$('armed');
  if(!d.slot){ box.className='clean'; box.textContent='running clean — arm a vector in the rack'; return; }
  if(!d.live){
    box.className='clean';
    box.innerHTML=`<b>${esc(d.slot)}</b> is armed but the fader is at 0 — <b>muted</b>, the next reply will be clean`;
    return;
  }
  box.className='live';
  box.innerHTML=`steering with <b>${esc(d.slot)}</b> at <b>${d.percent}%</b> of the residual at <b>L${d.layer}</b>, over ${esc(scopeWords(d.scope))}`;
}
function scopeWords(s){ return {turn:'your message',reply:'the reply as it is written',
  both:'your message and the reply',all:'the whole context'}[s]||s; }

function faderLook(){
  const v=+$('fader').value;
  $('pct').textContent=v;
  $('readingwrap').className='reading'+(v>150?' hotband':'');
  $('faderband').textContent = v===0 ? '0 is clean — the vector stays in the rack'
    : v>150 ? 'repetition-loop territory — above ~150% this model stops producing steered text'
    : (v>=25&&v<=75) ? '25–75% is the usual working band'
    : v<25 ? 'below 25% the forward pass is untouched and the reply is usually unchanged'
    : 'between 75% and 150% the reply degrades before it steers';
}

function refreshButtons(){
  if(busy) return;
  const named=$('slotname').value.trim().length>0;
  const has = src==='word' ? $('word').value.trim() : src==='prompt' ? $('xprompt').value.trim() : (S&&S.history>0);
  $('mk').disabled=!(named&&has);
  const op=$('cop').value, same=$('cl').value===$('cr').value&&op!=='x';
  $('cgo').disabled=!$('cinto').value.trim()||same||!$('cl').value;
  $('chint').dataset.warn = same?'1':'';
  $('send').disabled=!$('msg').value.trim();
  $('ab').disabled=!S||S.history<1;
  $('undo').disabled=!S||S.history<2;
}

async function refresh(){
  S=await fetch('/state').then(r=>r.json());
  const s=S;
  $('runfacts').textContent=`${s.model} · ${s.layers} layers · hidden ${s.hidden} · ${s.device} ${String(s.dtype).replace('torch.','')} · cap ${s.max_tokens??'?'} tok`;
  const lg=s.log||{};
  $('logfacts').innerHTML = lg.enabled
    ? `<span title="${esc(lg.path||'')}">recording ${esc((lg.path||'').split('/').slice(-2).join('/'))}</span>`
    : '<span class=bad>NOT RECORDING</span>';
  $('think').checked=!!s.thinking; $('think').disabled=!s.supports_thinking;
  $('thinwhy').textContent = s.supports_thinking?'':'(this model has no reasoning channel)';

  $('layer').max=s.layers;
  if(!$('layer').dataset.touched) $('layer').value = s.desk.layer || Math.round(s.layers*2/3);
  $('layerof').textContent=`of ${s.layers} · two-thirds is L${Math.round(s.layers*2/3)}`;
  if(document.activeElement!==$('fader')) $('fader').value=s.desk.percent??0;
  $('scope').value=s.desk.scope||'turn';
  faderLook(); armedText();

  const names=s.detail.map(d=>d.name);
  $('racklist').innerHTML = s.detail.length ? s.detail.map(d=>{
    const prov = d.kind==='concept' ? `concept &ldquo;${esc(d.word||d.name)}&rdquo;`
      : d.kind==='from chat' ? `residual from the live chat, position ${d.position}`
      : d.prompt ? `residual at position ${d.position} of &ldquo;${esc(String(d.prompt).slice(0,60))}&rdquo;`
      : esc(d.kind);
    return `<div class="slotrow${d.name===s.desk.slot?' armed':''}">`
      + `<button class=arm type=button data-arm="${esc(d.name)}">${d.name===s.desk.slot?'armed':'arm'}</button>`
      + `<span><span class=name>${esc(d.name)}</span><div class=prov>${prov} &middot; read at L${d.layer} &middot; |v| ${Math.round(d.norm).toLocaleString()}</div></span>`
      + `<button class=x type=button data-drop="${esc(d.name)}" title="drop">&times;</button></div>`;
  }).join('') : '<div class=hint>no vectors yet — build a concept or extract one below</div>';
  for(const b of document.querySelectorAll('[data-arm]'))
    b.onclick=async e=>{ const n=e.target.dataset.arm;
      await post('/desk',{slot:n===s.desk.slot?null:n,layer:+$('layer').value,
        percent:+$('fader').value,scope:$('scope').value}); refresh(); };
  for(const b of document.querySelectorAll('[data-drop]'))
    b.onclick=async e=>{ await post('/drop',{slot:e.target.dataset.drop}); refresh(); };

  const keep=(el,v,fallback)=>{ el.innerHTML=names.map(x=>`<option>${esc(x)}</option>`).join('');
    if(names.includes(v)) el.value=v; else if(fallback&&names.length>1) el.value=names[1];
    else if(names.length) el.value=names[0]; };
  keep($('cl'), s.desk.slot||$('cl').value);
  keep($('cr'), $('cr').value===$('cl').value?null:$('cr').value, true);
  $('herehint').textContent=`Reads the live conversation's own last position (${s.history} message(s)) — the move a single-shot console cannot make.`;
  const onscreen=log.querySelectorAll('.turn:not(.receipt):not(.ab)').length;
  $('sessionfacts').textContent=`${s.history} message(s) · ${onscreen} on screen`;
  $('chiptag').className='tag'+(s.desk.live?' live':'');
  $('chiptag').textContent = s.desk.live
    ? `${s.desk.slot} ${s.desk.percent}% L${s.desk.layer} ${s.desk.scope}`
    : s.desk.slot ? `${s.desk.slot} muted` : 'running clean';
  refreshButtons();
}

/* ---- the desk, debounced so a drag is not thirty POSTs at the generation's lock --------------- */
let deskTimer=null;
function syncDesk(now){
  clearTimeout(deskTimer);
  const fire=async()=>{ await post('/desk',{layer:+$('layer').value,percent:+$('fader').value,
    scope:$('scope').value}); refresh(); };
  if(now) fire(); else deskTimer=setTimeout(fire,140);
}
$('fader').oninput=()=>{faderLook(); syncDesk(false);};
$('fader').onchange=()=>syncDesk(true);
$('layer').oninput=()=>{$('layer').dataset.touched='1'; syncDesk(false);};
$('scope').onchange=()=>syncDesk(true);
$('layerdefault').onclick=()=>{ $('layer').value=Math.round(S.layers*2/3);
  $('layer').dataset.touched='1'; syncDesk(true); };
$('think').onchange=async()=>{ await post('/thinking',{on:$('think').checked});
  turn('','reasoning channel '+($('think').checked?'on':'off'),'receipt'); refresh(); };

/* ---- sending --------------------------------------------------------------------------------- */
/* ---- streaming ------------------------------------------------------------------------------
   A reply takes ten to thirty seconds, and the page used to show nothing for all of it. The turn
   is built empty, filled token by token as they arrive, and replaced by the painted version once
   the clean replay comes back with it -- the replay is a second pass over the finished reply and
   cannot itself be streamed. */
const THOUGHT_CLOSE='<channel|>', THOUGHT_OPEN='<|channel>thought';
function liveTurn(){
  clearOrient();
  const w=document.createElement('div'); w.className='turn';
  w.innerHTML='<div class=who>model</div>';
  const mk=kind=>{ const b=document.createElement('div'); b.className='channel '+kind;
    b.innerHTML=`<div class=head><span>${kind==='thought'?'reasoning':'answer'}</span>`
      +`<span class=count></span></div>`;
    const p=document.createElement('div'); p.className='painted'; b.appendChild(p);
    b.hidden=true; w.appendChild(b); return b; };
  const t=mk('thought'), a=mk('answer');
  const note=document.createElement('div'); note.className='meta';
  note.innerHTML='<dt>run</dt><dd>generating…</dd>'; w.appendChild(note);
  log.appendChild(w); toBottom();
  return {w, t, a, note};
}
function fillLive(live, raw, n, cap){
  let thought='', answer=raw;
  if(raw.indexOf(THOUGHT_CLOSE)>=0){
    let head=raw.slice(0,raw.indexOf(THOUGHT_CLOSE));
    if(head.indexOf(THOUGHT_OPEN)>=0) head=head.slice(head.indexOf(THOUGHT_OPEN)+THOUGHT_OPEN.length);
    thought=head.replace(/^\n+/,''); answer=raw.slice(raw.indexOf(THOUGHT_CLOSE)+THOUGHT_CLOSE.length).replace(/^\n+/,'');
  }else if(raw.indexOf(THOUGHT_OPEN)>=0){
    thought=raw.slice(raw.indexOf(THOUGHT_OPEN)+THOUGHT_OPEN.length).replace(/^\n+/,''); answer='';
  }
  if(thought){ live.t.hidden=false; live.t.querySelector('.painted').textContent=thought;
               live.t.querySelector('.count').textContent='writing…'; }
  // The answer block appears only once there is an answer, or once it is clear there is no
  // reasoning channel at all. An empty ANSWER heading sitting under a thought that is still being
  // written says the model has finished thinking, which it has not.
  const showAnswer = answer.length>0 || !thought;
  live.a.hidden = !showAnswer;
  if(showAnswer){ live.a.querySelector('.painted').textContent=answer;
                  live.t.querySelector('.count').textContent=thought?'done':''; }
  live.note.innerHTML=`<dt>run</dt><dd>${n} of ${cap} tokens…</dd>`;
  toBottom();
}

$('composer').onsubmit=async e=>{
  e.preventDefault(); if(busy) return;
  const text=$('msg').value.trim(); if(!text) return;
  lastSent=text; $('msg').value=''; grow(); turn('you',text,'you'); lock(true,'generating…');
  const live=liveTurn();
  let raw='', n=0, cap=(S&&S.max_tokens)||512, done=null, failed=null;
  try{
    const res=await fetch('/say_stream',{method:'POST',
      headers:{'content-type':'application/json'},body:JSON.stringify({text})});
    const reader=res.body.getReader(), dec=new TextDecoder();
    let buf='';
    for(;;){
      const {value,done:end}=await reader.read(); if(end) break;
      buf+=dec.decode(value,{stream:true});
      let nl;
      while((nl=buf.indexOf('\n'))>=0){
        const line=buf.slice(0,nl); buf=buf.slice(nl+1);
        if(!line.trim()) continue;
        const ev=JSON.parse(line);
        if(ev.t==='start'){ cap=ev.cap||cap; }
        else if(ev.t==='tok'){ raw+=ev.s; n++; fillLive(live,raw,n,cap); }
        else if(ev.t==='replaying'){ $('activitytext').textContent='replaying through the clean model…';
          live.note.innerHTML=`<dt>run</dt><dd>${ev.tokens} tokens · replaying through the clean model…</dd>`; }
        else if(ev.t==='error'){ failed=ev.error; }
        else if(ev.t==='done'){ done=ev; }
      }
    }
  }catch(err){ failed=String(err); }
  live.w.remove();
  if(failed) turn('','error: '+failed,'receipt bad'); else if(done) modelTurn(done);
  lock(false); refresh();
};
$('ab').onclick=async()=>{
  if(busy) return; lock(true,'running both arms…');
  try{ const r=await post('/ab',{});
       if(r.error) turn('','error: '+r.error,'receipt bad'); else abCard(r); }
  catch(err){ turn('','error: '+String(err),'receipt bad'); }
  finally{ lock(false); refresh(); }
};

/* ---- the maker ------------------------------------------------------------------------------- */
function setSrc(which){
  src=which;
  for(const [id,k] of [['srcword','word'],['srcprompt','prompt'],['srchere','here']])
    $(id).setAttribute('aria-pressed', String(k===which));
  for(const d of document.querySelectorAll('#srcfields [data-src]')) d.hidden = d.dataset.src!==which;
  refreshButtons();
}
$('srcword').onclick=()=>setSrc('word');
$('srcprompt').onclick=()=>setSrc('prompt');
$('srchere').onclick=()=>setSrc('here');
$('word').oninput=()=>{ if(!$('slotname').dataset.touched) $('slotname').value=squeeze($('word').value);
  showSqueeze(); refreshButtons(); };
$('xprompt').oninput=refreshButtons;
$('slotname').oninput=()=>{ $('slotname').dataset.touched='1'; showSqueeze(); refreshButtons(); };
function showSqueeze(){
  const raw=$('slotname').value.trim(), sq=squeeze(raw);
  $('squeezed').textContent = raw&&sq!==raw ? `will be filed as: ${sq}` : '';
}
$('mk').onclick=async()=>{
  if(busy) return;
  const slot=squeeze($('slotname').value); if(!slot) return;
  lock(true,'building the vector…');
  try{
    let r;
    if(src==='word') r=await post('/concept',{word:$('word').value.trim(),layer:+$('layer').value});
    else if(src==='prompt') r=await post('/extract',{slot,prompt:$('xprompt').value.trim(),
      layer:+$('layer').value, pos:$('xpos').value===''?null:+$('xpos').value});
    else r=await post('/extract_here',{slot,layer:+$('layer').value});
    turn('', r.note||('error: '+r.error), r.error?'receipt bad':'receipt');
    if(!r.error){ $('word').value=''; $('xprompt').value=''; $('slotname').value='';
      delete $('slotname').dataset.touched; showSqueeze(); }
  }catch(err){ turn('','error: '+String(err),'receipt bad'); }
  finally{ lock(false); refresh(); }
};

$('cop').onchange=()=>{ const scale=$('cop').value==='x';
  $('cr').hidden=scale; $('cfactor').hidden=!scale;
  $('chint').textContent = scale
    ? 'Scaling does not change a steered run — the fader rescales to per cent of the residual either way. × only changes the number in the rack.'
    : 'A difference of two states the model produced is a direction in the model’s own units — no contrastive frame, no magnitude convention. Try “I am cheerful” minus “I am despairing”.';
  refreshButtons(); };
for(const id of ['cl','cr','cinto']) $(id).oninput=$(id).onchange=refreshButtons;
$('cgo').onclick=async()=>{
  if(busy) return; lock(true,'combining…');
  try{ const r=await post('/combine',{into:squeeze($('cinto').value),left:$('cl').value,
        right:$('cr').value,op:$('cop').value,weight:+$('cfactor').value||2});
       turn('', r.note||('error: '+r.error), r.error?'receipt bad':'receipt');
       if(!r.error) $('cinto').value=''; }
  catch(err){ turn('','error: '+String(err),'receipt bad'); }
  finally{ lock(false); refresh(); }
};

/* ---- session --------------------------------------------------------------------------------- */
$('undo').onclick=async()=>{
  const r=await post('/undo',{});
  if(r.error){ turn('','error: '+r.error,'receipt bad'); return; }
  const turns=[...log.querySelectorAll('.turn:not(.receipt)')];
  for(const t of turns.slice(-2)) t.remove();
  turn('',`rolled back — ${r.history} message(s) left`,'receipt'); refresh();
};
$('clear').onclick=async()=>{ await post('/clear',{}); log.innerHTML=''; allRows=[];
  closeInspector(); refresh(); };

/* ---- stage head ------------------------------------------------------------------------------ */
function setPaint(mode){
  document.body.dataset.paint=mode;
  $('paintprose').setAttribute('aria-pressed',String(mode==='prose'));
  $('paintrank').setAttribute('aria-pressed',String(mode==='rank'));
}
$('paintprose').onclick=()=>setPaint('prose');
$('paintrank').onclick=()=>setPaint('rank');
$('onlychanged').onchange=()=>{ if($('onlychanged').checked) document.body.setAttribute('data-only-changed','');
  else document.body.removeAttribute('data-only-changed'); };
$('benchtoggle').onclick=()=>{ const hid=document.body.dataset.bench==='hidden';
  document.body.dataset.bench=hid?'':'hidden'; $('benchtoggle').textContent=hid?'hide bench':'show bench'; };
$('legend').innerHTML =
  '<span>Δ</span>' + [0,0.17,0.33,0.67,0.78].map((a,i)=>
    `<span class=swatch style="background:rgba(var(--heat-rgb),${a})"></span>`).join('')
  + '<span>0 · 1 · 2 · 4 · 8 nats</span>'
  + '<span style="margin-left:10px">rank</span>'
  + [1,2,3].map(px=>`<span class=swatch style="background:none;border-bottom:${px}px solid var(--dim)"></span>`).join('')
  + '<span>1–10 · 11–1k · &gt;1k</span>';

/* ---- keyboard -------------------------------------------------------------------------------- */
function grow(){ const t=$('msg'); t.style.height='auto'; t.style.height=Math.min(t.scrollHeight,window.innerHeight*0.3)+'px'; }
$('msg').oninput=()=>{grow(); refreshButtons();};
$('msg').onkeydown=e=>{
  if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); $('composer').requestSubmit(); }
  else if(e.key==='ArrowUp'&&!$('msg').value&&lastSent){ e.preventDefault(); $('msg').value=lastSent; grow(); refreshButtons(); }
};
for(const id of ['word','xprompt','slotname']) $(id).onkeydown=e=>{ if(e.key==='Enter'){e.preventDefault(); if(!$('mk').disabled) $('mk').click();} };
$('cinto').onkeydown=e=>{ if(e.key==='Enter'){e.preventDefault(); if(!$('cgo').disabled) $('cgo').click();} };
document.addEventListener('keydown',e=>{
  const typing=/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
  if(e.key==='Escape'){ closeInspector(); return; }
  if((e.metaKey||e.ctrlKey)&&e.key==='\\'){ e.preventDefault(); $('benchtoggle').click(); return; }
  if(typing) return;
  if(e.key==='/'){ e.preventDefault(); $('msg').focus(); }
  else if(e.key==='\\'){ e.preventDefault();
    const v=+$('fader').value; $('fader').dataset.was = v>0?v:($('fader').dataset.was||40);
    $('fader').value = v>0 ? 0 : +$('fader').dataset.was; faderLook(); syncDesk(true); }
  else if(e.key==='['||e.key===']'){ e.preventDefault();
    const d=(e.key===']'?1:-1)*(e.shiftKey?25:5);
    $('fader').value=Math.max(0,Math.min(250,+$('fader').value+d)); faderLook(); syncDesk(true); }
  else if((e.key==='ArrowLeft'||e.key==='ArrowRight')&&$('inspector').hasAttribute('data-open')){
    e.preventDefault(); step(allRows, e.key==='ArrowRight'?1:-1); }
  else if((e.metaKey||e.ctrlKey)&&e.key==='z'){ e.preventDefault(); if(!$('undo').disabled) $('undo').click(); }
});

setPaint('rank'); setSrc('word'); grow(); refresh();
</script>
"""


def _model_name(path: str) -> str:
    """The model's name out of a hub path, rather than the snapshot hash the path ends with.

    A hub snapshot lives at ``models--google--gemma-4-31b-it/snapshots/<40 hex>``, so the last path
    component is a hash. The name is in the ``models--`` directory.
    """
    for part in reversed(str(path).split("/")):
        if part.startswith("models--"):
            return part.split("--")[-1]
    return str(path).split("/")[-1] or "model"


class Service:
    """The model, behind one lock, because there is one card."""

    def __init__(self, chat: Chat, max_tokens: int, log_path: Path | None = None):
        self.chat, self.max_tokens = chat, max_tokens
        self.lock = threading.Lock()
        self.log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, kind: str, **fields) -> None:
        """Append one event. On by default: a session that is not written down did not happen."""
        if self.log_path is None:
            return
        desk = self.chat.desk
        row = {"t": time.time(), "kind": kind,
               "desk": {"slot": desk.slot, "layer": desk.layer,
                        "percent": desk.percent, "scope": desk.scope},
               **fields}
        with open(self.log_path, "a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def state(self) -> dict:
        c = self.chat
        return {"model": _model_name(getattr(c.model.config, "_name_or_path", "")),
                "layers": c.view.num_layers, "hidden": c.view.hidden_size,
                "device": str(c.device), "dtype": str(next(c.model.parameters()).dtype),
                "max_tokens": self.max_tokens,
                "log": {"path": str(self.log_path) if self.log_path else None,
                        "enabled": self.log_path is not None},
                "history": len(c.history), "slots": sorted(c.slots),
                "thinking": c.thinking, "supports_thinking": c.supports_thinking,
                # The rack shows where a vector came from, not only how long it is: a residual and
                # a concept of the same norm steer differently and a slot name does not say which.
                "detail": [{"name": n, "kind": e["kind"], "layer": e["layer"], "norm": e["norm"],
                            "word": e.get("word"), "prompt": e.get("prompt"),
                            "position": e.get("position")}
                           for n, e in sorted(c.slots.items())],
                "desk": {"slot": c.desk.slot, "layer": c.desk.layer,
                         "percent": c.desk.percent, "scope": c.desk.scope,
                         "live": c.desk.live}}

    def set_desk(self, body: dict) -> dict:
        desk = self.chat.desk
        if "slot" in body:
            desk.slot = self._resolve_slot(body.get("slot"))
        if "layer" in body:
            desk.layer = int(body.get("layer") or desk.layer or 1)
        if "percent" in body:
            desk.percent = float(body.get("percent") or 0)
        if "scope" in body:
            desk.scope = body.get("scope") or "turn"
        # A fader at zero is clean -- `Desk.live` reads the fader, so nothing here has to null the
        # slot to make that true, and the rack goes on showing what is patched in.
        return self.state()

    def _resolve_slot(self, name):
        """The slot this name means, or None. Concept slots are stored under a squeezed name.

        `/concept butt holes` is filed as `buttholes`, so a desk set from the same words the user
        typed would miss its own vector and run clean without saying so.
        """
        if not name:
            return None
        if name in self.chat.slots:
            return name
        squeezed = "".join(ch for ch in str(name).lower() if ch.isalnum())[:12]
        return squeezed if squeezed in self.chat.slots else None

    def _token_rows(self, prompt_ids: list[int], emitted: list[int], reply: str) -> list[dict]:
        """Per-token evidence, each row tagged with the channel it fell in.

        The channel tag is what makes this interesting on a reasoning model: the steering may bite
        hard in the thought and wash out by the answer, or the reverse, and a single painted strip
        over the whole reply would hide that.

        Both the prompt and the reply come in as the ids the steered run actually used. Rendering
        the prompt again or re-tokenising the decoded reply would replay a token sequence the
        steered run never visited, and a rank measured against that context is evidence for
        nothing.
        """
        if not emitted:
            return []
        rows = self.chat.clean_replay(prompt_ids, emitted)
        # A reply cut off mid-thought carries the open marker and no close: it is all thought.
        out, phase = [], ("thought" if opens_thought(reply) else "answer")
        for r in rows:
            token = r["steered_token"]
            out.append({"tok": token, "top": r["clean_top"], "rank": r["clean_rank_of_steered"],
                        "d": r["logprob_steered_token"] - r["logprob_clean_top"], "phase": phase})
            if THOUGHT_CLOSE in token:
                phase = "answer"
        return out

    def desk_snapshot(self) -> dict:
        """What the desk was when a run STARTED.

        `set_desk` does not take the lock, so the fader can move while a reply is generating. Read
        after the fact, a reply that was steered comes back described as clean and loses its token
        strip, and the transcript records the percentage the user happened to land on rather than
        the one that ran.
        """
        d = self.chat.desk
        return {"slot": d.slot, "layer": d.layer, "percent": d.percent,
                "scope": d.scope, "live": d.live, "label": d.label()}

    def say(self, text: str) -> dict:
        with self.lock:
            chat = self.chat
            desk = self.desk_snapshot()
            messages = chat.history + [{"role": "user", "content": text}]
            reply, emitted, prompt_ids, note, seconds, hook = chat.speak(messages, self.max_tokens)
            chat.last_turn = {"prompt_ids": prompt_ids, "emitted": emitted}
            rows = self._token_rows(prompt_ids, emitted, reply) if desk["live"] else []
            chat.history = messages + [{"role": "assistant", "content": reply}]
            return self._turn_payload(text, reply, emitted, prompt_ids, rows, note,
                                      seconds, desk, hook)

    def _turn_payload(self, text, reply, emitted, prompt_ids, rows, note, seconds, desk, hook):
        """What a finished turn is, recorded and returned. One builder for both paths."""
        changed = sum(1 for r in rows if r["rank"])
        summary = f"{note} · {len(emitted)} tok · {seconds:.1f}s"
        if rows:
            summary += f" · {changed} of {len(rows)} tokens were not the clean model's first choice"
        thought, answer = split_thought(reply)
        if thought and rows:
            thought_rows = [r for r in rows if r["phase"] == "thought"]
            bit = sum(1 for r in thought_rows if r["rank"])
            summary += f" · thought {bit}/{len(thought_rows)} changed"
            answer_rows = len(rows) - len(thought_rows)
            if answer_rows:
                summary += f", answer {changed - bit}/{answer_rows}"
            else:
                summary += ", answer not reached (the token cap cut the thought off)"
        self.record("turn", user=text, thought=thought, reply=answer, note=summary,
                    desk=desk, hook=hook.report() if hook else None,
                    changed=changed, total=len(rows), seconds=seconds, tokens=rows)
        return {"reply": answer.strip(), "thought": thought.strip(),
                "note": summary, "tokens": rows, "desk": desk,
                "hook": hook.report() if hook else None,
                "seconds": seconds, "emitted": len(emitted),
                "prompt_tokens": len(prompt_ids), "truncated": bool(thought) and not answer}

    def say_stream(self, text: str):
        """The same turn as `say`, yielded as it happens.

        A reply takes ten to thirty seconds and the page showed nothing at all until it was over.
        The events are newline-delimited JSON: `tok` as each token is produced, then one `done`
        carrying exactly the payload `say` would have returned. The clean replay cannot be streamed
        -- it is a second pass over the finished reply -- so the painting arrives with `done` and
        the page repaints what it has already shown.
        """
        with self.lock:
            chat = self.chat
            desk = self.desk_snapshot()
            messages = chat.history + [{"role": "user", "content": text}]
            queue: list[dict] = []
            yield {"t": "start", "desk": desk, "cap": self.max_tokens}
            # The generation loop is synchronous, so tokens are collected by the callback and
            # flushed between forward passes rather than yielded from inside it.
            def collect(index, _id, piece):
                queue.append({"t": "tok", "i": index, "s": piece})

            import threading
            done: dict = {}

            def run():
                try:
                    done["result"] = chat.speak(messages, self.max_tokens, on_token=collect)
                except Exception as exc:  # surfaced to the page rather than dying silently
                    done["error"] = f"{type(exc).__name__}: {exc}"

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            while worker.is_alive() or queue:
                while queue:
                    yield queue.pop(0)
                if worker.is_alive():
                    worker.join(0.05)
            if "error" in done:
                yield {"t": "error", "error": done["error"]}
                return
            reply, emitted, prompt_ids, note, seconds, hook = done["result"]
            chat.last_turn = {"prompt_ids": prompt_ids, "emitted": emitted}
            yield {"t": "replaying", "tokens": len(emitted)}
            rows = self._token_rows(prompt_ids, emitted, reply) if desk["live"] else []
            chat.history = messages + [{"role": "assistant", "content": reply}]
            payload = self._turn_payload(text, reply, emitted, prompt_ids, rows, note,
                                         seconds, desk, hook)
            yield dict(payload, t="done")

    def ab(self) -> dict:
        with self.lock:
            chat = self.chat
            desk = self.desk_snapshot()
            base, text = list(chat.history), None
            for index in range(len(base) - 1, -1, -1):
                if base[index]["role"] == "user":
                    text, base = base[index]["content"], base[:index]
                    break
            if text is None:
                return {"error": "nothing to re-run — say something first"}
            messages = base + [{"role": "user", "content": text}]
            clean, *_a = chat.speak(messages, self.max_tokens, steered=False)
            steered, emitted, prompt_ids, note, seconds, hook = chat.speak(
                messages, self.max_tokens, steered=True)
            rows = self._token_rows(prompt_ids, emitted, steered) if desk["live"] else []
            clean_thought, clean_answer = split_thought(clean)
            steered_thought, steered_answer = split_thought(steered)
            # An arm whose thought the cap cut off has no answer, and "(nothing)" then reads as the
            # model declining to speak rather than as the budget running out.
            clean_cut = bool(clean_thought) and not clean_answer
            steered_cut = bool(steered_thought) and not steered_answer
            result = {"clean": clean_answer.strip(), "steered": steered_answer.strip(),
                      "clean_thought": clean_thought.strip(),
                      "steered_thought": steered_thought.strip(), "note": note,
                      "clean_cut": clean_cut, "steered_cut": steered_cut,
                      "cap": self.max_tokens,
                      "history": len(base), "tokens": rows, "desk": desk,
                      "hook": hook.report() if hook else None, "seconds": seconds,
                      "emitted": len(emitted), "prompt_tokens": len(prompt_ids),
                      # Neither arm is committed. A/B is a probe, and the page has to say so.
                      "committed": False,
                      "same": clean.strip() == steered.strip()}
            self.record("ab", user=text, **{k: v for k, v in result.items() if k != "tokens"},
                        tokens=rows)
            return result

    def drop(self, slot: str) -> dict:
        """Forget a vector, and disarm the desk if that is the vector it was pointing at.

        Dropping the armed slot used to leave `desk.slot` naming a key that no longer exists, and
        the next message died in `build_injection` on the lookup.
        """
        with self.lock:
            self.chat.slots.pop(slot, None)
            if self.chat.desk.slot == slot:
                self.chat.desk.slot = None
            return self.state()

    def set_thinking(self, on: bool) -> dict:
        with self.lock:
            self.chat.thinking = on
            return self.state()

    def undo(self) -> dict:
        with self.lock:
            # history[:-2] on a one-message history empties it; say so rather than doing it.
            if len(self.chat.history) < 2:
                return dict(self.state(), error="nothing to roll back yet")
            self.chat.history = self.chat.history[:-2]
            self.chat.last_turn = None
            return self.state()

    def clear(self) -> dict:
        with self.lock:
            self.chat.history = []
            self.chat.last_turn = None
            return self.state()

    def _captured(self, call) -> str:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            call()
        return buffer.getvalue().strip()

    def extract(self, slot: str, prompt: str, layer: int, position: int | None) -> dict:
        """An arbitrary prompt in, a residual row out, into a named slot."""
        with self.lock:
            note = self._captured(lambda: self.chat.cmd_extract(slot, layer, position, prompt))
            self.record("extract", slot=slot, prompt=prompt, layer=layer, note=note)
            return {"slot": slot, "note": note}

    def extract_here(self, slot: str, layer: int) -> dict:
        """The residual at the live conversation's own last position.

        This is the move a single-shot console cannot make: a moment in an actual exchange becomes
        a steering vector, rather than a vector built from a synthetic carrier sentence.
        """
        with self.lock:
            chat = self.chat
            if not chat.history:
                return {"error": "no conversation yet — say something first"}
            ids = chat.render_chat(chat.history)
            vector = chat.residual_at(ids, layer, None)
            chat.slots[slot] = {"vector": vector, "layer": layer, "kind": "from chat",
                                "norm": float(vector.norm()),
                                "position": len(ids) - 1,
                                "prompt": f"<live conversation, {len(chat.history)} message(s)>"}
            note = (f"{slot}: residual at L{layer} from the live conversation "
                    f"(position {len(ids) - 1}), norm {float(vector.norm()):,.0f}")
            self.record("extract_here", slot=slot, layer=layer, position=len(ids) - 1,
                        history=len(chat.history), norm=float(vector.norm()), note=note)
            return {"slot": slot, "note": note}

    def combine(self, into: str, left: str, right: str | None, op: str, weight: float) -> dict:
        """Build a vector out of other vectors: a difference, a sum, or a scaling.

        A *difference of two states the model itself produced* is the construction this repo's
        confidence work settled on as the honest one: it needs no contrastive frame and no magnitude
        convention, because it is already scaled in units the model generates. `cheerful − despairing`
        is a direction built entirely from things that actually happened inside the model.
        """
        with self.lock:
            slots = self.chat.slots
            if left not in slots:
                return {"error": f"no slot '{left}'"}
            if op in {"-", "+"} and (right is None or right not in slots):
                return {"error": f"no slot '{right}'"}
            a = slots[left]
            if op in {"-", "+"}:
                b = slots[right]
                if a["layer"] != b["layer"]:
                    return {"error": f"'{left}' is at layer {a['layer']} and '{right}' at "
                                     f"{b['layer']} — a difference across layers is not a direction"}
                if a["vector"].shape != b["vector"].shape:
                    return {"error": "those vectors are different widths — different models?"}
                vector = (a["vector"] - b["vector"]) if op == "-" else (a["vector"] + b["vector"])
                kind = f"{left} {op} {right}"
            else:
                vector = a["vector"] * weight
                kind = f"{left} x {weight:g}"
            if float(vector.norm()) < 1e-6:
                return {"error": f"'{into}' would be the zero vector — a direction of length zero "
                                 "cannot steer, and the fader would divide by it"}
            slots[into] = {"vector": vector, "layer": a["layer"], "kind": kind,
                           "norm": float(vector.norm()), "prompt": kind}
            share = float(vector.norm()) / max(float(a["vector"].norm()), 1e-9)
            note = (f"{into} = {kind} at L{a['layer']}, norm {float(vector.norm()):,.0f} "
                    f"({share:.0%} of '{left}')")
            self.record("combine", slot=into, expression=kind, layer=a["layer"],
                        norm=float(vector.norm()), note=note)
            return {"slot": into, "note": note}

    def concept(self, word: str, layer: int) -> dict:
        with self.lock:
            slot = "".join(ch for ch in word.lower() if ch.isalnum())[:12] or "concept"
            # The squeeze is lossy: `self-awareness` and `self awareness` file under one name, so
            # the second silently replaced the first after spending twenty-five forward passes on
            # it. Refuse instead, and say what is in the way.
            if slot in self.chat.slots:
                held = self.chat.slots[slot]
                return {"error": f"slot '{slot}' already holds {held['kind']} at L{held['layer']} "
                                 f"— drop it first, or use a different word"}
            import io
            import contextlib

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.chat.cmd_concept(slot, layer, word)
            note = buffer.getvalue().strip()
            self.record("concept", slot=slot, word=word, layer=layer, note=note)
            return {"slot": slot, "note": note}


def handler_for(service: Service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # the console is for the model, not for access logs

        def _stream(self, events):
            """Newline-delimited JSON, flushed per event, so the page sees tokens as they land."""
            self.send_response(200)
            self.send_header("content-type", "application/x-ndjson")
            self.send_header("cache-control", "no-store")
            self.send_header("x-accel-buffering", "no")
            self.end_headers()
            try:
                for event in events:
                    self.wfile.write(json.dumps(event).encode() + b"\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # the tab was closed or reloaded mid-reply; the generation finishes either way
                pass

        def _send(self, payload, status=200, content="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", content)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/state"):
                self._send(service.state())
            else:
                self._send(PAGE.encode(), content="text/html; charset=utf-8")

        def do_POST(self):
            length = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                if self.path == "/say":
                    self._send(service.say(body.get("text", "")))
                elif self.path == "/say_stream":
                    self._stream(service.say_stream(body.get("text", "")))
                elif self.path == "/ab":
                    self._send(service.ab())
                elif self.path == "/desk":
                    self._send(service.set_desk(body))
                elif self.path == "/extract":
                    self._send(service.extract(body.get("slot", ""), body.get("prompt", ""),
                                               int(body.get("layer", 1)), body.get("pos")))
                elif self.path == "/extract_here":
                    self._send(service.extract_here(body.get("slot", ""), int(body.get("layer", 1))))
                elif self.path == "/combine":
                    self._send(service.combine(body.get("into", ""), body.get("left", ""),
                                               body.get("right"), body.get("op", "-"),
                                               float(body.get("weight", 1.0))))
                elif self.path == "/drop":
                    self._send(service.drop(body.get("slot", "")))
                elif self.path == "/concept":
                    self._send(service.concept(body.get("word", ""), int(body.get("layer", 1))))
                elif self.path == "/thinking":
                    self._send(service.set_thinking(bool(body.get("on"))))
                elif self.path == "/undo":
                    self._send(service.undo())
                elif self.path == "/clear":
                    self._send(service.clear())
                else:
                    self._send({"error": "no such endpoint"}, status=404)
            except Exception as exc:
                traceback.print_exc()
                self._send({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--port", type=int, default=8765)
    # A reasoning model spends its budget on the thought before it writes a word of the answer.
    # 200 cut most replies off mid-thought; 512 still did on anything with history behind it, and
    # the A/B card then showed two empty arms with no explanation.
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concept-baseline", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", type=Path, default=None,
                        help="JSONL transcript; defaults to sessions/<timestamp>.jsonl beside this "
                             "script's working directory. Pass 'none' to disable.")
    args = parser.parse_args(argv)

    if args.log is None:
        stamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
        args.log = Path("sessions") / f"steer-{stamp}.jsonl"
    log_path = None if str(args.log) == "none" else args.log

    chat = Chat(args.model, device=args.device, dtype=args.dtype,
                baseline_words=args.concept_baseline, seed=args.seed)
    service = Service(chat, args.max_tokens, log_path)
    if log_path:
        print(f"transcript: {log_path.resolve()}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(service))
    print(f"\nserving on http://127.0.0.1:{args.port} (localhost only — reach it with)\n"
          f"    ssh -N -L {args.port}:localhost:{args.port} rtx6000\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
