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
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE))

from steer_chat import Chat  # noqa: E402

PAGE = """<!doctype html><meta charset=utf-8><title>steering desk</title>
<style>
:root{--bg:#14161a;--panel:#1c1f26;--line:#2b2f3a;--ink:#e6e8ee;--dim:#8b93a7;--hot:#ff6b5a;--cool:#5ac8fa}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;display:grid;grid-template-columns:1fr 300px;height:100vh}
main{display:flex;flex-direction:column;min-width:0}
#log{flex:1;overflow-y:auto;padding:20px 24px}
.turn{margin-bottom:22px;max-width:70ch}
.who{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin-bottom:5px}
.you .who{color:var(--cool)}
.body{white-space:pre-wrap;word-wrap:break-word}
.meta{margin-top:7px;font-size:12px;color:var(--dim);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.tokens{margin-top:10px;padding:10px;background:var(--panel);border:1px solid var(--line);border-radius:8px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;line-height:2;overflow-x:auto}
.tok{padding:2px 1px;border-radius:3px;cursor:default;white-space:pre}
.tok:hover{outline:1px solid var(--dim)}
form{display:flex;gap:10px;padding:16px 24px;border-top:1px solid var(--line)}
input[type=text]{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:8px;color:var(--ink);padding:11px 14px;font:inherit}
button{background:var(--panel);border:1px solid var(--line);border-radius:8px;color:var(--ink);padding:11px 16px;font:inherit;cursor:pointer}
button:hover:not(:disabled){border-color:var(--dim)}
button:disabled{opacity:.45;cursor:default}
aside{background:var(--panel);border-left:1px solid var(--line);padding:20px;overflow-y:auto}
aside h2{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);margin:22px 0 9px;font-weight:600}
aside h2:first-child{margin-top:0}
label{display:block;font-size:12px;color:var(--dim);margin:9px 0 3px}
select,input[type=number],input[type=range]{width:100%;background:var(--bg);border:1px solid var(--line);border-radius:6px;color:var(--ink);padding:7px;font:inherit}
input[type=range]{padding:0}
.reading{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:22px;text-align:center;margin:6px 0}
.hint{font-size:11px;color:var(--dim);margin-top:5px}
.row{display:flex;gap:8px}.row>*{flex:1}
#state{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:var(--dim);white-space:pre-wrap}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;border:1px solid var(--line);font-size:11px;font-family:ui-monospace,monospace}
</style>
<main>
  <div id=log></div>
  <form id=f>
    <input type=text id=msg placeholder="say something to the model" autocomplete=off autofocus>
    <button id=send>send</button>
    <button id=ab type=button title="run the last message clean and steered on identical history">A/B</button>
  </form>
</main>
<aside>
  <h2>desk</h2>
  <label>vector</label><select id=slot></select>
  <label>layer</label><input type=number id=layer min=1 value=32>
  <label>fader &mdash; per cent of the residual norm at the site</label>
  <div class=reading><span id=pct>0</span>%</div>
  <input type=range id=fader min=0 max=250 step=5 value=0>
  <div class=hint>0 is clean. On the 12B at L32 a concept flips the reply into another language near 120%; a sustained injection collapses it near 30%.</div>
  <label>scope</label><select id=scope>
    <option value=turn>turn &mdash; over your message only</option>
    <option value=reply>reply &mdash; while generating</option>
    <option value=both>both &mdash; the paper's protocol</option>
    <option value=all>all &mdash; the whole context</option>
  </select>
  <h2>build a vector</h2>
  <div class=row><input type=text id=word placeholder="word"><button id=mk type=button>concept</button></div>
  <div class=hint>&ldquo;Tell me about {word}.&rdquo; minus the mean over 24 random words, at the layer above.</div>
  <h2>state</h2><div id=state></div>
  <h2></h2><div class=row><button id=undo type=button>undo</button><button id=clear type=button>clear</button></div>
</aside>
<script>
const $=id=>document.getElementById(id), log=$('log');
let busy=false;
const post=(path,body)=>fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body||{})}).then(r=>r.json());

function paint(rows){
  if(!rows||!rows.length) return '';
  const d=document.createElement('div'); d.className='tokens';
  for(const r of rows){
    const s=document.createElement('span'); s.className='tok'; s.textContent=r.tok;
    // saturation tracks how far the clean model was from choosing this token
    const heat=Math.min(1,Math.log10(1+r.rank)/4.2);
    if(r.rank>0){ s.style.background=`rgba(255,107,90,${0.10+0.62*heat})`; }
    s.title = r.rank===0 ? 'the clean model wanted this too'
      : `clean model wanted ${JSON.stringify(r.top)} — this token ranked ${r.rank.toLocaleString()}, Δlogprob ${r.d.toFixed(2)}`;
    d.appendChild(s);
  }
  return d;
}
function turn(who,text,meta,rows){
  const w=document.createElement('div'); w.className='turn '+(who==='you'?'you':'');
  w.innerHTML=`<div class=who>${who}</div><div class=body></div>`;
  w.querySelector('.body').textContent=text;
  if(meta){const m=document.createElement('div');m.className='meta';m.textContent=meta;w.appendChild(m);}
  if(rows&&rows.length){const p=paint(rows); if(p) w.appendChild(p);}
  log.appendChild(w); log.scrollTop=log.scrollHeight; return w;
}
function lock(on){busy=on;for(const b of ['send','ab','mk','undo','clear'])$(b).disabled=on;}
async function refresh(){
  const s=await fetch('/state').then(r=>r.json());
  $('slot').innerHTML='<option value="">(none)</option>'+s.slots.map(x=>`<option${x===s.desk.slot?' selected':''}>${x}</option>`).join('');
  $('state').textContent=`${s.model}\\n${s.layers} layers · hidden ${s.hidden}\\ntwo-thirds depth ≈ L${Math.round(s.layers*2/3)}\\n${s.history} message(s)\\n${s.slots.length} vector(s)`;
}
async function sync(){
  await post('/desk',{slot:$('slot').value||null,layer:+$('layer').value,percent:+$('fader').value,scope:$('scope').value});
}
$('fader').oninput=()=>{$('pct').textContent=$('fader').value; sync();};
for(const id of ['slot','layer','scope']) $(id).onchange=sync;
$('f').onsubmit=async e=>{
  e.preventDefault(); if(busy) return;
  const text=$('msg').value.trim(); if(!text) return;
  $('msg').value=''; turn('you',text); lock(true);
  try{const r=await post('/say',{text}); turn('model',r.reply,r.note,r.tokens);}
  catch(err){turn('error',String(err));} finally{lock(false); refresh();}
};
$('ab').onclick=async()=>{
  if(busy) return; lock(true);
  try{
    const r=await post('/ab',{});
    if(r.error){turn('error',r.error);}
    else{ turn('A — clean',r.clean,`on ${r.history} message(s) of history`);
          turn('B — steered',r.steered,r.note,r.tokens);
          turn('','',r.same?'identical — the desk changed nothing':'they differ'); }
  }catch(err){turn('error',String(err));} finally{lock(false);}
};
$('mk').onclick=async()=>{
  const w=$('word').value.trim(); if(!w||busy) return; lock(true);
  try{const r=await post('/concept',{word:w,layer:+$('layer').value}); turn('desk',r.note);}
  catch(err){turn('error',String(err));} finally{lock(false); refresh();}
};
$('undo').onclick=async()=>{await post('/undo',{}); turn('desk','rolled back one exchange'); refresh();};
$('clear').onclick=async()=>{await post('/clear',{}); log.innerHTML=''; refresh();};
refresh();
</script>
"""


class Service:
    """The model, behind one lock, because there is one card."""

    def __init__(self, chat: Chat, max_tokens: int):
        self.chat, self.max_tokens = chat, max_tokens
        self.lock = threading.Lock()

    def state(self) -> dict:
        c = self.chat
        return {"model": c.model.config._name_or_path.split("/")[-1] if hasattr(c.model.config, "_name_or_path") else "model",
                "layers": c.view.num_layers, "hidden": c.view.hidden_size,
                "history": len(c.history), "slots": sorted(c.slots),
                "desk": {"slot": c.desk.slot, "layer": c.desk.layer,
                         "percent": c.desk.percent, "scope": c.desk.scope}}

    def set_desk(self, body: dict) -> dict:
        desk = self.chat.desk
        desk.slot = body.get("slot") or None
        if desk.slot and desk.slot not in self.chat.slots:
            desk.slot = None
        desk.layer = int(body.get("layer") or desk.layer or 1)
        desk.percent = float(body.get("percent") or 0)
        desk.scope = body.get("scope") or "turn"
        if desk.percent <= 0:
            desk.slot = None  # a fader at zero is clean, whatever is patched into it
        return self.state()

    def _token_rows(self, messages: list[dict], reply: str) -> list[dict]:
        """Per-token evidence: where the clean model ranked each token the steered run emitted."""
        ids = self.chat.render_chat(messages)
        emitted = self.chat.tokenizer(reply, add_special_tokens=False)["input_ids"]
        if not emitted:
            return []
        rows = self.chat.clean_replay(ids, emitted)
        return [{"tok": r["steered_token"], "top": r["clean_top"],
                 "rank": r["clean_rank_of_steered"],
                 "d": r["logprob_steered_token"] - r["logprob_clean_top"]} for r in rows]

    def say(self, text: str) -> dict:
        with self.lock:
            chat = self.chat
            messages = chat.history + [{"role": "user", "content": text}]
            reply, emitted, _ids, note, seconds = chat.speak(messages, self.max_tokens)
            rows = self._token_rows(messages, reply) if chat.desk.live else []
            chat.history = messages + [{"role": "assistant", "content": reply}]
            changed = sum(1 for r in rows if r["rank"])
            summary = f"{note} · {len(emitted)} tok · {seconds:.1f}s"
            if rows:
                summary += f" · {changed} of {len(rows)} tokens were not the clean model's first choice"
            return {"reply": reply.strip(), "note": summary, "tokens": rows}

    def ab(self) -> dict:
        with self.lock:
            chat = self.chat
            base, text = list(chat.history), None
            for index in range(len(base) - 1, -1, -1):
                if base[index]["role"] == "user":
                    text, base = base[index]["content"], base[:index]
                    break
            if text is None:
                return {"error": "nothing to re-run — say something first"}
            messages = base + [{"role": "user", "content": text}]
            clean, _a, _b, _c, _d = chat.speak(messages, self.max_tokens, steered=False)
            steered, _e, _f, note, _g = chat.speak(messages, self.max_tokens, steered=True)
            rows = self._token_rows(messages, steered) if chat.desk.live else []
            return {"clean": clean.strip(), "steered": steered.strip(), "note": note,
                    "history": len(base), "tokens": rows,
                    "same": clean.strip() == steered.strip()}

    def concept(self, word: str, layer: int) -> dict:
        with self.lock:
            slot = "".join(ch for ch in word.lower() if ch.isalnum())[:12] or "concept"
            import io
            import contextlib

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.chat.cmd_concept(slot, layer, word)
            return {"slot": slot, "note": buffer.getvalue().strip()}


def handler_for(service: Service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # the console is for the model, not for access logs

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
                elif self.path == "/ab":
                    self._send(service.ab())
                elif self.path == "/desk":
                    self._send(service.set_desk(body))
                elif self.path == "/concept":
                    self._send(service.concept(body.get("word", ""), int(body.get("layer", 1))))
                elif self.path == "/undo":
                    service.chat.history = service.chat.history[:-2]
                    self._send(service.state())
                elif self.path == "/clear":
                    service.chat.history = []
                    self._send(service.state())
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
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--concept-baseline", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    chat = Chat(args.model, device=args.device, dtype=args.dtype,
                baseline_words=args.concept_baseline, seed=args.seed)
    service = Service(chat, args.max_tokens)
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
