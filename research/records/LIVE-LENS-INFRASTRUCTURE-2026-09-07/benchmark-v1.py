"""Per-mode fixed-history timing; preserves all timings and native-equivalence evidence."""
from pathlib import Path
from contextlib import nullcontext
import json
import statistics
import time
import numpy as np
PRIMARY=Path('/Users/daniel.tipton/Desktop/An app')
ROOT=Path('/private/tmp/codex-live-lens-01a079ee')
OUT=Path(__file__).parent
from local_llm_lab import runlock
runlock.PROJECT_ROOT=PRIMARY
runlock.hold_model_run_lock(session='Codex live-lens benchmark',command=__file__)
import mlx.core as mx
from local_llm_lab.arch import ArchitectureView, NativeCapture
from local_llm_lab.pipeline.live_lens.instruments import LensMaps, read_band
from local_llm_lab.pipeline.live_lens.session import CaptureSession,LensReadout,RecordWriter,_logit_hash
mx.set_cache_limit(2*2**30); mx.set_memory_limit(12*2**30)
model,tokenizer=runlock.load_weights('/Users/daniel.tipton/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3')
model.eval(); view=ArchitectureView.from_model(model)
band=read_band(ROOT/'configs/models/qwen35-4b.yaml',[view.layer_kind(i) for i in range(view.num_layers)])
layers=tuple(sorted({p for pair in band for p in pair}))
blocks=tuple(p-1 for pair in band for p in pair if view.layer_kind(p-1)=='attention')
lens=LensMaps.load(PRIMARY/'models/jlens/Qwen3.5-4B_jacobian_lens_n1000.npz',expected_sha256='381c089dcffead8147ee91f944496f468cce2c7d593e0a1b17230745055aea12',hidden_size=view.hidden_size,num_layers=view.num_layers)
reader=LensReadout(view,lens)
protocol=json.loads((OUT/'benchmark-protocol.json').read_text()); prompt=protocol['prefix']
ids=list(tokenizer.encode(prompt,add_special_tokens=tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)))
cache=view.make_cache(); chunks=[ids]; hashes=[]
for i in range(9):
 out=model(mx.array([chunks[-1]]),cache=cache); mx.eval(out)
 hashes.append(_logit_hash(out))
 if i<8: chunks.append([int(mx.argmax(out[0,-1]).item())])
class Sink:
 def __init__(self,mode): self.mode=mode
 def residual(self,layer,offset,h):
  mx.eval(h)
  if self.mode=='lens' and layer in layers:
   p=reader(h[0,-1],layer)
   np.argsort(-p,kind='stable')
 def attention(self,block,target,weights,written,total):
  mx.eval(weights)
  if written is not None: mx.eval(written)
 def output(self,*args): pass

def measure(mode,trial):
 mx.clear_cache(); mx.reset_peak_memory()
 cache=view.make_cache()
 sink=Sink(mode)
 session=None
 with (RecordWriter(OUT/f'benchmark-{mode}-{trial}.jsonl',{'mode':mode,'trial':trial}) if mode=='full_record' else nullcontext()) as write:
  if mode=='full_record':
   session=CaptureSession(view,reader,write,layers=layers,attention_blocks=blocks,top_k=10,audit_modulus=23,audit_seed=20260907)
   context=session.generation(model,tokenizer,prompt,turn_cache=None)
  elif mode=='native': context=nullcontext(model)
  else: context=NativeCapture(view,sink,layers=layers,attention_blocks=blocks if mode in {'attention','head_vectors'} else (),head_vectors=mode=='head_vectors')
  t=time.monotonic()
  with context as runner:
   actual=[]
   for i,chunk in enumerate(chunks):
    out=runner(mx.array([chunk]),cache=cache); mx.eval(out)
    actual.append(_logit_hash(out))
    if session is not None and i: session.emitted(chunk[0])
  elapsed=time.monotonic()-t
 assert actual==hashes, f'{mode} changed native outputs'
 return {'mode':mode,'trial':trial,'seconds':elapsed,'peak_memory_gib':mx.get_peak_memory()/2**30,'hashes_identical':True}

rows=[]
for mode in protocol['modes']: measure(mode,'warmup')
for trial in range(3):
 order=protocol['modes'][trial:]+protocol['modes'][:trial]
 for mode in order:
  row=measure(mode,trial); rows.append(row)
  print(json.dumps(row),flush=True)
median={mode:statistics.median(r['seconds'] for r in rows if r['mode']==mode) for mode in protocol['modes']}
summary={mode:{'median_seconds':value,'ratio_to_native':value/median['native']} for mode,value in median.items()}
result={'protocol':protocol,'prefix_tokens':len(ids),'decode_forwards':8,'rows':rows,'summary':summary}
with (OUT/'benchmark-v1.json').open('x') as f: json.dump(result,f,indent=2)
print(json.dumps({'summary':summary}),flush=True)
