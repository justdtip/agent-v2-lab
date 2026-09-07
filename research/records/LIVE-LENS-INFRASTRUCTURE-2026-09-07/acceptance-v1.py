"""Bounded native fidelity check, 2026-09-07. Uses the primary checkout's model lock."""

# RECORD, NOT A LAUNCHER (2026-09-08, issue 89). Kept so the result can be read; it takes the
# primary model-run lock and loads the checkpoint at module scope. Runs go through the package
# entry points under R47 and R48, in a window the Director declares, and take the model-run lock
# of issue 83.
import sys as _sys
if __name__ == "__main__" and "--i-am-a-record" not in _sys.argv:
    _sys.exit(
        "refusing to run: this file is a record of the 2026-09-07 live-lens fidelity run, "
        "not a launcher; runs go through the package entry points under R47 and R48, in a "
        "window the Director declares, and take the model-run lock of issue 83"
    )
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import time
import traceback
import numpy as np

PRIMARY = Path('/Users/daniel.tipton/Desktop/An app')
ROOT = Path('/private/tmp/codex-live-lens-01a079ee')
OUT = Path(__file__).parent
from local_llm_lab import runlock
# This linked worktree shares one physical machine and the primary model-run lock.
runlock.PROJECT_ROOT = PRIMARY
runlock.hold_model_run_lock(session='Codex live-lens fidelity', command=__file__)
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler
from local_llm_lab.arch import ArchitectureView, NativeCapture
from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.runner import run_task
from local_llm_lab.pipeline.tasks import make_tasks
from local_llm_lab.pipeline.live_lens.instruments import LensMaps, read_band, file_sha256
from local_llm_lab.pipeline.live_lens.session import CaptureSession, LensReadout, RecordWriter, replay, read_record, _logit_hash

mx.set_cache_limit(2 * 2**30)
mx.set_memory_limit(12 * 2**30)
report = {'protocol_sha256': file_sha256(OUT/'protocol.json'), 'started': time.time()}

def log(event, **payload):
    print(json.dumps({'event':event, **payload}, allow_nan=False), flush=True)

class ForwardLog(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.rows = []
    def __getattr__(self, key):
        try: return super().__getattr__(key)
        except AttributeError:
            if key == 'inner': raise
            return getattr(super().__getattr__('inner'), key)
    def __call__(self, ids, *args, **kwargs):
        result = self.inner(ids, *args, **kwargs)
        self.rows.append({'ids':ids[0].tolist(),'shape':list(result.shape),'sha256':_logit_hash(result)})
        return result

class Sink:
    def __init__(self):
        self.rows=[]
        self.head_error=0.
        self.head_rows=0
        self.final=None
    def residual(self, layer, offset, h):
        mx.eval(h)
        if layer == view.num_layers: self.final=h
    def attention(self, block, target, weights, written, total):
        self.head_error=max(self.head_error,float(mx.max(mx.abs(written.sum(0)-total)).item()))
        self.head_rows+=int(weights.shape[0])
    def output(self, offset, ids, logits):
        self.rows.append({'ids':ids,'shape':list(logits.shape),'sha256':_logit_hash(logits)})

class EpisodeCapture:
    def __init__(self):
        self.sink=Sink()
        self.contexts=[]
        self.tokens=[]
    def set_context(self, **context): self.contexts.append(context)
    @contextmanager
    def generation(self, received, tokenizer, prompt, *, turn_cache):
        assert received is model and turn_cache is None
        prompt_ids=tokenizer.encode(prompt,add_special_tokens=tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token))
        assert len(prompt_ids)<=4096
        with NativeCapture(view,self.sink,layers=layers,attention_blocks=blocks) as wrapped: yield wrapped
    def emitted(self, token): self.tokens.append(int(token))

try:
    model_path=PRIMARY/'.not-used'
    model_path=Path('/Users/daniel.tipton/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3')
    model, tokenizer=runlock.load_weights(str(model_path))
    model.eval()
    view=ArchitectureView.from_model(model)
    spec=load_model_spec('qwen35-4b'); resolved=spec.resolve(model,tokenizer)
    band=read_band(ROOT/'configs/models/qwen35-4b.yaml',[view.layer_kind(i) for i in range(view.num_layers)])
    layers=tuple(sorted({p for pair in band for p in pair}))
    blocks=tuple(p-1 for pair in band for p in pair if view.layer_kind(p-1)=='attention')
    report.update(model_snapshot=str(model_path), model_config_sha256=file_sha256(model_path/'config.json'),band=band,cache_strategy=resolved.cache_strategy)
    assert resolved.cache_strategy=='none'
    task=next(t for t in make_tasks('live-lens-fidelity',12,20260907,perturb=False,difficulty=0) if t.family=='pointer_chain')
    sampler=make_sampler(temp=0.)
    native_wrapper=ForwardLog(model)
    t=time.monotonic()
    native=run_task(native_wrapper,tokenizer,task,sampler=sampler,spec=spec,view=view,resolved=resolved,max_steps=8,max_tokens=160,keep_last=2)
    native_seconds=time.monotonic()-t
    log('native_episode',turns=native.turns,tokens=native.generated_tokens,seconds=native_seconds)
    capture=EpisodeCapture()
    t=time.monotonic()
    observed=run_task(model,tokenizer,task,sampler=sampler,spec=spec,view=view,resolved=resolved,max_steps=8,max_tokens=160,keep_last=2,capture=capture)
    captured_seconds=time.monotonic()-t
    native_turns=[s['raw'] for s in native.steps]
    captured_turns=[s['raw'] for s in observed.steps]
    same=native_wrapper.rows==capture.sink.rows and native_turns==captured_turns
    transition=any('hidden' in json.dumps(context['messages']).lower() for context in capture.contexts)
    report['episode']={'task_id':task.task_id,'turns':native.turns,'tokens':native.generated_tokens,'native_seconds':native_seconds,'captured_seconds':captured_seconds,'forward_count':len(native_wrapper.rows),'forward_hashes_identical':native_wrapper.rows==capture.sink.rows,'turns_identical':native_turns==captured_turns,'window_transition_exercised':transition,'head_rows':capture.sink.head_rows,'head_sum_max_abs_error':capture.sink.head_error}
    with (OUT/'episode-forward-hashes.json').open('x') as f: json.dump({'native':native_wrapper.rows,'captured':capture.sink.rows,'contexts':capture.contexts,'raw_turns':native_turns},f)
    log('episode_gate',**report['episode'])
    assert same, 'native capture changed forward outputs'
    assert transition, 'episode did not exercise the observation-window transition'

    ids=mx.array([tokenizer.encode('A short fixed prefix for native lens validation.')])
    baseline=model(ids,cache=view.make_cache()); mx.eval(baseline)
    sink=Sink()
    with NativeCapture(view,sink,layers=(view.num_layers,),injection=(20,1,np.zeros(view.hidden_size))) as wrapped:
        zero=wrapped(ids,cache=view.make_cache()); mx.eval(zero)
    report['zero_injection']={'max_abs_error':float(mx.max(mx.abs(baseline.astype(mx.float32)-zero.astype(mx.float32))).item()),'hash_equal':_logit_hash(baseline)==_logit_hash(zero)}
    assert report['zero_injection']['hash_equal']
    lens_path=PRIMARY/'models/jlens/Qwen3.5-4B_jacobian_lens_n1000.npz'
    lens=LensMaps.load(lens_path,expected_sha256='381c089dcffead8147ee91f944496f468cce2c7d593e0a1b17230745055aea12',hidden_size=view.hidden_size,num_layers=view.num_layers)
    reader=LensReadout(view,lens)
    final=reader.logits(sink.final,view.num_layers)
    report['final_identity']={'max_abs_error':float(mx.max(mx.abs(final-zero.astype(mx.float32))).item()),'hash_equal':_logit_hash(final)==_logit_hash(zero)}
    assert report['final_identity']['hash_equal']
    log('identity_and_zero',zero=report['zero_injection'],identity=report['final_identity'])

    from types import SimpleNamespace
    prompt='A short fixed prefix for native lens validation.'
    for repeat in (1,2):
        path=OUT/f'short-capture-{repeat}.jsonl'
        with RecordWriter(path,{'lens_sha256':lens.sha256,'population_sha256':json.loads((OUT/'population.json').read_text())['population_sha256'],'fixture':'fixed short prefix, not a behavioral endpoint'}) as write:
            session=CaptureSession(view,reader,write,layers=layers,attention_blocks=blocks,top_k=10,audit_modulus=23,audit_seed=20260907)
            with session.generation(model,tokenizer,prompt,turn_cache=None) as wrapped:
                cache=view.make_cache()
                logits=wrapped(ids,cache=cache)
                next_id=int(mx.argmax(logits[0,-1]).item())
                wrapped(mx.array([[next_id]]),cache=cache)
                session.emitted(next_id)
        log('short_capture',repeat=repeat,bytes=path.stat().st_size)
    identical=(OUT/'short-capture-1.jsonl').read_bytes()==(OUT/'short-capture-2.jsonl').read_bytes()
    report['record_reproducibility']={'byte_identical':identical,'sha256':file_sha256(OUT/'short-capture-1.jsonl')}
    assert identical
    report['replay']=replay(view,read_record(OUT/'short-capture-1.jsonl'),atol=0.,rtol=0.)
    report['status']='pass'
except BaseException as error:
    report['status']='fail'; report['error']=repr(error)
    traceback.print_exc()
    raise
finally:
    report['peak_memory_gib']=mx.get_peak_memory()/2**30
    report['elapsed_seconds']=time.time()-report['started']
    with (OUT/'acceptance-v1.json').open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    log('finished',status=report['status'],elapsed=report['elapsed_seconds'],peak_memory_gib=report['peak_memory_gib'])
