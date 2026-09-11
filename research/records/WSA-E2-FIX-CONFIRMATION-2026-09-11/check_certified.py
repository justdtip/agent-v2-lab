"""Confirm C1-C3 using frozen source bytes and small independent NumPy fixtures only."""
from __future__ import annotations
import argparse, hashlib, importlib.abc, json, platform, sys, types
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--snapshot',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--manifest',type=Path,default=Path(__file__).with_name('reviewed-files.json'))
args=parser.parse_args()
root=args.snapshot.resolve()
manifest=json.loads(args.manifest.read_text())
paths=['src/local_llm_lab/pipeline/state_programme/transport.py','src/local_llm_lab/pipeline/state_programme/transport_certified.py','tests/test_state_transport_certified.py','tests/test_e2_review_regressions.py','research/records/STATE-PLAN-PROGRESS-2026-09-09/AMENDMENT-2-CERTIFIED-DIRECTION.md']
source={p:(root/p).read_bytes() for p in paths}
E={'base_commit':manifest['base_commit'],'reviewed_state':'uncommitted snapshot identified by file hashes, not base commit alone','python':platform.python_version(),'numpy':np.__version__,'source_sha256':{p:hashlib.sha256(b).hexdigest() for p,b in source.items()},'checks':{},'real_capture_reads':0}
assert all(E['source_sha256'][p]==manifest['source_sha256'][p] for p in paths)
class NoModels(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','mlx','mlx_lm','transformers','scipy'}:raise RuntimeError('forbidden runtime import '+fullname)
sys.meta_path.insert(0,NoModels())
for n in ['local_llm_lab','local_llm_lab.pipeline','local_llm_lab.pipeline.state_programme']:
 p=types.ModuleType(n);p.__path__=[];sys.modules[n]=p
mods={}
for n in ['transport','transport_certified']:
 path='src/local_llm_lab/pipeline/state_programme/'+n+'.py';m=types.ModuleType('local_llm_lab.pipeline.state_programme.'+n);m.__file__=path;sys.modules[m.__name__]=m;exec(compile(source[path],path,'exec'),m.__dict__);mods[n]=m
base,draft=mods['transport'],mods['transport_certified'];C=E['checks']
real=base._leading_direction
calls=[]
def counted(a):calls.append(a.copy());return real(a)
base._leading_direction=counted
A=np.array([[3.,-3.,0.],[0.,0.,1.],[0.,0.,0.]])
B=np.array([[3.,-3.],[1.,1.]])
padded=np.zeros((64,64));padded[:2,:2]=B
close=np.array([[1+1e-8,-(1+1e-8)],[1.,1.]])
controls={'exact_nonleading':A,'nonleading':B,'padded64':padded,'near_degenerate':close,'zero':np.zeros((8,8)),'repeated':np.eye(8),'rank_one':np.outer(np.arange(1.,9.),np.arange(1.,9.)),'rectangular':np.arange(15.).reshape(3,5)}
for name,a in controls.items():
 before=a.copy();u0,s0=real(a);diagnostic=draft.leading_triplet(a);start=len(calls);u,s,certificate=draft.leading_direction(a)
 assert not diagnostic[3]['certified'] and not certificate['certified']
 assert certificate['fell_back_to_full_svd'] and len(calls)==start+1
 assert np.array_equal(u,u0) and s==s0 and np.array_equal(before,a)
 C['C1_'+name]={'diagnostic':diagnostic[3],'diagnostic_sigma':diagnostic[1],'reference_sigma':s0,'public_sigma':s,'public_array_equal':True,'reference_calls':1,'matrix_unchanged':True}
# Numerical diagnostic exceptions cannot bypass the reference or prevent its execution.
real_triplet=draft.leading_triplet
for exception in (ValueError,np.linalg.LinAlgError,FloatingPointError):
 def fail(a,exc=exception):raise exc('injected diagnostic failure')
 draft.leading_triplet=fail
 start=len(calls);u,s,c=draft.leading_direction(A);expected=real(A)
 assert c['fell_back_to_full_svd'] and not c['certified'] and len(calls)==start+1 and np.array_equal(u,expected[0]) and s==expected[1]
 C['C1_diagnostic_failure_'+exception.__name__]=c
draft.leading_triplet=real_triplet
with np.errstate(over='raise',invalid='raise'):
 u,s,c=draft.leading_direction(np.eye(2)*1e155)
assert np.isfinite(s) and s==1e155 and c['fell_back_to_full_svd']
C['C1_finite_cross_overflow_in_diagnostic_falls_back']=c
try:draft.leading_direction(np.array([[np.nan]]))
except ValueError:C['C1_nonfinite_reference_refuses']=True
else:raise AssertionError('nonfinite reference admitted')
# Actual full fitter at representative small ranks/precisions/shapes; do not patch its math.
base._leading_direction=real
checks=[]
fields=('mean','scale','target_mean','weights','loadings','target_loadings')
def compare_fit(label,x,y,rank):
 beforex,beforey=x.copy(),y.copy()
 ref=base.fit(x,y,rank);got,certs=draft.certified_fit(x,y,rank)
 assert ref.max_rank==got.max_rank and ref.leading_singular_values==got.leading_singular_values
 assert all(np.array_equal(getattr(ref,f),getattr(got,f)) for f in fields)
 assert np.array_equal(beforex,x) and np.array_equal(beforey,y)
 assert len(certs)==got.max_rank and all(c['fell_back_to_full_svd'] and not c['certified'] for c in certs)
 for r in range(1,got.max_rank+1):
  assert np.array_equal(ref.coefficients(r),got.coefficients(r))
  assert np.array_equal(ref.apply(x,rank=r),got.apply(x,rank=r))
  if x.shape[1]==y.shape[1]:assert np.array_equal(ref.apply(x,rank=r,times=2),got.apply(x,rank=r,times=2))
 checks.append({'name':label,'rank':got.max_rank,'array_equal_all_fields_coefficients_predictions':True,'all_retained_components_fell_back':True})
 return ref,got
x=np.array([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]])
ref,got=compare_fit('C2_tiny_sigma_floor',x,x*(8e-13/4),2);assert ref.max_rank==0
compare_fit('C1_full_fitter_nonleading_start',x,x@B/4,1)
for dtype in (np.float32,np.float64):
 for seed in range(6):
  rng=np.random.default_rng(seed);xx=rng.normal(size=(30,6)).astype(dtype);yy=(xx+.03*rng.normal(size=xx.shape)).astype(dtype)
  compare_fit('ordinary_'+np.dtype(dtype).name+'_'+str(seed),xx,yy,4)
rng=np.random.default_rng(31)
compare_fit('rectangular_target',rng.normal(size=(25,5)),rng.normal(size=(25,3)),3)
xx=rng.normal(size=(25,5));xx[:,4]=3
compare_fit('constant_coordinate',xx,xx+.03*rng.normal(size=xx.shape),5)
compare_fit('constant_target',rng.normal(size=(25,4)),np.ones((25,4)),4)
compare_fit('rank_one_target',rng.normal(size=(25,4)),np.outer(rng.normal(size=25),np.arange(1.,5.)),4)
C['C2_C3_fitter_identity']=checks
# Explicit component-finiteness refusal remains in the same place; static source comparison.
sealed=source[paths[0]].decode();current=source[paths[1]].decode();doc=source[paths[-1]].decode()
assert 'if sigma < DEFLATION_FLOOR:' in current and 'if tt < DEFLATION_FLOOR:' in current
assert 'if not (np.isfinite(p_load).all() and np.isfinite(c_load).all()):' in current
assert 'cross = (cross - np.outer(xt, c_load) - np.outer(p_load, ty)' in current
assert '+ tt * np.outer(p_load, c_load))' in current
C['C2_static_stop_finiteness_and_deflation_order_match']=True
assert 'always reports `certified=False`' in doc and '**always uses the reference full SVD**' in doc
assert 'former unsound bypass' in doc and '20-hour projection for 10,000 refits is withdrawn' in doc
assert 'not\na promise of bitwise identity across different' in doc and 'earlier SciPy estimate also remains withdrawn' in doc
C['C3_claims_revised_and_historical_speedup_withdrawn']=True
E['verdict']='PASS for C1-C3 correction on reviewed bytes; no optimized bypass or execution release'
args.out.write_text(json.dumps(E,indent=2,sort_keys=True)+'\n')
print(json.dumps({'verdict':E['verdict'],'independent_direction_controls':len(controls),'exact_fitter_fixtures':len(checks),'source_sha256':E['source_sha256']},indent=2))
