"""Reproduce the width audit from frozen JSONL and exact arithmetic; no model imports."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
BASIS="computed from pinned producer device records; no new device execution"
def verify(data,sha):
    if hashlib.sha256(data).hexdigest()!=sha:
        raise ValueError("source hash mismatch")
def key(r):
    return tuple(r[k] for k in ("precision","repo_layer","position","direction","cotangent","k"))
def read(name):
    b=(ROOT/'sources/calibration/artefacts'/name).read_bytes()
    assert b.endswith(b"\n"),"incomplete JSONL"
    rows=[json.loads(l) for l in b.splitlines()]
    result={key(r):r for r in rows}
    assert len(rows)==len(result)==648
    return result

def analyze():
    manifest=json.loads((ROOT/'SOURCE-MANIFEST.json').read_text())
    for name,e in manifest['sources'].items():
        verify((ROOT/'sources'/name).read_bytes(),e['sha256'])
    old,a,b=read('ladder.jsonl'),read('ladder-w1.jsonl'),read('ladder-w64.jsonl')
    assert set(old)==set(a)==set(b)
    assert all(old[k]['a_requested']==a[k]['a_requested'] and old[k]['d_h']==a[k]['d_h'] for k in a)
    for expected,data in [(1,a),(64,b)]:
        for row in data.values():
            assert row['width']==row['anchor_batch']==expected
            assert row['h']==row['h0']*2**(-row['k'])
            assert row['absolute_error']==abs(row['d_h']-row['a_requested'])
    summaries=[]
    for precision in ['native','float32']:
        for layer in [1,17,33]:
            ks=[k for k in a if k[0]==precision and k[1]==layer and k[-1]==0]
            assert len(ks)==18
            ad1,ad64=[a[k]['a_requested'] for k in ks],[b[k]['a_requested'] for k in ks]
            assert all(x!=0 for x in ad1)
            for k in ks:
                for rung in (0,2,4,6,8,10):
                    match=(*k[:-1],rung)
                    assert a[match]['a_requested']==a[k]['a_requested']
                    assert b[match]['a_requested']==b[k]['a_requested']
            relative=[abs(y-x)/abs(x) for x,y in zip(ad1,ad64)]
            hratios={b[k]['h0']/a[k]['h0'] for k in ks}
            assert len(hratios)==1
            ratio=hratios.pop()
            per_width=[]
            for width,data in [(1,a),(64,b)]:
                candidates=[]
                for rung in (0,2,4,6,8,10):
                    g=[v for k,v in data.items() if (k[0],k[1],k[-1])==(precision,layer,rung)]
                    assert len(g)==18
                    candidates.append({'k':rung,'h':g[0]['h'],'median_relative_error':statistics.median(x['relative_error'] for x in g)})
                best=min(candidates,key=lambda x:x['median_relative_error'])
                per_width.append({'width':width,'best_sampled_median':best,'last_rung_is_best':best['k']==10,'ladder':candidates})
            summaries.append({'precision':precision,'layer':layer,'projections':18,'basis':BASIS,
               'AD_relative_change_min':min(relative),'AD_relative_change_median':statistics.median(relative),
               'AD_relative_change_max':max(relative),
               'AD_projected_L2_relative_change':math.sqrt(sum((y-x)**2 for x,y in zip(ad1,ad64))/sum(x*x for x in ad1)),
               'h64_over_h1':ratio,'ratio_over_sqrt_batch':ratio/8,
               'effective_k_shift':math.log2(ratio),'per_width':per_width})
    # Exact fixture: repeating (3,4) 64 times multiplies its Euclidean norm by eight.
    norm_one,norm_64=Fraction(5),Fraction(40)
    h1,h64=norm_one/100,norm_64/100
    x=Fraction(3)
    fd=lambda h:((x+h)**3-(x-h)**3)/(2*h)
    exact=3*x*x
    assert norm_64/norm_one==8
    assert (fd(h64)-exact)/(fd(h1)-exact)==64
    # A single smooth function can have a 75.6% derivative change at different anchors.
    center=Fraction(1000);x1=center+1;x2=center+Fraction(439,250)
    ad=lambda t:t-center  # derivative of (t-center)^2/2
    assert (ad(x2)-ad(x1))/ad(x1)==Fraction(189,250)
    # Raw residual-relative error is not a readout-relative bound.
    # h=(100,1), delta=(0,1), L reads only the second coordinate.
    residual_relative=1/math.sqrt(10001)
    readout_relative=1.0
    assert residual_relative<.0124 and readout_relative==1
    return {'basis':BASIS,'model_execution':False,'width_one_AD_FD_identity_cells':648,
       'paired_cells':648,'summaries':summaries,
       'batch_step_counterexample':{'basis':'exact analytic fixture, not model data','source':[3,4],'copies':64,
          'norm_one':str(norm_one),'norm_repeated':str(norm_64),'h_ratio':8,
          'same_function_FD_truncation_error_ratio':64},
       'anchor_shift_counterexample':{'basis':'exact analytic fixture, not a dtype simulation','function':'(x-1000)^2/2',
          'x1':str(x1),'x2':str(x2),'gradient_relative_change':.756,'arithmetic_path_changed':False},
       'readout_counterexample':{'basis':'analytic illustration, not model data','residual_relative_change':residual_relative,
          'linear_readout_relative_change':readout_relative}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    payload=json.dumps(analyze(),indent=2,sort_keys=True)+'\n';out=ROOT/'analysis.json'
    if args.check:
        assert out.read_text()==payload
        m=json.loads((ROOT/'SOURCE-MANIFEST.json').read_text())['sources'];rejected=0
        for name,item in m.items():
            try:verify((ROOT/'sources'/name).read_bytes()+b'x',item['sha256'])
            except ValueError:rejected+=1
            else:raise AssertionError('corrupted source accepted')
        print(json.dumps({'status':'passed','paired_cells':648,'corrupted_sources_rejected':rejected,
                          'analytic_counterexamples':3,'model_execution':False}))
    else:
        out.write_text(payload);print('analysis.json written')
if __name__=='__main__':main()
