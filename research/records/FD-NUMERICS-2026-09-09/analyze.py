"""Read-only numerical diagnosis from sealed golden-test and step-sweep evidence."""
from __future__ import annotations
import argparse
import builtins
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_HASH = '577b8f9688ef5d2466928e596c0f2b39d0b92f25472a6f6414afa95388c9842c'
_import = builtins.__import__


def file_only(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'mlx', 'mlx_lm', 'transformers'}:
        raise ImportError('model libraries forbidden in file-only diagnosis')
    return _import(name, *args, **kwargs)


builtins.__import__ = file_only
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/codex-gpu-analysis-mpl')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(root=ROOT):
    raw = (root/'SOURCES.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_HASH, 'manifest hash mismatch')
    m = json.loads(raw)
    for p, h in m['files'].items():
        require(hashlib.sha256((root/p).read_bytes()).hexdigest() == h, 'source hash mismatch: '+p)
    card = root/'source/card'
    logs = {}
    for name in ('progress.jsonl', 'rows.jsonl'):
        raw = (card/name).read_bytes()
        require(raw.endswith(b'\n'), 'partial JSONL')
        logs[name] = [json.loads(line) for line in raw.splitlines()]
        require(logs[name][-1]['event'] == 'done', 'incomplete run snapshot')
    report = json.loads((card/'golden-report.json').read_text())
    nu = json.loads((card/'nu-finite-difference.json').read_text())
    exact = json.loads((card/'nu-exact.json').read_text())
    return report, nu, exact, logs


def derive(report, nu, exact, logs):
    require(nu['endpoint'] == exact['endpoint'], 'target convention mismatch')
    require(nu['corpus'] == exact['corpus'], 'corpus mismatch')
    require(nu['corpus']['rows'] == [39] and nu['corpus']['n_prompts'] == 1, 'unexpected corpus')
    require(nu['precision']['capture_dtype'] == exact['precision']['capture_dtype'] == 'native', 'mixed arithmetic')
    require(nu['precision']['dtype'] == exact['precision']['dtype'] == 'bfloat16', 'precision mismatch')
    require(nu['endpoint']['target_is_pre_final_norm'], 'endpoint changed')
    require(nu['position_weighting']['probe'][0]['runs'] == [[8,9],[127,128]], 'position selection changed')
    layers = report['finding']['per_layer']
    require([r['layer'] for r in layers] == list(range(1,34)), 'layer order/coverage changed')
    for r in layers:
        require(r['reference_norm'] > 0 and r['relative_difference'] >= 0 and abs(r['cosine']) <= 1, 'invalid map metric')
    cells = [r for r in logs['rows.jsonl'] if r['event'] == 'cell']
    ok = [r for r in cells if 'error' not in r]
    errors = [r for r in cells if 'error' in r]
    require(len(ok) == len(errors) == 8, 'unexpected control count')
    require(all(r['capture_dtype'] == 'native' for r in ok), 'unexpected successful control')
    require(all(r['capture_dtype'] == 'promoted-float32' and 'same dtype' in r['error'] for r in errors), 'unexpected failure')
    keys = [(r['repo_layer'], r['epsilon_scale']) for r in ok]
    require(len(set(keys)) == 8, 'duplicate sweep cell')
    require(set(keys) == {(l,c) for l in (1,33) for c in (.01,.001,.0001,.00001)}, 'incomplete sweep')
    for r in ok:
        require(math.isfinite(r['relative_difference']) and math.isfinite(r['cosine']), 'invalid sweep metric')
        if r['epsilon_scale'] == .01:
            base = layers[r['repo_layer']-1]
            require(r['relative_difference'] == base['relative_difference'] and r['cosine'] == base['cosine'], 'sweep fails to reproduce original')
    half = next(r for r in logs['progress.jsonl'] if r['event'] == 'epsilon_schedule')
    ratio = half['residual_coarse']/half['residual_fine']
    require(math.isclose(ratio, half['ratio'], rel_tol=1e-12), 'halving ratio mismatch')
    b_over_a = (1-ratio/4)/(2*ratio-1)
    require(b_over_a > 0, 'counterexample outside two-term positive range')
    reconstructed = (1+b_over_a)/(.25+2*b_over_a)
    require(math.isclose(reconstructed, ratio, rel_tol=1e-12), 'envelope counterexample arithmetic')
    first = layers[0]
    r, c = first['relative_difference'], first['cosine']
    require(r > 1, 'norm ratio would have two nonnegative branches')
    norm_ratio = c+math.sqrt(r*r-1+c*c)
    require(math.isclose(r*r, 1+norm_ratio**2-2*norm_ratio*c, rel_tol=1e-12), 'norm geometry inconsistent')
    dimension, seq = 2560, nu['corpus']['max_seq_len']
    result = {
        'scope':'instrument numerical diagnosis; one row/two positions; no new model execution',
        'measured':{'native_cells':len(ok),'promoted_cells_failed_before_result':len(errors),'successful_float32_controls':0,
                    'halving_ratio':ratio,'layer1_norm_ratio_inferred':norm_ratio,
                    'layer1_aligned_projection_fraction_inferred':norm_ratio*c},
        'mathematical_model':{
            'basis':'illustrative error envelope, not a fit to Gemma',
            'form':'A*h^2 + B/h + C with nonnegative coefficients; not an equality for observed vector errors',
            'normalized_current_h':1,'counterexample_C':0,'counterexample_B_over_A':b_over_a,
            'counterexample_halving_ratio':reconstructed,
            'counterexample_optimum_relative_h':(b_over_a/2)**(1/3),
            'warning':'The illustrative optimum is not a recommended step for the model; one ratio does not identify error components.',
            'original_step_over_global_coordinate_RMS':.01*math.sqrt(seq*dimension),
            'step_scale_basis':'exact norm/RMS identity for 128 by 2560 entries; not relative to the particular affected coordinate',
        },
        'interpretation':'Small-step instability is observed. Mixed truncation and arithmetic error is a leading explanation; forward-schedule mismatch remains unchecked. Failed dtype cells do not answer the float32 control.',
        'unmeasured':['realized perturbation and midpoint shift','zero-response fraction before aggregation','same-primal same-batch zero-step hook check','matched whole-float32 FD versus AD','native AD versus float32 AD at the same stored primal','identity-subtracted map agreement'],
    }
    return layers, ok, errors, result


def csv_write(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def plot(layers, cells, out):
    blue, gold = '#267CA1','#B8791F'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'fd-numerics-20260909'})
    fig, ax=plt.subplots(2,2,figsize=(13,9))
    fig.subplots_adjust(left=.085,right=.95,bottom=.14,top=.79,hspace=.48,wspace=.27)
    fig.text(.06,.948,'The step sweep reveals instability, not a resolved float32 control',fontsize=17,weight='bold')
    fig.text(.06,.907,'Native bf16 · one held row · 128 tokens · two selected positions · raw final-block residual target',fontsize=10.5)
    fig.text(.06,.866,'Eight promoted-residual attempts failed on a dtype mismatch; none produced a float32 comparison.',fontsize=10.5,color='#8E5912')
    depth=[r['layer'] for r in layers]
    ax[0,0].plot(depth,[r['relative_difference'] for r in layers],color=blue,marker='o',markersize=3)
    ax[0,0].set_ylabel('Relative Frobenius difference')
    ax[0,0].set_xlabel('Source layer (repository convention)')
    ax[0,0].set_title('Original step: 0.01 × full-sequence norm',loc='left',fontsize=11)
    ax[0,1].plot(depth,[r['cosine'] for r in layers],color=blue,marker='o',markersize=3)
    ax[0,1].set_ylabel('Map cosine')
    ax[0,1].set_xlabel('Source layer (repository convention)')
    ax[0,1].set_ylim(-.08,1)
    ax[0,1].set_title('Overall depth trend, with local reversals',loc='left',fontsize=11)
    for l,col,mark in [(1,blue,'o'),(33,gold,'s')]:
        rows=sorted([r for r in cells if r['repo_layer']==l],key=lambda r:r['epsilon_scale'])
        x=[r['epsilon_scale'] for r in rows]
        ax[1,0].loglog(x,[r['relative_difference'] for r in rows],color=col,marker=mark,label=f'Layer {l}')
        ax[1,1].semilogx(x,[r['cosine'] for r in rows],color=col,marker=mark,label=f'Layer {l}')
    ax[1,0].set_ylabel('Relative difference (log scale)')
    ax[1,1].set_ylabel('Map cosine')
    ax[1,0].set_title('Smaller steps amplify the error',loc='left',fontsize=11)
    ax[1,1].set_title('Layer 1 never reaches good alignment here',loc='left',fontsize=11)
    for a in ax[1]:
        a.set_xlabel('Step scale multiplying the full-sequence norm')
        a.legend(frameon=False)
    for a in ax.flat:
        a.grid(color='#E1E7E9',linewidth=.6)
        a.spines[['top','right']].set_visible(False)
    fig.text(.06,.074,'Layer 33 improves to 28.6% error at 10⁻³, then degrades. Layer 1 reaches 45.7× error at 10⁻⁵.',fontsize=11,weight='bold')
    fig.text(.06,.042,'Observed card results; lines join measurements. The fourfold halving reference was a tiny float32 fixture, not this model.',fontsize=9.5)
    fig.text(.06,.016,'Source: /workspace/wsd/out/{golden-4b,diagnose-4b}; immutable local snapshot with transfer hashes. No timing inference.',fontsize=8.5,color='#596D77')
    fig.savefig(out/'diagnosis.png',dpi=150,facecolor='white')
    svg=out/'diagnosis.svg'
    fig.savefig(svg,facecolor='white',metadata={'Date':None})
    svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT)
    args=p.parse_args()
    layers,cells,errors,result=derive(*load())
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    names=['analysis.json','depth.csv','sweep.csv','failed-controls.json','diagnosis.png','diagnosis.svg']
    require(not any((out/n).exists() for n in names),'use fresh output directory')
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'failed-controls.json').write_text(json.dumps(errors,indent=2)+'\n')
    csv_write(out/'depth.csv',layers);csv_write(out/'sweep.csv',cells)
    plot(layers,cells,out)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
