"""Verify sealed-data reductions and the diagnosis's mathematical counterexamples."""
from __future__ import annotations
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('fd_analysis',ROOT/'analyze.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
raw=m.load()
layers,cells,errors,result=m.derive(*raw)
checks=['all source hashes and finished logs','original-step sweep identity','33-layer metric coverage','shared corpus endpoint precision and selector','eight native results and eight failed promoted cells']


def reject(name,fn,needle):
    try:fn()
    except (ValueError,ImportError) as e:
        if needle not in str(e):raise AssertionError(str(e)) from e
    else:raise AssertionError('failed to reject '+name)
    checks.append(name)


bad=copy.deepcopy(raw);bad[3]['rows.jsonl'].append(copy.deepcopy(cells[0]))
reject('duplicate cell refusal',lambda:m.derive(*bad),'control count')
bad=copy.deepcopy(raw);bad[1]['endpoint']['target_is_pre_final_norm']=False
reject('changed endpoint refusal',lambda:m.derive(*bad),'target convention')
bad=copy.deepcopy(raw);bad[3]['rows.jsonl'][-2]['error']='other failure'
reject('unexpected failed control refusal',lambda:m.derive(*bad),'unexpected failure')
reject('model imports blocked',lambda:__import__('torch'),'model libraries forbidden')

q=result['mathematical_model'];b=q['counterexample_B_over_A']
assert math.isclose((1+b)/(.25+2*b),result['measured']['halving_ratio'],rel_tol=1e-12)
assert math.isclose(.01*math.sqrt(128*2560),q['original_step_over_global_coordinate_RMS'],rel_tol=1e-12)
# A rounded, asymmetric scalar intervention on a quadratic exposes the midpoint correction.
x,h,x_plus,x_minus=2.,.3,2.5,1.75
v_actual=(x_plus-x_minus)/(2*h);mid=(x_plus+x_minus)/2
fd=(x_plus*x_plus-x_minus*x_minus)/(2*h)
assert math.isclose(fd,2*mid*v_actual)
assert not math.isclose(fd,2*x*v_actual)
checks.extend(['halving-ratio mixed-error counterexample','global norm RMS identity','realized direction still needs midpoint correction'])
with tempfile.TemporaryDirectory(prefix='fd-numerics-verify-') as temp:
    temp=Path(temp)
    clone=temp/'clone';shutil.copytree(ROOT/'source',clone/'source')
    shutil.copyfile(ROOT/'SOURCES.json',clone/'SOURCES.json')
    with (clone/'source/card/golden-report.json').open('ab') as f:f.write(b' ')
    reject('changed source refused',lambda:m.load(clone),'source hash mismatch')
    out=temp/'reproduction'
    subprocess.run([sys.executable,str(ROOT/'analyze.py'),'--output-dir',str(out)],check=True,capture_output=True,text=True)
    for n in ['analysis.json','depth.csv','sweep.csv','failed-controls.json','diagnosis.png','diagnosis.svg']:
        assert (ROOT/n).read_bytes()==(out/n).read_bytes(),n
    checks.append('all six outputs reproduce byte for byte')
    rerun=subprocess.run([sys.executable,str(ROOT/'analyze.py'),'--output-dir',str(out)],capture_output=True,text=True)
    assert rerun.returncode!=0 and 'fresh output directory' in rerun.stderr
    checks.append('existing output refusal')
print(json.dumps({'checks_passed':len(checks),'checks':checks,'model_loads':0},indent=2))
