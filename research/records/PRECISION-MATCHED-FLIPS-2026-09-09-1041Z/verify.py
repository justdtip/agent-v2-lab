"""File-only adversarial checks and byte-for-byte reproduction of this analysis."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('precision_analysis', ROOT / 'analyze.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
manifest, original = module.load_sources()
passed = []


def rejects(name, operation, expected):
    try:
        operation()
    except (ValueError, ImportError) as err:
        if expected not in str(err):
            raise AssertionError(f'{name}: wrong refusal {err}') from err
    else:
        raise AssertionError(f'{name}: accepted bad input')
    passed.append(name)


rows, margins, reference = module.joined_data(original)
assert len(rows) == 24 and sum(r['bf16_cuda_agrees'] for r in rows) == 21
assert sorted(r['reference_gap_ulps'] for r in rows if not r['bf16_cuda_agrees']) == [1, 1, 3]
passed.append('all canonical joins and reductions')

bad = copy.deepcopy(original)
bad['mlx-bf16-arm.json']['positions'][0] = bad['mlx-bf16-arm.json']['positions'][1]
rejects('duplicate position refused', lambda: module.joined_data(bad), 'duplicate position')

bad = copy.deepcopy(original)
bad['mlx-bf16-arm.json']['positions'].pop()
rejects('missing joined position refused', lambda: module.joined_data(bad), 'position set differs')

bad = copy.deepcopy(original)
for name in ('mlx-bf16-arm.json', 'mlx-bf16-arm.jsonl'):
    rows_bad = bad[name]['positions'] if name.endswith('.json') else bad[name]
    rows_bad[0]['produced_token'] += 1
rejects('token identity drift refused', lambda: module.joined_data(bad), 'token/probability identity differs')

bad = copy.deepcopy(original)
bad['flip-margins.json']['positions'].pop()
rejects('missing margin row refused', lambda: module.joined_data(bad), 'margin subset differs')

bad = copy.deepcopy(original)
bad['flip-margins.json']['positions'][0]['mlx_logit_gap'] += .01
rejects('incorrect margin arithmetic refused', lambda: module.joined_data(bad), 'gap reconstruction differs')

rejects('model import forbidden', lambda: __import__('torch'), 'model imports forbidden')

with tempfile.TemporaryDirectory(prefix='precision-join-check-') as temp:
    temp = Path(temp)
    copied = temp / 'copied'
    shutil.copytree(ROOT / 'source', copied / 'source')
    shutil.copyfile(ROOT / 'SOURCES.json', copied / 'SOURCES.json')
    with (copied / 'source' / 'confident-flips.json').open('ab') as f:
        f.write(b' ')
    rejects('source tamper refused', lambda: module.load_sources(copied), 'source hash mismatch')
    with (copied / 'SOURCES.json').open('ab') as f:
        f.write(b' ')
    rejects('manifest tamper refused', lambda: module.load_sources(copied), 'manifest hash mismatch')
    reproduced = temp / 'reproduced'
    subprocess.run([sys.executable, str(ROOT / 'analyze.py'), '--output-dir', str(reproduced)], check=True, capture_output=True, text=True)
    outputs = ['joined.csv', 'margins.csv', 'analysis.json', 'README.md', 'precision-matched-24.png', 'precision-matched-24.svg', 'remaining-three.png', 'remaining-three.svg']
    for name in outputs:
        assert (ROOT/name).read_bytes() == (reproduced/name).read_bytes(), name
    passed.append('all eight outputs reproduce byte for byte')
    again = subprocess.run([sys.executable, str(ROOT/'analyze.py'), '--output-dir', str(reproduced)], capture_output=True, text=True)
    assert again.returncode != 0 and 'fresh output directory' in again.stderr
    passed.append('existing outputs protected')

print(json.dumps({'passed': len(passed), 'checks': passed, 'model_loads': 0}, indent=2))
