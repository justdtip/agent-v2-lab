"""Verify this file-only review; no producer code is imported or executed."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent
a = runpy.run_path(str(root / 'analyze.py'))
record = json.loads((root / 'analysis.json').read_text())
assert a['analyze'](root) == record
manifest = a['verify_sources'](root)
for item in manifest['sources']:
    raw = subprocess.check_output(['git', 'show', item['commit'] + ':' + item['repository_path']], cwd=root)
    assert raw == (root / item['snapshot']).read_bytes()
forbidden = ('torch', 'mlx', 'mlx_lm', 'transformers', 'local_llm_lab')
assert not any(k == p or k.startswith(p + '.') for k in sys.modules for p in forbidden)
with tempfile.TemporaryDirectory(prefix='wsa-map-audit-') as t:
    clone = Path(t) / 'record'
    shutil.copytree(root, clone)
    refused = []
    for item in manifest['sources']:
        p = clone / item['snapshot']
        raw = p.read_bytes()
        p.write_bytes(raw + b'\n')
        try:
            a['verify_sources'](clone)
        except ValueError:
            refused.append(item['snapshot'])
        else:
            raise AssertionError('changed source accepted')
        p.write_bytes(raw)
    assert len(refused) == 22
    assert a['analyze'](clone) == record
source = (root / 'source/golden_float32.py').read_text()
assert a['reference_reused'](source)['same_reference_expression']
changed = source.replace('reproduction=reference,', 'reproduction=independent_repeat,')
assert not a['reference_reused'](changed)['same_reference_expression']
raw = a['rows'](root / 'source', 'artefacts/displacement.jsonl')
corrupt = copy.deepcopy(raw)
corrupt[0]['delta_relative'] += 0.1
try:
    a['displacement_summary'](corrupt)
except ValueError:
    pass
else:
    raise AssertionError('corrupt ratio accepted')
raw = a['rows'](root / 'source', 'artefacts/same-anchor.jsonl')
corrupt = copy.deepcopy(raw)
corrupt[0]['a_w1_x1'] += 1.0
try:
    a['same_anchor_summary'](corrupt, True)
except ValueError:
    pass
else:
    raise AssertionError('changed raw reference accepted')
for filename in ['analyze.py', 'verify.py']:
    ast.parse((root / filename).read_text())
ruff = shutil.which('ruff')
if ruff is None:
    raise SystemExit('ruff must be on PATH; no package installation attempted')
lint = subprocess.run([ruff, 'check', '--no-cache', '--select', 'F,E9',
                       str(root / 'analyze.py'), str(root / 'verify.py')],
                      capture_output=True, text=True)
assert lint.returncode == 0, lint.stdout + lint.stderr
assert subprocess.run(['git', 'diff', '--check'], cwd=root, capture_output=True).returncode == 0
verification = {
    'schema_version': 1,
    'basis': 'executed locally, file-only; no model/native tests or device access',
    'source_count': 22, 'source_bytes_match_pinned_git': True,
    'all_changed_sources_refused': len(refused),
    'analysis_reproduces_exactly': True, 'relocated_copy_reproduces_exactly': True,
    'same_reference_bug_detected': True,
    'independent_reference_mutation_no_longer_detected_as_same': True,
    'corrupted_displacement_ratio_refused': True,
    'corrupted_same_anchor_raw_response_refused': True,
    'analytic_known_answer_cases': list(a['counterexamples']().keys())[1:],
    'python_syntax': 'passed', 'ruff_F_E9': lint.stdout.strip(),
    'git_diff_check': 'passed', 'model_imports_absent': True, 'scalars_checked': 270,
    'analysis_sha256': hashlib.sha256((root / 'analysis.json').read_bytes()).hexdigest(),
    'reviewer_parent': manifest['reviewer_parent'],
    'unexecuted': ['model computations', 'native pytest', 'matrix NPZ reanalysis',
                  'full-anchor norm reconstruction', 'device access'],
}
(root / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
print(json.dumps(verification, indent=2))
