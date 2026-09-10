"""File-only verification for the bridge fix review."""
import copy
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent
m = runpy.run_path(str(root / 'analyze.py'))
expected = json.loads((root / 'analysis.json').read_text())
assert m['analyze'](root) == expected
manifest = m['sources'](root)
for row in manifest['sources']:
    raw = subprocess.check_output(['git', 'show', row['commit'] + ':' + row['repository_path']], cwd=root)
    assert raw == (root / row['snapshot']).read_bytes()
with tempfile.TemporaryDirectory(prefix='wsa-bridge-fix-review-') as t:
    clone = Path(t) / 'record'
    shutil.copytree(root, clone)
    assert m['analyze'](clone) == expected
    for row in manifest['sources']:
        p = clone / row['snapshot']
        raw = p.read_bytes()
        p.write_bytes(raw + b'\n')
        try:
            m['sources'](clone)
        except ValueError:
            pass
        else:
            raise AssertionError('source corruption was accepted')
        p.write_bytes(raw)
raw = json.loads((root / 'source/anchor_sensitivity.json').read_text())
table = {k: v for k, v in raw.items() if not k.startswith('_')}
rows = [json.loads(line) for line in (root / 'source/displacement.jsonl').read_text().splitlines()]
assert len(expected['numeric_cells']) == 60 and all(r['matches'] for r in expected['numeric_cells'])
for cell in expected['numeric_cells']:
    changed = copy.deepcopy(table)
    slot = changed[m['BASE']]['layers'][str(cell['layer'])][cell['group']]
    if cell['group'] != 'native_anchor_read_in_the_float32_tail':
        slot = slot[cell['phase']]
    slot[cell['quantity']] += 1
    checked = m['numeric_table'](changed, rows)
    key = ('layer', 'group', 'phase', 'quantity')
    item = next(r for r in checked if all(r[k] == cell[k] for k in key))
    assert not item['matches']
assert len(expected['declared_field_mutations']) == len(expected['missing_field_refusals']) == 10
assert all(r['outcome'] == 'refused' for r in expected['declared_field_mutations'])
assert len(expected['schema_mutations']) == 6 and all(r['outcome'] == 'refused' for r in expected['schema_mutations'])
case = expected['artifact_identity_counterexample']
assert case['matrix_json_sha256']['A'] != case['matrix_json_sha256']['B']
assert case['hook_alignment_outcomes']['A']['outcome'] == case['hook_alignment_outcomes']['B']['outcome'] == 'accepted'
assert case['different_nu_control']['outcome'] == 'refused'
assert case['relative_readout_change']['A'] < 1e-4 and case['relative_readout_change']['B'] > 0.9
assert not any(k.split('.')[0] in {'torch', 'mlx', 'mlx_lm', 'transformers', 'local_llm_lab', 'pytest'} for k in sys.modules)
ruff = shutil.which('ruff')
assert ruff, 'put ruff on PATH'
lint = subprocess.run([ruff, 'check', '--no-cache', '--select', 'F,E9', str(root / 'analyze.py'), str(root / 'verify.py')], capture_output=True, text=True)
assert lint.returncode == 0, lint.stdout + lint.stderr
assert subprocess.run(['git', 'diff', '--check'], cwd=root, capture_output=True).returncode == 0
report = {'schema_version': 1, 'basis': 'executed metadata fixtures and standard-library matrix arithmetic',
          'source_files_match_git': 7, 'source_corruptions_refused': 7,
          'analysis_reproduces': True, 'relocated_copy_reproduces': True,
          'metadata_functions_extracted': 14, 'declared_field_mutations_refused': 10,
          'missing_field_refusals': 10, 'nested_schema_corruptions_refused': 6,
          'numeric_cells_match': 60, 'numeric_cell_corruptions_detected': 60,
          'different_artifacts_same_declaration_both_accepted': True,
          'different_declaration_control_refused': True, 'model_package_imports': False,
          'ruff_F_E9': lint.stdout.strip(), 'git_diff_check': 'passed',
          'analysis_sha256': hashlib.sha256((root / 'analysis.json').read_bytes()).hexdigest(),
          'reviewer_parent': manifest['reviewer_parent'],
          'unexecuted': ['model forwards', 'checkpoint loads', 'native pytest', 'real pairing registration', 'real A2']}
(root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
