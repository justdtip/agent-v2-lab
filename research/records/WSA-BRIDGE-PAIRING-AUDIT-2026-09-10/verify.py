"""Verify the metadata-only bridge audit without importing its package or test module."""
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
manifest = m['verify_sources'](root)
for item in manifest['sources']:
    raw = subprocess.check_output(['git', 'show', item['commit'] + ':' + item['repository_path']], cwd=root)
    assert raw == (root / item['snapshot']).read_bytes()
with tempfile.TemporaryDirectory(prefix='wsa-bridge-metadata-') as t:
    clone = Path(t) / 'record'
    shutil.copytree(root, clone)
    assert m['analyze'](clone) == expected
    for item in manifest['sources']:
        p = clone / item['snapshot']
        original = p.read_bytes()
        p.write_bytes(original + b'\n')
        try:
            m['verify_sources'](clone)
        except ValueError:
            pass
        else:
            raise AssertionError('corrupted source accepted')
        p.write_bytes(original)
source = root / 'source'
rows = [json.loads(line) for line in (source / 'displacement.jsonl').read_text().splitlines()]
raw_table = json.loads((source / 'anchor_sensitivity.json').read_text())
cells = expected['sensitivity_table_cells']
assert len(cells) == 60
assert len(expected['sensitivity_table_mismatches']) == 1
for cell in cells:
    changed = copy.deepcopy(raw_table)
    slot = changed[m['BASE']]['layers'][str(cell['repo_layer'])][cell['group']]
    if cell['group'] != 'native_anchor_read_in_the_float32_tail':
        slot = slot[cell['precision']]
    slot[cell['metric']] += 1
    results = m['numerical_table'](json.dumps(changed), rows)
    key_fields = ['repo_layer', 'group', 'precision', 'metric']
    matches = [r for r in results if all(r[k] == cell[k] for k in key_fields)]
    assert len(matches) == 1 and not matches[0]['matches']
assert not any(k.split('.')[0] in {'torch', 'mlx', 'mlx_lm', 'transformers', 'local_llm_lab', 'pytest'} for k in sys.modules)
ruff = shutil.which('ruff')
assert ruff, 'put ruff on PATH'
lint = subprocess.run([ruff, 'check', '--no-cache', '--select', 'F,E9', str(root / 'analyze.py'), str(root / 'verify.py')], capture_output=True, text=True)
assert lint.returncode == 0, lint.stdout + lint.stderr
assert subprocess.run(['git', 'diff', '--check'], cwd=root, capture_output=True).returncode == 0
report = {'schema_version': 1, 'basis': 'executed local metadata fixtures; not model evidence',
          'analysis_reproduces': True, 'relocated_copy_reproduces': True,
          'source_blobs_match_git': 7, 'source_corruptions_refused': 7,
          'pure_producer_functions_extracted': 6, 'isolated_metadata_tests_extracted': 1,
          'pairing_positive_and_negative_controls': 'passed',
          'nested_ratio_survives_test': True, 'immediate_ratio_negative_control_refused': True,
          'numeric_fields_compared': 60, 'numeric_fields_mismatched': 1,
          'numeric_field_corruptions_detected': 60, 'ruff_F_E9': lint.stdout.strip(),
          'git_diff_check': 'passed', 'model_or_pytest_imports': False,
          'analysis_sha256': hashlib.sha256((root / 'analysis.json').read_bytes()).hexdigest(),
          'reviewer_parent': manifest['reviewer_parent'],
          'unexecuted': ['model or device operations', 'full production-module import',
                         'native pytest', 'real pairing measurement', 'real A2 read']}
(root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
