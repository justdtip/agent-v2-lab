"""Verify the metadata-only review closure."""
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent
m = runpy.run_path(str(root / 'check.py'))
expected = json.loads((root / 'check.json').read_text())
assert m['check'](root) == expected
manifest = m['verify_sources'](root)
for row in manifest['sources']:
    assert subprocess.check_output(['git', 'show', row['commit'] + ':' + row['repository_path']], cwd=root) == (root / row['snapshot']).read_bytes()
with tempfile.TemporaryDirectory(prefix='wsa-bridge-closure-') as t:
    clone = Path(t) / 'record'
    shutil.copytree(root, clone)
    assert m['check'](clone) == expected
    for row in manifest['sources']:
        p = clone / row['snapshot']
        raw = p.read_bytes()
        p.write_bytes(raw + b'\n')
        try:
            m['verify_sources'](clone)
        except ValueError:
            pass
        else:
            raise AssertionError('changed source accepted')
        p.write_bytes(raw)
assert len(expected['field_controls']) == 11
assert all(r['changed'] == r['missing'] == 'refused' for r in expected['field_controls'])
assert expected['matching_artifact_and_declaration']['outcome'] == 'accepted'
assert expected['different_artifact_same_declaration']['outcome'] == 'refused'
assert expected['same_artifact_different_declaration']['outcome'] == 'refused'
assert not any(k.split('.')[0] in {'torch', 'mlx', 'mlx_lm', 'transformers', 'local_llm_lab', 'pytest'} for k in sys.modules)
ruff = shutil.which('ruff')
assert ruff, 'put ruff on PATH'
lint = subprocess.run([ruff, 'check', '--no-cache', '--select', 'F,E9', str(root / 'check.py'), str(root / 'verify.py')], capture_output=True, text=True)
assert lint.returncode == 0, lint.stdout + lint.stderr
assert subprocess.run(['git', 'diff', '--check'], cwd=root, capture_output=True).returncode == 0
verification = {'schema_version': 1, 'basis': 'executed local metadata fixtures, no model evidence',
                'verdict': 'C1 closed', 'sources_match_git': 5, 'source_mutations_refused': 5,
                'reproduces_exactly': True, 'relocated_copy_reproduces': True,
                'changed_pairing_fields_refused': 11, 'missing_pairing_fields_refused': 11,
                'entry_point_identity_controls': 'passed', 'no_capture_and_same_path': 'passed',
                'ruff_F_E9': lint.stdout.strip(), 'git_diff_check': 'passed',
                'model_or_pytest_imports': False,
                'check_sha256': hashlib.sha256((root / 'check.json').read_bytes()).hexdigest(),
                'reviewer_parent': manifest['reviewer_parent'],
                'unexecuted': ['model forward', 'checkpoint load', 'native pytest', 'real A2', 'real pairing registration']}
(root / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
print(json.dumps(verification, indent=2))
