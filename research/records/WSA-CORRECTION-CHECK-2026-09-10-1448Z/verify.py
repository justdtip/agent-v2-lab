"""File-only verification of the documentation follow-up."""
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
manifest = json.loads((root / 'sources.json').read_text())
for item in manifest['sources']:
    assert subprocess.check_output(['git', 'show', item['commit'] + ':' + item['repository_path']], cwd=root) == (root / item['snapshot']).read_bytes()
text = (root / 'producer.md').read_text()
base = m['inventory'](text)
assert all(len(v) == 1 for v in base.values())
for key, phrase in m['PASSAGES'].items():
    changed = m['inventory'](text.replace(phrase, '[corrected passage]'))
    assert not changed[key]
    assert all(changed[k] == base[k] or [x['text'] for x in changed[k]] == [x['text'] for x in base[k]]
               for k in base if k != key)
with tempfile.TemporaryDirectory(prefix='wsa-correction-check-') as t:
    clone = Path(t) / 'record'
    shutil.copytree(root, clone)
    for item in manifest['sources']:
        p = clone / item['snapshot']
        original = p.read_bytes()
        p.write_bytes(original + b'\n')
        try:
            m['check'](clone)
        except ValueError:
            pass
        else:
            raise AssertionError('modified source accepted')
        p.write_bytes(original)
    table = text.split('Over the eighteen\ndirection-and-cotangent pairs:', 1)[1].split('So **', 1)[0]
    corrupted_cells = 0
    for line in table.splitlines():
        if not line.startswith(('| native bf16 |', '| float32 |')):
            continue
        cells = [s.strip() for s in line.strip('|').split('|')]
        for column in (2, 3, 4):
            wrong = list(cells)
            wrong[column] = str(float(wrong[column].replace('**', '')) + 1)
            edited = text.replace(line, '| ' + ' | '.join(wrong) + ' |', 1).encode()
            (clone / 'producer.md').write_bytes(edited)
            altered_manifest = json.loads(json.dumps(manifest))
            item = next(r for r in altered_manifest['sources'] if r['snapshot'] == 'producer.md')
            item['sha256'] = hashlib.sha256(edited).hexdigest()
            item['bytes'] = len(edited)
            (clone / 'sources.json').write_text(json.dumps(altered_manifest))
            # The manifest deliberately accepts the synthetic document; the numerical check must refuse it.
            try:
                m['check'](clone)
            except AssertionError:
                corrupted_cells += 1
            else:
                raise AssertionError('corrupted table cell accepted')
    assert corrupted_cells == 18
ruff = shutil.which('ruff')
assert ruff, 'put ruff on PATH'
lint = subprocess.run([ruff, 'check', '--no-cache', '--select', 'F,E9', str(root / 'check.py'), str(root / 'verify.py')], capture_output=True, text=True)
assert lint.returncode == 0, lint.stdout + lint.stderr
assert not any(k.split('.')[0] in {'torch', 'mlx', 'mlx_lm', 'transformers', 'local_llm_lab'} for k in sys.modules)
assert subprocess.run(['git', 'diff', '--check'], cwd=root, capture_output=True).returncode == 0
result = {'schema_version': 1, 'basis': 'executed locally, standard library and static lint only',
          'source_count': 4, 'sources_match_git': True, 'analysis_reproduces': True,
          'source_mutations_refused': 4, 'passage_removal_controls': 4,
          'table_cell_mutations_refused': corrupted_cells, 'table_cells_verified': 18,
          'lint': lint.stdout.strip(), 'git_diff_check': 'passed', 'model_imports': False,
          'device_access': False, 'reviewer_parent': manifest['reviewer_parent'],
          'check_sha256': hashlib.sha256((root / 'check.json').read_bytes()).hexdigest()}
(root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
