"""Verify receipts, source tampering refusal, relocation and relevant semantic negative controls."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import check

HERE = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def main():
    result = check.compute()
    assert canonical(result) == canonical(json.loads((HERE/'check.json').read_text()))
    assert result['source_count'] == 16
    assert result['repeat']['failure_controls_refused'] == 4
    assert len(result['capture']['seam_controls_refused']) == 10
    assert sum(r['refused'] for r in result['capture']['resume_controls'].values()) == 5
    assert result['capture']['new_request_gap'] == {
        'fresh_refused': True, 'resume_complete': True, 'resume_verified': 1}
    assert result['gate_counts'] == {'cannot_certify_claimed_target': 5,
                                     'demonstrated_can_disagree': 4, 'unexecuted_or_unknown': 4}
    assert result['mathematics']['pca']['best_accuracy_from_PCA_h'] == 0.5
    assert result['mathematics']['pca']['accuracy_from_PCA_Jh'] == 1.0
    assert result['mathematics']['transfer_null']['unchanged_training_label_assignments'] == 36
    assert not (check.FORBIDDEN & set(sys.modules))
    blocked = []
    for module in sorted(check.FORBIDDEN):
        try:
            check.NoNative().find_spec(module)
        except RuntimeError:
            blocked.append(module)
        else:
            raise AssertionError(module)

    semantic_mutations_detected = []
    with tempfile.TemporaryDirectory(prefix='wsa-closure-verify-') as temp:
        replica = Path(temp)/'review'
        shutil.copytree(HERE, replica, ignore=shutil.ignore_patterns('__pycache__'))
        manifest = json.loads((replica/'sources.json').read_text())
        refused = []
        for row in manifest['inputs']:
            path = replica/row['snapshot']
            original = path.read_bytes()
            path.write_bytes(original+b'\nSOURCE CORRUPTION\n')
            try:
                check.verify_sources(replica)
            except ValueError as error:
                assert row['snapshot'] in str(error)
                refused.append(row['snapshot'])
            else:
                raise AssertionError(row['snapshot'])
            finally:
                path.write_bytes(original)
        run = subprocess.run([sys.executable, '-B', str(replica/'check.py')], cwd=temp,
                             text=True, capture_output=True, check=True)
        assert canonical(json.loads(run.stdout)) == canonical(result)

        # Restore the exact old verdict defect. Invoke the local semantic checks directly so this
        # tests detection of the defect, independently of the already-tested source-hash refusal.
        p = replica/'source/repeat_gate.py'
        original = p.read_text()
        broken = original.replace('within["relative_frobenius_difference"] == 0.0\n                   and ', '')
        assert broken != original
        p.write_text(broken)
        try:
            check.repeat_checks(replica)
        except AssertionError:
            semantic_mutations_detected.append('repeat ignores within-process failure')
        else:
            raise AssertionError('old repeat defect not detected')
        p.write_text(original)

        # Restore membership-only resume at the seam between metadata functions.
        ns = check.capture_namespace(replica)

        def membership_only(path, target, corpus):
            if not path.exists():
                return set(), -1
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            return {(r['task_id'], r['step']) for r in rows}, max(r['shard'] for r in rows)

        ns['_verified_captures'] = membership_only
        try:
            check.capture_checks(ns)
        except AssertionError:
            semantic_mutations_detected.append('resume reverts to manifest membership')
        else:
            raise AssertionError('old resume defect not detected')

        # Restore the config-only identity at both actual consumers by rewriting only their call ASTs.
        for name in ('runner.py', 'repeat_gate.py'):
            path = replica/'source'/name
            tree = ast.parse(path.read_text())
            call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and ((isinstance(n.func, ast.Name) and n.func.id == 'checkpoint_identity')
                             or (isinstance(n.func, ast.Attribute) and n.func.attr == 'checkpoint_identity')))
            call.args[0] = ast.parse("{'config.json': report['sha256']['config.json']}", mode='eval').body
            path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
        try:
            check.identity_checks(replica, check.capture_namespace(replica))
        except ValueError:
            semantic_mutations_detected.append('consumer passes config-only manifest')
        else:
            raise AssertionError('config-only consumer not detected')

    report = {'basis': 'fresh file-only verification', 'source_count': len(refused),
              'source_corruptions_refused': refused, 'recorded_output_reproduced': True,
              'relocated_output_reproduced': True,
              'semantic_mutations_detected': semantic_mutations_detected,
              'blocked_imports': blocked, 'forbidden_modules_imported': [],
              'check_sha256': hashlib.sha256((HERE/'check.py').read_bytes()).hexdigest(),
              'verify_sha256': hashlib.sha256((HERE/'verify.py').read_bytes()).hexdigest(),
              'unexecuted': result['unexecuted']}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
