"""Independent receipt/source controls for the coverage follow-up, without native imports."""
from __future__ import annotations

import copy
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
    recorded = json.loads((HERE/'check.json').read_text())
    assert canonical(result) == canonical(recorded)
    assert result['repeat']['ignored_within_process_failures'] == 2
    assert result['repeat']['against_saved_failures_refused'] == 2
    assert len(result['capture']['resume_cases']) == 5
    assert result['capture']['missing_manifest_negative_control']['outstanding'] == 1
    assert result['capture']['changed_messages_refused']
    assert result['gate_counts'] == {'cannot_certify_claimed_target': 7,
                                   'demonstrated_can_disagree': 1, 'unexecuted_or_unknown': 5}
    assert not (check.FORBIDDEN & set(sys.modules))
    for name in sorted(check.FORBIDDEN):
        try:
            check.NoNative().find_spec(name)
        except RuntimeError:
            pass
        else:
            raise AssertionError(name)
    with tempfile.TemporaryDirectory(prefix='wsa-coverage-verification-') as temp:
        replica = Path(temp)/'review'
        shutil.copytree(HERE, replica, ignore=shutil.ignore_patterns('__pycache__'))
        manifest = json.loads((replica/'sources.json').read_text())
        refused = []
        for row in manifest['inputs']:
            path = replica/row['snapshot']
            original = path.read_bytes()
            path.write_bytes(original+b'\nCORRUPTION\n')
            try:
                check.verify_sources(replica)
            except ValueError as error:
                assert row['snapshot'] in str(error)
                refused.append(row['snapshot'])
            else:
                raise AssertionError(row['snapshot'])
            finally:
                path.write_bytes(original)
        assert len(refused) == 10
        p = subprocess.run([sys.executable, '-B', str(replica/'check.py')], cwd=temp,
                           text=True, capture_output=True, check=True)
        assert canonical(json.loads(p.stdout)) == canonical(result)
    mutations = []
    for section in ('repeat', 'checkpoint', 'capture', 'gate_counts'):
        altered = copy.deepcopy(recorded)
        altered[section] = {'incorrect': True}
        assert canonical(altered) != canonical(result)
        mutations.append(section)
    report = {'basis': 'fresh model-free verification', 'source_corruptions_refused': refused,
              'source_count': 10, 'recorded_output_reproduced': True,
              'relocated_output_reproduced': True, 'output_receipt_mutations_detected': mutations,
              'forbidden_imports': sorted(check.FORBIDDEN), 'forbidden_modules_imported': [],
              'repeat_cases': 5, 'resume_cases': 5, 'missing_manifest_control': 'refused completeness',
              'check_sha256': hashlib.sha256((HERE/'check.py').read_bytes()).hexdigest(),
              'verify_sha256': hashlib.sha256((HERE/'verify.py').read_bytes()).hexdigest(),
              'unexecuted': result['unexecuted']}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
