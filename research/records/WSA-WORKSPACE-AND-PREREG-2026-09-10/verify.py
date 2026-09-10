"""Verify frozen identities, reproducibility, controls and import isolation; no native tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import audit

HERE = Path(__file__).resolve().parent


def serialized(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def main():
    observed = audit.compute()
    recorded = json.loads((HERE/'audit.json').read_text())
    assert serialized(observed) == serialized(recorded), 'recorded analysis differs'
    guard = audit.NoModels()
    for name in sorted(audit.FORBIDDEN):
        try:
            guard.find_spec(name)
        except RuntimeError:
            pass
        else:
            raise AssertionError('import guard did not refuse ' + name)
    assert not (audit.FORBIDDEN & set(sys.modules)), 'forbidden package imported'

    assert observed['capture']['resume_identity_or_storage_mismatches_refused'] == 0
    assert observed['capture']['recorded_path']['forward_batch'] == 1
    # The true primitive outputs are also checked independently of the recorded JSON.
    a = observed['analytic']
    assert a['softmax_crossing']['width_layers'] == 1
    assert a['six_class_rank']['contrast_rank'] == 5
    assert a['local_relay']['two_local_layers_reach'] and not a['local_relay']['broken_relay_reaches']
    assert a['attention_mass']['contributions'] == [0, 1]
    assert a['five_fold_dependence']['true_cv_mean_variance'] == 0.25
    assert a['five_fold_dependence']['fixed_score_bootstrap_variance_every_dataset'] == 0
    assert a['gram_spectrum']['nonnegative_squared_error_b'] == 1
    assert a['gram_spectrum']['nonnegative_squared_error_c'] == 0
    assert observed['carrier']['producer_cells']['ten_groups']['fully_ablated_episodes'] == 1
    assert observed['prereg']['workspace_residual_bytes_two_positions_excluding_embedding']['4b'] == 7629*2*34*2560*4

    with tempfile.TemporaryDirectory(prefix='wsa-review-verification-') as temp:
        relocated = Path(temp)/'record'
        shutil.copytree(HERE, relocated, ignore=shutil.ignore_patterns('__pycache__'))
        manifest = json.loads((relocated/'sources.json').read_text())
        rejected = []
        for row in manifest['inputs']:
            path = relocated/row['snapshot']
            original = path.read_bytes()
            path.write_bytes(original+b'\nCONTROL: changed input\n')
            try:
                audit.verify_sources(relocated)
            except ValueError as error:
                assert row['snapshot'] in str(error)
                rejected.append(row['snapshot'])
            else:
                raise AssertionError('changed source accepted: '+row['snapshot'])
            finally:
                path.write_bytes(original)
        assert len(rejected) == observed['source_files_verified'] == 22
        process = subprocess.run([sys.executable, str(relocated/'audit.py')], cwd=temp,
                                 text=True, capture_output=True, check=True)
        assert serialized(json.loads(process.stdout)) == serialized(observed)

    # Exercise the output receipt comparison on each top-level analytic result, not only a sum.
    output_rejections = []
    for key in sorted(observed['analytic']):
        changed = copy.deepcopy(recorded)
        changed['analytic'][key] = {'CONTROL': 'wrong result'}
        assert serialized(changed) != serialized(observed)
        output_rejections.append(key)

    result = {
        'basis': 'fresh model-free verification',
        'source_mutations_refused': rejected,
        'source_count': len(rejected),
        'recorded_output_reproduced': True,
        'relocated_output_reproduced': True,
        'independent_analytic_assertions_passed': True,
        'output_receipt_mutations_detected': output_rejections,
        'forbidden_import_guard_exercised': sorted(audit.FORBIDDEN),
        'forbidden_packages_imported': [],
        'analytic_script_sha256': hashlib.sha256((HERE/'audit.py').read_bytes()).hexdigest(),
        'verification_script_sha256': hashlib.sha256((HERE/'verify.py').read_bytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
