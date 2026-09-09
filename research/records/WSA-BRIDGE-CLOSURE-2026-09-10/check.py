"""Close the lens artifact/declaration identity review using pinned metadata-only call paths."""
import copy
import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
BASE = 'google/gemma-3-4b-it'


def verify_sources(root):
    manifest = json.loads((root / 'sources.json').read_text())
    for row in manifest['sources']:
        raw = (root / row['snapshot']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256'] or len(raw) != row['bytes']:
            raise ValueError('source changed: ' + row['snapshot'])
    return manifest


def check(root):
    manifest = verify_sources(root)
    # Reuse only the metadata extractor, digest and result helpers. The old audit's main does not run.
    helpers = runpy.run_path(str(root / 'source/previous-replay.py'))
    raw = json.loads((root / 'source/anchor_sensitivity.json').read_text())
    table = {k: v for k, v in raw.items() if not k.startswith('_')}
    assert all(not e['measured_pairings'] for e in table.values())
    env = helpers['metadata_env'](root, table)
    env['_validate_anchor_table'](table)
    nu = {'estimator': 'synthetic fixture',
          'precision': {'fit_dtype': 'float32', 'forward_batch': 1, 'anchor_batch': 1},
          'endpoint': {'source_layers_repo': [1], 'target_layer_repo': 2},
          'position_weighting': {'source_reduction': 'mean', 'target_reduction': 'sum'},
          'corpus': {'identity': 'synthetic fixture; no model data'}}
    reading = {'positions': [8, 127], 'reduction': 'one position at a time',
               'endpoint': 'synthetic residual endpoint', 'context_tokens': 128}
    arrays = {'A': [[1.0, 0.0], [0.0, 0.001]], 'B': [[0.001, 0.0], [0.0, 1.0]]}
    lenses = {name: SimpleNamespace(maps={1: matrix}, sha256=helpers['digest'](matrix),
                hidden_size=2, identity=SimpleNamespace(base=BASE, num_layers=2))
              for name, matrix in arrays.items()}
    dictionary = SimpleNamespace(config={'model_name': BASE}, hook_point='model.layers.0.output', hidden_size=2)
    def align(lens, declaration=nu, *, capture='bfloat16'):
        return helpers['result'](lambda: env['hook_alignment'](
            dictionary, lens, lens_fit_dtype='float32', nu=declaration,
            capture_dtype=capture, capture_batch=None if capture is None else 1, reading=reading))
    empty = align(lenses['A'])
    assert empty['outcome'] == 'refused'
    identity = env['reading_identity'](nu, lens_sha256=lenses['A'].sha256,
                                      capture_dtype='bfloat16', capture_batch=1, reading=reading)
    measurement = {'pair': identity, 'relative': 0.004, 'basis': 'synthetic fixture, not measured'}
    table[BASE]['measured_pairings']['1'] = measurement
    matching = align(lenses['A'])
    changed_artifact = align(lenses['B'])
    assert matching['outcome'] == 'accepted'
    assert changed_artifact['outcome'] == 'refused'
    assert 'lens_sha256' in changed_artifact['reason'] and 'nu_sha256' not in changed_artifact['reason']
    changed_declaration = align(lenses['A'], {**nu, 'changed_declaration': True})
    assert changed_declaration['outcome'] == 'refused' and 'nu_sha256' in changed_declaration['reason']
    no_hash_lens = copy.copy(lenses['A'])
    del no_hash_lens.sha256
    missing_hash = align(no_hash_lens)
    assert missing_hash['outcome'] == 'refused' and 'lens_sha256' in missing_hash['reason']
    no_capture = align(lenses['A'], capture=None)
    same_path = align(lenses['A'], capture='float32')
    assert no_capture['outcome'] == same_path['outcome'] == 'accepted'
    controls = []
    for field in env['PAIR_FIELDS']:
        changed = copy.deepcopy(identity)
        value = changed[field]
        changed[field] = value + 1 if isinstance(value, int) else value + ['changed'] if isinstance(value, list) else str(value) + '-changed'
        pairing, reason = env['path_pairing'](1, base=BASE, identity=changed)
        assert pairing is None and field in reason
        missing = {k: v for k, v in identity.items() if k != field}
        absent, why = env['path_pairing'](1, base=BASE, identity=missing)
        assert absent is None and field in why
        controls.append({'field': field, 'changed': 'refused', 'missing': 'refused'})
    assert len(controls) == 11
    table[BASE]['measured_pairings']['1'] = {**measurement, 'pair': {k: v for k, v in identity.items() if k != 'lens_sha256'}}
    legacy_registration = align(lenses['A'])
    assert legacy_registration['outcome'] == 'refused' and 'lens_sha256' in legacy_registration['reason']
    assert not any(n.split('.')[0] in {'local_llm_lab', 'torch', 'mlx', 'mlx_lm', 'transformers', 'pytest'} for n in env['_imports_attempted'])
    return {'schema_version': 1, 'verdict': 'C1 closed on metadata fixtures; no new finding',
            'basis': 'pinned entry-point replay with synthetic lens stand-ins, not device evidence',
            'source_commit': manifest['source_commit'], 'order_commit': manifest['order_commit'],
            'shipped_measured_pairings': 0, 'sources_verified': len(manifest['sources']),
            'empty_table': empty, 'matching_artifact_and_declaration': matching,
            'different_artifact_same_declaration': changed_artifact,
            'same_artifact_different_declaration': changed_declaration,
            'lens_without_archive_hash': missing_hash,
            'old_registration_without_archive_hash': legacy_registration,
            'no_capture': no_capture, 'same_path': same_path, 'field_controls': controls,
            'fixture_matrix_json_sha256': {name: lens.sha256 for name, lens in lenses.items()},
            'shared_declaration_sha256': env['nu_digest'](nu),
            'unexecuted': ['real pairing registration', 'real A2', 'checkpoint load', 'model forward', 'native pytest']}


if __name__ == '__main__':
    print(json.dumps(check(ROOT), indent=2, sort_keys=True, allow_nan=False) + '\n', end='')
