"""Metadata-only verification and a two-matrix identity counterexample; no model imports."""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
BASE = 'google/gemma-3-4b-it'
FUNCTIONS = {'_check_keys', '_check_cell', '_validate_anchor_table', 'anchor_sensitivity',
             'measured_layers', '_validate_pairing', 'nu_digest', 'reading_identity',
             'path_pairing', 'path_term_for_layer', 'lens_fit_precision', 'fit_precision_record',
             'layer_for_hook', 'hook_alignment'}
CONSTANTS = {'LENS_PATH_TERM', 'FIT_PRECISION_KEYS', 'PAIR_FIELDS', 'READING_FIELDS',
             '_CELL_KEYS', '_ARM_KEYS', '_LAYER_KEYS', '_MODEL_KEYS'}


def sources(root):
    manifest = json.loads((root / 'sources.json').read_text())
    for row in manifest['sources']:
        raw = (root / row['snapshot']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256'] or len(raw) != row['bytes']:
            raise ValueError('source changed: ' + row['snapshot'])
    return manifest


def metadata_env(root, table):
    source = (root / 'source/sae_bridge.py').read_text()
    parsed = ast.parse(source)
    definitions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    assert {n.name for n in definitions} == FUNCTIONS
    imports_attempted = []
    def guarded_import(name, *args, **kwargs):
        imports_attempted.append(name)
        if name.split('.')[0] in {'local_llm_lab', 'torch', 'mlx', 'mlx_lm', 'transformers', 'pytest'}:
            raise AssertionError('forbidden package import attempted: ' + name)
        return builtins.__import__(name, *args, **kwargs)
    env = {'__builtins__': {**vars(builtins), '__import__': guarded_import},
           'hashlib': hashlib, 'json': json, 'ANCHOR_SENSITIVITY_PATH': root / 'source/anchor_sensitivity.json',
           'anchor_table': lambda: table,
           # Fixture models already use the canonical base id; no registry is imported.
           'dictionary_base': lambda dictionary: dictionary.config['model_name']}
    for node in parsed.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name == '_HOOK':
                env[name] = re.compile(ast.literal_eval(node.value.args[0]))
            elif name in CONSTANTS:
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'frozenset':
                    env[name] = frozenset(ast.literal_eval(node.value.args[0]))
                else:
                    env[name] = ast.literal_eval(node.value)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *definitions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<frozen metadata-only call paths>', 'exec'), env)
    env['_imports_attempted'] = imports_attempted
    return env


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def result(call):
    try:
        value = call()
    except ValueError as error:
        return {'outcome': 'refused', 'reason': str(error)}
    return {'outcome': 'accepted', 'result': value}


def numeric_table(table, rows):
    checks = []
    for layer, groups in table[BASE]['layers'].items():
        for group, content in groups.items():
            if group == 'native_anchor_read_in_the_float32_tail':
                choices = [('float32', content, 'a_at_native_anchor')]
            else:
                column = 'a_plus_delta' if group == 'sampled_displacement' else 'a_plus_random'
                choices = [(phase, cell, column) for phase, cell in content.items()]
            for phase, declared, column in choices:
                part = [r for r in rows if r['phase'] == phase and r['repo_layer'] == int(layer)]
                values = [abs(r[column] - r['a_base']) / abs(r['a_base']) for r in part]
                computed = {'min': min(values), 'max': max(values), 'median': statistics.median(values), 'count': len(values)}
                for key, value in declared.items():
                    places = 6 if key == 'median' else 0 if key == 'count' else 4
                    rounded = round(computed[key], places)
                    checks.append({'layer': int(layer), 'group': group, 'phase': phase,
                                   'quantity': key, 'recorded': value, 'computed_rounded': rounded,
                                   'matches': value == rounded})
    assert len(checks) == 60
    return checks


def analyze(root):
    manifest = sources(root)
    shipped = json.loads((root / 'source/anchor_sensitivity.json').read_text())
    table = {k: v for k, v in shipped.items() if not k.startswith('_')}
    assert all(not entry['measured_pairings'] for entry in table.values())
    env = metadata_env(root, table)
    env['_validate_anchor_table'](table)
    nu = {'estimator': 'synthetic fixture',
          'precision': {'fit_dtype': 'float32', 'forward_batch': 1, 'anchor_batch': 1},
          'endpoint': {'source_layers_repo': [1], 'target_layer_repo': 2},
          'position_weighting': {'source_reduction': 'mean', 'target_reduction': 'sum'},
          'corpus': {'identity': 'synthetic fixture; no model data'}}
    reading = {'positions': [8, 127], 'reduction': 'one position at a time',
               'endpoint': 'synthetic residual endpoint', 'context_tokens': 128}
    identity = env['reading_identity'](nu, capture_dtype='bfloat16', capture_batch=1, reading=reading)
    def record(**overrides):
        args = {'declared': 'float32', 'layer': 1, 'base': BASE,
                'capture_dtype': 'bfloat16', 'capture_batch': 1, 'reading': reading}
        args.update(overrides)
        return env['fit_precision_record'](nu, **args)
    empty = result(record)
    assert empty['outcome'] == 'refused'
    table[BASE]['measured_pairings']['1'] = {'pair': identity, 'relative': 0.004,
                                           'basis': 'synthetic fixture; not measured'}
    matching = result(record)
    assert matching['outcome'] == 'accepted'
    fields = []
    for field in env['PAIR_FIELDS']:
        changed = copy.deepcopy(identity)
        original = changed[field]
        changed[field] = original + 1 if isinstance(original, int) else original + ['changed'] if isinstance(original, list) else str(original) + '-changed'
        measured, why = env['path_pairing'](1, base=BASE, identity=changed)
        assert measured is None and field in why
        fields.append({'field': field, 'outcome': 'refused', 'reason': why})
    missing = []
    for field in env['PAIR_FIELDS']:
        incomplete = {k: v for k, v in identity.items() if k != field}
        measured, why = env['path_pairing'](1, base=BASE, identity=incomplete)
        assert measured is None and field in why
        missing.append(field)
    front = {'capture_precision': result(lambda: record(capture_dtype='float16')),
             'capture_width': result(lambda: record(capture_batch=64)),
             'context': result(lambda: record(reading={**reading, 'context_tokens': 1400})),
             'neighbor': result(lambda: record(layer=17)),
             'missing_reading': result(lambda: record(reading=None))}
    assert all(v['outcome'] == 'refused' for v in front.values())
    same = result(lambda: record(capture_dtype='float32'))
    no_capture = result(lambda: record(capture_dtype=None, capture_batch=None))
    assert same['outcome'] == no_capture['outcome'] == 'accepted'

    # The two lenses have the same declaration/base/shape but different matrices and content hashes.
    arrays = {'A': [[1.0, 0.0], [0.0, 0.001]], 'B': [[0.001, 0.0], [0.0, 1.0]]}
    dictionary = SimpleNamespace(config={'model_name': BASE}, hook_point='model.layers.0.output', hidden_size=2)
    lenses = {name: SimpleNamespace(maps={1: matrix}, sha256=digest(matrix), hidden_size=2,
                                    identity=SimpleNamespace(base=BASE, num_layers=2))
              for name, matrix in arrays.items()}
    outcomes = {}
    for name, lens in lenses.items():
        outcomes[name] = result(lambda lens=lens: env['hook_alignment'](
            dictionary, lens, lens_fit_dtype='float32', nu=nu, capture_dtype='bfloat16',
            capture_batch=1, reading=reading))
    assert all(v['outcome'] == 'accepted' for v in outcomes.values())
    changed_nu = {**nu, 'changed_fixture_declaration': True}
    changed = result(lambda: env['hook_alignment'](
        dictionary, lenses['B'], lens_fit_dtype='float32', nu=changed_nu,
        capture_dtype='bfloat16', capture_batch=1, reading=reading))
    assert changed['outcome'] == 'refused'
    h, dh = [100.0, 1.0], [0.0, 1.0]
    def apply(matrix, x):
        return [sum(a * b for a, b in zip(row, x)) for row in matrix]
    def norm(x):
        return math.sqrt(sum(v * v for v in x))
    errors = {name: norm(apply(matrix, dh)) / norm(apply(matrix, h)) for name, matrix in arrays.items()}
    assert errors['A'] < errors['B']

    schema_paths = [('amplification_ratio',), ('layers', '1', 'amplification_ratio'),
                    ('layers', '1', 'sampled_displacement', 'amplification_ratio'),
                    ('layers', '1', 'sampled_displacement', 'float32', 'amplification_ratio'),
                    ('layers', '1', 'equal_norm_random_displacement', 'native', 'amplification_ratio'),
                    ('layers', '1', 'native_anchor_read_in_the_float32_tail', 'amplification_ratio')]
    schema = []
    for path in schema_paths:
        corrupted = copy.deepcopy(table)
        node = corrupted[BASE]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = 539
        check = result(lambda: env['_validate_anchor_table'](corrupted))
        assert check['outcome'] == 'refused'
        schema.append({'path': list(path), **check})
    rows = [json.loads(line) for line in (root / 'source/displacement.jsonl').read_text().splitlines()]
    numeric = numeric_table(table, rows)
    assert all(c['matches'] for c in numeric)
    assert not any(name.split('.')[0] in {'local_llm_lab', 'torch', 'mlx', 'mlx_lm', 'transformers', 'pytest'} for name in env['_imports_attempted'])
    return {'schema_version': 1, 'source_commit': manifest['source_commit'],
            'order_commit': manifest['order_commit'], 'sources_verified': len(manifest['sources']),
            'basis': 'metadata fixture replay and elementary matrix example; no model measurements',
            'empty_pairing_table': empty, 'matching_pairing': matching,
            'declared_field_mutations': fields, 'missing_field_refusals': missing,
            'front_door_controls': front, 'same_path': same, 'no_capture': no_capture,
            'schema_mutations': schema, 'numeric_cells': numeric,
            'artifact_identity_counterexample': {'matrix_values': arrays,
                'matrix_json_sha256': {name: lens.sha256 for name, lens in lenses.items()},
                'shared_nu_sha256': env['nu_digest'](nu), 'hook_alignment_outcomes': outcomes,
                'different_nu_control': changed,
                'h': h, 'delta_h': dh, 'relative_readout_change': errors,
                'basis': 'two-dimensional synthetic lens stand-ins; JSON hashes identify fixture matrix bytes, not NPZ files'},
            'package_imports_attempted': env['_imports_attempted'],
            'unexecuted': ['model forwards', 'checkpoint load', 'native pytest', 'real A2', 'real pairing registration']}


if __name__ == '__main__':
    print(json.dumps(analyze(ROOT), indent=2, sort_keys=True, allow_nan=False) + '\n', end='')
