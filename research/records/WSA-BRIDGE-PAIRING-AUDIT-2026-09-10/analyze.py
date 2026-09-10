"""Pure-metadata replay of a bridge guard. No package/model imports or forwards."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
PURE_FUNCTIONS = {'anchor_sensitivity', 'measured_layers', 'path_pairing',
                  'path_term_for_layer', 'lens_fit_precision', 'fit_precision_record'}
BASE = 'google/gemma-3-4b-it'


def verify_sources(root):
    manifest = json.loads((root / 'sources.json').read_text())
    for r in manifest['sources']:
        raw = (root / r['snapshot']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != r['sha256'] or len(raw) != r['bytes']:
            raise ValueError('source changed: ' + r['snapshot'])
    return manifest


def metadata_functions(source, table):
    """Compile only the six inspected metadata functions, never module-level imports/code."""
    parsed = ast.parse(source)
    defs = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in PURE_FUNCTIONS]
    assert {n.name for n in defs} == PURE_FUNCTIONS
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for d in defs for n in ast.walk(d))
    env = {'anchor_table': lambda: table}
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'LENS_PATH_TERM', 'FIT_PRECISION_KEYS'}:
                    env[target.id] = ast.literal_eval(node.value)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *defs], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<frozen metadata functions only>', 'exec'), env)
    return env


def attempt(env, *, dtype='float32', capture='bfloat16', width=1, layer=1):
    nu = {'precision': {'fit_dtype': dtype, 'forward_batch': 1, 'anchor_batch': 1}}
    try:
        result = env['fit_precision_record'](nu, declared=dtype, layer=layer, base=BASE,
                                              capture_dtype=capture, capture_batch=width)
    except ValueError as error:
        return {'outcome': 'refused', 'message': str(error)}
    return {'outcome': 'accepted', 'record': result}


def pairing_replay(source, shipped):
    table = copy.deepcopy(shipped)
    env = metadata_functions(source, table)
    checks = {'unmeasured_crossing': attempt(env),
              'same_dtype_same_width': attempt(env, capture='float32'),
              'no_capture': attempt(env, capture=None, width=None)}
    assert checks['unmeasured_crossing']['outcome'] == 'refused'
    assert checks['same_dtype_same_width']['outcome'] == 'accepted'
    assert checks['no_capture']['outcome'] == 'accepted'
    # This is synthetic evidence, not a measured crossing or a proposed production schema.
    # Its declared scope makes the non-transfer requirement explicit for the counterexample.
    measurement = {'relative': 0.004, 'basis': 'synthetic metadata fixture, not measured',
                   'fit_precision': {'fit_dtype': 'float32', 'forward_batch': 1, 'anchor_batch': 1},
                   'capture_precision': {'dtype': 'bfloat16', 'forward_batch': 1},
                   'context_tokens': 128, 'positions': [8, 127],
                   'lens_identity': 'fixture-lens-A', 'capture_path': 'fixture-native-A'}
    table[BASE]['measured_pairings']['1'] = measurement
    checks.update({'matching_fixture_scope': attempt(env),
                   'different_capture_precision': attempt(env, capture='float16'),
                   'different_capture_width': attempt(env, width=64),
                   'reversed_precision_direction': attempt(env, dtype='bfloat16', capture='float32'),
                   'neighbor_unmeasured': attempt(env, layer=17)})
    assert checks['neighbor_unmeasured']['outcome'] == 'refused'
    for name in ['matching_fixture_scope', 'different_capture_precision',
                 'different_capture_width', 'reversed_precision_direction']:
        assert checks[name]['outcome'] == 'accepted'
    table[BASE]['measured_pairings']['1'] = {'relative': 0.004}
    checks['measurement_missing_pair_identity'] = attempt(env)
    assert checks['measurement_missing_pair_identity']['outcome'] == 'accepted'
    checks['measurement_fixture'] = measurement
    checks['basis'] = 'executed pure metadata functions extracted unchanged from pinned source; no real pairings exist'
    return checks


def amplification_test_replay(test_source, shipped):
    name = 'test_no_amplification_ratio_survives_anywhere_in_the_shipped_measurements'
    parsed = ast.parse(test_source)
    target = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == name)
    # The selected test imports only json. No test module, conftest, or package is imported.
    imports = [n for n in ast.walk(target) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 1 and isinstance(imports[0], ast.Import)
    assert [a.name for a in imports[0].names] == ['json']
    def run(table):
        env = {'B': SimpleNamespace(anchor_table=lambda: table)}
        exec(compile(ast.Module(body=[target], type_ignores=[]), '<one metadata test>', 'exec'), env)
        try:
            env[name]()
        except AssertionError:
            return 'refused'
        return 'accepted'
    clean = copy.deepcopy(shipped)
    nested = copy.deepcopy(shipped)
    nested[BASE]['layers']['1']['sampled_displacement']['float32']['amplification_ratio'] = 539
    immediate = copy.deepcopy(shipped)
    immediate[BASE]['layers']['1']['amplification_ratio'] = 539
    result = {'clean_table': run(clean), 'nested_withdrawn_ratio': run(nested),
              'immediate_layer_ratio_control': run(immediate),
              'basis': 'extracted test replay with synthetic corruptions; shipped table was not changed'}
    assert result == {**result, 'clean_table': 'accepted', 'nested_withdrawn_ratio': 'accepted',
                      'immediate_layer_ratio_control': 'refused'}
    return result


def numerical_table(table_text, rows):
    table = json.loads(table_text, parse_float=Decimal)[BASE]['layers']
    output = []
    for layer, groups in table.items():
        for name, phases in groups.items():
            if name == 'native_anchor_read_in_the_float32_tail':
                selections = [('float32', phases, 'a_at_native_anchor')]
            else:
                target = 'a_plus_delta' if name == 'sampled_displacement' else 'a_plus_random'
                selections = [(phase, group, target) for phase, group in phases.items()]
            for phase, declared, field in selections:
                part = [r for r in rows if r['repo_layer'] == int(layer) and r['phase'] == phase]
                values = [abs(r[field] - r['a_base']) / abs(r['a_base']) for r in part]
                observed = {'median': statistics.median(values), 'min': min(values),
                            'max': max(values), 'count': len(values)}
                for metric, stored in declared.items():
                    places = -stored.as_tuple().exponent if isinstance(stored, Decimal) else 0
                    rounded = f'{observed[metric]:.{places}f}'
                    output.append({'repo_layer': int(layer), 'group': name, 'precision': phase,
                                   'metric': metric, 'stored_text': str(stored),
                                   'computed': observed[metric], 'rounded_at_declared_precision': rounded,
                                   'matches': Decimal(rounded) == stored})
    assert len(output) == 60
    return output


def analyze(root):
    manifest = verify_sources(root)
    source = root / 'source'
    table_text = (source / 'anchor_sensitivity.json').read_text()
    table = {k: v for k, v in json.loads(table_text).items() if not k.startswith('_')}
    assert all(not v['measured_pairings'] for v in table.values())
    rows = [json.loads(line) for line in (source / 'displacement.jsonl').read_text().splitlines()]
    assert len(rows) == 108
    numbers = numerical_table(table_text, rows)
    return {'schema_version': 1, 'source_commit': manifest['source_commit'],
            'order_commit': manifest['order_commit'], 'sources_verified': len(manifest['sources']),
            'shipped_measured_pairings': 0,
            'pairing_scope_replay': pairing_replay((source / 'sae_bridge.py').read_text(), table),
            'withdrawn_ratio_test_replay': amplification_test_replay((source / 'test_sae_bridge.py').read_text(), table),
            'sensitivity_table_cells': numbers,
            'sensitivity_table_mismatches': [row for row in numbers if not row['matches']],
            'unexecuted': ['model or device operations', 'native pytest', 'full module import',
                           'real A2 residual read', 'real cross-path pairing measurement']}


if __name__ == '__main__':
    print(json.dumps(analyze(ROOT), indent=2, sort_keys=True, allow_nan=False) + '\n', end='')
