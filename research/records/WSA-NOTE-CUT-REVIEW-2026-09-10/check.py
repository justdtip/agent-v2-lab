"""Pinned source, synthetic offsets/attention and reporter fixtures. No model imports."""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from reporter_checks import run_checks

ROOT = Path(__file__).resolve().parent


class Mask:
    def __init__(self, shape):
        assert shape[:2] == (1, 1) and shape[-1] == shape[-2]
        self.size = shape[-1]
        self.cells = set()

    def __setitem__(self, key, value):
        assert key[:2] == (0, 0) and value is True
        queries = range(*key[2].indices(self.size))
        self.cells.update((q, k) for q in queries for k in key[3])

    def clone(self):
        result = Mask((1, 1, self.size, self.size))
        result.cells = self.cells.copy()
        return result


def assignment(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets
    )


def compiled(nodes, label):
    return compile(ast.Module(body=nodes, type_ignores=[]), label, 'exec')


def outer_loop(tree):
    return next(n for n in tree.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Tuple))


def build_mask(tree, pieces, prompt_tokens=2):
    # Pieces are hand-labelled synthetic tokenisation, not a Gemma tokenizer claim.
    comp = ''.join(pieces)
    offsets = []
    end = 0
    for part in pieces:
        offsets.append((end, end + len(part)))
        end += len(part)
    body = outer_loop(tree).body
    start = next(i for i, n in enumerate(body) if assignment(n, 'note_end'))
    stop = next(i for i, n in enumerate(body) if assignment(n, 'n_all'))
    size = prompt_tokens + len(pieces)
    env = {'comp': comp, 'FENCE': '```json', 't_idx': len(pieces) - 1,
           'enc_c': SimpleNamespace(offset_mapping=offsets),
           'p_ids': list(range(prompt_tokens)), 'S': size, 'P_act': size - 2,
           'P_note': prompt_tokens - 1, 'kinds': ['format'] * size,
           'KINDS': ['older_note'],
           'torch': SimpleNamespace(zeros=lambda *s, dtype: Mask(s), bool=bool)}
    exec(compiled(body[start:stop], '<source mask builder>'), env)
    return env, offsets


def edge_set(size, start, keys):
    return {(q, k) for q in range(start, size) for k in keys}


def placement(actual, expected):
    return {'passes': actual == expected,
            'missing_edges': sorted(expected - actual),
            'unexpected_edges': sorted(actual - expected)}


def gate_assertions(tree, *, cells, p_act=6, masked_all=0.0, far_mass=0.0,
                    p_note_changed=False, open_mass=1.0, leaky_masked=0.0):
    # Execute the actual assertions and P_note comparison, not a rewritten verdict.
    # Inputs are analytic nonnegative attention sums, not native torch outputs.
    assertions = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    pick = lambda text: next(n for n in assertions if text in ast.unparse(n.test))
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "arm == 'current_note'")
    note_readout = {'model_six_note': [[1.0, 0, 0, 0, 0, 0], 1.0],
                    'lens_note': {'1': {'six': [1.0, 0, 0, 0, 0, 0], 'mass': 1.0}},
                    'margin_note_logits': 1.0}
    actual_note = copy.deepcopy(note_readout)
    if p_note_changed:
        actual_note['margin_note_logits'] = 2.0
    env = {'extra': object(), 'n_blk_act': sum(q == p_act for q, _ in cells),
           'masked_all': masked_all, 'far_mass': far_mass, 'i': 0,
           'arm': 'current_note', 'rec_arm': actual_note,
           'row': {'arms': {'unmasked': note_readout}}, 'json': json,
           'passes': open_mass > 0.0 and leaky_masked == 0.0}
    env['rec_arm']['gate'] = {'masked_all': masked_all, 'far_mass': far_mass}
    checks = [('nonempty_P_act', pick('n_blk_act')),
              ('masked_attention_and_local_window', pick('masked_all')),
              ('P_note_selected_readouts', block),
              ('leaky_carrier_control', pick('passes'))]
    outcomes = {}
    for name, node in checks:
        try:
            exec(compiled([node], '<source gate ' + name + '>'), env)
            outcomes[name] = True
        except AssertionError:
            outcomes[name] = False
    return outcomes


def two_layer_attention(cells, size=8):
    # Valid nonnegative causal one-hot rows; residual identity paths stay available.
    layers = [{(q, q): 1.0 for q in range(size)} for _ in range(2)]
    for layer, q, k in [(0, 4, 3), (1, 6, 4)]:
        layers[layer].pop((q, q))
        layers[layer][q, k] = 1.0
    for layer in layers:
        for q, k in list(layer):
            if (q, k) in cells:
                del layer[q, k]
                allowed = next(j for j in [q] + list(range(q)) if (q, j) not in cells)
                layer[q, allowed] = 1.0
        assert all(k <= q and value >= 0 for (q, k), value in layer.items())
        assert all(sum(v for (qq, _), v in layer.items() if qq == q) == 1 for q in range(size))
    reachable = {3}
    by_layer = []
    for layer in layers:
        reachable |= {q for (q, k), v in layer.items() if v > 0 and k in reachable}
        by_layer.append(sorted(reachable))
    return {'attention': [{f'{q},{k}': v for (q, k), v in sorted(layer.items())}
                          for layer in layers],
            'masked_sum': sum(v for layer in layers for edge, v in layer.items() if edge in cells),
            'reachable_after_layers': by_layer, 'note_reaches_P_act': 6 in reachable}


def main():
    sources = json.loads((ROOT / 'sources.json').read_text())
    for item in sources['files']:
        assert hashlib.sha256((ROOT / item['frozen_as']).read_bytes()).hexdigest() == item['sha256']
    tree = ast.parse((ROOT / 'source/workspace_w3b.py').read_text())
    fixtures = [
        ('ordinary', ['Go.', '\n', '```json', '\n', '{"name":"', 'read_file'], [2, 3], 4, 0),
        ('one_note_token', ['Go.\n', '```json', '\n', '{"name":"', 'read_file'], [2], 3, 0),
        ('straddling_fence', ['Go.', '\n```', 'json', '\n', '{"name":"', 'read_file'], [2, 3], 4, 1),
        ('no_prose', ['```json', '\n', '{"name":"', 'read_file'], [], 2, 0),
    ]
    builder_results = []
    for name, pieces, keys, q0, straddle in fixtures:
        for n_prompt in [2, 47, 100, 1500]:
            env, offsets = build_mask(tree, pieces, n_prompt)
            translated_keys = [k + n_prompt - 2 for k in keys]
            assert env['note_keys'] == translated_keys
            assert env['q0'] == q0 + n_prompt - 2 and env['straddle'] == straddle
            if keys:
                expected = edge_set(env['S'], q0 + n_prompt - 2, translated_keys)
                actual = env['arms']['current_note'].cells
                assert actual == expected
                verdict = 'exact_expected_edges'
            else:
                assert 'current_note' not in env['arms'] and 'current_note' in env['skipped']
                verdict = 'skipped_no_prose'
            builder_results.append({'case': name, 'prompt_tokens': n_prompt,
                                    'offsets': offsets, 'note_keys': env['note_keys'],
                                    'q0': env['q0'], 'P_act': env['P_act'], 'S': env['S'],
                                    'fence_straddle_tokens': env['straddle'], 'verdict': verdict})
    expected = edge_set(8, 4, [2, 3])
    actuals = {'correct': expected, 'delayed_query_start': edge_set(8, 6, [2, 3]),
               'wrong_keys': edge_set(8, 4, [4]),
               'missing_one_note_key': edge_set(8, 4, [2]),
               'extra_unrelated_key': edge_set(8, 4, [0, 2, 3]),
               'future_only_key': edge_set(8, 4, [7]), 'empty': set()}
    mutations = []
    for name, cells in actuals.items():
        gates = gate_assertions(tree, cells=cells)
        oracle = placement(cells, expected)
        assert oracle['passes'] == (name == 'correct')
        assert all(gates.values()) == (name != 'empty')
        mutations.append({'case': name, 'actual_edges': sorted(cells),
                          'producer_assertions': gates, 'independent_placement': oracle})
    controls = []
    for name, kwargs, rejected in [
        ('positive_masked_attention', {'masked_all': .5}, 'masked_attention_and_local_window'),
        ('positive_local_far_attention', {'far_mass': .5}, 'masked_attention_and_local_window'),
        ('changed_P_note_readout', {'p_note_changed': True}, 'P_note_selected_readouts'),
        ('leaky_open_key_zero', {'open_mass': 0.0}, 'leaky_carrier_control'),
        ('leaky_masked_key_positive', {'leaky_masked': .5}, 'leaky_carrier_control'),
    ]:
        gates = gate_assertions(tree, cells=expected, **kwargs)
        assert not gates[rejected]
        controls.append({'case': name, 'producer_assertions': gates})
    relay = {name: two_layer_attention(actuals[name]) for name in ['correct', 'delayed_query_start']}
    assert not relay['correct']['note_reaches_P_act']
    assert relay['delayed_query_start']['note_reaches_P_act']
    assert all(r['masked_sum'] == 0 for r in relay.values())
    old = ast.parse((ROOT / 'source/workspace_w3b_before.py').read_text())
    old_body = outer_loop(old).body
    start = next(i for i, n in enumerate(old_body) if assignment(n, 'note_keys'))
    old_env = {'kinds': ['format'] * 2 + ['note'] * 6, 'S': 8, 'arms': {},
               'torch': SimpleNamespace(zeros=lambda *s, dtype: Mask(s), bool=bool)}
    exec(compiled(old_body[start:start + 2], '<old source note builder>'), old_env)
    assert old_env['arms']['current_note'].cells == set()
    assert not gate_assertions(tree, cells=set())['nonempty_P_act']
    reporters = run_checks(ROOT)
    by_name = {c['name']: c for c in reporters['fixtures']}
    assert len(by_name) == 15
    assert all(c['execution']['completed_without_exception'] for c in by_name.values())
    for c in by_name.values():
        assert c['execution']['report']['arms']['current_note']
        assert 'w3b_analyzed' in c['execution']['stdout']
    get_report = lambda name: by_name[name]['execution']['report']
    assert get_report('failed_masked_edge_gate')['gate']['attention_on_masked_edges_all_queries_max'] == .125
    assert get_report('failed_window_gate')['gate']['local_mass_beyond_window_max'] == .25
    assert get_report('failed_nonempty_cut_gate')['gate']['rows_with_zero_blocked_at_P_act'] == 1
    assert get_report('failed_p_note_identity_gate')['current_note_p_note_identical_to_unmasked']['share'] == 0
    assert get_report('missing_gate')['gate'] == {'n': 0}
    assert get_report('schema_1')['schema_version'] == 1
    assert get_report('missing_expected_id')['n'] == 1
    assert get_report('missing_expected_id')['rows_written'] == 2
    assert get_report('duplicate_id_replaces_expected_id')['n'] == 2
    assert get_report('extra_id')['n'] == 2
    assert get_report('unexpected_id_with_matching_count')['n'] == 1
    hidden_leak = get_report('first_legacy_gate_hides_later_leak')
    assert 'attention_on_masked_edges_all_queries_max' not in hidden_leak['gate']
    assert hidden_leak['gate']['attention_on_masked_keys_max'] == 0
    for name in ['gained_winner_reported_still_top', 'unresolved_baseline_reported_still_top']:
        assert get_report(name)['arms']['current_note']['taken_still_top_at_act'] == {'share': 1.0, 'n': 1}
    assert get_report('empty_denominator_reported_zero_share')['arms']['current_note']['taken_still_top_at_act'] == {'share': 0.0, 'n': 0}
    forbidden_modules = {'torch', 'mlx', 'mlx_lm', 'numpy', 'transformers'}
    assert not forbidden_modules.intersection(sys.modules), 'native/model import observed'
    result = {'basis': 'synthetic source-AST and analytic fixtures; no model findings',
              'source_hashes_verified': len(sources['files']), 'model_imports': 0,
              'forbidden_modules_checked_absent': sorted(forbidden_modules),
              'scope': 'builder AST, gate assertions with analytic attention sums, full reporter AST on virtual files; no torch tensor/pre-hook or real tokenizer execution',
              'builder_fixtures': builder_results, 'placement_mutations': mutations,
              'gate_negative_controls': controls, 'two_layer_relay': relay,
              'original_empty_cut': 'reproduced; new nonempty assertion rejects',
              'reporter': reporters}
    (ROOT / 'check.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'builder_fixtures': len(builder_results), 'placement_cases': len(mutations),
                      'negative_controls': len(controls), 'sources_verified': len(sources['files']),
                      'model_imports': 0, 'result': 'review counterexamples reproduced'}))


if __name__ == '__main__':
    main()
