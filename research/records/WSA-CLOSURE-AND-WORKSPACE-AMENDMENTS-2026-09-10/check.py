"""Reproduce source-level closures and remaining counterexamples without native imports."""
from __future__ import annotations

import ast
from collections import Counter
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FORBIDDEN = {'torch', 'numpy', 'mlx', 'mlx_lm', 'transformers', 'local_llm_lab', 'jlens'}


class NoNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise RuntimeError('native/production import refused: ' + fullname)
        return None


sys.meta_path.insert(0, NoNative())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_sources(root):
    rows = json.loads((root/'sources.json').read_text())['inputs']
    for row in rows:
        data = (root/row['snapshot']).read_bytes()
        if sha(data) != row['sha256'] or len(data) != row['bytes']:
            raise ValueError('changed source: ' + row['snapshot'])
    return len(rows)


def capture_namespace(root):
    tree = ast.parse((root/'source/capture.py').read_text())
    functions = {'checkpoint_identity', 'prompt_digest', 'assert_contract',
                 'assert_seam_ran_the_declared_pass', 'capture_decisions', '_verified_captures'}
    constants = {'REQUIRED_CELL_FIELDS', 'REQUIRED_SEAM_FIELDS', 'REQUIRED_ROW_FIELDS',
                 'FORWARD_BATCH', 'CAPTURE_DTYPE'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in {'ContractViolation', 'ResumeUnverified'}:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants
                                                for t in node.targets):
            nodes.append(node)

    def writer(target, shard, buffer, cells, manifest_path):
        receipt = b'SYNTHETIC RECEIPT ONLY; NOT A TENSOR FILE'
        (target.directory/f'residuals-{shard:05d}.pt').write_bytes(receipt)
        with manifest_path.open('a') as stream:
            for cell in cells:
                stream.write(json.dumps({**cell, 'shard_sha256': sha(receipt)})+'\n')
        return len(cells)

    ns = {'__name__': 'isolated_metadata', 'hashlib': hashlib, 'json': json,
          'Path': Path, '_flush': writer}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<metadata only>', 'exec'), ns)
    return ns


def repeat_checks(root):
    tree = ast.parse((root/'source/repeat_gate.py').read_text())
    verdict_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'verdict' for t in n.targets))
    row_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                            and t.value.id == 'rows_out' for t in n.targets))
    assert isinstance(row_node.value, ast.Dict)
    row_pass = next(value for key, value in zip(row_node.value.keys, row_node.value.values)
                    if isinstance(key, ast.Constant) and key.value == 'passes')
    cases = [('baseline', None, None)] + [(f'{part}_L{layer}', part, layer)
             for part in ('within', 'against') for layer in (1, 33)]
    results = {}
    for name, part, changed_layer in cases:
        rows = {}
        for layer in (1, 33):
            metrics = {p: {'relative_frobenius_difference': float(p == part and layer == changed_layer)}
                       for p in ('within', 'against')}
            passes = eval(compile(ast.Expression(row_pass), '<row passes only>', 'eval'), metrics)
            rows[layer] = {'passes': passes}
        scope = {'rows_out': rows, 'boundary': {x: {'bitwise_identical': True} for x in rows},
                 'REPO_LAYERS': (1, 33), 'n_layers': 34}
        exec(compile(ast.Module(body=[verdict_node], type_ignores=[]), '<verdict only>', 'exec'), scope)
        results[name] = scope['verdict']['passes']
    assert results == {'baseline': True, 'within_L1': False, 'within_L33': False,
                       'against_L1': False, 'against_L33': False}
    binding_node = next(value for key, value in zip(row_node.value.keys, row_node.value.values)
                        if isinstance(key, ast.Constant) and key.value == 'binding')
    fields = [key.value for key in binding_node.keys if isinstance(key, ast.Constant)]
    return {'basis': 'actual row-pass expression and overall verdict AST; synthetic comparison metrics',
            'cases': results, 'failure_controls_refused': 4, 'binding_fields_in_source': fields,
            'real_repeat_boundary_and_binding_execution': 'unexecuted'}


def identity_checks(root, ns):
    base = {'config.json': sha(b'config'), 'model-00001.safetensors': sha(b'A'),
            'model-00002.safetensors': sha(b'B')}
    cases = {}
    for name in ('runner.py', 'repeat_gate.py'):
        tree = ast.parse((root/'source'/name).read_text())
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and ((isinstance(n.func, ast.Name) and n.func.id == 'checkpoint_identity')
                         or (isinstance(n.func, ast.Attribute) and n.func.attr == 'checkpoint_identity')))
        code = compile(ast.Expression(call), '<consumer identity call only>', 'eval')

        def consume(value):
            return eval(code, {'capture': SimpleNamespace(checkpoint_identity=ns['checkpoint_identity']),
                               'checkpoint_identity': ns['checkpoint_identity'], 'report': {'sha256': value}})

        original = consume(base)
        changed = []
        for field in base:
            item = dict(base)
            item[field] = sha(b'changed '+field.encode())
            other = consume(item)
            assert other['checkpoint_sha256'] != original['checkpoint_sha256']
            changed.append(field)
        assert consume(dict(reversed(list(base.items())))) == original
        refused = []
        for label, value in [('none', None), ('empty', {}), ('string', 'invalid'),
                             ('config_only', {'config.json': base['config.json']}),
                             ('weights_only', {'model.safetensors': base['model-00001.safetensors']})]:
            try:
                consume(value)
            except ns['ContractViolation']:
                refused.append(label)
            else:
                raise AssertionError(label)
        cases[name] = {'individual_file_changes_detected': changed,
                       'missing_identity_cases_refused': refused, 'order_invariant': True,
                       'file_manifest_retained_by_helper': original['checkpoint_files_sha256'] == base}
    return {'basis': 'actual consumer calls plus unchanged identity helper, synthetic loader manifests',
            'consumers': cases, 'loader_completeness_is_upstream_contract': True,
            'actual_checkpoint_files_read': False}


def capture_checks(ns):
    row = {'prompt': 'rendered task prompt',
           'messages': [{'role': 'user', 'content': 'task'}, {'role': 'assistant', 'content': 'expert'}],
           'metadata': {'task_id': 'fixture', 'step': 0, 'family': 'read'}}
    decision = {'task_id': 'fixture', 'step': 0, 'split': 'test', 'family': 'read',
                'variant': 'clean', 'difficulty': 2, 'recovery': False, 'rendered_rows': 1,
                'row_ordinals': [0], 'messages_sha256': ns['prompt_digest'](row['messages'])}
    weights = {'config.json': sha(b'cfg'), 'model.safetensors': sha(b'weights')}
    identity = ns['checkpoint_identity'](weights)
    calls = []

    def invoke(directory, current_row=None, current_decision=None, current_identity=None,
               override=None, ids=None):
        current_row = row if current_row is None else current_row
        current_decision = decision if current_decision is None else current_decision
        current_identity = identity if current_identity is None else current_identity
        ids = [10, 20, 30] if ids is None else ids

        def forward(received):
            calls.append(list(ids))
            observed = {'residuals': SimpleNamespace(shape=(2, 4)), 'seq_len': len(ids),
                        'token_index': len(ids)-1, 'layers': 2, 'd_model': 4, 'device': 'cpu',
                        'dtype': 'torch.bfloat16', 'forward_batch': 1, 'anchor_batch': 1,
                        'capture_dtype': 'native',
                        'rendered_prompt_sha256': sha(received['prompt'].encode()),
                        'token_ids_sha256': sha(json.dumps(ids).encode()), 'token_ids_length': len(ids)}
            return {**observed, **(override or {})}

        target = SimpleNamespace(directory=directory, entry='synthetic', identity=current_identity,
                                 decoding='teacher-forced', shard_size=256)
        return ns['capture_decisions'](decisions=[current_decision],
               corpus={('fixture', 0): current_row}, forward=forward, target=target)

    refusal_cases = {'forward_width': {'forward_batch': 64}, 'anchor_width': {'anchor_batch': 64},
                     'promoted': {'capture_dtype': 'promoted-float32'},
                     'fractional_position': {'token_index': 1.9}, 'bool_position': {'token_index': True},
                     'negative_position': {'token_index': -1}, 'past_end': {'token_index': 3},
                     'different_bytes_at_seam': {'rendered_prompt_sha256': '0'*64},
                     'token_length': {'token_ids_length': 2},
                     'residual_shape': {'residuals': SimpleNamespace(shape=(7, 3))}}
    refusals, resumes = {}, {}
    with tempfile.TemporaryDirectory(prefix='wsa-closure-') as temp:
        base = Path(temp)
        fresh = invoke(base/'positive')
        assert fresh['complete'] and fresh['verified'] == 1
        cell = json.loads((base/'positive/manifest.jsonl').read_text())
        assert cell['row_ordinals'] == [0]
        assert cell['messages_sha256'] == decision['messages_sha256']
        assert cell['token_ids_sha256'] == sha(json.dumps([10, 20, 30]).encode())
        for name, override in refusal_cases.items():
            try:
                invoke(base/name, override=override)
            except ns['ContractViolation'] as error:
                refusals[name] = str(error)
            else:
                raise AssertionError(name)
            assert not (base/name/'manifest.jsonl').exists()

        for name in ('clean', 'checkpoint', 'messages', 'rendered_bytes', 'missing_shard', 'altered_shard'):
            directory = base/('resume_'+name)
            invoke(directory)
            current_row, current_identity = copy.deepcopy(row), identity
            if name == 'checkpoint':
                current_identity = ns['checkpoint_identity']({**weights, 'model.safetensors': sha(b'changed')})
            elif name == 'messages':
                current_row['messages'][0]['content'] = 'new task'
            elif name == 'rendered_bytes':
                current_row['prompt'] += ' changed rendering'
            elif name == 'missing_shard':
                (directory/'residuals-00000.pt').unlink()
            elif name == 'altered_shard':
                (directory/'residuals-00000.pt').write_bytes(b'changed receipt')
            before = len(calls)
            try:
                summary = invoke(directory, current_row=current_row, current_identity=current_identity)
            except ns['ResumeUnverified'] as error:
                assert name != 'clean'
                resumes[name] = {'refused': True, 'reason': str(error)}
            else:
                assert name == 'clean'
                resumes[name] = {'refused': False, 'complete': summary['complete'],
                                 'verified': summary['verified'], 'captured': summary['captured']}
            assert len(calls) == before

        # The corpus and manifest agree, but the *new request* names a different semantic input.
        altered_request = {**decision, 'messages_sha256': sha(b'not the corpus messages')}
        try:
            invoke(base/'wrong_request_fresh', current_decision=altered_request)
        except ns['ContractViolation']:
            wrong_request_fresh_refused = True
        else:
            raise AssertionError('fresh request should refuse')
        invoke(base/'wrong_request_resume')
        before = len(calls)
        wrong_request_resume = invoke(base/'wrong_request_resume', current_decision=altered_request)
        assert wrong_request_resume['complete'] and wrong_request_resume['verified'] == 1
        assert len(calls) == before

        # Same text, same config/weights, a different tokenizer result. Fresh passes bind different
        # token identities; resume cannot consult the new result because it never prepares the input.
        invoke(base/'token_a', ids=[10, 20, 30])
        invoke(base/'token_b', ids=[10, 21, 30])
        cell_a = json.loads((base/'token_a/manifest.jsonl').read_text())
        cell_b = json.loads((base/'token_b/manifest.jsonl').read_text())
        assert cell_a['token_ids_sha256'] != cell_b['token_ids_sha256']
        before = len(calls)
        changed_token_resume = invoke(base/'token_a', ids=[10, 21, 30])
        assert changed_token_resume['complete'] and changed_token_resume['verified'] == 1
        assert len(calls) == before

    return {'basis': 'unchanged metadata functions, inert seam and tiny synthetic receipt writer',
            'seam_controls_refused': refusals, 'resume_controls': resumes,
            'positive_records_distinct_message_render_and_token_identities': True,
            'new_request_gap': {'fresh_refused': wrong_request_fresh_refused,
                                'resume_complete': wrong_request_resume['complete'],
                                'resume_verified': wrong_request_resume['verified']},
            'tokenizer_gap': {'fresh_token_hashes_differ': True,
                              'resume_complete': changed_token_resume['complete'],
                              'resume_verified': changed_token_resume['verified'],
                              'new_tokenization_not_consulted': True},
            'real_tokenizer_forward_or_tensor_storage': 'unexecuted'}


def mathematical_examples():
    # All four sign combinations: label is the sign of y, the smaller-variance component.
    points = [(2*a, b) for a, b in itertools.product((-1, 1), repeat=2)]
    labels = [int(y > 0) for x, y in points]
    scaled = [(x, 3*y) for x, y in points]  # invertible J=diag(1,3)

    def covariance(vectors):
        return [[sum(v[i]*v[j] for v in vectors)/len(vectors) for j in range(2)] for i in range(2)]

    cov_h, cov_jh = covariance(points), covariance(scaled)
    assert cov_h == [[4.0, 0.0], [0.0, 1.0]]
    assert cov_jh == [[4.0, 0.0], [0.0, 9.0]]
    # The top PCA component is x before J and y after it, with no eigenvalue ties.
    x_groups = {x: [label for (xx, _), label in zip(points, labels) if xx == x] for x, _ in points}
    assert all(sorted(group) == [0, 1] for group in x_groups.values())
    original_best = sum(max(group.count(0), group.count(1)) for group in x_groups.values())/len(points)
    transformed_accuracy = sum(int(y > 0) == label for (_, y), label in zip(scaled, labels))/len(points)
    assert original_best == 0.5 and transformed_accuracy == 1.0

    # A family-constant label survives every within-family shuffle, before *any* readout is fit.
    train = {'A': [0, 0, 0], 'B': [1, 1, 1]}
    permutations = [list(itertools.permutations(values)) for values in train.values()]
    unchanged = sum(list(a) == train['A'] and list(b) == train['B']
                    for a, b in itertools.product(*permutations))
    total = len(permutations[0])*len(permutations[1])
    assert unchanged == total == 36

    # Agreement at a state does not identify the local derivative; failure need not mean bad J.
    # f(x)=x^2+3 at x=2: f=7, f'=4. Exact J*x=8, while wrong J=3.5 gives exactly 7.
    state, true_readout, derivative = 2.0, 7.0, 4.0
    wrong_map = true_readout/state
    assert derivative*state != true_readout and wrong_map*state == true_readout
    return {'basis': 'exact finite toy examples; no model data or fitted real probes',
            'pca': {'points': points, 'labels': labels, 'invertible_map': [[1, 0], [0, 3]],
                    'covariance_h': cov_h, 'covariance_Jh': cov_jh, 'rank': 1,
                    'best_accuracy_from_PCA_h': original_best,
                    'accuracy_from_PCA_Jh': transformed_accuracy,
                    'new_information_in_full_vector': False,
                    'changed_information_retained_after_PCA': True},
            'transfer_null': {'within_family_permutations_enumerated': total,
                              'unchanged_training_label_assignments': unchanged,
                              'heldout_inputs_need_not_change_for_this_counterexample': True},
            'readout_not_derivative_gate': {'state': state, 'true_readout': true_readout,
                                           'true_derivative': derivative,
                                           'exact_map_readout': derivative*state,
                                           'wrong_map': wrong_map,
                                           'wrong_map_readout': wrong_map*state}}


def capture_enumeration(root):
    data = (root/'source/capture-set.jsonl').read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    logical = [[r['task_id'], r['step'], r['messages_sha256']] for r in rows]
    digest = sha(json.dumps(logical, sort_keys=True, ensure_ascii=False).encode())
    assert len(rows) == 7629
    assert digest == '1a7cfbdd4e21fc203eafbcc3ec50b96afcb214c169f21c7ba968ca83dfc9709a'
    assert sha(data) == '8bbc8062249ce1cc0e15050a866cd726a2c205a969f3eeab3840fcb5520cee3e'
    return {'basis': 'frozen decision metadata; no residual captures', 'decisions': len(rows),
            'logical_sha256_unchanged': digest, 'file_sha256': sha(data),
            'legacy_prompt_sha256_fields': sum('prompt_sha256' in r for r in rows)}


def compute(root=HERE):
    source_count = verify_sources(root)
    ns = capture_namespace(root)
    previous = json.loads((root/'source/previous-check.json').read_text())
    ledger = copy.deepcopy(previous['gate_ledger'])
    revisions = {
        'G9': ('demonstrated_can_disagree', 'check.json:capture.seam_controls_refused',
               'widths, promoted tag, position, shape and prompt mismatch rejected before writer',
               'honest seam metadata; actual tensor/model arithmetic remains unexecuted'),
        'G10': ('cannot_certify_claimed_target', 'check.json:capture.tokenizer_gap',
                'same prompt and weights, changed tokenizer ids: resume skips before preparation',
                'fresh token hashes are now recorded; current tokens are not verified on resume'),
        'G11': ('cannot_certify_claimed_target', 'check.json:capture',
                'new decision digest differs: fresh refuses, resume says complete and verified',
                'original checkpoint/prompt/shard failures now all refuse; current request still unchecked on skip'),
        'G12': ('demonstrated_can_disagree', 'check.json:repeat',
                'either comparison independently nonzero at either declared layer rejects',
                'actual verdict expressions with synthetic metrics; no device repeat executed'),
        'G13': ('demonstrated_can_disagree', 'check.json:identity',
                'change each config/weight digest separately; missing identity refused by both consumers',
                'complete loader manifest is upstream contract; no checkpoint files read'),
        'G8': ('cannot_certify_claimed_target', 'check.json:mathematics.transfer_null',
               'family-constant labels unchanged by every training-family permutation',
               'new transfer null remains degenerate on this structure; no programme data scored'),
    }
    for gate in ledger:
        if gate['id'] not in revisions:
            continue
        status, evidence, negative, scope = revisions[gate['id']]
        gate['previous_status'] = gate['status']
        gate['previous_review_commit'] = '332bdc96509e15375ad6eb3857974d61879ef1ab'
        gate.update(status=status, evidence=evidence, negative_control=negative, scope=scope,
                    latest_receipt='WSA-CLOSURE-AND-WORKSPACE-AMENDMENTS-2026-09-10/check.json')
    return {'source_count': source_count, 'repeat': repeat_checks(root),
            'identity': identity_checks(root, ns), 'capture': capture_checks(ns),
            'mathematics': mathematical_examples(), 'enumeration': capture_enumeration(root),
            'gate_ledger': ledger,
            'gate_counts': dict(sorted(Counter(g['status'] for g in ledger).items())),
            'gate_inventory_scope': 'same 13 IDs traced from 3494a2c and 332bdc9; not programme total',
            'unexecuted': ['real model forwards, exact fits and boundary checks', 'native suites',
                           'real tokenizer and tensor persistence', 'workspace analyses and captures',
                           'new ladder raw archives', 'all device results']}


if __name__ == '__main__':
    print(json.dumps(compute(), indent=2, sort_keys=True, allow_nan=False))
