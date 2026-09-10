"""Model-free checks of the new coverage/repeat verdicts; never import producer modules."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from collections import Counter

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
    sources = json.loads((root/'sources.json').read_text())['inputs']
    for row in sources:
        content = (root/row['snapshot']).read_bytes()
        if sha(content) != row['sha256'] or len(content) != row['bytes']:
            raise ValueError('changed source: ' + row['snapshot'])
    return len(sources)


def repeat_verdict(root):
    tree = ast.parse((root/'source/repeat_gate.py').read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'verdict' for t in node.targets))
    baseline = {layer: {'within_process': {'relative_frobenius_difference': 0.0,
                                          'bitwise_identical': True},
                       'against_saved_map': {'relative_frobenius_difference': 0.0,
                                             'bitwise_identical': True}}
                for layer in (1, 33)}
    results = {}
    cases = [('baseline', None, None)] + [(f'{part}_L{layer}', part, layer)
              for part in ('within_process', 'against_saved_map') for layer in (1, 33)]
    for name, part, layer in cases:
        rows = copy.deepcopy(baseline)
        if part:
            rows[layer][part] = {'relative_frobenius_difference': 1.0, 'bitwise_identical': False}
        scope = {'rows_out': rows, 'boundary': {v: {'bitwise_identical': True} for v in (1, 33)},
                 'REPO_LAYERS': (1, 33), 'n_layers': 34}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<verdict only>', 'exec'), scope)
        verdict = scope['verdict']
        results[name] = {'passes': verdict['passes'], 'coverage': verdict['coverage']}
    assert results['baseline']['passes']
    assert results['within_process_L1']['passes'] and results['within_process_L33']['passes']
    assert not results['against_saved_map_L1']['passes'] and not results['against_saved_map_L33']['passes']
    return {'basis': 'unchanged verdict AST, synthetic nonzero/zero comparison metrics',
            'cases': results, 'ignored_within_process_failures': 2,
            'against_saved_failures_refused': 2,
            'device_boundary_or_repeat_executed': False}


def checkpoint_selector(root):
    expressions = {}
    for name in ('runner.py', 'repeat_gate.py'):
        tree = ast.parse((root/'source'/name).read_text())
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'get'
                   and node.args and isinstance(node.args[0], ast.Constant)
                   and node.args[0].value == 'config.json']
        assert matches
        expressions[name] = matches
    reports = [{'sha256': {'config.json': sha(b'one config'),
                           'model-00001-of-00001.safetensors': sha(weights)}}
               for weights in (b'synthetic weights A', b'synthetic weights B')]
    selected = {}
    for name, nodes in expressions.items():
        selected[name] = [[eval(compile(ast.Expression(node), '<digest selector only>', 'eval'),
                                {'report': report}) for node in nodes] for report in reports]
        assert selected[name][0] == selected[name][1]
    assert reports[0]['sha256'] != reports[1]['sha256']
    return {'basis': 'actual config-hash selector expressions with synthetic loader reports',
            'loader_reports_have_different_weight_hashes': True, 'selected_hashes': selected,
            'selected_hash_identifies': 'config.json only',
            'loader_already_reports_weight_shard_hashes': True}


def capture_scope(root):
    tree = ast.parse((root/'source/capture.py').read_text())
    names = {'prompt_digest', 'assert_contract', 'capture_decisions', '_already_captured'}
    constants = {'REQUIRED_CELL_FIELDS', 'REQUIRED_ROW_FIELDS', 'FORWARD_BATCH', 'CAPTURE_DTYPE'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == 'ContractViolation':
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants
                                                for t in node.targets):
            nodes.append(node)
    observed_prompts, written_cells = [], []

    def writer(target, shard, buffer, cells, manifest_path):
        written_cells.extend(copy.deepcopy(cells))
        data = b'synthetic receipt, not a tensor file'
        (target.directory/f'residuals-{shard:05d}.pt').write_bytes(data)
        with manifest_path.open('a') as stream:
            for cell in cells:
                stream.write(json.dumps({**cell, 'shard_sha256': sha(data)})+'\n')
        return len(cells)

    scope = {'__name__': 'isolated_capture_metadata', 'hashlib': hashlib, 'json': json,
             'Path': Path, '_flush': writer}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<capture metadata only>', 'exec'), scope)
    row = {'messages': [{'role': 'user', 'content': 'task'},
                        {'role': 'assistant', 'content': 'expert completion'}],
           'prompt': 'rendered prompt A', 'metadata': {'task_id': 'fixture', 'step': 0, 'family': 'read'}}
    decision = {'task_id': 'fixture', 'step': 0, 'split': 'test', 'family': 'read',
                'variant': 'clean', 'difficulty': 2, 'recovery': False, 'rendered_rows': 1,
                'prompt_sha256': scope['prompt_digest'](row['messages']), 'row_ordinals': [0]}

    def invoke(directory, current_row=row, current_decision=decision, checkpoint='A'):
        def spy(received_row):
            observed_prompts.append(received_row['prompt'])
            return {'residuals': [[0, 0]], 'seq_len': 3, 'token_index': 2, 'layers': 0,
                    'd_model': 2, 'device': 'cpu', 'dtype': 'torch.bfloat16'}
        target = SimpleNamespace(directory=directory, entry='synthetic', checkpoint_sha256=checkpoint,
                                 decoding='teacher-forced', shard_size=256)
        return scope['capture_decisions'](decisions=[current_decision],
                                         corpus={('fixture', 0): current_row}, forward=spy, target=target)

    with tempfile.TemporaryDirectory(prefix='wsa-coverage-fixtures-') as temp:
        base = Path(temp)
        initial = invoke(base/'baseline')
        assert initial['complete'] and initial['requested'] == 1 and initial['outstanding'] == 0
        original = written_cells[-1]
        changed_render = {**row, 'prompt': 'different rendered prompt B'}
        render_result = invoke(base/'changed_render', current_row=changed_render)
        assert observed_prompts[-1] == changed_render['prompt'] and render_result['complete']
        assert written_cells[-1]['prompt_sha256'] == original['prompt_sha256']
        changed_messages = {**row, 'messages': [{'role': 'user', 'content': 'different task'},
                                               row['messages'][-1]]}
        try:
            invoke(base/'changed_messages', current_row=changed_messages)
        except scope['ContractViolation']:
            message_control = True
        else:
            raise AssertionError('changed messages should fail')
        resumes = {}
        for mode in ('same', 'checkpoint_changed', 'prompt_changed', 'shard_missing', 'shard_changed'):
            directory = base/mode
            invoke(directory)
            now_row, now_decision, checkpoint = row, decision, 'A'
            if mode == 'checkpoint_changed':
                checkpoint = 'B'
            elif mode == 'prompt_changed':
                now_row = changed_messages
                now_decision = {**decision, 'prompt_sha256': scope['prompt_digest'](now_row['messages'])}
            elif mode == 'shard_missing':
                (directory/'residuals-00000.pt').unlink()
            elif mode == 'shard_changed':
                (directory/'residuals-00000.pt').write_bytes(b'corrupted fixture receipt')
            before = len(observed_prompts)
            resumes[mode] = invoke(directory, now_row, now_decision, checkpoint)
            assert resumes[mode]['complete'] and resumes[mode]['captured'] == 0
            assert resumes[mode]['outstanding'] == 0 and resumes[mode]['already_present'] == 1
            assert len(observed_prompts) == before

        # Prove the new coverage field can expose a missing manifest line, in its stated scope.
        scope['_flush'] = lambda target, shard, buffer, cells, manifest_path: len(cells)
        absent_manifest = invoke(base/'no_manifest_written')
        assert not absent_manifest['complete'] and absent_manifest['outstanding'] == 1

    return {'basis': 'unchanged metadata functions, inert callback, synthetic receipt writer',
            'real_corpus_or_model_used': False,
            'row_interface_without_ids_works_in_fixture': True,
            'changed_rendered_prompt_accepted_under_unchanged_messages_digest': True,
            'changed_messages_refused': message_control,
            'resume_cases': resumes,
            'missing_manifest_negative_control': absent_manifest,
            'coverage_scope': 'manifest-line membership, not verified capture identity or storage'}


def compute(root=HERE):
    count = verify_sources(root)
    previous = json.loads((root/'source/previous-audit.json').read_text())
    ledger = copy.deepcopy(previous['gates'])
    for gate in ledger:
        gate['origin_review_commit'] = '3494a2cd597bcbc8183121bb4fc8f63c2f6004b6'
        gate['origin_record'] = 'research/records/WSA-WORKSPACE-AND-PREREG-2026-09-10'
        if gate['id'] in {'G10', 'G11'}:
            gate['latest_receipt'] = 'WSA-COVERAGE-REVIEW-2026-09-10/check.json:capture'
    ledger.extend([
        {'id': 'G12', 'gate': 'fresh and saved-map repeat verdict',
         'status': 'cannot_certify_claimed_target',
         'evidence': 'source/repeat_gate.py; check.json:repeat',
         'negative_control': 'only the second fresh fit disagrees; overall verdict still passes',
         'scope': 'against-saved comparison can reject; within-process failure does not gate'},
        {'id': 'G13', 'gate': 'runner and repeat checkpoint identity',
         'status': 'cannot_certify_claimed_target',
         'evidence': 'source/runner.py; source/repeat_gate.py; source/hf_text.py; check.json:checkpoint',
         'negative_control': 'same config, different weight-shard digests give identical recorded identity',
         'scope': 'config identity, not full checkpoint identity'},
    ])
    return {'source_count': count, 'repeat': repeat_verdict(root), 'checkpoint': checkpoint_selector(root),
            'capture': capture_scope(root), 'gate_ledger': ledger,
            'gate_counts': dict(sorted(Counter(g['status'] for g in ledger).items())),
            'gate_inventory_scope': '13 named checks accumulated from review 3494a2c; not programme total',
            'unexecuted': ['all model forwards and real tensor storage', 'real repeat and boundary checks',
                           'ladder archive array validation', 'native suites', 'workspace or plan-progress capture analysis']}


if __name__ == '__main__':
    print(json.dumps(compute(), indent=2, sort_keys=True, allow_nan=False))
