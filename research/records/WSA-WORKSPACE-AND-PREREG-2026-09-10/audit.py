"""Reproduce this review using frozen text, metadata and analytic counterexamples only."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import runpy
import sys
import tempfile
from collections import Counter, defaultdict
from fractions import Fraction
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FORBIDDEN = {'torch', 'mlx', 'mlx_lm', 'transformers', 'numpy', 'local_llm_lab'}


class NoModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise RuntimeError('forbidden analysis import: ' + fullname)
        return None


sys.meta_path.insert(0, NoModels())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_sources(root=HERE):
    manifest = json.loads((root / 'sources.json').read_text())
    for row in manifest['inputs']:
        content = (root / row['snapshot']).read_bytes()
        if digest(content) != row['sha256'] or len(content) != row['bytes']:
            raise ValueError('changed source: ' + row['snapshot'])
    return len(manifest['inputs'])


def safe_module(path):
    allowed = {'__future__', 'argparse', 'hashlib', 'json', 're', 'sys', 'pathlib', 'collections'}
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split('.')[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or '').split('.')[0]]
        else:
            continue
        if not set(names) <= allowed:
            raise ValueError('unexpected frozen-script import: ' + str(names))
    return runpy.run_path(str(path), run_name='frozen_metadata_only')


def rank(matrix):
    a = [[Fraction(v) for v in row] for row in matrix]
    r = 0
    for c in range(len(a[0])):
        pivot = next((i for i in range(r, len(a)) if a[i][c]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        pivot_value = a[r][c]
        a[r] = [v / pivot_value for v in a[r]]
        for i in range(len(a)):
            if i != r:
                multiplier = a[i][c]
                a[i] = [v - multiplier * w for v, w in zip(a[i], a[r])]
        r += 1
        if r == len(a):
            break
    return r


def token_graph_reachable(positions, edges, source, target):
    reachable = {source}
    for layer_edges in edges:
        previous = reachable
        reachable = set(previous)  # residual connections
        reachable.update(q for q, k in layer_edges if k in previous)
    return target in reachable


def analytic_examples():
    # A straight logit line can look like abrupt 'ignition' after softmax.
    logits = [4 * (layer - 4) for layer in range(1, 9)]
    probabilities = [math.exp(z) / (math.exp(z) + 5) for z in logits]
    crossing = {str(t): next((i + 1 for i, p in enumerate(probabilities) if p > t), None)
                for t in (0.1, 0.5, 0.9)}
    width = crossing['0.9'] - crossing['0.1']
    assert width == 1
    assert all(logits[i+2] - 2 * logits[i+1] + logits[i] == 0 for i in range(6))

    # Six softmax logits have only five independent contrasts, regardless of hidden width.
    w = [[int(i == j) for j in range(6)] for i in range(6)]
    centered = [[Fraction(v) - Fraction(1, 6) for v in row] for row in w]
    assert rank(w) == 6 and rank(centered) == 5

    # Direct local attention respects the mask, yet two layers relay a distant signal.
    edges = [[(900, 0)], [(1800, 900)]]
    assert all(0 <= q-k < 1024 for layer in edges for q, k in layer)
    assert not any((1800, 0) in layer for layer in edges)
    assert token_graph_reachable(1801, edges, 0, 1800)
    assert not token_graph_reachable(1801, [edges[0], []], 0, 1800)

    # Attention mass alone does not bound projected value contribution.
    mass = [0.99, 0.01]
    projected_values = [0.0, 100.0]
    contributions = [p*v for p, v in zip(mass, projected_values)]
    assert contributions == [0.0, 1.0]

    # A family conditional label that is constant survives within-family label permutation.
    family_labels = {'A': ['read'] * 4, 'B': ['finish'] * 4}
    assert all(list(reversed(labels)) == labels for labels in family_labels.values())

    # Five-fold example: every prediction omits its own independent episode, yet all losses agree.
    # Five independent Bernoulli labels; training rule predicts the parity of its four labels.
    losses = []
    for labels in itertools.product((0, 1), repeat=5):
        errors = [int((sum(labels[:i] + labels[i+1:]) % 2) != labels[i]) for i in range(5)]
        assert len(set(errors)) == 1
        losses.append(sum(errors) / 5)
    mean = sum(losses) / len(losses)
    variance = sum((v-mean)**2 for v in losses) / len(losses)
    assert mean == 0.5 and variance == 0.25

    # Identical observational traces do not identify a causal role for the note.
    # Model R decides from X then writes N; model C decides from N; in clean data N=X.
    report_clean = [x for x in (0, 1)]
    compute_clean = [x for x in (0, 1)]
    report_intervene = [x for x in (0, 1)]
    compute_intervene = [1-x for x in (0, 1)]
    assert report_clean == compute_clean and report_intervene != compute_intervene

    # Matching a Gram spectrum does not match signed overlaps or a nonnegative sparse cone.
    a = 1 / math.sqrt(2)
    b = [(1.0, 0.0), (0.0, 1.0), (a, a)]
    c = [(1.0, 0.0), (0.0, 1.0), (-a, -a)]
    dot = lambda x, y: sum(u*v for u, v in zip(x, y))
    gb = [[dot(x, y) for y in b] for x in b]
    gc = [[dot(x, y) for y in c] for x in c]
    signs = [1, 1, -1]
    assert all(gc[i][j] == signs[i]*gb[i][j]*signs[j] for i in range(3) for j in range(3))
    target = (-a, -a)
    # Every B coefficient is nonnegative, so B's output is in the positive quadrant.
    # Its optimal reconstruction of target is zero, with squared error one. C has target as atom 3.
    assert all(dot(v, target) <= 0 for v in b)
    assert abs(dot(target, target) - 1) < 1e-15 and c[2] == target

    # Positive monotonicity is not enough for a derivative to serve as an activation map.
    # f(h)=h^2 at h=3: J*h=18 whereas f(h)=9. This is not a defect in Jacobian estimation.
    return {
        'basis': 'analytic counterexamples; not Gemma measurements',
        'softmax_crossing': {'linear_logits': logits, 'probabilities': probabilities,
                             'first_crossings': crossing, 'width_layers': width,
                             'logit_second_difference': 0},
        'six_class_rank': {'raw_rank': 6, 'contrast_rank': 5,
                           'maximum_softmax_linear_contrast_rank': 5},
        'local_relay': {'window': 1024, 'edges_by_layer': edges, 'total_distance': 1800,
                        'direct_far_edge': False, 'two_local_layers_reach': True,
                        'broken_relay_reaches': False},
        'attention_mass': {'mass': mass, 'projected_values': projected_values,
                           'contributions': contributions},
        'within_family_constant_null': {'permutation_changes_labels': False},
        'five_fold_dependence': {'equally_likely_datasets': len(losses),
                                'all_losses_identical_within_every_dataset': True,
                                'expected_error': mean, 'true_cv_mean_variance': variance,
                                'naive_independent_mean_variance': 0.25/5,
                                'fixed_score_bootstrap_variance_every_dataset': 0.0},
        'note_causal_nonidentifiability': {'clean_report_actions': report_clean,
                                         'clean_computation_actions': compute_clean,
                                         'note_flip_report_actions': report_intervene,
                                         'note_flip_computation_actions': compute_intervene},
        'gram_spectrum': {'gram_b': gb, 'gram_c': gc, 'similarity_diagonal': signs,
                          'same_spectrum_by_orthogonal_similarity': True,
                          'pairwise_b': [gb[0][1], gb[0][2], gb[1][2]],
                          'pairwise_c': [gc[0][1], gc[0][2], gc[1][2]],
                          'target': target, 'nonnegative_squared_error_b': 1.0,
                          'nonnegative_squared_error_c': 0.0, 'sparsity_budget_c': 1},
        'derivative_vs_activation': {'h': 3, 'f_h': 9, 'j_times_h': 18},
    }


def prereg_metadata(root):
    source = root / 'source'
    rows = [json.loads(s) for s in (source / 'capture-set.jsonl').read_text().splitlines()]
    triples = [[r['task_id'], r['step'], r['prompt_sha256']] for r in rows]
    inputs = json.loads((source / 'prereg-inputs.json').read_text())
    logical = digest(json.dumps(triples, sort_keys=True, ensure_ascii=False).encode())
    assert logical == inputs['decisions']['capture_set_sha256']
    assert len({(r['task_id'], r['step']) for r in rows}) == len(rows) == 7629
    file_digest = digest((source / 'capture-set.jsonl').read_bytes())
    assert file_digest == '98c78f1674902575ba87135edcfec409d3e82c6331e155a3ab7da6d22b731a4d'
    episodes = {r['task_id']: r for r in rows}
    folds = json.loads((source / 'folds.json').read_text())
    recomputed = {}
    strata = defaultdict(list)
    for task, r in episodes.items():
        strata[r['split'], r['family'], r['variant']].append(task)
    for stratum, tasks in sorted(strata.items()):
        offset = int.from_bytes(hashlib.sha256(f"{folds['seed']}:{':'.join(stratum)}".encode()).digest()[:4], 'big') % 5
        for index, task in enumerate(sorted(tasks)):
            recomputed[task] = (index + offset) % 5
    assert recomputed == folds['fold_of']
    assignment_digest = digest(json.dumps(sorted(recomputed.items()), sort_keys=True).encode())
    assert assignment_digest == folds['assignment_sha256']
    # Run the actual safe metadata scripts on their own frozen sources.
    checker = safe_module(source / 'check_prereg.py')
    document = (source / 'PREREGISTRATION.md').read_text()
    assert checker['check'](document, verbose=False) == []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert checker['self_test']() == 0
    claim_count = len(checker['CLAIMS'])
    # The producer's self-test is supplemented with an independently chosen precise corruption.
    bad = document.replace('| **distinct decisions** | **7,629** |', '| **distinct decisions** | **7,628** |')
    assert bad != document and 'distinct decisions' in checker['check'](bad, verbose=False)
    # Its scope is pinned figures: it does not validate an unpinned scientific assertion.
    wrong_science = document + '\nAll out-of-fold errors are independent by construction.\n'
    assert checker['check'](wrong_science, verbose=False) == []
    with tempfile.TemporaryDirectory(prefix='wsa-fold-replay-') as temp:
        target = Path(temp) / 'folds.json'
        fm = safe_module(source / 'folds.py')
        with contextlib.redirect_stdout(io.StringIO()):
            assert fm['main'](['--decisions', str(source/'capture-set.jsonl'), '--out', str(target)]) == 0
        assert json.loads(target.read_text()) == folds
    # Extract one pure function, without importing the production module (which imports numpy).
    tree = ast.parse((source / 'tolerances.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'required_n')
    scope = {'math': math, 'ALPHA': 0.05}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<required_n only>', 'exec'), scope)
    requirements = {str(eps): scope['required_n'](12, eps) for eps in (0.17, 0.09, 0.11, 0.16, 0.15)}
    assert requirements == {'0.17': 214, '0.09': 763, '0.11': 511, '0.16': 242, '0.15': 275}
    return {'basis': 'recomputed from frozen metadata; no captures read',
            'decisions': len(rows), 'episodes': len(episodes), 'logical_digest': logical,
            'byte_digest': file_digest, 'fold_digest': assignment_digest,
            'fold_counts': dict(sorted(Counter(recomputed.values()).items())),
            'producer_fold_script_reproduced': True, 'independent_fold_recomputation_agrees': True,
            'reporter_pinned_claims': claim_count, 'reporter_self_test': buf.getvalue().strip(),
            'independent_pinned_cell_mutation_refused': True,
            'unsupported_science_sentence_passes_reporter': True,
            'approved_formula_required_n': requirements,
            'arbitrary_paired_range_minus1_plus1_sufficient_n': {
                str(eps): math.ceil(2*math.log(480)/eps**2) for eps in (0.17, 0.09, 0.11)},
            'range_comparison_is_not_a_new_ruling': True,
            'source_prompt_sha_hashes': 'canonical JSON messages[:-1], not rendered prompt bytes',
            'workspace_residual_bytes_two_positions_excluding_embedding': {
                '4b': len(rows)*2*34*2560*4, '12b': len(rows)*2*48*3840*4}}


def carrier_fixture(root):
    module = safe_module(root / 'source/carrier_ablation.py')
    data = []
    for family, groups in [('two_groups', 2), ('ten_groups', 10)]:
        for step in range(20):
            note = f'Phase {step * groups // 20}. Next: same-item. '
            data.append({'metadata': {'family': family, 'task_id': family, 'step': step},
                         'completion': note + '```json\n{"name":"read_file","arguments":{"path":"same"}}\n```'})
    with tempfile.TemporaryDirectory(prefix='wsa-carrier-fixture-') as temp:
        directory = Path(temp)
        (directory/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
        for split in ('valid', 'test'):
            (directory/(split+'.jsonl')).write_text('')
        target = directory/'out.json'
        with contextlib.redirect_stdout(io.StringIO()):
            assert module['main'](['--data', str(directory), '--out', str(target)]) == 0
        result = json.loads(target.read_text())
    cells = {f: result['by_family'][f]['whole note'] for f in ('two_groups', 'ten_groups')}
    assert cells['two_groups']['distinguishable_share'] == 0.1
    assert cells['ten_groups']['distinguishable_share'] == 0.5
    assert all(c['fully_ablated_episodes'] == 1 for c in cells.values())
    return {'basis': 'synthetic notes passed through frozen producer script; not corpus finding',
            'steps': 20, 'floor': 0.05, 'declared_floor_plus_one_over_n': 0.1,
            'producer_cells': cells, 'two_group_early_late_bit_retained': True,
            'two_group_calls_and_next_action_identical': True,
            'producer_checks_call_or_next_action_identity': False}


def capture_metadata_fixture(root):
    """Execute unchanged metadata functions with a list-returning spy and an in-memory writer.

    No production module, torch, tensor, model or actual storage implementation is imported.
    """
    source = root / 'source/capture.py'
    tree = ast.parse(source.read_text())
    functions = {'prompt_digest', 'assert_contract', 'rows_by_decision',
                 'capture_decisions', '_already_captured'}
    constants = {'REQUIRED_CELL_FIELDS', 'FORWARD_BATCH', 'CAPTURE_DTYPE'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == 'ContractViolation':
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants
                                                for t in node.targets):
            nodes.append(node)
    written, calls = [], []

    def flush(target, shard, buffer, cells, manifest_path):
        written.extend(dict(cell) for cell in cells)
        return len(cells)

    scope = {'__name__': 'isolated_capture_metadata', 'Path': Path, 'json': json,
             'hashlib': hashlib, '_flush': flush}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    row = {'metadata': {'task_id': 'fixture', 'step': 0, 'family': 'read'},
           'messages': [{'role': 'user', 'content': 'task'},
                        {'role': 'assistant', 'content': 'expert call'}], 'ids': [11, 12, 13]}
    decision = {'task_id': 'fixture', 'step': 0, 'split': 'test', 'family': 'read',
                'variant': 'clean', 'difficulty': 2, 'recovery': False, 'rendered_rows': 1,
                'prompt_sha256': scope['prompt_digest'](row['messages']), 'row_ordinals': [0]}

    def invoke(directory, current_row, current_decision, overrides=None, checkpoint='checkpoint-A'):
        def spy(ids):
            calls.append(list(ids))
            return {'residuals': [[0.0, 0.0]], 'seq_len': len(ids), 'token_index': len(ids)-1,
                    'layers': 0, 'd_model': 2, 'device': 'cpu', 'dtype': 'torch.bfloat16',
                    **(overrides or {})}
        target = SimpleNamespace(directory=directory, entry='gemma3-4b-cuda-bf16',
                                 checkpoint_sha256=checkpoint, decoding='teacher-forced', shard_size=256)
        return scope['capture_decisions'](decisions=[current_decision],
                                         corpus={('fixture', 0): current_row},
                                         forward=spy, target=target)

    with tempfile.TemporaryDirectory(prefix='wsa-capture-metadata-') as temp:
        base = Path(temp)
        invoke(base/'baseline', row, decision)
        baseline = dict(written[-1])
        # The isolated cell validator can fail; the producer erases the opposing evidence first.
        try:
            scope['assert_contract']({**baseline, 'forward_batch': 64})
        except scope['ContractViolation']:
            pass
        else:
            raise AssertionError('direct width control should fail')
        invoke(base/'wrong_path', row, decision,
               {'forward_batch': 64, 'anchor_batch': 64, 'capture_dtype': 'promoted-float32',
                'dtype': 'torch.float32'})
        relabelled = dict(written[-1])
        assert relabelled['forward_batch'] == relabelled['anchor_batch'] == 1
        assert relabelled['capture_dtype'] == 'native' and relabelled['dtype'] == 'torch.float32'
        invoke(base/'fractional', row, decision, {'token_index': 1.9})
        fractional_record = written[-1]['token_index']
        assert fractional_record == 1
        changed_ids = {**row, 'ids': [21, 22, 23]}
        invoke(base/'changed_ids', changed_ids, decision)
        assert calls[-1] == changed_ids['ids'] and written[-1]['prompt_sha256'] == baseline['prompt_sha256']
        moved_messages = {**row, 'messages': [{'role': 'user', 'content': 'changed'},
                                             row['messages'][-1]]}
        before = len(calls)
        try:
            invoke(base/'changed_messages', moved_messages, decision)
        except scope['ContractViolation']:
            pass
        else:
            raise AssertionError('changed messages must fail')
        assert len(calls) == before
        resumes = {}
        for kind in ('baseline', 'changed_checkpoint', 'changed_prompt', 'missing_shard', 'changed_shard'):
            directory = base/('resume_'+kind)
            directory.mkdir()
            shard = directory/'residuals-00000.pt'
            shard_bytes = b'fixture receipt bytes; not a tensor archive'
            shard.write_bytes(shard_bytes)
            cell = {**baseline, 'shard_sha256': digest(shard_bytes)}
            (directory/'manifest.jsonl').write_text(json.dumps(cell)+'\n')
            current_row, current_decision = row, decision
            if kind == 'changed_prompt':
                current_row = moved_messages
                current_decision = {**decision, 'prompt_sha256': scope['prompt_digest'](moved_messages['messages'])}
            if kind == 'missing_shard':
                shard.unlink()
            if kind == 'changed_shard':
                shard.write_bytes(b'changed receipt')
            before = len(calls)
            resumes[kind] = invoke(directory, current_row, current_decision,
                                   checkpoint='checkpoint-B' if kind == 'changed_checkpoint' else 'checkpoint-A')
            assert resumes[kind]['already_present'] == 1 and resumes[kind]['captured'] == 0
            assert len(calls) == before
    return {'basis': 'unchanged producer metadata functions; spy callback and in-memory shard writer',
            'production_module_imported': False, 'torch_imported': False,
            'direct_width_cell_mutation_refused': True,
            'callback_path': {'forward_batch': 64, 'anchor_batch': 64, 'dtype': 'torch.float32',
                              'capture_dtype': 'promoted-float32'},
            'recorded_path': {k: relabelled[k] for k in ('forward_batch', 'anchor_batch', 'dtype', 'capture_dtype')},
            'fractional_token_index_1_9_recorded_as': fractional_record,
            'changed_token_ids_accepted_under_same_messages_digest': True,
            'changed_messages_refused_before_callback': True,
            'resume_results': resumes,
            'resume_identity_or_storage_mismatches_refused': 0}


def gate_ledger():
    return [
        {'id': 'G1', 'gate': 'check_prereg pinned-claim consistency',
         'status': 'demonstrated_can_disagree', 'evidence': 'source/check_prereg.py; audit.prereg_metadata',
         'negative_control': '7,629 to 7,628 in the distinct-decisions cell',
         'scope': 'pinned documentary claims only; not statistical validity'},
        {'id': 'G2', 'gate': 'required_n closed-form examples as a confidence guarantee',
         'status': 'cannot_certify_claimed_target', 'evidence': 'source/tolerances.py; analytic five-fold example',
         'negative_control': 'correlated cross-fit losses leave every formula check unchanged',
         'scope': 'can test evaluation of a formula; cannot establish its assumptions'},
        {'id': 'G3', 'gate': 'carrier removal acceptance including preserved action',
         'status': 'cannot_certify_claimed_target', 'evidence': 'source/carrier_ablation.py; audit.carrier_fixture',
         'negative_control': '20 notes in ten groups are reported fully ablated; no action identity check',
         'scope': 'producer script is a screening diagnostic, not the promised acceptance gate'},
        {'id': 'G9', 'gate': 'capture writer actual width and precision',
         'status': 'cannot_certify_claimed_target', 'evidence': 'source/capture.py; capture_metadata_fixture',
         'negative_control': 'spy reports width 64 and promoted float32; writer stamps width 1 and native',
         'scope': 'direct cell-label mutations fail, but actual callback path is not checked'},
        {'id': 'G10', 'gate': 'capture input identity as exact consumed tokens',
         'status': 'cannot_certify_claimed_target', 'evidence': 'source/capture.py; capture_metadata_fixture',
         'negative_control': 'different token IDs with identical messages are accepted',
         'scope': 'message changes do fail; token identity is not bound'},
        {'id': 'G11', 'gate': 'capture resume identity and shard integrity',
         'status': 'cannot_certify_claimed_target', 'evidence': 'source/capture.py; capture_metadata_fixture',
         'negative_control': 'changed checkpoint or prompt; missing or altered shard all skip as done',
         'scope': 'resume trusts task and step plus shard number'},
        {'id': 'G4', 'gate': 'workspace local-mask structural gate', 'status': 'unexecuted_or_unknown',
         'evidence': 'source/workspace-order.md W-3', 'negative_control': 'not delivered',
         'scope': 'direct attention support only, never all causal paths'},
        {'id': 'G5', 'gate': 'checkpoint layer_types versus is_sliding agreement', 'status': 'unexecuted_or_unknown',
         'evidence': 'source/workspace-order.md W-3', 'negative_control': 'not delivered',
         'scope': 'architecture identity; count is stated but not independently rechecked here'},
        {'id': 'G6', 'gate': 'workspace mass-floor classification', 'status': 'unexecuted_or_unknown',
         'evidence': 'source/workspace-order.md mass floor', 'negative_control': 'not delivered',
         'scope': 'declared 1e-3 retained; tool IDs and boundaries still need fixtures'},
        {'id': 'G7', 'gate': 'workspace lens/capture manifest binding and seal', 'status': 'unexecuted_or_unknown',
         'evidence': 'source/workspace-order.md discipline', 'negative_control': 'not delivered',
         'scope': 'no capture-reader scripts in the reviewed commits'},
        {'id': 'G8', 'gate': 'W-2 family permutation and fold exclusion', 'status': 'unexecuted_or_unknown',
         'evidence': 'source/workspace-order.md W-2', 'negative_control': 'not delivered; constant-family example supplied',
         'scope': 'family split, class support and fit dependencies must be frozen'},
    ]


def compute(root=HERE):
    source_count = verify_sources(root)
    gates = gate_ledger()
    return {'source_files_verified': source_count,
            'scope': 'standing requests 1 and 3; prospective reviewer design audit, no captures',
            'analytic': analytic_examples(), 'prereg': prereg_metadata(root),
            'carrier': carrier_fixture(root), 'capture': capture_metadata_fixture(root), 'gates': gates,
            'gate_counts': dict(sorted(Counter(g['status'] for g in gates).items())),
            'gate_count_scope': 'eleven named checks in this review only, not a programme-wide total'}


if __name__ == '__main__':
    print(json.dumps(compute(), indent=2, sort_keys=True, allow_nan=False))
