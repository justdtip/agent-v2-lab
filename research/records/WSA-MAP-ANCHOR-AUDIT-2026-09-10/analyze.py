"""File-only audit. Standard library only; never imports the captured model code."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_json(root, name):
    return json.loads((root / name).read_text())


def rows(root, name):
    return [json.loads(line) for line in (root / name).read_text().splitlines() if line]


def verify_sources(root):
    manifest = load_json(root, 'sources.json')
    for item in manifest['sources']:
        data = (root / item['snapshot']).read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f"changed source: {item['snapshot']}")
        if len(data) != item['bytes']:
            raise ValueError(f"changed source length: {item['snapshot']}")
    return manifest


def close(a, b):
    if not math.isclose(a, b, rel_tol=2e-14, abs_tol=1e-12):
        raise ValueError(f'inconsistent recorded arithmetic: {a} != {b}')


def describe(values):
    valid = [abs(v) for v in values if v is not None]
    if not all(math.isfinite(v) for v in valid):
        raise ValueError('nonfinite value')
    return {'count': len(valid), 'undefined': len(values) - len(valid),
            'min_absolute': min(valid), 'median_absolute': statistics.median(valid),
            'max_absolute': max(valid)}


def reference_reused(source):
    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'golden_report':
                kw = {k.arg: k.value for k in node.keywords}
                equal = ast.dump(kw['reference']) == ast.dump(kw['reproduction'])
                matches.append({'call_line': node.lineno,
                                'reproduction_line': kw['reproduction'].lineno,
                                'same_reference_expression': equal})
    if len(matches) != 1:
        raise ValueError('expected one golden_report call')
    return matches[0]


def same_anchor_summary(data, with_raw):
    if len(data) != 54:
        raise ValueError('expected 54 scalar pairs')
    seen = set()
    for r in data:
        key = r['repo_layer'], r['direction'], r['cotangent']
        if key in seen:
            raise ValueError('duplicate scalar pair')
        seen.add(key)
        close(r['observed_shift'], r['anchor_term'] + r['arithmetic_term'])
        close(r['identity_residual'], abs(r['observed_shift'] - r['anchor_term'] - r['arithmetic_term']))
        if with_raw:
            a1, ax, aw = (r[k] for k in ['a_w1_x1', 'a_w64_x1', 'a_w64_x64'])
            close(r['observed_shift'], aw - a1)
            close(r['anchor_term'], aw - ax)
            close(r['arithmetic_term'], ax - a1)
            if a1:
                for kind in ['observed', 'anchor', 'arithmetic']:
                    term = r[kind + ('_shift' if kind == 'observed' else '_term')]
                    close(r[kind + '_relative'], term / abs(a1))
    result = {}
    for layer in sorted({r['repo_layer'] for r in data}):
        part = [r for r in data if r['repo_layer'] == layer]
        if len(part) != 18:
            raise ValueError('expected 18 checks at each layer')
        result[str(layer)] = {
            'basis': 'recomputed from committed scalar responses; shared anchors and cotangents',
            'opposing_anchor_and_arithmetic_terms': sum(r['anchor_term'] * r['arithmetic_term'] < 0 for r in part),
            **{kind: describe([r[kind + '_relative'] for r in part])
               for kind in ['observed', 'anchor', 'arithmetic']},
        }
    return result


def displacement_summary(data):
    if len(data) != 108:
        raise ValueError('expected 108 scalar displacement checks')
    keys = [(r['phase'], r['repo_layer'], r['direction'], r['cotangent']) for r in data]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate displacement check')
    for r in data:
        base = r['a_base']
        for name, column in [('delta', 'a_plus_delta'), ('random', 'a_plus_random')]:
            if base:
                close(r[name + '_relative'], abs(r[column] - base) / abs(base))
            elif r[name + '_relative'] is not None:
                raise ValueError('undefined relative response given a number')
    result = {}
    for phase in ['native', 'float32']:
        result[phase] = {}
        for layer in sorted({r['repo_layer'] for r in data}):
            part = [r for r in data if r['phase'] == phase and r['repo_layer'] == layer]
            if len(part) != 18:
                raise ValueError('expected 18 projections, not 18 independent displacement draws')
            row = {name: describe([r[name + '_relative'] for r in part])
                   for name in ['delta', 'random']}
            # Descriptive only: both derivatives below are computed through the float32 tail.
            # This does not compare a native-bf16 derivative with a float32 derivative.
            if phase == 'float32':
                row['native_anchor_inserted_in_float32_tail'] = describe([
                    abs(r['a_at_native_anchor'] - r['a_base']) / abs(r['a_base'])
                    if r['a_base'] else None for r in part])
            row['basis'] = 'recomputed scalar summaries; one displacement of each kind per layer'
            result[phase][str(layer)] = row
    return result


def counterexamples():
    # Equal to the producer's matrix-norm definition for these two diagonal matrices.
    reference = [1000.0, 1.0]
    candidate = [1000.0, 0.0]
    norm = math.sqrt(sum(x * x for x in reference))
    frobenius = math.sqrt(sum((a - b) ** 2 for a, b in zip(reference, candidate))) / norm
    cosine = sum(a * b for a, b in zip(reference, candidate)) / (norm * 1000)
    worst_column = max(abs(a - b) / abs(a) for a, b in zip(reference, candidate))
    assert frobenius < 0.001 and worst_column == 1.0
    # A second result differs. Reusing the first masks that difference completely.
    independent_relative = abs(2.0 - 1.0) / abs(1.0)
    reused_relative = abs(1.0 - 1.0) / abs(1.0)
    assert independent_relative == 1.0 and reused_relative == 0.0
    # F(p,z)=p*z, dF/dp=z. The selected position p does not move; the other does.
    old_p, old_z, new_p, new_z = 1.0, 1.0, 1.0, 1.1
    local_step = abs(new_p - old_p) / abs(old_p)
    full_step = math.hypot(new_p - old_p, new_z - old_z) / math.hypot(old_p, old_z)
    derivative_change = abs(new_z - old_z) / abs(old_z)
    assert local_step == 0 and full_step > 0 and derivative_change > 0
    # d(.5*x^2)/dx=x. Same-norm movements along x and y have different effects.
    equal_norm_changes = {'along_x': 0.1, 'along_y': 0.0}
    return {
        'basis': 'analytic known-answer examples, not model measurements',
        'weak_column': {'reference_diagonal': reference, 'candidate_diagonal': candidate,
                        'relative_frobenius': frobenius, 'cosine': cosine,
                        'worst_relative_column': worst_column},
        'independent_repeat': {'self_comparison': reused_relative,
                               'actual_second_result_comparison': independent_relative},
        'partial_input_norm': {'function': 'F(p,z)=p*z; selected derivative dF/dp=z',
                               'selected_position_relative_change': local_step,
                               'whole_input_relative_change': full_step,
                               'selected_derivative_relative_change': derivative_change},
        'one_random_direction_is_not_every_direction': equal_norm_changes,
    }


def analyze(root):
    manifest = verify_sources(root)
    source = root / 'source'
    maps = {}
    for layer in (1, 33):
        report = load_json(source, f'artefacts/golden-f32/golden-report-L{layer}.json')
        per_layer = report['finding']['per_layer']
        if len(per_layer) != 1 or per_layer[0]['layer'] != layer:
            raise ValueError('unexpected layer population')
        row = per_layer[0]
        close(report['finding']['worst_relative_difference'], row['relative_difference'])
        control = report['gates']['controls_separate']['transposed']
        close(control['margin_over_finding'], control['worst_relative_difference'] / row['relative_difference'])
        exact = load_json(source, f'artefacts/golden-f32/nu-exact-L{layer}.json')
        fd = load_json(source, f'artefacts/golden-f32/nu-finite-difference-L{layer}.json')
        for nu in (exact, fd):
            assert nu['precision']['forward_batch'] == nu['precision']['anchor_batch'] == 1
            assert nu['precision']['dtype'] == 'float32'
            assert nu['endpoint']['source_layers_repo'] == [layer]
            assert nu['position_weighting']['probe'][0]['first'] == 8
            assert nu['position_weighting']['probe'][0]['last'] == 127
            assert nu['position_weighting']['seq_len']['max'] == 128
        maps[str(layer)] = {
            **row,
            'metric': 'whole-layer relative Frobenius norm, NOT worst column',
            'basis': 'producer-reported matrix metric; underlying NPZ arrays absent from frozen Git record',
            'transposed_ratio_recomputed': control['worst_relative_difference'] / row['relative_difference'],
            'scientific_pass_threshold': report['finding']['threshold'],
            'stored_repeat_gate': report['gates']['exact_reproduces_itself'],
            'physical_step': report['declared']['finite_difference_epsilon'],
            'forward_batch': 1, 'anchor_batch': 1, 'seq_len': 128,
            'source_target_positions': [8, 127],
        }
    gate = reference_reused((source / 'golden_float32.py').read_text())
    assert gate['same_reference_expression']
    library = ast.parse((source / 'library/golden.py').read_text())
    metric_fn = next(n for n in library.body if isinstance(n, ast.FunctionDef) and n.name == 'layer_metrics')
    assert 'Frobenius norm' in ast.get_docstring(metric_fn)
    return {
        'schema_version': 1, 'producer_commit': manifest['producer_commit'],
        'order_commit': manifest['order_commit'], 'sources_verified': len(manifest['sources']),
        'full_maps': maps, 'repeat_gate_source_audit': gate,
        'same_anchor_4b_width_1_vs_64': same_anchor_summary(rows(source, 'artefacts/same-anchor.jsonl'), True),
        'same_anchor_12b_width_1_vs_8': {
            phase: same_anchor_summary(rows(source, f'artefacts/same-anchor-12b-{phase}.jsonl'), False)
            for phase in ['native', 'float32']},
        'native_displacement_transplant_4b': displacement_summary(rows(source, 'artefacts/displacement.jsonl')),
        'known_answer_examples': counterexamples(),
        'unexecuted_by_reviewer': ['model forward or derivative', 'device access',
                                  'recomputation of full maps from NPZ', 'full-anchor norm measurement',
                                  'independent exact-repeat run', 'preregistration seal review pending P1-P6'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
