"""Read embedded score arrays; never load a model, tokenizer, lens matrix or tensor archive."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
BLOCKED = {'torch', 'mlx', 'mlx_lm', 'transformers', 'local_llm_lab', 'jlens'}


class NoModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise RuntimeError('model import refused: ' + fullname)
        return None


sys.meta_path.insert(0, NoModels())


def source_check(root):
    inputs = json.loads((root/'sources.json').read_text())['inputs']
    for item in inputs:
        data = (root/item['snapshot']).read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('changed source: ' + item['snapshot'])
    return len(inputs)


def extract(root):
    source_check(root)
    text = (root/'source/jspace-heatmap.html').read_text()
    matches = re.findall(r'^const MODELS = (.*);$', text, re.M)
    if len(matches) != 1:
        raise ValueError('expected one literal MODELS array; JavaScript is never executed')
    return json.loads(matches[0])


def validate(d):
    n = d['n_generated']
    layers = d['repo_layers']
    assert isinstance(n, int) and n > 0
    assert len(set(layers)) == len(layers) and layers == sorted(layers)
    assert len(d['tokens']) == n and ''.join(d['tokens']) == d['generated_text']
    assert all(isinstance(t, str) for t in d['tokens'])
    cells = 0
    for method in ('lens', 'logit_lens', 'model'):
        rows = [d[method]] if method == 'model' else list(d[method].values())
        if method != 'model':
            assert sorted(map(int, d[method])) == layers
        for row in rows:
            for field in ('p_actual', 'rank_actual', 'top1', 'entropy'):
                assert len(row[field]) == n
            for p, rank, top, entropy in zip(row['p_actual'], row['rank_actual'], row['top1'], row['entropy']):
                assert isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1
                assert type(rank) is int and rank >= 1
                assert isinstance(top, str)
                assert isinstance(entropy, (int, float)) and math.isfinite(entropy) and entropy >= -1e-5
                # If rank r counts strictly better candidates plus one, total probability >= r*p.
                # Small allowance here checks serialized numerical consistency, not lens validity.
                assert rank*p <= 1 + 1e-5
            cells += n
    return cells


def summary(d):
    n, layers = d['n_generated'], d['repo_layers']
    per_layer = []
    paths = {}
    for method in ('lens', 'logit_lens'):
        first, stable, lost = [], [], []
        for i in range(n):
            successes = [l for l in layers if d[method][str(l)]['rank_actual'][i] == 1]
            hit = successes[0] if successes else None
            first.append(hit)
            stable.append(next((l for l in layers if all(d[method][str(k)]['rank_actual'][i] == 1
                                                        for k in layers if k >= l)), None))
            if hit is not None and any(d[method][str(k)]['rank_actual'][i] != 1 for k in layers if k > hit):
                lost.append(i+1)
        found = [x for x in first if x is not None]
        paths[method] = {
            'ever_rank1': len(found), 'never_rank1': n-len(found),
            'first_rank1_median_among_found': statistics.median(found) if found else None,
            'first_rank1_counts': dict(sorted(Counter(found).items())),
            'first_rank1_by_token': first, 'stable_rank1_from_layer': stable,
            'lost_rank1_later_count': len(lost), 'lost_rank1_later_token_numbers': lost,
            'never_rank1_token_numbers': [i+1 for i, x in enumerate(first) if x is None],
            'previous_token_string_matches_L1': sum(d[method]['1']['top1'][i] == d['tokens'][i-1]
                                                   for i in range(1, n)),
            'previous_token_comparison_denominator': n-1,
        }
    for layer in layers:
        row = {'layer': layer, 'tokens': n}
        for method in ('lens', 'logit_lens'):
            data = d[method][str(layer)]
            row[method] = {
                'rank1': sum(rank == 1 for rank in data['rank_actual']),
                'mean_p_actual': statistics.mean(data['p_actual']),
                'median_p_actual': statistics.median(data['p_actual']),
                'zero_p_actual': sum(p == 0 for p in data['p_actual']),
                'top1_decoded_matches_final_model': sum(x == y for x, y in zip(data['top1'], d['model']['top1'])),
            }
        a, b = d['lens'][str(layer)], d['logit_lens'][str(layer)]
        row['lens_gives_higher_p_count'] = sum(x > y for x, y in zip(a['p_actual'], b['p_actual']))
        per_layer.append(row)
    return {
        'basis': 'recomputed from embedded producer-reported arrays; retrospective descriptive quantities',
        'label': d['label'], 'forced': d['forced'], 'n_prompt': d['n_prompt'], 'n_tokens': n,
        'layer_count': len(layers), 'layer_min': min(layers), 'layer_max': max(layers),
        'generated_text': d['generated_text'], 'whitespace_word_count': len(d['generated_text'].split()),
        'model_rank1_of_scored_text': sum(r == 1 for r in d['model']['rank_actual']),
        'model_p_actual_mean': statistics.mean(d['model']['p_actual']),
        'model_p_actual_median': statistics.median(d['model']['p_actual']),
        'paths': paths, 'by_layer': per_layer,
        'tokens': d['tokens'],
        'final_model_disagreements': [{'token_number': i+1, 'scored_token': d['tokens'][i],
                                       'final_top1': d['model']['top1'][i], 'rank': r,
                                       'p_actual': d['model']['p_actual'][i]}
                                     for i, r in enumerate(d['model']['rank_actual']) if r != 1],
    }


def selected_examples(models):
    by_name = {d['label']: d for d in models}
    selections = [('4b', 24, 50), ('4b', 27, 7), ('12b', 36, 6), ('12b', 36, 4), ('12b', 35, 42)]
    examples = []
    for model, layer, token_number in selections:
        d = by_name[model]; i = token_number-1
        examples.append({'model': model, 'layer': layer, 'token_number': token_number,
                         'scored_text_token': d['tokens'][i], 'lens_top1': d['lens'][str(layer)]['top1'][i],
                         'prefix_suffix': ''.join(d['tokens'][:i])[-70:],
                         'basis': 'post-hoc selected examples, not a predeclared category count'})
    return examples


def compute(root=HERE):
    models = extract(root)
    assert [d['label'] for d in models] == ['4b', '12b']
    assert models[0]['tokens'] == models[1]['tokens'] and models[0]['chat'] == models[1]['chat']
    assert models[0]['forced'] is False and models[1]['forced'] is True
    results = [summary(d) for d in models]
    a, b = results
    assert a['paths']['lens']['ever_rank1'] == 76 and a['paths']['lens']['never_rank1'] == 20
    assert a['paths']['lens']['first_rank1_median_among_found'] == 29
    assert a['paths']['logit_lens']['ever_rank1'] == 86
    assert a['paths']['lens']['lost_rank1_later_count'] == 22
    assert b['model_rank1_of_scored_text'] == 68
    first_layer_hits = [i+1 for i, r in enumerate(models[0]['lens']['1']['rank_actual']) if r == 1]
    assert first_layer_hits == [82, 83, 84, 85]
    assert all(models[0]['tokens'][i-1] == models[0]['tokens'][i-2] == '0' for i in first_layer_hits)
    band = [r for r in a['by_layer'] if 20 <= r['layer'] <= 30]
    return {
        'source_count': source_check(root),
        'scope': 'one selected prompt, one 96-token 4B continuation, 12B teacher-forced on that continuation',
        'independent_prompt_count': 1,
        'array_cells_checked_including_model_and_both_readouts': sum(validate(d) for d in models),
        'models': results,
        'selected_examples': selected_examples(models),
        'early_zero_run': {'token_numbers': first_layer_hits, 'all_equal_preceding_token_string': True,
                           'previous_token_matches_L1': a['paths']['lens']['previous_token_string_matches_L1'],
                           'previous_token_denominator': 95},
        'band20_30_4b': {
            'lens_more_rank1_at_layers': [r['layer'] for r in band if r['lens']['rank1'] > r['logit_lens']['rank1']],
            'logit_more_rank1_at_layers': [r['layer'] for r in band if r['logit_lens']['rank1'] > r['lens']['rank1']],
            'lens_higher_mean_p_layers': [r['layer'] for r in band if r['lens']['mean_p_actual'] > r['logit_lens']['mean_p_actual']],
            'lens_higher_median_p_layers': [r['layer'] for r in band if r['lens']['median_p_actual'] > r['logit_lens']['median_p_actual']],
        },
        'nominal_positions': {'convention': 'one-based preceding-token positions inferred from n_prompt and n_tokens',
                              'first': 47, 'last': 142, 'beyond_128_cap': 14,
                              'exact_fit_position_selector_and_capture_alignment_verified': False},
        'provenance_limits': ['no producer capture script or original numerical archives in this source',
                              'no lens file hashes or full checkpoint weight manifest in this source',
                              'no token IDs; token identity matches refer to decoded strings',
                              'fit CUDA widths 32/16 versus reported CPU width-one capture are not an identical schedule',
                              'no repeat runs, matched interventions, random/rotated lens null or held-out prompt sample',
                              'all new metrics and examples chosen after inspecting the data'],
        'unexecuted': ['model/tokenizer/forward/lens reconstruction', 'checkpoint and matrix digest validation',
                       'real path/precision/width acceptance for these maps and this capture',
                       'workspace causality, generalisation or behavioural steering test'],
    }


if __name__ == '__main__':
    print(json.dumps(compute(), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))
