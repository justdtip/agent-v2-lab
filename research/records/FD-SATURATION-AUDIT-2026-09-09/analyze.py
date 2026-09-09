"""File-only audit. Standard library only; never loads maps, weights, or a model."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent


def ranks(values):
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        for k in range(i, j):
            out[ordered[k][0]] = (i + j - 1) / 2 + 1
        i = j
    return out


def pearson(a, b):
    if len(a) != len(b) or len(a) < 2:
        raise ValueError('paired observations required')
    aa = [v - statistics.mean(a) for v in a]
    bb = [v - statistics.mean(b) for v in b]
    return math.fsum(x*y for x, y in zip(aa, bb)) / math.sqrt(
        math.fsum(x*x for x in aa) * math.fsum(y*y for y in bb))


def value(number, basis='derived-from-device-record'):
    return {'value': number, 'basis': basis}


def analyze():
    source = json.loads((ROOT / 'SOURCES.json').read_text())
    for entry in source['files']:
        data = (ROOT / entry['local_path']).read_bytes()
        assert len(data) == entry['bytes']
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
    z = json.loads((ROOT / 'source/zero-response.json').read_text())['per_layer']
    nu = json.loads((ROOT / 'source/nu-finite-difference.json').read_text())
    repo = nu['endpoint']['source_layers_repo']
    upstream = nu['endpoint']['source_layers_upstream']
    eps = nu['upstream']['epsilon_per_layer']
    scale = nu['upstream']['epsilon_scale']
    seq = nu['corpus']['max_seq_len']
    assert len(repo) == len(upstream) == len(z)
    assert [r['repo_layer'] for r in z] == repo
    # Reproduce the README's proxy; it is not a measured target-token norm.
    proxy = eps[str(upstream[-1])]['mean'] / scale / math.sqrt(seq)
    rows = []
    for r, block in zip(z, upstream):
        h = eps[str(block)]['mean']
        excursion = h * r['median_col_norm_exact']
        ratio = r['frobenius_fd'] / r['frobenius_exact']
        assert h > 0 and excursion > 0 and ratio > 0
        rows.append(dict(repo_layer=r['repo_layer'], upstream_block=block,
            epsilon=h, median_exact_column_norm=r['median_col_norm_exact'],
            exact_frobenius=r['frobenius_exact'], fd_frobenius=r['frobenius_fd'],
            reduced_linear_excursion=excursion, preceding_layer_rms_proxy=proxy,
            excursion_over_proxy=excursion/proxy, fd_over_exact_norm=ratio,
            basis='derived-from-device-record'))
    x = [r['excursion_over_proxy'] for r in rows]
    y = [r['fd_over_exact_norm'] for r in rows]
    reversals = []
    for previous, current in zip(rows, rows[1:]):
        dx = current['excursion_over_proxy'] - previous['excursion_over_proxy']
        dy = current['fd_over_exact_norm'] - previous['fd_over_exact_norm']
        if dy < 0:
            reversals.append({'from_repo_layer':previous['repo_layer'],
                'to_repo_layer':current['repo_layer'], 'delta_excursion':dx,
                'delta_compression':dy,'opposite_sign':dx*dy < 0,
                'exact_tie':dx == 0 or dy == 0, 'basis':'derived-from-device-record'})
    # Known-answer counterexample: F_l(x)=a_l*x, J_l=a_l*I, broken estimate G_l=I.
    # No saturation exists, but shared scale creates a perfect inverse relationship.
    a = [1., 4., 2., 8., 3., 16., 5.]
    fake_x, fake_y = a, [1/v for v in a]
    inverse_pearson = pearson(list(map(math.log, fake_x)), list(map(math.log, fake_y)))
    inverse_spearman = pearson(ranks(fake_x), ranks(fake_y))
    assert math.isclose(inverse_pearson, -1, abs_tol=1e-14)
    assert math.isclose(inverse_spearman, -1, abs_tol=1e-14)
    assert ranks([4, 1, 1, 8]) == [3., 1.5, 1.5, 4.]
    assert math.isclose(pearson([1, 2, 3], [2, 4, 6]), 1, abs_tol=1e-14)
    # Another logical counterexample, not a Gemma simulator: rounding to nearest integer.
    # J has diagonal 1 and off-diagonal .49; central differences at x=0,h=1 give I.
    d, off = 256, .49
    assert round(off) == 0 and round(-off) == 0 and round(1.) == 1
    quant_ratio = 1 / math.sqrt(1 + (d-1)*off*off)
    assert quant_ratio < .13
    # A zero aggregate does not imply that each contribution was zero.
    first, second = (1., 0.), (-1., 0.)
    assert all((a+b)/2 == 0 for a,b in zip(first,second))
    assert any(first) and any(second)
    result = {'source_commit':source['source_commit'],
        'scope':'reproduction of summary statistics and logical counterexamples; no mechanism identified',
        'saved_zero_columns':value(sum(r['zero_columns'] for r in z)),
        'preceding_layer_rms_proxy':value(proxy),
        'spearman':value(pearson(ranks(x),ranks(y))),
        'pearson_logs':value(pearson(list(map(math.log,x)),list(map(math.log,y)))),
        'excursion_over_proxy_range':value([min(x),max(x)]),
        'shared_exact_size_log_correlation':value(pearson(
            [math.log(r['median_exact_column_norm']) for r in rows],
            [math.log(r['exact_frobenius']) for r in rows])),
        'downward_compression_steps':value(len(reversals)),
        'opposite_sign_steps':value(sum(r['opposite_sign'] for r in reversals)),
        'exact_ties_among_downward_steps':value(sum(r['exact_tie'] for r in reversals)),
        'downward_steps':reversals,
        'linear_counterexample':{
            'construction':'F_l(x)=a_l*x; true J_l=a_l*I; erroneous reported G_l=I; fixed epsilon and target scale',
            'a':value(a,'synthetic-logical-counterexample'),
            'spearman':value(inverse_spearman,'synthetic-logical-counterexample'),
            'pearson_logs':value(inverse_pearson,'synthetic-logical-counterexample')},
        'rounding_counterexample':{
            'construction':'linear F(x)=Jx; diag(J)=1, offdiag(J)=.49; nearest-integer output rounding; x=0,h=1; FD=I',
            'width':value(d,'synthetic-logical-counterexample'),
            'zero_columns':value(0,'synthetic-logical-counterexample'),
            'fraction_components_lost':value((d-1)/d,'synthetic-logical-counterexample'),
            'fd_over_exact_norm':value(quant_ratio,'synthetic-logical-counterexample')},
        'verification':'input digests, layer join, rank ties, correlation known answers, linear coupling, rounding and cancellation counterexamples passed'}
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (ROOT/'layers.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ('downward_steps','linear_counterexample','rounding_counterexample')},indent=2))


if __name__ == '__main__':
    analyze()
