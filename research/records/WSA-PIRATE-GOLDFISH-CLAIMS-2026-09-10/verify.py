"""Independent count checks and adversarial source/array tests; no model imports."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile

import analyze

HERE = Path(__file__).resolve().parent


def main():
    # JSON object keys are strings; normalize the in-memory histogram before receipt comparison.
    result = json.loads(json.dumps(analyze.compute()))
    assert result == json.loads((HERE/'analysis.json').read_text())
    models = analyze.extract(HERE)
    for d, computed in zip(models, result['models']):
        n = d['n_generated']
        for method in ('lens', 'logit_lens'):
            hit_sets = [{i for i, rank in enumerate(d[method][str(layer)]['rank_actual']) if rank == 1}
                        for layer in d['repo_layers']]
            ever = set.union(*hit_sets)
            assert len(ever) == computed['paths'][method]['ever_rank1']
            first = {i: min(layer for layer, hit in zip(d['repo_layers'], hit_sets) if i in hit)
                     for i in ever}
            assert statistics.median(first.values()) == computed['paths'][method]['first_rank1_median_among_found']
            loses = {i for i in ever if any(i not in hit for layer, hit in zip(d['repo_layers'], hit_sets)
                                           if layer > first[i])}
            assert len(loses) == computed['paths'][method]['lost_rank1_later_count']
            for row, hits in zip(computed['by_layer'], hit_sets):
                assert row[method]['rank1'] == len(hits)
                assert abs(sum(d[method][str(row['layer'])]['p_actual'])/n-row[method]['mean_p_actual']) < 1e-12
    assert result['array_cells_checked_including_model_and_both_readouts'] == 15552
    assert result['early_zero_run']['token_numbers'] == [82, 83, 84, 85]
    assert result['early_zero_run']['previous_token_matches_L1'] == 68
    assert result['band20_30_4b']['logit_more_rank1_at_layers'] == list(range(20,31))
    assert result['band20_30_4b']['lens_higher_mean_p_layers'] == [20]
    assert result['models'][1]['model_rank1_of_scored_text'] == 68
    assert not (analyze.BLOCKED & set(sys.modules))
    altered_arrays_refused = []
    for case in ('missing_token', 'negative_probability', 'nan_probability', 'zero_rank',
                 'negative_entropy', 'impossible_probability_rank', 'broken_text_join'):
        d = copy.deepcopy(models[0])
        if case == 'missing_token': d['lens']['1']['p_actual'].pop()
        if case == 'negative_probability': d['lens']['1']['p_actual'][0] = -0.1
        if case == 'nan_probability': d['lens']['1']['p_actual'][0] = float('nan')
        if case == 'zero_rank': d['lens']['1']['rank_actual'][0] = 0
        if case == 'negative_entropy': d['lens']['1']['entropy'][0] = -1
        if case == 'impossible_probability_rank':
            d['lens']['1']['rank_actual'][0] = 3
            d['lens']['1']['p_actual'][0] = 0.8
        if case == 'broken_text_join': d['tokens'][0] = 'changed'
        try: analyze.validate(d)
        except AssertionError: altered_arrays_refused.append(case)
        else: raise AssertionError(case)
    with tempfile.TemporaryDirectory(prefix='goldfish-verify-') as temp:
        replica = Path(temp)/'record'
        shutil.copytree(HERE, replica, ignore=shutil.ignore_patterns('__pycache__'))
        rows = json.loads((replica/'sources.json').read_text())['inputs']
        corrupted_sources_refused = []
        for item in rows:
            p = replica/item['snapshot']; original = p.read_bytes(); p.write_bytes(original+b'changed')
            try: analyze.source_check(replica)
            except ValueError: corrupted_sources_refused.append(item['snapshot'])
            else: raise AssertionError(item['snapshot'])
            finally: p.write_bytes(original)
        run = subprocess.run([sys.executable,'-B',str(replica/'analyze.py')],cwd=temp,
                             text=True,capture_output=True,check=True)
        assert result == json.loads(run.stdout)
    print(json.dumps({'basis':'model-free verification of recorded arrays, not original measurement reproduction',
                      'source_corruptions_refused':corrupted_sources_refused,
                      'array_mutations_refused':altered_arrays_refused,
                      'independent_count_formulations_agree':True,'relocated_output_reproduced':True,
                      'recorded_output_reproduced':True,'model_modules_imported':[],
                      'analyze_sha256':hashlib.sha256((HERE/'analyze.py').read_bytes()).hexdigest(),
                      'verify_sha256':hashlib.sha256((HERE/'verify.py').read_bytes()).hexdigest(),
                      'experimental_workspace_gates_passed_by_this_audit':0,
                      'unexecuted':result['unexecuted']},indent=2,sort_keys=True))


if __name__ == '__main__':main()
