#!/bin/bash
# v3.2 control regression on the CPU: one row of the three-row test capture (row 0: step 1, search — carriers present), then the
# v3.2 analyzer on that output, on the 4B v3.1 pass (expect admitted, control labelled degenerate) and on the schema-2 fixtures.
set -u; cd /workspace/chief
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cpu HF_HOME=/workspace/.hf_home CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
rm -rf out/w3btest5
echo "$(date -u +%H:%M:%SZ) v3.2 one-row CPU test (row 0 of wtest2: step 1, search)"
$PY workspace_w3b_v32.py "$S4B" /workspace/rendered-corpus/agent_v2e-gemma3-4b/test.jsonl out/lens4b-f32/exact-maps.npz out/wtest2 out/w3btest5 --device cpu --max-sample 1 > w3btest5.log 2>&1; echo "v3.2 rc=$?"
grep -v "Loading weights\|^\[transformers\]" w3btest5.log | cut -c1-400 | tail -n 6
$PY workspace_w3b_analyze_v32.py out/w3btest5 out/w3btest5/analysis.json; echo "analyzer rc=$?"
$PY - <<'PYEOF'
import json; r=json.loads(open('out/w3btest5/w3b.jsonl').readline()); print('placement_oracle', r['placement_oracle'], 'q0', r['q0'], 'P_note', r['P_note'], 'prose', r['note_prose_tokens']); print('random_control', json.dumps(r['random_control']))
for arm, x in r['arms'].items():
    g=x['gate']; rc=x.get('receipt', {})
    print('%-30s blocked@P_act=%4s total=%6s masked_all=%s above_diag=%s far_all=%s receipt_pass=%s p_note_identical=%s' % (arm, g['n_blocked_edges_at_P_act'], g['n_blocked_edges_total'], g['attention_on_masked_edges_all_queries'], g['attention_above_diagonal'], g['local_mass_beyond_window_all_queries'], rc.get('passes'), x.get('p_note_identical_to_unmasked')))
car=set(r['arms']['all_carriers']['receipt']['expected_keys'])
for arm in ('random_equal_count','random_equal_count_note'):
    ks=r['arms'][arm]['receipt']['expected_keys']; print(arm, 'keys', len(ks), 'min', min(ks), 'max', max(ks), 'overlap with carriers', len(set(ks)&car), 'query_interval', r['arms'][arm]['receipt']['query_interval'])
a=json.load(open('out/w3btest5/analysis.json')); print('verdict', a['verdict'], '| script_version', a['script_version'], '| control_status', a['control_status'][:70])
for arm,d in a['arms'].items(): print('  %-30s paired control %s n %s' % (arm, (d['paired_vs_random'] or {}).get('control_arm'), (d['paired_vs_random'] or {}).get('n')))
PYEOF
echo "--- the v3.2 analyzer on the 4B v3.1 pass: expect admitted 300, control labelled degenerate"
$PY workspace_w3b_analyze_v32.py captures/4b-w3b out/w3b-4b-v31-analysis-by-v32.json | cut -c1-300; echo "rc=$?"
$PY -c "
import json; a=json.load(open('out/w3b-4b-v31-analysis-by-v32.json')); print('verdict', a['verdict'], 'eligible', a['admission']['eligible'], '| script_version', a['script_version'], '| control_status', a['control_status'][:90]); print('current_note paired control', a['arms']['current_note']['paired_vs_random']['control_arm'], a['arms']['current_note']['paired_vs_random']['n'], '| previous_note paired', a['arms']['previous_note']['paired_vs_random']['control_arm'], a['arms']['previous_note']['paired_vs_random']['n'])"
echo "--- analyzer fixtures (schema-2 outputs and corrupted copies)"
if [ -f analyzer_test.sh ]; then sed "s#workspace_w3b_analyze.py#workspace_w3b_analyze_v32.py#g" analyzer_test.sh > analyzer_test_v32.sh; bash analyzer_test_v32.sh 2>&1 | tail -n 14 | cut -c1-300; else echo "no analyzer_test.sh on the card"; fi
echo "$(date -u +%H:%M:%SZ) v3.2 test done"
