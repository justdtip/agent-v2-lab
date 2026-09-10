#!/bin/bash
set -u; cd /workspace/chief
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cpu HF_HOME=/workspace/.hf_home CUDA_VISIBLE_DEVICES=""
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
rm -rf out/w3btest3
echo "$(date -u +%H:%M:%SZ) v3 one-row CPU test"
$PY workspace_w3b_v3.py "$S4B" /workspace/rendered-corpus/agent_v2e-gemma3-4b/test.jsonl out/lens4b-f32/exact-maps.npz out/wtest2 out/w3btest3 --device cpu --max-sample 1 > w3btest3.log 2>&1; echo "v3 rc=$?"
grep -v "Loading weights\|^\[transformers\]" w3btest3.log | cut -c1-300 | tail -n 5
$PY workspace_w3b_analyze.py out/w3btest3 out/w3btest3/analysis.json; echo "analyzer rc=$?"
$PY -c "
import json; r=json.loads(open('out/w3btest3/w3b.jsonl').readline()); print('placement_oracle', r['placement_oracle'], 'q0', r['q0'], 'prose', r['note_prose_tokens'], 'syntax', r['note_syntax_tokens'])
for arm, x in r['arms'].items():
    g=x['gate']; rc=x.get('receipt', {})
    print('%-30s blocked@P_act=%4s total=%6s masked_all=%s above_diag=%s far_all=%s pre_cut=%.4g receipt=%s' % (arm, g['n_blocked_edges_at_P_act'], g['n_blocked_edges_total'], g['attention_on_masked_edges_all_queries'], g['attention_above_diagonal'], g['local_mass_beyond_window_all_queries'], g['pre_cut_queries_mass_on_masked_keys'], {k: rc[k] for k in ('passes','missing_edges','unexpected_edges','causality_preserved','pre_cut_queries_still_attend','window_preserved_all_queries')} if rc else None))
a=json.load(open('out/w3btest3/analysis.json')); print('analysis verdict', a['verdict'], 'receipts_present_arms', a['admission']['receipts_present_arms'])"
echo "$(date -u +%H:%M:%SZ) v3 test done"
