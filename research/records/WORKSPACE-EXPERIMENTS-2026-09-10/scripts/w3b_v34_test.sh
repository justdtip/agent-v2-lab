#!/bin/bash
# v3.2 control regression on the CPU: one row of the three-row test capture (row 0: step 1, search — carriers present), then the
# v3.2 analyzer on that output, on the 4B v3.1 pass (expect admitted, control labelled degenerate) and on the schema-2 fixtures.
set -u; cd /workspace/chief
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cpu HF_HOME=/workspace/.hf_home CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
rm -rf out/w3btest8
echo "$(date -u +%H:%M:%SZ) v3.4 one-row CPU test (row 0 of wtest2: step 1, search)"
$PY workspace_w3b_v34.py "$S4B" /workspace/rendered-corpus/agent_v2e-gemma3-4b/test.jsonl out/lens4b-f32/exact-maps.npz out/wtest2 out/w3btest8 --device cpu --max-sample 1 > w3btest8.log 2>&1; echo "v3.4 rc=$?"
grep -v "Loading weights\|^\[transformers\]" w3btest8.log | cut -c1-400 | tail -n 6
$PY workspace_w3b_analyze.py out/w3btest8 out/w3btest8/analysis.json; echo "analyzer rc=$?"
$PY - <<'PYEOF'
import json; r=json.loads(open('out/w3btest8/w3b.jsonl').readline()); print('placement_oracle', r['placement_oracle'], 'q0', r['q0'], 'P_note', r['P_note'], 'prose', r['note_prose_tokens']); print('random_control', json.dumps(r['random_control']))
for arm, x in r['arms'].items():
    g=x['gate']; rc=x.get('receipt', {})
    print('%-30s blocked@P_act=%4s total=%6s masked_all=%s above_diag=%s far_all=%s receipt_pass=%s p_note_identical=%s' % (arm, g['n_blocked_edges_at_P_act'], g['n_blocked_edges_total'], g['attention_on_masked_edges_all_queries'], g['attention_above_diagonal'], g['local_mass_beyond_window_all_queries'], rc.get('passes'), x.get('p_note_identical_to_unmasked')))
car=set(r['arms']['all_carriers']['receipt']['expected_keys'])
for arm in ('random_equal_count','random_equal_count_note'):
    ks=r['arms'][arm]['receipt']['expected_keys']; print(arm, 'keys', len(ks), 'min', min(ks), 'max', max(ks), 'overlap with carriers', len(set(ks)&car), 'query_interval', r['arms'][arm]['receipt']['query_interval'])
a=json.load(open('out/w3btest8/analysis.json')); print('verdict', a['verdict'], '| script_version', a['script_version'], '| control_status', a['control_status'][:70])
for arm,d in a['arms'].items(): print('  %-30s paired control %s n %s' % (arm, (d['paired_vs_random'] or {}).get('control_arm'), (d['paired_vs_random'] or {}).get('n')))
PYEOF
echo "--- the v3.2 analyzer on the 4B v3.1 pass: expect admitted 300, control labelled degenerate"
$PY workspace_w3b_analyze.py captures/4b-w3b out/w3b-4b-v31-analysis-by-v34.json | cut -c1-300; echo "rc=$?"
$PY -c "
import json; a=json.load(open('out/w3b-4b-v31-analysis-by-v34.json')); print('verdict', a['verdict'], 'eligible', a['admission']['eligible'], '| script_version', a['script_version'], '| control_status', a['control_status'][:90]); print('current_note paired control', a['arms']['current_note']['paired_vs_random']['control_arm'], a['arms']['current_note']['paired_vs_random']['n'], '| previous_note paired', a['arms']['previous_note']['paired_vs_random']['control_arm'], a['arms']['previous_note']['paired_vs_random']['n'])"
echo "--- analyzer fixtures (schema-2 outputs and corrupted copies)"
if [ -f analyzer_test.sh ]; then sed "s#workspace_w3b_analyze.py#workspace_w3b_analyze.py#g" analyzer_test.sh > analyzer_test_v34.sh; bash analyzer_test_v34.sh 2>&1 | tail -n 14 | cut -c1-300; else echo "no analyzer_test.sh on the card"; fi
echo "$(date -u +%H:%M:%SZ) v3.2 test done"
echo "--- v3.3 analyzer on the 4B v3.2 pass: expect admitted 300, short rows 55"
$PY workspace_w3b_analyze.py captures/4b-w3b-v32 out/w3b-4b-v32-analysis-by-v34.json | cut -c1-200; echo "rc=$?"
$PY -c "
import json; a=json.load(open('out/w3b-4b-v32-analysis-by-v34.json')); print('verdict', a['verdict'], 'eligible', a['admission']['eligible'], '| carrier_control_short_rows', a['carrier_control_short_rows'], '| subsampled rows', a['all_carriers_matched_subsampled_rows'])
rows=[json.loads(l) for l in open('out/w3btest8/w3b.jsonl')]
for r in rows: rc=r['random_control']; print('test row', r['i'], 'step', r['step'], 'pool', rc['pool_size'], '| carriers', rc.get('all_carriers_matched'), '| control', rc.get('random_equal_count'), '| arms', [k for k in r['arms']])"
echo "$(date -u +%H:%M:%SZ) v3.3 test done (again)"
echo "--- digest guard on the real captures (metadata only): 4b, 12b (running), and a corrupted copy: expect source = capture-time event twice, then refused"
$PY - <<'PYEOF'
import json, sys, importlib.util, types, re
src=open("/workspace/chief/workspace_w3b_v34.py").read()
i=src.index("def capture_corpus_digest"); j=src.index("corpus_sha = hashlib.sha256(Path(args.corpus)")
ns={"json": json, "Path": __import__("pathlib").Path}; exec(src[i:j], ns)
for d in ("/workspace/chief/captures/4b", "/workspace/chief/captures/12b"):
    print(d, ns["capture_corpus_digest"](d, json.load(open(d+"/manifest.json")) if __import__("os").path.exists(d+"/manifest.json") else {"corpus": "/x"}))
import tempfile, shutil
with tempfile.TemporaryDirectory() as t:
    open(t+"/progress.jsonl","w").write('{"event": "corpus", "corpus_sha256": "' + "0"*64 + '"}\n')
    print("wrong digest reported (the assert then refuses):", ns["capture_corpus_digest"](t, {"corpus": "/workspace/chief/all-splits.jsonl"})[0][:12])
with tempfile.TemporaryDirectory() as t:
    open(t+"/progress.jsonl","w").write('{"event": "loaded"}\n')
    try: ns["capture_corpus_digest"](t, {"corpus": "/workspace/chief/all-splits.jsonl"}); print("BUG: accepted without a digest")
    except SystemExit as e: print("refused as expected:", str(e)[:80])
PYEOF
echo "$(date -u +%H:%M:%SZ) v3.4 test done"
