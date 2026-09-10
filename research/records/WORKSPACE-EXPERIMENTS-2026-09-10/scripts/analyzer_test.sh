#!/bin/bash
set -u; cd /workspace/chief; export CUDA_VISIBLE_DEVICES=""; PY=/workspace/agent-v2-lab/.venv/bin/python
echo "--- schema-2 test output (no receipts): expect admitted"
$PY workspace_w3b_analyze.py out/w3btest2 out/w3btest2/analysis_v2.json; echo "rc=$?"
$PY -c "
import json; a=json.load(open('out/w3btest2/analysis_v2.json')); print('verdict', a['verdict'], '| admission', {k:v for k,v in a['admission'].items() if k!='ineligible_rows'})
for arm, d in a['arms'].items(): print(arm, 'transition', d['expert_top_transition_paired'], 'retained', d['retained_share_of_jointly_resolved'], 'agree', d['masked_expert_top_agreement'], 'paired_vs_random n', (d['paired_vs_random'] or {}).get('n'))"
echo "--- corrupted copy: a nonzero masked-edge attention on one arm, a duplicated row, a stale manifest: expect refused, rc=1"
rm -rf out/w3bcorrupt; cp -r out/w3btest2 out/w3bcorrupt
$PY -c "
import json; rows=[json.loads(l) for l in open('out/w3bcorrupt/w3b.jsonl')]; r=rows[0]; r['arms']['latest_result']['gate']['attention_on_masked_edges_all_queries']=0.5
open('out/w3bcorrupt/w3b.jsonl','w').write(json.dumps(r)+'\n'+json.dumps(rows[0])+'\n')"
$PY workspace_w3b_analyze.py out/w3bcorrupt out/w3bcorrupt/analysis.json; echo "rc=$? (expect 1)"
$PY workspace_w3b_analyze.py out/w3bcorrupt out/w3bcorrupt/analysis_diag.json --diagnostic; echo "diagnostic rc=$? (expect 0)"
$PY -c "
import json; a=json.load(open('out/w3bcorrupt/analysis_diag.json')); print('verdict', a['verdict'], '| set_problems', a['admission']['set_problems'], '| ineligible', a['admission']['ineligible_rows'])"
echo "--- gained-winner fixture: expert tool not top unmasked, top under the mask: expect gained=1, retained share null"
rm -rf out/w3bgain; cp -r out/w3btest2 out/w3bgain
$PY -c "
import json; rows=[json.loads(l) for l in open('out/w3bgain/w3b.jsonl')]; r=rows[0]; ti=r['tool_idx']
u=r['arms']['unmasked']['model_six_act']; s=[0.05]*6; s[ti]=0.1; s[(ti+1)%6]=0.65; u[0]=s
open('out/w3bgain/w3b.jsonl','w').write(json.dumps(r)+'\n')"
$PY workspace_w3b_analyze.py out/w3bgain out/w3bgain/analysis.json; $PY -c "
import json; a=json.load(open('out/w3bgain/analysis.json')); d=a['arms']['current_note']; print('current_note transition', d['expert_top_transition_paired'], 'retained', d['retained_share_of_jointly_resolved'], 'agree', d['masked_expert_top_agreement'])"
