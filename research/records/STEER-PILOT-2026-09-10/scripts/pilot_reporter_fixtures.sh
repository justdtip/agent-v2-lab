#!/bin/bash
# the reporter on corrupted copies of the smoke output: duplicate row, failed same-state (void cell), missing manifest, empty store
set -u; cd /workspace/chief; PY=/workspace/agent-v2-lab/.venv/bin/python; SRC=out/pilottest
mk() { rm -rf "$1"; cp -r "$SRC" "$1"; }
mk out/pt_dup; $PY -c "
rows=open('out/pt_dup/pilot.jsonl').read().splitlines(); open('out/pt_dup/pilot.jsonl','w').write('\n'.join(rows+[rows[0]])+'\n')"
$PY steer_pilot_analyze.py out/pt_dup out/pt_dup/a.json >/dev/null 2>&1; echo "duplicate row: rc=$? (expect 1)"; $PY -c "import json; a=json.load(open('out/pt_dup/a.json')); print('  ', a['verdict'], a['admission']['problems'])"
mk out/pt_void; $PY -c "
import json; rows=[json.loads(l) for l in open('out/pt_void/pilot.jsonl')]; rows[0]['layers']['20']['same_state']['identical_to_baseline']=False
open('out/pt_void/pilot.jsonl','w').write('\n'.join(json.dumps(r) for r in rows)+'\n')"
$PY steer_pilot_analyze.py out/pt_void out/pt_void/a.json >/dev/null 2>&1; echo "failed same-state at (row0, L20): rc=$? (expect 0, admitted with the cell void)"; $PY -c "
import json; a=json.load(open('out/pt_void/a.json')); print('  ', a['verdict'], '| void', a['admission']['void_row_layers'], '| L20 void_rows', a['per_layer_arm']['20']['void_rows'], '| L20 operation n', a['per_layer_arm']['20']['operation']['n'], '| operation switches L20', a['switch_counts_by_layer']['operation']['20'], '(was 2)')"
mk out/pt_noman; rm -f out/pt_noman/manifest.json
$PY steer_pilot_analyze.py out/pt_noman out/pt_noman/a.json >/dev/null 2>&1; echo "no manifest: rc=$? (expect 1)"; $PY -c "import json; a=json.load(open('out/pt_noman/a.json')); print('  ', a['verdict'], a['admission']['problems'])"
mk out/pt_empty; : > out/pt_empty/pilot.jsonl
$PY steer_pilot_analyze.py out/pt_empty out/pt_empty/a.json >/dev/null 2>&1; echo "empty store: rc=$? (expect 1)"; $PY -c "import json; a=json.load(open('out/pt_empty/a.json')); print('  ', a['verdict'], a['admission']['problems'][:2])"
mk out/pt_applied; $PY -c "
import json; rows=[json.loads(l) for l in open('out/pt_applied/pilot.jsonl')]; rows[1]['layers']['8']['opposite_target']['patch_applied']=0
open('out/pt_applied/pilot.jsonl','w').write('\n'.join(json.dumps(r) for r in rows)+'\n')"
$PY steer_pilot_analyze.py out/pt_applied out/pt_applied/a.json >/dev/null 2>&1; echo "patch_applied=0 on one arm: rc=$? (expect 1)"; $PY -c "import json; a=json.load(open('out/pt_applied/a.json')); print('  ', a['verdict'], a['admission']['row_problems'])"
echo "=== digests on the card vs the record"; sha256sum steer_pilot.py steer_pilot_analyze.py | cut -c1-12,66-
