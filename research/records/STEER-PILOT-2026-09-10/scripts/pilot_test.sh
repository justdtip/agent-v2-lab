#!/bin/bash
# CPU smoke of the steering pilot: 1 recipient, 2 layers, 20 new tokens
set -u; cd /workspace/chief
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cpu HF_HOME=/workspace/.hf_home CUDA_VISIBLE_DEVICES=""
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
rm -rf out/pilottest
echo "$(date -u +%H:%M:%SZ) pilot smoke"
$PY steer_pilot.py "$S4B" /workspace/chief/all-splits.jsonl out/pilottest --device cpu --n-recipients 1 --layers 8,20 --max-new-tokens 20 > pilottest.log 2>&1; echo "pilot rc=$?"
grep -v "Loading weights\|^\[transformers\]" pilottest.log | cut -c1-400 | tail -n 6
$PY - <<PYEOF
import json
r = json.loads(open("out/pilottest/pilot.jsonl").readline())
print("recipient:", r["recipient"]["task_id"], "step", r["recipient"]["step"], "expert path", r["recipient"]["path"])
print("baseline text:", repr(r["baseline"]["text"][:120]), "| tool", r["baseline"]["tool"], "| path", r["baseline"]["path"], "| valid", r["baseline"]["valid_json"], "| first top", r["baseline"]["first_top_tool"])
for L, lay in r["layers"].items():
    for arm, x in lay.items():
        print(f"L{L} {arm:17s} tool={x['tool']} path={x['path']} class={x['path_class']} tool_pres={x['tool_preserved']} identical={x.get('identical_to_baseline')} applied={x['patch_applied']} text={x['text'][:70]!r}")
PYEOF
echo "$(date -u +%H:%M:%SZ) pilot smoke done"
