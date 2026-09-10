#!/bin/bash
# CPU regression of capture v2.1 (logits at two positions) against wtest3 (v2, full logits), then w3b v3.1 on it
set -u; cd /workspace/chief
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cpu HF_HOME=/workspace/.hf_home CUDA_VISIBLE_DEVICES=""
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
CORP=/workspace/rendered-corpus/agent_v2e-gemma3-4b/test.jsonl; MAPS=/workspace/chief/out/lens4b-f32/exact-maps.npz
rm -rf out/wtest4 out/w3btest4
echo "$(date -u +%H:%M:%SZ) capture v2.1, 3 rows"
$PY workspace_capture_v21.py "$S4B" "$CORP" "$MAPS" out/wtest4 --model 4b --device cpu --max-rows 3 --sample-per-family 1 > wtest4.log 2>&1; echo "capture rc=$?"; grep "logits_to_keep_check" out/wtest4/progress.jsonl | cut -c1-160
$PY - <<PYEOF
import json, numpy as np
a = [json.loads(l) for l in open("out/wtest3/index.jsonl")]; b = [json.loads(l) for l in open("out/wtest4/index.jsonl")]
print("index identical to v2 (full logits):", a == b)
if a != b:
    for r, q in zip(a, b):
        for k in r:
            if r[k] != q[k]: print("  differs:", k, str(r[k])[:80], "|", str(q[k])[:80]); break
for f in ("residual_note", "residual_act"): print(f, "identical:", np.array_equal(np.load(f"out/wtest3/{f}.npy"), np.load(f"out/wtest4/{f}.npy")))
print("sample.jsonl identical:", open("out/wtest3/sample.jsonl").read() == open("out/wtest4/sample.jsonl").read())
PYEOF
echo "$(date -u +%H:%M:%SZ) w3b v3.1, 1 row"
$PY workspace_w3b_v31.py "$S4B" "$CORP" "$MAPS" out/wtest4 out/w3btest4 --device cpu --max-sample 1 > w3btest4.log 2>&1; echo "w3b rc=$?"
$PY - <<PYEOF
import json
a = json.loads(open("out/w3btest3/w3b.jsonl").readline()); b = json.loads(open("out/w3btest4/w3b.jsonl").readline())
same = {arm: all(json.dumps(a["arms"][arm][f]) == json.dumps(b["arms"][arm][f]) for f in ("model_six_note", "model_six_act", "margin_act_logits", "margin_note_logits", "lens_act")) for arm in a["arms"]}
print("w3b arms identical to v3 (full logits):", same)
print("receipts pass:", all(x.get("receipt", {}).get("passes", True) for x in b["arms"].values()), "| placement oracle:", b["placement_oracle"])
PYEOF
$PY workspace_w3b_analyze.py out/w3btest4 out/w3btest4/analysis.json; echo "analyzer rc=$?"
echo "$(date -u +%H:%M:%SZ) recover test done"
