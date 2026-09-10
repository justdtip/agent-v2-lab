#!/bin/bash
# regression of workspace_capture_v2.py against the existing wtest2 capture (same 3 rows, CPU): identical index and memmaps; then a ranged run
set -u
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cpu HF_HOME=/workspace/.hf_home CUDA_VISIBLE_DEVICES=""
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
CORP=/workspace/rendered-corpus/agent_v2e-gemma3-4b/test.jsonl
MAPS=/workspace/chief/out/lens4b-f32/exact-maps.npz
rm -rf /workspace/chief/out/wtest3 /workspace/chief/out/wtest3r
echo "$(date -u +%H:%M:%SZ) full 3-row run"
"$PY" /workspace/chief/workspace_capture_v2.py "$S4B" "$CORP" "$MAPS" /workspace/chief/out/wtest3 --model 4b --device cpu --max-rows 3 --sample-per-family 1 > /workspace/chief/wtest3.log 2>&1; echo "full rc=$?"
echo "$(date -u +%H:%M:%SZ) ranged run rows 1-3, no sample"
"$PY" /workspace/chief/workspace_capture_v2.py "$S4B" "$CORP" "$MAPS" /workspace/chief/out/wtest3r --model 4b --device cpu --rows-from 1 --rows-to 3 --skip-sample > /workspace/chief/wtest3r.log 2>&1; echo "ranged rc=$?"
"$PY" - <<PYEOF
import json, numpy as np
a = [json.loads(l) for l in open("/workspace/chief/out/wtest2/index.jsonl")]; b = [json.loads(l) for l in open("/workspace/chief/out/wtest3/index.jsonl")]
print("index identical (v1 vs v2, 3 rows):", a == b, "| rows", len(a), len(b))
for f in ("residual_note", "residual_act"):
    x = np.load(f"/workspace/chief/out/wtest2/{f}.npy"); y = np.load(f"/workspace/chief/out/wtest3/{f}.npy"); print(f, "memmap identical:", x.shape == y.shape and np.array_equal(x, y), x.shape)
r = [json.loads(l) for l in open("/workspace/chief/out/wtest3r/index.jsonl")]; print("ranged index i:", [q["i"] for q in r], "| equals rows 1,2 of the full index:", r == b[1:3])
z = np.load("/workspace/chief/out/wtest3r/residual_act.npy"); y = np.load("/workspace/chief/out/wtest3/residual_act.npy"); print("ranged memmap shape", z.shape, "| equals full rows 1,2:", np.array_equal(z, y[1:3]))
m = json.load(open("/workspace/chief/out/wtest3r/manifest.json")); print("ranged manifest:", {k: m[k] for k in ("decisions", "rows_from", "rows_to", "corpus_decisions", "memmap_row")})
PYEOF
echo "$(date -u +%H:%M:%SZ) test done"
