#!/bin/bash
# (1) the --sample-ids-file path reproduces wtest3's sample row; (2) a simulated death at row 1 merged back from the ranged tail wtest3r equals wtest3
set -u
cd /workspace/chief
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cpu HF_HOME=/workspace/.hf_home CUDA_VISIBLE_DEVICES=""
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
CORP=/workspace/rendered-corpus/agent_v2e-gemma3-4b/test.jsonl
MAPS=/workspace/chief/out/lens4b-f32/exact-maps.npz
rm -rf /workspace/chief/out/wtest3s /workspace/chief/out/wtest3m; echo "[0]" > /workspace/chief/out/wtest3-ids.json
echo "$(date -u +%H:%M:%SZ) sample-ids run rows 0-3"
"$PY" /workspace/chief/workspace_capture_v2.py "$S4B" "$CORP" "$MAPS" /workspace/chief/out/wtest3s --model 4b --device cpu --rows-from 0 --rows-to 3 --sample-ids-file /workspace/chief/out/wtest3-ids.json > /workspace/chief/wtest3s.log 2>&1; echo "sample-ids rc=$?"
"$PY" - <<PYEOF
import json, numpy as np
a = open("/workspace/chief/out/wtest3/index.jsonl").read(); b = open("/workspace/chief/out/wtest3s/index.jsonl").read(); print("index identical:", a == b)
print("sample.jsonl identical:", open("/workspace/chief/out/wtest3/sample.jsonl").read() == open("/workspace/chief/out/wtest3s/sample.jsonl").read())
print("attn_0 identical:", np.array_equal(np.load("/workspace/chief/out/wtest3/attn_0.npy"), np.load("/workspace/chief/out/wtest3s/attn_0.npy")))
m = json.load(open("/workspace/chief/out/wtest3s/manifest.json")); print("manifest sample:", m["sample"])
PYEOF
echo "$(date -u +%H:%M:%SZ) simulated death: copy wtest3 -> wtest3m, zero rows 1-2, drop its index; reconstruct; merge the tail wtest3r"
cp -r /workspace/chief/out/wtest3 /workspace/chief/out/wtest3m; rm -f /workspace/chief/out/wtest3m/index.jsonl /workspace/chief/out/wtest3m/manifest.json
"$PY" - <<PYEOF
import numpy as np
for f in ("residual_note", "residual_act"):
    M = np.load(f"/workspace/chief/out/wtest3m/{f}.npy", mmap_mode="r+"); M[1:3] = 0.0; M.flush(); del M
print("zeroed rows 1-2")
PYEOF
"$PY" /workspace/chief/reconstruct_index.py /workspace/chief/out/wtest3m "$CORP" "$S4B" /workspace/chief/out/wtest3m/index.jsonl 2>&1 | grep -v "Loading\|transformers\]"; echo "reconstruct rc=$?"
"$PY" /workspace/chief/merge_tail_capture.py /workspace/chief/out/wtest3m /workspace/chief/out/wtest3r /workspace/chief/out/wtest3m/index.jsonl; echo "merge rc=$?"
"$PY" - <<PYEOF
import json, numpy as np
for f in ("residual_note", "residual_act"):
    print(f, "merged memmap equals wtest3:", np.array_equal(np.load(f"/workspace/chief/out/wtest3m/{f}.npy"), np.load(f"/workspace/chief/out/wtest3/{f}.npy")))
a = [json.loads(l) for l in open("/workspace/chief/out/wtest3/index.jsonl")]; b = [json.loads(l) for l in open("/workspace/chief/out/wtest3m/index.jsonl")]
det = ("i", "task_id", "step", "family", "variant", "recovery", "tool", "tool_idx", "P_note", "P_act", "n_prompt_tokens", "n_note_tokens", "in_sample")
print("merged index deterministic fields equal wtest3:", [{k: r[k] for k in det} for r in a] == [{k: r[k] for k in det} for r in b], "| rows", len(b), "| row0 reconstructed:", b[0].get("reconstructed"), "| rows 1-2 full:", all(b[j]["lens_act"] is not None for j in (1, 2)))
PYEOF
echo "$(date -u +%H:%M:%SZ) test2 done"
