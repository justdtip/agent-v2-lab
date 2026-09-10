#!/bin/bash
# The 4B W-3b v3.1 pass, gated on /workspace/chief/GO-W3B-4B so it cannot start into the D-CRO's timed slot.
set -u
cd /workspace/chief
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cuda:0 HF_HOME=/workspace/.hf_home AGENT_V2_BOX_STATE_DIR=/workspace/box-chief-ws
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
LOG=/workspace/chief/overnight.log
status() { echo "$(date -u +%H:%MZ) STATUS $*" | tee -a "$LOG"; }
until [ -s /workspace/chief/captures/4b/manifest.json ]; do sleep 60; done
status "4B capture manifest present; W-3b 4B waiting for /workspace/chief/GO-W3B-4B"
until [ -f /workspace/chief/GO-W3B-4B ]; do sleep 60; done
status "start ws-w3b-4b (v3.1)"
"$PY" -m local_llm_lab.runlock run --seat chief-ws --purpose "chief: W-3b v3.1 masked-carrier arms with receipts, 4B, the 300-sample (shared card)" --minutes 120 -- \
  "$PY" /workspace/chief/workspace_w3b.py "$S4B" /workspace/chief/all-splits.jsonl /workspace/chief/out/lens4b-f32/exact-maps.npz /workspace/chief/captures/4b /workspace/chief/captures/4b-w3b --device cuda:0 > /workspace/chief/ws-w3b-4b.log 2>&1 < /dev/null
status "end ws-w3b-4b rc=$? $(grep -E '"event": "done"' /workspace/chief/ws-w3b-4b.log | tail -1 | cut -c1-120)"
