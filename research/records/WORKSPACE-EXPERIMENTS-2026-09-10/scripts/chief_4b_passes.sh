#!/bin/bash
# The 4B workspace passes, beside c3 (width 8, ~60 GiB): capture (all decisions, the 300-sample) then W-3b on the sample.
set -u
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cuda:0 HF_HOME=/workspace/.hf_home AGENT_V2_BOX_STATE_DIR=/workspace/box-chief-ws
mkdir -p /workspace/box-chief-ws
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
CORP=/workspace/rendered-corpus/agent_v2e-gemma3-4b
LOG=/workspace/chief/overnight.log
status() { echo "$(date -u +%H:%MZ) STATUS $*" | tee -a "$LOG"; }
while pgrep -f "fit_lens_f32.py .*lens12b-f32-c2" > /dev/null; do sleep 30; done
sleep 60
cat "$CORP/train.jsonl" "$CORP/valid.jsonl" "$CORP/test.jsonl" > /workspace/chief/all-splits.jsonl
status "start ws-capture-4b (beside c3, shared card)"
"$PY" -m local_llm_lab.runlock run --seat chief-ws --purpose "chief: workspace capture 4B, all decisions + 300-sample, float32 width 1 (shared card)" --minutes 180 -- \
  "$PY" /workspace/chief/workspace_capture.py "$S4B" /workspace/chief/all-splits.jsonl /workspace/chief/out/lens4b-f32/exact-maps.npz /workspace/chief/captures/4b --model 4b --device cuda:0 > /workspace/chief/ws-capture-4b.log 2>&1 < /dev/null
status "end ws-capture-4b rc=$? $(grep -E '"event": "done"' /workspace/chief/ws-capture-4b.log | tail -1 | cut -c1-120)"
status "start ws-w3b-4b"
"$PY" -m local_llm_lab.runlock run --seat chief-ws --purpose "chief: W-3b masked-carrier arms, 4B, the 300-sample (shared card)" --minutes 120 -- \
  "$PY" /workspace/chief/workspace_w3b.py "$S4B" /workspace/chief/all-splits.jsonl /workspace/chief/out/lens4b-f32/exact-maps.npz /workspace/chief/captures/4b /workspace/chief/captures/4b-w3b --device cuda:0 > /workspace/chief/ws-w3b-4b.log 2>&1 < /dev/null
status "end ws-w3b-4b rc=$? $(grep -E '"event": "done"' /workspace/chief/ws-w3b-4b.log | tail -1 | cut -c1-120)"
status "4b passes finished"
