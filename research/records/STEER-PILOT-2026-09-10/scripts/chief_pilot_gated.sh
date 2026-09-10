#!/bin/bash
# The steering pilot (experiment 1, minimal form), 24 recipients × 8 layers × 5 patched arms, greedy complete calls.
# Starts when the 4B W-3b pass has started (phase 0), beside it: ~20 GiB against W-3b's ~25 on an otherwise empty card.
set -u
cd /workspace/chief
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cuda:0 HF_HOME=/workspace/.hf_home AGENT_V2_BOX_STATE_DIR=/workspace/box-chief-pilot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/box-chief-pilot
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
LOG=/workspace/chief/overnight.log
status() { echo "$(date -u +%H:%MZ) STATUS $*" | tee -a "$LOG"; }
until grep -q "start ws-w3b-4b (v3.1)" "$LOG" 2>/dev/null; do sleep 30; done
sleep 60
status "start steer-pilot-4b (beside W-3b)"
"$PY" -m local_llm_lab.runlock run --seat chief-pilot --purpose "chief: steering pilot, whole-residual donor patches at P_act, 4B, 24 recipients x 8 layers (shared card)" --minutes 120 -- \
  "$PY" /workspace/chief/steer_pilot.py "$S4B" /workspace/chief/all-splits.jsonl /workspace/chief/captures/pilot-4b --device cuda:0 --n-recipients 24 --layers 4,8,12,16,20,24,28,32 --max-new-tokens 48 > /workspace/chief/steer-pilot-4b.log 2>&1 < /dev/null
status "end steer-pilot-4b rc=$? $(grep -E '"event": "done"' /workspace/chief/steer-pilot-4b.log | tail -1 | cut -c1-120)"
