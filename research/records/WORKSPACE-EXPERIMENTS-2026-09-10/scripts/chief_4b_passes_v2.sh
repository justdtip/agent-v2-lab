#!/bin/bash
# v2 (after the 03:22Z OOM death of the first 4B capture beside chunk c3): the 4B workspace passes again — capture with the
# index written per row and the model's logits computed only at the two positions read (the 4.2 GiB full-vocabulary logits
# of a 4,321-token row was the fatal allocation), then W-3b v3.1 with receipts. Runs from /workspace/chief, expandable segments.
set -u
cd /workspace/chief
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cuda:0 HF_HOME=/workspace/.hf_home AGENT_V2_BOX_STATE_DIR=/workspace/box-chief-ws
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
LOG=/workspace/chief/overnight.log
status() { echo "$(date -u +%H:%MZ) STATUS $*" | tee -a "$LOG"; }
status "start ws-capture-4b (v2.1: index per row, logits at two positions; beside c3, expandable segments)"
"$PY" -m local_llm_lab.runlock run --seat chief-ws --purpose "chief: workspace capture 4B v2.1, all decisions + 300-sample, float32 width 1 (shared card)" --minutes 240 -- \
  "$PY" /workspace/chief/workspace_capture.py "$S4B" /workspace/chief/all-splits.jsonl /workspace/chief/out/lens4b-f32/exact-maps.npz /workspace/chief/captures/4b --model 4b --device cuda:0 > /workspace/chief/ws-capture-4b.log 2>&1 < /dev/null
rc=$?; status "end ws-capture-4b rc=$rc $(grep -E '"event": "done"' /workspace/chief/ws-capture-4b.log | tail -1 | cut -c1-120)"
if [ "$rc" != "0" ]; then status "capture failed; W-3b not started"; exit 1; fi
status "start ws-w3b-4b (v3.1)"
"$PY" -m local_llm_lab.runlock run --seat chief-ws --purpose "chief: W-3b v3.1 masked-carrier arms with receipts, 4B, the 300-sample (shared card)" --minutes 120 -- \
  "$PY" /workspace/chief/workspace_w3b.py "$S4B" /workspace/chief/all-splits.jsonl /workspace/chief/out/lens4b-f32/exact-maps.npz /workspace/chief/captures/4b /workspace/chief/captures/4b-w3b --device cuda:0 > /workspace/chief/ws-w3b-4b.log 2>&1 < /dev/null
status "end ws-w3b-4b rc=$? $(grep -E '"event": "done"' /workspace/chief/ws-w3b-4b.log | tail -1 | cut -c1-120)"
status "4b passes v2 finished"
