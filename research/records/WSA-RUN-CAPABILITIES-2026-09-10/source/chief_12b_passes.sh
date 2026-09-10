#!/bin/bash
# The 12B workspace passes: wait for chunk c3 and the merge, then capture (all decisions + the 300-sample) and W-3b.
set -u
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cuda:0 HF_HOME=/workspace/.hf_home AGENT_V2_BOX_STATE_DIR=/workspace/box-chief-ws
PY=/workspace/agent-v2-lab/.venv/bin/python
S12B=/workspace/.hf_home/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80
MAPS=/workspace/chief/out/lens12b-f32/exact-maps.npz
LOG=/workspace/chief/overnight.log
status() { echo "$(date -u +%H:%MZ) STATUS $*" | tee -a "$LOG"; }
until [ -s "$MAPS" ] && grep -q "merge rc=0" "$LOG" 2>/dev/null; do sleep 60; done
while pgrep -f "fit_lens_f32.py" > /dev/null; do sleep 30; done
sleep 30
status "start ws-capture-12b (merged lens $(sha256sum $MAPS | cut -c1-12))"
"$PY" -m local_llm_lab.runlock run --seat chief-ws --purpose "chief: workspace capture 12B, all decisions + 300-sample, float32 width 1 (shared card)" --minutes 300 -- \
  "$PY" /workspace/chief/workspace_capture.py "$S12B" /workspace/chief/all-splits.jsonl "$MAPS" /workspace/chief/captures/12b --model 12b --device cuda:0 > /workspace/chief/ws-capture-12b.log 2>&1 < /dev/null
status "end ws-capture-12b rc=$? $(grep -E '"event": "done"' /workspace/chief/ws-capture-12b.log | tail -1 | cut -c1-120)"
status "start ws-w3b-12b"
"$PY" -m local_llm_lab.runlock run --seat chief-ws --purpose "chief: W-3b masked-carrier arms, 12B, the 300-sample (shared card)" --minutes 180 -- \
  "$PY" /workspace/chief/workspace_w3b.py "$S12B" /workspace/chief/all-splits.jsonl "$MAPS" /workspace/chief/captures/12b /workspace/chief/captures/12b-w3b --device cuda:0 > /workspace/chief/ws-w3b-12b.log 2>&1 < /dev/null
status "end ws-w3b-12b rc=$? $(grep -E '"event": "done"' /workspace/chief/ws-w3b-12b.log | tail -1 | cut -c1-120)"
status "12b passes finished"
