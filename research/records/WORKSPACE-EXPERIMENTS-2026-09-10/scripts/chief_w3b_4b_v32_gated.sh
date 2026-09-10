#!/bin/bash
# The 4B W-3b pass repeated with the v3.2 control (the v3.1 pass's random control was degenerate: its pool was the carrier
# spans themselves). Gated on /workspace/chief/GO-W3B-4B-V32, touched by the Chief on the D-CRO's message (phases are entered on
# messages, never on a clock); refuses to run unless workspace_w3b.py is the v3.2 script by digest, and waits for >= 30 GiB free.
# Untimed: it may share the card with an untimed job. The v3.2 analyzer runs on the CPU when the pass ends.
set -u
cd /workspace/chief
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cuda:0 HF_HOME=/workspace/.hf_home AGENT_V2_BOX_STATE_DIR=/workspace/box-chief-ws
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
LOG=/workspace/chief/overnight.log
OUT=/workspace/chief/captures/4b-w3b-v32
V32=0a7869c0d8ed
status() { echo "$(date -u +%H:%MZ) STATUS $*" | tee -a "$LOG"; }
until [ -e /workspace/chief/GO-W3B-4B-V32 ]; do sleep 30; done
D=$(sha256sum /workspace/chief/workspace_w3b.py | cut -c1-12)
if [ "$D" != "$V32" ]; then status "REFUSED ws-w3b-4b-v32: workspace_w3b.py is $D, not the v3.2 script $V32 (promote it first)"; exit 1; fi
status "GO-W3B-4B-V32 seen; W-3b v3.2 4B waits for >= 30 GiB free on the card"
while :; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0); FREE=${FREE:-0}
  [ "$FREE" -ge 30720 ] && break
  sleep 30
done
[ -e "$OUT" ] && mv "$OUT" "$OUT.superseded-$(date -u +%H%MZ)"
status "start ws-w3b-4b-v32 (v3.2 control; card free ${FREE} MiB)"
"$PY" -m local_llm_lab.runlock run --seat chief-ws --purpose "chief: W-3b v3.2 (corrected random control + note control) 4B, the 300-sample (shared card, untimed)" --minutes 90 -- \
  "$PY" /workspace/chief/workspace_w3b.py "$S4B" /workspace/chief/all-splits.jsonl /workspace/chief/out/lens4b-f32/exact-maps.npz /workspace/chief/captures/4b "$OUT" --device cuda:0 > /workspace/chief/ws-w3b-4b-v32.log 2>&1 < /dev/null
RC=$?
status "end ws-w3b-4b-v32 rc=$RC $(grep -E '"event": "done"' /workspace/chief/ws-w3b-4b-v32.log | tail -1 | cut -c1-120)"
if [ "$RC" -eq 0 ]; then
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$PY" /workspace/chief/workspace_w3b_analyze.py "$OUT" /workspace/chief/captures/4b/analysis/w3b-v32.json > /workspace/chief/ws-w3b-4b-v32-analyze.log 2>&1
  status "w3b-v32 analyzer rc=$? $(tail -n 1 /workspace/chief/ws-w3b-4b-v32-analyze.log | cut -c1-160)"
fi
