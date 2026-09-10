#!/bin/bash
# Steering pilot re-run. The 06:28Z attempt died at model load (CUDA OOM: c3 still held 62 GiB beside W-3b's 23 GiB;
# the driver had gated on W-3b's start instead of c3's end). This one is gated on /workspace/chief/GO-PILOT, which the
# Chief touches on the D-CRO's message (phases are entered on messages, never on a clock), and it refuses to launch
# until the card shows at least 30 GiB free, checking every 30 s. Untimed: it may share the card with an untimed job.
set -u
cd /workspace/chief
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cuda:0 HF_HOME=/workspace/.hf_home AGENT_V2_BOX_STATE_DIR=/workspace/box-chief-pilot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/box-chief-pilot
PY=/workspace/agent-v2-lab/.venv/bin/python
S4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
LOG=/workspace/chief/overnight.log
OUT=/workspace/chief/captures/pilot-4b
status() { echo "$(date -u +%H:%MZ) STATUS $*" | tee -a "$LOG"; }
until [ -e /workspace/chief/GO-PILOT ]; do sleep 30; done
status "GO-PILOT seen; pilot waits for >= 30 GiB free on the card"
while :; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0); FREE=${FREE:-0}
  [ "$FREE" -ge 30720 ] && break
  sleep 30
done
[ -e "$OUT" ] && mv "$OUT" "$OUT.superseded-$(date -u +%H%MZ)"
status "start steer-pilot-4b (re-run, untimed; card free ${FREE} MiB)"
"$PY" -m local_llm_lab.runlock run --seat chief-pilot --purpose "chief: steering pilot re-run, whole-residual donor patches at P_act, 4B, 24 recipients x 8 layers (shared card, untimed)" --minutes 240 -- \
  "$PY" /workspace/chief/steer_pilot.py "$S4B" /workspace/chief/all-splits.jsonl "$OUT" --device cuda:0 --n-recipients 24 --layers 4,8,12,16,20,24,28,32 --max-new-tokens 48 > /workspace/chief/steer-pilot-4b.log 2>&1 < /dev/null
RC=$?
status "end steer-pilot-4b rc=$RC $(grep -E '"event": "done"' /workspace/chief/steer-pilot-4b.log | tail -1 | cut -c1-120)"
