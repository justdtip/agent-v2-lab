#!/bin/bash
# v3: wait for chunk c2's python to exit -> c3 at width 8 (same speed as 16, 16 GB less) -> merge the chunks.
set -u
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src LLL_BACKEND=torch LLL_DEVICE=cuda:0 HF_HOME=/workspace/.hf_home AGENT_V2_BOX_STATE_DIR=/workspace/box-chief
PY=/workspace/agent-v2-lab/.venv/bin/python; FIT=/workspace/chief/fit_lens_f32.py
CORPUS=/workspace/lens-corpus/prose-gemma3-4b-cuda-bf16.json
S12B=/workspace/.hf_home/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80
LOG=/workspace/chief/overnight.log
status() { echo "$(date -u +%H:%MZ) STATUS $*" | tee -a "$LOG"; }
while pgrep -f "fit_lens_f32.py .*lens12b-f32-c2" > /dev/null; do sleep 30; done
status "v3: c2 finished: $(grep -E '"event": "(fit_done|done)"' /workspace/chief/lens12b-c2.log | tail -1 | cut -c1-120)"
status "start lens12b-c3 (width 8)"
"$PY" -m local_llm_lab.runlock run --seat chief --purpose "chief: 12B float32 lens chunk c3 rows 140-201, dim_batch 8" --minutes 480 -- \
  "$PY" "$FIT" "$S12B" "$CORPUS" /workspace/chief/out/lens12b-f32-c3 --dim-batch 8 --rows-from 140 --rows-to 201 --label lens12b-f32-c3 > /workspace/chief/lens12b-c3.log 2>&1 < /dev/null
status "end lens12b-c3 rc=$? $(grep -E '"event": "(fit_done|done)"' /workspace/chief/lens12b-c3.log | tail -1 | cut -c1-120)"
status "merge chunks"
"$PY" /workspace/chief/merge_chunks.py /workspace/chief/out/lens12b-f32 /workspace/chief/out/lens12b-f32-c1 /workspace/chief/out/lens12b-f32-c2 /workspace/chief/out/lens12b-f32-c3 > /workspace/chief/merge12b.log 2>&1
status "merge rc=$? $(tail -1 /workspace/chief/merge12b.log | cut -c1-120)"
status "v3 chain finished"
