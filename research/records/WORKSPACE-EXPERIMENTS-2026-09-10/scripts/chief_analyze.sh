#!/bin/bash
# Run every analysis for one model's capture on the card's CPU: W-1/W-3/W-4 (workspace_analyze.py),
# W-2 and W-4 primary (workspace_w2.py), and W-3b's aggregate when its pass exists.
# usage: chief_analyze.sh 4b|12b
set -u
M=$1
export PYTHONPATH=/workspace/chief/ws-d:/workspace/chief/ws-d/src CUDA_VISIBLE_DEVICES=""
PY=/workspace/agent-v2-lab/.venv/bin/python
CAP=/workspace/chief/captures/$M; OUT=/workspace/chief/captures/$M/analysis; mkdir -p "$OUT"
if [ "$M" = "4b" ]; then MAPS=/workspace/chief/out/lens4b-f32/exact-maps.npz; SNAP=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767; else MAPS=/workspace/chief/out/lens12b-f32/exact-maps.npz; SNAP=/workspace/.hf_home/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80; fi
CORPUS=/workspace/chief/all-splits.jsonl
LOG=/workspace/chief/analyze-$M.log
{
  echo "=== $(date -u +%H:%MZ) analyze $M"
  "$PY" /workspace/chief/workspace_analyze.py "$CAP" "$OUT" --corpus "$CORPUS" --snapshot "$SNAP" && echo "W1/W3/W4/W5 ok"
  "$PY" /workspace/chief/workspace_w2.py "$CAP" "$MAPS" "$OUT" && echo "W2 ok"
  if [ -s /workspace/chief/captures/$M-w3b/w3b.jsonl ]; then "$PY" /workspace/chief/workspace_w3b_analyze.py /workspace/chief/captures/$M-w3b "$OUT/w3b.json" && echo "W3b ok"; fi
  echo "=== $(date -u +%H:%MZ) done $M"
} > "$LOG" 2>&1
tail -3 "$LOG"
