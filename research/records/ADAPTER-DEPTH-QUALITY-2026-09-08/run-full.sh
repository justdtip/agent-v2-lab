#!/bin/zsh
# Issue 88 arm 1: the winning checkpoint (800 rows) across the whole 180-task test split, so the
# 19-task answer has a wider reading behind it. A separate directory name so the 19-task file
# stays: stage_eval's stem is the adapter directory's name.
set -u
cd "/Users/daniel.tipton/Desktop/An app"
PY=.venv/bin/python
eval "$($PY -m local_llm_lab.runlock announce --seat deputy \
  --purpose '88 arm 1: checkpoint 800 across the full 180-task test split' --minutes 90)"
trap '$PY -m local_llm_lab.runlock end' EXIT INT TERM
echo "window open: $AGENT_V2_BOX_WINDOW"
echo "=== full split starting $(date -u +%H:%M:%SZ) ==="
.venv/bin/agent-pipeline --config configs/agent_v2e_qwen35_4b_top8.yaml eval \
  --adapter outputs/agent-v2e-qwen35-4b-top8/checkpoints/ckpt-0000800-full --split test --limit 180 \
  || echo "=== exited non-zero ==="
echo "=== full split done $(date -u +%H:%M:%SZ) ==="
