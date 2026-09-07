#!/bin/zsh
# Issue 88 arm 1: the three checkpoints through the paired-task evaluation, on the 19 tasks the
# divergence record used. mlx-lm's load_adapters takes a DIRECTORY (adapter_config.json +
# adapters.safetensors), never a bare checkpoint file, so each checkpoint is presented as one:
# outputs/agent-v2e-qwen35-4b-top8/checkpoints/ckpt-<step>/, config copied and weights hard-linked.
# The directory name becomes stage_eval's stem, so the three write to distinct evals/*.json.
# One window for the whole job, ended by a trap on exit whatever happens.
set -u
cd "/Users/daniel.tipton/Desktop/An app"
PY=.venv/bin/python
PIPE=.venv/bin/agent-pipeline
CFG=configs/agent_v2e_qwen35_4b_top8.yaml
D=outputs/agent-v2e-qwen35-4b-top8/checkpoints

eval "$($PY -m local_llm_lab.runlock announce --seat deputy \
  --purpose '88 arm 1 paired eval: top-8 checkpoints 800, 1200, 400 on the divergence 19' \
  --minutes 120)"
trap '$PY -m local_llm_lab.runlock end' EXIT INT TERM
echo "window open: $AGENT_V2_BOX_WINDOW"

for CKPT in 0000800 0001200 0000400; do
  echo "=== ckpt-$CKPT starting $(date -u +%H:%M:%SZ) ==="
  $PIPE --config "$CFG" eval --adapter "$D/ckpt-$CKPT" --split test --limit 19 \
    || echo "=== ckpt-$CKPT exited non-zero ==="
  echo "=== ckpt-$CKPT done $(date -u +%H:%M:%SZ) ==="
done
echo "ALL THREE DONE $(date -u +%H:%M:%SZ)"
