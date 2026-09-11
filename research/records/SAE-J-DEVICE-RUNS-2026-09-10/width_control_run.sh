#!/bin/bash
# The dictionary width/sparsity control, overnight 2026-09-10: for each fetched dictionary, the runner on the 600
# out-of-domain cells (both stages where examples exist, A2 alone for 262k which ships none) after its own dry-run,
# then the in-domain control on the fit prompts. From lab-a2 (aa2daa4). CPU for the runner, GPU for the control forward.
set -u; cd /workspace/chief
export PYTHONPATH=/workspace/chief/lab-a2/src LLL_BACKEND=torch LLL_DEVICE=cuda:0 HF_HOME=/workspace/.hf_home AGENT_V2_BOX_STATE_DIR=/workspace/box-chief-ws OMP_NUM_THREADS=8
PY=/workspace/agent-v2-lab/.venv/bin/python; LOG=/workspace/chief/overnight.log; SCRIPT=/workspace/chief/lab-a2/scripts/device_sae_bridge.py; B=/workspace/chief/bridge
status() { echo "$(date -u +%H:%MZ) STATUS $*" | tee -a "$LOG"; }
S4=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
S12=/workspace/.hf_home/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80
H4=/workspace/.hf_home/hub/models--google--gemma-scope-2-4b-it/snapshots/3e94b68be95290aada5b7525cf431d3040f81bb1
H12=/workspace/.hf_home/hub/models--google--gemma-scope-2-12b-it/snapshots/4c419f1ba0be8b7754d4151d4f26c23b92a9029e
R4=/workspace/.hf_home/dictionaries/google/gemma-scope-2-4b-it; R12=/workspace/.hf_home/dictionaries/google/gemma-scope-2-12b-it
runner() { # model snapshot hubroot receiptroot lens folder stage tag capture
  local m=$1 snap=$2 hub=$3 rec=$4 lens=$5 folder=$6 stage=$7 tag=$8 cap=$9
  $PY /workspace/chief/bridge_configs_width.py $m $folder $B/config-$tag.json > /dev/null || return 1
  local args="--model-snapshot $snap --dictionary-dir $hub/$folder --dictionary-receipt $rec/$folder/DIGEST.json --lens-archive $lens --config $B/config-$tag.json --capture-dir /workspace/chief/captures/$cap --positions /workspace/chief/captures/$cap/positions-sample.json --corpus /workspace/chief/all-splits.jsonl --pairings /workspace/chief/captures/$cap/pairings-sample.json --stage $stage"
  $PY $SCRIPT $args --output-dir $B/dry-$tag --dry-run > $B/dry-$tag.log 2>&1 < /dev/null || { status "skip $tag: dry-run refused: $(tail -n 1 $B/dry-$tag.log | cut -c1-160)"; return 1; }
  status "start width-$tag ($stage, $folder)"
  $PY $SCRIPT $args --output-dir $B/run-$tag > $B/run-$tag.log 2>&1 < /dev/null
  status "end width-$tag rc=$? $(tail -n 1 $B/run-$tag.log | cut -c1-160)"
}
control() { # model snapshot lensdir hubsite recsite suffix layers tag
  local m=$1 snap=$2 lensdir=$3 hub=$4 rec=$5 suffix=$6 layers=$7 tag=$8
  local corpus=/workspace/lens-corpus/prose-gemma3-4b-cuda-bf16.json
  status "start indomain-$tag ($suffix, layers $layers)"
  "$PY" -m local_llm_lab.runlock run --seat chief-ws --purpose "chief: SAE-J width control in-domain, $tag (untimed)" --minutes 120 -- \
    "$PY" /workspace/chief/indomain_control_any.py $B/config-$m.json $layers $B/indomain-$tag.json $snap $corpus $lensdir $hub $rec $suffix > $B/indomain-$tag.log 2>&1 < /dev/null
  status "end indomain-$tag rc=$? $(tail -n 1 $B/indomain-$tag.log | cut -c1-160)"
}
L4=/workspace/chief/out/lens4b-f32/admitted-maps.npz; L12=/workspace/chief/out/lens12b-f32-remerged/admitted-maps.npz
# 4B: 16k big at 18 and 24
runner 4b $S4 $H4 $R4 $L4 resid_post_all/layer_17_width_16k_l0_big both 4b-l18-16k-big 4b
runner 4b $S4 $H4 $R4 $L4 resid_post_all/layer_23_width_16k_l0_big both 4b-l24-16k-big 4b
control 4b $S4 /workspace/chief/out/lens4b-f32 $H4/resid_post_all $R4/resid_post_all width_16k_l0_big 18,24 4b-16k-big
# 4B: the 65k ladder at 18 (resid_post site)
for l0 in small medium big; do
  runner 4b $S4 $H4 $R4 $L4 resid_post/layer_17_width_65k_l0_$l0 both 4b-l18-65k-$l0 4b
  control 4b $S4 /workspace/chief/out/lens4b-f32 $H4/resid_post $R4/resid_post width_65k_l0_$l0 18 4b-65k-$l0
done
# 4B: 262k small at 18 and 24 (no examples: A2 alone)
runner 4b $S4 $H4 $R4 $L4 resid_post_all/layer_17_width_262k_l0_small a2 4b-l18-262k-small 4b
runner 4b $S4 $H4 $R4 $L4 resid_post_all/layer_23_width_262k_l0_small a2 4b-l24-262k-small 4b
control 4b $S4 /workspace/chief/out/lens4b-f32 $H4/resid_post_all $R4/resid_post_all width_262k_l0_small 18,24 4b-262k-small
# 12B: 16k big and 262k small at 24 and 47
runner 12b $S12 $H12 $R12 $L12 resid_post_all/layer_23_width_16k_l0_big both 12b-l24-16k-big 12b
runner 12b $S12 $H12 $R12 $L12 resid_post_all/layer_46_width_16k_l0_big both 12b-l47-16k-big 12b
control 12b $S12 /workspace/chief/out/lens12b-f32-remerged $H12/resid_post_all $R12/resid_post_all width_16k_l0_big 24,47 12b-16k-big
runner 12b $S12 $H12 $R12 $L12 resid_post_all/layer_23_width_262k_l0_small a2 12b-l24-262k-small 12b
runner 12b $S12 $H12 $R12 $L12 resid_post_all/layer_46_width_262k_l0_small a2 12b-l47-262k-small 12b
control 12b $S12 /workspace/chief/out/lens12b-f32-remerged $H12/resid_post_all $R12/resid_post_all width_262k_l0_small 24,47 12b-262k-small
status "end width-control-all"
