#!/bin/bash
# After the width driver exits: (a) the suite-matched 16k ladder at 4B block 17 (resid_post site) — runner on the 600 cells
# and the in-domain control; (b) behavioural tests (substitution v2 and complete call) on the lowest-margin sample at
# layer 18 with the 65k big and 262k small dictionaries, so width is read on behaviour, not only on shares.
set -u; cd /workspace/chief
export PYTHONPATH=/workspace/chief/lab-a2/src LLL_BACKEND=torch LLL_DEVICE=cuda:0 HF_HOME=/workspace/.hf_home AGENT_V2_BOX_STATE_DIR=/workspace/box-chief-ws OMP_NUM_THREADS=6
PY=/workspace/agent-v2-lab/.venv/bin/python; LOG=/workspace/chief/overnight.log; SCRIPT=/workspace/chief/lab-a2/scripts/device_sae_bridge.py; B=/workspace/chief/bridge
status() { echo "$(date -u +%H:%MZ) STATUS $*" | tee -a "$LOG"; }
while pgrep -f "bash /workspace/chief/width_control_run.sh" > /dev/null; do sleep 60; done
S4=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767
H4=/workspace/.hf_home/hub/models--google--gemma-scope-2-4b-it/snapshots/3e94b68be95290aada5b7525cf431d3040f81bb1; R4=/workspace/.hf_home/dictionaries/google/gemma-scope-2-4b-it; L4=/workspace/chief/out/lens4b-f32/admitted-maps.npz
for l0 in small medium big; do
  folder=resid_post/layer_17_width_16k_l0_$l0; tag=4b-l18-rp16k-$l0
  $PY /workspace/chief/bridge_configs_width.py 4b $folder $B/config-$tag.json > /dev/null
  args="--model-snapshot $S4 --dictionary-dir $H4/$folder --dictionary-receipt $R4/$folder/DIGEST.json --lens-archive $L4 --config $B/config-$tag.json --capture-dir /workspace/chief/captures/4b --positions /workspace/chief/captures/4b/positions-sample.json --corpus /workspace/chief/all-splits.jsonl --pairings /workspace/chief/captures/4b/pairings-sample.json --stage both"
  $PY $SCRIPT $args --output-dir $B/dry-$tag --dry-run > $B/dry-$tag.log 2>&1 < /dev/null || { status "skip $tag: $(tail -n 1 $B/dry-$tag.log | cut -c1-160)"; continue; }
  status "start width-$tag (both, $folder)"; $PY $SCRIPT $args --output-dir $B/run-$tag > $B/run-$tag.log 2>&1 < /dev/null; status "end width-$tag rc=$? $(tail -n 1 $B/run-$tag.log | cut -c1-160)"
  status "start indomain-4b-rp16k-$l0"; $PY /workspace/chief/indomain_control_any.py $B/config-4b.json 18 $B/indomain-4b-rp16k-$l0.json $S4 /workspace/lens-corpus/prose-gemma3-4b-cuda-bf16.json /workspace/chief/out/lens4b-f32 $H4/resid_post $R4/resid_post width_16k_l0_$l0 > $B/indomain-4b-rp16k-$l0.log 2>&1 < /dev/null; status "end indomain-4b-rp16k-$l0 rc=$? $(tail -n 1 $B/indomain-4b-rp16k-$l0.log | cut -c1-160)"
done
for spec in "resid_post width_65k_l0_big 65k-big" "resid_post_all width_262k_l0_small 262k-small" "resid_post width_65k_l0_small 65k-small"; do
  set -- $spec; site=$1; suffix=$2; tag=$3
  status "start subst2-4b-$tag (lowest-margin sample, layer 18, $site/$suffix)"
  DICT_SITE=$site DICT_SUFFIX=$suffix SAMPLES=lowest_margin_32 $PY /workspace/chief/substitution_test_v2.py $B/config-4b.json 18 $B/subst2-4b-$tag.json > $B/subst2-4b-$tag.log 2>&1 < /dev/null; status "end subst2-4b-$tag rc=$? $(tail -n 1 $B/subst2-4b-$tag.log | cut -c1-300)"
  status "start calltest-4b-$tag (lowest-margin sample, layer 18, $site/$suffix)"
  DICT_SITE=$site DICT_SUFFIX=$suffix SAMPLES=lowest_margin_32 $PY /workspace/chief/call_test.py $B/config-4b.json 18 $B/subst2-4b.json $B/calltest-4b-$tag.json > $B/calltest-4b-$tag.log 2>&1 < /dev/null; status "end calltest-4b-$tag rc=$? $(tail -n 1 $B/calltest-4b-$tag.log | cut -c1-300)"
done
status "end width-followup-all"
