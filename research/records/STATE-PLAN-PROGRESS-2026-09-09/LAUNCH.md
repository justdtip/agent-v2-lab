# The exact commands for the slot, written before it, so none is composed at 06:50Z

Four launches, in this order, each one a line to run rather than a line to assemble. Composing a
command at the moment of running it is how a `cd` gets forgotten and a process ends up in the shared
checkout, and how an environment variable that should be set is not.

**Two constraints that apply to every line below** (Chief, 02:45Z):

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.** It changes the allocator's segment
  strategy and not any kernel, so it is not expected to change a number; it is what turns most
  "allocation failed, freed cache, retried" events into non-events on a card at its ceiling. The
  capture run records it in `run.json` and in an `allocator` event, because *not expected to* is
  precisely why it belongs in the manifest — if a later pass disagrees with this one, an environment
  variable nobody wrote down is the answer nobody finds.
- **`cd` into the worktree first.** The card's ssh login shell starts in `/workspace/agent-v2-lab`,
  the shared checkout, and a process sitting there trips the pull gate. Everything below runs from
  `/workspace/wsd/ws-d`, which is this branch's worktree, with `PYTHONPATH` pointing at its `src`.

```bash
# 0. shared prelude for every launch
cd /workspace/wsd/ws-d
export PYTHONPATH=/workspace/wsd/ws-d/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VENV=/workspace/agent-v2-lab/.venv/bin/python
SNAP4B=/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767/
SNAP12B=/workspace/.hf_home/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80/
LENS=/workspace/lens-corpus/prose-gemma3-4b-cuda-bf16.json
CORPUS=/workspace/rendered-corpus/agent_v2e-gemma3-4b
```

## 1. The repeat gate — alone, it measures time

```bash
$VENV -m local_llm_lab.runlock run --seat d-cro \
  --purpose "gate 1 executed: float32 no-op boundary, then two independent exact fits per layer" \
  --minutes 20 -- $VENV research/records/WSD-FD-CALIBRATION-2026-09-10/repeat_gate.py \
  "$SNAP4B" "$LENS" /workspace/wsd/out/golden-f32d /workspace/wsd/out/repeat-gate
```

## 2. The ladder re-run — alone, it measures time

```bash
$VENV -m local_llm_lab.runlock run --seat d-cro \
  --purpose "the width-1 ladder re-run for the unreduced per-cell archive" --minutes 20 -- \
  $VENV research/records/WSD-FD-CALIBRATION-2026-09-10/ladder.py \
  "$SNAP4B" "$LENS" /workspace/wsd/out/ladder-archive --width 1
```

**Message the Chief when 1 and 2 end.** Their 12B capture starts on a GO file they create, beside
the 4B pass below.

## 3. The 4B capture — beside the Chief's 12B capture, neither measures time

```bash
$VENV -m local_llm_lab.runlock run --seat d-cro \
  --purpose "plan-progress capture, 4B, native bf16, width 1, nothing read" --minutes 180 -- \
  $VENV scripts/state_capture.py --checkpoint "$SNAP4B" \
  --capture-set research/records/STATE-PLAN-PROGRESS-2026-09-09/capture-set.jsonl \
  --corpus "$CORPUS" --out /workspace/captures/gemma3-4b-cuda-bf16 \
  --entry gemma3-4b-cuda-bf16 --decoding teacher-forced
```

## 4. The 12B capture — **only if the headroom rule holds**

The Chief's rule: their measured peak plus this pass's measured first-shard peak must leave at least
9 GiB, so the sum must be ≤ 86 GiB. Both numbers are measured by then — theirs from their progress
log every 100 rows, mine from the `shard` event, which carries the device high-water mark and by the
first shard already includes a full forward at the longest row seen. If the sum exceeds 86, this
waits for their capture to end or runs beside their 12B W-3b later.

```bash
$VENV -m local_llm_lab.runlock run --seat d-cro \
  --purpose "plan-progress capture, 12B, native bf16, width 1, nothing read" --minutes 240 -- \
  $VENV scripts/state_capture.py --checkpoint "$SNAP12B" \
  --capture-set research/records/STATE-PLAN-PROGRESS-2026-09-09/capture-set.jsonl \
  --corpus "$CORPUS" --out /workspace/captures/gemma3-12b-cuda-bf16 \
  --entry gemma3-12b-cuda-bf16 --decoding teacher-forced
```

## What is being captured, and what is not read

7,629 distinct decisions, one position each, every layer, native bf16 at width 1. §16.18 permits the
captures to be made before the seal so long as **none is read**, and none is: these four commands
write shards and manifests and nothing opens them. The seal follows Codex's file-only review.

Both capture passes are resumable and every resumed cell is verified against the checkpoint identity,
the semantic record, the rendered bytes, the request's own enumeration, the shard's bytes and the ids
this run's tokenizer produces — so an interruption costs the shard in flight and not the pass.
