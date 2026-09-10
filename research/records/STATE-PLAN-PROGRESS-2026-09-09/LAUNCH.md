# The exact commands for the slot, written before it, so none is composed at 06:50Z

## The handshake, because two seats share one card and neither can see the other's queue

Settled with the Chief at 03:45Z. Four phases, each entered on a message, not on a clock — their
re-run 4B capture ends about fifteen minutes after c3, and my first two jobs need the card alone
because they measure time.

| # | starts on | what runs | roughly |
|---|---|---|---|
| 1 | **Chief messages "card clear"** (their capture's manifest exists) | my repeat gate, then my ladder — **alone**, both measure time | minutes |
| 2 | **I message "timed jobs done"** | my 4B capture **and** their 12B capture together | 8 + ~60 GiB |
| 3 | **I message that my 4B pass has ended** | their 4B W-3b beside their 12B capture | ~25 GiB, ~1 h |
| 4 | **Chief messages that W-3b has ended** | my 12B capture beside their 12B capture, **on the rule** | sum ≤ 86 GiB |

**The rule at phase 4** is measured process memory on both sides, not the allocator's high-water
mark, and I report mine on the first shard. Their W-3b is gated on a GO file of their own so it
cannot start into phase 1 or 2; my phases start only on their message, never on my own estimate of
when their job ends. **A clock is not a signal here** — both of us have already had a job run long
tonight, and the whole point of the handshake is that neither seat guesses.


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

## 0a. First, and it is not a formality: is the worktree the tree these commands describe?

Written commands do not protect against a stale tree. When these four were first written the card's
worktree sat eleven commits behind and contained **neither** `capture.py` **nor**
`state_capture.py` — the commands were right and would have failed at the moment the slot opened,
with a file-not-found and the window burning. The commands say what to run; this says what to run it
*from*, and it is the half that was missing.

```bash
cd /workspace/wsd/ws-d
git fetch -q origin && git checkout -q --detach origin/cuda-ws-d && git log --oneline -1
# then prove the code is actually importable from this tree, not merely present:
PYTHONPATH=/workspace/wsd/ws-d/src /workspace/agent-v2-lab/.venv/bin/python -c \
  "from local_llm_lab.pipeline.state_programme import capture; \
   import pathlib; assert pathlib.Path('scripts/state_capture.py').exists(); print('ready')"
```

If that last line does not print `ready`, nothing below is worth starting.

```bash
# 0b. shared prelude for every launch
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

**On the lease, because it was set carelessly the first time.** `--minutes` is **advisory**: a window
past its expected end is labelled `overdue` in `runlock status` and nothing is killed
(`WINDOW_OVERDUE_SECONDS`, "reported, never enforced"). So an under-lease does not end the pass — it
makes the window a *false promise to the other seat*, which is worse, since the window exists so
another seat can tell whether the card is spoken for. It is leased long here, and `runlock extend`
takes it further if the measured rate says so.

**The rate is measured, not assumed.** The first `shard` event carries its elapsed time, so 256
decisions in, the true seconds-per-decision is known and the remaining time is arithmetic. The
laptop's own anchor is 19 ms per 128-token forward at width 1 on the 4B; these rows are ~1,200 tokens
median with quadratic attention, so the honest range before measuring is wide and the lease reflects
that rather than a point estimate.

```bash
$VENV -m local_llm_lab.runlock run --seat d-cro \
  --purpose "plan-progress capture, 4B, native bf16, width 1, nothing read" --minutes 420 -- \
  $VENV scripts/state_capture.py --checkpoint "$SNAP4B" \
  --capture-set research/records/STATE-PLAN-PROGRESS-2026-09-09/capture-set.jsonl \
  --corpus "$CORPUS" --out /workspace/captures/gemma3-4b-cuda-bf16 \
  --entry gemma3-4b-cuda-bf16 --decoding teacher-forced
```

## 4. The 12B capture — **only if the headroom rule holds**

The Chief's rule: their measured peak plus this pass's measured first-shard peak must leave at least
9 GiB, so the sum must be ≤ 86 GiB — **read as process memory, not as the allocator's high-water
mark**. Their 4B capture died at 03:22Z on a 4.20 GiB allocation with 3.41 GiB free, and the numbers
around it are the reason: c3 showed 62.25 GiB of process memory against 59.72 allocated, and their
own capture 29.30 against 22.12 with 6.52 reserved and unhanded-out. A rule read on `allocated`
plans with a figure 2.5 GiB smaller than the one that decides. The runner therefore reports
`peak_allocated_gib`, `peak_reserved_gib` and the device's own `device_used_gib` / `device_free_gib`
on every shard event; the last two need no proxy and are what the rule is evaluated on.

*This pass has no logits allocation.* `HFLensModel.forward` calls the text module, not the causal LM,
so the decoder stack runs and the vocabulary-sized head does not — there is no spike on a long row of
the kind that ended the Chief's capture. Checked in the source rather than assumed. Both numbers are measured by then — theirs from their progress
log every 100 rows, mine from the `shard` event, which carries the device high-water mark and by the
first shard already includes a full forward at the longest row seen. If the sum exceeds 86, this
waits for their capture to end or runs beside their 12B W-3b later.

```bash
$VENV -m local_llm_lab.runlock run --seat d-cro \
  --purpose "plan-progress capture, 12B, native bf16, width 1, nothing read" --minutes 600 -- \
  $VENV scripts/state_capture.py --checkpoint "$SNAP12B" \
  --capture-set research/records/STATE-PLAN-PROGRESS-2026-09-09/capture-set.jsonl \
  --corpus "$CORPUS" --out /workspace/captures/gemma3-12b-cuda-bf16 \
  --entry gemma3-12b-cuda-bf16 --decoding teacher-forced
```

## What is being captured, and what is not read

7,629 distinct decisions, one position each, every layer, native bf16 at width 1. §16.18 permits the
captures to be made before the seal so long as **none is read**, and none is: these four commands
write shards and manifests and nothing opens them. The seal follows Codex's file-only review.

**If a pass runs past its lease**, extend it rather than letting the window go stale:

```bash
$VENV -m local_llm_lab.runlock extend --minutes N     # the nonce comes from `runlock status`
```

Both capture passes are resumable and every resumed cell is verified against the checkpoint identity,
the semantic record, the rendered bytes, the request's own enumeration, the shard's bytes and the ids
this run's tokenizer produces — so an interruption costs the shard in flight and not the pass.
