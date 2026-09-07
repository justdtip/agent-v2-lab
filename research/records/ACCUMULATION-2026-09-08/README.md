# Accumulation carries a third of the gap, and the larger part is unexplained (issue 94)

2026-09-08, Deputy Chief of AI Research. Two arms under the model-run lock, announced in the
heartbeat with an R47(b) projection and closed with an end line.

## The question

The landed memory envelope reads **9.6370 GiB** at 2,688 tokens for the chunkwise recurrence. Arm
A's own training run reached **9.7260 GiB**. Every calibration point was measured, per
`_CALIBRATION_PROCEDURE`, as "one row, one optimiser step"; arm A trains at
`grad_accumulation_steps: 4`. Accumulation was the obvious candidate and it was not the only one.

## What was run

Two arms, identical but for accumulation: ten optimiser steps at 2,688 tokens, chunkwise recurrence
at chunk 256, batch 1, gradient checkpointing on, LoRA rank 16 scale 32 on the resolved keys at
full depth, AdamW. Arm 1 does ten micro-batches, arm 4 does forty. Each loads fresh and resets the
peak after a warm-up step, so the load's own peak is excluded.

| arm | peak | of the working set | median step | elapsed |
| ---: | ---: | ---: | ---: | ---: |
| accumulation 1 | **9.3822 GiB** | 0.5283 | 30.42 s | 5.1 min |
| accumulation 4 | **9.5050 GiB** | 0.5352 | 96.78 s | 16.2 min |

**Difference: +0.1228 GiB.**

## Accumulation costs one gradient tree, not two

The LoRA gradient tree is 32,464,896 float32 parameters, **0.1209 GiB**. The measured difference is
0.1228, or **1.02 trees**. So accumulation holds the accumulator and reuses the incoming gradient's
memory rather than holding both. The R47(b) projection used two trees as a ceiling, and it was a
ceiling rather than an estimate — the right shape for a projection.

## The finding is not the one the issue anticipated

| condition | peak | envelope at 2,688 |
| --- | ---: | --- |
| one optimiser step per row — **the calibration procedure's own condition** | 9.3822 | clears by 0.2548 |
| accumulation 4, ten steps | 9.5050 | clears by 0.1320 |
| arm A: accumulation 4, long run | 9.7260 | **under by 0.0890** |

**The envelope is not under-predicting the condition it describes.** It clears the procedure's own
condition by a quarter of a gibibyte. What it fails to clear is arm A's long run.

Of the **0.3438 GiB** between the procedure's condition and arm A:

- **0.1228 is accumulation**, measured here;
- **0.2210 is unattributed** — the larger part, and it belongs to running long rather than to
  accumulating.

The allocator hypothesis is not confirmed by this. It is the only candidate left standing, and it
is recorded as unexplained rather than assumed. Separating it needs a long run at accumulation 1,
which is hours on the box and is not worth them for 0.22 GiB unless something else wants the run.

## Caveats

- Ten steps is not a long run, which is the point of the comparison and also its limit: whatever
  grows with a run's length shows up in neither arm.
- Batch 1 throughout. The term is per sequence and nothing here models batch.
- Synthetic token ids, not the arm's rows, so the row length is exact rather than distributed. Arm
  A's cap is 2,688 and its rows average about 1,100, which is part of why its long-run peak is not
  directly comparable to a fixed 2,688-token step.
- Median step time rises 30.42 s → 96.78 s, close to but under 4×, which is the accumulation ratio.
  Not investigated; it is a throughput observation, not the measurement.

## Files

- `arm1.json`, `arm4.json` — the two arms, with the recipe and the working set recorded.
- `accumulation_probe.as-run.py.txt` — the script, guarded under the issue-89 rule.
- `run-arm4.log` — the second arm's log.
