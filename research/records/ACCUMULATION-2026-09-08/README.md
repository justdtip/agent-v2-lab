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

---

## Attribution, appended 2026-09-08: the residual is not row length, and no model is needed to say so

The Chief pointed at data already on disk: the trainer's own stdout in
`outputs/agent-v2e-qwen35-4b/train.log` reports `Peak mem` every ten iterations, cumulative. Arm
A's running peak rose in **four discrete steps and nowhere else**, across 780 iterations:

| iteration | peak | step |
| ---: | ---: | ---: |
| 10 | 8.1630 GiB | — |
| 40 | 9.3281 GiB | +1.1651 |
| 110 | 9.5274 GiB | +0.1993 |
| 310 | 9.6047 GiB | +0.0773 |
| 730 | 9.7267 GiB | +0.1220 |

Nothing at the validation pass at iteration 400.

### The model-free half, which settles the total

`iterate_batches` truncates every sequence to `max_seq_length` and pads to
`1 + 32·ceil(len/32)`, capped at the same, so **2,688 tokens is the widest batch the trainer can
build**. The dataset attains it: tokenised with the model's own tokenizer, the 6,648 training rows
run 80 to 2,764 tokens, median 1,193, and **14 rows sit at or over the cap**, giving a largest
padded width of exactly 2,688 across 77 distinct widths.

This evening's arm measured that worst case directly: **9.5050 GiB at 2,688 tokens, accumulation
4**, the same recipe. Arm A reached **9.7267**.

**A measurement against a measurement, with no model in between: arm A exceeds the widest row the
trainer can build, at the same accumulation, by 0.2217 GiB. No row length can supply that.**

### The inversion, which attributes the individual steps

Taking the peak's shape in the row width from the landed calibration — the floor's slope plus the
measured attention term — and anchoring it on this evening's own point so the accumulation cost
and every fixed overhead cancel, each step implies a row width:

| iteration | peak | implied padded width | possible under the 2,688 cap |
| ---: | ---: | ---: | --- |
| 10 | 8.1630 | 2,157 | yes |
| 40 | 9.3281 | 2,619 | yes |
| 110 | 9.5274 | 2,697 | **no** |
| 310 | 9.6047 | 2,727 | **no** |
| 730 | 9.7267 | 2,774 | **no** |

So the first two steps are new longest rows and **the last three are not**. This half depends on
the shape being right; the model-free comparison above does not, and the two agree on the total.

### What that leaves

**The allocator hypothesis is confirmed on the record rather than left as the last candidate**, and
it is sharper than "long runs use more". The peak grows in rare discrete steps at iterations 110,
310 and 730, long after the widest row has been seen, on a workload whose batch widths take 77
distinct values. That is a pool acquiring block sizes it cannot reuse, not a run steadily consuming
more.

Two things it is not. It is not the validation pass, which left no step at 400. And the final
step's 0.1220 GiB is close to one gradient tree (0.1209), but so is a two-bucket width change at
2.114 MiB per token, so that coincidence carries no weight and is recorded only to say it was
noticed and discarded.

### The row order is not reproducible, which is why this is an inversion

The Chief's suggested test — read the row lengths at iterations 40, 110, 310 and 730 — cannot be
run. `iterate_batches` sorts by length, chunks into batches, then permutes the batch order with
`np.random.permutation`, and the trainer's main loop calls it **without a seed**
(`trainer.py:273`), so the permutation depends on numpy's global state at that moment rather than
on the run's declared seed. The order is not deterministic from the config, and the peaks had to
be attributed by their size rather than by their position.
