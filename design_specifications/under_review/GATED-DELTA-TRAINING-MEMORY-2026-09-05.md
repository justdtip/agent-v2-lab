# B4 cannot train on this machine as the library stands — diagnosis and a proposed slice

**Deputy → Chief and Director, 2026-09-05 evening. Ruling requested before dispatch.**

## What happened

Three live B4 attempts on the Qwen3.5-4B, all Deputy-run on the Director's acceptance of
condition 6. Attempts 1 and 2 died on our own code before iteration one (unused test split
over the ceiling; mlx-lm 0.31.3 dataset protocol) — fixed and committed (`9d5828c`, #47).
Attempt 3 passed the loader, resolved 12 LoRA targets (32.5M trainable), completed the first
validation over 24 batches of two rows up to ~2,400 tokens (val loss 0.7207), and died at the
first optimiser step: `[METAL] Command buffer execution failed: Insufficient Memory`.
`health.json`: status error, verdict incomplete (the #35 amendment on its first exercise).

## Measurements (real train loop, batch 1, one row, one step, checkpointing on)

| adapted layers | row tokens | result | peak |
| --- | --- | --- | --- |
| 32 | 495 | OK | 11.38 GB |
| 32 | 997 | OOM | 19.17 GB |
| 8 | 997 | OOM | 19.69 GB |
| 16 | 997 | OOM | 21.70 GB |

Device: M4 Pro, 24 GiB; Metal recommended working set 17.8 GiB, max buffer 13.3 GiB. The
registry budget for `qwen35-4b` is 22 GiB — above what the device grants; the preflight's
`within_budget` (3.85 GiB, model plus one forward) was judged against the wrong number.
Training split: median row 824 tokens, 95th percentile 1,592, max 2,257 (with completion).
Batch size and adapter depth do not change the outcome; the 3B peaked at 9.4 GB at batch 2.

## Mechanism (read in the installed library)

`GatedDeltaNet.__call__` calls `gated_delta_update(..., use_kernel=not self.training)`
(`mlx_lm/models/qwen3_5.py`). The Metal kernel (`mlx_lm/models/gated_delta.py`,
`gated_delta_kernel`, a raw `mx.fast.metal_kernel`) has no gradient, so in training mode the
library runs `gated_delta_ops`: a Python loop `for t in range(T)` of `_gated_delta_step_ops`,
each step producing a new `(B, Hv, Dv, Dk)` float32 state (2 MiB for the 4B) and several
state-sized temporaries, all retained in the autograd graph for the backward. Memory is
linear in T per layer; `grad_checkpoint` wraps the whole `DecoderLayer` (one class for both
hybrid layer types, so all 32 are covered) but the peak is one layer's unrolled recurrence.
The forward fits because inference takes the kernel. mlx-lm 0.31.3 is the latest release on
PyPI (mlx 0.32.2): no upstream fix exists to pin.

## Options

1. **Chunked, checkpointed recurrence for training (recommended).** A training-time
   implementation of `gated_delta_ops` that processes the sequence in chunks (e.g. 64 tokens)
   under `mx.checkpoint`, so the graph holds only chunk-boundary states and recomputes each
   chunk's steps during backward. Same ops, same order, same numbers as the reference loop;
   memory O(T/chunk × state + chunk × step temporaries) per layer — about 30 MiB of boundary
   states plus one chunk's graph at T = 1,000, against ~19 GB today. Installed by our code
   only during training (the `use_kernel=False` path), leaving inference on the kernel.
2. Cap `max_seq_length` near 700 with prompt truncation — discards most rows and the
   state-carrying context the design is about; not the pre-registered recipe. Rejected.
3. Defer B4; run D3 (3B + D data) first — D data is absent (#17). Does not unblock the 4B.
4. A larger machine — not available.

## Proposed slice (option 1), for the Chief's ruling and SPEC-001 §8-style amendment

- Files: new `src/local_llm_lab/training/gated_delta_chunked.py` (or under `arch.py`'s
  package), an installer that swaps `mlx_lm.models.gated_delta.gated_delta_ops` (or the model's
  `gated_delta_update` in training mode) for the chunked function for the duration of
  `stage_train`, with `train.gated_delta_chunk: 64` in the arm config (recorded in
  provenance and `lora.yaml`); tests in `tests/test_gated_delta_chunked.py`.
- Exactness (R31: drive the library's real functions): forward equals `gated_delta_ops` bit
  for bit on random inputs for both gating shapes, with and without a mask, across chunk sizes
  that do and do not divide T; gradients equal via `mx.grad` on a small case; the final state
  equal. The real `GatedDeltaNet` in training mode produces the same output with and without
  the swap on a small random config.
- Memory: one bounded probe on the lane after the tests are green — one step at 997, 1,592
  and 2,287 tokens at batch 1, then one at batch 2 — reporting peaks; success = the longest
  training row steps under the device's working set with headroom.
- Registry: `qwen35-4b` `memory.budget_gib` corrected to the device's recommended working set
  (17.8), and the preflight's memory block gains the recurrence's training footprint estimate
  (state bytes × T for the longest row) so the next arm cannot pass preflight and fail at step
  one.
- Invariants: no change to inference; no change to the recipe (batch 2, accumulation 2, 2688);
  fakes-only unit tests plus one lane probe; red-first (the reference loop's memory at T = 997
  is the red).

## Rulings requested

(a) Approve option 1 as a slice under SPEC-001 (training-time backbone patch), with the
Chief's chain. (b) Correct the registry budget to the device working set. (c) Whether the
chunk size is an arm-config field (recorded) or a fixed constant. (d) Whether the memory
estimate becomes a preflight gate (R15 condition 1) for every arm.

Until ruled, the lane is free and B4 is blocked at the framework level, not by the recipe.
