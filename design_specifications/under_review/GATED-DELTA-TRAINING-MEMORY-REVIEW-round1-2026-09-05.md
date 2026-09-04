# Gated-delta recurrence in training (issue #50): ruling R32 and efficiency recommendations (2026-09-05 10:30)

Reviewer: Claude (Chief). Evidence: issue #50's measurements and the Chief's direct read of
`mlx_lm/models/gated_delta.py:127-283` and the call site `qwen3_5.py:183-193`.

## Diagnosis confirmed

`gated_delta_update(..., use_kernel=not self.training)` routes training to `gated_delta_ops`:
`for t in range(T)` over `_gated_delta_step_ops`, each step computing `state * decay`,
`state * k`, `state + k * delta`, `state * q` on a `(B, 32, 128, 128)` float32 state (2 MiB) and
returning a new state, all retained for backward. Memory per layer is linear in T; the
per-`DecoderLayer` checkpoint bounds the peak to one layer's unrolled loop, which is what the
19 GB at 997 tokens is. The Metal kernel has no VJP. Nothing upstream to pin.

## Ruling R32

(a) **Option 1 approved as a SPEC-001 training-time backbone slice, in two stages, both
tested against the library's real `gated_delta_ops` (R31):**

- **Stage 1, now: chunked checkpointing of the reference loop.** `gated_delta_ops_chunked`
  processes the sequence in chunks of `C` tokens under `mx.checkpoint`, keeping only
  chunk-boundary states; identical ops in identical order, so forward and final state match
  the reference bit for bit and gradients match `mx.grad` on small cases. Installed only for
  the duration of `stage_train` (the `use_kernel=False` path); inference untouched. Acceptance:
  the tests in the proposal plus a bounded lane probe at 997, 1,592 and 2,287 tokens at batch
  1, then batch 2, reporting peak memory **and step time**.
- **Stage 2, gated on stage 1's step time: the chunkwise-parallel form.** The per-token loop
  is also slow: at T ≈ 1,000 and 24 linear layers, forward plus backward recompute is on the
  order of 50,000 small state ops per row, and a 400-iteration run may take many hours. The
  standard training algorithm for gated delta rules (Yang et al., the WY/UT-transform chunk
  form used by every DeltaNet trainer) computes within-chunk outputs with matrix products and
  passes states between chunks: same O(T/C) state memory, matmul-bound compute. It is exact
  in exact arithmetic but not bit-identical to the loop, so its tests are tolerance-based
  (float32 relative ≤ 1e-4 on outputs, states and gradients against the reference loop).
  Dispatch it if stage 1's measured step time puts 400 iterations over about two hours.
- **Stage 3, later: a VJP for the Metal kernel** (the backward formulas exist in the
  literature), which is the real fix and upstreamable. Not on the B4 critical path.

(b) **Registry budget corrected.** `memory.budget_gib` for every model becomes
`min(registry value, device recommended working set)` resolved at preflight from
`mx.metal.device_info()`; the registry value is a cap, not a constant. For this machine that
is 17.8 GiB, and the preflight artifact records both numbers.

(c) **Chunk size is an arm-config field** `train.gated_delta_chunk` (default 64), recorded in
`lora.yaml` and provenance; a fixed constant would hide a variable that trades memory for time.

(d) **The training footprint estimate becomes part of R15 condition 1.** Preflight computes,
for the longest training row at the configured batch, the attention/MLP activation estimate
plus the recurrence's training footprint under the configured chunk (boundary states × T/C +
one chunk's step temporaries per layer) and fails when it exceeds the device working set with
10% headroom. An arm can no longer pass preflight and die at step one.

## Efficiency recommendations (ranked; none changes the recipe's numbers)

1. **Validation on the kernel path.** mlx-lm's in-loop `evaluate` runs with the model still
   in training mode, so validation also takes the Python loop. Wrap the trainer's evaluate to
   `model.eval()` for the pass and `model.train()` after: no gradient is needed, the kernel is
   exact for inference, and 24 batches of up to 2.4k tokens every 100 iterations become
   seconds instead of minutes. Zero effect on loss values.
2. **Batch 1 × accumulation 4** for the 4B. The recurrence's footprint is per row, but the
   attention and MLP activations halve; same effective batch, same optimiser steps, same
   recipe. Record as the arm's memory setting, not a recipe change.
3. **Chunk size sweep in the lane probe** (32/64/128): memory falls with C, time rises with
   recompute; pick the largest C that fits with headroom.
4. **Keep the state in float32.** `mamba_ssm_dtype: float32` is the model's own choice; a
   bfloat16 state would halve memory and silently change numerics. Not permitted.
5. **Length bucketing is already in place** (rows sorted by token count); keep
   `steps_per_report` at 10 so the health rules see enough reports before the first eval.
6. **Not viable, for the record:** freezing the DeltaNet LoRA targets does not help, because
   backward must still flow through the recurrence to reach earlier layers' adapters; prompt
   truncation discards the state-carrying context the design exists to test; sequence packing
   is unsafe across rows for a recurrence whose mask freezes rather than resets the state.

## Timeline note for the Director

Stage 1 is a day of implementer work plus one lane probe. If its step time is prohibitive,
stage 2 is several days. A dense model of the same size (Qwen3-4B) would train today with the
3B's memory profile and could carry the "parameters" arm meanwhile; that is a change to the
base-model directive and therefore the Director's call, offered as a fallback, not a
recommendation to abandon Qwen3.5.
