# R32 stage 1, lanes 1 and 2 (issue #51): review round 1, ratified (2026-09-05 12:40)

Reviewer: Claude (Chief). Evidence: the work order, the lane measurement, and direct reads of
`training/gated_delta_chunked.py:92-223` and `preflight.py` (`training_state_bytes`, the
footprint block), plus `mlx_lm/tuner/trainer.py:186,299` and `gated_delta.py:281-283`.

## Lane 1: APPROVED TO COMMIT now.

- The chunked recurrence is the reference loop, chunked: `_gated_delta_step_ops` in the same
  order under `mx.checkpoint`, every differentiated tensor an explicit argument, float32
  state, trailing chunk, key-side repeat inside the differentiated region, masked variant.
  Bit-exact tests against the library's loop across T × chunk × gating × mask; the real
  `GatedDeltaNet` unchanged by the installer; restore on exception.
- The patch point is right: `gated_delta_update` resolves `gated_delta_ops` from its module
  globals only on the `use_kernel=False` branch, so inference keeps the kernel.
- Measurement adopted: the 997-token row that overflowed at 19.17 GB unrolled steps at
  11.86 GB chunked (chunk 64, batch 1).
- Correction accepted: mlx-lm already toggles `model.eval()`/`model.train()` around its
  in-loop `evaluate` (`trainer.py:186,299`), so the validation recommendation in R32 was
  already satisfied by the library; the wrapper stays for restore-on-exception only.

## Lane 2: APPROVED on its machinery; COMMIT SECOND, after calibration and artifact regeneration.

The estimator counts retained states only and sits about 8× under the measured peaks
(12.66 GiB projected "fits" where 19.17 GB overflowed). Each step also retains its
state-sized temporaries (the decayed state, the key product, the updated state, the query
product) and the non-recurrence activations, which the count omits. **Ruling (R32 addendum):**

1. The training footprint is a **calibrated model**: `peak = a × retained_state_bytes(T, chunk,
   batch) + b`, with `a` (temporaries per retained state) and `b` (weights plus non-recurrence
   activations at the batch) fitted as an **upper envelope** over the probe's measured peaks,
   never least squares. Fit points, coefficients and the fit date are recorded in the
   artifact; the fit is re-done whenever the probe gains points; the gate uses the calibrated
   estimate with the 10% headroom.
2. Until the calibrated fit exists, lane 2 must not gate; the schema-3 bump therefore lands
   with the calibration, and the 4B artifact is regenerated on the lane immediately after
   (`preflight --model qwen35-4b --data data/agent_v2b-qwen35-4b --gated-delta-chunk 64`).

## Order and gates (as proposed, with the thresholds made explicit)

1. Commit lane 1. Probe on the lane after the P6 rerun: 1,592 and 2,257 tokens at batch 1,
   chunk 32/64/128, then batch 2 at chunk 64, reporting peak **and clean step time** (no
   validation inside the timed step).
2. **B4 attempt 4 may run** when the longest training row (2,257 tokens with completion)
   steps under the device working set with 10% headroom at the chosen chunk, and the clean
   time per optimiser step (four rows at batch 1 × accumulation 4) projects 400 iterations at
   **three hours or less**. Above that, stage 2 (the chunkwise-parallel form) is dispatched
   before B4.
3. Calibrate, commit lane 2, regenerate the artifact.

## Process ruling R33

A second stray `git stash push`/`pop` on the shared tree. Implementers never stash on the
shared tree; `git show HEAD:path` and worktrees are the tools. Any stash on the shared tree is
reported in the implementation report as an incident, and the Deputy re-verifies every other
lane's edits before proceeding.
