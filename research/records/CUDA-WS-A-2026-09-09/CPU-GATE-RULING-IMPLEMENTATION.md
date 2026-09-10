# Native-dtype seam gate after the review ruling

Written before the new checkpoint run, following `cd8055d` in the WS-A order.
Calibration02 is retained unchanged as the earlier cross-precision experiment.

## Finding under test

At fixed stored weight values and fixed input IDs, the native-dtype hand-run path
must reproduce every native block output exactly. This separates instrumentation
agreement from the previously measured float32-versus-bf16 arithmetic difference.
It does not choose Q5 or change the public `residuals()` result or signature.

## Material, method and gates

- Same official local bf16 checkpoint and the same frozen `calibration-token-ids.json`.
  Use the shared `hf_text.load_text_causal_lm`; verify its returned audit and hashes.
- CPU, eager attention, torch 2.14.0, one thread, deterministic settings; no MLX,
  MPS or CUDA. Fresh process under an owned running window and model lock.
- At 64 and 1,400 input IDs, observe native entry, block arguments and block outputs.
  Keep the native reference unchanged. Run each block through `_block` at native
  dtype; compare every residual site to the native reference. **All errors must be
  exactly zero**, including entry. Missing sites or non-finite values refuse.
- Run the existing promoted block path using the same entry and argument bundles.
  Report `max(abs(loop-native))/max(abs(native))` for each layer as a descriptive
  precision difference, with the existing rotary-rounding evidence beside it.
  It has no acceptance threshold and cannot justify relaxing exactness.
- Controls are global-mask dispatch, hook-site off-by-one, and omitted entry
  transform. They change only the controlled reading/path, never the native
  reference. At 1,400, each must have a maximum per-layer relative difference
  strictly greater than the promoted path's maximum. Record both values, their
  difference and per-layer margins. This literal outside-the-floor test adds no
  chosen multiplier. If a control fails it, stop; do not amplify its perturbation.
- The global-mask control at 64 must remain exactly zero at native dtype. Hook-site
  and entry errors can exist below the window: the dated ruling requires those
  controls at 1,400 and supersedes the earlier blanket short-context restriction.
- Reuse observed native tensors/arguments between arms and discard each loop output
  after comparison where possible. This is frozen-input reuse inside a diagnostic,
  not generation cache reuse. Hooks must be removed on every exit.

## Resources and scope

Retain the conservative **9.8 GiB** projection and explicit **10.656005859375 GiB**
cap. The previous complete run peaked at **7.2386322021484375 GiB** in 1,785.2
seconds. The new comparison schedule needs its own measured peak and duration;
those earlier numbers are its basis, not a guarantee. Persist scalar results
atomically at phase boundaries and check the lifetime process peak before
continuing. Native text capture avoids materializing unnecessary vocabulary logits.

Gate 1 checks discovery against the loaded config. The criteria above decide gate
2. Gates 3–4 and the graph-once estimator remain unexecuted until their prerequisites
pass. The unresolved capture-dtype choice is not part of this diagnostic change.
