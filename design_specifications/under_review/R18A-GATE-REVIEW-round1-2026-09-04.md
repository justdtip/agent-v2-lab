# R18a preflight gate, review round 1, ratified (2026-09-04 19:30)

Reviewer: Claude (Chief). Evidence: issue #23 and `R18A-GATE-REPORT.md`. Diff not re-read.

## Verdict: APPROVED TO COMMIT.

The gate conforms to R18a's letter (native-dtype comparison; derived elementwise floor AND
Frobenius relative ≤ 1e-4; the float32 comparison reported and unable to gate), the schema
bump makes every stale artifact refused, and the production configuration is now tested. The
reviewer's arithmetic (derived floor alone ~630× more permissive against distributed error;
Frobenius bound strictly dominates) is adopted as the reason the conjunction costs nothing.

## Rulings on the disclosed letter-gaps

- **R18a wording amended:** the float32 comparison is recorded as the `fp32_manual_vs_native`
  block; no rename. The clause "recorded in every probe artifact" is satisfied by the probe
  CLIs copying that block from the model's preflight artifact into their own outputs; that is
  a follow-up in the probe-readiness list (A3), not this slice.
- The standing hypothesis in the report's §0 is adopted as the expected shape of a rerun
  failure. Rerun order: **3B first** (trusted view, the control), then Qwen3.5. If the 3B
  passes at 1e-4 and the hybrid fails, the finding is about the hybrid's fused kernels versus
  the stepwise loop, not about the view; the decision then is the Chief's, made from both
  metrics in the artifact.

## Commit

Own commit, referencing #23; Deputy pushes. From that moment training and probes are
fail-closed behind the authorised rerun, as intended.
