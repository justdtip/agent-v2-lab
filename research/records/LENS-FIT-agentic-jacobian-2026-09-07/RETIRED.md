# Qwen Jacobian line retired — findings preserved

Director's Gemma pivot, 8 September 2026. No further Qwen fitting, runtime diagnostics or benchmark
optimization is planned. Prior records and source are historical evidence, not Gemma calibration.

The step-size diagnosis, local-norm plateau sweep and fresh six-comparison self-check succeeded
within their registered scope. Full fitting stopped during its cost benchmark, and produced no
lens artifact. The fit rule and cost proposal in earlier files are superseded for future work.

The transferable findings and reproduction technique are separated from implementation in
../GEMMA3-FITTING-2026-09-08/FINDINGS-AND-TECHNIQUE.md. In particular: the perturbed object determines
the relevant numerical scale; the complete execution/intervention history defines the derivative
reference; and a reference passing at sampled directions is not a completed lens validation.

Recurrence alone is not a proof that Jacobians are ambiguous or unavailable. Fixing history and
state defines a function; unspecified or incorrectly reused state can change it. The successful
corrected Qwen checks are evidence for that narrower conclusion. Neither their coefficient nor
their hardware cost is transferred unchanged to the next architecture.
