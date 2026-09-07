# Source inspection before lens fitting

The three training-split JSONs contain 72 trajectories and600 steps (200/201/199). Their summary fields declare split, model, keep_last, adapter and generator metadata; source-inspection.json preserves exact task listings and hashes. No fitting weights were loaded.

ArchitectureView.embed and run_block promote to float32. residuals captures post-block residuals and leaves the final norm out, as required for regression. ArchitectureView.tail includes the norm and cannot be used unchanged for this Jacobian. jacobian_vector_product accepts a view-like object exposing residuals and tail, so a small pre-norm-tail adapter can reuse the existing finite-difference step rule without changing the probe module. Its epsilon is derived from the entire primal tensor; the cached/full-sequence self-check must deliberately use identical perturbation sizes rather than obtaining different epsilons from tensors of different lengths.

response_agreement explicitly requires the same source/target averaging as the fitted map. A mean Jacobian must therefore be validated against the mean of held-out derivative responses under the same span weighting, not independently against every local Jacobian. All32 random directions can be fixed once and used on each sampled context before averaging; the record must state this. Only layer_verdict's map failure can request refitting.

The installed native caches retain batch dimensions in their tensor state, while offsets and mask metadata have different representations. A batched Jacobian tail must copy/broadcast both cache state and metadata and prove that sources remain unmodified. Merely repeating K/V does not suffice for the recurrent blocks. Full-cache cloning from the amended history patch should be reused after it lands if its interface remains suitable.

The original pilot atlas aggregates token rows. Its exact identity output is required as a compatibility check, but the new scientific profiles must carry separate episode-level counts and rates, with matched final-distribution base rates. The two outputs need explicit scope labels.

Ridge scaling remains unresolved: literal3.2 adds the stated lambda directly to raw XTX; the unrun prototype multiplies it by n. No regression solve has been implemented pending the Director's ruling. No final scientific reading, cached-Jacobian equivalence, runtime-cost claim or checkpoint acceptance is asserted by these source notes.
