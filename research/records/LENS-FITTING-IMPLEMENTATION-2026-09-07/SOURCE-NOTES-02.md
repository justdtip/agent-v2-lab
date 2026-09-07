# Cached-tail construction cautions

Source-only analysis, before any Jacobian fit or checkpoint self-check.

At residual layer L, the tail begins at block index L. The cached state used for a perturbation at position t must describe positions strictly before t at every tail block. Obtaining the current primal through the first L blocks advances those early cache entries; reusing that partly advanced cache to build masks is unsafe because ArchitectureView.masks obtains offsets from the first matching attention/recurrent entry in the full list, which may be an early block outside the tail. Keep an untouched full prefix snapshot for each perturbation, and obtain the layer-L primal through a separate clone. Every perturbation arm, including plus/minus and batch-size comparisons, starts from the same full prefix state.

For a batch of standard basis vectors, finite-difference output row i is the response to input basis i, so it is a COLUMN of the conventional Jacobian. Transpose the stacked responses when writing J. This is the same orientation as a row-oriented regression solve X W = Y followed by J = W.T; a nonsymmetric linear stand-in is required to catch a double transpose.

The existing finite-difference helper scales epsilon by the norm of the entire primal tensor. A source-position perturbation in a full sequence and the same perturbation in a one-token cached tail must use the same explicitly computed epsilon for the self-check. Independently invoking the step rule on differently shaped primals compares different finite differences, even in exact arithmetic.

Cache broadcast needs the batch axis of recurrent state, convolution history and KV tensors, plus lengths/left-padding metadata. Attention offsets are sequence positions and must not be multiplied by batch size. Compare the broadcast and restored-per-call methods on equal inputs and against the full-sequence reference before timing can choose either. A method that is faster but fails the reference check is ineligible.

No choice about ridge scaling or numerical self-check bounds has been made in this note. Its statements are construction invariants to turn into tests, not evidence that the real cached derivative is equivalent.
