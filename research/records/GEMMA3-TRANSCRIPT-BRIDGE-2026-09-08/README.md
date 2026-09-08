# Gemma transcript lenses and dictionary bridge

Status: source and small tests in progress; no model run or full dictionary readout has started.

## Finding, technique, implementation

**Finding so far.** The existing agentic corpus builder would drop turns longer than its cap,
repeat earlier history in the estimator, and reconstruct generated token IDs from decoded text.
Those are properties of the data preparation, not findings about Gemma. The old prose lens does
not answer the transcript-distribution or above-window questions.

**Technique.** Freeze the successful forward inputs separately from emitted tokens. Construct
bounded contexts from those IDs, count each newly introduced eligible position once, and obtain
residual pairs from each checkpoint's own forward on the same frozen contexts. Fit the unchanged
ridge grid at both precisions. Compare corresponding maps relative to their norms. This is native
replay with a declared context/window policy, not a claim that a batched replay reproduces every
live generation activation under its original partitions.

**Implementation.** Worktree `/Users/daniel.tipton/worktrees/gemma-transcript-bridge`, branch
`codex/gemma-transcript-bridge`, based on `3ef0d73`, with observation and lineage changes through
`7d2a18a` merged. Numeric bridge commit `80cc9ad`: 46 small NumPy tests passed with MLX imports
blocked; independent source review found no substantive numerical-core issue. Real acceptance
is still unrun. Work is committed in the home directory for crash recovery.

## Frozen decisions before any new model or matrix run

- Training-split evaluation tasks only, 24 steps, greedy, 200 emitted tokens per turn; no cache
  reuse. The exact cohort and generation provenance must be registered before launch.
- Observation gate: the landed fix at `7d2a18a`, lineage at `a960d80`, and confirming record
  `research/records/GEMMA3-JSPACE-MAP-2026-09-08/DIAGNOSTIC-RERUN.md`. That run shows the corrected
  rendering functions; all three tasks failed. Rendering and the step ceiling changed together,
  so it does not isolate a causal effect of rendering. Its episodes are not training data.
- Context cap 2,048. Record full-length and short contexts, scoring versus input counts, spans,
  and the number of scored positions actually beyond 1,024 tokens. Do not merely stamp a cap.
- Same frozen token IDs and ridge estimator for `gemma3-4b-bf16` and `gemma3-4b`; source `native`.
  Canonical base `google/gemma-3-4b-it`, no additional training, depth 34, width 2560.
- A1 dictionary: **every-layer suite** `resid_post_all/layer_17_width_16k_l0_big`, width 16,384,
  L0 target 120, HF revision `3e94b68be95290aada5b7525cf431d3040f81bb1`. Use the `big` variant
  consistently if later layers are authorized. Only this one dictionary is authorized now.
  This follows the order's amendment, not the exploratory cached deep-dive configuration.
- A1 lens: hosted Jacobian archive, digest
  `a14ab264fc47de9c59b3183a8783a60ca6f1dee2ea635e88223d5b2d67b03193`, matching probe layer 18.
  It was fitted on 128-token prose; it cannot support sliding/global layer claims.
- A1 named feature **0**, top **10** largest signed scores, lower token ID breaks ties.
  Negative control substitutes probe-layer **1** against the same feature/decoder/readout.
  No search for a favorable feature after observing whether that control bites.
- Independent reference: `(W_chunk @ J) @ d_0` in float64 from canonical float32 operands,
  compared with `W @ (J @ d_0)` through the production composed decoder. Complete score vectors
  must meet `abs(error) <= 0.001 + 0.0001 * abs(reference)` and relative L2 error <= 0.0001.
  Report reference norm and maximum absolute error. These bounds are fixed before real scores.
  The wrong-layer vector must fail the same gate and change the top-10 set; otherwise acceptance
  is unmet. Never silently widen the bounds or change the control.
- W is the raw tied unembedding weight. Scores omit final RMS normalization, soft caps, biases,
  and softmax. They are linear scores, not actual token probabilities or causal derivatives.

## Coordinates and labels

The pinned upstream SAELens loader at commit
`ee45e7406165ce267b33d33b7371bbb63ef24db8`,
[`pretrained_sae_loaders.py`](https://github.com/decoderesearch/SAELens/blob/ee45e7406165ce267b33d33b7371bbb63ef24db8/sae_lens/loading/pretrained_sae_loaders.py),
loads Gemma Scope 2 residual `w_dec` unchanged, with `normalize_activations: none` and no
finetuning scaling factor. Stored rows are feature directions; transpose once for D. Do not
renormalize them. Both hooks must be `model.layers.17.output`.

The pinned registry has no Neuronpedia mapping for this all-layer dictionary. Its published
deep-dive medium mapping describes different features and must not be borrowed. Exact all-layer
label availability is being checked; absent verified labels, emit an explicit missing-label field.

## Resources and remaining acceptance

No MLX import is needed for A1. The selected BF16 embedding is 262,208 by 2,560. Float32 W is
2.69 GB, D and JD are about 0.168 GB each. The loader and independent reference add bounded
scratch; loading every hosted map transiently adds about 0.865 GB. Never allocate the 17.2 GB
vocabulary-by-feature transfer. A1's full CPU multiplication requires an R61 window. A measured
small feature batch must establish a throughput projection rather than assuming the order's
150-second estimate applies to column-wise matrix-vector products.

The 12.44 GiB peak of the earlier 128-token fit is a starting datum, not a projection for 2,048.
Calibrate native replay plus scoring masks and all-layer sufficient statistics before announcing
the combined fit window. Include load, solve and serialization in the projection.

A2 waits for the transcript lens and run activations. B and the causal-abstraction programme are
not implemented. Small averaged-Jacobian readings can hide cancellation of strong contextual
sensitivities. Correlation across times with a retained transcript does not identify memory.
