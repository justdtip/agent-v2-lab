# Lens validation across architectures: findings, technique, implementation

8 September 2026. Preparation record for the approved Gemma pivot. This record is not a
Gemma fitting result. Qwen fitting and runtime measurement are retired. The Scope dictionary
composition remains later context; no dictionary was downloaded or composed here.

## Findings and their limits

### What the Qwen measurements established

The original finite-difference step used the norm of the whole residual sequence while changing
one token position. On the registered input, that made the perturbation about 23% of the local
residual norm. A sweep using the selected position's norm found a common stable coefficient of
0.03 at three registered depths. A fresh production self-check then passed all six cached/reference
comparisons under the original coordinatewise bounds. Full-basis fitting never began: its
concurrent benchmark exceeded the cost gate. There is no completed Jacobian fit from that attempt.

These are findings about an instrument on a particular checkpoint, input, execution schedule,
precision, sampled directions and depths. They do not certify all inputs or every layer. The
concurrent cost projection is not a standalone runtime measurement. Evidence remains in
../LENS-FIT-agentic-jacobian-2026-09-07/FIT-RETRY-01-RESULT.md and its bound raw records.

The transferable lesson is to define the perturbed object, normalize to that object, and distinguish
finite-step bias from numerical cancellation and changed execution history before reading a lens.
The coefficient 0.03 is not a universal constant and is not carried into Gemma.

### A necessary correction to the architectural claim

Recurrence does not make a Jacobian mathematically undefined. Fix the complete input history,
initial state and computation, and the derivative is defined wherever that computation is
differentiable. Holding a prefix state fixed while perturbing an input that produced it defines a
different intervention from recomputing that state. A reference is ambiguous when these conditions
are unspecified, not simply because the architecture contains recurrence.

The Qwen self-check's success is direct evidence against treating recurrence as an impossibility
result. The narrower observed difficulty was reproducing the intended function and derivative in
this hybrid implementation and its numerical execution. Operator support for autodifferentiation
is also an implementation property, not a theorem that hybrid models cannot have Jacobian lenses.

All-attention models simplify prefix reuse when the retained state, position indices and masks
preserve the intended function. They still need numerical conformance evidence: mathematical
function equality does not imply bit-identical floating-point execution across schedules. Gemma's
rotating caches and two mask regimes make that qualification concrete.

### Different fitting objectives can disagree with neither implementation broken

Our regression minimizes squared error between same-position residual values, with ridge
regularization. The upstream Jacobian reference averages derivatives, including a sum across valid
current and later target positions. Thus a hosted Jacobian lens is an independent comparison
instrument, not a numerical ground truth for a regression map. Agreement is supporting evidence;
a localized disagreement tells us where to investigate but does not by itself identify a defect.

Two exact counterexamples are reproduced by reproduce_methods.py. For y=x^3 at x=-2,-1,1,2,
unregularized value regression has slope 17/5 while the mean derivative is 15/2. For the linear
causal map y1=x1, y2=x1+x2 over all four sign pairs, pooled same-position regression is 1 and the
mean of summed target derivatives is 3/2. Neither example is a checkpoint measurement.

Reference consulted: https://github.com/anthropics/jacobian-lens/blob/main/jlens/fitting.py,
8 September 2026. Its documented estimator averages source-position gradients of summed target
residuals. The hosted run's local fit-config records its command and settings but not an immutable
upstream source revision; a new reference run must pin that revision instead of assuming today's
main is the historical implementation.

## Technique, independent of this repository

1. Define a residual site by the computation that produced it: the output of each complete
   decoder block, before final output normalization. Report normalized depth L/N, and retain the
   integer index only to reproduce the selected model. Identify the preceding block's receptive
   field and other differing operations. Do not infer all information at a site came from that
   block; earlier global layers can already have mixed distant context.
2. Fix the model revision, conversion, precision, tokenizer, rendered bytes, token IDs, sample
   membership and reduction over positions. Record actual context lengths and which selected
   positions have causal history exceeding the local window. A long file or a high maximum-length
   setting alone does not establish that the instrument was exercised beyond the window.
3. Prove the observed forward matches the model's own forward on fixed inputs below and above the
   window. Deliberately break local/global mask dispatch and show the above-window check catches
   it. Preserve the unmodified native final normalization and tied/untied unembedding behavior.
4. Fit the value-regression lens using only its fit rows. For each depth accumulate X^T X, X^T Y
   and target energy in a stated precision; solve (X^T X + alpha*mean(diag(X^T X))*I)W=X^T Y.
   Choose alpha from the fixed grid using the selection rows, and represent the lens as J=W^T.
   Those selection rows are not an independent generalization test.
5. Verify that fit with an independent solver on the same small frozen residual matrices, direct
   reconstruction errors, a nonsymmetric known map (which catches transposition), and replication
   of rows (which tests penalty scaling). These are correctness checks of the same objective.
6. Compare independently fitted instruments on the same held records and native readout. Report
   each layer, the final-layer identity/base rate, the corpus and precision mismatches, and the
   existing preregistered agreement and foreknowledge summaries. Preserve the regression h=1 bias
   warning. Report matrix differences as descriptive comparisons, with no invented equality gate.
7. Repeat the method on a second model with the same semantic definitions. Agreement across two
   sizes is evidence of transfer across those sizes; evidence across families tests a broader
   claim. Neither makes the finding universal to transformers or proves a causal training benefit.

For a derivative reference, additionally declare what is held fixed: the prefix state before the
intervention, perturbed positions and directions, differentiated target positions, final-normalization
boundary and forward partitioning. If using finite differences, sweep a relative step, compare
adjacent halvings and independent execution paths, and freeze the resulting rule before the fit.
If using automatic differentiation, test the actual operators and intervention boundary. The
method transfers; the need for each implementation check depends on the new stack.

## What the port itself is testing

| Assumption exposed by the port | Evidence now | Portable detector |
| --- | --- | --- |
| Embeddings already are the block-0 input | Gemma native entry includes a scale missing from our manual loop; source inspection | Compare the native block-0 input with the instrument's input |
| One causal mask/cache exemplar fits every block | Gemma native path selects local and global masks/caches; source inspection | Compare above the window and deliberately misdispatch masks |
| Dimensions identify a lens's model | Existing record shows a Qwen map accepted against Gemma dimensions | Wrong-model negative control with authenticated model identity |
| Every tokenizer accepts the same protocol roles/suffix | Gemma template differs; generation-prefix and role work is pending | Render exact protocol messages and audit token/span alignment |
| A historical token count is a portable sample definition | Gemma tokenizer preview changes window and held counts | Freeze source bytes and tokenizer; recount before fitting |

The architecture-port numerical gates have not run in this record. The Deputy owns the port;
its source and measured negative control are prerequisites for our real fitting run.

## Implementation and evidence in this worktree

The prose builder, reader and CLI now accept an explicit positive integer window length through
chunk_tokens / --prose-chunk-tokens. The 1,024 default, manifest structure, source hashes,
one-based every-fifth selection rule, source boundaries and tail accounting are preserved.
Nondefault lengths were already refused by the older reader; it continues to fail closed on them.
The revised reader derives coverage, starts, row lengths and tails from the bound manifest value.
A configurable length enables a study; it does not authorize a particular run or certify a window
contrast. Changing length changes the partition and therefore requires a new manifest.

Eight focused tests first failed because the API/CLI parameter did not exist. After the change,
all 70 corpus tests passed with native imports blocked. One initial broader attempt had 63 passes
and one import-availability failure: Transformers asked whether MLX was installed and our guard
raised. The final wrapper reports that package unavailable while rejecting actual native imports.
The frozen old corpora read back through the changed reader with their original counts: agentic
418 fit / 100 held sequences, prose 204 fit / 51 held. This is an archive-compatibility check,
not new measurement of the retired model. Ruff and whitespace verification accompany the handoff.

Reproduce the mathematical checks with Python 3 using reproduce_methods.py; reproduce the focused
suite with the project's Python using verify_cpu.py; run audit_inputs.py --primary <primary-checkout>
to repeat the source/tokenizer/count audit. The audit imports the tokenizer backend and reads cached
text/tokenizer assets, but forbids MLX, MLX-LM and Torch. It neither imports a model nor loads weights.
