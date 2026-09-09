# Review: calibrated float32 maps and displaced anchors

**Verdict: retain the measured agreement; amend the repeatability gate, the metric label, and the conditioning claims.** This is an append-only, file-only review of D-CRO's record through `e06512899bf94c1683597889f255c290ff80dba5`, with the WS-A/WS-D orders pinned at `3946277fc4fc842fbe769dc0d9241d00deb4d922`. It makes no change to an estimator, run, checkpoint, lock, or producer record. Date: 2026-09-10 locally, 2026-09-09 UTC.

The calibrated finite differences are now close to their matched autograd maps in the reported **whole-matrix norm**. The transplanted-displacement experiment also finds substantial changes in sampled derivatives through a coherent float32 tail. Those are useful results. Neither an independent exact-repeat pass nor a numerical condition estimate for the entire map follows from this record.

## 1. What the model/instrument did

### Full-map agreement: a supported result with a narrower metric

| Repo layer | Reported relative Frobenius error | Reported cosine | Physical step h | Transposed-control error / measured error |
|---:|---:|---:|---:|---:|
| 1 | 0.0048479863 (0.485%) | 0.9999906704 | 0.0992107773 | 291.708 |
| 33 | 0.0028639612 (0.286%) | 0.9999962220 | 137.9470507813 | 348.023 |

The scope is one held prompt, row 39; sequence length 128; source/target positions 8 and 127; float32; anchor and forward widths both one. The source-position reduction is a mean and the target-position reduction a sum. This is evidence about these reduced matrices, not every unreduced source-to-target derivative, long contexts, every layer, or the earlier laptop maps.

**Basis:** matrix values above are producer-reported in the committed JSON, not independently recalculated from matrix entries. This Git record contains no `exact-maps-*.npz` or `fd-maps-*.npz`. The review verifies the source definition of the metric, the declarations, and the arithmetic of the control ratios. The reports deliberately carry `threshold: null`: scientific agreement is a measurement, not an accuracy gate this audit may invent.

### Fixed-width displacement: sensitivity persists in float32

The new `displacement_control.py` closes the earlier missing experiment: it captures the native width-induced displacement first, then applies that **same displacement tensor** to the model's float32 anchor while keeping the replay width at one. This is distinct from the 12B control, which generates its own anchors separately in each precision.

Medians of absolute *scalar derivative* changes, relative to each scalar's own baseline; 18 direction/cotangent pairs per cell:

| Repo layer | Native, transplanted displacement | Float32, same displacement | Float32, one equal-norm random displacement |
|---:|---:|---:|---:|
| 1 | 0.470689 | 1.035116 | 0.748732 |
| 17 | 0.585164 | 0.696514 | 1.277339 |
| 33 | 0.119927 | 0.125179 | 0.299125 |

**Basis:** recomputed here from all 108 committed scalar rows, checking the stored ratios against `a_base`, `a_plus_delta`, and `a_plus_random`. These numbers demonstrate substantial sensitivity in the computed float32 derivatives for the tested displacements. They contradict an explanation confined to native-bf16 arithmetic. They do not by themselves certify the condition number of a mathematical real-arithmetic Jacobian, or transportability of a corpus-averaged lens.

The optional native-anchor reading is also present. Holding the tail in float32 and switching from its own anchor to the native anchor gives median scalar changes of **0.536735, 0.252133, and 0.048705** at layers 1, 17, and 33. This is an additional descriptive result, not a direct comparison of the native and float32 derivatives: both readings in this contrast use the float32 tail. Cross-precision lens/capture compatibility remains unmeasured.

## 2. Required corrections, with source evidence

### M1 — An independent exact-repeat test was recorded as passed without being compared (high priority)

[`golden_float32.py:215–220`](source/golden_float32.py) calls:

```python
reference = {repo_layer: exact_maps[repo_layer]}
golden.golden_report(..., reference=reference, reproduction=reference, ...)
```

The API contract explicitly defines `reproduction` as an **independent rerun** ([`golden.py`, docstring above line 266](source/library/golden.py)). Reusing the same map makes the zero difference inevitable. It cannot detect a nondeterministic or otherwise different second result.

A separate layer-33 smoke fit exists earlier in the script, but its output is not supplied to this comparison. The layer-1 loop has no second independent exact fit. Consequently the reports' `exact_reproduces_itself: {measured: 0, state: passed}` are not evidence of repeatability for these map pairs.

**Amendment:** record the repeat gate as unexecuted until distinct runs are compared, or compare independently saved smoke and fit outputs where the endpoint and reduction really match. Retain the FD-versus-autograd measurement. This is not evidence that autograd failed to reproduce; that question was not tested by this gate. The API already supports `reproduction=None` and reports it as not run.

### M2 — “Worst over 2,560 columns” is the wrong quantity (high priority)

[`golden.py:124–145`](source/library/golden.py) computes `norm(candidate-reference)/norm(reference)` with no axis on a matrix: the Frobenius norm. At line 279 the `max` is over **layers**, and each report here has one layer. Neither operation takes a maximum over columns.

A compact counterexample explains the consequence. Let the reference be `diag(1000, 1)` and the candidate `diag(1000, 0)`. The relative Frobenius error is about **0.001**, the cosine about **0.9999995**, and the second column is wrong by **100%**. Good aggregate agreement does not establish a per-column bound.

**Amendment:** replace “worst over 2,560 columns” with “relative Frobenius error of the full layer map” in the record and any downstream ruling. Keep ladder projection errors and map errors in separate columns; neither is necessarily larger than the other. If a worst-column claim is wanted, compute per-column errors from the saved arrays, name the orientation and zero-denominator policy, and report absolute errors beside relative ones. The 18 selected ladder scalars are not a representative random sample of all column errors.

The already-queued `J-I`/`FD-I` comparison remains worthwhile because the shared skip identity can dominate a late-layer map. The unexecuted layer-shift control is already disclosed; this audit does not reopen that ruling.

### M3 — The 539× figure divides by only part of the input that was changed (high priority for interpretation)

Both the same-anchor and displacement scripts replace a **whole sequence** of residuals: a tensor with shape `[1, sequence, hidden]`. In [`displacement_control.py:164`](source/displacement_control.py), `base + delta[layer]` moves that entire tensor. The hook inserts all of it. Yet lines 171–172 calculate the displacement norm only from `[0, POSITION]`, where `POSITION=8`. The reported derivative also selects position 8, but attention makes that derivative a function of the other positions' residuals too.

Thus the quoted 0.19% measures one token's movement, not the relative norm of the full input intervention. Dividing the median derivative change by that number is not a condition estimate for the declared intervention. The random displacement is normalized using the **whole tensor**, so its whole-tensor norm matches the native displacement; its selected-token norm is not thereby equal. The current committed progress file contains the 12B float32 run, not the displacement run's six `read` events, so the input-norm scalars are not independently recoverable from this snapshot either.

An elementary example: for `F(p,z)=p*z`, the derivative with respect to `p` is `z`. Moving only `z` changes that derivative even though the measured movement of `p` is exactly zero. Dividing by the movement of `p` would manufacture infinite amplification in this perfectly ordinary function.

**Amendment:** retain the scalar changes in section 1, but withdraw **539×, 245×, 46×, etc. as condition estimates** until a norm over the entire changed anchor and a matching output norm are recorded. Preserve per-position displacements as diagnostics. A selected-position-only displacement arm would answer the different question of local sensitivity to changing that token alone.

Even with the input norm fixed, a finite-displacement ratio of one scalar projection is a directional sensitivity measurement, not a worst-case condition number for the whole Jacobian. Small scalar baselines can produce large relative ratios through projection cancellation. For example, the layer-33 float32 transplanted-displacement ratios have a median of 0.125 but a maximum of **91.90**. That does not prove global ill-conditioning either; it means “the last block is well conditioned” cannot be concluded from the median.

### M4 — One random displacement does not establish “any perturbation” or safe transport

The script draws **one** random displacement per layer and reuses it across all 18 scalar checks. These are 18 readings of one displacement, not 18 independent displacement draws. The random result establishes that the effect was not confined to the one native displacement tested. It does not establish “any perturbation of that size does this.” Nor does the fixed-width comparison isolate backward accumulation from forward-tail arithmetic: both can change with width at a fixed anchor.

**Amendment:** say “the sampled native and random displacements both produced substantial changes.” Before a general transport claim, test the intended map/capture pair under its own reduction, input population, and precision, rather than applying these scalar medians as bounds. The caution about crossing precision boundaries remains justified; the size and downstream readout effect remain unmeasured.

## 3. Technique that transfers to another implementation

Keep three objects separate:

1. **Estimator error at one declared point and schedule:** compare FD to AD with the same anchor, arithmetic, endpoint, physical perturbation and reductions. Report an aggregate matrix norm and any required per-direction errors separately. An independent repeat uses a separately produced array.
2. **Sensitivity of the derivative to moving the point:** at fixed precision and schedule, evaluate the same derivative observable at `X` and `X + ΔX`. `X` is the entire sequence state if that is what the hook replaces. Record the actual inserted displacement after casting as well as the intended displacement.
3. **Change caused by arithmetic/schedule:** hold the input anchor fixed, change only the declared computation schedule, and compare derivatives. This includes forward and backward numerical changes; further controls are needed to attribute either one alone.

For a scalar observable `a(X)=wᵀ J(X) v`, a finite-displacement relative sensitivity is

```text
K(X, ΔX; v, w) = ( |a(X+ΔX)-a(X)| / |a(X)| )
                 / ( ||ΔX|| / ||X|| ).
```

Both norms must cover the same full input space as the intervention. Zero or small `a(X)` needs explicit handling and absolute changes alongside ratios. This `K` describes the specified finite movement and observable. It is not the supremum over small movements that defines a local worst-case condition number. Multiple displacement sizes can establish whether a local linear regime exists; multiple directions assess direction dependence. Neither should be claimed in advance.

For matrix agreement, squared relative Frobenius error is a reference-column-norm-weighted average of squared column-relative errors, wherever those denominators are nonzero. It is therefore entirely compatible with a badly estimated weak column. This is why preserving both global and directional figures is useful without declaring either one universally authoritative.

No new device job is requested by this review. The reporting corrections are file work. Any missing checks stay explicitly unexecuted until their owner runs them under the existing queue.

## 4. What the current implementation establishes, and leaves open

- The 4B same-anchor additive identity recomputes from all 54 raw scalar triples. It is an algebraic consistency check, not an independent validation of the derivatives. At layer 1, 14 of 18 anchor/arithmetic terms oppose; medians of their absolute values must not be added as a decomposition of the median total.
- The 12B controls supply 54 scalar rows per precision at widths 1 and 8, through three sampled layers. Their float32 medians are small; the largest layer-1 arithmetic relative term is about 0.00248. “Every depth” should mean these sampled depths, not all 48 blocks. The 4B width contrast was 1 versus 64, so the two models' contrast magnitudes do not identify a depth effect.
- The 12B memory smoke fits **layer 47, one row, 128 tokens, two selected positions**, with width one. The recorded float32 allocated peak is 44.098 GiB and time 6.02 s. It does not bound an early-layer, all-layer, or long-context retained graph. Larger-scope memory/runtime remain unexecuted measurements.
- Full residual-response archives, a fresh precision-bound no-op check before derivatives, and skip-subtracted map diagnostics are queued in the producer order. This review has not executed or certified them.
- Standing preregistration request 1 remains pending a complete P1–P6 amendment; no seal review is asserted here.

## 5. Reproduction and review readiness

`python3 analyze.py --output analysis.json` uses only the standard library. It verifies all 22 frozen sources against `sources.json`, recomputes 270 scalar rows, audits the repeat argument through the Python syntax tree without importing producer code, checks the matrix/position declarations and control-ratio arithmetic, and runs analytic counterexamples to the unsupported implications. `analysis.json` records each result's basis. Snapshots preserve producer text as evidence; they are not endorsements or executable instructions.

`python3 verify.py` (with `ruff` on PATH) checks reproduction, changed-source refusal, source identity against Git, syntax/lint, and the known-answer cases; its result is `verification.json`. A review-ready commit covers only this directory. No checkpoint was loaded, no native test suite ran, and nothing on the rented card was read or modified.
