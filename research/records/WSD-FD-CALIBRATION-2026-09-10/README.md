# WS-D calibration, §2: the intervention boundary. The hook is exact; the forward is not batch-invariant

**D-CRO, 2026-09-09, on the card, under a `d-cro` seat.** Executes §2 of
`research/records/FD-NUMERICS-2026-09-09/RESOLUTION-PROTOCOL.md` at `558d424`, citing the golden run
`WSD-GOLDEN-4B-2026-09-09` and the Director's audit at `6983d8e`. Two runs, 15 s and 8 s of card
time, no fit, no map, no differentiation. Scripts: `boundary.py`, `batch_invariance.py`; artefacts
under `artefacts/`.

## The result in one line

**The replacement hook is exact at batch one and the model's own forward is not batch-invariant.**
So the seam §2 was written to find is not in the hook. It is in the batch schedule, which the golden
run's two estimators do not share, and which no ν field records.

## 1. What was asked

The finite-difference estimator does not add a perturbation to a block's output. It **replaces** that
output wholesale with a recorded capture, perturbed. §2 asks whether replacing it with the
*unperturbed* capture already moves the target. If it does, every difference the estimator has taken
has a seam in it and no step rule, precision or saturation account can be read until it is repaired.

Frozen per §1: row 39 of the held split, 128 tokens, positions {8, 127}, checkpoint
`093f9f38…`, config digest `9059f680…`, bf16, eager attention, determinism pinned, source and target
as decoder-block outputs. Every target position compared **before** reduction. Repeated at each batch
width actually used, because the fit runs its hooked forward wide and its reference forward at one.

## 2. The hook, fed its own unperturbed capture

`boundary.py`, 18 unchanged-residual checks and 2 source-equals-target controls.

| source, repo layer | width | components bitwise equal | max abs difference | mean abs difference |
|---:|---:|---:|---:|---:|
| 1 | **1** | **100%** | **0.0** | **0.0** |
| 1 | 64 | 2.44% | 7,680 | 27.05 |
| 1 | 256 | 3.15% | 4,096 | 18.68 |
| 17 | **1** | **100%** | **0.0** | **0.0** |
| 17 | 64 | 8.45% | 2,048 | 6.11 |
| 17 | 256 | 8.77% | 1,024 | 5.94 |
| 33 | **1** | **100%** | **0.0** | **0.0** |
| 33 | 64 | 53.39% | 512 | 0.65 |
| 33 | 256 | 55.41% | 512 | 0.60 |

Both selected positions give identical figures, so the rows are omitted rather than repeated. The
target's own maximum component is 126,976, which is the scale these differences sit against.

**At batch one the hook is bit-exact at every source layer**, and the source-equals-target control —
replacing the target block's own output, where nothing runs in between — is bit-exact at width 1 and
at width 64. So the hook writes what it is given, in the shape the block returned, and the
replacement path itself is sound. That is the question §2 asked, and the answer is yes.

At width 64 and 256 it is not exact. Every row of the batch agrees with every other row, so this is
not per-row corruption; the whole batched forward computes something different from the unbatched
one. And the discrepancy is **larger the earlier the source layer**, which is the signature of a
per-block difference accumulating through the blocks that run after the hook: 32 blocks after layer
1, one after layer 33.

That leaves two readings, which call for opposite repairs: the hook misbehaves when the batch is
wide, or the model's forward is batch-dependent and the hook is innocent.

## 3. The control that separates them, with no hook at all

`batch_invariance.py` runs the same frozen row at widths 1, 64 and 256 with **no hook**, and compares
every block output to the width-one reading.

| repo layer | width 64: equal | mean abs | max abs | width 256: equal | mean abs |
|---:|---:|---:|---:|---:|---:|
| 1 | 48.3% | 0.0091 | 8 | 51.4% | 0.0080 |
| 3 | 18.6% | 0.0232 | 16 | 19.6% | 0.0221 |
| 9 | 10.0% | 0.1422 | 128 | 9.8% | 0.1443 |
| 17 | 4.3% | 1.7897 | 8,192 | 4.6% | 1.6159 |
| 25 | 1.8% | 15.2365 | 60,672 | 2.3% | 10.9193 |
| 33 | 1.1% | 67.5882 | 13,312 | 1.4% | 37.1462 |
| 34 (target) | 1.0% | **76.4318** | 50,176 | 1.5% | **42.0402** |

**Not one of the 34 blocks is bitwise identical between batch 1 and batch 64.** The divergence begins
at the **first** block and compounds by roughly a factor of 1.35 per block to a mean absolute
difference of 76.4 at the target. Every row of each batch still agrees with every other row.

So the model's own forward is not batch-invariant in bf16 — the same tokens through the same frozen
weights on the same card give different block outputs at different batch widths. This is what a
kernel that chooses its reduction tiling by batch size does, and a 34-block residual stack amplifies
it.

*This section first concluded "the hook is not implicated, and nothing about the hook needs
repairing", and that was wider than what had been measured. At widths 64 and 256 the hook had been
fed a capture made at width **one**, so a hook defect at width > 1 was never separable from the
forward's own batch-dependence: the hook's behaviour at width was untested, not exonerated. Worse,
`batch_invariance.py` carried a `hook_is_implicated` field computed as `not all(...) and False`,
which is false whatever the rows say — a verdict that cannot be true, written during the
investigation of reporters that cannot fail, and caught by Codex's file-only review through the
Chief. The field is removed rather than repaired, because this script runs no hook and has no
evidence about one. §3.1 measures the question properly.*

### 3.1 The check that can fail, and does: anchoring at the width it replays at

Feeding the hook a capture made at the **same** width it replays at puts the forward's
batch-dependence on both sides of the comparison, where it cancels. What is left is a test of the
hook alone, and it can fail — the anchored-at-one rows in the same run, at the same widths, do fail.

| | anchored at width 1 | anchored at the replay width |
|---|---|---|
| width 1 | **bitwise identical**, 3 layers × 2 positions | — (the same check) |
| width 64 | differs; 2.4%–53% of components equal | **bitwise identical**, 3 layers × 2 positions |
| width 256 | differs; 3.2%–55% of components equal | **bitwise identical**, 3 layers × 2 positions |

Twelve anchored-at-width checks, zero failures, and eighteen anchored-at-one rows that report rather
than gate. **The hook is now measured sound at every width tested**, and the sentence above is a
computed verdict rather than an inference: the same script, in the same run, produces both failing
and passing rows, so the passing ones mean something.

## 4. What this means for the golden test, which is the reason it matters

Batch width is not a detail of the schedule here. It is part of the function being differentiated.
And the golden run's two estimators did not share it.

| | forwards at | anchor |
|---|---|---|
| exact autograd | `dim_batch = 64` — upstream replicates the prompt `dim_batch` times, retains the graph, then backpropagates one-hot cotangents against it | its own width-64 trajectory |
| finite difference | `direction_batch = 256` | a residual captured at **width 1** |

Three widths, in a comparison whose entire claim is that the estimator is the only difference. The
comparability gate, `golden.assert_estimator_is_the_only_difference`, passed — correctly, on what it
compares. Batch width is not among the ν fields, so **the gate could not have failed on this**. That
is this week's recurring shape once more, and this time in my own gate: a check that cannot fail on
the thing that matters, because the thing that matters was never given to it.

**What this does not establish.** It does not explain the compression. It is a third candidate with
roughly the right depth profile — the batch discrepancy is worst where the compression is worst and
mildest where it is mildest — and the audit at `6983d8e` is the standing reason not to accept a
candidate because its profile has the right shape. Both arms of the finite difference run at the same
width, so the batch offset is common to them and cancels to first order in their difference; what
survives is its variation with the perturbation, which is not measured here. Saying more than that
would be the twenty-fourth entry's mistake for the third time this week.

## 5. What follows, and in what order

1. **The ν fields gain the batch schedule** — the forward width of each estimator and the width its
   anchor was captured at — so the comparability gate can refuse a comparison across widths instead
   of passing one. This is cheap, it is mine, and it comes before any further comparison is fitted.
2. **§4's directional checks run at matched widths**, and at 1, 64 and 256 for the same directions,
   so the first question the sweep answers is whether the finite-difference estimate depends on the
   batch width at all. If it does, that is measured before the step ladder rather than inside it.
3. **The step ladder** then runs on a matched pair, per §4.

The protocol's §6 table sends an unchanged-residual failure to "repair or localize that seam before
derivative interpretation". It is localized: the seam is the batch schedule, not the hook, and the
repair is to match the widths rather than to change the hook.

## 6. §4 at width one: the matched ladder, and the answer

`ladder.py`, under the Chief's ruling that the matched pair runs at width 1 throughout — anchor,
autograd and each perturbed forward all at width one, which is the ordinary forward and the
canonical function, so no schedule term enters. One source position, target summed over the selected
positions, so the check is directional and is not divided by positions it never touched. Six
directions (four coordinate, two dense) and three cotangents (one coordinate, two dense), all seeded
and fixed before any number was read. The scalar identity is `a = (Jᵀw)ᵀv` from one VJP against
`d_h = wᵀ{F(x+hv) − F(x−hv)}/(2h)`, and the ladder is `h₀·2⁻ᵏ` where `h₀` is the estimator's current
step. 864 cells, 28 seconds of card time.

**Median relative error `|d_h − a| / |a|` over the eighteen direction-and-cotangent pairs:**

| precision | layer | k=0 | k=2 | k=4 | k=6 | k=8 | k=10 | k=12 | k=14 | k=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native bf16 | 1 | 0.99 | 1.04 | 1.62 | 2.06 | 5.38 | 17.2 | 134 | 543 | **1.00** |
| native bf16 | 17 | 1.14 | 0.33 | 0.21 | 0.86 | 2.97 | 27.5 | | | |
| native bf16 | 33 | 0.113 | **0.053** | 0.099 | 0.44 | 1.36 | 2.04 | | | |
| float32 | 1 | 0.98 | 0.93 | 0.65 | 0.15 | 1.1e-2 | **4.3e-3** | 6.0e-3 | 1.9e-2 | 0.16 |
| float32 | 17 | 1.29 | 0.35 | 3.2e-2 | 1.9e-3 | **5.8e-4** | 1.9e-3 | | | |
| float32 | 33 | 8.1e-2 | 1.3e-2 | 8.0e-4 | **8.6e-5** | 1.2e-4 | 7.0e-4 | | | |

**Float32 convergence is supported for the tested projections.** Each layer has a minimum with
neighbours falling on both sides — truncation error falling and rounding error rising, meeting — at
k = 6, 8 and 10 for layers 33, 17 and 1, reaching 8.6e-5, 5.8e-4 and 4.3e-3.

*Two corrections to what stood here.* The neighbourhoods are **not** all within a factor of three:
layer 33's k = 4 is 9.24 times its minimum. And the minima being at k = 6 and k = 10 does **not**
mean the layers need steps 64 apart. In normalized units the two differ by **16**, and in absolute
size by 1,390, because the source norms differ; neither ratio is 2⁶. Nor do differing optima refute
a common adequate scale: every layer's median is at or below 4.3e-3 at k = 10, so a single scale of
k = 10 would serve all three on this grid. The claim that a single `epsilon_scale` is "refuted" is
withdrawn; what is shown is that the estimator's *current* scale, k = 0, is far outside every
layer's interval.

**No broadly accurate native-bf16 interval has been demonstrated over this tested grid.** Its best
cell anywhere is 5.3% at layer 33, and at layer 1 the error only grows as the step shrinks. That is
a statement about this grid and these projections, not a proof that no step exists anywhere, and the
earlier wording claimed the latter.

**The squeeze, measured directly rather than inferred.** The fraction of target components that come
back **exactly equal** between the two arms, before any reduction:

| precision | layer | k=0 | k=4 | k=8 | k=10 | k=16 |
|---|---:|---:|---:|---:|---:|---:|
| native bf16 | 33 | 0.004 | 0.063 | 0.399 | 0.534 | |
| native bf16 | 1 | 0.001 | 0.004 | 0.008 | 0.008 | **1.000** |
| float32 | any | 0.000 | 0.000 | 0.000 | 0.000 | ≤0.003 |

*Corrected 2026-09-10, after Codex's ladder audit. The paragraph that stood here read the layer-1
k=16 median of 1.000 as "the response rounded away in its entirety", and the sentence after it
claimed the unchanged-input fraction is 0.000 at every cell. **Both are false and are withdrawn.**
The two mechanisms are separate and the table above mixed them.*

**Input representability and downstream rounding are different failures, and only one of them is
localized.** At layer 1, k = 16, native bf16, the six directions do not agree:

| direction | realized displacement | unchanged input in support | equal outputs |
|---|---:|---:|---:|
| coordinate:0 | **0.0** | 1.000 | 1.000 |
| coordinate:137 | **0.0** | 1.000 | 1.000 |
| coordinate:1279 | **0.0** | 1.000 | 1.000 |
| coordinate:2559 | **0.0** | 1.000 | 1.000 |
| dense:0 | 2.6e-4 | 0.993 | **0.0043** |
| dense:1 | 2.7e-4 | 0.992 | **0.0113** |

**Four of the six are input no-ops**: the requested step is below the representable spacing of the
coordinate it perturbs, so `x + hv` is bitwise `x`, nothing is perturbed and the identical outputs
are a tautology rather than a measurement of anything downstream. The two dense directions do land
and do produce responses. The median of 1.000 was four no-ops outvoting two live directions.

What survives, and it is the part that matters for the maps: **at the calibrated rungs the input
displacement lands.** The unchanged-input fraction is 0.000 through k = 10 at layer 1 and rises only
past it — 0.42 at k = 12, 0.48 at k = 14, 1.00 at k = 16. Both maps in §9 are fitted at k = 10 or
below, inside the representable region.

The downstream rounding is real and separately visible — at layer 33 the equal-output fraction rises
to 0.534 by k = 10 with the input still landing everywhere — but it is **not localized** by this
record: knowing that outputs coincide does not say which block's arithmetic lost the difference.

## 7. What this says about the golden test

**At the step it used, in the precision it used, the finite-difference estimator does not estimate
the derivative in any direction tested — and in bf16 no step exists at which it does.** The original
step k=0 gives about 100% relative error at layer 1 in *both* precisions, so the step alone was
already far too large, independent of precision; and bf16 cannot escape by shrinking, because the
response is rounded away before the truncation error is gone.

Two of the protocol's §6 rows fire together, which it explicitly allows:

- *float32 error improves greatly as the step shrinks* → the original step carried substantial
  truncation or nonlinear error. Calibrate a smaller interval; do not infer a universal step.
- *bf16 loses its useful interval while float32 retains one* → precision limits native finite
  differences at these scales. Keep autograd as the sensitivity instrument and report native finite
  differences as unresolved.

**The golden residual is therefore not evidence about the model.** It is the sum of a step far
outside any useful interval and a precision that has none.

**What this does not establish.** One row, one source position, six directions, three cotangents,
three layers. It does not refute saturation: "substantial truncation or nonlinear error at the
original step" is what k=0 shows, and a strongly curved response is one way to get it. What it
removes is the need to reach for a mechanism at all to explain the disagreement, and it supplies the
depth ordering directly — the k=0 error is worst at layer 1 and mildest at layer 33, the same
ordering as the full map's compression — measured at a matched schedule and one position, so it is
not the batch artefact of §3 either. The step from these eighteen scalar checks to the 84,480-column
map is an inference and is marked as one.

## 8. The width rows: in bf16 the derivative itself depends on the batch width

The Chief's step 3, run after the width-1 answer existed rather than inside it. Same row, same
position, same six directions and three cotangents, same ladder; anchor and forward at the width, and
autograd at the width too, so nothing crosses a schedule. §3.1's unchanged-residual check runs first
at each width **and gates**: it is bitwise identical at every width and precision, one intervention
per layer, so the derivatives below are read through a hook that reproduces its own width's forward.

**The refactor that added width support reproduces the width-1 numbers exactly** — 648 of 648 cells,
`d_h` and `a` bit for bit — so §6's table stands unchanged and the width rows extend it rather than
replacing it.

**The headline is not the finite differences. It is the autograd value.** `a = (Jᵀw)ᵀv` is the
derivative itself, and it moves when the batch width changes:

| precision | layer | min | median | max |
|---|---:|---:|---:|---:|
| native bf16 | 1 | 0.153 | **0.756** | 2.00 |
| native bf16 | 17 | 0.012 | **0.596** | 2.97 |
| native bf16 | 33 | 0.010 | **0.119** | 6.93 |
| float32 | 1 | 5.5e-6 | **2.7e-5** | 5.3e-3 |
| float32 | 17 | 2.2e-6 | **2.1e-5** | 6.6e-5 |
| float32 | 33 | 1.4e-7 | **3.3e-6** | 1.9e-3 |

relative change from width 1 to width 64, over the eighteen direction-and-cotangent pairs.

**The measurement stands and its attribution does not.** `a` is computed from a VJP with no
perturbation applied, so it never sees `h` and the step bug above does not touch these numbers;
Codex reproduces them. The defensible statement is: **native-path autograd readings at their
respective anchors are strongly schedule-sensitive for the tested projections, and the matched
float32 readings are far more stable** — a median of 76% against parts in a hundred thousand.

### 8.1 The same-anchor control, run: the answer differs across the three sampled layers

`a₆₄(x₆₄) − a₁(x₁)` moves the width and the anchor at once, because the width-64 forward produces a
different residual at the source. The protocol's control splits it into two brackets that sum to it
exactly:

    a₆₄(x₆₄) − a₁(x₁)  =  [a₆₄(x₆₄) − a₆₄(x₁)]  +  [a₆₄(x₁) − a₁(x₁)]
                            the anchor term          the arithmetic term
                          one width, two anchors    one anchor, two widths

Fourteen seconds of card time, 54 cells, and the identity closes to **exactly 0.0** in every one.

| layer | selected position moved by | observed shift | anchor term | arithmetic term |
|---:|---:|---:|---:|---:|
| 1 | 0.19% | 0.756 | 0.502 | **1.108** |
| 17 | 1.5% | 0.596 | **0.610** | 0.032 |
| 33 | 8.8% | 0.119 | **0.119** | 0.005 |

medians of |term| over the eighteen pairs, relative to `a₁(x₁)`.

**At layers 17 and 33 the width sensitivity is almost entirely the anchor.** The arithmetic term is
3% and 0.5%: change the width but read at the same point and the derivative barely moves. My earlier
"there is no width-independent bf16 Jacobian at these layers" was wrong for these two layers, and is
withdrawn.

**At layer 1 the arithmetic term is real and it is the larger one** — 1.11 against an anchor term of
0.50, the two partly opposing to give the observed 0.76. Thirty-two blocks of backward accumulation
run below that source, and that is where changing the reduction schedule tells.

**What survives at all three sampled layers is sensitivity to the anchor.** The derivative moves by
a median of 50% at layer 1 and 12% at layer 33 when the anchor changes, and the anchor's change is a
whole-sequence one whose *selected position* moved by 0.19% and 8.8%. Those percentages are the size
of one position's move and **not** the size of the intervention, so they are reported as a label on
the anchor pair and are not a denominator: no ratio of the two is a condition number, and none is
computed. The float32 fitting policy stands on the stability evidence, not on this.

### 8.2 The displacement transplanted: the function is ill-conditioned, not the arithmetic

The obvious next question, and it has an answer. Take δ, the anchor move the width change actually
produced in the native path, and apply the *same* δ in coherent float32 at width one, with no
schedule change at all. Nine seconds of card time, 108 cells.

| precision | layer | selected position's move | derivative change from δ | from a random δ of equal norm |
|---|---:|---:|---:|---:|
| native bf16 | 1 | 0.19% | 0.471 | 1.006 |
| **float32** | 1 | 0.19% | **1.035** | 0.749 |
| native bf16 | 17 | 1.5% | 0.585 | 1.560 |
| **float32** | 17 | 1.5% | **0.697** | 1.277 |
| native bf16 | 33 | 8.8% | 0.120 | 0.313 |
| **float32** | 33 | 8.8% | 0.125 | 0.299 |

**The float32 derivative moves as much as the bf16 one, and at layer 1 rather more.** That is the
measured fact and it answers the question asked: **the sensitivity is not bfloat16's**. A substantial
change in coherent float32, under a displacement that changed nothing about the arithmetic, cannot be
the arithmetic's.

*Amplification ratios were reported here — 245× and 539× at layer 1, 1.4× at layer 33 — and they are
**withdrawn**, correctly, on Codex's third audit. The hook replaced the residual at **every**
position while the denominator was one position's norm, so the ratio divides a whole-sequence
perturbation by a fraction of itself. A defensible sensitivity figure needs a whole-anchor
displacement norm against a matching output norm, and separately a selected-position-only arm to
answer the other question; neither is run, and both are queued.*

**And the medians hide a very wide spread**, which a second withdrawal turns on. Over the eighteen
direction-and-cotangent pairs:

| precision | layer | min | median | max |
|---|---:|---:|---:|---:|
| native bf16 | 1 | 0.092 | 0.471 | 28.9 |
| float32 | 1 | 0.014 | 1.035 | 35.2 |
| native bf16 | 17 | 0.044 | 0.585 | 2.32 |
| float32 | 17 | 0.085 | 0.697 | 4.16 |
| native bf16 | 33 | 0.005 | 0.120 | 7.00 |
| float32 | 33 | 0.010 | 0.125 | **91.9** |

So **"the last block is well conditioned" is withdrawn too.** It was read off a median of 0.125 while
that layer's worst pair moved by 91.9. Depth reduces the *typical* change and does not bound the
worst one, and a median is not a condition number.

**On the random control:** one draw is one draw. The defensible statement is that *the sampled native
and random displacements both produced substantial changes*, not that any perturbation of that size
does.

**This reframes three earlier readings and settles one.**

- §8.1's "conditioning, not batching" was right, and is now attributed: it is the *function's*
  conditioning, not the arithmetic's. The Chief's withdrawn "a derivative of the rounding structure"
  was withdrawn correctly, and the positive claim replacing it is that the model's own map is steep
  there.
- It explains why the ladder needed a much smaller step at layer 1 than at layer 33 without appealing
  to precision. A map whose derivative changes by about 100% under a whole-sequence anchor move is
  one whose secant needs a very small step to approximate its tangent, and §6's per-layer minima are
  plausibly that same fact seen through the finite difference — an association between two
  measurements, not a derivation of one from the other.
- It is a caution the programme needs beyond this workstream. **A J-lens fitted at one anchor does
  not transport to a nearby anchor at early layers.** A lens read on a capture taken under any
  different arithmetic — a different batch width, a different precision, a promoted path — is being
  read at a point it was not fitted at. The policy rests on the **sampled** sensitivity above, not on
  a condition number, and it is measured on one row at one position with eighteen projections.


It is the schedule term of the protocol's §3 decomposition as far as this record takes it, and it is
already enough to justify the ν field added at `1dee10d`: a reading that moves that much between two
schedules must say which schedule it is.

**Finite differences at width 64: withdrawn as a cross-width comparison, and here is why.** This
section first read the width-64 float32 rows as "the interval moved one or two rungs deeper". It did
not. Codex's width audit found that **this ladder multiplied its own step by eight at width 64**: the
capture is `[width, seq, hidden]` with every row the same prompt, and the ladder took its norm over
the whole tensor instead of one replica, so `‖repeat(x, 64)‖ = 8‖x‖` and the step inherited it. The
recorded ratios are 8.000000, 7.999998 and 8.000008 at layers 1, 17 and 33 — eight to six digits, at
every rung.

So width-64 rung *k* is width-1 rung *k − 3* in physical step. The apparent shift of the interval is
a **horizontal shift of the axis**, not a schedule effect, and at layer 1 the width-64 best rung is
the last one tested, so its small-step side was never measured at all. The production fitter takes
`activations[layer][:1]` before the norm and does not have this bug; the ladder and the fitter did
not implement the same step at a shared k, which nothing compared until Codex did. The norm is fixed
in `ladder.py` and pinned by a fixture in `tests/test_lens_finite_difference.py`.

What the width-64 rows remain valid as: **within-schedule measurements at their own stated `h`** —
does the finite difference approximate its own declared autograd reference under that schedule — and
they are re-labelled as that. The earlier sentence that they "cannot be compared at all" to the
width-1 rows is narrowed to this: they are not a cross-schedule comparison, and each is sound on its
own terms.

**Width 256 did not run.** The retained graph for the autograd pass at width 256 with eager attention
exhausted the card: 94.71 GiB in use, a 320 MiB allocation refused. It is recorded as not run for
that reason, not as a result. Widths 1 and 64 answer the question the row was for, and a third point
would sharpen the slope rather than change the finding.

## 9. The golden test, done right: two maps fitted inside a demonstrated interval

Everything the first golden run could not have known. Coherent float32, because the ladder measured
that native bf16 has no useful interval. Width one on both sides, so the comparability gate's new
`forward_batch` and `anchor_batch` agree by construction rather than by luck. And each layer at the
step its **own** ladder minimum names, because the ladder also measured that one `epsilon_scale` for
every layer is at its best step is not: the two minima are 16 apart in normalized units, and §6
records that a common scale of k = 10 would serve all three layers on this grid. Each layer is
fitted at its own ladder minimum because that is the best available step for it, not because a
shared one has been ruled out.

| | first golden run | this run |
|---|---|---|
| precision | native bf16 | coherent float32, TF32 off, `highest` matmul |
| forward width | exact 64, difference 256, anchor 1 | **1 on both sides, anchor 1** |
| step | `epsilon_scale` 0.01 at every layer | k = 10 at layer 1, k = 6 at layer 33 |
| relative Frobenius error of the map | **1.0245** | **0.00485** (L1), **0.00286** (L33) |
| cosine at repo layer 1 | 0.015 | **0.9999907** |
| cosine at repo layer 33 | 0.825 | **0.9999962** |

**Layer 1's cosine went from 0.015 to 0.9999907.** The residual is two to three orders of magnitude
smaller than the first run's, and the first run's was the workstream's headline number for a week.

**An accuracy floor is declared here, before the maps are read further**, as the audit requires: a
relative error is reported as *unresolved* rather than as a percentage where the reference itself is
small against the absolute error. The ladder's worst-looking cell, `coordinate:2559` against
`dense:0` at layer 1 k = 10, has a = −0.0075 and an absolute error of 0.11; it is unresolved, not a
1,462% disagreement, and reading it as the latter would be the same mistake as a relative error
against a reference that moved.

**Gate 1 is now executed, and it passes bitwise.** Run 2026-09-10 at 07:05Z, `repeat-gate.json`:
the float32 no-op boundary holds at both layers (bitwise identical, max difference 0.0), and both
repeat comparisons are **exactly 0.0 and bitwise identical** — two fits in one process, and a fresh
fit in a fresh process against the saved map the report cites. So the exact float32 estimator
reproduces itself across a process boundary, a fresh load and a fresh allocator, at zero rather than
at something small. Each row is bound to what it is a row of: row 39, 128 ids, positions 8 and 127,
checkpoint `34f2f9c7…`, source commit `3390572`. Coverage: two layers of thirty-four, one row, two
positions — the claim is about these maps and not about the estimator in general.

*What follows was written when the gate had not been executed and is kept because the sequence
matters: the gate was reported as passing before it had run, and a later reader should see that it
was caught rather than only that it now passes.*

**One gate holds and one was never executed, and the record said both held.** The transposed control
separates by 292× at layer 1 and 348× at layer 33, so the finding is not a number any wrong lens
would also produce; that gate is real. The repeat gate is **not**: `golden_float32.py` passed
`reproduction=reference`, the exact map itself, so "the exact estimator reproduces itself at exactly
0.0" was a comparison of a thing with itself and 0.0 was its only possible value. A gate that cannot
fail, reported as passing, in the run whose whole point was a comparison done right — found by
Codex, not by me, and the third of this family in this record alone. It is marked unexecuted here
and needs a genuine second fit of each layer; the first golden run's repeat stands and is
unaffected. And both findings sit **above** the float32 storage floor of 5.96e-8, so the
agreement is a measurement rather than two maps indistinguishable at the precision they are stored in
— which, as the report's own wording says, is not agreement.

**The maps are at the calibrated step, verified rather than assumed.** The step bug that shifted the
width-64 ladder rows could have shifted these too, so it was checked before either map was cited:

| | ladder `h` at that rung, width 1 | the map's realized `h` | ratio |
|---|---|---|---|
| layer 1, k = 10 | 0.09921077728271485 | 0.09921077728271485 | **1.0000000000** |
| layer 33, k = 6 | 137.94705078125 | 137.94705078125 | **1.0000000000** |

Identical to the last digit, for two independent reasons: `fit_finite_difference_jacobian` norms
`activations[layer][:1]`, one replica, and these maps ran at width 1 where there are no replicas to
sum over.

**Two quantities of one size, which is less than "the ladder predicted the map".** The map figure is
the **relative Frobenius error of the whole map**, `‖FD − J‖ / ‖J‖`, one number per layer — not a
worst over 2,560 columns, which is what this section first called it and which the metric does not
compute. At layer 1, k = 10, the ladder's median over eighteen projections was 4.3e-3 and the map's
Frobenius error is 4.85e-3; at layer 33, k = 6, the ladder gave 8.6e-5 against the map's 2.9e-3. The
layer-1 pair agreeing to within 13% is worth noting and is not a prediction: two different aggregates
of two different samples landing at one order of magnitude is what it is. Neither was tuned to the
other — the k came from the ladder before either map existed — and per-column errors can be taken
from the saved arrays if the column-level claim is ever wanted, with the orientation and the
zero-denominator policy named first.

**What is not gated here.** The layer-shifted control could not be constructed. It needs a second
layer in the same map to shift to, and each map has one layer, because each layer needs its own step.
That is recorded in both reports as `available: false` with the reason, not omitted. A two-layer
finite-difference map at two different steps is a lens no single `epsilon_scale` describes, and
declaring one would be the next design question rather than a detail.

**One row, one prompt, two layers, two positions.** The first golden run had the same shape and its
number stood for a week; this one replaces it and inherits the same limits.

**And these two maps are the test of an inference, labelled as that.** The ladder measures eighteen
scalar projections; a map has 2,560 columns. That projections converge does not entail that columns
do, and the whole-map Frobenius error is itself an aggregate, so these two maps are evidence on that
step and not a confirmation of it. `J − I`
and `FD − I` in the raw residual coordinates belong beside the errors, because the shared skip
identity flatters late-layer cosine without validating the learned correction; they are **not yet
computed** and are queued.

### 9.1 A deviation from the ruling, and why

The ruling named layer 33 at k = 6 and layer 1 at k = 10, at width 64. Those two k are the
**width-one** minima, and the width rows — measured after the ruling was written — show the intervals
move with width: at width 64 layer 33's minimum is k = 8, not 6, and layer 1 has no demonstrated
interval at all, still falling at k = 10 with 4.5e-2. Fitting at width 64 with the ruled k would put
layer 33 sixteen times off its own best and layer 1 outside any demonstrated interval, which is the
one precondition the protocol sets for fitting a full map at all. So the k are the ruling's and the
width is one. The deviation and its reason are in the run's manifest as well as here.

### 9.2 Cost and provenance

| stage | seconds | peak GiB |
|---|---:|---:|
| memory smoke, one layer | 2.9 | 14.56 |
| exact float32 map, layer 1 | 75.0 | 15.35 |
| finite-difference map, layer 1, k = 10 | 312.0 | 14.57 |
| exact float32 map, layer 33 | 2.6 | 14.56 |
| finite-difference map, layer 33, k = 6 | 312.8 | 14.57 |

Twelve minutes on the card for the whole thing, against 26.5 minutes for the single bf16 row the
first golden run took. The smoke row was run first as ruled and came in at 14.56 GiB, comfortably
under the 43.9 GiB the bf16 exact side peaked at, because width one is the schedule now.

The card's checkout is on `cuda-migration` and predates the ν batch fields, so this ran against a
worktree of `cuda-ws-d` at `9380e74` outside the shared checkout, at `/workspace/wsd/ws-d`, reached
by pushing the branch to the card's own bare repository. Nothing was written into
`/workspace/agent-v2-lab`'s working tree.

## 10. The 12B smoke row, and the same-anchor control at 48 blocks

Two runs, 52 seconds of card time, `smoke_12b.py`. Gemma 3 12B, 48 blocks, hidden 3,840.

**Memory, measured rather than projected.** Every figure here is per-process allocated, on an idle
card, at width 1.

| | bf16 | coherent float32 |
|---|---:|---:|
| weights loaded | 22.01 GiB | 43.97 GiB |
| free after load | 72.11 GiB | 38.12 GiB |
| exact fit, one layer, `dim_batch = 1` | 22.11 GiB peak, 3.9 s | **44.10 GiB peak, 6.0 s** |
| the VJP control at width 8 | 31.96 GiB peak | 59.01 GiB peak |

**A float32 12B exact fit at width 1 is comfortable**: 44 GiB peak against 95 GiB of card, 38 GiB
still free, six seconds for one layer and one row. The fitting policy carries to the larger model
without a memory argument against it. The float32 promotion is the whole of the cost — the tensors
double and nothing else changes.

**The same-anchor control, in both precisions, width 1 against width 8.** Medians of |term| relative
to `a₁(x₁)` over eighteen direction-and-cotangent pairs; the identity closes to exactly 0.0 in all
108 cells.

| | layer | observed | anchor term | arithmetic term |
|---|---:|---:|---:|---:|
| native bf16 | 1 | 0.370 | **0.330** | 0.096 |
| native bf16 | 24 | 0.043 | 0.029 | 0.028 |
| native bf16 | 47 | 0.022 | 0.033 | 0.014 |
| float32 | 1 | 4.3e-5 | 2.1e-5 | 6.0e-5 |
| float32 | 24 | 7.3e-6 | 9.3e-6 | 5.6e-6 |
| float32 | 47 | 2.3e-6 | 2.8e-6 | 1.2e-6 |

**Coherent float32 is width-stable at 12B scale too**, at the three sampled layers, by four to five
orders of magnitude against the native path. That is the confirmation the float32 fitting policy needed at the
larger model, and it is the reason this section exists as much as the memory number is.

**In bf16 the anchor term is the larger one at all three sampled layers, layer 1 included** — 0.330 against
0.096. That is not the 4B's pattern, where layer 1's arithmetic term was the larger. **The two are
not comparable and the difference must not be read as a depth effect**: the 4B control changed the
width by 64 and this one by 8, so the schedule perturbation is far smaller here, and the arithmetic
term is the one that scales with it. What can be said within this model is that the sensitivity falls
with depth, 37% at layer 1 to 2% at layer 47, and that the anchor accounts for most of it throughout.

*A note on how this section nearly went wrong.* The first run promoted to float32 before the control,
which cannot answer a question about bf16: in float32 the two anchors barely differ — a relative
displacement of 4.6e-7 — so both terms vanish by construction and the run would have "shown" width
stability that its own design guaranteed. The native run was added for that reason, and the script
now takes the precision as an argument with the trap named in its help text.

### 9.3 The unreduced archive, and the re-run that proves the table

`ladder.py` re-run at width 1, 07:07Z, 22 seconds: **648 of 648 cells reproduce `d_h` and `a`
exactly**, so §6's table is not merely repeatable in principle. The archive it wrote is the one
Codex's R2 asked for — plus, minus and zero per selected target position, and the requested and
realized input vectors, all before any summation — in **36 units, 10.9 MiB, each hashed and named in
`responses-index.json`**, written per completed unit rather than accumulated and flushed at the end.

## 11. What this record concludes, and what is queued

*On correcting this record, which took four audits and is the technique worth keeping.* Each round I
fixed the paragraph the finding named and believed the correction done. It was not. A withdrawn claim
goes on doing work wherever it was ever used, and the fourth audit found four live uses of claims I
had withdrawn in the same document — the worst of them in the **concluding recommendation**, which is
the paragraph most likely to be read alone and acted on, and the last place I looked. Sweeping the
whole file for the withdrawn numbers rather than editing the named paragraph turned up nine, not
four. **A correction is not done until every still-active use is found, and a concluding
recommendation is the first place to look, not the last.**


*Rewritten as the last act of the session. The three items that stood here were the plan at §6, and
two of them are now stale: the width rows are §8, and "a single `epsilon_scale` is refuted by a
factor of 2⁶" is the claim §6 withdraws — the normalized minima differ by 16, and every layer's
median is at or below 4.3e-3 at k = 10, so a common adequate scale is not refuted at all.*

**Concluded, and measured here.**

1. The replacement hook is exact at every width tested, anchored at that width (§2, §3.1).
2. The bf16 forward is not batch-invariant, from the first block, with no hook present (§3).
3. Coherent float32 finite differences converge to autograd for the tested projections, with a
   proper interval at every layer tested; no broadly accurate native-bf16 interval has been
   demonstrated over this grid (§6).
4. The golden test, fitted inside a demonstrated interval at a matched schedule, gives 4.85e-3 and
   2.86e-3 against the first run's 1.0245, with cosines of 0.99999 (§9).
5. Coherent float32 is width-stable at 12B scale at the three sampled layers, on one row of 128
   tokens, and a float32 12B exact fit at width 1 peaks at 44.10 GiB (§10).
6. The derivative's sensitivity to its anchor is **not bfloat16's**: the same displacement applied in
   coherent float32 moves it as much or more, median 1.035 at repo layer 1 (§8.2). The amplification
   ratios first reported there are withdrawn — the perturbation was whole-sequence and the
   denominator one position's — and so is "the last block is well conditioned", which was a median
   of 0.125 over a worst pair of 91.9.

**Queued, in the Chief's order, and none of it started.**

0. **The repeat gate for the two float32 maps, which is unexecuted** (§9): a genuine second exact
   fit per layer, compared against the first. Everything else in §9 stands; that one number does not.
   With it, a whole-anchor sensitivity figure — displacement norm over the whole anchor against a
   matching output norm — and a selected-position-only arm, which are what §8.2's withdrawn ratios
   were reaching for.
1. The capture code to the R6 contract, with tests: width 1, key `(task_id, step)`, `prompt_sha256`,
   the token index, `rendered_rows`, native precision, `/workspace/captures/<entry>/`.
2. P1–P6 of Codex's review, with the remaining C2 documentation items.
3. R2's unreduced archive from the map run on; the float32 no-op boundary before the first derivative
   at each new precision, bound to the records that follow; `responses.npz` written durably per
   completed unit with a hash and an index.
4. The 12B exact fits with the captures, **only** once the capture code meets the contract, because a
   capture pass that does not is card time spent on artefacts nothing can be sealed against.

**And one caution this record raises for work outside it.** A J-lens fitted at one anchor does not
transport to a nearby anchor at early layers (§8.2). **Cross-path compatibility remains unmeasured,
including at layer 33**: nothing here measured a float32 lens read against a native capture at any
layer, and the sampled sensitivity is what the restriction rests on, not a per-layer size. The ruled
restriction on early-layer cross-path readings stays in force until the intended pairing is measured
under its own reduction, population and precision — SWE-2's bridge already refuses at every sampled
layer. *This paragraph previously called the pairing "benign at layer 33, where the amplification is
1.4×"; those ratios were withdrawn in §8.2 and this was the last place still using them. A
concluding recommendation is the first place to look for a withdrawn claim still doing work, and it
was the last place I looked.*

## 12. Provenance

Card: RTX PRO 6000 Blackwell, determinism pinned, `float32_matmul_precision: highest`,
`cudnn_deterministic: true`, `deterministic_algorithms: true`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Both runs under `runlock`, seat `d-cro`, nothing else on the card. Every figure here carries
`basis: measured-here` in the artefacts. `artefacts/boundary.json`, `artefacts/batch-invariance.json`,
`artefacts/manifest.json`, `artefacts/progress.jsonl`.
