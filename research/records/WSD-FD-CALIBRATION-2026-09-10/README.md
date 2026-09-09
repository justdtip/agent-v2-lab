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

**Float32 has a useful interval at every layer tested, and it is a proper one** — a minimum with
neighbours on both sides within a factor of about three, which is truncation error falling and
rounding error rising, meeting. It sits at k=6 at layer 33, k=8 at layer 17 and k=10 at layer 1: the
shallower the source, the smaller the step it needs and the looser its best agreement, 8.6e-5 to
4.3e-3 across the three. That is the protocol's "coherent float32 FD converges to coherent float32
AD": the tested float32 local derivatives are supported.

**Native bf16 has no useful interval at any layer tested.** Its best cell anywhere is 5.3% at layer
33, and at layer 1 the error only grows as the step shrinks. It is squeezed from both sides at once
and the two sides overlap.

**The squeeze, measured directly rather than inferred.** The fraction of target components that come
back **exactly equal** between the two arms, before any reduction:

| precision | layer | k=0 | k=4 | k=8 | k=10 | k=16 |
|---|---:|---:|---:|---:|---:|---:|
| native bf16 | 33 | 0.004 | 0.063 | 0.399 | 0.534 | |
| native bf16 | 1 | 0.001 | 0.004 | 0.008 | 0.008 | **1.000** |
| float32 | any | 0.000 | 0.000 | 0.000 | 0.000 | ≤0.003 |

At layer 1 in bf16, by k=16 **every** target component is bitwise identical between `F(x+hv)` and
`F(x−hv)`. The difference is exactly zero, so `d_h` is exactly zero and the relative error is exactly
1.00 — which is why the bf16 row above stops falling and pins there. The response has been rounded
away in its entirety.

This is the quantity the Director's audit said the zero-column count could not reach: not whether an
aggregated column vanished, but what fraction of the individual responses did, before aggregation.
It is now measured, and the answer is that in bf16 it goes to one.

The input side is never the problem: the fraction of perturbed coordinates that did not move is
**0.000 at every cell, both precisions, every step**. The displacement always lands; the loss is
downstream.

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

**In bf16 there is no width-independent Jacobian at these layers.** Change nothing but the batch
width of the forward and the directional derivative moves by a median of 76% at layer 1, 60% at layer
17 and 12% at layer 33, and by more than 100% in the worst pairs. The map is a property of the
schedule as much as of the model. In float32 the same change moves it by parts in a hundred thousand,
which is the arithmetic noise of a different reduction order and nothing more.

That is the schedule term of the protocol's §3 decomposition, measured. It also settles what the ν
field added at `1dee10d` is worth: not a bookkeeping nicety, but up to a factor of two.

**Finite differences at width 64, for completeness.** Float32 still converges, with the interval
moved one or two rungs deeper and the best agreement two to five times looser — 2.0e-4 at layer 33
(k=8, against 8.6e-5 at k=6 at width 1), 6.1e-4 at layer 17, 4.5e-2 at layer 1 and still falling.
Both arms share the width, so the batch offset is common to them; what does not cancel is its
variation with the perturbation, and that is the larger floor. The bf16 rows at width 64 are not
comparable to the bf16 rows at width 1 **at all**, because their reference `a` is a different number:
a relative error against a reference that moved 76% measures the pair, not the estimator.

**Width 256 did not run.** The retained graph for the autograd pass at width 256 with eager attention
exhausted the card: 94.71 GiB in use, a 320 MiB allocation refused. It is recorded as not run for
that reason, not as a result. Widths 1 and 64 answer the question the row was for, and a third point
would sharpen the slope rather than change the finding.

## 9. The golden test, done right: two maps fitted inside a demonstrated interval

Everything the first golden run could not have known. Coherent float32, because the ladder measured
that native bf16 has no useful interval. Width one on both sides, so the comparability gate's new
`forward_batch` and `anchor_batch` agree by construction rather than by luck. And each layer at the
step its **own** ladder minimum names, because the ladder also measured that one `epsilon_scale` for
every layer is refuted — the two minima are a factor of 64 apart.

| | first golden run | this run |
|---|---|---|
| precision | native bf16 | coherent float32, TF32 off, `highest` matmul |
| forward width | exact 64, difference 256, anchor 1 | **1 on both sides, anchor 1** |
| step | `epsilon_scale` 0.01 at every layer | k = 10 at layer 1, k = 6 at layer 33 |
| worst relative difference | **1.0245** | **0.00485** (L1), **0.00286** (L33) |
| cosine at repo layer 1 | 0.015 | **0.9999907** |
| cosine at repo layer 33 | 0.825 | **0.9999962** |

**Layer 1's cosine went from 0.015 to 0.9999907.** The residual is two to three orders of magnitude
smaller than the first run's, and the first run's was the workstream's headline number for a week.

Both gates hold. The exact estimator reproduces itself at exactly 0.0. The transposed control
separates by 292× at layer 1 and 348× at layer 33, so the finding is not a number any wrong lens
would also produce. And both findings sit **above** the float32 storage floor of 5.96e-8, so the
agreement is a measurement rather than two maps indistinguishable at the precision they are stored in
— which, as the report's own wording says, is not agreement.

**The scalar ladder predicted the map.** At layer 1, k = 10, the ladder's median relative error over
eighteen direction-and-cotangent pairs was 4.3e-3 and the map's worst over 2,560 columns is 4.85e-3.
At layer 33, k = 6, the ladder gave 8.6e-5 and the map's worst is 2.9e-3; a worst over 2,560 columns
being some tens of times a median over eighteen scalars is what those two statistics do, and the
layer-1 agreement is the more informative of the two. Neither was tuned to the other: the k came from
the ladder before either map existed.

**What is not gated here.** The layer-shifted control could not be constructed. It needs a second
layer in the same map to shift to, and each map has one layer, because each layer needs its own step.
That is recorded in both reports as `available: false` with the reason, not omitted. A two-layer
finite-difference map at two different steps is a lens no single `epsilon_scale` describes, and
declaring one would be the next design question rather than a detail.

**One row, one prompt, two layers, two positions.** The first golden run had the same shape and its
number stood for a week; this one replaces it and inherits the same limits.

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

## 10. What follows

1. **The width rows**, per the Chief's step 3: the same directions, cotangents and a few steps at
   widths 64 and 256, anchor and forward at that width, with §3.1's anchored-at-width check repeated
   first. That measures the schedule term after the matched answer exists rather than inside it.
2. **No further full bf16 map is justified.** The instrument has no useful interval in that
   precision, so another map would measure the same floor at more expense.
3. **The step rule is not a constant.** Its useful value moved by 2⁶ across three layers here, so a
   single `epsilon_scale` for every layer is refuted by this table whatever else is true.

## 11. Provenance

Card: RTX PRO 6000 Blackwell, determinism pinned, `float32_matmul_precision: highest`,
`cudnn_deterministic: true`, `deterministic_algorithms: true`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Both runs under `runlock`, seat `d-cro`, nothing else on the card. Every figure here carries
`basis: measured-here` in the artefacts. `artefacts/boundary.json`, `artefacts/batch-invariance.json`,
`artefacts/manifest.json`, `artefacts/progress.jsonl`.
