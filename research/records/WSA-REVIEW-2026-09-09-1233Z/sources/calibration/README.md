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
it. **The hook is not implicated, and nothing about the hook needs repairing.**

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

## 6. Provenance

Card: RTX PRO 6000 Blackwell, determinism pinned, `float32_matmul_precision: highest`,
`cudnn_deterministic: true`, `deterministic_algorithms: true`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Both runs under `runlock`, seat `d-cro`, nothing else on the card. Every figure here carries
`basis: measured-here` in the artefacts. `artefacts/boundary.json`, `artefacts/batch-invariance.json`,
`artefacts/manifest.json`, `artefacts/progress.jsonl`.
