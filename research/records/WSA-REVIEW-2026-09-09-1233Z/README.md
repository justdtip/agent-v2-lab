# Review: batch calibration and the plan-progress pre-registration

**Codex, 2026-09-09. Review complete; changes requested before the pre-registration seal.**
The current device evidence establishes batch-width sensitivity and passes the unchanged-residual
check at width one at the three sampled source layers. It does not yet validate either derivative
estimator, and does not exonerate the replacement hook at every width. The pre-registration carries
the corrected corpus definition, but omits two new requirements and needs a ruling on its confidence
calculation and an explicit episode-held-out evaluation procedure.

This is a **preliminary review of the producer-branch snapshot** at `8b98129`, including the
calibration introduced at `4a2cfe2`, against the orders at `c030bed` and protocol at `558d424`.
The calibration has not yet landed on the fetched integration tip `4559f02`. No live status or
uncommitted card result is inferred from Git. No source experiment, order, pre-registration, lock,
checkpoint or model process was changed. All device figures below are computations from the frozen
producer record, not a new device run. Full source commits and byte hashes are in
[SOURCE-MANIFEST.json](SOURCE-MANIFEST.json).

## 1. Calibration: retain the narrow pass, correct the broad claim

### C1 — the batch-width effect is measured; hook exoneration is not

The native-forward control changes only batch width and uses passive activation recorders without
a replacement intervention. Every one of the 34 block outputs differs from width one at both tested
widths. Mean absolute target-residual difference is **76.4317856 at width 64** and **42.0402336 at
width 256**, in raw activation units, not percentages. All copies within each batch agree. This is
positive evidence of a schedule-dependent numerical forward on this fixed row and configuration.
It does not identify the particular kernel or reduction tiling responsible.

The boundary test replaces a source residual with a width-one capture. Its six width-one entries
pass exactly under `torch.equal`, corresponding to three distinct zero-step source interventions
(repo layers 1, 17, 33). Each is repeated with two position labels, but at zero step the position
label does not change the replaced full tensor. Twelve entries with widths 64/256 fail. None is a
width-64 anchor returned to a width-64 forward, or a width-256 anchor returned to width 256.

**Finding:** [batch_invariance.py](sources/calibration/batch_invariance.py), the `verdict` dictionary,
sets `hook_is_implicated` to `not all(...) and False`. The field is always false, for either
possible outcome of the native control. The report's assertion that nothing about the hook needs
repairing is therefore stronger than its evidence. Native width dependence and a wider-batch
hook defect could coexist. Replace that field with an explicitly untested/undetermined wider-width
status until same-width capture/replacement tests exist; preserve the valid width-one result.
The analysis checks both possible Boolean outcomes without importing or running producer code.

This does **not** reverse the Chief's width-one ruling. Keeping the anchor, ordinary forward,
autograd reference and both finite-difference arms at width one removes the measured schedule
confound from the next matched comparison. The original three-width comparison cannot attribute
its discrepancy to precision alone. Nor does this control demonstrate that width mismatch explains
the observed norm compression: the two perturbed arms share a width, and sensitivity to the
perturbation remains to be measured.

### C2 — state which pieces of the derivative protocol are still unexecuted

The source-equals-target checks replace a tensor by itself at zero step. They establish copying;
they do not establish an identity derivative for a nonzero realized displacement. The sign control
reruns the same positive and negative inputs in swapped order. Its odd-response reversal is useful
reproducibility evidence, but not an independent check of derivative magnitude or accuracy. Its
comment promises an unchanged even remainder, but the flipped even remainder is not calculated.

The current record compares components and selected positions before reduction, then saves summary
statistics. It **does not preserve the individual response vectors** needed to revisit a proposed
mechanism after averaging. This matters for the requested protocol §5 record. There are no saved
matched FP32/FP32 derivative comparisons, graph-once/sequential checkpoint comparisons, nonzero
identity derivative controls, step ladder with realized displacements/midpoints/spacing, or
per-direction autograd predictions in this boundary-stage record. These are **unexecuted**, not
failed measurements. Do not label the full derivative protocol passed from the boundary result.
The failed all-width boundary verdict is retained as run; no exception is treated as a scientific
branch in this review.

Before the next run, preserve the per-position/per-direction quantities and declare all forward
and anchor widths in the identity. Keep native bf16 and whole-path FP32 matched pairs separate.
The native-control JSON also lacks its own linked runtime manifest and explicit frozen-token digest;
link those to the boundary manifest before treating the two runs as a complete provenance chain.

## 2. Pre-registration: requirements missing from the current draft

### P1 — carry the new batch and readout-position fields into the capture contract

[PREREGISTRATION.md §2 and §8](sources/plan/PREREGISTRATION.md) specify native precision and identify
the last prompt token as the capture unit, but do not declare **`forward_batch: 1`**, now required
by the last WS-D ruling at `c030bed`. Add it to the capture execution and manifest contract.
Also include the actual token index in each cell, as the original R6 requires. The prose definition
of last prompt token is useful but does not replace the saved position after tokenization.

### P2 — retain the descriptive transport distance beside retrieval hits

The amended [plan order](sources/orders/PLAN.md) accepts E2 retrieval on the explicit condition that
transport distance is also reported descriptively. Draft §4.2 only specifies the hit/chance score.
Add the distance, its metric, and its aggregation. This does not add a new confirmatory estimand.
Freeze the retrieval candidate set and tie policy as well: including all N episode decisions,
including the current source, is consistent with the stated 1/N uniform-guess reference; excluding
the source would require N−1. A strict nearest-target rule makes ties misses. Name that policy so
an implementation cannot silently substitute argmin tie-breaking. Uniform guessing is a declared
reference, not a measured geometric null for this representation.

### P3 — refresh §13 to reflect rulings already made

A1's withdrawal, A3's revised capture key, the E2 retrieval proposal, and epsilon_sub = 0.11 have
already been ruled on. The golden-control design has landed. Replace the obsolete open-item list
with the current sequence: review and resolve the remaining issues, seal, then first reading.
The order permits capture before the seal; this review does not impose a new capture hold.

## 3. A statistical ruling is needed; the approved epsilon arithmetic is unchanged

### P4 — decision count is not a demonstrated independent sample count

The arithmetic in §7 reproduces: M = 12 and alpha = 0.05 yield required counts 1,715, 3,859 and 511
at the approved epsilon values 0.06, 0.04 and 0.11. The unestablished premise is that each counted
observation supplies an independent trial. The E1 population has **1,781 decisions in 240 episodes**;
ordinary E2 has **4,122 transitions in 840 training episodes**. Holding out entire episodes prevents
one kind of leakage but does not make decisions inside a held-out episode independent.
The standard bounded-sum Hoeffding result assumes independent summands; a dependent-data extension
would need its own assumptions and bound. [Hoeffding, 1963, original paper and abstract](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830).

A simple counterexample explains the issue: copy one fair binary observation eight times. The
average remains the same one observation, with variance 1/4. Treating its eight copies as independent
would report variance 1/32. This is an analytic illustration, not a claim that the actual corpus's
within-episode correlation equals one.

For context only, if independent bounded episode means receive decision weights w_e = n_e/N, the
sum of squared weights is sum(n_e²)/N². Its reciprocal is **169.7234 for E1** and **549.6533 for
ordinary E2** in this capture set. These are arithmetic quantities for a conservative weighted-episode
bound, **not estimated effective sample sizes**, and not replacement required_n values. Independence
between episodes and the treatment of learned/cross-fitted scores also require justification.

**Requested ruling:** define whether the target is a description of this finite corpus or an
inference to new episodes; specify the resampling or concentration unit and how dependence is
handled. An episode-level uncertainty procedure is a candidate, but its design and attainable
precision must be declared rather than inferred from row count. Keep the approved epsilons visible;
mark whether their claimed coverage is supported after that procedure is specified. This is a
methodological objection to an accepted ruling, not a claim that the author failed to transcribe it.

The pooled corrective set has 553 transitions in 553 episodes. The specific multiple-observations
per-episode objection does not inflate that pooled count. Its mandatory subgroups have 244 and 309
transitions, so the pooled 0.11 assurance cannot simply be attached to each subgroup.

### P5 — make every claimed held-out prediction actually held out by episode

Draft §4.2 fits ordinary transitions on an episode-held-out split of training episodes, while §7
counts **all 4,122** ordinary transitions as held out and §4.2 evaluates all 553 corrective
transitions as out of sample. A single fit/holdout partition cannot make the full ordinary census
held out. Excluding corrective rows alone also does not exclude the ordinary rows of their episodes.

Specify episode-level cross-fitting if every transition is to receive an out-of-fold prediction,
or declare a single held-out episode set and its smaller evaluation count. Freeze the folds,
family stratification and seeds before the seal, and apply the exclusion to every learned stage
including the representation probe and transport rule. Explain how shared training across folds
is handled in the uncertainty calculation. This resolves the ambiguity without pretending the
existing prose proves leakage occurred.

### P6 — label the capture-set digest's scope

The named digest `1a7cfbdd…9709a` correctly reproduces the serialized ordered triples
`[task_id, step, prompt_sha256]`. It is a logical-set digest, not the file's byte digest.
The captured JSONL's SHA-256 is `98c78f16…731a4d`, reproduced in `analysis.json`. Family, recovery,
variant and multiplicity lie outside the former hash. Keep the logical digest with its serialization
schema and add the full byte hash; this is a provenance clarification, not evidence of corruption.

## 4. What carries correctly, and what this review did not run

The draft carries the revised distinct-decision unit, exclusion of chat replay, recovery
oversampling correction, gap-aware transition census, task-provided horizon, R2 decodability and
nulls, within-pass H2 with the cache-exchange dependency, fixed rank/depth ladder and headline,
twelve exploratory episode IDs chosen by rule, corpus archive digest and native-precision capture.
The fitted update rule's ordinary/corrective distinction is stated, and the two corrective groups
are separately enumerated. The 7,629 distinct decisions, 1,128 episodes and 6,501 transitions
reproduce from the frozen metadata. The existing producer record states the device count rerun
passed; this review did not repeat that device run or inspect the original corpus rows.

No model was loaded. No capture was read for plan-progress outcomes. No fit, precision control,
new derivative measurement, 12B smoke row, or performance measurement was performed here. The
source snapshot is append-only; requested changes are handed to its owners, not applied to their
records by this review.

## Reproduction and verification

Run `python3 analyze.py` to regenerate `analysis.json`, then `python3 analyze.py --check` to compare
it exactly and prove every source hash check refuses corrupted bytes. Only the Python standard
library is required. The saved corpus file contains metadata and hashes, not prompt text. The
snapshotted producer scripts are evidence only and must not be executed to reproduce this review.
The analysis also inspects the reporter expression as syntax, without importing it, and evaluates
both Boolean possibilities of its trailing-false form. See `VERIFICATION.json` for the final
file-only verification outcome.
