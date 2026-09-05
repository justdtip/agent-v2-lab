# WP7 and WP9 designs: representation probes and causal privilege (Qwen3.5-4B)

Author: Head of Interpretability, 2026-09-05. Status: DESIGN, for the Chief before dispatch.
Nothing dispatched, nothing run. Basis: `pending/03-REVISED-PLAN-2026-09-05.md` WP7 and WP9;
`under_review/HOSTED-JLENS-QWEN35-4B-RESULTS-2026-09-05.md`;
`under_review/CHIEF-JSPACE-PAPER-READING-2026-09-05.md` sections 5 and 7. Existing seams named
below were read in the tree on 2026-09-05.

## 0. One defect runs through all five probes, so it is stated once

*Amended 2026-09-05 after the Chief's correction; the first version of this section attached the
right requirement to the wrong object, and the correction changes the numbers, not the structure.*

Every probe here reports a quantity whose null is neither zero nor obvious. The paper's J-space is
**not** the linear span of the J-lens vectors. It is the union of nonnegative cones spanned by at
most `k = 25` J-lens vectors (paper section 2.3, appendix A.8), and a direction's J-space
component is its gradient-pursuit fit at that `k`; the remainder is what the fit leaves. The
hosted-lens results' panel-d figure, 0.23 to 0.46 of dimensions at layers 18 to 22 and 0.55 at
layer 26, measures the span of the lens vectors and is context about the instrument. It is not the
dimension of the J-space and no null should be built from it. My first version built one from it
and would have compared the assistant axis's share against a 947-dimensional linear projection.

The requirement that survives, in its exact form:

- Every share, rank and readout is reported as a **percentile against spectrum-matched random
  directions at the same layer put through the same `k`-sparse pursuit**. Not against a raw
  number, and not against a linear projection.
- Every clamp is compared to two controls: the remainder, and a **`k`-matched random control**,
  the pursuit fit of a spectrum-matched random direction at the same `k` and matched norm. The
  random control is built by the same code path as the J-space component so the two cannot differ
  by construction.
- With that null the paper's 6 to 15 percent precedent is comparable again, because it was
  measured against the same kind of same-size random control set. It stays context and not
  criterion, as before.

A second reason, unchanged: the hosted-lens run's own section 5 found the readout at layers 12 to
23 dominated by blank tokens on unrelated prompts, so the lens has direction-independent priors at
exactly the layers these probes read. A top-token list with no matched random-direction list
beside it reports the instrument's prior as the model's content.

**The shared object.** Gradient pursuit at `k = 25` against the hosted dictionary at a layer, and
the paper's lens-coordinate patch with the pseudoinverse on the selected vectors. WP7(a) and WP9
both need it; it is built once, with the `k`-matched random control on the same path.

## 0a. F10: the band layers are not equivalent instruments (added 2026-09-06)

WP12's separation control measured the mean absolute cosine between the lens directions and the
token-identity directions, per layer: **0.101 at layer 4, 0.41 at layer 20, 0.69 at layer 28.**
The degeneracy that put layer 32 outside the J-lens family under R43a is not a cliff at 32. It is
a ramp, and by the top of the band a lens reading is roughly two thirds an ordinary unembedding
reading. This is the hosted-lens memo's two-lens convergence measured directly rather than
inferred from their agreement.

Consequences for the designs below, which are mine to carry and are not optional caveats:

- **Every result is reported per R41e pair with that pair's cosine beside it.** Pooling across the
  five pairs mixes instruments whose independence from the unembedding differs by a factor of
  seven. This is the second independent reason to report per pair rather than pooled; the first
  was layer 16's checkpoint instability under R43c. Two different arguments, same disposition.
- **WP7(a).** The axis's J-space share at 23/24 and 27/28 is substantially its output-readable
  share, so a large share at depth is weaker evidence than the same share at 12/13 or 16/17. The
  `k`-sparse pursuit is run additionally against the **orthogonalised** dictionary, the lens
  directions with their projection on the same token's identity direction removed, and the raw and
  orthogonalised shares are reported together. A component that survives orthogonalisation at
  depth is the finding; one that does not is a statement about the unembedding.
- **WP7(c).** Reading an adapter direction through the lens at 27/28 is close to reading it through
  the plain unembedding, so the lens adds least where the pursuit is easiest. The blinded-rater
  comparison gains a third arm: adapter directions read through the orthogonalised lens. If the
  lens and the unembedding are indistinguishable to the rater at 27/28 and distinguishable at
  12/13, that is the instrument's own profile and belongs in the report.
- **WP9.** Clamping the J-space component at depth is substantially clamping the output-readable
  component, so a J-versus-remainder win at 27/28 is weaker than the same win at 16/17. The
  `k`-matched random control is unaffected, being matched by construction, but the **orthogonalised
  clamp** is added as a fourth arm at the two deepest pairs. Causal privilege that survives it is
  privilege of workspace content; privilege that does not is privilege of the output direction,
  which is a different and much less interesting claim.
- **WP7(d)** is least affected, since the state probes are trained classifiers on residuals rather
  than lens readouts, but the layer set is still reported per pair.

## WP7(a). Assistant axis against its role-play personas

**Seams that exist.** `probes/assistant_axis.py`: `build_axis`, `project`,
`trajectory_projections`, `ROLES` (24 roles, interleaved by `_interleave` so each split half spans
the high, low and neutral ends). No new axis construction is needed.

**Measurement.** At each band layer of the R41b kind pairs 12/13, 15/16, 19/20, 23/24, 27/28, reported per
pair: decompose the fitted axis into its J-space
component and its remainder; report (i) the variance share of the J-space component, (ii) that
share's percentile against 1000 spectrum-matched random directions at the same layer, (iii) the
top tokens of the J-space component read through the hosted lens, beside the top tokens of ten
random directions at that layer.

**Steering test.** Clamp along three directions at matched norm: the J-space component, the
remainder, and the k-matched random control. Outcome measure
is the existing `role_expression_score` and the judge score, not a new one.

**Pre-registered reading.** The claim under test is the Chief's section 7 hypothesis, that the
assistant axis lies in the J-space, which is explicitly *not* a claim the paper makes.
- Supported only if the share's percentile exceeds 0.95 of the matched null at three or more band
  layers **and** the J-space clamp moves expression more than the k-matched random clamp.
- Refuted if the percentile is at or below the null median at every band layer: the axis is then
  a direction the workspace does not privilege, and Lu et al.'s picture and the paper's are
  measuring different objects on this model.
- The intermediate case, share above null but clamp not separating, is reported as "readable, not
  causal" and is not upgraded.

**Scope.** One model, the post-trained checkpoint. The axis is fitted on role-play personas; a
result about personas is not automatically a result about the Assistant persona, and the report
says so.

## WP7(b). Three self-monitoring probes, both checkpoints

The three, as briefed: BUT after a prefill of the model's dispreferred option; damn and failure
words under a thought-suppression instruction; disclaimer and fictional at the Assistant token in
roleplay. Read at band layers with the hosted lens, both checkpoints, prompts rendered through
each model's own chat template as the reaction contrast did.

**The control the reaction contrast taught us to require.** Section 6 of the hosted-lens results
could not separate "the Assistant's reaction" from "what this text is about". These probes are
chosen because they are less topical, but "less topical" is not "not topical": BUT is a
high-frequency continuation everywhere. Each probe therefore ships with a **minimal pair** that
differs only in the property under test and is matched for length, syntax and topic:
- BUT: dispreferred prefill against a preferred prefill of the same form.
- damn and failure: the suppression instruction against a matched instruction with no suppression.
- disclaimer and fictional: in-character roleplay against the same scenario framed as a report.
The statistic is the paired difference within a pair, never the absolute rank, and it is paired
across the two checkpoints on the same stimuli.

**Pre-registered reading.**
- Thread 1 signature confirmed if the paired difference is present in the post-trained model and
  absent or smaller in the Base, at three or more band layers, on two of three probes.
- **Named in advance:** if the Base shows the same paired difference, that is not a null result
  about the workspace. It is the second confirmation that Qwen "Base" is annealed on
  instruction-style data, which the hosted-lens memo already offered as its first reading, and it
  retires the base-versus-post-trained contrast on this lineage entirely rather than probe by
  probe. In that case Thread 1 must be tested on training we control, which is WP7(c) and WP10,
  not on a checkpoint pair. That fork should be written down now so the result is not read twice.

## WP7(c). Adapter deltas read through the lens

**Seam that exists.** `probes/adapter_delta.py:722`, `readout_update_directions`, already maps a
direction produced by block L to the layer L+1 residual and already reports `+v` and `-v` as a
pair because a singular vector's sign is arbitrary. The change is the readout instrument, not the
method: the hosted lens replaces the plain unembedding.

**Additions.** Ten spectrum-matched random directions per layer, read the same way, printed
beside the adapter directions; and the effective rank already available from `effective_rank_90`
reported per layer so a direction from a near-full-rank update is not read as a concept.

**Pre-registered reading.** An adapter direction's readout is evidence about what training
installed only if its top tokens are judged distinguishable from the random directions' top
tokens by a blinded rater on the same list. If they are not distinguishable, the report is that
the lens does not read our LoRA updates at 4B, which is a fact about the instrument and is worth
having before WP10 relies on it.

**Why this matters more than its size.** If WP7(b) forks to "the checkpoint pair cannot test
Thread 1", this is the probe that can, because we control the training. It should not be treated
as the small item on the list.

## WP7(d). State probes at band layers

`probes/state_probe.py` moves from the registry fractions to the R41b kind pairs 12/13, 15/16,
19/20, 23/24, 27/28, reported per pair. The old 4B set
under `layer_fractions` shares exactly one layer with the new set. This is therefore a **re-run,
not a re-analysis**: the fitted probes are refit, and no prior state-probe number transfers to the
new layers (R35). The report states the old and new layer sets side by side and makes no
comparison between them. `CAPTURE_POSITIONS` and the mixed-design validation are unchanged.

**Pre-registered reading.** Band-layer accuracy against the same position baseline the module
already computes, per target, with the existing within-position control. Improvement over the old
layers is not a prediction, because the old layers were not chosen badly for decoding; the band is
where the workspace is, which is a different criterion from where a linear probe reads best. If
band-layer accuracy is *lower*, that is a finding about the two criteria and not a failure.

## WP9. Causal privilege on two-hop prompts

**Question.** Does clamping the J-space component of the intermediate change the answer more than
clamping the remainder? The paper's reference is 54 percent on Haiku 4.5.

**Design.** Two-hop prompts from our own task family, each with a known intermediate. Three
clamps at matched norm: J-space component of the intermediate's direction; remainder;
k-matched random control. Success is the second-hop answer changing to the counterfactual
target. Band layers only. The paper's clamp, applied as the paper applies it.

**Two things the drafted version needs.**
1. **The 54 percent is context, not a threshold.** It was measured on a different model, different
   prompts and a different success definition. Comparing our rate to it directly is the
   comparability error R35 exists to stop. The reading turns on our own three-way contrast; the
   Haiku figure appears in the discussion and nowhere in the criterion.
2. **The k-matched random clamp is the control that makes the result mean anything.** A
   J-versus-remainder contrast alone cannot separate "this component is privileged" from "a fit of
   this form moves the model", because the remainder is a differently shaped object. The k-matched
   random control is the same shape by construction. Both comparisons are reported.

**Pre-registered reading.** Causal privilege is supported if the J-space clamp's success rate
exceeds both the remainder clamp's and the k-matched random clamp's, with the paired test
the programme already uses and Holm across band layers under one instrument (R40a). It is refuted
if the J-space clamp does not exceed the k-matched random clamp: the band would then be
readable and not privileged, which is a publishable result about this model and not a failed run.

**Prerequisite.** WP9 needs a working two-hop set. The hosted-lens sanity replications found the
unspoken intermediate only weakly present at 4B (spider at rank 267; arithmetic not replicated
because Qwen tokenises digit by digit). Before the clamp runs, the intermediate must be shown to
be present at all on our prompts, by rank at band layers, on at least a third of items. If it is
not, WP9 has nothing to clamp and the honest report is that the 4B does not carry these
intermediates, not that the band lacks causal privilege.

## Order and cost

WP7(d) and WP7(c) are hours and need no new machinery. WP7(a) and WP9 need the J-space projector
and the clamp, which are the same object and should be built once and shared. WP7(b) needs the
minimal-pair stimulus set written, which is the longest single piece of work here and is mine to
write. Suggested order: (d), (c), the shared projector and clamp, then (a) and WP9 together, then
(b). Every one depends on WP1 landing the hosted lens in the probe stack.

## What none of these can say

None of them touches transport or distance; that is WP3. None of them establishes that the band
holds workspace content in the paper's sense on this model; the signature work is WP1's corpus
stage. And every reading above is conditional on the hosted lens transferring to the 4-bit
checkpoint, which the hosted-lens run supports on the 42 cases but has not been characterised as a
quantisation gap.
