# Why did restricting the adapter to the top eight layers turn a losing model into a winning one?

**In progress, 2026-09-08.** The Director's question, and his hypothesis: *training every single
layer to do agent stuff makes the model forget other items.* This record separates what was
measured, the technique that measured it, and our implementation, per the deliverable rule.

## The result being explained

On the same 180 held-out tasks, greedy, same base model, same seed, same prompts:

| | passes of 180 |
|---|---|
| base, no adapter | 123 |
| arm A, LoRA on all 32 layers | not run at 180; **5 of 19** on the divergence subset where base scores 15 |
| arm 1, LoRA on the top 8 layers | **159** |

Full depth produced a model worse than its own base. The top eight produced one that beats it by
thirty-nine tasks to three. The recipes differ in two lines: the output directory and the adapted
depth.

---

## Findings so far, none of which needed the machine

### 1. Full-depth training moved every layer by the same amount

Relative perturbation `||dW_layer|| / ||W_layer||` for arm A at its evaluated checkpoint is
**0.199 to 0.246 across all thirty-two layers, mean 0.227**, with no depth structure and no
difference between the attention blocks and the recurrent ones. The lowest twenty-four layers
moved slightly *more* on average (0.234) than the top eight (0.208).

**A uniform-rank, uniform-learning-rate LoRA distributes its change uniformly across depth. It
does not discover which layers the task needs.** Every layer moved by about a fifth of its own
weight norm whether or not it had anything task-specific to learn.

Arm 1 moved its eight layers **less** than arm A moved the same eight: 0.154 at matched
iterations and 0.167 at its evaluated checkpoint, against arm A's 0.208. Fewer trainable
parameters did not force harder movement; it reached a lower validation loss with a smaller
perturbation.

### 2. The two arms did not learn the same thing in the layers they share

Cosine between the two adapters' weight deltas at the same layer and module:

| | cosine |
|---|---|
| arm A against arm 1, matched iterations | +0.0019 |
| arm A against arm 1, each at its evaluated checkpoint | +0.0021 |
| **control**: arm 1 against its own later checkpoint | **+0.9245** |
| **null**: independent random deltas of the same shapes | +0.0008, sd 0.0033 |

The measured similarity is within one standard deviation of independent random matrices, and the
control shows the measurement detects similarity when it is there. **In weight space the two
solutions share nothing.** Full-depth training did not learn arm 1's solution and additionally
damage the lower layers; it found an entirely different one.

### 3. The change is diffuse in both, so that is not the difference

Effective rank of the delta is about 12 of a possible 16 in both arms. Neither concentrated its
change into a few directions.

---

## What this does not yet establish, and the experiment that would

**Weight-space orthogonality is weak evidence of functional difference.** Two rank-16 subspaces in
2,560 dimensions are nearly orthogonal by dimension counting alone, so two adapters could
implement similar functions through unrelated weight directions. The cosine result rules out the
*simplest* form of the forgetting story — that arm A learned arm 1's solution plus damage — but it
does not tell us where arm A's capability went.

**The decisive experiment is a post-hoc depth ablation of the trained adapter**, and it costs
minutes. Take arm A's adapter and zero every tensor outside a chosen depth band, changing nothing
else, then evaluate. Two slices are built and verified in `outputs/depth-ablation/`:

- `armA-top8`: arm A's layers 24-31 exactly, layers 0-23 exactly zero.
- `armA-bottom24`: arm A's layers 0-23 exactly, layers 24-31 exactly zero.

Read against base 15 and arm 1 18 on the divergence nineteen:

- **`armA-top8` scores well** → arm A's capability loss lives entirely in the lower layers, and
  the Director's hypothesis is confirmed in its precise form.
- **`armA-top8` scores badly** → arm A's top layers are themselves the problem, co-adapted to
  lower layers that its own training moved, and the story is not forgetting but a solution that
  cannot be decomposed.
- **`armA-bottom24` scores badly on its own** → the lower-layer change is harmful by itself.

---

## The technique, stated without reference to our code

Given two parameter-efficient adapters trained on the same data from the same base, differing in
which layers they were allowed to change:

1. **Per-layer relative perturbation.** Aggregate the delta norms and the base norms separately in
   quadrature over each layer's adapted modules, then divide. Do not average per-module ratios and
   do not sum them in quadrature; the latter inflates by roughly the square root of the module
   count, which was this author's first error here and is easy to make.
2. **Cross-adapter cosine in weight space**, computed inside the low-rank core so no full delta is
   ever formed, reported only alongside two references: a null from independent random deltas of
   the same shapes, and a control from one adapter against its own earlier checkpoint. A cosine
   without both is uninterpretable.
3. **Post-hoc depth ablation.** Zero the trained adapter outside a depth band and evaluate. This
   separates *where the damage is* from *what was learned*, needs no retraining, and is the only
   one of the three that measures function rather than geometry.

All three are architecture-independent and method-independent: they need a base checkpoint, an
adapter expressed as per-module deltas, and an evaluation. Nothing above is specific to this model
family, to LoRA's particular factorisation, or to our pipeline.

**Read the library for the scaling convention rather than assuming it.** The multiplier here is
the recorded scale itself, not scale divided by rank, which the fusion routine settles in one
line. Getting this wrong scales every number by the rank.

---

## Implementation

`adapter_geometry.py` beside this file, which loads no model and runs beside a job. The slices were
built by zeroing tensors outside the band and copying the adapter configuration unchanged, so the
loader sees the same structure and a zero delta is exactly no change.

---

## A third reading, weaker than the first two, with one sharp consequence

Both arms' updates were read in token space by taking the leading left singular vectors of the
delta for the modules that write into the residual stream, scaling by the learned final norm and
projecting through the tied unembedding. This asks what a trained update *can* write, not what it
does write on any input, so it is suggestive rather than evidential.

At layer 31 both arms' leading directions land on the protocol's own vocabulary. Arm A's top
direction reads `update / Update / 更新 / {"`; arm 1's second reads ` ``` / approved / values /
update`. At layer 27 both have a direction reading `plan / Plan / planning`. Several other
directions are multilingual noise, which is what a delta of effective rank twelve should give:
the leading direction carries only ten to fifteen per cent of the mass, so reading the top one is
weak by construction.

**The sharp consequence is methodological, and it changes how the orthogonality result above
should be read.** Two updates whose weight-space cosine is indistinguishable from random
nonetheless write toward the same token families. A 2,560-dimensional residual read through a
248,320-token unembedding is many-to-one, so orthogonal weight directions can carry the same
function. **Weight-space orthogonality is therefore not evidence of functional difference**, and
the earlier finding must be stated as what it is: the two solutions share no weight-space
structure, which rules out arm A having learned arm 1's solution *by the same route*, and says
nothing about whether they compute the same thing.

It also yields a prediction worth writing down before the ablation runs: **if arm A's top layers
learned protocol-relevant directions much like arm 1's, arm A's top-eight slice should score
well**, and its capability loss should be attributable to the layers below. That is the
Director's hypothesis in falsifiable form, and the ablation tests it.
