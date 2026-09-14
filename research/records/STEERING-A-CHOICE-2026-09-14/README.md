# A concept direction moves a forced choice, and a recipe-matched null does not

2026-09-14, WS-D. Gemma 4 31B, base checkpoint, no adapter. 117 complete trials per layer.

## The setup, and why it has no room for the confounds that ate the introspection work

The model is asked to pick one of two topics to write a sentence about, and answers with a number.
Nothing in the prompt or the system turn mentions injection, activations or introspection; it is
never asked about itself. The answer is read from the logits at the answer position, so every trial
gives a continuous log-odds rather than a coin flip.

The pair comes from the bank's own geometry: the NEAR option is the vocabulary word whose direction
has the highest cosine with the injected concept, mean removed, and the FAR option is drawn from
the forty least aligned. Mean cosine of the near option, 0.797.

Each pair is asked in **both orders** and averaged, because a model that prefers whichever option
is listed first would otherwise produce exactly the signature of a steering effect. Every arm scores
the **same pair**, so the statistic is paired; the pair-to-pair spread is about 6 log-odds against
an effect of order 1, and an unpaired difference of means carries a standard error forty times too
large to see it.

The null is **mean-matched** spectrum noise: same covariance, same norm, and the shared-direction
scalar that our older null leaves free is neutralised. This is the null the trained introspection
adapter could not tell from a real concept, so it carries the recipe's signature and not the
concept's content.

## The result, at −0.45 nats, injected at every prompt position

| layer | concept − clean | t | trials moving the right way | null − clean | t |
|---|---|---|---|---|---|
| 20 | +2.724 ± 0.636 | 4.28 | 86 / 117 | −0.081 | −0.13 |
| 32 | +3.537 ± 0.410 | 8.63 | 92 / 117 | −0.181 | −0.39 |
| 40 | +0.822 ± 0.097 | 8.50 | 88 / 117 | −0.057 | −0.65 |
| 48 | −0.280 ± 0.166 | −1.69 | 53 / 117 | −0.221 | −1.42 |

Three quarters of individual trials move toward the semantically nearer topic at layers 20, 32 and
40, and the null moves nothing at any layer. The "117" is not 120 minus rounding: the pairing key
is (concept, layer, near, far) and `near` is determined by the other two, so trials that draw the
same (concept, layer, far) cell collide and one is silently lost. Expected distinct cells for 120
draws from 2,080 is 116.6, which is what we are seeing. The loss is unbiased but it was not
declared, and the key now carries the trial index. At layer 48, four fifths of the way through the stack,
the effect is gone and the two arms are indistinguishable.

## Scope is the whole difference

The identical experiment injecting at ONE token — the last of the prompt — is a flat null across
the entire strength ladder from −0.03 to −1.2 nats, every shift within 0.01 of zero with t below
0.21. Training, the introspection evaluation and the first choice run all used that scope. The
prior work sustains, and this project's own base-model grid injected at every prompt token.

So: **a single-token injection does not steer this decision at any strength, and a sustained one
does at three of four layers.** Every null this programme has reported was a statement about the
scope it happened to use, and nobody had said so.

## What this does NOT establish

**The damage label is not the damage.** The strength was calibrated by the meter, which injects at
one site, and then applied at roughly two hundred positions. Measured 2026-09-14, that costs
between 1.3 and 11.4 times the label depending on layer and strength: at layer 20 the −0.45 tier
really costs −3.46 nats and the −1.2 tier costs −11.8, while at layer 40 the same tiers cost −0.79
and −2.15. So the large shifts at layers 20 and 32 were measured under a far heavier hand than the
number says, and layer 40 is the one place the label is nearly honest.

**CORRECTION, 2026-09-14.** An earlier version of this paragraph said "the concept and the null
share the scale and the scope exactly, so the contrast stands". The second half is true and the
first is false. They share the TIER, not the scale: `steer_choice.py` solves a scale for the
concept and a separate one for the null, and they cannot share a scale, because `spectrum_matched`
norm-matches the arms and the solver divides by the vector's norm, so matching on damage is
precisely what forces the two magnitudes apart. Each solve also clamps independently and only the
concept's clamp note was recorded, so a divergence was undiagnosable from the output file. What the
arms genuinely share is the scope and the tier they were solved for. Whether they are matched on
delivered damage at that scope was not checked in this run and is now measured and printed per arm.

**The injection disrupts as well as tilts.** Position bias — the preference for whichever option is
listed first — nearly doubles under injection, 2.58 against 1.49 clean. Asking both orders removes
it from the mean. It is still evidence that the choice mechanism is being perturbed and not only
biased, and a disruption reported as a preference would be a real error.

**It is a preference, not an action.** Choosing which of two topics to write about is a decision,
but a light one with no consequence. Whether a concept can move a choice that costs the model
something is a different and harder question.

**The magnitudes are large against a very variable baseline.** 3.5 log-odds at layer 32 is a
thirty-fold odds shift, but the pair-to-pair spread of the clean reading is about 6, so this is a
shift inside a noisy quantity rather than a clean flip.
