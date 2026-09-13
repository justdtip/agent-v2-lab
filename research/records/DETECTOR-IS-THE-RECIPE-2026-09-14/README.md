# The trained detector fires on a direction containing no concept at all

2026-09-14, WS-D. Gemma 4 31B with the introspection adapter, evaluated at damage matched per arm
against the loaded model.

## The result

| arm | −0.03 | −0.08 | −0.20 |
|---|---|---|---|
| held-out concept | 100% | 100% | 100% |
| trained concept | 100% | 100% | 100% |
| spectrum noise, bank mean REMOVED | 15% | 12% | 8% |
| **spectrum noise, bank mean RESTORED** | **100%** | **98%** | **98%** |
| **the bank mean alone, no concept** | **100%** | **100%** | **60%** |
| nothing injected | 0% | — | — |
| factual control | 0% | 0% | 0% |

At the −0.08 tier the concept arm and the mean-matched null are matched on damage to within 2 per
cent, with a between-arm spread of 7 per cent. So this is like for like: **a random direction
carrying no concept, at the same cost to the forward pass, gets the same answer as a real concept.**

## What was being detected

Every concept vector is a carrier residual minus a baseline mean, and the bank's own mean is 70 to
76 per cent of a typical concept's norm (BANK-GEOMETRY-2026-09-13). Our null generator draws from
the *centred* covariance, so it has no component along that mean and a concept has a large one.

The separation is not merely large, it is exact. `drawn` lies in rowspace(bank − mu) by
construction, so for the unit vector w in span(bank) orthogonal to that rowspace, ⟨v_i, w⟩ = ‖P·mu‖
for every concept including held-out ones, and ⟨drawn, w⟩ = 0 identically. One scalar, available to
a linear readout, never looking at which concept is present, separates the arms perfectly and
survives the bf16 cast with a margin of about 1e4.

The adapter found it. That is not a failure of training; it is the cheapest available solution to
the objective we wrote.

## Read together with the naming result

The same adapter, asked which concept is present, ranks the true one 129.9 of 240 against 130.8 for
no injection at all, and answers "walnuts" for all 137 concepts it was shown. So:

- it detects the presence of the generator's shared direction, reliably;
- it carries no information about which concept that direction belongs to;
- the headline 100 per cent against 12 per cent needs no concept content to explain, and now has an
  explanation that does not use any.

## What this closes and what it does not

**Closed:** this design cannot demonstrate introspective access to concept content. Holding out
concepts does not hold out the generator, and the generator's signature is a single linear
direction the model can read without introspecting at all.

**Not closed:** the untrained model's zero remains a clean null result on its own terms — 0 of 813
cells across two harnesses, with a correct prompt, a correct scorer and a batch-matched meter. And
nothing here bears on whether larger models do this; the capacity is reported as emergent, and a
31B is below where it has been demonstrated.

**Pre-committed and honoured.** `INTROSPECTION-TRAINING-PLAN-2026-09-12` named this outcome as the
likely one and said it should be treated as a result rather than a failure: "Guard explicitly
against the adapter learning ... the statistical signature of how our vectors were built rather
than what they mean." The guard was the right one and it fired.

## What a future attempt would have to do

Hold out the **generator**, not the concept. Any experiment whose positives all come from one
recipe is measuring recipe recognition until proven otherwise, and the proof costs one arm: inject
a direction built a different way and see whether the detector still fires.

Three defects in this run, from the review, all fixed at HEAD and none of them able to change the
conclusion: the common-mode arm emitted one measurement per layer as though it were twenty rows;
the two noise arms drew independent coefficients rather than differing only by the mean; and the
comment in `vectors.py` quoted a synthetic AUC as though it had been measured on a real bank.
