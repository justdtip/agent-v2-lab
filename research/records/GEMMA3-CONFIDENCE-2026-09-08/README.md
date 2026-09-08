# Gemma is not uncertain and wrong. It is certain, and stays certain, through twenty-three refutations

**2026-09-08, Chief.** The first result of this programme that is about Gemma's internal state rather
than about our instruments. It cost no box time: it is arithmetic over a record already on disk.

## What was read

In `update-0028` under the corrected rendering, the model emits the wrong path-opening token `' "/'`
(id 9560) at a fork in each of 24 turns, receives an unambiguous `ERROR: file not found` each time,
and emits it again. The record stores, at layer 34, **the model's own softmax** rather than a lens
read, because the final layer's readout is the identity. So the model's own probability of the token
it is about to emit is recoverable at every fork with no model load and no lens involved.

| turn | position | rank | P(wrong token) |
|---:|---:|---:|---:|
| 0 | 555 | 1 | 0.9998727 |
| 1 | 637 | 1 | **0.9890071** |
| 2 | 728 | 1 | 0.9999998 |
| 3 | 835 | 1 | 0.9999969 |
| … | | | |
| 21 | 2545 | 1 | 0.9999999 |
| 22 | 2640 | 1 | 0.9999999 |
| 23 | 2735 | 1 | **1.0000000** |

Twenty-four of twenty-four at rank 1. The lowest value in the whole episode is **0.989**, at turn 1.
At 22 of 24 forks it is at or above **0.9999969**, and at the last fork it saturates float32.

## What it means

**Twenty-three unambiguous refutations do not move the distribution.** The register's sharpest
sentence so far was *"the model does not switch strategy under repeated unambiguous failure"*, and
that sentence was compatible with three mechanically different stories: a flat distribution over
several candidates, a self-reinforcing one, or one moving in the right direction too slowly to matter.
It is none of the interesting ones. **The model is not uncertain and picking wrong. It is certain, and
the errors do not touch its certainty.**

**"Present but beaten" understated the correct alternative by orders of magnitude.** The correct
opener `' "'` sits at rank 2 at layer 34 at 23 of 24 forks. At the late forks the winner is at
1 − 10⁻⁷, so the correct branch is behind by up to seven orders of magnitude. Reporting it as "rank 2"
made it sound like a near miss. It is not a near miss.

**And it changes how any intervention must be designed.** Fork 1, at P = 0.989, is the only soft
target in the episode; forks 15 to 23 are saturated. An intervention that flips fork 1 and reports a
success has measured the one fork where a flip was cheap. **Any causal experiment on this fixed point
stratifies by the model's own confidence or it is not a result.** That constraint did not exist
before this reading and would have been discovered afterwards, expensively.

## A correction, made before publishing rather than after

The agent that first computed this series described it as hardening **monotonically** after the first
two turns. It does not: turn 2 is 0.9999998 and turn 3 is 0.9999969, and there are several such
reversals. Checked explicitly and the monotonicity assertion returns **False**.

The finding does not need monotonicity and is stronger without it. The claim is **saturation**, not a
trend: at 22 of 24 forks the model is above 0.9999969 and it never falls below 0.989. A trend claim
would have invited a slope, a test and a confound; the saturation claim needs none of them.

That is the fifth clean-looking number today that changed on the smallest possible check, and it is
the first one caught in someone else's work before it entered a record rather than after. See
[`../METHOD-2026-09-08/README.md`](../METHOD-2026-09-08/README.md).

## What this does not establish

It is one episode and one fixed point, which is one observation repeated, not twenty-four. It is
under 4-bit weights. It says nothing about whether layers 23 and 24 *cause* the commitment — the
final-layer probability is the model's output, so this is a statement about what the model concludes,
not about where or how it concludes it. That question needs an intervention, and we have not run one
on Gemma's internals.

---

## What this fixes in the intervention design, before one is ordered

The D-CRO raised the right objection to the injection seam: it is additive, single-layer,
single-position, so it steers rather than ablates, and against a distribution at 1 − 10⁻⁷ the
magnitude needed may be large enough that the arm stops being a perturbation. That is correct and it
is quantifiable rather than a matter of taste.

**How much work each fork actually demands.** The logit gap an intervention must close is
`ln(p / (1-p))`:

| fork | P(wrong) | gap to close |
|---:|---:|---:|
| 1 — the softest | 0.9890071 | **4.50 nats** |
| 0 | 0.9998727 | 8.97 nats |
| 3 | 0.9999969 | 12.68 nats |
| 2 | 0.9999998 | 15.42 nats |
| 23 — saturated | ≥ 0.9999999 | **16.12 nats** |

**A fixed-magnitude steering arm is therefore not one experiment. It is a perturbation at fork 1 and
a sledgehammer at fork 23, differing by 3.6x in the work required**, and a design that applies one
magnitude across forks would report a flip rate that is a map of the confidence profile rather than
of anything causal.

### Two design rules, fixed now

**1. The dependent variable is the minimum perturbation that flips, not whether a fixed one does.**
Report, per fork, the smallest multiple of the injected direction that changes the emitted token. A
binary flip rate under a fixed magnitude confounds the intervention with the saturation; a threshold
is graded, is comparable across forks, and its *shape against the confidence profile* is the actual
finding. If the required magnitude tracks the logit gap exactly, the intervention is doing nothing
the output distribution does not already say. If it departs from that curve, the departure is the
result.

**2. The perturbation is a donor difference, not a synthetic direction.** The seam takes an arbitrary
delta at a position, so `delta = h_donor − h_target` is expressible, which makes it activation
patching in additive clothing. Using the residual from another fork — or from a run where the model
emitted the correct opener — keeps the perturbation inside the distribution the model actually
produces. **A synthetic steering vector large enough to close 16 nats is not a statement about the
representation; it is a statement about how hard you pushed.** A donor difference is scaled in units
the model itself generates, which is what makes the threshold in rule 1 interpretable.

**What the seam still cannot do.** It cannot ablate, so it tests sufficiency and never necessity: it
can show the fixed point is movable and not what the representation was carrying. That limit is
structural, it is not fixed by either rule above, and any record from this arm states it rather than
letting "we changed the output" read as "we found the cause".
