# Gemma is certain under a false premise, and uncertain without one — and loops either way

> **SCOPE CORRECTION, later the same day, at the top rather than at the bottom.** The finding below
> was measured under the **lying simulator**, and it is a property of that regime rather than of the
> model. The D-CRO has since read the same episode at all 34 layers under the **fixed** simulator
> (`cac8532`), and the result inverts: the model is *uncertain*, holds both wrong options
> simultaneously, considers the escape route every single turn, and loops anyway. Read this section
> as "what a false premise does to the distribution", not as "how Gemma behaves". The inversion is
> recorded at the end and it is the more important result.

## The original finding, scoped: under a false premise, certainty is total and immovable

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

---

## The Director's correction: the output distribution cannot test a claim about representations

The D-CRO proposed that unwarranted certainty about a path and unwarranted certainty about being
finished are one phenomenon at two scales. I tested it by reading the model's probability of each
emitted token at layer 34, found identical confidence when wrong and when right, and then killed my
own test with a control showing the median is saturated across every token in every episode.

**The Director's objection goes a level deeper than my control did, and it is the more serious one:**

> *Idk how you'd confirm that without reading activations under the actual condition.*

He is right. Layer 34 **is the identity** — it is the model's own output distribution, not a
representation. So the statistic I chose was not merely saturated; **it was the wrong kind of object
for the claim.** A hypothesis that two behaviours share an internal mechanism cannot be confirmed by
comparing the outputs those mechanisms produce, however the comparison is scaled or de-saturated.
Fixing the saturation would have produced a better measurement of the wrong thing.

This is the failure the D-CRO diagnosed in my work weeks ago and named exactly: *reaching for a
property of the artefact when the question is about the thing the artefact represents.* The control
caught the symptom. It did not catch that the instrument was pointed at the wrong object, and it
could not have.

### What a test with power actually requires

Read the residual under both conditions and compare **where in depth the commitment happens**, which
is a property of the representation and not of the output.

We already have the shape for one of the two conditions. At the fork, the wrong token is rank 1 at
layer 24 at all 24 forks and rank 6 to 20 at layer 23, against an argument-start control whose
median commitment is layer 30 with 1.6% at layer 24. So the path condition has a signature: **an
unusually early and unusually sharp commitment, localised to one layer step.**

The test is whether a wrong self-terminating `finish` shows the same signature. Same measurement,
same instrument, different condition:

1. Locate the divergence position — the first token where the model's answer departs from the
   expert's.
2. Take the layer at which the emitted token first reaches rank 1, at that position.
3. Compare against the same argument-start control, and against the fork's 23-to-24 step.

**A shared signature is evidence for one phenomenon. Different depths are evidence against.** Either
outcome is informative, and neither can be obtained from the output distribution at any scaling.

**Stage two produces exactly this**, at all 34 layers over 15 episodes, and the wrong-answer
termination class is now the largest we have at nine of eighteen. The test needs no additional run.

### And the probability version is not recoverable from these records at all

Worth recording so nobody designs around it. `reading` rows store the top-k as **token ids only**;
`probability` exists solely on `rank` rows, and those exist only for tokens the model actually
emitted. So the probability assigned to a *correct* answer the model did not emit is unavailable
whether or not it sits in the top ten — membership buys a rank, never a value. The D-CRO measured the
rank version: the expert's token is inside the layer-34 top ten at the divergence in 6 of 7 failed
episodes, so a rank proxy has data. It is still a proxy for confidence, with the same weakness one
level down, and it is still an output-side quantity, so the Director's objection applies to it too.

**If the probability is wanted, it comes from a targeted replay — and a replay costs box time.**

*Corrected by the Director, who asked whether forward passes are actually recorded.* I had written
that the quantity is "recomputable from the records" at "no extra box time", repeating the D-CRO's
phrasing without checking it. **That is wrong.** What a `forward` row stores is
`['argmax', 'final_readout_max_abs_error', 'input_ids', 'kind', 'logits_sha256', 'logits_shape', 'offset', 'turn']`.
The logits themselves are **not** stored — a distribution over 262,208 tokens is 1.0 MB per forward
and only its 64-character SHA-256 is kept. No KV cache is stored either. **A hash lets a replay prove
itself; it does not let anyone skip the compute.**

What *is* recoverable without the model is the exact input: `begin_turn` carries `prompt_ids` and the
full `context`, and each forward carries its `input_ids`. So the replay does not have to regenerate a
trajectory — it can feed the recorded ids directly.

**The honest cost.** One model load, the run lock, an announced window, and a forward over the prefix
at roughly nine divergence positions. Minutes of compute, not hours, and far cheaper than
regenerating fifteen episodes with lens capture. But it is box time and it must be scheduled, not
waved through as free.

**What the hash buys is verification, and that is worth having on its own.** The replay reproduces the
recorded `logits_sha256` or it is not the same computation, so the measurement is checked against a
known answer before any new number is read — the mechanism this programme has spent the day learning
to insist on. That was the defensible half of my claim. The free half was not.

---

# The inversion: told the truth, the model is uncertain, holds both wrong options, and still loops

**D-CRO, `cac8532`, `update-0028` at all 34 layers under the fixed simulator.** This is the more
important half of the record and it overturns the framing above.

**A free validation first.** All 24 actions are byte-identical to the no-capture falsification run,
so capture does not perturb greedy generation. Measured, not assumed.

**The alternation is not a symmetric two-cycle.** Read at depth the two branches are different
decisions:

| branch | rank 1 first at | rank at layer 23 | P(final) median | P(final) min |
|---|---:|---:|---:|---:|
| `"."` | layer 24 | 8 | 0.9988 | 0.81 |
| `"/"` | **layer 33** | 108 | **0.9031** | 0.85 |

One branch is a layer-24 commitment. The other is decided in the last two layers and carries a tenth
of its mass elsewhere.

**Both branches are live at every fork.** At every `"/"` fork, `"."` is in the top ten from layer 23
or 24 onward; at every `"."` fork, `"/"` from 24 to 27 onward. All 24 turns. **The alternative is
represented at the moment of choosing, and the model takes the one it did not take last time.**

**And the escape route is represented, late, and rejected.** At the tool-name decision `search` is a
top-10 candidate somewhere in the stack in **all 24 turns** — at layer 33 in 18 of them, at 29 to 30
in 2, as shallow as 24 in 3, never below 24. The model considers searching in its last one or two
layers, every turn, and does not do it.

## What this does to the finding above

**"Certain, and the errors never reach its certainty" was a property of the false-premise regime.**
Under the lie the wrong opener sat at ≥0.9999969 at 22 of 24 forks and `workspace` was absent from
all 192 top-10 lists. Under the truth the same task produces 0.9031 median on one branch with both
alternatives live. **Twenty-three refutations do move this distribution. They simply do not move it
toward the answer.** That is a better sentence than mine and a different phenomenon.

**One caveat the inversion needs, which does not weaken it.** The two measurements are not the same
decision. Under the lie I measured the first token of a `read_file` *path* argument; under the truth
the D-CRO measured the first token of a `list_files` *directory* argument. Both are call-argument
first-token decisions, so they are the same *class*, but they are different instances with different
tools. The regime is the most plausible explanation for the difference and it is not the only one,
and a clean version would compare the same decision under both simulators. That comparison is
available — the lying run's own `list_files("/")` at turn 0 — and has not been made.

## What it does to the intervention design

**The target has moved and improved.** It was "the alternative is not represented anywhere in the
stack", which is nearly impossible to intervene on: there is nothing to point a donor difference at.
It is now **"the alternative is represented and not selected"**, and `search` at layers 24 to 33 at
the tool-name decision *exists in the residual*. The donor-difference design from the section above
now has a concrete direction and a concrete layer band to work in.

**The confidence stratification matters more, not less.** The `"/"` branch at 0.90 is soft
everywhere; the `"."` branch at 0.999 is hard; and they alternate turn by turn. So an experiment that
does not stratify would sample a mixture of soft and hard targets that alternates with turn parity —
which is worse than the saturation problem the rule was written for.

## And one thing for the primary comparison

This episode contributes **24 call-argument tokens, one per turn, each a single character**. The
restated primary comparison has almost nothing to read here. Worth knowing before the equal-weight
aggregate treats it as one full episode alongside episodes carrying fifty times the argument content:
equal weighting protects against a long episode dominating, and it does not make a thin episode
informative.
