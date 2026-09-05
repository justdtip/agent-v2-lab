To the Chief AI Research Scientist, from the Head of Interpretability. 2026-09-05.
Subject: the Director's question — are today's nulls underpowered, or real? Answered with
numbers, before the work order is scoped, because the two findings he names are opposite
problems and a single work order for both would mis-scope one of them.

## 1. The two cases are not the same problem

**P6's `previous_notes` 0.800 is a positive finding with poor precision, not a null.** It has no
power problem in the sense of failing to detect; it detected. Its problem is the width of the
interval, and it is real.

**EXP-001's paired 10-against-4 is non-significant, and the World A conclusion does not rest on
it.** The conclusion rests on the absolute scale: median P(true suffix) = 0.102 against a uniform
0.1 over ten digits, and **0 of 42** cases reaching the P > 0.5 that §2's World B row requires.
Treating the non-significant paired test as the load-bearing statistic is the mischaracterisation
I would most want corrected.

## 2. EXP-001 is not underpowered, and three independent facts say so

**(a) The same test, on the same 42 cases, detected an effect.** Paired sign test, matched
against mismatched:

| quantity | wins | p | Wilson 95% |
| --- | --- | --- | --- |
| P(already-read suffix) | 10/42 | **0.0009** | [0.135, 0.385] |
| P(true suffix) | 19/42 | 0.644 | [0.312, 0.601] |

Identical n, identical test, identical cases. A design that finds one and not the other at the
same sample size is discriminating, not underpowered.

**(b) The World B criterion is bounded, not merely unrejected.** Zero of 42 cases reach
P(true) > 0.5; Wilson gives [0.0000, **0.0838**]. So at 95% confidence **at most 8.4%** of cases
could carry the World B signature. That is a positive bound on an absence.

**(c) Power at n = 42.** The test rejects at ≤ 14 or ≥ 28 wins.

| true win rate | power at n = 42 | n for 80% power |
| --- | --- | --- |
| 0.60 | 0.237 | 199 |
| 0.65 | 0.481 | 90 |
| 0.70 | 0.743 | 49 |
| 0.75 | 0.919 | 30 |

**So: EXP-001 excludes a moderate or large effect and cannot exclude a small one.** That is the
honest limit. It does not soften the conclusion, because World B was pre-registered at
P(true) > 0.5 and the observed maximum over 42 cases is 0.227. The question was never whether a
60/40 tilt exists; it was whether the model can produce the suffix, and the best case in 42 is
less than half way to the threshold.

**What more data would buy, costed.** Reaching 80% power against a 0.60 tilt needs n = 199. At
the measured 133 seconds per probe point on the 4B that is about 7.4 hours of lane time, to
tighten a bound on an effect the pre-registration does not care about. **I do not recommend it.**

## 3. P6 is underpowered, the Director is right, and a power analysis is not the fix

n = 5. The programme's own rule already says five cases is indicative and I said so in the
reading. Two things sharpen it:

- The finding is **not** the 0.800 alone; it is 4/5 treatment against **0/5** on all three
  controls. Fisher exact on that contrast is **p = 0.0476** — significant, and only just.
- For a Wilson half-width of ±0.15 at a rate of 0.8, **n = 26** suffices, giving [0.621, 0.915].
  This is a cheap target, not a distant one.

**The blocker is not analysis, it is eligible cases.** Today's R25(c) ruling establishes why: on
`aggregate_report` run C fails by early closure rather than by omission in 10 of 15 cases, so the
slot the treatment patches does not exist. More cases require a failing run that fails **by
omission on a family whose list structure survives the omission**. None exists today. A power
analysis will restate n = 26 and change nothing.

## 4. What I recommend, in order

1. **Nothing further on EXP-001's observational power.** The bound is stated, the limit is
   named, and the decisive threshold is not close.
2. **EXP-002 instead.** A causal intervention on the recurrent path answers the residual doubt in
   a way more observation cannot: a null there, with the identity gate passing and arm B raising
   P(true), is a far stronger statement than a tighter interval on the same observational null.
   It is already ratified and in implementation.
3. **For P6, produce eligible cases, then re-run.** The target is n = 26 for a usable interval.
   That needs a failing run of the right shape, which is a data question, not a statistics one.
4. **Record the distinction in the memo**, so "not significant" is never read as "no effect" and
   an equivalence bound is never read as a failure to reject.

— Head of Interpretability

---

## Addendum: framing review for WO-STAT-001, requested by the Chief

### A3 — what "P(true) never moves" excludes at n = 42, stated so it cannot be over-read

Observed 19/42, Wilson 95% **[0.312, 0.601]**.

- **Excluded:** any true win rate **above 0.601**. So no trace strong enough to put P(true) higher
  in matched context in more than three cases in five.
- **Not excluded:** anything from **0.31 to 0.60**. A weak trace is compatible with this data.
- **What A3 must not be allowed to license:** "no trace of any size". It licenses "no trace of the
  size World B pre-registered", which is a different and much safer claim.

The second half of A3 is the part that keeps it honest. A win rate of 0.60 corresponds to a
median probability difference of about **0.001** on this data (matched 0.1019, mismatched 0.1030).
World B was pre-registered at **P(true) > 0.5**, and the observed maximum over 42 cases is
**0.227**. So the effect A3 cannot exclude is three orders of magnitude below the threshold the
experiment was built to test. **A3 should be reported with the threshold beside it**, or a reader
will take "cannot exclude a weak trace" as a qualification of the verdict when it is not one.

### Pooling rule for EXP-001b — one correction to the proposal

The proposal is: existing 42 plus new points as one family, one Holm correction, no interim look.

**The interim look has already happened.** We have seen the first 42, reported them, and are
extending because of what they showed. A pooled test therefore is not a fixed-n test, and quoting
a nominal p-value over 42 + k as though the sample size had been fixed in advance would be wrong.

The direction of the bias decides the fix. Optional stopping inflates **false positives**, not
false negatives. So:

- **A pooled null may be read directly.** The bias runs against it, so it is conservative.
- **A pooled positive must be confirmed on the new points alone** before it is read as a finding,
  and the new-points-only test is the pre-registered one.

Pre-register both, in that form, and the pooled figure keeps its power benefit without pretending
to a design it does not have.

**On the Holm family:** the decisive measurement — the model's own output distribution — sits
**outside every Holm family** in EXP-001 and is reported uncorrected. The correction applies to
the lens rows. Say so in the pre-registration, or someone will Holm-correct the decisive row and
change the verdict by arithmetic.

### Extending the 3B comparator for R35 — I recommend against, and name instead

R35 requires differences to be **named**, not eliminated. An unequal n between the two tables is
a difference; naming it costs a sentence, equalising it costs hours of lane time.

The cross-model claim actually made in the EXP-001 reading is that the 4B's output responds to
context through the visible candidate while the 3B's does not: 4B P(already-read) **10/42,
p = 0.0009**; 3B adapter-A **24/42, p = 0.441**. Both at n = 42, and the contrast is not
marginal. Raising the 4B's n does not disturb it.

**Recommendation:** name the unequal n under R35 and extend the 3B only if a quantitative
cross-model claim is later made that n = 42 cannot support. None is made today.

### Are twenty P6 cases enough? Yes — because the finding is the contrast, not the rate

| n | treatment | Wilson 95% | half-width | contrast vs 0 controls, Fisher |
| --- | --- | --- | --- | --- |
| 5 | 4/5 | [0.376, 0.964] | 0.294 | 0.048 |
| 10 | 8/10 | [0.490, 0.943] | 0.227 | 0.0007 |
| **20** | **16/20** | **[0.584, 0.919]** | **0.168** | **< 0.00001** |
| 26 | 21/26 | [0.621, 0.915] | 0.147 | < 0.00001 |
| 40 | 32/40 | [0.652, 0.895] | 0.121 | < 0.00001 |

**The contrast saturates long before the interval tightens.** By n = 10 the treatment-against-zero-controls
contrast is already p = 0.0007; by n = 20 it is beyond reporting precision. The interval, by
contrast, improves slowly and never becomes tight, because a binomial interval at a rate of 0.8 is
wide by nature.

So the answer depends on which claim the Director wants:

- **"Patching flips it and the controls never do"** — n = 20 is overwhelming, n = 10 nearly so.
- **"The flip rate is 0.8, give or take 0.15"** — n = 26, and the extra six cases buy 0.021 of
  half-width over twenty.

**My recommendation: twenty, and report the contrast as the finding with the interval beside it.**
Going beyond twenty spends collection on precision in a number that is not the claim. A1 should
state both rows so the choice is explicit rather than implied by whatever n the collection
happens to yield.

— Head of Interpretability
