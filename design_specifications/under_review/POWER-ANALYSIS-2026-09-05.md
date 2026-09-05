# POWER-ANALYSIS: what the small-sample results exclude, and at what n (2026-09-05)

WO-STAT-001 Part A. Computed by `src/local_llm_lab/probes/power.py`; every figure below is
reproduced by a test in `tests/test_power.py`, which is the acceptance criterion. No model, no
network, no new dependency: the module imports `math` and nothing else, and every number is an
exact enumeration over the discrete distribution rather than a normal approximation.

**This is not observed power.** Power evaluated at the effect that was observed is a monotone
function of the p-value and licenses no sentence the p-value did not already license. What
follows is, for a range of *hypothesised* effects, the n at which each becomes detectable — so
the Director can name which effects the recorded nulls exclude and which they do not.

## 0. Sources (R38)

Every observed statistic below was read out of the artifact, not transcribed from the work
order. The two that the work order names:

| figure | value | artifact | field |
| --- | --- | --- | --- |
| P6 `previous_notes` flip rate | 4/5 = 0.8 | `outputs/probes/patch-C-r27-2026-09-05/patch.json` | `cells["6:previous_notes"].treatment.numerator` / `.denominator` |
| P6 Wilson 95% | [0.3755283, 0.9637768] | same | `cells["6:previous_notes"].treatment.wilson_95` |
| P6 controls | 0/5 on all three | same | `cells["6:previous_notes"].controls.{unrelated_task,random_positions,content_swap}.numerator` |
| EXP-001 paired, matched-only | 10 | `outputs/probes/jspace-qwen35-4b-base-2026-09-05-rerun/sweep.json` | derived from `per_case[*].{matched,mismatched}.model_output.{true,false}` |
| EXP-001 paired, mismatched-only | 4 | same | same |
| EXP-001 decomposition | 19/42 | same | `per_case[*]`, count of `matched.model_output.true > mismatched.model_output.true` |
| EXP-001 decisive marginals | 30/42 and 24/42 | same | `results.model_output.{matched_wins,mismatched_wins}` |

Recomputed from those fields: the paired exact two-sided p is **0.1795654** (§0 of the work
order says 0.180 — agrees), and the decomposition's is **0.6439690** (says 0.64 — agrees). The
P6 rate, its interval and the zero controls agree with §0 as written. **No §0 figure is
contradicted by its artifact.**

The run artifacts are untracked, so `tests/test_power.py` carries the observed statistics as
literal inputs and reproduces the provenance check only when the artifacts are present
(`test_recorded_statistics_match_their_artifacts`, which skips otherwise). That test was run
against the real artifacts and passes; the paths and fields above are how a reader repeats it.

## 1. Three different objects

A reader will assume the three grids measure one thing. They do not, and the word "power" means
something different in each.

- **A1 — two-sample Fisher.** Two independent arms of n, treatment ~ Bin(n, π_T) and control ~
  Bin(n, π_C). The *test* conditions on the total number of successes and reads the
  hypergeometric tail; the *power* is unconditional, summing the joint probability of every
  (a, b) table whose conditional p clears α. One-sided (treatment > control) per §1 A1.
- **A2 — exact McNemar.** Each probe point is discordant with probability d; given D discordant
  pairs the matched-only count is Bin(D, q), tested by the exact two-sided binomial against a
  fair coin. The test conditions on D — that is what makes it McNemar — but D is random when you
  are choosing n, so the planning power sums over D ~ Bin(n, d) as well. Both are reported in
  §3.3, because they differ and in the observed regime they differ in the *unintuitive*
  direction.
- **A3 — unconditional sign test.** All n cases yield a comparison, wins ~ Bin(n, r), exact
  two-sided binomial against a fair coin. n is the whole sample, not a subset, which is why A3
  reaches a given power at a far smaller n than A2 does.

α = 0.05 throughout; the power target is 0.80. Every "n for 80%" is a **stable** crossing — the
smallest n from which the target holds for every larger n — not a first crossing, for the reason
in §6.2.

## 2. A1 — P6's flip rate

### 2.1 Exact power of Fisher's one-sided test, n per arm

| (π_T, π_C) | n=5 | n=10 | n=15 | n=20 | n=30 | n=40 | n for 80% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (0.8, 0.0) | 0.7373 | 0.9991 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **6** |
| (0.6, 0.0) | 0.3370 | 0.9452 | 0.9981 | 0.9997 | 1.0000 | 1.0000 | **8** |
| (0.5, 0.1) | 0.1210 | 0.4713 | 0.6732 | 0.8375 | 0.9542 | 0.9895 | **19** |
| (0.4, 0.1) | 0.0548 | 0.2911 | 0.4627 | 0.6190 | 0.8009 | 0.9151 | **30** |

### 2.2 Wilson 95% half-width of the treatment rate

At the success count the hypothesised rate implies for that n (half-up; 0.8 of 26 is 20.8 → 21).

| π_T | n=5 | n=10 | n=15 | n=20 | n=30 | n=40 | n for half-width < 0.15 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8 | 0.2941 | 0.2266 | 0.1907 | 0.1677 | 0.1390 | 0.1213 | **26** |
| 0.6 | 0.3258 | 0.2596 | 0.2221 | 0.1973 | 0.1654 | 0.1453 | **38** |
| 0.5 | 0.3258 | 0.2634 | 0.2254 | 0.2007 | 0.1685 | 0.1480 | **39** |
| 0.4 | 0.3258 | 0.2596 | 0.2221 | 0.1973 | 0.1654 | 0.1453 | **38** |

### 2.3 The two rows side by side, per §4

These answer different questions and are not collapsed into one recommended n. The contrast
column is the p at the observed configuration (0.8 flip rate against zero controls) at that n.

| n | flips | contrast p, one-sided | contrast p, two-sided | Wilson half-width |
| --- | --- | --- | --- | --- |
| 5 | 4/5 | 2.381e-02 | 4.762e-02 | 0.2941 |
| 10 | 8/10 | 3.572e-04 | **7.144e-04** | 0.2266 |
| 15 | 12/15 | 5.261e-06 | 1.052e-05 | 0.1907 |
| 20 | 16/20 | 7.709e-08 | 1.542e-07 | 0.1677 |
| 26 | 21/26 | 3.426e-10 | 6.852e-10 | **0.1468** |
| 30 | 24/30 | 1.647e-11 | 3.294e-11 | 0.1390 |
| 40 | 32/40 | 3.510e-15 | 7.020e-15 | 0.1213 |

§4's "p 0.0007 at n = 10" is the **two-sided** figure (7.144e-04). §1 A1 specifies the
*one-sided* alternative, which at n = 10 is 3.572e-04. Both are correct for their own tail and
they differ by exactly a factor of two on a 2×2 with an empty control arm; the report states
both so that no figure is quoted without its tail. §4's other two P6 figures reproduce exactly:
half-width 0.294 at five, 0.168 at twenty, 0.147 at twenty-six, and "beyond reporting precision
at n = 20" holds on either tail.

## 3. A2 — EXP-001's paired comparison

Discordance d = 14/42 = 0.3333 (observed). Matched-only share q; observed q = 10/14 = 0.7143.

| q | n=42 | n=84 | n=126 | n=168 | n=252 |
| --- | --- | --- | --- | --- | --- |
| 0.60 | 0.0706 | 0.1328 | 0.2090 | 0.2786 | 0.4074 |
| 0.65 | 0.1343 | 0.2817 | 0.4411 | 0.5715 | 0.7604 |
| **0.7143 (observed)** | **0.2709** | 0.5561 | 0.7692 | 0.8862 | 0.9754 |
| 0.80 | 0.5422 | 0.8815 | 0.9780 | 0.9965 | 0.9999 |

- **n for 80% power at the observed q: 135 probe points** (0.7981 at 134, 0.8014 at 135).
- **Smallest q detectable at 80% power with n = 42: 0.8763.** At n = 84 it is 0.7731.

### 3.1 This grid is fixed-n power, and the planned EXP-001b test is not fixed-n

Per §4's corrected pooling rule, the interim look has already happened: the first 42 points were
seen, reported, and are the reason for extending. A pooled test over 42 + k is therefore **not**
a fixed-n test, and the table above is the power of a *fresh* design at that n. Optional
stopping inflates false positives, not false negatives, so the direction of the bias matters:

- a pooled **null** may be read directly against this table — the bias runs against it, so the
  table if anything understates how much a pooled null excludes;
- a pooled **positive** may **not**. Read as fixed-n power, this grid would overstate what a
  pooled positive licenses. Per §4 a pooled positive counts only if it is confirmed on the new
  points alone — and the power of *that* confirmation is the row at n = k, not at 42 + k.

So a collection to 135 total points buys 80% power only for the pooled-null reading. To confirm
a positive on the new points alone at the observed q needs 135 *new* points, i.e. 177 in total.

### 3.2 The 3B comparator is not extended

Per §4, R35 requires the difference to be named rather than eliminated, so the unequal n stands.
The cross-model claim actually made holds at n = 42 on both sides and is not disturbed by
raising the 4B's n: 4B 10/42 at p = 0.000941, 3B 24/42 at p = 0.440799 (both recomputed here
and matching the recorded values).

### 3.3 Conditional and unconditional power differ, and cross over

| n | unconditional (D random) | conditional on D = round(n·d) |
| --- | --- | --- |
| 42 | 0.2709 | 0.1904 |
| 84 | 0.5561 | 0.5939 |
| 126 | 0.7692 | 0.8053 |
| 168 | 0.8862 | 0.9062 |
| 252 | 0.9754 | 0.9777 |

At n = 42 the unconditional figure is the **larger** of the two — the exact test on fourteen
discordant pairs is lumpy enough that the luckier draws of D more than repay the unluckier ones
— and from n = 84 up the usual ordering resumes. Neither substitutes for the other. The planning
figure is the unconditional one.

## 4. A3 — EXP-001's decomposition

Sign test on "P(true) higher under matched" at true rate r.

| r | n=42 | n=84 | n=168 |
| --- | --- | --- | --- |
| 0.60 | 0.2366 | 0.4061 | 0.6996 |
| 0.65 | 0.4808 | 0.7626 | 0.9696 |
| 0.70 | **0.7430** | 0.9565 | 0.9995 |

- **Smallest r detectable at 80% power with n = 42: 0.7130.** At n = 84: 0.6565. At n = 168:
  0.6119.
- n for 80% power at r = 0.70: **54**.

### 4.1 The threshold beside the result, per §4

The power threshold and the interval answer different questions and the interval is the tighter
of the two here:

- **What this result excludes.** Wilson on 19/42 is [0.3122316, 0.6005114] (recomputed against
  the repository's own `wilson`, which `power.wilson_interval` restates and a test pins). The
  result excludes any true win rate above **0.601** and does not exclude 0.31 to 0.60.
- **What a design of this size reliably detects.** 0.713 at 80% power.

The interval is tighter than the power threshold because the observed 19/42 came in *below* 0.5,
which is evidence against a high r beyond what the design alone guarantees. Quoting only the
power threshold would understate what the recorded result excludes.

The sentence A3 licenses is therefore **"no trace of the size World B pre-registered"** — P(true)
above 0.5, against an observed maximum P(true) of 0.227368 (`per_case[*].matched.model_output.true`)
— not "no trace of any size". "Cannot exclude a weak trace" is not a qualification of the
verdict, and must not be printed without the 0.601 bound beside it.

## 5. A4 — Holm

`holm_alpha(α, m) = α/m` is the level a row must clear on its own inside a family of m tests:
Holm rejects the smallest p exactly when it clears α/m and rejects nothing at all otherwise, so
this is the exact threshold for the row that would carry the family and a floor for every other.
The recorded lens rows are sign tests at n = 42, so A4 is A3 re-run at that level.

| family m | α/m | r=0.60 | r=0.65 | r=0.70 | smallest r at n=42 | n for 80% at r=0.70 |
| --- | --- | --- | --- | --- | --- | --- |
| 5 (3B `all`, `future`) | 0.010000 | 0.0860 | 0.2411 | 0.4957 | 0.7574 | 76 |
| 6 (3B `self`) | 0.008333 | 0.0860 | 0.2411 | 0.4957 | 0.7574 | 79 |
| 8 (4B `all`, `future`) | 0.006250 | 0.0449 | 0.1499 | 0.3632 | 0.7793 | 82 |
| 9 (4B `self`) | 0.005556 | 0.0449 | 0.1499 | 0.3632 | 0.7793 | 85 |

**Family sizes corrected.** §1 A4 says "six on the 3B and eight on the 4B, per readout". The
artifacts say the size depends on the readout: 3B `all`=5, `future`=5, `self`=6; 4B `all`=8,
`future`=8, `self`=9 (counted as the `jlens_*` rows carrying `matched_p_holm` in each sweep's
`results`). The final layer contributes only a `self` row — `all` and `future` are excluded
there because no decoder block remains in the tail — so the `self` family is one larger. The
work order quoted the per-model maximum as though it were uniform.

It costs nothing in the power column: at n = 42 the sign test is discrete enough that α/5 and
α/6 fall between the same two attainable p-values, as do α/8 and α/9, so each pair has an
identical rejection set and identical power. It does move the n each family would need (76 vs
79, 82 vs 85).

**The decisive row is not in this table.** Per §4 the decisive measurement sits outside every
Holm family and is reported uncorrected; A4 describes the lens rows only. Nothing here should be
read as a corrected p for the decisive row.

## 6. The three answers in plain terms

### 6.1 P6 — at n = 5, what is indistinguishable from the controls?

Against controls at 0/5, Fisher's one-sided test at α 0.05 rejects for **only two** of the six
possible treatment counts:

| flips | 0/5 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 |
| --- | --- | --- | --- | --- | --- | --- |
| one-sided p | 1.0000 | 0.5000 | 0.2222 | 0.0833 | **0.0238** | **0.0040** |

So at n = 5 **every flip rate at or below 3/5 is indistinguishable from controls at zero**, and
4/5 is the smallest count that separates at all. The observed result sits exactly on the edge of
what the design could have shown.

Worse for the claim: the smallest *true* flip rate that n = 5 detects at 80% power against zero
controls is **0.8314** — above the observed 0.8. Power at n = 5 against a true rate of 0.8 is
only 0.7373. **"4 of 5 with controls at zero" is allowed to claim that the flip rate is not
zero, and essentially nothing about its size.** The interval says the same: [0.376, 0.964] is
consistent with a true rate anywhere from "just over a third" to "almost always".

At the §4 recommendation of twenty cases, the contrast is beyond reporting precision (7.7e-08
one-sided) and the half-width is 0.168; twenty-six is where the half-width clears 0.15.

### 6.2 EXP-001 paired — is "10 versus 4, p = 0.18" a null?

**No. It is an absence of evidence.** The smallest matched-only share detectable at 80% power
with n = 42 is **0.8763**. The observed share is 0.7143 — far below the design's detection
threshold. Power against the observed effect is 0.2709: had the true share been exactly what was
observed, this design would have missed it roughly three times in four.

The design could not have shown an effect of the observed size, so its failure to show one is
not evidence that none exists. Detecting the observed share at 80% needs **135 probe points**
(fixed-n), or 135 *new* points if the finding must be confirmed on new data alone per §3.1.

### 6.3 EXP-001 decomposition — confirm or refute the work order's claim?

The claim (§1 A3): "at n = 42 the observed 19/42 excludes a strong trace (r ≥ ~0.7) and not a
weak one."

**Confirmed in substance, with the threshold corrected.** The actual 80%-power threshold at
n = 42 is **r = 0.7130**, not 0.70. At exactly 0.70 the power is 0.7430 — short of the 0.80
convention, so 0.70 itself is marginally *not* excluded at the conventional bar. The work
order's "≳" covers 0.713 and the claim survives as written, but the precise number is 0.713 and
the report states it rather than rounding it into a claim the arithmetic misses.

The stronger and more useful statement is §4's, which comes from the interval rather than the
power curve: the result **excludes any true win rate above 0.601** and does not exclude 0.31 to
0.60. That bound must be printed beside the verdict.

## 7. Sanity checks

Run before the grids were trusted; all four are pinned by tests.

1. **Sign test at r = 0.5 returns the attained size.** 0.043559 (n=42), 0.037530 (n=84),
   0.036927 (n=168) — each ≤ α, approaching it from below because the discrete binomial has no
   outcome at the boundary. A value above α would mean the rejection set was built wrong. The
   paired object behaves the same way: 0.026523 at n = 42, q = 0.5.
2. **Fisher power rises monotonically in the effect.** Verified at n = 20 across π_T from 0.50
   to 1.00 in hundredths: the sequence is sorted.
3. **Fisher power does *not* rise monotonically in n — the work order's sanity check is wrong
   on this point.** At three cases per arm, 3/3 against 0/3 has p exactly 0.05 and rejects, so
   power is 0.8³ = 0.5120. At four, that table's p falls to 0.0143 but the next one in (3/4
   against 0/4) is 0.0714 and cannot reject, so only the perfect table rejects and power falls
   to 0.8⁴ = 0.4096. A second, tiny violation sits at n = 15 → 16. This is the ordinary sawtooth
   of an exact discrete test, not a bug, and it is why every "n for 80%" above is a stable
   crossing rather than a first crossing.
4. **The restated helpers match the repository's.** `power.wilson_interval` equals
   `pipeline.evaluate.wilson` and `power.sign_test_p` equals `probes.jspace_sweep.sign_test` on
   every configuration used here, so the power figures describe the same tests that produced the
   recorded artifacts. The module's cached fast paths are separately checked against
   straightforward per-cell computation.

## 8. Claims checked against the artifacts

Confirmed: §0's P6 rate, interval and zero controls; §0's paired 10/4 and p 0.180; §0's
decomposition 19/42 and p 0.64; §4's Wilson [0.312, 0.601] on 19/42; §4's half-widths 0.294 /
0.168 / 0.147 at five / twenty / twenty-six; §4's observed maximum P(true) 0.227; §4's
cross-model pair (10/42 at p 0.0009, 24/42 at p 0.44); §4's "beyond reporting precision at
n = 20".

Corrected:

- **§1 A4's Holm family sizes.** Not "six on the 3B and eight on the 4B per readout" — the
  artifacts give 5/5/6 on the 3B and 8/8/9 on the 4B by readout (§5). No effect on the power
  column, some effect on the required n.
- **§4's "p 0.0007 at n = 10"** is the two-sided Fisher figure; §1 A1 specifies one-sided, which
  is 0.00036 (§2.3). Both are reported with their tail named.
- **§1 A3's "r ≥ ~0.7"** is 0.713 at the 80% convention; power at exactly 0.70 is 0.743 (§6.3).
- **§1's implied monotonicity of Fisher power in n** does not hold (§7.3).

Not reproduced:

- **§4's "median probability difference of about 0.001".** No reading of `per_case`
  `model_output` gives 0.001. The candidates: median signed difference **−0.003843**; median
  absolute difference **0.017468**; median of the positive differences **0.023562**; and, under
  the reading that seems intended — the median after shifting the distribution so the win rate
  reaches the Wilson upper bound — **+0.002549** (at 25/42) or **+0.002849** (at 26/42). The
  closest is ~0.0025, roughly 2.5× the quoted figure. The differences nearest the decision
  boundary do run from 0.0004 to 0.004, which is likely where "about 0.001" came from as an
  order-of-magnitude statement. **The qualitative claim survives under every reading** — the
  effect A3 cannot exclude is two to three orders of magnitude below the pre-registered
  threshold of P(true) > 0.5 — but the specific figure should be restated as ~0.0025, or the
  sentence rewritten to name the range rather than a median.
