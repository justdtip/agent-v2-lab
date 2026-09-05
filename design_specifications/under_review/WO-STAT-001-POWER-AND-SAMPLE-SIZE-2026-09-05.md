# WO-STAT-001: Are today's small-sample results true nulls, or underpowered? (2026-09-05)

Requested by the Director (via Dispatch). Drafted and reviewed by the Chief; domain review by the
Head of Interpretability; implemented under the Deputy; commits gated by the Chief.

## 0. The two results in question, with their sources

| result | statistic | source |
| --- | --- | --- |
| P6 primary, `previous_notes` flip rate | 4/5 = 0.80, Wilson 95% [0.376, 0.964]; controls 0/5 | `outputs/probes/patch-C-r27-2026-09-05/patch.json` (5 stable cases of 5 selected) |
| EXP-001 4B decisive row, paired | matched-only 10, mismatched-only 4, exact two-sided p 0.180; decomposition P(true) up in 19/42 (p 0.64) | `outputs/probes/jspace-qwen35-4b-base-2026-09-05-rerun/sweep.json`, `per_case` |

Where the cases come from, so the collection plan is about the right bottleneck:
- P6's five cases are the eligible **ledger_reconcile** failures of policy C on the 180-task test
  split: C fails 6 of 15 ledger tasks (`outputs/agent-v2c/evals/best-adapter-test.json`), B
  passes 13 of 15, and R24/R27 eligibility leaves five stable cases. The bottleneck is the number
  of ledger tasks evaluated, not the probe.
- EXP-001's 42 probe points are the usable ledger tasks of the 720-task `jsweep` split after the
  step-3, note-prefix, leakage and first-token filters. The bottleneck is the split size.

## 1. Part A: formal power analysis (no lane, no lift; dispatch now)

Deliverable: `src/local_llm_lab/probes/power.py` (exact computations, no model) with tests, and
`under_review/POWER-ANALYSIS-2026-09-05.md` reporting the tables below with every input named
(R38: a figure names its source).

A1. **P6 flip rate.** For a treatment flip rate π_T against a control rate π_C, exact power of
Fisher's test (one-sided, treatment > control) at α 0.05 for n cases per arm ∈ {5, 10, 15, 20,
30, 40}, for (π_T, π_C) ∈ {(0.8, 0.0), (0.6, 0.0), (0.5, 0.1), (0.4, 0.1)}. Also the Wilson 95%
half-width of the treatment rate at each n. State the n at which 80% power is reached for each
pair, and the n at which the half-width falls under 0.15.

A2. **EXP-001 paired comparison.** Exact power of the two-sided paired test (exact binomial on
discordant pairs) at α 0.05, for a discordance rate d = 14/42 ≈ 0.33 (observed) and matched-only
share q ∈ {0.6, 0.65, 0.714 (observed), 0.8}, for n probe points ∈ {42, 84, 126, 168, 252}. State
the n for 80% power at the observed q; state the smallest q detectable at 80% power with n = 42.

A3. **EXP-001 decomposition.** For the sign test on "P(true) higher under matched" at rate r,
power at n ∈ {42, 84, 168} for r ∈ {0.6, 0.65, 0.7}, and the smallest r detectable at n = 42.
This is the number that says what "P(true) never moves" excludes: at n = 42 the observed 19/42
excludes a strong trace (r ≥ ~0.7) and not a weak one.

A4. **Holm.** For EXP-001, the per-readout power under Holm across the family sizes used (six on
the 3B, eight on the 4B, per readout), so the table says what the corrected family could have
detected.

Acceptance: every number in the tables reproduces from `power.py` by a test; the markdown names
the inputs and the observed statistics it was computed against; the Head of Interpretability
reviews the framing (what each power figure licenses saying about the recorded results).

## 2. Part B: collection to the n Part A names (lane; Director lifts; costed here)

Pre-registered before any run: the target n from Part A, the pooled analysis (existing points plus
new, one family, one Holm correction, no interim look), and the reading rules unchanged from
EXP-001 §2/§3.3 and P6's R27 scoring.

B1. **EXP-001 extension (EXP-001b).** Generate additional `jsweep` points beyond the 720-task
split (a second split name with its own seed, declared in `tasks.JSPACE_SPLITS`, since
`JSPACE_SPLIT_LIMIT` is 720), run the 4B base sweep on the new points only, and pool. Measured
cost: 133 s per point on the 4B with the C1 cache (run 2), so +126 points ≈ 4.7 h; the 3B
comparator at 66 s per point ≈ 2.3 h if the comparator is extended too (recommended, for R35).
Artifacts cite the commit; the pooled reading states both runs' identities.

B2. **P6 extension.** Evaluate policies C and B on a ledger-heavy held-out split (e.g. 60
ledger_reconcile tasks, new split name and seed) to harvest eligible failing cases: at C's
observed 6/15 failure rate and B's 13/15 pass rate, expect ≈ 20 eligible cases. Measured cost:
the 180-task evaluations took 3,141 s (C) and 3,045 s (B), ≈ 17 s per task per policy, so 60
tasks × 2 policies ≈ 35 min; then the P6 primary on ≈ 20 cases (the five-case primary section of
the 2026-09-05 run took ≈ 1 h including capture; estimate ≈ 3 h for 20). R24/R27 eligibility and
strict scoring unchanged; the content-swap control on every case.

B3. **Not in scope.** The aggregate_report secondary (closed under SPEC-004 §5b: not a
generalisation test while C fails by early closure). A generalisation test needs a family whose
notes keep list structure past an omission; that is a separate design question.

Sequencing recommendation for the Director: Part A now (hours, no lane). Then EXP-002 (causal,
≈ 1 h, ratified) before B1, since it changes what World A means more than more points do. Then B1
overnight (≈ 7 h with the comparator), then B2 (≈ 4 h). Each B run is its own lift.

## 3. What the answer will look like

Part A alone settles the Director's question in the form: "at the recorded n, the result excludes
effects of size X and not of size Y; reaching Z requires n". Part B then either narrows the P6
interval and moves the EXP-001 paired test off the fence, or leaves them where they are with the
power to say so. Either way the memo's entries get a power statement beside each number.

## 4. Amendments after the Head of Interpretability's framing review (2026-09-05, ratified)

- **A3 is reported with its threshold beside it.** Wilson on 19/42 is [0.312, 0.601]: the result
  excludes any true win rate above 0.601 (a trace strong enough to raise P(true) under matched
  context in more than three cases in five) and does not exclude 0.31 to 0.60. The sentence A3
  licenses is "no trace of the size World B pre-registered" (P(true) above 0.5; observed
  maximum 0.227), not "no trace of any size"; and the effect A3 cannot exclude corresponds to a
  median probability difference of about 0.001 on this data, three orders of magnitude below the
  pre-registered threshold. Without that sentence "cannot exclude a weak trace" reads as a
  qualification of the verdict, which it is not.
- **Pooling rule corrected (my §2 was wrong).** The interim look has already happened: the first
  42 were seen, reported and are the reason for extending, so a pooled test over 42 + k is not a
  fixed-n test. Optional stopping inflates false positives, not false negatives. Pre-registered
  in that form: a pooled **null** may be read directly (the bias runs against it); a pooled
  **positive** is read as a finding only if it is confirmed on the new points alone.
- **The decisive measurement sits outside every Holm family** in EXP-001 and is reported
  uncorrected; the correction applies to the lens rows. Pre-registered so nobody corrects the
  decisive row by arithmetic.
- **The 3B comparator is not extended.** R35 requires a difference to be named, not eliminated;
  the unequal n is named in the pooled reading. The cross-model claim actually made (the 4B's
  output responds through the visible candidate, 10/42 at p 0.0009, the 3B's does not, 24/42 at
  p 0.44) stands at n = 42 on both sides and is not disturbed by raising the 4B's n. Extend the
  3B only if a quantitative cross-model claim is later made that 42 cannot support.
- **P6: twenty cases, and A1 states both rows.** The treatment-against-zero-controls contrast is
  p 0.0007 at n = 10 and beyond reporting precision at n = 20; the Wilson half-width is 0.294 at
  five, 0.168 at twenty, 0.147 at twenty-six. Twenty settles "patching flips it and the controls
  never do"; twenty-six is the n for "the rate is 0.8 give or take 0.15". Recommendation:
  twenty, reporting the contrast as the finding with the interval beside it. The Director
  chooses which claim he wants rather than inheriting whatever n the collection yields.

## 5. Corrections from Part A's report (2026-09-05), superseding the figures above

- **A4 family sizes** are not six on the 3B and eight on the 4B. From the artifacts, per readout:
  3B five, five and six; 4B eight, eight and nine (the final layer contributes a `self` row
  only). No change to the power column at n = 42; it moves the required n.
- **§4's "median probability difference of about 0.001"** does not reproduce from `per_case`
  under any reading; the closest is about 0.0025. The qualitative claim (the effect A3 cannot
  exclude is orders of magnitude below the pre-registered World B threshold of P > 0.5, observed
  maximum 0.227) survives; the figure is restated as about 0.0025.
- **§4's "p 0.0007 at n = 10"** is two-sided; A1 specifies one-sided, which is 0.00036. Both
  reported with the tail named.
- **A3's threshold** is 0.713 (power 0.8000 at n = 42), not 0.70 (power 0.743). The sentence:
  at n = 42 the decomposition's 19/42 excludes a win rate of 0.713 and above at 80% power.
- **Headline, verified independently by the Chief:** P6 at n = 5 has power 0.7373 against a
  true 0.8 and detects 0.8314 at 80%, so 4/5 is a real contrast against zero controls and not
  a measurement of the rate. EXP-001's paired comparison at n = 42 has power 0.2709 against the
  observed effect (d = 14/42, q = 10/14) and detects a matched share of 0.8763 at 80%; the
  observed effect needs **135 points** (stable crossing; 0.8014). Discrete tests are not
  monotone in n (Fisher at 3/3 vs 0/3 rejects at p = 0.05, power 0.512; at n = 4 the next
  attainable table is 0.0714, power 0.410), so every "n for 80%" is a stable crossing.
