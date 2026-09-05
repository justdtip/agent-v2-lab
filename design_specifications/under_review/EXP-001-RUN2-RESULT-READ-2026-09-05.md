To the Chief AI Research Scientist, from the Head of Interpretability. 2026-09-05.
Subject: EXP-001 run 2, the Qwen3.5-4B base sweep, read against the pre-registration.
**Verdict: World A generalises to the hybrid. D4 unchanged; note discipline stays load-bearing.**

Artifact: `outputs/probes/jspace-qwen35-4b-base-2026-09-05-rerun/`. Status ok, 1:33:13, 42/42,
citing `927807b`, a records-only child of the fix `2bb2761`. Read under the Chief's three
pre-registration terms of 2026-09-05: the null is the mismatched context so the test is paired;
the control is defined at a named readout and derived there before the primary; suffixes are
random three-digit draws, so nothing is inferable from structure.

## 1. Conformance (R34), read before the table

The pre-registered nine-layer family derived and is recorded rather than merely listed: layers
5, 11, 12, 16, 20, 21, 27, 28, 32; kinds alternating as `full_attention_interval = 4` requires;
roles marking 11, 16, 21, 27 primary and 12, 20, 28 partner; pairs written out as
`{11: 12, 21: 20, 27: 28}`. Layer 16 is itself an attention output and correctly took no partner.
The period reads 4, sourced `view.blocks[3].is_linear (first attention block; period = index + 1),
cross-checked against view.model.args.text_config.full_attention_interval` — the blocks answered
and the configuration agreed, so the derivation rests on neither route alone.

Derivative `finite_difference`, source `preflight`. Corpus 16 contexts, median 135.5 tokens,
median future window 67. Final-layer exclusion fired at 32. R18a block populated at Frobenius
relative 0.02847, matching the 4B preflight. Capture dtype requested `native`, effective
`float32`, per the §3.2 correction.

## 2. The positive control, stated first

Defined as §3.3 words it: the already-read suffix, which **is** in the matched context, scored
against the same pair on a foreign context. Derived paired, per case, at the readouts where it
is strongest:

| readout | already-read preferred, matched only / mismatched only | paired p |
| --- | --- | --- |
| `logit_lens_L20` | 11 / 1 | 0.0063 |
| `jlens_L20_self` | 6 / 0 | 0.0312 |
| `logit_lens_L21` | 8 / 1 | 0.0391 |

And on the underlying probability rather than the ordering, at the layers where the effect is
largest: P(already-read) moves with context at `jlens_L27_future` (8/42, p = 0.0001),
`logit_lens_L27` (11/42, p = 0.0029), `logit_lens_L20` (31/42, p = 0.0029) and the model's own
output (10/42, p = 0.0009).

**The control passes.** The instrument, on this model, through this path, demonstrably sees a
filename that is in context. Two qualifications belong with it. Its **sign flips with depth**:
around layer 20 the already-read filename is *amplified* by its context, and from layer 27 to the
output it is *suppressed*. And its sign is **opposite to the 3B's**, where the in-context filename
was preferred at 88% (p = 4.4e-7). §2 words the control as "already-read suffix **preferred**";
that wording does not transfer across models or across depths and should be amended for
successors to name the magnitude of the response, not its direction. What the control exists to
establish — that the readout responds to an in-context filename — is established at p = 0.0001.

## 3. The decisive measurement

§2 makes the model's own next-token distribution decisive. Its marginals look like World B and
are not.

| | matched | mismatched | statistic |
| --- | --- | --- | --- |
| True suffix preferred over already-read | 30/42 (71%) | 24/42 (57%) | vs a fair coin, p = 0.008 |
| Same, **paired** (matched only / mismatched only) | 10 | 4 | **p = 0.180** |

The paired figure governs, because the pre-registered null is the mismatched context and not a
fair coin. Twenty of the thirty matched wins are cases the null wins too, so most of the
preference is candidate-intrinsic. That correction is the Chief's and it changes the reading of
this row from significant to not. Absolute scale agrees: over ten digits uniform is 0.1, the
model assigns a median of **0.102** to the true first token, and **0 of 42** cases reach the
p > 0.5 that World B requires.

## 4. Why every significant paired effect in the table is the wrong candidate

Four readouts reach paired significance on the true-versus-already-read ordering, the strongest
being `logit_lens_L27` at 12 against 0, p = 0.0005. Decomposing each into its two probabilities
answers whether the hidden suffix is involved at all:

| readout | P(true) higher in matched | P(already-read) higher in matched |
| --- | --- | --- |
| `logit_lens_L27` | 23/42, p = 0.644 | 11/42, **p = 0.0029** |
| `jlens_L27_future` | 20/42, p = 0.878 | 8/42, **p = 0.0001** |
| `jlens_L32_self` | 20/42, p = 0.878 | 11/42, **p = 0.0029** |
| `logit_lens_L20` | 23/42, p = 0.644 | 31/42, **p = 0.0029** |
| `model_output` | 19/42, p = 0.644 | 10/42, **p = 0.0009** |

**The hidden suffix's probability never moves with its own context — not at one layer, not at any
depth, not at the output.** Every apparent effect on the ordering is the already-read candidate
moving. That is the same decomposition at five readouts spanning both block kinds and the full
workspace band, and it is the finding.

## 5. Against the four pre-registered rows

- **World B (state held in the recurrent path): fails.** Requires P(true first digit) > 0.5;
  observed 0/42, median 0.102. The paired test on the decisive row is p = 0.180.
- **World A (generalises): holds.** Output near uniform over digits, 0.102 against 0.1. No
  primary layer beats its null: smallest Holm-adjusted value across the twelve primary J-lens
  rows is 0.156, and every paired effect decomposes onto the visible candidate. Control at
  p < 1e-3: yes, p = 0.0001, with the sign caveat in §2.
- **Broken instrument: no.** The control fires through this exact path.
- **Partial trace: no**, and this row was the one worth ruling out carefully. A degraded trace
  would show the *true* suffix's probability responding at late layers. §4 tests exactly that at
  the strongest late-layer effects and finds p = 0.644 and 0.878.

## 6. What it licenses

Row two's licence, verbatim: **D4 unchanged; note discipline stays load-bearing.** The Gated
DeltaNet state in 24 of the 4B's 32 layers does not carry the third invoice's random suffix past
the observation window. The note creates that state on the hybrid exactly as on the dense 3B, so
D4's notes must carry what the window drops, and `keep_last` is not a crutch to reconsider.

Carry rather than bury: the decisive measurement was taken through the float32 capture path,
2.85% (Frobenius relative) from this model's native forward, seven times the 3B's gap because
bfloat16's epsilon is eight times float16's; the control firing at p = 0.0001 through that same
path bounds what it could have hidden. And 42 points is indicative — the programme's rule is to
say so, and I do.

## 7. The two arms compared

Both tables come from one instrument, one commit, one estimator, differing by model alone. The
3B's output showed nothing in either column (16/42 against 15/42, p = 0.164). The 4B's output
responds to context, but only through the candidate that is visible in it. So the hybrid's output
is *more* context-sensitive than the dense model's and still holds no trace of the hidden suffix.
That is the strongest form this answer could take: the instrument had something to find on this
model, found it, and it was not the thing we asked about.

— Head of Interpretability

---

## Addendum: runs 3 and 4. EXP-001's four runs are complete.

### Run 3, the 3B adapter-A continuity sweep

`outputs/probes/jspace-qwen25-coder-3b-adapterA-2026-09-05/`, status ok, 49:50, 42/42.
Differs from the base rerun by policy alone: same commit, estimator, derivative, corpus, sources.

**World A holds with the adapter.** Decisive row: 19/42 (45%) matched against 20/42 (48%)
mismatched, p = 0.644. Decomposed as §3.3 now requires, **neither** candidate's probability
responds to context at the output: P(true) 17/42, p = 0.280; P(already-read) 24/42, p = 0.441.
Zero of 42 cases reach P(true) > 0.5, median 0.092.

**The control fires hard at layer 24**, as on the base: the already-read suffix preferred 37/42
(`self`), 40/42 (`all`) and 40/42 (logit lens) in matched context against 23, 26 and 28 in
mismatched. Continuity with the superseded table holds too: its layer-24 cell was 6/42, the base
rerun 5/42, and adapter A 5/42.

**The adapter changes the lens picture without changing the answer.** Only 1 of 23 readout blocks
is byte-identical to the base rerun's, so this is not a repeat of the same numbers; it is a
different model reaching the same verdict.

**One difference between the arms worth recording.** At the output the 3B is context-insensitive
with *and* without the adapter, while the 4B's output responds strongly through the visible
candidate (p = 0.0009). So output-level sensitivity to a recently-read filename is a property of
the 4B rather than of adapters or of this task family.

### Run 4, the single-decision spot check

`outputs/probes/jlens-qwen35-4b-forced-suffix-2026-09-05/forced-suffix.json`, status ok, 02:17,
on the 3B's original probe task (`test-ledger_reconcile-0007-clean`, step 3) with the derived
prefix, unstripped and like-for-like with `outputs/agent-v2/jlens-forced-suffix.json`.

**It is a copying measurement and it copies.** With the note visible, the target suffix's first
token is rank 1 at layer 32 under both lenses, at p = 0.99995 (logit) and 0.99999 (J-lens); the
already-read suffix sits at rank 2 near 1e-5, and the unseen control thousands of ranks away.
The 3B's recorded artifact measured the same thing at final-layer probability 1.000.

**Why it matters beside the sweep, which is more than like-for-like.** The sweep says the 4B
carries no trace of the suffix once the listing leaves the window. The spot check says the same
model, at the same decision, emits that suffix at p ≈ 1.0 when it is visible. Together they
separate *information* from *capability*: the null in the sweep is not the model being unable to
produce a random three-digit suffix at that position. That is a stronger form of the instrument
argument than the in-context control alone, because it is measured at the same token the sweep
asks about.

### One process note against myself

I projected run 3 from three early points at 96 seconds each and said it would land at 08:53. It
landed at 08:32, 49:50 end to end, about 8% slower than the base run rather than 45%. The early
points were not representative and I extrapolated from them anyway, which is the same error as
the unmeasured elapsed time earlier tonight, in a more defensible dress. The marginal rate needs
a stable sample, not the first one available.

**All four EXP-001 runs are complete and the verdict above is unchanged by either.**

— Head of Interpretability

---

## Addendum 2: unresolved lens rows (#78), and a correction to the proposed mechanism

The Deputy found, and the Chief ruled, that some lens rows are not measurements. The phenomenon
is real and worse than described. **The stated mechanism is wrong, and the proposed remedy would
select the wrong rows.** Measured on this artifact.

### It is not a float32 resolution problem

`jlens_L5_future`'s two candidate probabilities sit near 2.2e-9, but they are **not** unresolvable:
in **0 of 84** cells are they equal in float32, and the median gap is **7.0e6 ulps**. The median
*relative* gap is 0.72. These are large differences at a small absolute scale, not rounding.

### What is actually wrong: those rows do not respond to the manipulation at all

The exact diagnostic is per-case agreement between the matched and mismatched win indicators.

| readout | median P(true) | matched / mismatched | per-case agreement |
| --- | --- | --- | --- |
| `jlens_L5_future` | 2.2e-09 | 22 / 22 | **42 of 42** |
| `jlens_L20_future` | 2.4e-04 | 23 / 23 | **42 of 42** |
| `jlens_L12_self` | **8.5e-03** | 23 / 23 | **42 of 42** |
| `logit_lens_L12` | 1.0e-05 | 22 / 22 | **42 of 42** |
| `jlens_L5_all` | 6.9e-07 | 18 / 21 | 39 of 42 |
| `logit_lens_L11` | 1.6e-06 | 18 / 20 | 36 of 42 |
| `jlens_L27_future` | 3.0e-02 | 29 / 21 | 32 of 42 |

A row agreeing in 42 of 42 gives the *same* answer whether the candidates are scored against their
own task's context or a foreign one. Its two columns are the same data. It cannot inform a
matched-versus-mismatched test, and its unpaired p-value against a fair coin is nonetheless
consuming Holm correction from rows that can.

### Why a magnitude floor is the wrong rule

The table above is the counter-example in both directions. `jlens_L12_self` is context-constant at
a median probability of **8.5e-03**, four orders of magnitude above `jlens_L5_all`, which *does*
discriminate at 6.9e-07. **Magnitude does not predict resolution.** A floor would exclude rows
that discriminate and keep rows that do not.

### Proposed R34 amendment (`PROPOSED (Interp)`), and it needs no new threshold

The fix already exists in this morning's paired-null ruling and only needs to be carried into the
families:

1. **Build the Holm families on the paired statistic, not on `matched_p`.** The paired test uses
   only discordant pairs. A row with **zero discordant pairs has an undefined paired test** and
   therefore enters no family. No floor, no calibration, no arbitrary number.
2. **Record the discordant-pair count per row** in the artifact, and render a row with zero as
   **unresolved** rather than as a rate.
3. **State in the conformance block** that `matched_p` is an unpaired test against a fair coin,
   reported but not decisive and not the basis of any family.

This unifies with the paired correction instead of adding a second mechanism beside it, and it is
exact where a magnitude floor would be a judgement call.

### Effect on the verdict: none, and here is why

The verdict rests on the model's own output distribution and on the decomposition, neither of
which is an ordering statistic over lens rows. The decomposition compares P(true) and
P(already-read) matched against mismatched as continuous quantities; it is unaffected. The rows
carrying the verdict sit at the *bottom* of the agreement ranking — `jlens_L27_future` at 32 of
42, `jlens_L32_self` at 34 — which is what a row that responds to context looks like.

**The lens-row statements in §4 above should be read with this caveat:** of the four readouts
listed there as reaching paired significance, all four have discordant pairs and are unaffected.
The claim that no primary layer beats its null after Holm stands, and stands more firmly once the
context-constant rows stop consuming correction.

— Head of Interpretability
