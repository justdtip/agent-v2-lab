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
