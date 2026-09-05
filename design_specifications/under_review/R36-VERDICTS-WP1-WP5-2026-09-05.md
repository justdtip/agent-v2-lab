# R36 verdicts: WP1 (hosted lens in the probe stack) and WP5 (EXP-003 re-specification)

Author: Head of Interpretability, 2026-09-05. These are reviews of specifications, not of code:
neither work package has an implementation yet. Basis: `pending/03-REVISED-PLAN-2026-09-05.md`
WP1 and WP5; `under_review/HOSTED-JLENS-QWEN35-4B-RESULTS-2026-09-05.md`; R40a and R40b;
`pending/EXP-003-DISTANCE-CURVE-QWEN35-4B.md` as it stands.

## WP1. Verdict: proceed, with four amendments

**A1. The 42-case table is a validation run, not a fixture, and the spec calls both "the
fixture".** Reproducing it needs the model and 42 forward passes at a median of 1547 tokens, about
154 seconds by the memo's own timing. The project's tests do not load models. WP1 therefore needs
two separable artefacts: a unit fixture on the module's seams, in R31 form against the real
`ArchitectureView` and the real lens loader; and a validation run whose artifact is compared to
the recorded table. Only the first is a test.

**A2. "Within one case" licenses a nondeterminism nobody has characterised.** The comparison is a
rank ordering of two candidates, which is deterministic given the same activations, lens file and
dtype. Any discrepancy at all is either a defect or a named source of nondeterminism. Replace the
tolerance with exact reproduction of the per-case boolean vector, and if a case does move, the
run does not pass until the cause is named. A count tolerance on a deterministic quantity hides
exactly the class of error the fixture exists to catch.

**A3. Layer 32 is a member of the family where the instrument is not itself.** The lens files
cover blocks 0 to 30, so layers 1 to 31; layer 32 is the identity by construction, which is why
the memo's own table labels it "hosted = logit lens = model output". The spec says the Holm family
is "the layers under one instrument". State whether layer 32 is in it. My recommendation: it is
reported, and it is excluded from the J-lens family, because a J-lens claim at a layer where the
J-lens is the identity is a claim about the output and not about the workspace.

**A4. The quantisation gap is still unmeasured and every downstream reading rests on it.** The
lens was fitted on the bfloat16 `Qwen/Qwen3.5-4B`; we apply it to a 4-bit conversion. P0 of the
paper-reading memo proposed measuring that gap and the hosted-lens run did not: its transfer check
was Base against post-trained, both 4-bit, which is a different question. This is not a reason to
stop, and I am not asking for the full corpus stage on a bfloat16 checkpoint that is not cached
and would cost a 9 GB pull at the rates the memo recorded. A bounded version is enough: a few
tens of prompts through the bfloat16 post-trained model, hosted-lens top-k agreement and rank
correlation against the 4-bit run at band layers. Until it exists, every WP7 and WP9 reading
carries the conditional, and the report should say so in one line rather than leave it implied.

Two smaller points, not conditions. The registry `lens:` block must record which of the two lens
files was used, since the 417-prompt and n1000 files differ, and a later result must not be
silently compared across them. And instrument comparisons, hosted against logit lens, are
descriptive: they sit outside every corrected family, including the comparison the memo used
evidentially to retire the finite-difference estimator.

## WP5. Verdict: do not run as specified; it measures the channel we already understand

**B1. The design objection stands and WP3 sharpens it rather than resolving it.** The Chief's own
earlier objection was that an open render measures attention's retrieval plus the recurrent state,
so the expected distance curve is near-flat. WP5 keeps the open render. If WP3 confirms F4, that
objection becomes a prediction: attention's native window is 262144 tokens and our longest level is
2048, so attention retrieves at every level and the curve is flat by construction. EXP-003 as
specified would then measure attention's retrieval, which is not in doubt, and would say nothing
about the recurrent horizon, which is the question.

The experiment that separates the channels is EXP-003's distance axis run with the in-band
attention blocks' access to the distant span ablated, leaving the recurrent path as the only
carrier. That is EXP-002's arm-A machinery on EXP-003's axis, and it is a masked arm, not a new
experiment. My verdict is therefore conditional and ordered, and the Chief has accepted the ordering: WP5
does not run before WP3 reports, while the re-specification proceeds meanwhile so the dry run is
ready when the curve is;
if WP3 confirms F4, WP5 acquires a masked arm or it is withdrawn; if WP3 refutes F4, the open
render is informative and WP5 runs as written.

**B2. Settled by R41b.** Raised: under R40b the layers were 13, 16, 20, 24, 28, whose writer
kinds are linear, attention, attention, attention, attention, so four fifths of the readouts were
the residual immediately after a full attention block and layer 13 was the only recurrent-output
readout. An experiment whose whole question is which channel transports content cannot read four
fifths of its layers at one channel's output. Settled: R41b replaces the set with the kind pairs
12/13, 15/16, 19/20, 23/24, 27/28, one DeltaNet-written and one attention-written layer at each
depth, named as pairs in the registry and reported per pair. The confound is removed at the source
rather than annotated, and WP5 no longer needs the partner set carried separately.

**B3. Settled by R41b.** Raised: the partner rule had no answer for layer 13, whose nearest
attention-written neighbours are layer 12, one below the band's onset, and layer 16, already a
primary. Settled: R41b admits layer 12 as 13's pair. That is the option I preferred, for the
reason given, that a labelled out-of-band partner is honest and an unmatched primary silently
reintroduces the confound the rule exists to remove. Tables should continue to note that 12 sits
one layer below the persistence onset.

**B4. Two statistical specifics.** The paired Wilcoxon is on log-rank, which is undefined for a
censored observation, so the spec must require full-vocabulary ranks rather than a top-k dump. The
memo says full-vocabulary ranks at every layer are cheap, so this costs nothing and removes the
censoring rule entirely. And "recovery depth, the first band layer at which the target enters the
top 25" reintroduces a threshold that full-vocabulary ranks make unnecessary; keep the continuous
rank as the primary outcome and report recovery depth as descriptive, over five band layers plus
"never", with no significance claim attached to a six-valued ordinal.

**B5. One thing WP5 gets right that should not be lost.** The levels, 32 to 2048 geometric, now
sit inside WP3's gap bins, including the top level, which fell outside them until the bins were
extended to 2688. The per-kind predictions the plan wants from WP3 are therefore available at
every level, which they would not have been.
