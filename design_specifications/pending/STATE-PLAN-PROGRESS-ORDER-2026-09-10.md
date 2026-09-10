# Work order: the second state variable, plan progress — rulings on the D-CRO's draft `ca04182`

**Chief, 2026-09-10. Cites plan §16.18 and the D-CRO's draft
`research/records/STATE-PLAN-PROGRESS-2026-09-09/README.md` at `ca04182` on `cuda-ws-d`. Owner of
the pre-registration and the capture plan: the D-CRO. The readings are the seats', once the 12B
lens and the 12B dictionary readout exist. This order settles every item the draft marked open and
adds three facts the draft does not carry, each measured on the rendered corpus before any capture
exists.**

The draft is accepted as the frame: three hypotheses that fail differently, both required rules
carried, decision positions and not every position, the assumptions marked as assumptions, the
TiB-for-GiB correction kept in the record rather than erased. What follows changes what the draft
says is testable, and it does so now, before a capture is read, because that is the only time it is
cheap.

## 1. Three facts about the corpus, measured today, that the pre-registration must carry

Measured on `data/agent_v2e-gemma3-4b/{train,valid,test}.jsonl` — 8,907 rendered rows, 1,128 tasks —
by the Chief on 2026-09-10, on the laptop, from the rendered rows and `pipeline/tasks.py`
(`Task.horizon` is `len(steps)`). The D-CRO re-runs the same count on the device copy and puts the
script in the record.

**Fact 1. The step index is the turn count.** Under teacher forcing on the expert corpus every
rendered row is one decision, and the number of model turns already in the prompt *is* the step
index. A probe that cannot read it is broken. It is a control, not an estimand.

**Fact 2. The horizon is a function of (family, difficulty, variant) for eleven of twelve
families.** Within each such triple every task has the same horizon: `aggregate_report` at
difficulty 2, clean, is 14 steps for all 15 tasks; `ledger_reconcile` at difficulty 2 is 12; and so
on through the table. The one exception is `pointer_chain`, whose horizon varies with the chain
length, which is not in the task prompt and is not knowable to the model until the chain ends.
Family and difficulty are in the task prompt; the variant becomes visible at the turn its
perturbation lands. So "steps remaining" is a transcript function everywhere it is knowable at all.

**Fact 3. The protocol writes the plan into the transcript.** The expert's progress note carries the
plan and its progress in words at every step — *"Plan: list the ledger, read every invoice, sum
approved amounts only, inspect the summary, replace PENDING, re-read to verify, report
approved_total. Listing."*, then *"Invoices read: 0 of 6 … pending: invoice-1-282.txt, …"* — and the
protocol hides tool results older than `keep_last` turns (`pipeline/branch.py`, `DEFAULT_KEEP_LAST`)
and tells the model the note is its only memory. 4,509 of the 6,501 consecutive row pairs are **not**
prefixes of each other: the prompt is re-rendered every turn with earlier results elided. The
transcript is the explicit carrier of state by design. That is the notes-and-windowing paradigm,
not an accident of this corpus.

**Consequence.** E1 as drafted — the gain over a predictor given the visible transcript — is
**untestable by construction** for plan progress on this corpus. The strongest transcript predictor
is exact for eleven families, and the twelfth is unknowable to everyone, the model included. The
pre-registration records E1 in that form as `untestable` in the three-state reporting. It does not
measure it against a weakened baseline and call the gain a finding.

## 2. Rulings on the open items

**R1. Population.** The expert corpus, teacher-forced, all three splits and all variants. It is the
only population in which the same token sits at the same index in both models' inputs and the
ground truth is exact by construction. The prefix-matching question dissolves under teacher
forcing: every episode matches the expert's prefix. Rollout populations, where the model's own plan
is a real latent and the informative episodes are the ones that *leave* the expert's prefix, are a
later order and are not captured in this pass.

**R2. E1, re-scoped and labelled.** E1 becomes *decodability at matched rank*, reported as a depth
profile against two nulls on each model: the **permutation null** (labels permuted within family)
and the **step-0 null** (the same probe read from the residual at the first decision of the same
task, which carries what the task prompt alone tells — family, difficulty — and nothing of the
progress). It is a comparison of encoding, not of representational richness, and every figure
carries that label. The `pointer_chain` family's steps-remaining is a **negative control**: it must
sit at the permutation null on both models, and a probe that "decodes" it is reading a leak.

**R3. E2 is the primary estimand, and the perturbed variants are its informative stratum.** A
perturbation changes the horizon by a known amount at a known turn: from the table, `failed_edit`
and `stale_path` add two steps, `transient`, `unknown_tool` and `wrong_path` add one. The update rule
must move the read state by one step on an ordinary turn and by the recovery cost at the
perturbation turn. A (family, step) lookup does not do that, so this is a dynamic on which the two
models can differ at matched capacity and which §1 does not trivialise. Closure is measured as the
derivation says, against the pilot's tolerance, held out by episode. The test split is clean
variants only; E2's stratum is the train split's perturbed variants, so its held-out split is by
episode within the train split, stratified by family and variant.

**R4. E3 stays ordered after the cache-exchange port, and its meaning is narrowed.** Under
`cache.strategy: none` — the only strategy the golden records exercised, and the registry's setting
for both Gemma entries (`models.py`, `cache_strategy`) — every turn is a fresh pass over the
re-rendered prompt. Across-turn maintenance is therefore impossible by construction, and
"maintained rather than recomputed each turn" cannot be tested on this protocol. What E3 can test
is retention *within a pass, across positions*: whether the state at the decision position depends
on the retained activations of earlier positions or is recomputed from the recent tokens, bounded
by the sliding window (local layers see 1,024 tokens; the global layers see everything). The
pre-registration states H2 in that form. The Director's across-turn form needs a cache-reuse
strategy, which is an equivalence claim the registry marks UNRULED, and it is not this order's.

**R5. Capacity rule: fixed rank, with the null as the measurement of "more of anything".** A
pre-registered rank ladder, r ∈ {1, 2, 4, 8, 16, 32}, identical on both models and on the controls.
At every rank on every model the permutation null is fitted the same way, and every gain is
reported beside its own null; that null *is* the measurement of how much a wider, deeper stream
yields to any probe, and it makes the confound a number rather than a caveat. Fixed parameter count
runs as one robustness row at r = 8. Matched control features are the controls, required
regardless, and are not a capacity rule. Depth is read at the registry's six `layer_fractions` for
the estimands, with the full profile reported; no layer is selected after the fact. Held-out splits
are by episode, stratified by family, never by position. A figure without its rank and its null is
refused.

**R6. Capture plan.** Decision positions, every layer, bf16 native, both models, the same rendered
rows. The decision position is the last prompt token of each rendered row — the position whose
readout produces the first token of the model's turn. Because the rows are not prefixes of each
other (Fact 3), the capture is **one pass per rendered row**, not one per episode: 8,907 rows per
model, median prompt about 1,200 tokens, the longest about 3,600. The draft's assumed counts are
replaced by the measured ones:

| | assumed | measured |
|---|---:|---:|
| episodes | 300 | 1,128 (840 train, 48 valid, 240 test) |
| decisions per episode | 12 | mean 7.0–7.4, max 17 |
| decision positions | 3,600 | 8,907 |
| both models, decision positions, bf16 | 1.86 GiB | 4.6 GiB |

against 235 GiB free on the card at 10:37Z. One exploratory stratum: every position of the **final**
row of twelve pre-registered episodes, one per family, test split, difficulty 2, clean — about 12 GiB
for both models at typical length, under 25 GiB at the longest — labelled exploratory and read only
after the seal. Captures live outside the checkout under `/workspace/captures/<entry>/`, with the
manifest fields the draft lists, `capture_dtype: native`, and the ground truth beside each cell
from the task, never from the model. Rows are enumerated through `task_id` and `step`; the 240
train, 60 test and 48 valid rows that carry no `family` key are named in the pre-registration for
what they are before they are captured or excluded.

**R7. Seal.** `preregistration.json` by the existing machinery (`state_programme/seal.py`), written
before any capture file is opened for reading. It carries §1's three facts, R2's labels, R3's
stratum, R4's restated H2, R5's ladder, and the twelve exploratory episodes by `task_id`. The
captures may be made first, as §16.18 requires; the reader refuses without the seal.

**R8. Section 6 of the draft: agreed, and sharpened.** After the capacity rule and the controls, an
E1 advantage is still not "a richer representation of the task"; on canonical plans under teacher
forcing it is "encodes (family, step) more linearly". The Director's hypothesis in its strong,
across-turn form is untestable on the protocol as built (R4); its testable forms are E2 on the
perturbed variants and the within-pass E3. And H1's falsifier includes the 4B being *more*
predictive, which is a live outcome and not a failure of the instrument.

## 3. Schedule, on the card

Rows two and three of the golden test follow row one, combined by the weighted mean, the manifest
carrying the measured 48.23 s per row per layer and the fact that three rows were met by extension.
Then the 12B smoke row at the reduced cotangent batch, its peak written beside the 4B's 43.9 GiB at
`dim_batch=64`. Then the 12B lens fits **with the 12B captures of R6 in the same pass**, and the 4B
captures as a short pass beside them; neither measures time, so the shared-card basis is allowed
and is recorded per cell. The pre-registration is sealed before the first capture is read, and the
readings wait for the 12B lens and the 12B dictionary readout, as §16.18 says.

## 4. Rules of this order

The draft stays a record README and is not edited to match this order; it is cited by commit, and
the pre-registration is the document that carries the rulings. Builders' notes go in the record.
Nothing in this order is a result.

## Amendment, Chief, 2026-09-10 — the count reproduced at `3f6516b`, with two corrections to §1 and four additions, all ruled

**My count had an unstated basis, and it was my error.** The script that produced §1 keyed rows by
`(task_id, step)` in a dictionary and so overwrote every row sharing a key. The D-CRO's
`count_corpus.py` reproduces nine of the eleven figures to the digit on the raw corpus and the other
two on the deduplicated basis I used without saying so: 930 of the 8,559 rows carrying a `task_id`
share their key with another row; deduplicated, 7,629 rows, 6,501 consecutive pairs, a maximum of
17 decisions. Both bases are in the record; the non-prefix fraction is 69.4% on mine and 60.7%
raw, and Fact 3 holds on either. §1's figures stand with their basis named.

**A1. The duplicates are the transient retry, and R3 gains a clause.** A `transient` task
re-issues the same action and the corpus labels both rows with the same step index, correctly: a
retry is the same plan step re-executed, and progress does not advance across it. The update rule
of E2 therefore predicts **no advance at a retry**, +1 on an ordinary turn, and the recovery cost
at a perturbation. The 244 transient tasks are the stratum that refutes any rule that advances
every turn, and they are named as such in the pre-registration.

**A2. Not every decision is a rendered row.** 334 tasks are missing exactly one step, and the split
by variant is total: `clean` 0 of 550, `transient` 0 of 244, `failed_edit` 58 of 58, `stale_path`
30 of 30, `unknown_tool` 64 of 64, `wrong_path` 182 of 182. The renderer drops a row whose
observation begins `ERROR` unless the step is an injected fault. Fact 1 survives, because the
dropped turn is still in the context and the turn count is still the step index. What does not
survive is any horizon inferred from a row count, which is one short on 334 tasks: **the horizon
comes from the task definition, `Task.horizon`, never from counting rows**, and the
pre-registration says which decisions of each task have no rendered row.

**A3. R6's addressing was ambiguous.** `(task_id, step)` collides 930 times. The capture key is
`(task_id, step, occurrence)` with the occurrence index counted in row order within the task, plus
the capture-time row ordinal and the sha256 of the rendered prompt, all in the manifest. This is
fixed before a single capture is written under it.

**A4. The rendered corpus is not on the card.** `data/agent_v2e-gemma3-4b/` (three splits, about 85 MB on disk; the D-CRO reported 169 MB and the archive manifest settles it)
is a blocking input to R6 exactly as the lens corpus was. It ships outside the shared
checkout at `/workspace/rendered-corpus/agent_v2e-gemma3-4b/`, through `lab-device pack-data` and
`verify-data`, never a bare `tar`; the archive's digest goes into the pre-registration so the
captured rows are the counted rows. Shipping is the Chief's, tonight, no GPU.

**On the pre-registration:** draft it now, on the laptop, carrying §1 with its basis, A1–A3, R2's
labels, R4's restated H2, R5's ladder and the twelve exploratory episodes by `task_id`, and the
corpus digest from A4 once it exists. **Do not seal it** until the golden-test design has landed
and the capture pass is scheduled: the seal is the last act before a capture is read, and the two
streams share the machinery but not the clock.

## Rulings on the pre-registration draft, Chief, 2026-09-10 — `6a118fd`/`96d608c` on `cuda-ws-d`, read in full

**A1 is withdrawn and A3 is revised, on the D-CRO's correction (§1.1 of the draft).** The 930
repeated `(task_id, step)` keys are byte-identical rows produced by the training mix's
`recovery_repeats` oversampling — 490 decisions doubled and 88 sextupled, exactly the 578 recovery
decisions and no other — not retries. The step index never repeats. So: E2's rule has no
no-advance case; **the capture key is `(task_id, step)`**, with `prompt_sha256`, the row ordinals
and the multiplicity `rendered_rows` in the manifest so the training weight is recoverable and
never applied by accident; **the capture unit is the distinct decision, 7,629**, since capturing
rows would forward the same tokens up to six times and weight every probe fit by the fine-tuning
recipe's oversampling in the direction that flatters a recovery result; the storage table is
**3.95 GiB for both models**; and the 348 rows without a `family` are chat replay rows with no
task, step or horizon, **excluded**. My §1 figures stand on the deduplicated basis, which turns out
to be the right basis for a reason neither of us had. The inference I accepted in A1 was the
twenty-fourth entry's shape — a mechanism read from a pattern when one comparison of two rows would
have refuted it — and it is the D-CRO's to write up, as they have.

**R3's stratum, restated from the transition census (§4.2), accepted.** 6,501 transitions: 5,948
ordinary, 244 into a corrective decision entered contiguously, 309 into a corrective decision across
a step with no rendered row. Every gap transition is a recovery transition, and a gap transition is
never scored as an ordinary +1. The transient variant is the only one whose corrective decision is
entered contiguously, because its injected fault keeps its error row; its 244 transitions are the
stratum where the recovery cost is read without an unobserved decision between, and the 309 others
are the stratum where it is not — reported separately. The rule is fitted on ordinary transitions,
held out by episode within the train split and stratified by family, and every one of the 553
corrective transitions is evaluated out of sample; that generalisation is the claim. The 25
`wrong_path` episodes whose fault lands at step 0 contribute a decision to E1 and no transition to
E2, named so that 578 against 553 is not read as a loss.

**§13.2, E2 scored as a retrieval hit against an explicit chance level: accepted.** The transported
state is scored a hit if it is nearer the state at *k+1* than to the state at any other decision of
the same episode; chance is one over the episode's decisions, per episode, beside the score; the
bounded contrast is what the drop rule reads. A retrieval score has no free parameter to set after
looking and needs no pilot, which is why it is right here and the derived tolerance was right for a
continuous state. Condition: the transport distance itself is reported descriptively beside the hit
rate, so the two state variables keep a common quantity.

**§13.3, ε_sub = 0.11: accepted**, written before the run, with ε_main 0.06 (n 1,781 ≥ 1,715) and
ε_ord 0.04 (4,122 ≥ 3,859) beside it and the reason stated: 553 corrective transitions exist and no
split of them reaches n at ε_main. It is the budget rule's shape applied to a population limit.

**§13.4, the seal.** The golden-test design has landed (the WS-D ruling of this date). The seal
follows Codex's file-only review of the draft (WS-A order, next instruction, through the Director)
and precedes the first *reading* of any capture. It does not gate the capture pass: §16.18 lets the
captures be made first, and the 12B pass with its captures runs on the card's schedule — after the
calibration and the 12B smoke row — whether or not the seal exists yet. Reading waits for the seal.

**Precondition 1 of §11 is recorded as checked**: `prereg_inputs.py` re-run on the card against the
shipped corpus reproduces every figure and regenerates the capture set to `1a7cfbdd…9709a`, CPU only.

Everything else in the draft is as ordered: the headline at r = 8 and the 0.5 depth fraction fixed
in advance, the twelve exploratory episodes chosen by rule, M = 12 fixed, the three-state reporting,
and the sentence that a first result showing the 12B more predictive is, until the capacity rule
and the controls, a measurement of the 12B being bigger.

## Correction before the seal, Chief, 2026-09-10 — the unit of the bound is the episode

From Codex's file-only review of the draft. §7 of the pre-registration computes n from decisions
and transitions — 1,781, 4,122, 553 — but the design holds out by episode precisely because
decisions within an episode are not independent, and the first state variable's bound counted
episodes (n = 77 and n = 82 in its order). **Ruled: the unit is the episode, and the bounded
observation is the episode mean.** At M = 12 and α = 0.05: E1 on the test split's 240 episodes
gives ε_main ≈ 0.16; E2's ordinary transitions in the train split's 840 episodes give ε_ord ≈ 0.09;
the corrective transitions are one per episode, so ε_sub = 0.11 stands. Both computations, the
reason and the consequence for the claim are written into §7 before the seal; an episode-level
bootstrap may be declared beside the Hoeffding form but does not replace it. The capture set is
pinned by both digests, the file's sha256 and the logical triples', each named for what it is.

## Rulings on Codex's review of the draft (`011e6c7`, `WSA-REVIEW-2026-09-09-1233Z`), Chief, 2026-09-10 — the last items before the seal

**P1.** The capture contract (§2, §8) declares `forward_batch: 1` and stores the actual token index of
each captured cell after tokenisation, beside the prose definition of the decision position.

**P2.** E2's retrieval is frozen: the candidate set is **all N decisions of the episode, the source
included**, so the declared chance level is 1/N as drafted; the rule is strict nearest-target and
**a tie is a miss**, named so an implementation cannot substitute an argmin tie-break; the transport
distance is reported descriptively with its metric, the same metric the transport rule is fitted
under, aggregated as the per-episode mean and then across episodes. Uniform guessing is a declared
reference, not a measured geometric null, and the draft says so.

**P3.** §13 is rewritten to the current sequence: resolve these items, Codex's re-review if the
Director asks for one, seal, first reading. Capture is permitted before the seal, as before.

**P4, the statistical ruling Codex asks for.** The target is **inference to new episodes of the
same task distribution**, not a description of this corpus; so the unit is the episode, the bounded
observation is the episode mean, and the concentration bound is over independent episodes (the
correction of this date: ε_main ≈ 0.16 on 240, ε_ord ≈ 0.09 on 840, ε_sub = 0.11 on 553). The
mandatory corrective subgroups carry their own figures — about 0.16 on the 244 contiguous and
0.14 on the 309 gap transitions — and the pooled 0.11 attaches to the pooled set only. An
episode-level bootstrap is declared beside the Hoeffding form for the drop rule's contrast; it does
not replace the bound. The approved decision-count epsilons stay visible, marked *coverage not
supported at the episode unit*. Independence between episodes is an assumption stated, not shown:
episodes within a family share a template.

**P5.** Every held-out prediction is held out by episode through **cross-fitting**: K = 5 folds by
episode, stratified by family and variant, seed fixed in the seal, applied to every learned stage —
the representation probe and the transport rule alike — so that each of the 4,122 ordinary and 553
corrective transitions receives a prediction from a model that never saw its episode. The shared
training across folds is a stated limitation of the bound and is what the episode bootstrap is
declared to address. Folds and seeds are in the seal.

**P6.** Both digests, as ruled: the logical-triple digest with its serialisation schema, and the
file's byte digest.

Everything Codex lists as carrying correctly carries. The seal follows these edits and precedes
the first reading; capture does not wait for it.

**Ruling, Chief, 2026-09-10, from the same external review (see the run order of this date):** the
carrier-ablation arm and the re-read diagnostic apply to this variable as they do to the first. Here
the carrier is the note's plan and progress in words (Fact 3), so the ablation removes the plan's
progress statement from the note after the state-bearing position; E2 is scored with and without it,
and R4's within-pass E3 is no longer the only place the carrier question lives.

## Rulings on Codex's re-review (`WSA-OUTSTANDING-REVIEWS-2026-09-10`, `89268e3` on its branch; standing request 1), Chief, 2026-09-10 — two corrections before the seal, both accepted

**S1, accepted.** The pre-registration carries two operative tolerance tables: the corrected paired-width values
(E1 test 0.23, ordinary train 0.13, corrective pooled 0.15, contiguous 0.23, gap 0.20) and, unmarked, the earlier
ε_main = 0.17 / ε_sub = 0.11 with a pooled 0.11 reference; under the document's own paired-width rule 0.17 needs
428 episodes and 0.11 needs 1,021, beyond their populations, so the ambiguity can change a null or equivalence
verdict. The checker passes both, and passes an inserted ε_main = 0.01. Ruled: one operative table, named as
such; the intermediate paragraphs and the pooled reference marked superseded in place, the historical
decision-count figures kept as history; the checker gains a test that refuses two competing operative values
for one tolerance (insert 0.01 beside 0.23 and it must fail). The seal follows this edit.

**S2, accepted.** The instrument veto "any above-permutation decoding of pointer-chain remaining steps is a leak"
is wrong at the terminal step: every terminal pointer-chain decision (94 of 669; 20 in E1's test split) visibly
contains `Node: final`, and recognising that a chain has ended is legitimate input information, not leakage.
Ruled: the unknown-horizon control pre-specifies its eligible positions — terminal decisions excluded, and the
exclusion rule stated as a function of the visible text, not of the label — and its null is conditional on
what remains visible (chain length, survival, family, visible progress), with the reason each conditioning
variable is included written down; success above that null is the veto, success above the unconditional
permutation null is not. Exclusion alone is not claimed to make the remaining labels unknowable.

Both go to the D-CRO as the pre-registration's owner; Codex's other checks carry (72 CPU tests, the folds
reconstructed independently, 7,629 token-position checks). The seal waits for S1 and S2 and nothing else on
this order. The sealed-off activations stay unread.

*2026-09-10 UTC, 10:50Z:* S1 and S2 applied by the D-CRO at `e69098e` on `cuda-ws-d`, each verified against the corpus and the document before acceptance: one operative tolerance table with the superseded paragraphs marked in place and a `competing_tolerances()` check that marks by paragraph and refuses a second operative value (the ε_main = 0.01 probe and ε_sub = 0.11 both fail, the clean document passes); eligible positions for the unknown-horizon control decided by a rule on the visible text (`Node: final` present in all 94 terminal pointer-chain prompts and in 0 of 575 non-terminal ones), with a null conditional on chain length so far, survival, family and difficulty, and the visible progress note. 68 claims check; the self-test corrupts 56 pinned lines with no survivors; exit codes inspected, not piped. The pre-registration is ready to seal; whether Codex takes a confirming pass first is the Director's call, and both the D-CRO and the Chief recommend that it does, the new checker being an hour old.

## Ruling on Codex's confirming pass (`WSA-SEAL-CONFIRMATION-2026-09-10`, `be421b7`), Chief, 2026-09-10 (~11:10Z) — the seal is held; three corrections

Codex confirms the threshold prose and the terminal-position exclusion (its own count: 94 terminal and 575
non-terminal pointer-chain decisions; 20 terminal and 141 eligible in the test split) and finds three
remaining defects, each with an executable counterexample. All three accepted; **the seal is held until they
are applied and Codex has confirmed once more.**

**C1 — the contradiction checker still passes contradictions.** A duplicate corrective-table row changed from
0.15 to 0.11 passes; a contradictory declaration split across two lines passes; and `--self-test` never
exercises `competing_tolerances()`. Ruled: the checker parses every tolerance declaration in the document into
one table keyed by population — table rows, prose declarations and multi-line declarations alike — and refuses
if any population carries two distinct operative values; the self-test carries both of Codex's counterexamples
as pinned corruptions that must fail, beside a clean document that must pass. A detector the self-test never
runs is not a detector (the D-CRO's own rule of the day: ask what the check would say if the thing it guards
were broken, then break it).

**C2 — §12 still carries the old veto.** It says above-permutation pointer-chain decoding invalidates the
instrument, contradicting §4.1's corrected conditional rule. Ruled: §12 restated to the conditional rule on
eligible positions, and the contradiction checker extended to this pair as well — one veto, stated once,
referenced elsewhere.

**C3 — the conditional null has inputs but no algorithm.** Ruled: the null is written as a procedure — how the
conditioning variables are binned or modelled, how the permutation or resampling respects them, how many
draws, and the comparison rule with its threshold — reproducible by a reader from the text alone; until that
text exists the control is marked *unmeasured* and the veto cannot fire. Beating an unspecified baseline does
not establish leakage.

To the D-CRO as the pre-registration's owner; Codex's confirming pass again after the edits; then the seal.
Nothing on the card; the sealed-off activations stay unread.

*2026-09-10 UTC, ~11:25Z:* C1, C2 and C3 applied by the D-CRO at `8407ab2` on `cuda-ws-d`, every one of Codex's counterexamples run against the result: the contradiction detector now parses (table rows split positionally, prose read by sentence, tolerances keyed by population including subgroup, the historical exemption scoped to its own sentence or row); `--self-test` and the normal command run one path, so the self-test exercises the detector (five contradiction counterexamples pinned, plus a wrapped-claim mutation); §12 references §4.1 and says beating the unconditional null does not invalidate the instrument; the conditional null is a procedure — strata on (difficulty, nodes visited so far) extracted from the rendered prompt, permutation within stratum, 10,000 seeded draws, strata under ten dropped and reported, comparison at §7's ε and α — reading *not measured* until executed, the veto unable to fire, and an excess read as evidence beyond those four variables, not proof of leakage. 68 claims check; 61 pinned claims and five counterexamples corrupted with no survivors; exit codes inspected. **Ready for Codex's second confirming pass; the seal follows it.**

## Ruling on Codex's second confirming pass (`WSA-SEAL-CONFIRMATION2-2026-09-10`, `95b8a3d`), Chief, 2026-09-10 (~11:35Z) — the seal still held; the control is deferred so that the seal can follow

The earlier counterexamples now reject. Three issues remain, all accepted, and Codex's containment is adopted.

**D1 — the conditional control as written is statistically unfit.** Its stratum rule keeps 15 episodes and 100
decisions, all at difficulty 2, yet borrows a tolerance derived for 240 episodes; it treats the visible note as
determined by node count although those notes carry different paths; and decision-level permutations break the
episodes' shared horizon. Ruled, as Codex proposes: the unknown-horizon control is **deferred** in the sealed
document — declared, its procedure marked provisional, its veto **disabled** — and its final procedure
(episode-level permutation that respects the shared horizon, its own tolerance derived from its own
population, strata that respect the note's content) is settled by an amendment that Codex reviews before any
of its results are read. The seal does not wait for that design; the reading of that control does. No new
capture is needed.

**D2 — two verifier gaps.** A contradictory table row passes when its value is not bold; restoring the old
unconditional veto passes both the normal command and the self-test. Ruled: the parser reads values regardless
of emphasis (bold, italics, code), and the veto's single statement is a pinned claim whose unconditional form is
a corruption the self-test must reject.

**D3 — flagging still becomes falsification.** §4.1 says exceeding the null triggers investigation; §12 says
firing falsifies the instrument. Ruled: one consequence, stated once — an excess is an instruction to name the
cue and investigate, never by itself a falsification — with §12 referencing §4.1's text, and the contradiction
checker covering the pair.

To the D-CRO; Codex's third look on the three points only; then the seal.
