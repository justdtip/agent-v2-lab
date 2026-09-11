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

*2026-09-10 UTC, ~11:15Z:* D1–D3 applied by the Chief at Daniel's instruction, `6ee2563` on `cuda-ws-d` (branch `chief/prereg-d1-d3`, fast-forwarded into `cuda-ws-d` and pushed): the unknown-horizon control deferred with its veto disabled, its defects named in §4.1 and its final procedure left to an amendment reviewed before any of its results are read (§7 keeps M = 12; §10 and §13 carry the state); §12 states one consequence — a diagnostic flag that starts §4.1's investigation, never by itself falsification, with invalidation needing positive evidence of a failed measurement path; the checker reads each table's declared-ε column by its header regardless of emphasis and fails closed on a tolerance row without one, pins the veto's form in §4.1, §10 and §12, and refuses any sentence that makes decoding above the permutation null a falsification or a leak; the self-test carries Codex's unbolded duplicate row, the veto restored in two wordings, the veto added beside the new text, and each pin removed — 68 claims check, 67 corruptions caught, none survive; both entry points one path. **Ready for Codex's third look, on the three points; then the seal.**

*2026-09-10 UTC, ~11:30Z:* the D-CRO's review of 6ee2563 (three amendments, all accepted and applied by the Chief at `4977111` on `cuda-ws-d` (rebased onto the D-CRO's method entry 4a08374)): the declared-ε header rule had never executed (a separator already ending in a pipe defeated the match) and now resolves §7's columns; the unconditional-veto regex is stated as a backstop with negation honoured only in the verdict's clause, and catches four further wordings the D-CRO put to it; the DEFERRED status now precedes the provisional procedure, with steps 1 and 5 marked inline. 68 claims; 71 corruptions, none survive. This is the commit for Codex's third look.

*2026-09-10 UTC, ~11:40Z:* **Codex approves sealing** (`ba7b7b9`, `WSA-SEAL-CONFIRMATION3-2026-09-10`) at `6ee2563`, the commit of its third look: the 67-case self-test passes, its 24 negative cases reject through both entry points, deferral, diagnostic consequence and M = 12 consistent throughout; one non-blocking verifier defect — a valid table with separate exact and declared columns is rejected — which is the inert header rule the D-CRO found and `4977111` fixes. **Ruling: the seal is taken at `4977111`, not `6ee2563`**, because a seal digests the text it seals and the branch has moved by the three reviewed-by-the-D-CRO amendments; Codex confirms the delta `6ee2563..4977111` (the document reordering in §4.1 and the checker's header rule, backstop regex and pins — small) in a delta look, then the D-CRO seals at `4977111` and the permitted readings begin. The deferred control's amendment remains a separate reviewed item before any of its results are read.

*2026-09-10 UTC, ~11:50Z:* the D-CRO's two corrections to 4977111 applied by the Chief at `acc130a` on `cuda-ws-d`: an inverse counterexample that must not fail — a trailing decimal cell on an operative row, ignored by the live header-column rule and a false failure under the fallback, so the suite now notices if the rule dies (proved by killing the rule in a scratch copy) — and the passive wording "the instrument is invalid" caught. 68 claims; 73 corruptions and inverse cases, none survive. **The seal is taken at `acc130a`; Codex's delta look covers `6ee2563..acc130a`.**

*2026-09-10 UTC, ~11:55Z:* the seal builder `make_seal.py` is on `cuda-ws-d` at `fbb5192` (prepared, not written): every sealed quantity recomputed rather than copied — the folds re-derived from `capture-set.jsonl` and required to match `folds.json` and §7.1's digest (c5e9622f…), the seed read from the run, the capture set's two digests kept apart (triples 1a7cfbdd…, file 8bbc8062…), every ε recomputed from the bound at M = 12 and α = 0.05 at the paired width and required to reproduce needs-n and spare (234, 731, 549, 234, 309), the document required to pass its own checker, thirteen files digested, `--baseline` refusing on any byte drift, `--write` refusing to overwrite. Ruled: **the seal takes the branch tip** (the text unchanged since 4977111, the checker the corrected one); **Codex's delta look is `6ee2563..fbb5192`**, covering the checker corrections and the builder; the D-CRO seals with `--baseline` at the then-tip on the Chief's relay of Codex's confirmation.

*2026-09-10 UTC, ~12:05Z:* the seal is prepared at `acc130a` from the tip `0de825a` (the two later commits builder-only; `--baseline acc130a` finds all thirteen sealed files byte-identical), digest 58734285b8cd…, deterministic across runs; the builder is recorded beside the seal as `produced_by` and is not itself sealed — it postdates the reviewed text. Ruled: the seal names acc130a; Codex's delta look is `6ee2563..0de825a`; and **the seal-consuming reader is built before the first reading** — precondition 5 (no capture is read unsealed) becomes a function that raises unless the seal's digest, its baseline files and the document's own checker all agree, not a sentence in the text. `--write` waits on the Chief's relay of Codex's confirmation.

*2026-09-10 UTC, ~12:12Z:* the tip is `4dc39d5` (the builder resolves `--baseline` to a full object name before recording it, so the seal digest no longer depends on how many characters of the commit were typed; the method record's tenth instance at 3d5ed51); no sealed file moved. **Codex's delta look is `6ee2563..4dc39d5`**; the seal is taken at the then-tip with `--baseline` set to it; the seal-consuming reader precedes the first reading.

*2026-09-10 UTC, ~12:20Z:* **Codex's delta review carries the approval forward to acc130a** — the 73-case self-test passes, all 29 reviewed bad edits reject through both entry points, and disabling the header rule makes the new positive case fail; "no further review hold on this seal". Relayed to the D-CRO: seal at the tip with `--baseline` set to it, commit seal.json, build the seal-consuming reader, then begin the permitted readings. The deferred control's amendment remains separate.

*2026-09-10 UTC, ~12:30Z:* the seal-consuming read gate is built at `f2a4265` (`state_programme/read_gate.require_seal`): five claims each on its own path — the seal exists and is complete, its bytes digest as expected, every sealed file still has its bytes, those files still match the baseline commit, and the sealed document still passes the sealed checker loaded from the sealed `check_prereg.py` — no argument disables a check; fifteen tests, two isolating single paths (a document matching its digest but failing its checker; a file moved after the baseline with the seal digest updated); git calls through `spawn.run` after the fork guard caught `subprocess` under a loaded model. The seal digest 58734285… is superseded by the baseline normalisation (4dc39d5); the final digest is reported at sealing. Relayed: seal now at the tip with `--baseline acc130a`.

*2026-09-10 UTC, ~12:45Z:* **SEALED.** Seal digest `998b3bcafa9d6aaffa21ebd43df3935b1634ba7b12fe187073837acd942ce521`, written at `f2a4265` with baseline `acc130a4c2057717c462564e5f4ed5dc35fa60ed` (the stronger form: all thirteen sealed files byte-identical to the commit Codex reviewed; the tip recorded beside it), committed `5525ac8` and pushed; verified through `require_seal` on the real seal, refusing by name on any other expected digest; the seal fixes the seed 20260910, the fold assignment c5e9622f… recomputed, both capture-set digests, the five operative tolerances recomputed from the bound, and thirteen file digests; the builder recorded as `produced_by`, not sealed. The permitted readings run on the card through the gate, under the shared-card rule beside the Chief's 12B masking pass and 4B repeat; the deferred control untouched.

## First sealed reading, E1, and three rulings — Chief, 2026-09-10 (~13:15Z; the D-CRO's record §0.1, `5b1c9a1`)

The seal travels: `require_seal` passes on the card's own clone with the baseline re-derived from its objects.
Both captures are the sealed set, 7,629 of 7,629, precondition 2 re-derived on the card (identical ids and read
position at every decision). E1 at the pre-registered headline (§5: rank 8, depth 0.5), fitted on the train
split's 5,515 decisions and scored on the 240 clean test episodes:

| quantity | probe | over the permutation null | against ε_main = 0.23 |
|---|---|---|---|
| 12B step index | 0.789 | 0.646 | resolved |
| 4B step index | 0.732 | 0.611 | resolved |
| 12B steps remaining | 0.325 | 0.073 | not resolvable |
| 4B steps remaining | 0.266 | 0.030 | not resolvable |
| 12B − 4B, step index (paired) | | 0.036 | not resolvable |
| 12B − 4B, steps remaining (paired) | | 0.043 | not resolvable |

The residual at mid-depth carries the step index on both models; it carries nothing about the horizon this corpus
can distinguish from zero; and the difference between the models lies inside the resolution — the hypothesis is
unresolved at this n, neither supported nor refuted, which is what the document's opening said a null would mean.

**Ruling 1 — the r = 32 figure is not a finding.** The 4B reaches 0.270 on steps remaining at rank 32, above
ε_main, where the pre-registered rank 8 does not. Reading it would be choosing the rank after seeing the answer,
the thing §5's fixed headline exists to prevent. It stays in the record marked in place as a figure that must
not be read as a finding; a claim about it needs its own pre-registration, with its own seal, before any capture
is read for it.

**Ruling 2 — the r = 1 → r = 2 drop stays on the open list, as a diagnostic.** Step index falls from rank 1 to
rank 2 on both models (0.419 → 0.215, 0.434 → 0.223) and then climbs. The rounding hypothesis (exact match on an
integer moving across boundaries as a second component enters) may be checked as a diagnostic on the training
folds — never on the test episodes — and reported as a diagnostic, not as an explanation, until it is.

**Ruling 3 — the supervised subspace stands, and is now fixed.** §5 fixed the rank and not its construction; the
D-CRO built the rank as a supervised subspace (PLS1, greedy, nested, the first r of 32 components equal to the
r-component fit, pinned by a test), so that r measures the capacity the probe is allowed rather than the share of
the stream's variance, and the wider 12B residual is matched on capacity rather than penalised for its width.
Accepted, for the reason given, and fixed for E2 and every later estimand: it is now part of the instrument and
cannot change after a result is seen. Both nulls were fitted the same way at every rank, as §4.1 requires.

E2 next; E3 waits on WS-B; the deferred control untouched.

## E2's transport rule cannot pass by construction — rulings, Chief, 2026-09-10 (~13:50Z; the D-CRO's record §0.2, `e35644f`)

E2 as run returned a hit rate of exactly 0.000 in every stratum on both models, and the D-CRO reports it as an
instrument result, not a reading; correct. §4.2 puts the source in the candidate set, so a transport rule can win
only by moving more than halfway to the successor; the rule the D-CRO wrote as the literal reading of "advance by
+1, or by the recovery cost" — `T(x, m) = x + m·d`, `d` the mean per-step displacement of the fitting folds'
ordinary transitions — moves a fifth to under a third of the way (the mean displacement's norm is 313 against
per-transition displacements of 1,548 on the 4B; coherent fraction 0.202 on the 4B, 0.290 on the 12B), so the source
wins every time. The corpus is not the problem: with the source excluded and no transport, the true successor is
the nearest other decision of its episode in 0.430 (4B) and 0.455 (12B) of ordinary training transitions against
a chance of 0.172. §4.2 fixed the candidate set, the tie rule, the metric and the strata and not the transport
rule's functional form; the D-CRO chose the form in the open, saw the null and stopped rather than choose again.
That stop is the pre-registration working.

**R1 — E2 reads *not measured*, by instrument; the form is fixed by amendment before any corrective stratum is
scored again.** The amendment makes the transport a **fitted map at the matched rank**: the supervised-subspace
instrument of ruling 3 generalised to a vector target (the successor's residual), fitted out of fold on the
fitting folds' ordinary transitions, applied `m` times for a transition of cost `m` — the exact form, the
handling of `m > 1`, and whether the map is shared across depths, written down in the amendment and not
chosen after a score. The D-CRO drafts it; Codex reviews it file-only; it is sealed as an addendum with the same
builder against its own baseline; then the corrective strata are scored **once**.

**R2 — the 553 corrective transitions are not spent, and the reason is written down.** A rule that cannot pass
by its geometry conveys nothing about which rule would: the outcome was fixed by the displacement statistics of
the ordinary training transitions, not by anything in the corrective strata, and no choice about the amended
rule can be informed by having seen zeros that were certain in advance. They therefore remain usable for exactly
one scoring under the amended rule, and the record says that this is why. Had the rule been capable of passing
and merely failed, the strata would have been spent.

**R3 — the no-transport comparison becomes a declared descriptive baseline in the amendment.** "Is the true
successor the nearest other decision of its episode, source excluded?" is reported beside E2 at the same rank
and depth, labelled as a baseline for local order and not as an estimand, with its chance level; until the
amendment is sealed it stays a diagnostic on training folds, as ruling 2 has it. It is not a tolerance-bearing
quantity and M stays 12.

E1 stands as read. E3 waits on WS-B. The deferred control is untouched.

## Amendment 1 (the transport rule): the capability gate — ruling, Chief, 2026-09-10 (~14:25Z)

The D-CRO's draft makes the rule the identity plus a rank-r supervised correction, `T_r(x) = x + Δ_r(x)`, with
`Δ_r` fitted from the source residual to the displacement — right, and for the right reason: a rank-r map from
`x_k` to `x_{k+1}` must spend its rank on the identity, which is full rank, and lands near the training mean; the
rule's content is the change, and the withdrawn constant displacement is `Δ` at rank zero, the floor of the same
ladder. R2 confirmed on the data: 0 of 4,122 ordinary train transitions would have beaten the source under the
withdrawn rule, so the corrective strata are unspent.

The draft's capability gate — "the rule clears the source on at least half of ordinary training transitions" —
was written at the headline rank and depth before measurement, then measured out of fold on ordinary train
transitions only: 0.586 on the 4B and 0.493 on the 12B at rank 8, 0.904 and 0.901 at rank 32. The D-CRO asks
which form stands rather than choosing after the number. Ruled:

**The gate tests the form, and a property of the form is read where the form is least constrained: at the top of
the ladder (form B).** A rule whose geometry forbids a hit scores near zero at every rank; this one reaches 0.90.
That is what the gate exists to reject, and it does not.

**The headline-rank fraction is not a gate; it is a capability report, and it is mandatory.** The E2 hit rate at
the pre-registered rank 8 is reported beside the fraction of ordinary training transitions on which the rule
clears the source at that rank — 0.586 and 0.493 — unconditionally, so that a low hit rate on the corrective
strata is read against a rule that beats its own source only about half the time there, and no reader takes a
capability ceiling for a null about update closure. No estimand is added and M stays 12.

**The record says what happened.** The amendment states that the gate was first written at the headline rank,
measured, and reclassified from gate to report on this ruling, with the reason above; the reason is not the
seven thousandths. The ladder (ranks 1 … 32) and the fold discipline are as the D-CRO measured them; nothing
corrective has been touched. With this the amendment is ready for Codex's file-only review; then the seal
builder seals it as an addendum against its own baseline; then the corrective strata are scored once.

## Amendment 1, the transport rule's form reverted — ruling, Chief, 2026-09-10 (~15:10Z)

The D-CRO withdraws the identity-plus-correction form and the argument for it: measured out of fold on ordinary
train transitions at the headline depth with the fitter's restart bug fixed, the direct rank-r supervised map
from `x_k` to `x_{k+1}` clears the source more often than the correction form at every rank on both models
(0.769 and 0.751 at rank 8 against 0.586 and 0.493), because the successor is dominated by a large, predictable
shared component and the displacement is small and only a fifth coherent; the earlier claim came from a
synthetic case in the wrong regime. Ruled:

**The form is the direct rank-r supervised map to the successor**, on the plain reading of "transport the
state read at k" — the choice rests on the reading and on the absence of any reason to constrain it, not on
the capability numbers, which are a two-point comparison on training folds and cannot arbitrate between two
capable forms; the retrieval score is not consulted, being ε_ord's own population before the seal.

**The gate returns to form A** — half of ordinary training transitions cleared at the headline depth and rank,
both models — as first drafted; it passes. The ruling of ~14:25Z reclassifying it is superseded and stays in
the order as history. The headline-rank capability fraction is still reported beside the E2 headline.

**The amendment carries its own history**: the correction form, the argument, its refutation, and the fitter's
restart defect (power iteration restarting from the deflated direction, able to stall short of the asked
rank), with the fix and its test.

**One question answered before Codex's review.** The restart defect lived in the supervised-subspace fitter that
E1 used. The D-CRO states whether E1's sealed reading was produced by the defective code and, if so, re-runs E1
through the gate with the fixed fitter and reports both numbers side by side — E1's reading stands only if the
fix does not move it, and moves are recorded either way. Then Codex's file-only review; then the addendum seal;
then the corrective strata once.

*2026-09-10 UTC, ~15:40Z:* **E1 stands**: its fitter has a scalar target with a closed-form direction (no iteration, no restart), `read_e1.py` never imports the module the defect lived in, and E1 re-run on the fixed tree reproduces its sealed readings byte for byte (digest ba841d7c…; 0.646, 0.073, 0.611, 0.030 unchanged). **Amendment 1, revision 4, is at `615116f` on `cuda-ws-d` and ready for Codex's file-only review**: §2 the direct rank-r supervised map to the successor, chosen on the plain reading and not on the capability table; §5 the gate at half the ordinary training transitions cleared at the headline depth and rank (0.769 / 0.751), the top-of-ladder reading kept as superseded history; the headline-rank fraction printed beside the E2 headline as a ceiling; §10 the whole detour with both withdrawn tables and the fitter defect, fix and test. Thirty-seven tests, the checker at 68 claims, the seal verifying. After Codex: the addendum seal against its own baseline, the gate, then the corrective strata once.

*2026-09-10 UTC, 13:15Z (card clock):* **Codex's review of Amendment 1 revision 4 (657a08a, `WSA-TRANSPORT-AMENDMENT-REVIEW-2026-09-10`): four findings, all correct, applied by the Chief on Daniel's instruction as revision 5, `e7d3e67` on `cuda-ws-d`.** F1: the leading direction is now the leading left singular vector of the deflated cross-product from a full SVD, sign-fixed, with finite guards — no start vector, so neither the zero-start NaN nor the certified non-leading fixed point can occur. F2: the tests now discriminate — Codex's restart witness, both F1 witnesses (finite and matching the SVD reference; alignment 1 with the leading direction), a nested 32-component ladder, full rank equals least squares, a narrower-than-source target, refusal of non-finite input; the fitter at b0cd61c fails seven of the fifteen, the current one passes all. Two further defects surfaced on the way and are fixed: the old fitter reached 11 of 12 components on a full-rank fixture, and its target-loading array was sized by the source width. F3: §7 refers to §5's current table and repeats no figure; `check_amendment.py` (with a self-test) fails the document if a §5 table figure recurs outside §5 and §10. F4: §5 now states the capability bound's exact scope — same rows, depth, rank, transition cost and averaging weights — and calls the figure a capability reference elsewhere; "at every depth … a ceiling" is gone. **Consequence:** §5's gate table was produced by the fitter F1 describes and is marked provisional; the D-CRO re-measures it with the SVD fitter before the seal and the record says whether the figures moved. Then Codex's file-only re-check of revision 5, then the addendum seal, then the corrective strata once. The D-CRO's hold on `transport.py`, its tests and the amendment is lifted.

*2026-09-10 UTC, 14:20Z (card clock):* **Amendment 1 revision 5 is ready for Codex's file-only re-check, at `1d93608` on `cuda-ws-d`.** The D-CRO re-measured §5's capability table with the SVD fitter (`measure_gate.py`, committed at 0fe5a42 before the run, refusing without the seal or against any fold assignment other than the sealed one; output `gate-table.json` beside the table): every fold yields 32 components on both models and the table is identical to the provisional one at every quoted digit, the largest cell difference 0.00049. §5 states that as a fact about this population and not as a defence of the old fitter, and §10's "consequence for §5" is discharged. The seal re-verifies (thirteen files, baseline acc130a, checker passing) and E1 re-run on the SVD-fitter tree is byte-identical (digest ba841d7c…), so E1 is independent of the defect and of its fix. The Chief re-ran both checkers and their self-tests on the fast-forwarded worktree: all four exit 0; the D-CRO also planted a withdrawn §10 figure in §7 and the amendment checker rejects it. Cost noted for the record only: the SVD fitter is about a hundred times the power iteration per component, twenty minutes for one depth on both models, an hour and a half for E2's six depths, measured. The sequence stands: Codex's re-check, then the addendum seal against its own baseline, then the gate, then the corrective strata once.

*2026-09-11 UTC, 02:00Z (card clock):* **Codex: PASS on Amendment 1 revision 5 at 1d93608** (review 6d9d714 on `codex/cuda-torch-seam`; checked set: 43 CPU tests, the old-fitter negative control, both document checkers and their self-tests, the 13 sealed files, E1 byte identity). The D-CRO proceeds: the addendum seal against its own baseline, reported first and separately; then the gate; then the corrective strata once, as ruled on 2026-09-10.

*2026-09-11 UTC, 02:05Z (card clock; an earlier stamp of 02:20Z was the Chief's estimate, corrected against the card):* **the addendum is sealed** — digest `def0b7e92961f4eb8a6a8634aa68eb65f6e4a0e313e0c07f3ab1a0cdf39d8dcd`, baseline `1d93608` (the commit Codex read), amending the parent seal `998b3bca…`, committed d7bdf7d; seven files fixed (the amendment, `check_amendment.py`, `measure_gate.py` and `gate-table.json`, `read_e2.py`, `transport.py`, the rule's tests); `make_addendum.py` recomputes and cross-checks (the printed table against the artefact, the artefact against this seal and fold set, the parent seal, the checker, every file's bytes at the baseline) and refuses to overwrite. Verified by the Chief: a dry recomputation reproduces all seven file digests; the addendum digest itself carries the writing commit, so a reviewer recomputing at a later HEAD gets a different digest by construction — a `--verify` mode that uses the recorded commit is asked for after the strata. On the record, the D-CRO's own caveat: §5's table and `gate-table.json` are both theirs, so the cross-check catches a table edited after measurement or measured against another seal or fold set, and does not make the figure independently attested. **The word given:** §5's gate as a restatement of the sealed figures, said plainly; then the four strata once, reported together with tails.

*2026-09-11 UTC, ~02:15Z (card clock):* **addendum `def0b7e9` is NOT good; nothing has been read.** The D-CRO found, on starting the strata, that `read_e2.py` — one of the addendum's seven sealed files — at the baseline 1d93608 still implements the withdrawn rule (`features[source] + m·d`, line 78) and never imports the transport module; `measure_gate.py` does. The addendum therefore sealed an amendment and a reader that contradict each other, and every check in `make_addendum.py` passed because each checks identity and provenance, none checks that the sealed code implements the sealed rule (entry 34's family). The Chief confirms the finding and records a miss of their own: the Chief read that line of `read_e2.py` on 2026-09-10 while locating the gate script and did not connect it to the amendment. Codex's PASS covered the amendment, `transport.py` and the tests; this file was not read against the rule by anyone. The parent seal's thirteen files do not include `read_e2.py`; it and E1 are unaffected. **Ruled:** the reader is updated to §2's rule and committed (not a parent-sealed path); addendum-1 is marked superseded in place, never deleted; the builder gains the import/withdrawn-signature refusal (a proxy, said so) and an executable conformity check — a synthetic fixture on which the two rules differ, run through the reader's own rule function, must return §2's answer; Codex re-checks `read_e2.py` against §2 file-only before addendum 2 is written; then addendum 2 against the new baseline, the gate restatement, the strata. Nothing is read until addendum 2 verifies.

*2026-09-11 UTC, ~02:40Z (card clock):* **the reader is corrected and the builder now checks meaning** (`cuda-ws-d` at bf6fd06; `read_e2.py` at 6fc5e0a). Verified by the Chief: the reader imports the sealed rule module and isolates the rule in `transport_rule`, fits the rank-r map out of fold on the sealed assignment and applies it m times, and reports per-episode contrast distributions (5th/95th, median, extremes, shares at zero and one); the withdrawn lines are gone. `addendum-1.json` is byte-identical to what was issued and `addendum-1-SUPERSEDED.md` sits beside it — the D-CRO's judgement not to edit a file whose digest is its identity is upheld. `make_addendum.py` gains the import/withdrawn-signature proxy (said to be a proxy in its docstring) and an executable conformity check: a fixture on which `x + m·d` and §2's rule differ, run through the reader's own `transport_rule`, refusing unless §2's answer comes back, refusing if the reader has no rule function or it raises, and refusing if the fixture ever stops separating the two rules; all three refusals were exercised against the superseded reader and pass on the current one; a dry build against bf6fd06 passes. **Codex** now re-checks `read_e2.py` against §2, file-only (the paste is with Daniel); then addendum 2 against the tip Codex read, the gate restatement, the strata once with tails. Nothing read.

*2026-09-11 UTC, ~03:00Z (card clock):* **Codex: FINDINGS on the corrected reader** (`read_e2.py` at 6fc5e0a; review 3af4c56, `WSA-E2-READER-RECHECK-2026-09-11`, with an executable harness that reproduces every finding on synthetic fixtures and reads no capture). The supervised fit, fold exclusion, m applications and the builder's three refusals check out. Six findings, all upheld: F1 the reader verifies only the parent seal, never an active addendum, and carries the void addendum's string; F2 the capability gate never gates scoring; F3 no stratum carries its sealed tolerance, bound, α, M or sufficiency status; F4 short-rank folds silently shrink the scored population while the transition count is reported whole; F5 retrieval runs in float64 where §4.2 declares float32; F6 "share at zero" on contrasts is the share at or below chance. **Ruled** (sent to the D-CRO): fix all six on `cuda-ws-d`; on F1 the circularity is resolved by the reader hashing its own bytes at runtime and requiring the addendum beside the record to verify against the parent seal, to carry no supersession marker, and to list this reader's and `transport.py`'s current digests, so the addendum is written after the reader is final; on F2 the headline gate on both models is enforced before any stratum, or a receipt bound to the same seal, folds, reader and fitter is verified; on F5 the declared float32 is kept, not amended. Codex re-checks the fix; then addendum 2; then the gate restatement and the strata once with tails. Nothing read.

*2026-09-11 UTC, ~03:20Z (card clock):* **all six findings fixed** (`cuda-ws-d` at c465d35; reader 074005d), verified by the Chief on the fast-forwarded worktree: the reader requires an active addendum that lists its own current bytes and the rule module's, refuses a superseded one by name and two active ones as "not for a reader to choose", and verifies every file the addendum lists; both models' headline gate is computed in a phase that ends before any stratum is scored and a failure refuses the run; every stratum carries its tolerance read from the seal with M, α, the range width, the episode requirement, a sufficiency status and the registered paired bound, the headline marked distinct from the exploratory profile and `ordinary_test` descriptive; coverage is reported against the request with unavailable folds named; float32 at the retrieval boundary; `share_at_or_below_chance` with the exact mass at chance beside it. `check_findings.py` beside the reader carries Codex's six witnesses: all pass on the fixed reader, all fail on 6fc5e0a (the Chief reproduces the six passes); the D-CRO states that five of the six fail on the old reader by absence of the machinery rather than by observing the defect, and that Codex's own harness stops at its fixture allowlist, not at the reader — whether to adapt that harness is Codex's call in the re-check. `require_addendum` currently refuses everything, correctly, until addendum 2 exists. Codex re-checks c465d35; nothing read.

*2026-09-11 UTC, ~03:35Z (card clock):* **Codex re-check da24c8b (`WSA-E2-READER-RECHECK2-2026-09-11`): F1, F4, F5, F6 closed; F2 and F3 remain, upheld.** F2: the gate admits whatever model directories exist and passes on the fraction alone — one model, no models, and a model gated on a fifth of its requested transitions all exit 0 and score. Ruled: the model pair is declared, a missing model or empty set refuses, and the gate requires complete coverage on both models (complete, scored equal to requested, no unavailable fold at the headline rank) or refuses; no conservative substitute. F3: the bootstrap resamples fixed episode arrays without refits or (split, family, variant) stratification, and the verdict ignores the interval (a control interval of [−0.4, 0.6] still reads "resolved"). Ruled: §7 as sealed — stratified episode resampling, M resamples, out-of-fold refits within each, both bounds reported, the wider governing; "resolved" only when the governing bound lies wholly beyond or within ε, "not resolvable at this n" when it straddles, "insufficient" below the episode requirement; scoped to the registered headline (depth 0.5, rank 8, eight components) for the four strata on both models, the exploratory profile descriptive without bounds; `ordinary_test` descriptive at the headline too. The D-CRO estimates the card time before running. Same loop; nothing read.

*2026-09-11 UTC, ~03:50Z (card clock):* **F2 and F3 fixed (`cuda-ws-d` at 1a48f17), and §7's declared interval is measured to cost 8.2 days of the card.** Verified by the Chief: the model pair is declared by the sealed capture budget and a missing, unexpected or empty set refuses by name; the gate requires complete coverage beside the fraction (1.0 on twenty of a hundred now fails); §7's bootstrap resamples episodes within `(split, family, variant)` and refits out of fold inside every resample, both intervals reported with the wider governing, and the reading is resolved / not resolvable at this n / insufficient from that interval rather than the point estimate; `is_registered` is extracted so the headline's registration is witnessed directly, a gap the D-CRO found in Codex's own reproduction and reported against themselves; `check_findings.py` carries both rounds and passes. Codex's harness runs to completion against the fix with six adaptations the D-CRO lists (two model-name literals, one stub signature, and four assertions that encoded the now-absent defects); its stored verdict string is left as Codex's. **The cost, measured on the card rather than extrapolated** (§0.4 of the record): one 8-component fit 27.7 s at 4B and 85.7 s at 12B; one resample over five folds 138.5 s and 428.5 s; 10,000 resamples 385 h and 1,190 h serial, 1,575 h in total, about 197 h — 8.2 days — across the eight physical cores. M is not trimmed. A lever exists and is not the Chief's to take: the fitter computes a full p×p singular value decomposition to obtain one leading triplet, where a Krylov method returns the same triplet with a residual certificate — which is what F1 required — measured at 14× on that step, which is about 97% of the fit, putting the run near 20 hours; it edits a sealed file, so it is an amendment to Amendment 1, a Codex review and a new addendum. **Ruled:** the 8.2-day run does not start; the Krylov routine is prepared as an unsealed, unrun draft with the full decomposition kept as its reference and the certificate's threshold declared; the spending decision is the Research Director's.

*2026-09-11 UTC, ~04:05Z (card clock):* **Amendment 2 drafted (`cuda-ws-d` at 1d2af61, `AMENDMENT-2-CERTIFIED-DIRECTION.md`), unsealed and unrun; `transport.py` untouched.** A NumPy Lanczos on `A Aᵀ` with a Rayleigh–Ritz step returns the leading triplet with a certificate — relative residual below 1e-10 and Ritz gap above 1e-6, both declared with reasons — and anything failing either threshold falls back to the full decomposition, so the sealed answer is reproduced exactly or reproduced exactly by the fallback, with no third outcome. §3 of the draft states plainly what the certificate does *not* prove: a small residual bounds the distance to *a* triplet and the Ritz value cannot overstate σ, so leadingness rests on the fallback rather than on the certificate. The Krylov dimension is a declared schedule 48/96/192/384 grown until certified; a fixed 24 leaves the direction wrong at the 12B width (cosine 0.32) and the certificate refuses it, which is the evidence that it does work. **A correction the D-CRO made to their own estimate:** the 14× figure came from `scipy`, which is absent from the card and undeclared in this repository, so a routine built on it could not run where the reading runs. Re-measured on the card at production shapes, single-threaded: 27.0 s → 2.9 s (4B) and 85.0 s → 8.6 s (12B), 9.4× and 9.8×, eight of eight components certified, no fallbacks, worst direction disagreement 1.1 × 10⁻¹⁵. §7's interval becomes about 160 h serial, ~20 h across the eight cores, against 8.2 days. Twelve tests compare the routine with the full decomposition in both directions and pass beside the sealed rule's own. **Asked before review:** the fitted rule's coefficients under the two routines compared on the conformity fixture at production widths, so "no declared quantity changes" is measured. Codex reviews 1a48f17 and this draft; §7 does not run under either routine until the Research Director rules on the cost.

*2026-09-11 UTC, ~04:20Z (card clock):* **Amendment 2's claim is now measured, not argued** (`cuda-ws-d` at 7fc7aa2, §4 of the draft). Fitting 3,298 rows at rank 8 and applying to 400 held out, at production widths on the card: the coefficient matrices of the sealed and certified routines agree to 2.3 × 10⁻¹¹ (4B) and 6.4 × 10⁻¹¹ (12B), and the quantity §5's table prints is **bit-identical** on both models — 0.132500 and 0.055000 under either routine. A test pins both at an affordable width, with the gate compared by exact equality rather than a tolerance. On the conformity fixture the coefficients agree to 6.6 × 10⁻¹¹ and 9.6 × 10⁻¹⁴. **The D-CRO's own note on how the number was obtained, recorded because it is the week's recurring shape:** their first attempt measured the gate on the conformity fixture at production width and got 0.000000 under both routines — agreement that was worthless, since that fixture's map is too ill-conditioned for a rank-8 rule to beat the source at all, so both arms score zero by construction. They distrusted a gate of exactly zero because it is the shape the withdrawn displacement rule produced, and re-measured on a fixture where the gate can move. The Chief verifies: 28 tests pass beside the sealed rule's, all four checkers and their self-tests exit 0, `check_findings.py` passes both rounds. Codex reviews 1a48f17 and 7fc7aa2; §7 does not run under either routine until the Research Director rules on the cost; the 553 corrective transitions remain unspent.

*2026-09-11 UTC, ~04:30Z (card clock):* **method entry 35 written and read** (`dce2401`, `METHOD-2026-09-08`, thirty-fifth): *an agreement is evidence only when the fixture could have disagreed.* Identity between two computations is evidence about the computations only if the fixture can tell them apart; a comparison both arms pass by construction — the precision too coarse, the regime too easy, or the fixture degenerate — measures the fixture. The procedure is one line: before reporting an agreement, perturb one arm and watch the number move. The entry notes that a separating fixture, running a test against the unfixed code, and an admitting-default negative control are three names for that one step, and records the tell that caught the third case when nothing else did: an exactly round number in a noisy measurement is a question, not an answer. Its three cases are this week's: the capability table whose quoted digits partly guaranteed the agreement (largest cell difference 0.00049, inside the rounding of three decimals), the nine tests that passed on the defective fitter, and the two routines that returned 0.000000 because an ill-conditioned fixture pinned both. The Chief's part in all three is recorded in the entries it cites.

*2026-09-11 UTC, ~04:40Z (card clock):* **the device was never questioned, and the Research Director's standing rule now governs it.** Reported by the D-CRO as Daniel's words of 2026-09-11: "unless there is a positive, scientific/epistemic integrity mediated reason to run something on the CPU, we should ALWAYS run on the GPU." The 8.2-day figure of §0.4 was a CPU-only estimate that never asked why the CPU, while the card's GPU sat at 1 MiB; the D-CRO records that correction as theirs. First measurement, float64, eight components at production shapes: 4B sealed CPU 15.3 s, certified CPU 2.2 s, certified GPU 0.32 s; 12B sealed CPU 56.3 s, certified CPU 8.0 s, certified GPU 0.23 s; agreement with the sealed routine 3.1 × 10⁻¹¹ and 2.5 × 10⁻¹¹. §7's interval would be about 7.6 hours serial on the GPU — one overnight run. **Held for re-measurement before it is quoted:** the 12B GPU time is faster than the 4B's on a wider matrix, which cannot be right in the direction it points; CUDA launches are asynchronous and the first call in a process carries context initialisation, so the timings are to be retaken with a discarded warm-up and explicit synchronisation (entry 35's tell). **Ruled, on the D-CRO's two questions.** (1) The device is part of Amendment 2: it proposes algorithm and device together, with the certificate and the fallback as the correctness guarantee in place of bitwise identity, and states what is given up — GPU float64 does not replay the CPU bitwise (1.3 × 10⁻¹³ on the vector, 1.8 × 10⁻⁸ on σ). (2) One reading, one implementation, one device: the headline point estimate and every resample run on the GPU under the certified routine; fitting the headline on the CPU and the interval on the GPU is refused, because an interval must bracket the estimator it is computed from. (3) The CPU becomes a declared replay: after the GPU reading the headline alone is refitted on the CPU under the sealed routine and both figures are printed side by side with their difference at §5's printed precision, agreement being the evidence that the device did not move the result and disagreement a finding that stops the reading. (4) What stays on the CPU stays for a stated reason — the sealed artefacts and their byte-identical replay checks, which is a property the programme uses — not by default.

*2026-09-11 UTC, 04:10Z (card clock):* **Codex 3cb6247 — FINDINGS on both the reader and Amendment 2 (`WSA-E2-READER-CERTIFIED-REVIEWS-2026-09-11`, with independent harnesses for each).** F2's model-pair and coverage gates close. **C1, high, reproduced: the certificate can certify a non-leading direction and skip the fallback.** The Lanczos starts from `A · 1`, the same start vector the power iteration used, and on `[[3,−3,0],[0,0,1],[0,0,0]]` — the shape of Codex's own F1 witness against Amendment 1 — it returns σ = 1 against a true 4.2426, cosine 0 to the leading direction, relative residual exactly 0, Ritz gap exactly 1, certified, fallback not taken; the gap is 1 because a one-dimensional Krylov space has no second Ritz value and the code supplies zero. **The amendment written to make Amendment 1 affordable reintroduced the defect Amendment 1 was written to fix, and certified it.** C2, high: the draft stops on the Frobenius norm where the sealed fitter stops on the largest singular value, returning rank 1 where the seal returns rank 0, a stopping rule §6 calls unchanged. C3: 1e-11 agreement is not "exact reproduction" and the amendment must not say so. On the reader, F3-A/B/C, all high and reproduced: the bootstrap loses draw multiplicities at the final reduction and resamples the wrong population for corrective strata; the sealed Hoeffding interval is missing from the governing pair; a resample that loses folds or produces nothing still yields a registered reading, `--bootstrap-resamples` is unadmitted against the sealed count, and a `None` capability fraction crashes the formatter before its named refusal. **Ruled: Amendment 2 as drafted is withdrawn, not repaired** — a certificate that cannot bound the spectrum outside the explored subspace cannot establish leadingness, and no threshold on residual or Ritz gap repairs that. **The measurement that was missing:** sealed-on-CPU and certified-on-GPU were measured, sealed-on-GPU never was. A full decomposition returns the leading triplet by construction, so `torch.linalg.svd` in float64 on the card is to be measured at both production widths against the CPU reference — time, coefficient agreement, and §5's printed quantity — before anything else. If it lands near the certified timings, §7 runs under the sealed algorithm on the GPU and the only amendment required is the device one, with no new mathematics. The reader's F3 findings are fixed in parallel and are not device-dependent. Nothing read; §7 not started.

*2026-09-11 UTC, ~04:45Z (card clock):* **the review findings are fixed and confirmed, and the fixes are uncommitted.** The Research Director addressed F3 and C1–C3 directly in the `cuda-ws-d` worktree (plan at `docs/superpowers/plans/2026-09-11-e2-review-fixes.md`, with `tests/test_e2_review_regressions.py`): bootstrap weights, training population, interval comparison and incomplete-refit refusal repaired; **false certification removed and the full decomposition made mandatory for every returned direction**, with the sealed fitter's stopping rule, floors and deflation order restored; the draft's accuracy and speed claims withdrawn. Codex confirms PASS at 1d2c820 (`WSA-E2-FIX-CONFIRMATION-2026-09-11`): 84 tests, eight reader witnesses, additional resampling and fitter checks, no remaining scoped finding. **The verdict is tied to the sha256 of the uncommitted files, not to a commit.** The Chief verifies all four recorded identities against the worktree on disk: `read_e2.py` 43459950…, `transport_certified.py` 5e90554a…, `AMENDMENT-2-CERTIFIED-DIRECTION.md` 76509aab…, and the **unchanged sealed `transport.py` 6e8b7135…** — the reviewed bytes are intact and the seal is untouched. The D-CRO's branch tip is dce2401, pushed, and they correctly declined either to commit another author's work under their name or to discard it. **Standing risk, recorded rather than acted on under the Director's hold:** a PASS anchored to uncommitted files in a shared worktree is lost to any checkout, stash or clean, and with it the only durable link between the review and the bytes it read. Nothing read; §7 not started; addendum 2 does not exist, so the reader still refuses every path into E2.

*2026-09-11 UTC, ~04:55Z (card clock):* **the merge has not landed and the hold stands for both seats.** `origin/cuda-ws-d` is dce2401, unchanged; the Research Director's fixes remain uncommitted in the worktree; Codex's 1d2c820 is fetchable but not on this branch. The D-CRO declined the Chief's instruction to take the GPU measurement, on the ground that the hold was given to them by the Director in his own words and a relayed lift is not the thing he gave: a peer's word is not the user's approval, and it does not become one because the peer is relaying accurately. **The Chief withdraws the instruction and records the refusal as correct.** Two errors of the Chief's, recorded: the instruction was written as though the merge had landed, and the Director's lift to the Chief was conditional on that same merge — "once this is done, the hold is lifted" — so neither seat is released and the Chief is not taking the measurement either. The D-CRO verified the sealed fitter on disk against **addendum 1's own file table** rather than against the Chief's message, so the two agree independently: `transport.py` 6e8b7135…. Digests of the seven uncommitted files are recorded as the anchor for Codex's hash-tied PASS: amendment 76509aab, `check_findings.py` bd9794a0, `read_e2.py` 43459950, `transport_certified.py` 5e90554a, its tests 9c5bf75e, the plan bfb66572, the regressions 1fab8ae7. Nothing read; §7 not started; the 553 unspent.

*2026-09-11 UTC, ~05:05Z (card clock):* **the fixes are committed as fd0e630 and the hold lifts for both seats.** The Research Director committed "Fix E2 bootstrap inference and require reference SVD fallback" on `cuda-ws-d` (local tip; not yet on origin, being brought into `cuda-migration` with the PASS record) after fresh verification — 84 tests, eight reader checks, the independent statistical and matrix checks. **Step 1 verified by the Chief against Codex's manifest rather than against any worktree**, on the D-CRO's improvement to the Chief's check: `reviewed-files.json` at 1d2c820 records a `source_sha256` per path over **forty** paths, not merely the seven — the pre-registration, both amendments, addendum 1 and its supersession notice, the capture artefacts and all four checkers — so the merged tree is compared to what Codex read with nobody's disk in the loop. Result: 0 paths missing at fd0e630, 0 digest mismatches, 7 paths changed by the commit and all 7 inside the reviewed set, 0 changed-but-unreviewed, the sealed `transport.py` still 6e8b71351cea4609, and the worktree clean. The hash-tied PASS now has a durable anchor. **The D-CRO proceeds to the one measurement never taken:** the sealed full decomposition, float64, on the GPU, both production widths, eight components, CPU reference from the same process and session, discarded warm-up and explicit synchronisation after the inverted timings, with coefficient agreement and §5's printed quantity beside the times. §7 does not start on that report; addendum 2 is written against the merged baseline when the Chief rules on the sequence. Nothing read; the 553 unspent.

*2026-09-11 UTC, ~05:20Z (card clock):* **the GPU is slower for this operation, measured, and the acceleration question is closed.** The D-CRO mirrored `transport.fit` operation for operation on the GPU in float64 — same σ and tᵀt stopping rules, same deflation order, same sign convention — against a CPU reference in the same process, warm-up discarded, synchronised: at width 2,560, 15.10 s CPU against 28.92 s GPU (0.52×); at width 3,840, 56.62 s against 213.87 s (0.26×). **The penalty grows with width**, which is the opposite of the direction that would make the device worth having; this card's double-precision throughput is a small fraction of its single-precision and a dense decomposition parallelises poorly. Coefficient agreement across devices 2.3 × 10⁻⁶ and 2.8 × 10⁻⁶; §5's printed quantity identical on both. The earlier 0.32 s and 0.23 s GPU figures are retired as the unsynchronised artefact the Chief flagged. **With the certified path withdrawn as unsound and the sealed path slower on the GPU, §7's cost stands at the CPU figure.** On the Research Director's standing rule: the CPU now has two measured arguments — two to four times faster for this operation at the sealed precision, and byte-identical replay against ~10⁻⁶ cross-device agreement — where before it had only an inheritance, which is what the rule asks for. **Step 1 also passes on the stronger anchor**: 40 pinned paths, 0 missing, 0 differing, 7 changed and 0 of them unreviewed, hashing tree objects rather than any worktree (`verify.py` reads a worktree by default and cannot anchor a merge). Both remote tips now carry it, `origin/cuda-ws-d` at fd0e630 and `origin/cuda-migration` at c177f90, verified by the Chief. **Ruled before any spend: compute the Hoeffding interval first.** §7 declares it beside the bootstrap with the wider governing; it is closed-form and costs seconds, and where it is the wider interval the 10,000 refits cannot change that cell's reading. The D-CRO reports, per stratum and model, the Hoeffding half-width against ε and whether a bootstrap of any width could move the verdict. §7 does not start either way; addendum 2 is written after, once, against final bytes. Nothing read; the 553 unspent.
