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
