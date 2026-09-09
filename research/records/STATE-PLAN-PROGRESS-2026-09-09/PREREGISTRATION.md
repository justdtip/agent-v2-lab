# Pre-registration: plan progress as the state programme's second state variable

**UNSEALED DRAFT. D-CRO, 2026-09-09.** Owner: D-CRO, under
`design_specifications/pending/STATE-PLAN-PROGRESS-ORDER-2026-09-10.md` (R7 and the Chief's
amendment). It is **not sealed** and must not be: sealing waits on the golden-test control design
and on a scheduled capture pass. The two streams share the machinery but not the clock.

Nothing here has been read from a capture, because none exists. Every number in §§1–2 and §§7–9 is
computed by the two scripts in this directory from the rendered corpus alone, on the laptop, with no
model and no card:

| artefact | what it is |
|---|---|
| `prereg_inputs.py` → `prereg-inputs.json` | the corpus facts, the census, the twelve episodes |
| `capture_budget.py` → `capture-budget.json` | what capturing the pre-registered set costs |
| `capture-set.jsonl` | the capture set itself, one line per decision, 7,629 lines |
| `count_corpus.py` → `corpus-count.json` | the order §1 count, re-run and falsified per claim |

Corpus on the card: `/workspace/rendered-corpus/agent_v2e-gemma3-4b/`, archive
`7fe6e64b89749b997854638b260d7654316636da3d8eaa59924ad9ed62f99120`. Per-file digests are in
`prereg-inputs.json` under `file_sha256`, and they match the corpus manifest's own `outputs` block,
so the counted rows and the shipped rows are the same bytes.

The capture set has **two** digests and they are not interchangeable, so both are pinned and each is
labelled:

| | digest | what it fixes |
|---|---|---|
| the set | `1a7cfbdd4e21fc203eafbcc3ec50b96afcb214c169f21c7ba968ca83dfc9709a` | the logical triples `(task_id, step, prompt_sha256)`, in order — what is to be captured |
| the file | `98c78f1674902575ba87135edcfec409d3e82c6331e155a3ab7da6d22b731a4d` | the bytes of `capture-set.jsonl` — what was shipped |

The first is the one that must survive a re-run on another machine, since it is invariant to
formatting; the second is the one that says the file on the card is the file written here.

---

## 1. The corpus, and the three facts the order requires this document to carry

Measured on the three rendered splits. Basis stated on every figure, because the week's own record
is that a count without a basis is a count nobody can reproduce.

| | |
|---|---:|
| rendered rows | 8,907 |
| agentic rows (carry a `task_id`) | 8,559 |
| chat replay rows (carry no `family`) | 348 |
| **distinct decisions** | **7,629** |
| episodes | 1,128 (840 train, 48 valid, 240 test) |
| transitions between consecutive decisions | 6,501 |

**Fact 1. The step index is the turn count.** Under teacher forcing every rendered row is one
decision and the number of model turns already in the prompt is the step index. It is a control, not
an estimand: a probe that cannot read it is broken.

**Fact 2. The horizon is a function of (family, difficulty, variant) for eleven of twelve families.**
`pointer_chain` is the exception, its horizon varying with a chain length that is not in the task
prompt and is not knowable to the model until the chain ends.

**Fact 3. The protocol writes the plan into the transcript.** The progress note carries the plan and
its progress in words at every step, and the protocol hides tool results older than `keep_last = 2`
turns. 4,509 of the 6,501 consecutive pairs are not prefixes of each other. The transcript is the
explicit carrier of state by design.

### 1.1 A correction to the amendment: the repeated step indices are not retries

The order's A1 rests on a reading I supplied and got wrong, and the correction changes what A1 and A3
should say. It is here rather than in a later note because the capture key depends on it.

I reported that `(task_id, step)` collides 930 times because a `transient` task re-issues the same
action and the corpus labels both rows with the same step. **It does not.** Every one of the 930
repeats is a **byte-identical row** — same prompt, same completion, same metadata, verified by
digest over the whole row, with zero exceptions. Their multiplicity is 2 or 6 and never anything
else, and the decisions that carry them are **exactly** the 578 rows flagged `recovery`, every one
of them, and no other row in the corpus.

The mechanism is in the renderer and is deliberate. `pipeline/data.py` applies, in the training
split only:

```python
if split_spec.role == "train":
    expert_rows = [row for row in expert_rows for _ in range(_repeats(row, recovery_repeats))]
```

and the corpus provenance records the setting it ran under:
`recovery_repeats: {transient: 2, wrong_path: 2, unknown_tool: 2, stale_path: 6, failed_edit: 6}`.
That reproduces the observed counts exactly: 490 decisions duplicated twice (transient 244 +
wrong_path 182 + unknown_tool 64) and 88 duplicated six times (failed_edit 58 + stale_path 30),
giving 490 + 88×5 = 930 extra rows. `random.Random(f"{seed}:{split}").shuffle(rows)` then scatters
them, which is why the copies are far apart in the file and why row order within a task carries no
meaning.

**So it is a training-mix weight, not a second decision.** Three consequences, and they are the
reason this section exists:

1. **The step index never repeats.** There is no retry transition anywhere in the corpus. A1's
   clause — that E2's rule must predict no advance at a retry — has nothing to predict, and the 244
   transient tasks are not a stratum that refutes a rule advancing every turn. What the transient
   variant actually contributes is stated in §4.2, and it is still the informative stratum, for a
   different and better reason.
2. **The capture unit is the distinct decision**, 7,629 of them. Capturing rows would forward the
   same tokens up to six times for nothing, and would weight any probe fit by the fine-tuning
   recipe's oversampling — silently, and in the direction that most flatters a recovery result.
3. **A3's occurrence index is unnecessary and misleading.** `(task_id, step)` addresses a decision
   uniquely once the copies are recognised as copies. The key is `(task_id, step)`; the manifest
   carries the sha256 of the rendered prompt and the multiplicity `rendered_rows`, so the training
   weight is recoverable without being applied by accident.

The Chief's own §1 figures are unaffected and were right: 7,629 rows, 6,501 pairs, maximum 17
decisions is the deduplicated basis, and deduplicating by `(task_id, step)` is the correct operation
after all — for a reason neither of us had. My Addition 1 in this record's appendix is withdrawn and
is superseded by this section.

### 1.2 The 348 rows that carry no `family`, named before they are captured or excluded

They are chat replay rows: `source: pre-expansion-policy-replay`, `prompt_id`
`chat-{split}-{category}-NNNN`, 58 in each of six categories — `code`, `compare`, `explain`,
`follow_up`, `plan`, `rewrite` — distributed 240 train, 48 valid, 60 test. They have no task, no
step and no horizon, so no ground truth for this state variable exists for them at all.
**Disposition: excluded from the capture.** They are in the mix to hold general capability during
fine-tuning; they are not agentic episodes and nothing here reads them.

---

## 2. The capture set

**Unit.** One decision position per distinct decision: the last prompt token of the rendered row,
the position whose readout produces the first token of the model's turn.

**Key.** `(task_id, step)`. The manifest carries, per captured cell: `task_id`, `step`, `split`,
`family`, `variant`, `difficulty`, `recovery`, `rendered_rows` (the oversampling multiplicity),
`prompt_sha256`, the row ordinals the decision was rendered at, the checkpoint digest, the layer,
the decoding mode, `capture_dtype: native`, the device reading, and the `basis` field per cell. The
ground truth is stored beside the cell and comes from the task, never from the model.

**Set.** `capture-set.jsonl`, 7,629 lines, digest above. It is enumerated once, before the first
capture, and the capture pass reads it rather than re-deriving the enumeration on the card.

---

## 3. The hypotheses

| | claim | how it can be false |
|---|---|---|
| H1 | the 12B's residual stream is **more predictive** of plan progress than the 4B's, at matched probe capacity | both equally predictive once capacity is matched; or the 4B is more predictive |
| H2 | that state is **retained across positions within a pass**, not recomputed at the decision position from the recent tokens | the state at the decision position is recoverable from the sliding window alone |
| H3 | H1 or H2 hold **only after fine-tuning** | they hold in the base models, or in neither |

**H2 is restated from the Director's form, per R4, and the restatement is a narrowing.** Under
`cache.strategy: none` — the registry's setting for both Gemma entries, and the only strategy the
golden records exercised — every turn is a fresh pass over the re-rendered prompt. Across-turn
maintenance is impossible by construction and cannot be tested on this protocol. What is testable is
retention *within a pass, across positions*. The Director's across-turn form needs a cache-reuse
strategy, which the registry marks UNRULED, and is not this document's.

H1's falsifier includes the 4B being *more* predictive. That is a live outcome, not an instrument
failure.

---

## 4. The estimands

### 4.1 E1 — decodability at matched rank (not "predictive sufficiency")

E1 as originally drafted — the gain over a predictor given the visible transcript — is
**untestable by construction** on this corpus and is recorded as `untestable`, not measured against
a weakened baseline. Facts 2 and 3 are why: the strongest transcript predictor is exact for eleven
families, and the twelfth is unknowable to the model as well as to the baseline. E1 is not measured
in that form and no gain over a weakened baseline is reported as a finding.

What is measured is **decodability at matched rank**, a comparison of encoding and not of
representational richness, and every figure carries that label.

- **Targets**, both read from the task: the step index, and the steps remaining.
- **Primary score**: exact-match accuracy on the integer target, bounded in [0, 1] so the
  concentration bound in §7 applies. Mean absolute error and R² are reported beside it, descriptively.
- **Two nulls, fitted the same way at every rank on every model.** The **permutation null**: labels
  permuted within family. The **step-0 null**: the same probe read from the residual at the first
  decision of the same task, which carries what the task prompt alone tells — family, difficulty —
  and nothing of the progress.
- **Negative control**: `pointer_chain` steps-remaining must sit at the permutation null on both
  models. A probe that decodes it is reading a leak, and the finding is the leak.
- **Evaluation set**: the test split's 1,781 decisions, 240 episodes, clean variants only. Held out
  by episode by construction, never by position.

### 4.2 E2 — update closure. The primary estimand

Does the state read after decision *k* move to the state the task's own dynamics imply for decision
*k+1*? The rule predicts +1 on an ordinary transition and the recovery cost at a corrective one.

**Score, and why it needs no tolerance.** For each held-out transition, transport the state read at
*k* by the rule and ask whether the transported state is nearer to the state read at *k+1* than to
the state read at any other decision of the same episode. A hit is bounded in [0, 1] and its chance
level is `1 / (decisions in the episode)`, computed per episode and reported beside the score. This
is a deliberate departure from the first state variable's derived-tolerance design: a retrieval
score against an explicit chance level has no free parameter to set after looking, and no pilot is
needed to set one. The departure is flagged for the Chief before sealing.

**The census, which is what says whether the stratum can carry the claim.** 6,501 transitions:

| | total | train | valid | test |
|---|---:|---:|---:|---:|
| ordinary (step advances by one) | 5,948 | 4,122 | 285 | 1,541 |
| into the corrective decision | 244 | 244 | 0 | 0 |
| into the corrective decision across a step with no rendered row | 309 | 309 | 0 | 0 |

Two things follow, and neither was in the order.

**Every gap transition is a recovery transition.** There are no transitions that skip a step for any
other reason: the dropped row is always the failing turn, and the decision after it is always the
corrective one. So "a gap in the rendered rows" and "the model has just failed" are the same event
in this corpus, and a transition spanning a gap is never scored as an ordinary +1.

**The transient variant is the only variant whose corrective decision is entered contiguously**,
because its fault is injected and its error row is therefore kept, while the other four earn their
errors and lose the row. That is what the 244 transient tasks actually contribute: not a retry, but
the only 244 recovery transitions in the corpus where the decision immediately before the correction
is itself observed. They are the stratum on which the recovery cost can be read without an
unobserved decision in between, and the 309 others are the stratum where it cannot.

**Stratum and split**, per R3: the rule is fitted on **ordinary** transitions only, held out by
episode within the train split and stratified by family. Every recovery transition is therefore
out-of-sample by construction, and all 553 of them are evaluated. Fitting on ordinary transitions and
evaluating on corrective ones is the claim: that the update rule generalises to the perturbation.

**25 episodes have no usable predecessor.** In 25 `wrong_path` tasks the fault lands at step 0, its
row is dropped, and the episode's first rendered decision is already the corrective one. They
contribute a decision to E1 and no transition to E2, and they are named here so that a count of 578
recovery decisions against 553 recovery transitions is not later read as a loss.

### 4.3 The carrier-ablation arm and the re-read diagnostic

Ruled 2026-09-10 from an external reading of the programme description, and it closes a hole this
document had. The transcript is a **redundantly available** carrier: `keep_last = 2` hides a tool
result two turns on, but the model's own note carries the state in words and attention can re-read
it. So a substitution test on residual features can report "no state" when the state is simply being
re-read from the text, and clamping features does not stop that. Three parts:

1. **A carrier-ablation arm.** The transcript re-rendered with the state-bearing fact removed from
   the note after the state-bearing position, so that a surviving effect must be carried by the
   model. For plan progress the removal targets the plan's progress statement.
2. **A direct re-read diagnostic at every decision position**: attention mass from the decision
   position onto the carrier tokens, per layer and head. The sliding window is the handle — the
   local layers see 1,024 tokens, so a carrier further back is reachable only through the global
   layers, and that is a testable difference rather than an assumption.
3. **E2 scored with and without the arm**, the difference reported as the carrier's share.

**Part 2 is orderable as it stands. Part 1 is not yet, and this section says why rather than
promising a rule that does not exist.**

**The note is not a note with a progress field. It is a running state summary in which nearly every
clause is a function of the step.** `pipeline/tasks.py` composes it as one `thought` string per step,
and one episode of `batch_update` shows the shape:

```
step 1: Inspected 0 of 4. Next: worker-0.ini mode=observe -> mode=strict. queue: worker-0.ini …; pending: …
step 6: Applied 1 of 4.  Next: worker-1.ini mode=safe -> mode=fast.       queue: worker-1.ini …; pending: …
```

The step index is recoverable from the counter, from the phase word, from which item is named next,
from the queue's remaining length and from the pending list — five carriers in one sentence, and
`aggregate_report` and `conditional_update` carry two counters at once ("values so far: 18; split
after 3 of 6", "loads so far: …; highest so far: …; Reading service 3 of 5"). Removing the counter
alone removes a label.

**So a removal rule is accepted only against a measurement, and `carrier_ablation.py` is it.** For
each episode it strips a candidate clause from every note and reports the mean share of steps still
told apart by what is left: 1.0 means the rule removed a label and left the carrier, and the floor of
1/n means the notes have become indistinguishable and the step is no longer readable from them.

| candidate rule | families moved off 1.0 |
|---|---|
| the explicit counter | 1 of 12 (`batch_update`, to 0.777) |
| counter and queue | the same 1 |
| counter, queue and next item | the same 1 |

**Eleven of twelve families are untouched by every rule I wrote**, and the diagnostic says why: the
counter pattern matches no note at all in eight of them, because their progress is carried by
accumulator lists ("values so far", "loads so far", "approved:"), by a shrinking pending list, or by
the terminal answer itself. This is a negative result about my candidate rules and not about the
ruling.

**The tension the pre-registration must state, because it may not be resolvable on this corpus.**
The clause that names the next action *is* the progress: "Next: worker-1.ini" is meaningful only
because worker-0 is done. Removing the progress may therefore not leave a well-formed note, and an
ablation that also removes the next action changes what the model is asked to do rather than what it
knows. Where no rule reaches the floor while leaving the action intact, the family is reported as
**arm not constructible**, with the measured share beside it, and E2 there is scored without the arm
and labelled as unablated. That is a worse result than a clean ablation and it is better than an arm
that removes a label and is read as removing the carrier.

**The acceptance criterion, fixed here before any rule is written:** a removal rule is used only if it
drives `distinguishable_share` to within 1/n of its floor on the family's episodes **and** leaves the
tool call and the next-action clause byte-identical. Both halves are checked by the harness, per
family, and the passing rules with their measured shares go into the seal.

### 4.4 E3 — retention within a pass. Ordered after the cache-exchange port, and not before

Whether the state at the decision position depends on the retained activations of earlier positions
or is recomputed from the recent tokens, bounded by the sliding window: the local layers see 1,024
tokens and the global layers see everything. WS-B's cache exchange is the separating control and is
unported. E3 is listed here so that its absence is a stated gap rather than a silence, and it is
**not** scheduled by this document.

**The fine-tuning arm** repeats E1–E3 on a fine-tuned checkpoint and is H3. It is last: it costs a
training run and means nothing until the base measurement exists.

---

## 5. The capacity rule, the depth, and the splits

**Fixed rank, with the null as the measurement of "more of anything".** The ladder is
r ∈ {1, 2, 4, 8, 16, 32}, identical on both models and on the controls. At every rank on every model
the permutation null is fitted the same way, and every gain is reported beside its own null. That
null is the measurement of how much a wider, deeper stream yields to any probe, which makes the
confound a number rather than a caveat.

- Fixed parameter count runs as **one robustness row at r = 8**. It is not the capacity rule.
- Matched control features are required regardless and are controls, not a capacity rule.
- **Depth**: the registry's six `layer_fractions` — 0.167, 0.333, 0.5, 0.667, 0.833, 1.0, identical
  in both entries — giving layers 6, 11, 17, 23, 28, 34 on the 4B and 8, 16, 24, 32, 40, 48 on the
  12B. The full profile is reported. **The headline is read at the 0.5 fraction and at r = 8, fixed
  here in advance**, so no layer and no rank is selected after the fact.
- **Splits**: held out by episode, stratified by family, never by position.
- **A figure without its rank and its null is refused.**

---

## 6. Ground truth

The horizon comes from the task definition — `Task.horizon`, which is `len(self.steps)` — and never
from a count of rendered rows, which is one short on 334 tasks (A2).

`prereg-inputs.json` lists, under `a2_decisions_without_a_rendered_row`, which decision of each of
those 334 tasks has no rendered row: one each, and the split by variant is total — `clean` 0 of 550,
`transient` 0 of 244, `failed_edit` 58 of 58, `stale_path` 30 of 30, `unknown_tool` 64 of 64,
`wrong_path` 182 of 182. Fact 1 survives, because the dropped turn still enters the context and the
turn count is still the step index; what does not survive is any horizon inferred from a row count.

---

## 7. The reported set, and the tolerances

**M = 12 pre-registered quantities**, fixed here and never added to after the fact:

| # | quantity |
|---|---|
| 1–4 | E1 on the 4B: {step index, steps remaining} × {permutation null, step-0 null} |
| 5–8 | E1 on the 12B: the same four |
| 9–10 | E2 closure on the 4B and on the 12B |
| 11–12 | the `pointer_chain` negative control on the 4B and on the 12B |

**The bound** is the first state variable's, unchanged: Hoeffding over M fixed quantities at
confidence 1 − α, `n ≥ ln(2M/α) / ε²`, with α = 0.05, computed by
`state_programme/tolerances.required_n`, whose two closed-form checks pass.

**The unit is the episode, not the decision.** The first draft of this section counted decisions and
transitions, and that was wrong: the design holds out **by episode**, precisely because decisions
within one episode are not independent — they share a task, a prompt prefix and a plan — and a
concentration bound over dependent observations is a bound over nothing. The first state variable's
own order counted episodes for the same reason. So each episode contributes one bounded observation,
its own mean, and n is the number of held-out episodes. Both computations are below, because the
correction is the interesting part and hiding the first one would hide it.

| | evaluation set | unit | n | ε declared | needs n ≥ |
|---|---|---|---:|---:|---:|
| ε_main | E1 on the test split | episodes | 240 | **0.17** | 214 |
| ε_ord | E2 ordinary transitions, train split | episodes | 840 | **0.09** | 763 |
| ε_sub | E2 corrective transitions | episodes (one each) | 553 | **0.11** | 511 |

*Superseded, kept for the comparison:* counting decisions gave n = 1,781 and ε_main = 0.06,
transitions gave n = 4,122 and ε_ord = 0.04. Those are two to three times tighter than the corpus
supports, and every one of them was an artefact of counting dependent observations as independent.

**ε_main is 0.17 and not 0.16.** The bound at 240 episodes is 0.1604, so 0.16 would be the honest
figure to a reader — but declaring 0.16 requires n ≥ 242, and there are 240 test episodes. Two short.
The declared value rounds **up** or the guarantee is not met, and a threshold that misses by two is
exactly the kind that gets rounded into existence after the fact.

**ε_sub stands at 0.11 and is still a loosening**, now for a plainer reason than before: there are
553 corrective transitions in the whole corpus, one per recovery episode, and no split of them
reaches n at ε_main. The alternative to loosening is not reporting the primary estimand's informative
stratum at all. This is the existing budget rule's shape — loosen ε_sub, write both numbers and the
reason, before the run — applied to a population limit rather than a device-time one.

**What the run can claim, given these.** At ε_main = 0.17, E1 can support a claim that one model's
decodability exceeds a null by more than about seventeen points of accuracy, and cannot support a
claim about a smaller gap. That is a real weakening against the draft's 0.06 and it is the true
resolution of this corpus at this M. If the eventual gap is smaller than that, the finding is
"below the pre-registered resolution", not "no effect" — and it is reported in those words.

*An episode-level bootstrap would be tighter than Hoeffding here, since Hoeffding assumes only
boundedness and ignores the observed variance. If the Chief prefers one, these Hoeffding numbers stay
beside it as the distribution-free floor; the seal can carry both. Flagged, not chosen.*

**The drop rule is inherited unchanged**: a quantity whose bootstrap bound on its contrast does not
exceed zero is dropped with its reason, and nothing is added to M after the pilot.

---

## 8. The capture plan and its budget

Decision positions, every layer, bf16 native, both models, the same rendered rows. Depth and width
come from the registry entries and the published configs, never from memory; both are checked
against `model.config` on the card before the first capture is written.

| | layers | hidden | residuals per position | KiB per position | GiB, 7,629 decisions |
|---|---:|---:|---:|---:|---:|
| `gemma3-4b-cuda-bf16` | 34 | 2,560 | 35 | 175.0 | 1.273 |
| `gemma3-12b-cuda-bf16` | 48 | 3,840 | 49 | 367.5 | 2.674 |
| both | | | | 542.5 | **3.947** |

This replaces R6's 4.6 GiB, which was correct arithmetic on 8,907 rendered rows; on the 7,629
distinct decisions, with the 348 chat replay rows excluded and the 930 duplicates recognised as
copies, it is 3.95 GiB. Against 235 GiB free on the card at 10:37Z.

Captures live outside the checkout at `/workspace/captures/<entry>/`, because `data/` was ignored
only after this corpus was made and the resume key hashes untracked files.

**Prompt lengths are measured in characters here, not tokens**, because no Gemma tokenizer is
present on the laptop: minimum 1,824, median 4,738, maximum 14,316. At 4.0 characters per token
those give a median of about 1,185 tokens and a longest of about 3,579, which reproduces the Chief's
independently measured 1,200 and 3,600 to within one per cent. The token figures are re-measured on
the card at capture time and the measured values go in the manifest beside these.

---

## 9. The exploratory stratum: twelve episodes, named now

Every position of the **final** decision of twelve pre-registered episodes, one per family, test
split, difficulty 2, clean. Selected by rule — the first `task_id` in sorted order within each family
— so the choice is reproducible and was not a pick. **Labelled exploratory and read only after the
seal.**

| family | task_id | decisions | final step |
|---|---|---:|---:|
| aggregate_report | `test-aggregate_report-0011-clean` | 14 | 13 |
| batch_update | `test-batch_update-0010-clean` | 14 | 13 |
| calculate | `test-calculate-0002-clean` | 2 | 1 |
| conditional_update | `test-conditional_update-0009-clean` | 10 | 9 |
| cross_reference | `test-cross_reference-0008-clean` | 11 | 10 |
| ledger_reconcile | `test-ledger_reconcile-0007-clean` | 12 | 11 |
| list | `test-list-0005-clean` | 3 | 2 |
| pointer_chain | `test-pointer_chain-0006-clean` | 8 | 7 |
| read | `test-read-0000-clean` | 2 | 1 |
| search | `test-search-0001-clean` | 3 | 2 |
| synthesis | `test-synthesis-0003-clean` | 4 | 3 |
| update | `test-update-0004-clean` | 4 | 3 |

None of the twelve has a dropped decision or an oversampled one, which follows from their being
clean: only recovery decisions are duplicated and only recovery variants lose rows.

About 15,776 positions in total at 4.0 characters per token, so **about 8.2 GiB for both models** —
below R6's estimate of about 12 GiB, on the same basis that the main stratum is below its estimate.
The figure is re-computed from the measured token counts on the card before the stratum is written.

---

## 10. Reporting

Three states, as the first state variable's: **measured**, **untestable**, **not measured**, with the
reason attached to each. Fixed here:

- E1 in its original predictive-sufficiency form: **untestable**, by Facts 2 and 3.
- E1 as decodability at matched rank: **measured**, with rank, depth and null on every figure.
- E2: **measured**, with ε_ord and ε_sub named separately.
- E3: **not measured**, pending the cache-exchange port.
- H3 and the fine-tuning arm: **not measured**, not ordered.

After the capacity rule and the controls, an E1 advantage is still not "a richer representation of
the task". On canonical plans under teacher forcing it is "encodes (family, step) more linearly", and
the report says so in those words.

---

## 11. Preconditions, checked on the card before the first capture is written

Each fails closed and names itself.

1. **The corpus is the counted corpus. Checked, 2026-09-09, and it passes.** All five per-file
   digests in `prereg-inputs.json` match the files under
   `/workspace/rendered-corpus/agent_v2e-gemma3-4b/`, and `prereg_inputs.py` run on the device copy
   reproduces every figure in §1 and regenerates `capture-set.jsonl` to the same
   `capture_set_sha256`. CPU only, seconds, no card time; nothing was written into the shared
   checkout. This is the re-run on the device copy that order §1 asks the owner of the
   pre-registration to do.
2. **The two models share the rendering byte for byte.** The 12B entry asserts it in a comment with
   the tokenizer digests; the check compares the two snapshots' `tokenizer.json` and
   `tokenizer.model` and re-renders one episode under each entry, comparing the digest. The corpus
   was rendered for the 4B entry, so this is a precondition and not an assumption.
3. **Depth and width match the registry.** 34 × 2,560 and 48 × 3,840 against `model.config`;
   `capture_budget.json` carries `verified_on_device: false` until this passes.
4. **`cache.strategy: none` on both entries**, since H2's restatement depends on it.
5. **The seal exists.** The reader refuses without it. The captures may be made first, as §16.18
   requires, but they are not looked at.

---

## 12. What would falsify each hypothesis

- **H1** is falsified by the two models' decodability agreeing at every rank on the ladder once the
  null is subtracted, and equally by the 4B exceeding the 12B.
- **H2**, in its restated within-pass form, is falsified by the state at the decision position being
  recoverable from the sliding window alone at the same rank.
- **The instrument** is falsified before either: by the step index failing to decode (Fact 1's
  control), or by `pointer_chain` steps-remaining decoding above the permutation null.

The programme's own record of this week is that the instrument fails before the model does. **A first
result showing the 12B more predictive than the 4B is, until the capacity rule is applied and the
controls run, a measurement of the 12B being bigger.**

---

## 13. Open before sealing

1. **§1.1's correction is a change to a ruling, not a note.** A1's retry clause has nothing to
   predict and A3's occurrence index addresses copies. Both need the Chief's word before the seal.
2. **E2's retrieval score replaces a derived tolerance** (§4.2). Mine, and flagged.
3. **The tolerances are recomputed on the episode** (§7), which loosens ε_main from 0.06 to 0.17
   and ε_ord from 0.04 to 0.09; ε_sub stands at 0.11. The first draft counted decisions and
   transitions as independent observations when the design holds out by episode, which they are not.
   The correction is the Chief's, through Codex's file-only review, and it is applied rather than
   argued with. Whether to put an episode-level bootstrap beside the Hoeffding floor is open.
4. **The carrier-ablation arm's removal rule** (§4.3) does not exist for eleven of twelve families,
   and may not exist for some of them at all, because the clause that names the next action is the
   progress. The acceptance criterion and the harness are fixed; the rules are not written. The
   re-read diagnostic is unaffected and is orderable now.
5. The golden-test control design has landed. The capture pass is not scheduled, and the seal follows
   Codex's file-only review of this draft. Until then this document stays unsealed.
