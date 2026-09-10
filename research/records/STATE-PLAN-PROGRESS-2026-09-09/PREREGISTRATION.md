# Pre-registration: plan progress as the state programme's second state variable

**UNSEALED DRAFT. D-CRO, 2026-09-09.** Owner: D-CRO, under
`design_specifications/pending/STATE-PLAN-PROGRESS-ORDER-2026-09-10.md` (R7 and the Chief's
amendment). It is **not sealed**: sealing follows Codex's file-only review of this draft and
precedes the first *reading* of any capture. The golden-test control design has landed and is no
longer a blocker; the capture pass does not wait for the seal, because §16.18 permits captures to be
made before it so long as none is read.

Nothing here has been read from a capture, because none exists. Every number in §§1–2 and §§7–9 is
computed by the two scripts in this directory from the rendered corpus alone, on the laptop, with no
model and no card:

| artefact | what it is |
|---|---|
| `prereg_inputs.py` → `prereg-inputs.json` | the corpus facts, the census, the twelve episodes |
| `capture_budget.py` → `capture-budget.json` | what capturing the pre-registered set costs |
| `capture-set.jsonl` | the capture set itself, one line per decision, 7,629 lines |
| `count_corpus.py` → `corpus-count.json` | the order §1 count, re-run and falsified per claim |

## What this study can and cannot say, before anything else

**E1 can support a claim that one model's decodability exceeds its null by more than about
twenty-three points of accuracy. It can support nothing about a smaller gap.** That is the resolution
this corpus affords at this number of pre-registered quantities, and it is a large weakening of what
the first draft imagined — 0.06 — which came from counting decisions as independent observations when
the design holds out by episode, and from bounding a paired difference as though it were a single
[0, 1] mean. Neither was defensible; the twenty-three points are.

If the eventual gap is smaller than that, the finding is **"below the pre-registered resolution"**,
which is not the same as "no effect" and is reported in those words. And the epsilons are **declared
resolutions, not coverage statements**: the episodes are not independent draws, the cross-fitted
scores share training data, and §7 states which assumption is doing the work.

The comparable numbers for the rest: 0.13 on E2's ordinary transitions, 0.15 on its corrective ones,
0.23 on the contiguous subgroup and 0.20 on the across-a-gap subgroup, that last met by exactly the
309 episodes it requires and no more.

---

Corpus on the card: `/workspace/rendered-corpus/agent_v2e-gemma3-4b/`, archive
`7fe6e64b89749b997854638b260d7654316636da3d8eaa59924ad9ed62f99120`. Per-file digests are in
`prereg-inputs.json` under `file_sha256`, and they match the corpus manifest's own `outputs` block,
so the counted rows and the shipped rows are the same bytes.

The capture set has **two** digests and they are not interchangeable, so both are pinned and each is
labelled:

| | digest | what it fixes |
|---|---|---|
| the set | `1a7cfbdd4e21fc203eafbcc3ec50b96afcb214c169f21c7ba968ca83dfc9709a` | the logical triples `(task_id, step, prompt_sha256)`, in order — what is to be captured |
| the file | `8bbc8062249ce1cc0e15050a866cd726a2c205a969f3eeab3840fcb5520cee3e` | the bytes of `capture-set.jsonl` — what was shipped |

The first is the one that must survive a re-run on another machine, since it is invariant to
formatting; the second is the one that says the file on the card is the file written here. Renaming
`prompt_sha256` to `messages_sha256` moved the second and left the first untouched, which is what
that distinction is for.

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
`row_ordinals`, the checkpoint identity, the layer count and width, the decoding mode, the device
reading, and the `basis` field. The ground truth is stored beside the cell and comes from the task,
never from the model.

**Three digests, because they are three different claims.** `messages_sha256` is canonical JSON over
`messages[:-1]` — the semantic record the capture set was enumerated under, and **not** the model's
input; two rows with one message record and different rendered prompts share it. It was called
`prompt_sha256`, which reads as the model's input and is neither. `rendered_prompt_sha256` is the
byte digest of the string that was tokenized. `token_ids_sha256`, with `token_ids_length`, is the ids
actually consumed. The capture boundary checks the second against the row it is about to forward, and
the resume check verifies both, because a semantic digest cannot see a change in the bytes.

**The checkpoint identity is a digest over the loader's complete manifest**, every weight shard and
the config, with the config's own digest recorded separately as `config_sha256`. The config digest
alone is not an identity: two checkpoints with one config and different weights would share it.

**The seam attests; the writer checks.** The batch width, the anchor width, the arithmetic path, the
position, the dtype and the shape are recorded from what the forward pass reports about itself, and
the writer refuses any that disagree with this contract. It does not write the contract's constants
into the cell — a writer that supplies the answers checks only itself, and a pass that ran at another
width or on a promoted path would be recorded as conforming.

**Resume verifies rather than trusts, and names what it verified.** Every *requested* decision is
checked on both paths, not only the one that captures it: the corpus row against the request's own
digest, and then, for a decision a previous pass already captured, the checkpoint identity, the
semantic record, the rendered bytes, the enumeration the request now makes, the shard's presence and
its bytes, **and the ids this run's tokenizer would produce**. That last one needs the model input
prepared *before* the reuse decision, because two tokenizers give different ids for the same bytes
and the checkpoint identity covers no tokenizer asset. A pass run without that preparation still
resumes, and its summary says in words that the consumed ids were not verified rather than reporting
a verification it did not perform.

**Two fields the first draft did not carry, both required.**

- **`forward_batch: 1`**, and `anchor_batch: 1` beside it. The bf16 forward is not batch-invariant —
  measured on the card, no block of the stack bitwise identical between two widths
  (`WSD-FD-CALIBRATION-2026-09-10`) — so a captured residual is a residual *of a width*. A batched
  capture would carry a width-dependent arithmetic term into every probe fitted on it, silently.
  Captures are made at width 1, which is the ordinary forward, and the manifest says so per cell.
- **The token index**, after tokenization, per cell. "The last prompt token" is a rule for finding
  the position and is not a record of which position was found. A prose definition cannot be checked
  against the capture; an integer can.

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
- **The unknown-horizon control, restated. The version that stood here was wrong at the last step.**
  It said `pointer_chain` steps-remaining must sit at the permutation null on both models and that
  any decoding above it is a leak. That is false at the terminal decision: the chain's end is
  **visible in the prompt**. Measured on the corpus — all **94** terminal `pointer_chain` decisions
  (20 of them in E1's test split) contain `Node: final` in the text the model is given, and **0 of
  575** non-terminal ones do. Recognising that a chain has ended is reading the input, not leaking
  the label, and the old veto would have condemned a probe for doing the obvious right thing.

  **Eligible positions are pre-specified by a rule on the visible text, never on the label.** A
  `pointer_chain` decision is ineligible for this control exactly when its rendered prompt contains
  the terminal marker. That rule is applied to the prompt, so it can be evaluated without knowing
  the horizon; using `step == max(step)` would be the label wearing a rule's clothes, and would also
  be circular — the label is what the control exists to protect.

  **The null is conditional on what stays visible**, not permutation. Each conditioning variable is
  named with why it is legitimately available: the **chain length so far**, which the model can
  count from the transcript; **survival** — that the chain has not ended, which is the absence of the
  marker above; the **family and difficulty**, both in the task prompt; and the **visible progress
  note**, which the protocol writes at every step. A probe may know all of that without knowing the
  horizon.

  **The null is a conditional randomization, specified here as a procedure.** Naming its inputs is
  not enough to run it, and an unspecified null cannot govern a verdict.

  1. **Strata.** Eligible decisions are grouped by `(difficulty, nodes visited so far)`. `family` is
     constant by construction, `survival` is constant given eligibility, and the visible progress
     note is a deterministic function of nodes visited — so it adds no stratum, and saying that is
     part of the specification rather than an omission from it.
  2. **Extraction.** `nodes visited so far` is read from the rendered prompt as the count of
     `read_file` observations before the decision, never from the label or the step index.
  3. **The randomization.** Within each stratum the steps-remaining labels are permuted across
     decisions, **10,000 draws**, seeded from the seal. Permuting within the stratum is what makes
     the null conditional: everything the conditioning variables carry is preserved exactly, and
     only the association with the residual is destroyed.
  4. **Unsupported strata.** A stratum with fewer than **10** eligible decisions is dropped, with its
     size reported; strata are never merged to reach the threshold, because merging changes the
     conditioning set after seeing it.
  5. **The comparison.** The probe's accuracy on eligible decisions against the permutation
     distribution of the same quantity, at the ε and α of §7's table. The veto fires when the probe
     exceeds the null's `1 − α` quantile by more than ε.

  **What firing supports, stated narrowly.** Exceeding this null is evidence that the residual
  carries steps-remaining information **beyond what the four conditioning variables carry** — it is
  not a proof of leakage in general, because the conditioning set may be incomplete and a null built
  on an incomplete set can be beaten by a probe reading a legitimate cue nobody listed. The veto is
  therefore an instruction to **investigate and name the cue**, and the instrument is called into
  question only if no visible cue accounts for the excess. Codex's objection is right and this is the
  narrower claim that answers it: naming several conditioning variables does not establish their
  sufficiency.

  **Until this procedure has been executed the control reads *not measured* and the veto cannot
  fire** (§10). It is specified before any capture is read, which is the condition that matters.
- **Evaluation set**: the test split's 1,781 decisions, 240 episodes, clean variants only. Held out
  by episode by construction, never by position.

### 4.2 E2 — update closure. The primary estimand

Does the state read after decision *k* move to the state the task's own dynamics imply for decision
*k+1*? The rule predicts +1 on an ordinary transition and the recovery cost at a corrective one.

**Score, and why it needs no tolerance.** For each held-out transition, transport the state read at
*k* by the rule and ask whether the transported state is nearer to the state read at *k+1* than to
the state read at any other decision of the same episode. A hit is bounded in [0, 1], which is what
makes §7's bound apply. This is a deliberate departure from the first state variable's
derived-tolerance design: a retrieval score against an explicit chance level has no free parameter to
set after looking, and no pilot is needed to set one.

**Frozen here, because each of these would otherwise be a choice made after seeing the score:**

- **The candidate set is all *N* decisions of the episode, the source included.** The chance level is
  then exactly `1/N`, which is the uniform-guess reference this document already quotes. Excluding
  the source would make it `1/(N−1)` and the two must not be mixed.
- **Strict nearest target: a tie is a miss.** Named because the alternative — splitting credit — is
  defensible and gives a different number, and because ties are not rare in a bounded space.
- **The metric is named in the space it is computed in.** The retrieval and the distance are both in
  the **residual space at the read layer**, on vectors as captured — native bf16 promoted to float32
  for the arithmetic and not otherwise transformed — under the Euclidean norm. If the transport rule
  is fitted with any normalisation of its own (a whitening, a per-layer scale), that normalisation is
  part of the metric and is declared with the rule and applied identically to the candidate set; a
  distance in a space the rule normalised and the candidates did not is not a distance between them.
- **The transport distance is reported descriptively beside the hit rate**, in that metric,
  aggregated as the mean within an episode and then the mean across episodes. Never
  pooled across decisions, which would weight long episodes more heavily and is the same mistake as
  counting decisions instead of episodes in §7. It is descriptive: it adds no confirmatory estimand
  and enters no bound.

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
promising a rule that does not exist.** What is pre-registered here is the **text-removal** arm and
only that. An arm built some other way — masking the carrier spans as attention keys, say, which
truncates the carrier without touching a token — is a different intervention with a different
failure mode, and if one is ever ruled into E2 it is ruled in by an amendment that says so. This
document does not promise it and must not be read as anticipating it.

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

**So a removal rule is accepted only against a measurement, and `carrier_ablation.py` is the
*screen* for it — not the gate.** For each episode it strips a candidate clause from every note and
reports the mean share of steps whose note remains a **distinct string**: 1.0 means the rule
certainly did not remove the carrier. A low share is a screen passed and not the gate met, because
distinctness is a property of strings — a step can be recoverable from text that repeats, or
unrecoverable from text differing in an irrelevant token. **The two-part gate below is not
implemented**: the byte-identity half is checked by nothing today, and no figure from this script
should be read as the gate having been applied.

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
- **Training eligibility, stated rather than implied.** The 240 **clean test episodes** are never in
  any fit: not the probe's, not the null's, not the transport rule's, and not any fold's training
  part. §7.1's cross-fitting partitions the *train* split; the test split is evaluation only. This is
  written down because "held out by episode" describes the mechanism and not the population, and a
  fold assignment that happened to place a test episode in a training part would satisfy the
  mechanism while destroying the claim.
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

**Every one of the M = 12 quantities is a paired contrast, and the bound must say so.** Each is a
score and its null computed on the *same* held-out episodes, so the observation is a difference
spanning [−1, 1] and not a mean spanning [0, 1]. The helper's `ln(2M/α)/ε²` is the single-mean form;
a paired difference of accuracies needs `2 ln(2M/α)/ε²`, and `required_n` now takes the range width
rather than carrying a second formula. Bounding a contrast as though it were a single [0, 1] mean
claims a resolution its own range does not support, which is what the first draft did.

### THE OPERATIVE TOLERANCE TABLE — the only one, and every other value in this document is superseded

| | evaluation set | unit | n | ε declared | needs n ≥ | spare |
|---|---|---|---:|---:|---:|---:|
| ε_main | E1 on the test split | episodes | 240 | **0.23** | 234 | 6 |
| ε_ord | E2 ordinary transitions, train split | episodes | 840 | **0.13** | 731 | 109 |
| ε_sub | E2 corrective transitions | episodes (one each) | 553 | **0.15** | 549 | 4 |
| ε_sub, contiguous (`transient`) | subgroup | episodes | 244 | **0.23** | 234 | 10 |
| ε_sub, across a gap | subgroup | episodes | 309 | **0.20** | 309 | 0 |

*A second table survived below when §7 was recomputed at the paired range width, unmarked, and Codex
found it: it declared, at range width 1 and now superseded, ε_main = 0.17 and ε_sub = 0.11. Under
this document's own rule those superseded values are not merely stale but **unmet**: 0.17 needs 428
episodes against 240, and 0.11 needs 1,021 against 553. A reader
taking the wrong table would have claimed a resolution the corpus does not support, and the two
tables could flip a verdict between them. Every superseded value below is now marked in place rather
than deleted, and `check_prereg.py` refuses a document that declares two operative values for one
tolerance.*

*A note on the convention, because it is not the textbook one, and it is now ruled.* At range width
1 the helper returns `ln(2M/α)/ε²`, which is **twice** the textbook Hoeffding `w²ln(2M/α)/(2ε²)`. At
width 2 the two coincide exactly. The Chief has ruled that it stays: every quantity in this
pre-registration is at width 2 and therefore standard, the closed-form checks and every earlier
figure stand, and a future single-mean claim inherits a conservative resolution rather than a moved
number. Which form is which is stated here so that a later reader is not left to rediscover it.

*Superseded, kept for the comparison, and marked:* counting decisions gave n = 1,781 and
ε_main = 0.06, transitions gave n = 4,122 and ε_ord = 0.04. **Coverage is not supported at those
values under the episode unit** — that is the label they carry wherever they appear — and every one
of them was an artefact of counting dependent observations as independent.

**Subgroups carry their own ε; the pooled one attaches to the pooled set only.** E2's corrective
stratum is two populations that differ structurally (§4.2): the contiguous transitions, where the
decision before the correction is itself observed, and those spanning a step with no rendered row.
Reporting either against the **pooled 0.15** would claim a resolution its own n does not support.
*(This sentence read "the pooled 0.11" — a superseded single-mean value — until Codex found it.)*

| stratum | episodes | exact ε | **declared** | needs n ≥ | spare |
|---|---:|---:|---:|---:|---:|
| corrective, pooled | 553 | 0.1494 | **0.15** | 549 | 4 |
| contiguous (`transient`) | 244 | 0.2250 | **0.23** | 234 | 10 |
| across a gap | 309 | 0.1999 | **0.20** | 309 | **0** |

**The across-a-gap subgroup is met exactly and with nothing to spare**: 0.20 requires 309 episodes
and there are 309. Every row here carries the n it needs beside the n it has, because at the previous
range width two of these declarations were rounded to a value the corpus did not support, and a row
that shows only the declared ε cannot be checked by a reader at all.

**Independence between episodes is an assumption, and it is stated rather than relied on silently.**
The bound treats each episode's mean as one independent bounded observation. Episodes within a family
share a generator template and differ in their sampled files, values and paths, so they are not
independent draws from an arbitrary distribution; what the bound covers is inference **to new
episodes of the same task distribution**, which is the target this document claims and not a broader
one. A family-level effect would not be detected by it.

**An episode-level bootstrap is declared beside the Hoeffding form, for the drop rule, and it is
paired.** The drop rule reads a bootstrap over episodes — resampling episodes, not decisions — at the
same α. Because every quantity is a score and its null on the same episodes, the resampling index is
drawn **once and applied to both arms**: `paired_bootstrap_bounds`, which requires equal lengths and
refuses otherwise. The production helper named `paired_bootstrap_lower_bound` previously called the
independent-arm version, so the name asserted a property the code did not have; it is fixed, and the
independent-arm function keeps its own name for the case it is right for. Frozen here: the pairing,
the scope (episodes), the seed (20260910), that the folds are refitted within each resample, and the
stratification by `(split, family, variant)`.

Both are reported for every quantity, and where they disagree the wider one governs the claim.

**SUPERSEDED — single-mean values, kept for the reasoning only.** The two paragraphs that stood here
declared ε_main 0.17 and ε_sub 0.11. Both were computed at range width 1 and neither is operative;
under the paired width they are not even attainable (0.17 needs 428 episodes against 240; 0.11 needs
1,021 against 553). The operative values are 0.23 and 0.15 in the table above.

What survives is the rule they were written to state, which applies unchanged at the new width: **the
declared value rounds up or the condition is not met**. At width 1 that turned 0.16 into 0.17 because
0.16 needed 242 episodes against 240 — two short. At the paired width it is why the across-a-gap
subgroup is 0.20 and met by exactly its 309, and why every row above carries the n it needs beside
the n it has. A threshold that misses by two is exactly the kind that gets rounded into existence
after the fact.

And the loosening's reason survives too: there are 553 corrective transitions in the whole corpus,
one per recovery episode, so ε_sub is loosened against a **population** limit rather than a
device-time one — the budget rule's shape, both numbers and the reason written before the run.

**What the run can claim, given these.** At ε_main = 0.23, E1 can support a claim that one model's
decodability exceeds a null by more than about twenty-three points of accuracy, and cannot support a
claim about a smaller gap. That is a substantial weakening against the draft's 0.06, and it is the
resolution **declared** for this corpus at this M — not a coverage statement, because the assumptions
below do not hold exactly. If the eventual gap is smaller, the finding is "below the pre-registered
resolution", not "no effect", and it is reported in those words.

*An episode-level bootstrap is usually tighter than Hoeffding, which assumes only boundedness and
ignores the observed variance. Both are carried, and neither is a guarantee here: see the assumptions
below and §7.1 — the episodes are not independent draws and the cross-fitted scores share training
data, so **every ε in this section is a declared resolution and not a coverage statement**.*

**The drop rule is inherited unchanged**: a quantity whose bootstrap bound on its contrast does not
exceed zero is dropped with its reason, and nothing is added to M after the pilot.

### 7.1 Cross-fitting, so that every transition gets an out-of-fold prediction

Holding out one split would leave most of the corpus scored in-sample or not scored at all. Every
held-out prediction is instead produced by **cross-fitting over episodes**: K = 5 folds, assigned by
episode, stratified by `(split, family, variant)`, and applied to the probe and to the transport rule
alike, so that all 4,122 ordinary and all 553 corrective transitions receive a prediction from a
model that never saw their episode.

The assignment is computed, not described: `folds.py` walks each stratum's episodes in sorted
`task_id` order, round-robin, with the stratum's starting fold offset by a digest of the seed and the
stratum name — a function of the corpus and the seed and of nothing else, with no shuffle and no
dictionary order in it. Seed **20260910**, and the assignment's own digest is
`c5e9622f7bef663c96614a99bab7e120873ad9eb79617912526c02e48016372c`, which goes into the seal so the
folds cannot move afterwards.

| fold | episodes | clean | transient | wrong_path | unknown_tool | failed_edit | stale_path |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 220 | 108 | 47 | 38 | 10 | 11 | 6 |
| 1 | 232 | 111 | 50 | 38 | 15 | 12 | 6 |
| 2 | 226 | 110 | 50 | 34 | 15 | 11 | 6 |
| 3 | 224 | 112 | 48 | 34 | 12 | 12 | 6 |
| 4 | 226 | 109 | 49 | 38 | 12 | 12 | 6 |

**And the limitation the bound inherits from cross-fitting, which is why §7's epsilons are declared
resolutions and not coverage.** The K models share training episodes with one another, so the
out-of-fold scores are **not** independent across folds even though each is out-of-sample for its own
episode. Codex's parity example makes the size of it concrete: five held-out errors that are
identical to one another have variance 0.25 where independence would give 0.05. The concentration
bound is applied to the episode-level scores as if they were independent, and the sharing makes that
an **approximate, assumption-dependent summary** rather than a guarantee — the word "guarantee" does
not appear about it anywhere in this document, and neither does "distribution-free floor", because
neither is true of what is computed here.

Three assumptions are therefore stated rather than relied on: that episodes are independent draws
from the task distribution, which families sharing a generator template make approximate; that the
out-of-fold scores are independent, which the shared training episodes make false and which the
paired bootstrap does not repair either; and that the scores are bounded, which alone is exact. What
the numbers support is a comparison at a declared resolution, and the reader is told which of the
three is doing the work.

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

**Measured, both passes complete**, 7,629 of 7,629 each, `complete: true`, `whole_set: true`, 30
shards apiece:

| | predicted | on disk | elapsed | ms/decision | peak reserved |
|---|---:|---:|---:|---:|---:|
| 4B | 1.273 GiB | **1.3 GB** | 7.7 min | 75.6 (alone) | 9.52 (est. 8.1) |
| 12B | 2.674 GiB | **2.7 GB** | 42.2 min | 331.6 (shared) | 26.45 (est. 23.7) |

The storage table was right to the tenth of a gigabyte on both. The memory estimates ran about 15%
light, in the direction the 4B measurement already predicted. The 12B's rate is on a shared card and
is not a property of the model.

This replaces R6's 4.6 GiB, which was correct arithmetic on 8,907 rendered rows; on the 7,629
distinct decisions, with the 348 chat replay rows excluded and the 930 duplicates recognised as
copies, it is 3.95 GiB. Against 235 GiB free on the card at 10:37Z.

Captures live outside the checkout at `/workspace/captures/<entry>/`, because `data/` was ignored
only after this corpus was made and the resume key hashes untracked files.

**Prompt lengths, measured on the card at capture time**, from the 4B pass's own manifest — the
assumed figures are kept beside them because the assumption was declared and should be scored:

| | assumed (4.0 chars/token) | **measured** |
|---|---:|---:|
| minimum | 456 | **416** |
| median | ~1,185 | **1,310** |
| 99th percentile | — | **3,300** |
| longest | ~3,579 | **4,274** |

The implied ratio is **3.617** characters per token, not 4.0, so every token figure derived from the
assumption ran about 10% light and the longest row is 19% longer than assumed. The assumption was
declared as one and is now scored rather than quietly replaced.

Three things the manifest establishes that no arithmetic could: the captured position is the last
token of the prompt in **7,629 of 7,629** cells, every cell reports `forward_batch` and
`anchor_batch` of 1, every cell reports `capture_dtype: native`, and all 7,629 carry **one**
checkpoint identity. Those are the contract's own claims, read back from what was written.

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

**Measured from the 4B capture's manifest: 18,018 positions**, against the 15,776 assumed at 4.0
characters per token — 14% more, in the same direction as the length correction above. At the
per-position costs of §8 that is about **9.3 GiB for both models**, still below R6's estimate of
about 12 GiB. The stratum is read only after the seal; the figure is now a measurement rather than a
projection.

---

## 10. Reporting

Three states, as the first state variable's: **measured**, **untestable**, **not measured**, with the
reason attached to each. Fixed here:

- E1 in its original predictive-sufficiency form: **untestable**, by Facts 2 and 3. This one is
  fixed now, because it is a property of the corpus and not of a run.
- The unknown-horizon control of §4.1: **not measured**, and **its veto cannot fire** until the
  conditional randomization there has been executed. Its procedure is specified before any capture is
  read, which is what the seal is for; its result is not.
- E1 as decodability at matched rank: **not measured**. It becomes *measured* when it has been
  executed, and not before. A pre-registration that records its own estimands as measured is
  describing an intention; the state is a fact about the run and this document has had none.
- E2: **not measured**, on the same reasoning.
- E3: **not measured**, pending the cache-exchange port.
- H3 and the fine-tuning arm: **not measured**, not ordered.

When E1 and E2 are executed their state changes with the figures beside it — rank, depth and null for
E1, and ε_ord, ε_sub and the two subgroup epsilons for E2 — and the reason for any that stay *not
measured* is written in the same line.

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
2. **The two models share the rendering byte for byte. Checked, 2026-09-10, and it passes — by a
   stronger test than the one planned here.** Rather than compare tokenizer files and re-render one
   episode, both capture passes recorded the digest of the ids they actually consumed, per decision.
   Those digests are **identical at 7,629 of 7,629 decisions**, and the read position is identical at
   7,629 of 7,629. So the two models were given the same input, token for token, at every decision
   the comparison will use — which is the thing the tokenizer check was a proxy for, measured
   directly on the artefacts that will be read.
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
  control), or by the unknown-horizon veto of §4.1 firing. **That veto is stated once, in §4.1, and
  in its conditional form only.** Beating the *unconditional* permutation null on `pointer_chain`
  steps-remaining does **not** invalidate the instrument: the chain's end is visible in the text at
  the terminal decision, so a probe that reads it is reading its input. This paragraph carried the
  unconditional form until Codex found the contradiction with §4.1; the two said different things
  and the checker certified the document anyway. E1's ordinary permutation nulls are unaffected —
  they serve a different purpose and are not this veto.

The programme's own record of this week is that the instrument fails before the model does. **A first
result showing the 12B more predictive than the 4B is, until the capacity rule is applied and the
controls run, a measurement of the 12B being bigger.**

---

## 13. Where this stands, and what is open before sealing

**Settled by measurement rather than by choice.** The corpus facts and their basis (§1); the
correction that the repeated step indices are oversampled copies and not retries (§1.1); the capture
unit, key and contract, with its three digests, the checkpoint identity over the whole manifest, and
the seam attesting while the writer checks (§2); E1's relabelling with its two nulls and the
`pointer_chain` negative control (§4.1); E2's frozen retrieval, its metric named in its space, and
the training eligibility mask (§4.2, §5); the rank ladder and fixed headline depth (§5); the
tolerances at the paired range width, with their subgroups, their assumptions and the n each needs
beside the n it has (§7); the cross-fitting folds, computed and digested (§7.1); the storage table on
the corrected unit (§8); the twelve exploratory episodes by rule (§9); the three-state reporting,
with E1 and E2 reading *not measured* until they are executed (§10); and precondition 1, checked on
the device copy and passing (§11).

**Open, each for a stated reason.**

1. **The carrier-ablation arm's removal rule** (§4.3). Measured not to exist for eleven of twelve
   families, and possibly not constructible where the clause naming the next action is the progress.
   The screen exists; the gate's byte-identity half is implemented by nothing, and the document says
   so rather than implying otherwise. The re-read diagnostic does not depend on it and is orderable.
2. **E3** (§4.4): not measured, not scheduled, waiting on WS-B's cache exchange.
3. **The seal**, which follows Codex's file-only review of this draft and precedes the first
   *reading* of any capture. The capture pass itself does not wait for it: §16.18 permits captures to
   be made before the seal so long as none is read.

**Mine, and flagged as mine.** E2's retrieval score in place of a derived tolerance (§4.2); the
subgroup epsilons and the decision to declare the rounded-up value at every stratum (§7); and the
fixed headline rank and depth (§5). Any is the Chief's to overturn before the seal. The range-width
convention was mine to flag and is now ruled, so it has left this list.

**What this document is not.** Nothing in it has been run. Every figure is computed from the rendered
corpus on the laptop, or measured on the card and cited to the record that measured it. The estimands
read *not measured* because they are.
