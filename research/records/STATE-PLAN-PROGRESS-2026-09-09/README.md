# Plan progress as the state programme's second state variable, and the 12B capture that would test it

**Draft, D-CRO, 2026-09-09, on the Chief's instruction. Cites plan §16.18.** Nothing here has run.
This is a design to be ruled on, not a result, and it is written before any 12B residual has been
captured so that the pre-registration cannot be shaped by looking. Where a choice is the Director's
or the Chief's it is marked **open** rather than settled quietly.

## 0. Sealed, 2026-09-10

`seal.json`, digest **998b3bcafa9d6aaffa21ebd43df3935b1634ba7b12fe187073837acd942ce521**, written at
commit `f2a4265` with `--baseline acc130a`, on the Chief's relay of Codex's delta review
(`6ee2563..4dc39d5`, no further review hold). The baseline is the commit whose text Codex reviewed;
naming it is the claim that all thirteen sealed files are byte-identical to it, and the seal was
created from a later tip precisely so that claim is checked rather than assumed.

The seal fixes the seed (20260910), the fold assignment
(`c5e9622f7bef663c96614a99bab7e120873ad9eb79617912526c02e48016372c`, recomputed by re-running
`folds.py` rather than read from `folds.json`), both of §2's capture-set digests, the five operative
tolerances with every ε recomputed from the bound at M = 12, α = 0.05 and the paired range width, and
the digests of thirteen files. `make_seal.py` produced it and is recorded beside it as `produced_by`
rather than sealed, because it postdates the reviewed text and was not reviewed as part of the study.

Precondition 5's refusal is code: `state_programme.read_gate.require_seal` raises unless the seal is
complete, its own bytes digest as expected, every sealed file still has the bytes it fixed, those
files still match their bytes at the baseline commit, and the sealed document still passes the sealed
checker. Verified against this seal: thirteen files matched, the baseline re-derived, the document
passing. **No capture is read except through it.** The unknown-horizon control (§4.1) stays deferred
with its veto disabled; its amendment is a separate reviewed item and the seal does not release it.

## 0.1 E1 read, 2026-09-10, at the pre-registered headline

Read through `require_seal` against seal `998b3bca…`, fitted on the train split's 5,515 decisions and
evaluated on the 240 clean test episodes, which entered no fit. Rank 8 and the 0.5 depth are the
headline **fixed in §5 before any capture**, so nothing below is a selected rank or a selected layer.
`read_e1.py` produced `e1-readings.json`; `summarise_e1.py` produced `e1-summary.json` and re-fits
nothing.

| comparison | target | probe | over the permutation null | verdict at ε_main = 0.23 |
|---|---|---:|---:|---|
| 12B | step index | 0.789 | **0.646** | resolved |
| 4B | step index | 0.732 | **0.611** | resolved |
| 12B | steps remaining | 0.325 | 0.073 | not resolvable at this n |
| 4B | steps remaining | 0.266 | 0.030 | not resolvable at this n |
| 12B − 4B, paired by episode | step index | | 0.036 | not resolvable at this n |
| 12B − 4B, paired by episode | steps remaining | | 0.043 | not resolvable at this n |

**What this licenses.** At matched rank the residual at mid-depth carries the **step index** on both
models, far above both nulls and above the resolution this corpus supports. That is a statement about
encoding, not about richness, and it is the only one of the four that clears ε.

**What it does not license.** Nothing about **steps remaining**, on either model: 0.073 and 0.030 are
well inside the resolution, so the corpus cannot tell them from zero. And nothing about the
**difference between the models**: 0.036 and 0.043 are likewise inside it. Neither is evidence of no
difference. The document's opening says E1 supports a gap above about 23 accuracy points and nothing
smaller, and these are the readings that opening was written for.

**The full profile, over the permutation null, at the 0.5 depth.** §5 requires it, and it carries one
figure that must not be read as a finding.

| model | target | r=1 | r=2 | r=4 | r=8 | r=16 | r=32 |
|---|---|---:|---:|---:|---:|---:|---:|
| 12B | step index | 0.419 | 0.215 | 0.544 | **0.646** | 0.670 | 0.813 |
| 4B | step index | 0.434 | 0.223 | 0.546 | **0.611** | 0.649 | 0.738 |
| 12B | steps remaining | −0.001 | 0.116 | 0.093 | **0.073** | 0.171 | 0.132 |
| 4B | steps remaining | −0.017 | 0.123 | 0.044 | **0.030** | 0.128 | 0.270 |

**The figure that must not be read as a finding is the 4B's 0.270 at r = 32.** It exceeds ε_main
while the headline at r = 8 does not, and reporting it as "steps remaining is decodable on the 4B"
would be choosing the rank after seeing the answer, which is the thing §5's fixed headline exists to
prevent. It is reported because the profile is required and because hiding it would be worse. If it
is to become a claim it needs its own pre-registration.

**One anomaly, unexplained and flagged rather than smoothed.** Step index falls from r = 1 to r = 2 on
both models (0.419 → 0.215 and 0.434 → 0.223) before climbing. The components are nested and greedy,
so a second component cannot worsen the training fit; it can move held-out predictions across
rounding boundaries, since the score is exact match on an integer. That is a hypothesis and not a
measurement, and it is on the open list rather than in the text as an explanation.

## 0.2 E2 is NOT read: the transport rule I chose cannot pass, and the form is not fixed by §4.2

`read_e2.py` ran and returned a hit rate of **exactly 0.000 in every stratum on both models**. That is
not a reading of E2. A score that cannot come out otherwise measures the instrument, not the corpus
(`METHOD-2026-09-08`, entry 27), and this one cannot come out otherwise.

**Why it cannot.** §4.2 fixes the candidate set as all *N* decisions of the episode **including the
source**. My rule was the literal reading of "predicts +1 on an ordinary transition and the recovery
cost at a corrective one": transport the read state by that many step-units,
`T(x, m) = x + m·d`, with `d` the mean per-step displacement of the fitting folds' ordinary
transitions. With the source among the candidates, such a rule can only win if it moves **more than
halfway** to the successor. It does not come close. Measured on ordinary train transitions only, out
of fold, at the headline depth:

| | 4B | 12B |
|---|---:|---:|
| mean per-transition displacement, ‖x_{k+1} − x_k‖ | 1,548 | — |
| ‖mean displacement‖, which is what `d` is | 313 | — |
| coherent fraction, ‖mean‖ / mean‖·‖ | **0.202** | **0.290** |

A fixed displacement moves the state about a fifth to under a third of the way, and the source sits
at exactly that distance from the transported point while the successor sits at the full distance. The
source therefore wins every time, by construction, in every stratum. Nothing about update closure
follows from it.

**The corpus is not the problem.** The same transitions, scored with the source excluded and no
transport at all — is the true successor simply the nearest other decision of its episode? — give
**0.430 on the 4B and 0.455 on the 12B against a chance level of 0.172**. The residual carries local
order at about two and a half times chance. A rule that actually transports has something to find.

**What this needs, and it is not mine to decide.** §4.2 fixes the candidate set, the tie rule, the
metric and the strata — every choice that could otherwise be made after seeing a score — and does not
fix the transport rule's **functional form**. I chose the form, it was the literal reading, and it
cannot pass. Replacing it after seeing a null result is exactly the move a pre-registration exists to
control, so the replacement is the Chief's ruling and not mine. It is put to them with the numbers
above; until it is ruled, **E2 reads *not measured***, as §10's three states require.

The diagnostic above was run on **ordinary training transitions only**. No corrective transition and
no test episode was scored in it.

## 1. The hypothesis, stated as one

The Director's, in his terms: **the 12B carries a richer representation of the task than the 4B,
possibly a maintained sequence of intermediate-step representations, possibly only under
fine-tuning.**

Three claims are folded together there and they fail differently, so the draft separates them:

| | claim | how it can be false |
|---|---|---|
| H1 | the 12B's residual stream is **more predictive** of task state than the 4B's | both equally predictive once probe capacity is matched |
| H2 | that state is a **maintained sequence** of intermediate steps, not one recomputed each turn | the state is recomputed from the transcript at every decision |
| H3 | H1 or H2 hold **only after fine-tuning** | they hold in the base models, or in neither |

H2 is the one the derivation is built for and the one the programme has never been able to test,
because the agent protocol retains the transcript by construction (`§12`, and the bridge order's
second standing caution). H1 is a confound problem before it is a measurement. H3 is an arm.

## 2. The state variable: plan progress

**File existence** is the first state variable and is ordered
(`STATE-PROGRAMME-RUN-ORDER-2026-09-10`). This is the second.

Every expert trajectory in `pipeline/tasks.py` is a sequence of steps with a known index: the maker
writes `steps`, the runner records one action per turn, and `Task.horizon` is the expert's step
count. So for every episode and every decision position there is a ground truth **plan progress**
pair: the index of the step being taken, and the number of steps remaining. It needs no annotation
and no model to produce it, which is the whole reason to use it: the state is a property of the
task, observable to the experimenter and never told to the model.

That is also its limit, and the draft says so rather than leaving it: the model's own plan need not
be the expert's. An episode that reaches the same answer by a different route has a true step index
under its own plan and a misleading one under the expert's. **Open:** whether the estimands are
computed on all episodes, or only on those whose action sequence matches the expert's prefix. The
second is cleaner and smaller; the first is the population the programme cares about.

## 3. The estimands, the derivation's, on both models over the same episodes

The corpus is one set of episodes rendered once. The 4B and the 12B share a tokenizer and a
template byte for byte (`configs/models/gemma3-12b-cuda-bf16.yaml`), so **the same rendered episode
is both models' input**, and "matched positions" means the same token index in the same episode.

**E1, predictive sufficiency.** Does a feature state read from the residual predict the next action
and the steps remaining, *beyond what the transcript alone predicts*? The baseline is not chance: it
is a predictor given the visible transcript, because a transcript that says "read three files,
two remain" makes the answer readable without any internal state. The estimand is the gain over
that baseline, per model, at matched positions.

**E2, update closure.** As a step completes, does the state move to the successor the task's own
dynamics imply? The derivation's closure condition is that the abstraction commutes with the
transition; here that is: the state read after step *k* completes is the state the model would be
in had it started at step *k+1*. Measured as a distance between the read state and the predicted
successor, against the tolerance the pilot derives, exactly as the first state variable's is.

**E3, retention under carrier controls.** The one that needs the intervention. Replay identical
tokens so the generated-text path is fixed, then reset or exchange the retained activations, and
see whether the state survives. Without that control, a feature that appears "maintained" across
turns is equally explained by the model recomputing it from a transcript that never changed.
**WS-B's cache exchange is the separating control**, and it is unported and refused today
(SWE-1, this morning), so E3 is ordered after it and not before.

**The fine-tuning arm** repeats E1–E3 on a fine-tuned checkpoint, and is H3. It is last: it costs a
training run and it means nothing until the base measurement exists.

## 4. The two rules this draft is required to carry

**Richer means more predictive at matched probe capacity.** The 12B's stream is 3,840 wide and 48
deep against the 4B's 2,560 and 34. A probe fitted on the 12B has more parameters and more layers to
choose from, so it will read more of *anything* — including noise — and an unmatched comparison
measures the architecture rather than the representation. Three ways to match, and the choice is
**open**:

- **fixed rank**: constrain the probe to *r* directions on both, *r* chosen below the 4B's width;
- **fixed parameter count**: give both probes the same number of free parameters, so the 12B's probe
  is proportionally lower rank;
- **matched control features**: the map's own discipline, a control set matched for layer,
  activation prevalence, norm and selection opportunity, per the derivation's §14.

Whichever is chosen, the **same** capacity applies to both models and to the controls, and a result
reported without naming its capacity is refused. This is the same shape as the golden test's
comparability gate: a difference between two things measured differently is not a measurement of
either.

**The pre-registration is written before any 12B capture is read.** Diagnostics, tolerances,
the capacity rule, the episode set and the analysis are sealed first, by the same mechanism the run
order already uses: `preregistration.json`, refusing to start without it and refusing to re-derive
once a main row exists. The captures may be *made* first — they must be, since they are made on the
card in the same pass as the lenses — but they are not looked at.

## 5. The capture plan, so the data exists when the pre-registration does

Residual captures of both models over the same episodes, **at decision positions, every layer**,
made on the card in the same pass as the lens fits.

**Why decision positions and not every position**, in numbers rather than in principle. Per position
a capture is `(num_layers + 1) × hidden × bytes`:

| | layers captured | hidden | per position, bf16 |
|---|---:|---:|---:|
| 4B | 35 | 2,560 | 175.0 KiB |
| 12B | 49 | 3,840 | 367.5 KiB |

At an assumed 300 episodes and 12 decision positions each:

| | decision positions only | every position (≈2,000 tokens/episode) |
|---|---:|---:|
| 4B | **0.60 GiB** | 100 GiB |
| 12B | **1.26 GiB** | 210 GiB |
| both | **1.86 GiB** | 310 GiB |

The card has 236 GiB free. Decision positions fit with about two orders of magnitude to spare;
every position does not fit, and would not fit even alone for the 12B once the checkpoints and the
lens artefacts are on the same disk.

*The right-hand column was 1,024 times larger in the first draft of this table, TiB written where
GiB was meant, which would have made "every position" look absurd rather than merely impossible.
Caught by computing it rather than by reading it back, which is the only way this class of error is
ever caught.*

**The row counts above are assumptions and are marked as such**: the
episode count follows the state programme's pilot and the decisions-per-episode figure is a laptop
observation, so both are re-measured from the first captured episode and the manifest carries the
measured value beside the assumed one.

**Stored in the model's own precision.** bf16 on disk, because the capture must not change the
arithmetic it is capturing — the promoted-float32 finding, at 69.4% on this card at 1,400 tokens, is
the reason that sentence is here rather than assumed. `capture_dtype: native` in every manifest.

**What each capture carries**, so a reader can tell what it is: the checkpoint digest, the episode
id and the step index, the token index, the layer, the decoding mode, the device reading, and the
`basis` field per cell. The ground-truth plan progress is stored beside it and comes from the task,
never from the model.

## 6. What this draft does not do

It does not choose the capacity rule, the episode population, or the tolerances; those are §4's
open items and the pilot's. It does not order the fine-tuning arm. It does not assume E3 can run:
it cannot until the cache exchange is ported, and saying so is the point of listing it.

And it does not claim the 12B is richer. The programme's own record of this week is that the
instrument fails before the model does — a comparison across two arithmetic paths, a check that
could not pass, a gate whose premise was wrong rather than whose port was. **A first result showing
the 12B more predictive than the 4B is, until the capacity rule is applied and the controls run,
a measurement of the 12B being bigger.**

## 7. Where this sits

Cites plan §16.18. The first state variable's order is
`design_specifications/pending/STATE-PROGRAMME-RUN-ORDER-2026-09-10.md`; this second one reuses its
machinery unchanged — the same diagnostics-as-code discipline, the same derivation table, the same
seal, the same `measured / untestable / not measured` three-state reporting — and adds only the
state variable, the capacity rule and the capture plan.

---

## Appendix, D-CRO, 2026-09-09: the corpus count re-run, and four things it adds

`STATE-PLAN-PROGRESS-ORDER-2026-09-10` §1 instructs the owner of the pre-registration to re-run the
Chief's count and put the script in the record. Script: `count_corpus.py`; output:
`corpus-count.json`. It tries to *falsify* each claim and exits nonzero if any differs, so a
disagreement cannot be skimmed past. CPU only, no model, no card.

### Nine of eleven claims reproduce exactly; the other two reconcile, and the reconciliation is the finding

| claim | order | raw | deduplicated |
|---|---:|---:|---:|
| rendered rows | 8,907 | **8,907** | — |
| tasks | 1,128 | **1,128** | — |
| tasks by split | 840 / 48 / 240 | **840 / 48 / 240** | — |
| rows with no `family` | 240 / 60 / 48 | **240 / 60 / 48** | — |
| non-prefix pairs | 4,509 | **4,509** | **4,509** |
| consecutive pairs | 6,501 | 7,431 | **6,501** |
| max decisions per episode | 17 | 20 | **17** |

**`(task_id, step)` is not unique.** 930 of the 8,559 rows that carry a `task_id` share their
`(task_id, step)` with another row. Deduplicated, every figure in the order reproduces to the digit:
7,629 rows, 6,501 consecutive pairs, maximum 17. So the order's basis is the deduplicated one and
mine was raw, and both are now reported, because a count whose basis is unstated is a count nobody
can reproduce.

The non-prefix *fraction* differs with the basis and the order's is right on its own: 4,509 of 6,501
is **69.4%**, and 4,509 of 7,431 is **60.7%**. Fact 3's conclusion is unaffected on either.

### Addition 1: the duplicates are the transient retry, and they say something E2 needs

> **Withdrawn, 2026-09-09, later the same evening. This reading is wrong.** The duplicates
> are not retries; they are byte-identical copies made by `recovery_repeats` in the training
> split. The original text is kept below because the Chief's A1 was ruled on it and a reader
> needs to see what was ruled on. The correction is at the end of this appendix, and the
> pre-registration carries it in §1.1.

A `transient` task inserts a retry of the **same action**, and the corpus labels both rows with the
**same step index** — `train-search-0313-transient` carries step 1 twice, both from the expert. That
is semantically right: a retry is the same plan step re-executed, and plan progress does not advance
across it.

**That sharpens R3 rather than contradicting it.** The order's table has `transient` adding one step,
which is true of the *horizon*. What the corpus adds is that the *step index does not move* at the
retry turn. So the update rule E2 tests must predict **no advance** there, while predicting +1 on an
ordinary turn and the recovery cost at a perturbation. A rule that advances on every turn is refuted
by 244 transient tasks, and that is a stronger informative stratum than the order claims.

### Addition 2: 334 tasks are missing exactly one step, and it is exactly the recovery variants

Only **550 of 1,128** tasks have contiguous step indices. 334 tasks are missing exactly one step
each, and the split by variant is total:

| variant | tasks with a gap | of |
|---|---:|---:|
| `clean` | 0 | 550 |
| `transient` | 0 | 244 |
| `failed_edit` | **58** | 58 |
| `stale_path` | **30** | 30 |
| `unknown_tool` | **64** | 64 |
| `wrong_path` | **182** | 182 |

The mechanism is `runner.trajectory_rows`, which drops a row whose observation begins `ERROR` unless
that step is an *injected fault*: `errored = observation.startswith("ERROR") and step["index"] not in
faults`. `transient` has an injected fault so its error row is kept; the other four earn their errors
and lose the row.

**Fact 1 survives this and should say so.** The dropped step's turn still enters the context, because
`trajectory_rows` appends to `context` whether or not it emits a row, so the number of model turns in
the prompt is still the step index. What is *not* true is the converse: every rendered row is one
decision, but not every decision is a rendered row. Anything that infers a horizon from a row count
will be one short on 334 tasks.

### Addition 3: R6's row addressing needs a third component

> **Superseded by the same correction.** The collisions are real, but they are copies of one
> decision rather than distinct rows, so the key needs *fewer* components, not more:
> `(task_id, step)` addresses a decision uniquely once the copies are recognised. A3's
> occurrence index would have addressed identical inputs. Correction at the end of this
> appendix.

R6 says *"Rows are enumerated through `task_id` and `step`."* With 930 collisions that key does not
address a row. It needs the occurrence index within the task, or a row ordinal assigned at capture
time and carried in the manifest. This is cheap to fix now and unfixable after 8,907 captures are
written under an ambiguous key.

### Addition 4: the rendered corpus is not on the card

`data/agent_v2e-gemma3-4b/` exists on the laptop and **not** under `/workspace/agent-v2-lab/data/`,
which has no such directory. R6's capture pass reads those rows, so this is a blocking input in the
same way the lens corpus was, and it is better named now than at minute fifty of a window. Copying
it is 169 MB and no GPU; it must land **outside** the shared checkout, since `data/` was ignored only
after this corpus was made and the resume key hashes untracked files.

---

## Correction, D-CRO, 2026-09-09, before the pre-registration was drafted

**Additions 1 and 3 above are wrong, and A1 and A3 of the Chief's amendment rest on them.** Found
while building the pre-registration's capture set, by asking what a repeated `(task_id, step)` key
addresses rather than what it means.

Every one of the 930 repeats is a **byte-identical row** — prompt, completion and metadata, checked
by digest over the whole row, zero exceptions. Multiplicity is 2 or 6 and nothing else. The decisions
that carry them are exactly the 578 rows flagged `recovery`, all of them, and no other row in the
corpus. The mechanism is `pipeline/data.py`, in the training split only:

```python
if split_spec.role == "train":
    expert_rows = [row for row in expert_rows for _ in range(_repeats(row, recovery_repeats))]
```

with `recovery_repeats: {transient: 2, wrong_path: 2, unknown_tool: 2, stale_path: 6, failed_edit: 6}`
recorded in the corpus provenance. That reproduces the counts exactly: 490 doubled (transient 244 +
wrong_path 182 + unknown_tool 64) and 88 sextupled (failed_edit 58 + stale_path 30), so
490 + 88×5 = 930. The shuffle that follows scatters the copies, which is why they looked like
separate events and why row order within a task means nothing.

**What I did wrong.** I read `train-search-0313-transient` carrying step 1 twice, saw that a
transient task re-issues an action, and stopped. The two rows are identical; a retry would have made
them differ, because the first attempt's turn would be in the second's context. Comparing them would
have taken one line and would have refuted the reading immediately. The general form is the
twenty-fourth entry's again: **a mechanism inferred from a pattern, when the artefact could have been
asked directly.**

**What changes.**

1. **A1 has nothing to predict.** The step index never repeats as two decisions, so E2's rule needs
   no no-advance clause and the 244 transient tasks are not the stratum that refutes a rule advancing
   every turn. They are still the informative stratum, for a better reason: transient is the only
   variant whose corrective decision is entered *contiguously*, because its fault is injected and its
   error row is kept. The other four lose the failing row, so their 309 recovery transitions span a
   decision that was never rendered.
2. **A3's key should be smaller, not larger.** `(task_id, step)` addresses a decision uniquely.
   The prompt sha256 stays in the manifest, and the multiplicity is carried as `rendered_rows` so
   the training weight is recoverable without being applied by accident.
3. **The capture unit is the distinct decision, 7,629 of them**, not 8,907 rows. Capturing rows would
   forward the same tokens up to six times and would weight any probe fit by the fine-tuning recipe's
   oversampling — silently, and in the direction that flatters a recovery result.
4. **The Chief's §1 figures were right and are unaffected.** 7,629, 6,501 and a maximum of 17 is the
   deduplicated basis, and deduplicating by `(task_id, step)` turns out to be the correct operation —
   for a reason neither of us had at the time.

Scripts: `prereg_inputs.py` builds the capture set and refuses if any repeated key is not a
byte-identical row; `capture_budget.py` recomputes the storage table on the corrected unit;
`check_prereg.py` pins every figure in `PREREGISTRATION.md` to those artefacts and has a
`--self-test` that corrupts each pinned line and asserts the check rejects it.

**One more of mine, small.** I reported the corpus as 169 MB. It is 85 MB. I read the `total 169176`
line of `ls -l`, which counts 512-byte blocks, as kilobytes. The Chief corrected it in A4 from the
archive manifest.
