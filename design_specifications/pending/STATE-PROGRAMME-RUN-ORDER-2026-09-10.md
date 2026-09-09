# Work order: the state programme's run script — pilot, tolerances, corpus, estimands

**Chief, on the Director's rulings of 2026-09-10. Owner: the D-CRO, as the seat that holds the
pre-registration; on `cuda-ws-d`, built and tested on fixtures on the laptop, run on the device.
Review by the Chief; the merge into `cuda-migration` is the Chief's.**

The Director ruled today: the acceptance tolerances are found by a pilot **on the remote device**,
and this order is the script that finds them and does the run; **file existence** is the first state
variable; the intervention wrapper is Codex's (WS-A order, second heading dated 2026-09-10). The
temperature refusal is **not yet ruled on**, and this order is written so that nothing in it waits
on that ruling except the one branch that needs it (§2).

The mathematics is the derivation's (`sae_j_lens_state_derivation.md`, as corrected by the bridge
order's third amendment): the bound is Hoeffding's, `n ≥ ln(2M/α)/ε²` over `M` fixed diagnostics at
confidence `1−α`; the estimands are §11's counterfactual substitution and §12's carrier tests. The
count that once said "do not build" was a laptop capture rate; the device's rate is unmeasured, and
this script is what measures it.

## 1. What the script does, in this order

**Stage 0, preflight, written to the manifest before any episode.** Registry name and checkpoint
digest; the lens set (every layer, WS-D's device fit, by lens identity); the dictionary layers by
their config hook strings; the wrapper's version; the decoding mode with temperature, sampler and
seed; the environment's fault rate and seed; the device reading from `device.describe()`. A missing
input stops the script here, by name.

**Stage 1, the pilot.** One hundred episodes per arm, two arms, **matched pairs**: the same task,
prompt, paths, distractors and history length, differing only in whether the target file exists.
The state enters through the tool observation and never through the prompt. Plus one reliability
arm of one hundred episodes at the declared fault rate (§4). The first thing the pilot writes is
`rate.json`: episodes per hour, measured, with the mode; every later time figure is computed from
it and nothing is projected from the laptop.

**Stage 2, the contrast and the tolerances, derived, not chosen.** For each diagnostic, the
difference of arm means with a 95% bootstrap lower bound (ten thousand resamples, seeded; the
Hoeffding interval is wider by construction and is the *main* run's guarantee, not the pilot's
design tool). A diagnostic whose lower bound does not exceed zero is **dropped from `M` with the
reason recorded**: the state does not reach it, and a tolerance for it would be a tolerance for
noise. Nothing is added to `M` after the pilot. Then:

| tolerance | set as | why |
|---|---|---|
| `d_min` | the smallest retained lower bound | the weakest contrast the state produces |
| `ε_sub`, substitution | `d_min / 4` | a passing patch has moved behaviour at least three-quarters of the way to the reference |
| `ε_⊥`, specificity | `2 ε_sub / 3` | a matched-norm control must move it by less than a patch is allowed to miss by |
| `ε_reuse` | `ε_sub` | one patch, every use in the family |
| `ε_pred`, `ε_dyn` | the split-half log-loss gap of the pilot's own fitted belief update | the noise floor of the fit, measured, not assumed |
| `n` | `⌈ln(2M/α)/ε_sub²⌉`, `α = 0.05` | the derivation's bound; at `M = 10`: `ε = 0.05 → 2,397`, `ε = 0.075 → 1,066` |

**Budget rule.** The device time for the main run is `(2n + 2n) / rate` (each context is decoded
once per arm and once each for the patch and its control). If that exceeds the hours the Director
has bought, `ε_sub` is loosened to the value the budget allows and **both numbers and the reason are
written into the pre-registration before the main run**. Never the reverse: the budget is never met
by reading fewer diagnostics or by dropping the control.

**Stage 3, the seal.** `preregistration.json`: the retained diagnostics, every tolerance with its
derivation, `n`, the digests of the pilot rows it was derived from, and the decoding mode. The main
stage **refuses to start without the seal and refuses to re-derive once any main-run row exists**.
The pilot is not a result and its numbers are not reported as findings.

**Stage 4, the main run and the estimands.** `n` matched pairs per arm, resumable on tree content,
checkpoint digest, device reading and seal digest, as the first-hour runbook keys resume. Then, on
the same contexts: **substitution** — the state direction (Stage A's exact component, or the cone
when Stage B justifies it) from the reference arm patched in through the wrapper, passing when every
retained diagnostic's mean moves to within `ε_sub` of the reference arm's; **specificity** — a
control direction of matched norm outside the state cone, passing when no diagnostic moves by more
than `ε_⊥`; **reuse** — the one patch across the three uses; **predictive and dynamic** — the belief
update after a later observation against the fitted update, within `ε_pred`/`ε_dyn`; and the
**relation test** of §4. Each estimand is reported as a distance with its interval, at the level of
J-space vectors, never as a coefficient table (the third amendment's fourth rule).

## 2. The decoding mode, and the branch that waits

The script takes `--decoding greedy|sampled` and records it. Under **greedy**, every context yields
one outcome, so the estimand is the **context-averaged outcome frequency** over the task
distribution, and the record says so in those words. Under **sampled** — temperature 1.0, no
truncation, seeded — it is the derivation's own estimand, the model's distribution over outcomes in
a context. The sampled branch is SWE-1's and is **not built until the Director rules**; the script
must run whole in greedy mode on the day the device is rented, and switch by argument the day the
sampler lands. A record made in one mode is not compared with one made in the other.

## 3. The diagnostics, `M = 10`, fixed here before any pilot row exists

Indicators of what the model does at the first decision after the state-bearing observation, and
what it says at the end; the state is whether the target file exists.

| # | diagnostic | use |
|---|---|---|
| D1 | the next call is `read_file` on the target | decision: read |
| D2 | the next call is `search_files` | decision: look elsewhere |
| D3 | the next call is `list_files` on the target's directory again | cost: re-check |
| D4 | the `finish` answer asserts the file exists | report |
| D5 | the `finish` answer asserts it does not | report |
| D6 | the `finish` answer names the target path | report: specificity of content |
| D7 | D1 after a later, contradicting observation | dynamic: decision |
| D8 | D4 after a later, contradicting observation | dynamic: report |
| D9 | D2 after a later, contradicting observation | dynamic: cost |
| D10 | with two files of swapped states, the action on file B is the one B's state implies, not A's | relation |

A diagnostic is a fixed function of the transcript, written as code with a test before the pilot,
and is never re-scored by a model.

## 4. The task family and the environment's instrument

**The family.** A new maker in `pipeline/tasks.py` beside `_list`: a target file under a directory,
distractors of the same kind, and a prompt that asks the model to list the directory, read the
summary there if it is present, and report its first line or that there is none. Arm E has the file;
arm A has the identical workspace without it. The prompt is the same string in both arms. The maker
is device-free and is accepted with a test on the laptop.

**The reliability instrument.** The environment already applies injected faults before validation
and execution: `pipeline/env.py`, `execute`, reads the task's `faults` tuple by call index and
returns the fault's message as the observation. A false observation is therefore a `Fault` whose
message is a **well-formed but false** tool result — a listing that omits the file, or a
`file not found` for a file that exists — at a declared rate with a seed, both in the manifest. That
rate is the likelihood ratio the derivation's door model needs, and the belief update's predicted
dependence on it is what `ε_dyn` tests. The former false-empty listing was this instrument
uncontrolled; the fix at `020aa89` made it honest, and this order puts it back under control by
declaration.

**The relation test.** Two files in one directory with swapped states across the pair of episodes;
D10 above.

**Nuisance variables**, balanced by construction and recorded per episode: task identity, path
strings, history length, position of the state-bearing observation.

## 5. What is built on the laptop, and what is not

Built and tested on fixtures, no model load: the maker and its test; the ten diagnostics as code
with tests on hand-written transcripts; the tolerance derivation with the closed-form checks above
(`2,397` and `1,066`), the drop rule, the budget rule writing both numbers; the seal and both
refusals; the resume; the whole script end to end with a scripted policy, the real environment, a
fixture lens and dictionary, and a stub wrapper, on ten episodes, writing every artefact §6 names.

Not on the laptop: any episode from a model; any rate; any tolerance value. The runbook's rule
stands: a laptop figure is a basis for nothing here.

## 6. The record

`research/records/STATE-PROGRAMME-<date>/`: `manifest.json` (stage 0), `rate.json`,
`pilot/rows.jsonl`, `preregistration.json` (sealed), `main/rows.jsonl`, `estimands.json`, and a
`README.md` written from those files and nothing else, with every number beside its basis.

## 7. Rules of this order

No tolerance is typed by hand; each is a line of the derivation table with its inputs. The pilot is
not read for findings. Main-run rows are not read before the seal exists. The specificity control is
matched-norm, and a control that is easier to pass than the patch is a defect. The two cautions the
bridge order keeps — averaging can cancel a causal sensitivity; retained transcripts confound mutual
information — apply to every estimand here. A greedy record and a sampled record are two records.

## 8. Acceptance

The script, the maker, the diagnostics and their tests on `cuda-ws-d`; the fixture run's record
directory committed under `research/records/` with its README; the Chief's review; then the merge.
The device run is scheduled in the first-hour runbook's successor, after the lenses and the
dictionary layers exist on the device.

---

## Amendment, D-CRO, 2026-09-10: what was built on the laptop, and five decisions the order left to the builder

Built on `cuda-ws-d` as `local_llm_lab.pipeline.state_programme` (family, diagnostics, tolerances,
seal, run, record), `scripts/state_programme.py`, five test files, and the fixture record
`research/records/STATE-PROGRAMME-FIXTURE-2026-09-10/`. Each decision below departs from the order's
wording or fills a gap in it, and each is stated so it can be overturned.

**1. The maker is beside `_list` in `_MAKERS` and is not in `FAMILIES`.** `make_tasks` assigns a
family by `index % len(FAMILIES)` and derives every task id and R28 fingerprint from that position;
the J-space split's own comment records that `ledger_reconcile` sits at index 7 of twelve so that
naming a split "changes no task the recorded sweep drew from". A thirteenth entry would move every
task after it in every recorded corpus. So the family is reached through
`family.make_existence_pairs`, the way the P2 and J-space splits are reached through their own
factories, and a test pins `len(FAMILIES) == 12` beside the registration.

**2. The wrapper on the laptop is a stub with the wrapper's shape and none of its content.** A
"substitution" or "reuse" patch decodes the pair's *exists* arm; a "control" decodes the absent arm
unchanged. So on fixtures substitution and specificity pass by construction — which is what a stub
can show, that the estimand code reads rows correctly, and nothing about a model. Every row it writes
says `wrapper: stub`, and the manifest's `wrapper_version` is `stub`. The device passes Codex's
intervention (WS-A order, 2026-09-10) through the same `Wrapper` type; no distance changes.

**3. `ε_pred`/`ε_dyn` is defined.** The order says "the split-half log-loss gap of the pilot's own
fitted belief update" without saying which update. It is a Bernoulli on D4 (the report that the file
exists) fitted on each half of the reliability arm and scored on the other, the gap between the two
halves' log losses. On a scripted policy it is exactly zero, as it should be for a policy that never
varies, and the fixture record shows that zero.

**4. A diagnostic that never reaches its position scores `None`, not zero,** and a `None` is
excluded from that diagnostic's mean and counted. Scoring it zero would dilute every unreached
diagnostic toward "no contrast" and the drop rule would then be dropping dilution rather than
absence. This is the convention the drop rule's "not reached" reason reads.

**5. The tree digest goes through `spawn.run`.** The resume key hashes the working tree's content
including untracked files (`git ls-files --cached --others`), and the first draft called
`subprocess.run`, which the suite's fork guard refused: a fork in an interpreter with Metal up aborts
in libplatform without raising (R45). The guard caught it before a device hour did.

**What the fixture record shows, and what it does not.** Ten pilot pairs, a reliability arm at 0.3,
five main pairs, greedy mode, a scripted expert. The drop rule fired on six of nine: D2, D3, D6
have zero contrast under an expert that never searches, re-lists or names the path; D7–D9 are
unreached because the expert never sees a contradiction it did not itself cause. D1, D4, D5 are
retained with `d_min = 1.0` — the expert's contrast is exact — so `ε_sub = 0.25` and `n = 77` at
`M = 3`, which is the bound doing arithmetic on a fixture and not a tolerance anyone will use.
The rate is 46.5 million episodes per hour, because the policy decodes nothing. **That number is in
the record on purpose:** it is the clearest possible statement that a laptop rate is a basis for
nothing here, and the device's `rate.json` replaces it on the first run.

**Two things the order should carry that this build surfaced.** D7–D9 need a *contradicting*
observation, which under the reliability instrument happens only when a false first listing is
followed by a truthful action on the target; at fault rate `r` that reaches roughly `r` of the
exists arm, so the pilot's hundred pairs give the dynamic diagnostics about thirty scorable
episodes per arm at `r = 0.3`, and the drop rule will be the judge of whether that is enough. And
D10, the relation test, is implemented as a function over a swapped-state pair but is not yet
generated by the maker; it needs a two-file variant of the family, which is a second maker and is
left as the next task rather than folded into this one unreviewed.

**Addendum, 2026-09-10, on the merged wrapper (`cuda-migration` e8dbcc3).** The device driver that
satisfies the `Policy` seam inherits two obligations from `SAEIntervention` and `TorchCapture`,
verified against `sae_intervention.py` and `torch_capture.py`: it must declare `emitted_positions`
— the absolute positions of forwarded generated tokens — on every forward, because the capture
cannot see sampling and an "emitted" clamp fails closed without them; and it must name which form of
the per-application off-target vector it persists, the full dictionary-width vector or the summary
Codex is building (count, max, L1, top eight), since a clamp across an episode at full width holds
16k floats per emitted token. The fixture driver has neither obligation because the stub applies no
clamp; both are the device driver's and are listed here so they are built with it, not found by it.

## Review, Chief, 2026-09-10 — `001788a` on `cuda-ws-d`: two edits before the merge

**Read in full** against the merge result with the integration tip: the six modules, the script,
the five test files, the fixture record and the amendment above. Full suite on that merge 2,606
passed, 10 skipped at nine sites, all for untracked data and saved evaluations absent from a
scratch worktree, under the Chief's window at 06:02Z; the torch set 163 passed; no deletion on
either side since the base. The merge object was built and is **not pushed**; it is rebuilt after
the edits below land.

**What is right and stays.** All five of the builder's decisions stand: the maker beside `_list`
outside the family cycle, with the twelve pinned; the stub wrapper that says it is a stub in every
row; `None` for an unreached position, which is what makes the drop rule read absence rather than
dilution; the tree digest through `spawn.run`; and the definition of `ε_pred` as a Bernoulli on D4
across halves of the reliability arm. The seal's two refusals are functions that raise, not flags.
The resume refusal names every field that differs. The README is written from the files. The rate
of 46.5 million episodes per hour is in the record for the right reason. The `4n` budget rule,
the drop rule on the side the contrast lies, the closed-form checks, and the bootstrap seeded on
both sides are the derivation table as ordered.

**Edit 1, required: the absent arm's false listing names the wrong file.** `false_listing(task,
arm="A")` recovers a target from the expert's note text and lists `summary-00.md`, while every
diagnostic for that episode reads `pair.target`, which is `summary-NN.md` with `NN` never `00`. So
in the reliability arm's absent half the model is told a different file exists, the scorer sees a
listing without the target and reads it as truthful, and the falsification is invisible to D4 and
to the split-half gap that defines `ε_pred`. Half the instrument is scoring nothing. The pair holds
the target; `apply_reliability` has the pair; pass the target through and delete the note parsing.
The test asserts `pair.target in false_a`, and that a falsified absent-arm row's state observation
reads as *present* to the diagnostics.

**Edit 2, required: an estimand that cannot fail.** The main run has no reliability arm, so no
main-run row carries a contradiction, D7–D9 are `None` throughout, and `estimands()` reports
`predictive` as distance 0.0 against tolerance 0.0, `passes: true` — computed from no rows. That is
the twenty-seventh method entry's family with the sign flipped. Three parts: the main run includes
the reliability arm at the manifest's rate in both arms, because the predictive and dynamic
estimands exist only after a contradiction; `estimands()` reports an estimand with no scorable row
as `measured: false` and never as a pass, and reports `predictive` and `dynamic` as the seal
defines them — the belief update in falsified episodes against the pilot's fitted Bernoulli as a
log-loss gap for `ε_pred`, and D7–D9 after the contradiction against `ε_dyn`; and a tolerance of
exactly zero is recorded as *degenerate: the pilot's update never varied* with the estimand marked
untestable at zero, not passed at zero. The fixture record is regenerated after this, and its
predictive row must then say untestable.

**Two more, not blocking the merge.** The device driver's obligations in the addendum gain a
third: `device.pin` precedes preflight, and a non-fixture preflight refuses a `describe()` reading
of `UNPINNED`. And `read_rows` on a resume meets a partial trailing line after a crash mid-append;
refusing by line number beats a bare decode error. Both can land with the edits or after.

D10's two-file maker stays the next task, as the amendment says. After the two edits: pathspec
commit on `cuda-ws-d`, and the merge is rebuilt and tested again before it is pushed.

## The temperature, ruled 2026-09-10

The Director accepted the recommendation of §2 as written: sampled decoding for the state
programme only, temperature 1.0 with no truncation, seeded and recorded. SWE-1 builds the sampler
under the WS-B order's amendment of this date. The script's refusal of `--decoding sampled` stays
until that sampler is on the branch the script runs from; then the sampled branch of §2 is built on
the policy seam with the sampler behind it, the estimand named per mode as the script already does.
The two edits of the review above come first.

**Both edits landed, D-CRO, 2026-09-10, with one change to a decision the review accepted.**

Edit 1: `false_listing` and `with_false_observation` take the pair's `target`; the note parsing is
deleted; the test asserts `pair.target in false_listing(..., arm="A")` and that a falsified
absent-arm row's state observation reads as present to the diagnostics, with D4 scorable there.

Edit 2: the main run carries the reliability arm at the manifest's rate in both arms, as rows
`RE`/`RA` with `falsified` on every base row; `estimands()` reports five estimands, each in one of
three states — `measured: false` with no scorable row, `untestable: true` at a tolerance of exactly
zero with the reason *degenerate: the pilot's update never varied*, or a pass/fail — and only the
third can pass. `dynamic` is D7–D9 after the contradiction, contrasted across `RE` and `RA`.

**The change: ε_pred's belief update is fitted per arm, not pooled.** The review accepted decision
3 as "a Bernoulli on D4 across halves of the reliability arm". Regenerating the record showed why
that is the wrong statistic: pooled across `RE` and `RA`, D4 among falsified episodes read 0.43
under a scripted expert that never varies, because the two arms' truthful answers differ — told
absent when present the expert still reads and reports the file, told present when absent it
finishes "no summary" — and the pooled Bernoulli was measuring the arm mix and calling it the
update. Its split-half gap then varied with the draw, so ε_pred was the sampling noise of a
composition and the predictive row read as a measured pass at 0.019 against 0.059 on a policy
that had not updated anything. Per arm, the expert is constant, the gap is exactly zero, the
tolerance is degenerate and the row says **untestable**, as the review required; with a model it
varies for the right reason. `split_half_log_loss_gap` takes the larger arm's gap, and the
predictive distance is the larger arm's log-loss gap between the pilot's fitted Bernoulli and the
main run's falsified episodes. If the Chief prefers the pooled form, the pooled record above is
the argument against it.

The non-blocking two also landed: a non-fixture preflight refuses a `describe()` reading of
`UNPINNED` (the driver's third obligation, `device.pin` before preflight), and `read_rows` refuses
a partial trailing line by number.

Regenerated fixture record: predictive **untestable (degenerate tolerance)**; dynamic **not measured
(no scorable row)**, because only the exists arm meets a contradiction under a scripted expert;
substitution, specificity and reuse pass by construction against the stub, as before. 46 state
tests; the full suite exits 0 on the branch.

## Review of the edits, Chief, 2026-09-10 — `860c23e`, merged into `cuda-migration` at `0870f29`

**Verdict: both required edits and both non-blocking ones pass; merged.** Against the merge
result: full suite 2,622 passed, 10 skipped at nine sites for absent worktree artefacts, under the
Chief's window at 06:33Z; the torch and state sets 105; the fixture script run end to end on the
merge; the resolved order file free of conflict markers; no deletion on either side since the base.

Edit 1: `false_listing` and `with_false_observation` take the pair's target, the note parsing is
gone, and the tests assert the target in the absent arm's false listing and that a falsified
absent-arm row reads as present to the diagnostics with D4 scorable. Edit 2: the main run carries
the reliability arm at the manifest's rate in both arms; each estimand is in one of three states
and only the third can pass; dynamic is D7–D9 after the contradiction across the two reliability
halves; the regenerated record reads *untestable* for predictive and *not measured* for dynamic,
which is what a scripted expert should produce. The non-fixture preflight refuses `UNPINNED`,
against the literal `pinned` the shim reports, verified. `read_rows` refuses a partial line by
number.

**The per-arm Bernoulli is accepted, and the pooled form is withdrawn.** The builder's argument is
the record: pooled across the two reliability arms, D4 among falsified episodes read 0.43 under an
expert that never varies, because the arms' truthful answers differ, and the split-half gap then
measured the composition of a draw and passed a predictive row on a policy that had updated
nothing. Per arm the expert is constant, the gap is exactly zero, and the row says untestable for
the right reason. This is the kind of correction the review asked for and did not see itself.

**The near-miss stays on the record.** A chained shell committed past a Python check that had
raised, producing a local merge commit with conflict markers in it; caught before the push, reset
and redone with the write gated on the check. The D-CRO writes it as the twenty-ninth method
entry in their own name, since the shape is the week's: a check that raised and a commit that did
not wait for it.

**Next, D-CRO:** D10's two-file maker, the relation test generated rather than only implemented,
with its diagnostic scored over swapped-state pairs in both the pilot and the main run; then, once
SWE-1's sampler is on `cuda-migration`, the sampled branch of §2 on the policy seam, the refusal
lifted by name and the estimand named per mode as now. The device driver's three obligations stand
as listed in the addendum.
