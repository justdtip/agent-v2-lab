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
