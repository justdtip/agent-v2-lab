# The three diagnostic episodes, run fairly: two loops gone, one worse, no passes

**2026-09-08 09:41Z to 10:14Z. Gemma 3 4B at 4-bit, eight layers, seed 20260902, on a box nobody
else was using. Stage one is the control and was not touched.**

## What changed, and why this run cannot separate the two changes

Two corrections landed together and both are independently right, so neither was held back to keep
a comparison clean:

- **The step ceiling went from 12 to 24.** Twenty-four is `run_task`'s own default and the height
  that produced every Qwen number this is read against. Four of Gemma's eleven stage-one failures
  were exhaustion at a ceiling half its comparator's.
- **The observation now renders as the workspace answering the model's own call.** The wrapper
  names the tool, and the system prompt carries a convention paragraph saying that a turn the
  format marks as the user's may be the workspace reporting the result of the model's last action.

So this run answers *can Gemma do these tasks when the harness and the rendering are fair to it*.
It does **not** answer which of the two errors mattered. Attribution is a cheap follow-up.

## Outcomes

| episode | steps | distinct calls | longest run of one call | loop detected | pass |
|---|---|---|---|---|---|
| `batch_update-0166` | 12 → 7 | 4 → 7 | 8 → 1 | yes → no | no → no |
| `ledger_reconcile-0163` | 12 → 9 | 5 → 8 | 8 → 2 | yes → no | no → no |
| `update-0028` | 12 → 24 | 4 → 2 | 5 → 23 | no → yes | no → no |

**Still nought for three.** And the failure shapes moved a long way, in both directions.

## Two of three lost their loops completely

`batch_update-0166` read one file eight times in stage one and tripped the loop detector. Now it
reads the manifest, performs the first replacement, reads the file back to verify it, attempts the
other three workers, receives three `old text not found` errors, and finishes. No repetition
anywhere. It fails by answering `updated-and-verified=4` when one update succeeded and three
errored — which breaks its own system prompt's rule about never claiming a write succeeded until a
tool result confirms it.

`ledger_reconcile-0163` issued `replace_text` with an empty `old` eight times in stage one and
tripped the detector. Now it makes that same error **once**, and then recovers: it reads
`summary.txt`, finds `approved_total=PENDING`, replaces `PENDING`, re-reads to confirm
`approved_total=277`, and finishes. The recovery is exactly the behaviour the system prompt asks
for and stage one never produced. It fails on the content: the expert reads six invoice files and
sums four approved amounts to 498, and the model read two, summed 155 + 122, and reported `277`
rather than the `approved_total=277` line the prompt asked for.

**These two are what the Director's account predicts.** The model stopped behaving as though a
person were answering it and started executing a plan — and then failed at the plan, which is a
different and more tractable failure than not being able to proceed at all.

## The third got worse, and the reason turned out to be ours

`update-0028`'s prompt gives the path verbatim: *In `workspace/test/0028/config.ini`, change mode
from fast to audit.* Four expert steps: read, replace, re-read, finish. The corrected run reads
`/test/0028/config.ini` instead — a dropped component and an added slash — and then issues that
identical call twenty-three times running.

**This section said the model regenerates paths rather than copying them. That is refuted and the
sentence is struck.** Per-call path fidelity is indistinguishable between the two models: Gemma
54/67 at 0.806, Qwen base 1,127/1,400 at 0.805. Only the error *shape* differs, and every leading
slash in the whole Gemma corpus belongs to this one task.

**What actually happened is in the model's own context at turn 2**, and the environment is the
author of it:

> **[system]** Rules: Inspect state with tools instead of guessing paths, file names, or values.
> **[user]** In `workspace/test/0028/config.ini`, change mode from fast to audit.
> **[assistant]** Progress note: Need to inspect the configuration file to see its current mode.
> `{"name": "list_files", "arguments": {"directory": "/"}}`
> **[tool]** `FILES: (none)`
> **[assistant]** Progress note: The file list is empty, so there is no configuration file to
> inspect. I need to create a configuration file to test.
> `{"name": "read_file", "arguments": {"path": "/test/0028/config.ini"}}`

`list_files` answered an unsatisfiable directory with an empty listing rather than an error, so the
model was told the workspace was empty. It had been instructed to inspect rather than guess. It
inspected. **The premise it reasoned from was false and we supplied it**, and this repository then
recorded the resulting loop as the model's failure. The Chief's fix is at `020aa89`: both
simulators now raise `directory not found`, echoing the caller's own string.

Everything downstream is a consequence of that turn, not of a copying defect.

Its notes make the failure legible in a way an outcome cannot. From the third turn on, every note
says a variant of *"The file still doesn't exist. I need to create it"* — and the action attached
to that note is `read_file`, twenty-three times. There is no tool in this workspace that creates a
file.

**That note is not a confabulation, and calling it one would have been the wrong finding.** The
observation it follows reads `ERROR: file not found: /test/0028/config.ini`, so *"the file still
doesn't exist"* is an accurate reading of what the model was handed, and *"I need to create it"* is
a sound inference from it. The premise is false, not the reasoning. The CRO raised this distinction
and it is settled here from the transcript rather than left open: the model reads its instrument
correctly and reasons correctly from a corrupted path.

**What remains after that correction is the divergence itself.** The stated plan is to create; the
emitted call is a read; and the two do not implement each other, twenty-three times running.

**And the corrupted path was never out of sight.** The literal `workspace/test/0028/config.ini`
is present in the rendered prompt at every turn, including the last at 2,701 tokens — the task
statement is in the opening turn and windowing never removes it. So this is not a memory failure
and not an artefact of hiding old observations. The string the model needed was in front of it on
every one of the twenty-four decisions, and a different string came out.

## What this hands the map

The map's primary comparison is calls against notes. This run supplies a case where the two
demonstrably disagree in plain text — a stated intention to create, an emitted call to read — and
another where a literal string present in the context is not the string that gets emitted. Both
are well-posed questions about what the residual carries at the decision, and neither needed the
map to be noticed, which is the right order.

**A hypothesis, not a finding:** the corrected rendering lengthens every prompt, and `update-0028`
is the episode whose critical literal sits furthest from the decision by the end. Whether path
corruption worsens with distance is testable and untested.

## Timings

1,341.9 s, 253.6 s and 392.8 s, 1,992.7 s in total, on a box with no other work on it. Peak 3.8 to
4.1 GiB. These are usable, unlike the superseded run's, and the reason that matters is R61(b):
contention moves seconds and cannot move a verdict.

---

## The fixed point, read off the records we already hold

**Added after the CRO reframed the path failure. No model, no box: this is the corpus stage one
and the re-run already wrote, read with `fixed_point.py`.**

The wrong path is not twenty-three decisions. It is **one token**, taken twenty-three times. The
correct path needs `' "'` after `"path":` and then `workspace`; the model emits the single token
`' "/'`, and every character after it is forced by that choice. So the whole error lives at the
position before that token, and the record carries what each read layer had there.

| | |
|---|---|
| forks in the episode | 24 |
| where the emitted wrong token is already rank 1 | layer 24, at all 24 |
| its rank one layer earlier, at 23 | 14 to 20 (twice lower: 6 and 2) |
| forks where the correct `' "'` is in any layer's top-10 | 24, and only at layer 34 |
| forks where `workspace` is in any layer's top-10 | 0 |

**The control, without which the first row means nothing.** Across the 1,149 other emitted tokens
in the same episode, the layer at which the emitted token first reaches rank 1 is spread: 9 by
layer 18, 183 at 23, 296 at 24, 408 at 30, 253 not until the identity at 34. Layer 24's median
rank for an ordinary token is 2. So "rank 1 by layer 24" is where about a quarter of tokens land
and is unremarkable on its own — what is not unremarkable is that **every one of the 24 forks
commits in the same 23-to-24 step, none earlier and none later**, while ordinary tokens scatter
across five layers.

**And "beaten" understates it by orders of magnitude, which the record said before the model's own
probabilities were read.** The final layer stores the model's own softmax, so the fork's confidence
is already in these records:

| | P(wrong opener) |
|---|---:|
| lowest of the 24 forks (turn 1) | 0.9890071 |
| turn 0 | 0.9998727 |
| at or above 0.9999969 | **22 of 24 forks** |
| turn 23 | 1.0000000, saturating float32 |

**Twenty-three unambiguous refutations do not move the distribution.** "The model does not switch
strategy under repeated unambiguous failure" was compatible with a flat distribution, a
self-reinforcing one, or one moving too slowly to matter. It is none of those: the model is not
uncertain and choosing wrongly. It is certain, and the errors never reach its certainty. At the
late forks the correct opener sits behind by up to seven orders of magnitude, which is not a near
miss and must not be read as one.

**It is saturation and not a trend, and the difference matters.** Turn 2 is 0.9999998 and turn 3 is
0.9999969, so the series is not monotone and no hardening slope exists to fit. The Chief caught
that in an agent's write-up before it entered a record. The claim is stronger without the slope,
because a slope would have invited a test and a confound.

**Consequence for any intervention, decided before the compute rather than after.** An experiment
on this fixed point must **stratify by the model's own confidence**. Fork 1 at 0.989 is the only
soft target in the episode; forks 15 to 23 are saturated. Flipping fork 1 and reporting success
would measure the fork where flipping was cheap.

**The word is never a candidate; the punctuation that would have allowed it is, but only at the
end.** `workspace` is absent from every read layer's top-10 at every one of the 24 forks — 192
top-10 lists and it is in none of them. The `' "'` that would have opened a bare path is a
different story and an earlier version of this record got it wrong: it is in the top-10 at **all
24 forks, and only at layer 34**, where it sits directly behind the token that wins. Layer 34 is
the identity, so that is the model's own output distribution: the correct opener is the runner-up
in the model's own final decision, and it is not a candidate at any depth below.

So the shape is not "the right branch was never live". It is narrower and stranger. At the moment
of choosing, the alternative that keeps the path correct is present and beaten; the word the path
actually needed is nowhere at all, at any depth. Whatever the residual carries about this path, it
does not contain `workspace` in any form the readout can see.

**Two limits, both real.** The top-10 is a narrow window and absence from it bounds the
alternative's probability rather than proving it absent. And these twenty-four are one decision
repeated, not twenty-four independent samples: it is one fixed point observed many times, which is
what makes it a good target and also what stops it being a population.

**What it hands stage two.** A known position, a known wrong token, a known correct alternative
that never appears, and a commitment localised to a single layer step — in an episode already
recorded. Whatever stage two measures at 23 against 24, this is the case where the answer is
already legible in the readout and can be checked against it.

### The obvious next question, asked and answered no

Does a path the model gets **wrong** commit earlier than one it gets right? Every path-slot
decision in both runs, across all fourteen episode-runs, classified by whether the resulting call
returned `not found`, and scored by the layer at which the emitted token first reaches rank 1:

| | n | median commitment layer | committed by layer 24 |
|---|---:|---:|---:|
| path resolved | 73 | 30 | 7% |
| path not found | 34 | 24 | 85% |

That looks like a finding and it is not one. **Twenty-nine of the thirty-four unresolved decisions
are `update-0028`, in the two runs, which is the fixed point counted twenty-nine times.** The
remaining five are three from `batch_update-0166`, committing at layer 30, and two singletons from
stage one at 30 and 34 — none of which behaves like the fixed point at all.

Deduplicated to one entry per distinct path-slot token per episode, so a repeated decision counts
once:

| | n | median commitment layer | committed by layer 24 |
|---|---:|---:|---:|
| path resolved | 13 | 30 | 15% |
| path not found | 5 | 30 | 40% |

**The difference disappears.** At that n there is nothing to test and no claim to make. The
aggregate table above is what the pooled version of the caveat looks like when the caveat is
ignored, and it is kept here so that the next reader meets it already refuted rather than
rediscovering it as a result.

So the early commitment at layer 24 is a property of **this fixed point**, not of wrong paths.
Whether it generalises is a question for a corpus with more than one of them, which stage two
would supply and this pair of runs does not.


### A correction to this record's own method, and what it cost

The first version of the table above said the correct opener appeared at 2 of 24 forks. It appears
at all 24. The error was not in the reading of the result but in the join that produced it, and it
is worth writing down because the failure is silent and the corpus invites it.

**The record has two coordinate systems.** A `rank` row's position counts the turn's own stream, so
the row at `p - 1` scores the token emitted at `p`, and it carries that token's id — which makes
the alignment checkable rather than assumed. A `reading` row's position is the *forward's* offset,
and with no cache across turns every turn re-encodes from zero, so **the same position occurs once
per turn**. Keyed in one flat dictionary, later turns silently overwrite earlier ones and every
lookup answers about the last turn instead of the one asked about.

Every rank-based number in this record was checked against the rows' own token ids: 8,207 emitted
tokens across every episode of both runs, zero mismatches. So the commitment layers, the
argument-start baseline and the identity check are unaffected. The reading-based presence counts
were not, and are now recomputed with readings grouped by turn.

**The check that catches it is the identity, again.** At layer 34 the readout is the model's own
distribution, so its top-1 must be the emitted token. Grouped by turn it is, for 100% of 1,173
emissions. Keyed flat it was 4%. A join this wrong announces itself the moment a boundary
condition is asked of it, which is the argument for having one.
### Re-read after the confound register: the false premise was ours

**Amended 2026-09-08 after the Chief's audit found the cause. The reading above stands and its
subject changes.**

`list_files` answered an unsatisfiable directory with `FILES: (none)` rather than an error.
Workspace paths carry no leading slash, so every rooted guess matched nothing and the model was
told the workspace was empty. `read_file` errors honestly; `list_files` did not. And there is no
discoverable root: no tool lists the workspace top level, so the exact string `workspace` had to be
guessed and every near miss was answered with a confident falsehood.

**The corrected run's `update-0028` opens with `list_files("/")` and is told the workspace is
empty.** Stage one's opens with a failed read and then `list_files("/workspace/test/0028")`, and is
told the same. Everything downstream — *"there is no configuration file"*, *"I need to create it"*,
the twenty-three repetitions — follows from an observation this repository manufactured.

So the sentence *"Gemma does not copy a literal path, it regenerates one"* is not supported by this
episode. The leading slash is **consistent with the belief the environment installed**: the model
had been told the root was `/` and that it was empty. What looked like a copying defect is a
correct inference from a false premise, one level further out than the previous amendment placed
it. The Chief has fixed both simulators so that `list_files` raises, echoing the caller's own
string.

**The fork finding is not weakened by this and is better posed because of it.** The measurements
are unchanged — the commitment at layer 24, the argument-start baseline it is read against, the
absence of `workspace` from all 192 top-10 lists. What changes is the subject.

It is no longer *the model cannot re-emit a string*. It is: **a false premise installed by the
environment at turn 0 still fully determines the representation twenty-three turns later, and the
alternative it ruled out never returns to candidacy at any read depth.** That is a question about
how long a belief persists and how completely it closes off what it excludes, and it is a better
interpretability target than the one it replaces because every part of it is identified — the
premise, the turn it entered, the moment of commitment, and the absent alternative.

**And it may explain the one row that was always the strongest, though this part is an
interpretation and not a measurement.** The Chief's reading: `list_files("/")` returned without an
error, which tells the model that `/` is a real directory here, so the prompt's `workspace/` reads
as the name of the sandbox rather than as a path component, and the file becomes
`/test/0028/config.ini`. On that account `workspace` is absent from all 192 top-10 lists **because
the model had already ruled it out as a directory name, correctly, on our authority** — not
because it could not recall the string.

That is coherent and it is a claim about the model's reasoning inferred from its outputs, which the
readout does not measure. What the readout establishes is the absence itself. The
explanation is the Chief's and is recorded as one, because the difference between "the word was not
available" and "the word had been excluded" is exactly the kind of thing a map should be asked to
settle rather than told.

---

## The environment fix does not fix the episode, and the causal story is refuted

**2026-09-08 11:05Z. One episode, no lens and no capture, fixed simulator, corrected rendering, 24
steps, greedy, seed 20260902. 65.7 s. `environment_falsification.py`, result in
`falsification.json`. Run because the Chief's account made a prediction that could fail.**

The account was: `list_files("/")` returned a false empty, the model concluded the workspace was
empty, and everything downstream followed from that. The prediction was that with `list_files`
raising instead, the belief never forms and the episode behaves differently.

**It behaves differently and fails identically, and the loop is tighter than before.**

| | corrected rendering, lying simulator | corrected rendering, fixed simulator |
|---|---|---|
| turns | 24 | 24 |
| distinct calls | 2 | 2 |
| loop detected | yes | yes |
| passed | no | no |
| shape | `read_file` on one wrong path, 23 times | `list_files("/")` and `list_files(".")` alternating, 24 times |

Every turn is now an error. The model never reaches a read at all, never tries `search_files` —
which returns the true path — and never emits the literal the prompt gave it. It alternates two
root guesses until the ceiling.

**So the specific chain is wrong.** The false empty was a lie and removing it was right, but it was
not what caused the failure. Told the truth, the model fails sooner and more completely.

**I then drew the wrong conclusion from this and it is corrected here rather than below.** I wrote
that the environment has no discoverable root, and urged that it be promoted to the register's
headline. It is false. No tool lists the workspace top level, which is what I checked, but
`search_files` reaches the full true path from any of several words in the task's own prompt:

| call | result |
|---|---|
| `search_files("config")` | `MATCHES: workspace/test/0028/config.ini` |
| `search_files("0028")` | `MATCHES: workspace/test/0028/config.ini` |
| `search_files("mode")` | `MATCHES: workspace/test/0028/config.ini` |
| `search_files("workspace")` | `MATCHES: workspace/test/0028/config.ini` |

Swept over the difficulty-2 test split, **165 of 180 tasks are reachable by a single search of a
word taken from their own prompt**, 15 of 15 in every family except `calculate`, which has no files
and needs no path. Reproduced independently by the Chief and here. And the route is named in the
very rule the model was obeying: *if a tool returns an error, read it and recover: list or search
to find the right path.*

**So the finding is not that the model cannot find the workspace. It is that the model does not
switch strategy under repeated unambiguous failure.** Twenty-four honest errors on two guesses,
with the escape route in its own instructions and its own tool list, and it alternates the two
rejected calls to the ceiling. That is a statement about the agent, measured under an environment
that no longer lies to it — which is the thing that could not be said this morning.

**Consequences.**

The register's causal paragraph needs revising. My suggestion that its "no discoverable root"
question be promoted to the headline was wrong and is withdrawn above; promoting it would have
promoted something false.

The fixed point's frozen records are untouched and remain what they were. This run deliberately
writes its own directory and reads nothing from theirs. What it does change is the claim they
support: the false premise is no longer the identified cause, so the belief-persistence framing
above must be read as being about **a** false premise the environment installed, with the question
of which one now open.

And a note on the metric this run exposed. `longest_identical_run` is 1 here, because the loop is a
two-cycle rather than a repetition. A loop measure that only counts consecutive identical calls
misses an alternation, and the earlier tables in this record use exactly that measure. The loop
detector caught it; my summary column would not have.


### How the false zero happened, since it nearly went out as a refutation

Checking the Chief's reachability claim, my first sweep reported **0 of 540 tasks reachable** and I
was one step from sending it. The script called `Simulator.act`, which does not exist, inside a
`try/except Exception: continue`. Every task raised, every task was scored unreachable, and the
result was a clean table of zeros that agreed with what I already believed.

Written down because it is the day's third instance of one shape: a broad aggregate that is more
persuasive than the thing it is made of. The pooled commitment table, the flat reading join, and
this. In each case the number was wrong in the direction of the hypothesis, and in each case what
caught it was asking a question whose answer was known in advance — here, that a single task's
search returns a path, which takes one call to check and which I ran only because the zero was too
clean.
---

## Does the strategy-switching class carve the failures? Not yet, and one way of asking is broken

**The Chief flagged `list-0149` as a third shape — failed without looping, without searching, and
without entering a repeated-failure state — and said it was worth watching at n=1 rather than
acting on. The class can be computed retroactively from every episode we hold, so it is worth more
than watching. It is computed here, and the first result is that the retroactive version does not
work.**

Applying `_strategy` to all 18 completed agentic episode-runs across the three runs:

| failure shape | episode-runs |
|---|---:|
| failed and terminated on its own, with a wrong answer | **9** |
| passed | 4 |
| loop detected | 3 |
| exhausted without the loop flag | 2 |

**The strategy-switching class holds 3 of 18** — `ledger_reconcile-0163` and `update-0028` and
`batch_update-0166`, all in the two earlier runs. **The modal failure is none of the shapes the
class was built for: nine episode-runs call `finish` with a wrong answer and stop.** That is not a
model failing to switch strategy. It is a model that believes it is done.

### Why the retroactive computation cannot be trusted across runs

`repeated_failure_step` counts **consecutive observations beginning `ERROR`**. Under the old
simulator, `list_files` answered an unsatisfiable directory with `FILES: (none)` — a falsehood, not
an error. So the detector was blind to precisely the failure the old runs contained most of:

| run | steps | ERROR observations | `FILES: (none)` |
|---|---:|---:|---:|
| stage one | 89 | 16 | 6 |
| corrected rendering | 40 | 27 | 1 |
| stage two | 15 | 3 | 0 |

**A "stuck" measure defined on errors cannot see an environment that answers falsely instead of
erroring.** So the three earlier runs' strategy fields are not comparable with stage two's, and the
3-of-18 above pools two incompatible definitions. Only stage two's rows measure what the field was
designed to measure, and there are four of them so far, none in the class.

**So the Chief's caution stands and for a better reason than n=1.** It is not that the sample is
small. It is that the sample from before the simulator fix is measuring a different thing, and the
question needs stage two's own episodes to answer.

**What survives the confound and is worth carrying forward.** The nine wrong-answer terminations
are counted the same way under both simulators, because they turn on the verdict and not on error
text. That the modal failure is confident premature completion rather than any form of being stuck
is a statement about the model that the pre-registration's strategy measurement does not reach, in
any run. It should be measured beside it rather than instead of it.

---

## `update-0028` at all thirty-four layers, under the environment that tells the truth

**Stage two, 2026-09-08. 24 turns, 1,050 tokens, loop detected, never searched, stuck from step 2.
5,264.8 s. `episode_at_depth.py`, one row per decision, spans through the validated joiner.**

**Capture does not perturb generation.** The falsification run took this task with no lens
attached. All 24 actions are byte-identical between that run and this one. Greedy decoding plus a
capture that only reads makes the trajectory reproducible, and now that is measured rather than
assumed.

### The alternation is not a symmetric two-cycle

The model alternates `list_files("/")` and `list_files(".")` to the ceiling. Read at depth, the
two branches are different decisions:

| branch | n | layer where it first reaches rank 1 | rank at layer 23 | P(final), median | P(final), min |
|---|---:|---:|---:|---:|---:|
| `"."` | 12 | 24 (range 24 to 32) | 8 | 0.9988 | 0.8096 |
| `"/"` | 12 | 33 (range 24 to 34) | 108 | 0.9031 | 0.8512 |

**One branch is a layer-24 commitment. The other is decided in the last two layers, and carries a
tenth of its mass elsewhere.** Every fork has its worst rank at layer 5, between 958 and 4,751.

### Both branches are live at every fork

At every `"/"` fork, `"."` is in the top ten from layer 23 or 24 onward. At every `"."` fork,
`"/"` is in the top ten from layer 24 to 27 onward. The alternative is represented at the moment of
choosing, in every one of the 24 turns, and the model takes the one it did not take last time.

**This is the opposite of the fixed point under the lying simulator**, where `workspace` was absent
from all 192 top-10 lists and the wrong opener sat at or above 0.9999969 in 22 of 24 forks. Told
the truth, the model is not saturated: it holds two candidates, alternates between them, and its
final probability on the `"/"` branch sits around 0.90. Twenty-three refutations do move this
distribution. They do not move it toward the answer.

### The escape route is represented, late, and rejected

At the tool-name decision in each turn, `search` is in the top ten somewhere in the stack in every
one of the 24 turns:

| shallowest layer where `search` is a top-10 candidate | turns |
|---:|---:|
| 24 | 3 |
| 29 or 30 | 2 |
| 33 | 18 |
| 34 only | 1 |

So the route named in the model's own recovery rule is a candidate in its last one or two layers in
three quarters of turns, and never below layer 24. It is not that the model never considers
searching. It considers it at the end, every time, and does not do it.

### What this changes

The earlier finding — *certain, and the errors never reach its certainty* — was a property of the
false-premise regime. Under an honest environment the same task produces a model that is uncertain,
holds both wrong options and the right tool in candidacy, and still loops. The failure moved from
"the alternative is not represented" to "the alternative is represented and not selected", which
is a different object and a more tractable one. An intervention on this episode has a target that
exists in the residual: `search` at layers 24 to 33 at the tool-name decision.

Span counts for the episode: 546 note, 480 call skeleton, **24 call argument** — one per turn, the
directory literal. The primary comparison has almost nothing to read here, because the model never
writes anything but a single character as an argument.

---

## `pointer_chain-0018`: the wrong kind of answer is decided before the wrong answer

**Stage two, all 34 layers. 9 turns, 482 tokens, failed, no loop, not exhausted, never stuck,
never searched. The first stage-two instance of the modal failure class — terminated on its own
with a wrong answer — read at full depth.**

**The navigation is perfect.** The task says: begin at node 0, follow the Next path in each node,
and when a node contains a Result field report that Result exactly. The model reads node 0 through
node 6 in order, following each Next correctly, seven reads with no error and no repetition.

Node 6 says:

```
Node: final
Result: artifact-93330
```

No Next field. **The model answers `lab/test/0018/chain/node-7-288.txt`** — a file that does not
exist, in the chain's own naming convention, invented after the chain has explicitly ended.

So this is not a navigation failure and not a retrieval failure. **After seven turns of extracting a
Next path, the model extracts an eighth Next path where there is none, and reports it as the
answer.** The schema it has been executing outlives the task that needed it.

### Where that happens in the stack

At the content fork — the token where `lab` was emitted and `artifact` was wanted — the top
candidates by layer:

| layers | what the model is choosing among |
|---|---|
| 19 to 23 | punctuation and closing quotes: `.",` `"},` `)"` `{}".` |
| 24 to 25 | `{}".` `)"` `Information` `filename` |
| **26 to 29** | **`filename` `Analysis` `Located` `Here` `The` `Location`** |
| 30 | `lab` `Lab` `laboratory` — the path branch enters |
| **31** | **`artifact` enters, immediately behind `lab`** |
| 32 to 34 | both in the top four |

**The wrong kind of answer is settled about five layers before the wrong answer itself.** By layers
26 to 29 the model has committed to *reporting a location* — `filename`, `Located`, `Location` —
and the choice between the path and the Result has not yet been made. The correct content becomes a
candidate only at layer 31, three layers from the end.

### And this is the softest decision in the corpus

| | |
|---|---|
| P(`lab`), the model's own output | **0.775** |
| rank of `artifact` at the final layer | **2** |
| rank of `lab` at layer 23 | 21,462 |
| rank of `lab` at layer 24 | 590 |

Against `update-0028`'s forks at 0.90 to 0.9999999, this is close to a coin flip with the correct
answer as runner-up. **It is the first target in this programme where the right answer is live, near,
and unsaturated**, which is exactly the condition the intervention design needs and the one that
episode could not supply.

**A caution on the layer-26-to-29 reading.** Those are the top-10 lists of a lens readout, and
"the model has committed to reporting a location" is an interpretation of them, not a measurement.
What is measured is which tokens are in candidacy at which depth. Whether the schema is *causally*
held there is what an intervention at layers 26 to 29 would test, and it is a better-posed
experiment than any this episode's outcome alone would suggest.
