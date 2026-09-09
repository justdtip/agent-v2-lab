# What caught eight wrong claims in one day

**2026-09-08. Chief and D-CRO.** The programme pivoted to Gemma 3 and spent a day auditing its own
instruments. Eight claims were made and refuted between two seats. None was caught by review,
argument or seniority. Every one was caught by a measurement whose answer was known in advance.

**"None reached a conclusion" is too kind and the D-CRO has struck it.** Claim 3 travelled: it was
not merely believed, it was *recommended between seats*. The D-CRO urged that it be promoted from
an aside in the register to the register's headline, and the Chief was about to do it. It was one
step from a published document at the moment it was refuted. The record understates itself by
implying these stayed local; one of them was already moving.

This record is the technique rather than the findings, because the technique is what transfers. The
findings are about Gemma 3 4B; this is about how a small team stops itself believing things.

> **If you read one entry, read the last one.** Seventeen of the eighteen errors here were caught by
> a control on the number: deduplicate it, permute it, compare it to a baseline, ask it a boundary
> question. The eighteenth could not be, because the number was correct — real, reproducible, and
> agreeing with an independent derivation to three decimals. It was a measurement of the wrong
> artefact, and no operation on the value reveals that. The only question that catches it is *what
> is this a measurement of*, and it was answered by the person who owns the thing measured, minutes
> before a machine was committed on it. **Precision is not evidence of relevance.** Every other
> mechanism in this record assumes you are looking at the right object; that one assumption is the
> one nothing below can check.

---

## The eight

| # | the claim | who | what killed it |
|---|---|---|---|
| 1 | "Gemma does not copy literal strings" | Chief | per-call path fidelity: **0.806** against Qwen's **0.805** |
| 2 | "Our false empty listing caused the loop" | Chief | one episode re-run on the fixed simulator: it still fails, differently |
| 3 | "The environment has no discoverable root" | Chief, D-CRO | `search_files` returns the true path from any prompt word, on 165 of 180 tasks |
| 4 | "`n_valid_positions = 111` bounds the fitted position" | Chief | the upstream source: it is a count; `SKIP_FIRST_N_POSITIONS = 16` |
| 5 | "Stereotyped JSON commits earlier in depth, being predictable" | Chief | a control of the same syntactic kind: it commits **later**, 30 against 24 |
| 6 | "Wrong paths commit earlier than right ones" | D-CRO | dedup: 29 of 34 were one fixed point counted 29 times; the effect vanished |
| 7 | "The correct alternative appears at 2 of 24 forks" | D-CRO | the identity check at a boundary: 100% grouped by turn, 4% keyed flat |
| 8 | "0 of 540 tasks are reachable by search" | D-CRO | a single-task check, run because the zero was too clean |

Four of the eight were the Chief's, three the D-CRO's, one shared. That distribution is not the
point. **The shape is the point, and it is the same eight times.**

---

## The shape

> **A broad aggregate is more persuasive than the thing it is made of, it is wrong in the direction
> of the hypothesis, and it is caught by asking something whose answer was already known.**

Each half does work.

**More persuasive than its parts.** Claim 6 was a clean 24-against-30 split on exactly the layer pair
the map was built to examine. Claim 8 was 0 of 540, a perfect result. Claim 1 was 165 of 180 tasks
carrying a literal path, a real number that licensed a false one. An aggregate hides its own
composition, and the larger it is the more it hides.

**Wrong in the direction of the hypothesis.** Not one of the eight was a random error. Every one made
the thing its author already believed look true. Claim 8 is the sharpest: a bare `except Exception:
continue` around a call to a method that does not exist scored every task unreachable, and the author
believed the root was unreachable. **A bug that disagrees with you is found in minutes. A bug that
agrees with you is published.**

**Caught by a known answer.** This is the operative half, because the first two describe the disease
and this one is the treatment.

---

## The three mechanisms, in order of how much they caught

### 1. A check whose answer you already know, run at a boundary

Claim 7 was caught because the final layer's readout *is* the model's own distribution, so its top-1
*must* be the emitted token. Grouped correctly it was 100% of 1,173 emissions; keyed the wrong way it
was 4%. Nothing else in the chain would have revealed that two coordinate systems had been conflated,
because every other number was plausible.

The check was, separately and correctly, judged near-worthless as validation: it asserts that
decoding is greedy, that the horizon is subtracted rather than added, and that the record's logits
line up with the tokens emitted — all bookkeeping, and nothing about the lens. **Both are true, and
the tension is the lesson.** A check that cannot fail teaches nothing about the thing it points at,
and is the only thing that catches a join, an index convention or an off-by-one. Keep it, label it
honestly as what it asserts, and never let its passing stand in for validation it does not perform.

The resolution was not to choose between the two readings but to build the missing half beside the
existing one: the readout's identity branch now also runs at the final layer and is compared against
the model's own logits every forward, recorded as a number and gated. The bookkeeping assertion and
the instrument validation are now two checks that fail for different reasons, which is what they
should always have been.

Claim 8 was caught the same way, informally: 0 of 540 is a value the author knew could not be right.

### 2. A control of the same kind as the thing measured

Claim 5 died against a control built from argument-start tokens specifically, rather than from a
pooled background of prose and JSON together. Against the pooled background the effect read as
unremarkable at 0.258; against its own syntactic kind it was 0.016. **The control did not dissolve
the finding, it sharpened it by a factor of sixteen** — which is the argument for building the right
control rather than the available one, in both directions.

The generalisation: before comparing two spans, check that neither is a compound of populations that
behave oppositely. A "call span" turned out to be exactly that — skeleton tokens committing at layer
24 and argument tokens at layer 30 — so any single figure over it averaged away the effect. This
forced a pre-registered comparison to be restated before it was ever computed.

### 3. Deduplication before any count is called a sample size

Claim 6 died because 29 of 34 observations were one decision repeated. **A decision a model repeats
is one observation, not many.** Twenty-three identical calls are twenty-three copies of one
commitment, and pooling them produced a clean, false result on the layer pair under study.

This is now a rule: no count over positions is quoted as a sample size without deduplicating
byte-identical repeated decisions within an episode. It is aimed at the most persuasive artefact in
the corpus, because that is the one that needs it.

**And the sharpest fact about claim 6 is missing, which the D-CRO supplies against himself.** The
rule already existed when the claim was made. He had written *"these twenty-four are one decision
repeated, not twenty-four independent samples"* into the fixed-point record **about an hour
earlier**, and then pooled twenty-nine repetitions of that same decision at the first opportunity.

**That fact is the most important one in this record and it is carried to the end**, because it
refutes the record's own first thesis rather than any single claim. See *The thesis, in its corrected
form*.

---

## Two practices that made the catching possible

**Run the experiment that threatens your own account, first.** This is listed first on the D-CRO's
argument, which I accept: it is the only item here that prevented a **downstream** cost rather than
correcting an **upstream** claim. Every other mechanism repairs a number that is already wrong. This
one stops a wrong number being built on. Claim 2 died to a test the Chief asked the D-CRO to run
*because* it risked the Chief's own explanation: 66 seconds, against a 4.6-hour run and a published
register that would otherwise have been read through a false causal story.

**Keep the refuted version beside the corrected one.** Both authors kept their wrong tables in the
record rather than deleting them, so the next reader meets the persuasive version already refuted
instead of rediscovering it as a result. A record that merely omits a refuted claim leaves the trap
armed.

---

## What transfers, stated without reference to this code

1. Run the cheap experiment that could refute you, before building on the claim. It is the only
   item here that prevents a cost rather than correcting a claim.
2. Before believing an aggregate, deduplicate it and recompute. If the effect is carried by repeats
   of one event, there is no effect.
3. Build the control from the same class as the thing measured, not from everything else.
4. Keep at least one check whose answer is known in advance, and run it at a boundary. Label it as an
   assertion about bookkeeping, never as validation of the instrument.
5. Treat a clean number as a reason for suspicion proportional to how much you wanted it — and
   discharge the suspicion with the smallest possible check, not a better version of the same
   computation. Claim 8's 0 of 540 was killed by running **one** search on **one** task by hand.
   Re-running the sweep more carefully would have reproduced the bug.
6. A bare `except: continue` around a measurement is a machine for producing agreeable falsehoods.
7. When a claim is refuted, strike it where it was made. A correction two hundred lines below a bold
   sentence leaves the bold sentence doing the work.

---

## The thesis, in its corrected form

The first version of this record said mechanisms beat care. The D-CRO's second correction shows that
is not enough, and the correction is against his own entry, so it is worth stating where the weaker
version was rather than only where he made it.

**A written rule is not a mechanism.** The deduplication rule already existed when claim 6 broke it.
It had been written into the fixed-point record an hour earlier, in the author's own words, about
that exact episode. It sat in a committed document and did not fire. It became a mechanism only when
it became a **computation that must execute before a number exists** — dedup first, then count.

So the ladder has three rungs and only the top one holds:

| | fires when | held today |
|---|---|---|
| care | you remember | no, eight times |
| a written rule | you remember at the moment of temptation | no — claim 6 broke a rule its author had written an hour earlier |
| a computation that must run before the number exists | always | yes |

**A rule you have to remember at the moment of temptation is care wearing a mechanism's clothes.**
That is the sentence this record exists for, and it was reached by one author breaking his own rule
and then saying so.

---

## The count kept growing, and that is the honest result

This record began at eight refuted claims. By the end of the day it was thirteen, and the additions
came from both seats at roughly the rate the first eight did:

| # | claim | who | what killed it |
|---|---|---|---|
| 9 | scored positions re-enter the sink regime at turn boundaries | Chief | rank rows exist only for generated tokens; smallest prompt is 420 |
| 10 | wrong answers are emitted as confidently as right ones | Chief | control: median P ≈ 0.999999 on *every* token, so the statistic is saturated |
| 11 | the logits are recoverable from the records at no box cost | Chief | the Director asked; only a SHA-256 is stored, never the distribution |
| 12 | 3.4 GiB peak and 2.1 s/token for stage two | D-CRO | outcome: 4.70 GiB and 3.68 s/token |
| 13 | Gemma is certain and unmoved by refutation | Chief | the same episode under an honest simulator: 0.9031, both branches live |

**We did not get better at avoiding wrong claims. We got better at catching them.** That is worth
saying plainly, because the opposite reading — that a day of method work reduces the error rate — is
the flattering one and it is not what happened. The mechanisms fire; the claims keep coming.

## A fourth mechanism, and a fourth shape

The three mechanisms above catch errors in a **number already computed**. Claim 12 is a different
animal: a projection made *before* a run, from a measurement taken where it was cheapest to take.

**The shape: measured where it was cheap, not where it was representative.** The D-CRO calibrated
stage two's memory and rate on `calculate-0158` — 2 turns, 103 tokens, the shortest episode in the
set. Both projected quantities scale with context, and the run then reached 2,749 positions. One
mistake, made twice, missing by 38% on memory and 2.3x on rate. It is the deduplication error's
sibling: not *correlated observations counted as independent*, but *an unrepresentative instance
treated as typical*. Both are sampling failures that favour the cheap measurement.

**The mechanism that caught it: a declaration that carries the measurement it rests on.** R60(c)
requires a projection to name its evidence, not merely to state a ceiling. Because the declaration
said *3.4 GiB, measured on `calculate-0158`*, the miss was **diagnosable rather than merely visible**
— the flaw could be named as a method error instead of guessed at as bad luck. A bare ceiling would
have been breached identically and taught nothing.

That is the argument for R60(c) rather than an embarrassment under it, and it generalises past this
repository: **a projection without its basis cannot be learned from, only failed.**

**The corollary, now a rule: calibrate at the largest instance the run will reach, and identify what
the cost actually scales with.** Stage two's driver is episode *length*, not episode *count*, so any
projection from a count is wrong in the same direction every time. The same error was live in a
second seat within the hour — a 21 GiB projection for a 2,816-token lens fit, on a box whose R47 stop
threshold is 10.66 GiB — and was caught by applying this rule before the run rather than after it,
which is the first time today a mechanism prevented a claim instead of refuting one.

---

## Eighteenth, and the most consequential: I authorised a run on a measurement of the wrong artefact

Codex caught it before a checkpoint loaded, which is the only reason it is a method entry and not an
incident.

**What I told the Director.** "Authorise Codex's transcript fit at batch 4. That configuration
measures at about 5.5 GiB, half the stop threshold." Both halves were false. **Nothing measured
5.5 GiB**; it was my arithmetic. **No batch-4 configuration exists.** I cited `regression.py:201` as hardcoding `batch_size=1`; the
D-CRO found no batching concept in the fitter at all, and both are half right. The string is there in
Codex's worktree — as a **manifest annotation** inside `counts.update(batch_size=1, ...)`, a label
describing the run for a later reader. It is not a parameter and nothing reads it. The fitter
accumulates row by row through `accumulate(view, rows, ...)`. So "a batching choice" named a lever
that does not exist, and my citation of the line was a grep hit read as a setting. I said "measures" about a number I had projected, four hours after writing a
rule that a projection must carry the measurement it rests on.

**What was actually measured, and of what.** The D-CRO's calibration ran `NativeCapture` with a sink
that retains each layer's residual in float32 — one 2,816-token window, 33 layers, **2.817 GiB
peak**. Its own docstring says *"which is what a fit holds"*. **That is the error, and we both made
it.** A fit does not hold residuals; it holds the model, the unembedding, and a 2,560-by-2,560
accumulator pair per layer. Codex's fitter, measured on the earlier 128-token prose fit, peaked at
**12.44 GiB**. At 128 tokens the residuals are 0.04 GiB. The other twelve gigabytes are the fitter.

So the 21 GiB projection for 2,816 tokens was not a batching artefact. It was a reasonable
extrapolation from 12.44 at 128, and my "21 GiB is a batching choice, not a floor" inverted the
truth: **there is no batch, and the floor is roughly 12 GiB before a single residual is retained.**

**The shape.** A measurement of one artefact taken as a calibration of another. It is the pattern
the D-CRO diagnosed in my work weeks ago — reaching for a property of the thing in hand when the
question is about a different thing — and it is the first time today it nearly cost a machine rather
than a claim. The 2.817 GiB number was real, precise, cross-checked against my own derivation to
three decimal places, and about the wrong object. **A measurement's precision says nothing about
whether it measures the thing you need.**

**What is true now.** The bf16 fit at 2,816 tokens needs, at minimum, the 7.3 GiB bf16 model, the
2.69 GiB fp32 unembedding, ~1.7 GiB of accumulators, and 0.89 GiB of residuals: about **12.6 GiB
before attention workspace**, on a box whose R47 stop is 10.66. **It may not fit on this machine at
all.** The 4-bit fit replaces 7.3 with roughly 2.5 and lands near 7.8 GiB, which does. That is a
decision the order did not anticipate and it follows the diagnostic, not the other way round.

---

## Nineteenth and twentieth, on the second day: a mislabelled commit, and the push that carried it

**The D-CRO's, self-reported before anyone found it.** Commit `4383fda`, described as a documentation
change — "upstream's defaults verified against the clone" — silently carried 880 lines of a
half-written adapter that a subagent was actively writing, because the commit used `git add -A` in a
shared checkout. The commit message misdescribes the commit, which is worse than the file being in the
wrong place: a wrong place is discoverable and a wrong message is not. Forward fix, honest messages,
no history rewrite, because the commit was already on origin.

**The Chief's, which is why it was on origin.** Given standing authority to push, I pushed the
twenty-three commits behind it without reading the range. `4383fda` was among them. **Standing
authority to push is not authority to push unreviewed**: a `git log --stat` over the range before a
push is the check, it takes ten seconds, and it would have shown 880 lines of source under a
documentation message. Recorded against me beside the D-CRO's `git add -A`, because the two errors
compose — one put the wrong thing in a commit and the other put the commit where it could not be
undone.

**Two rules, both mechanisms rather than care.** A seat in a shared checkout stages by explicit path,
never `-A`, and the pre-commit hook may refuse `-A` there if anyone wants to make it unforgettable.
And a push is preceded by `git log --stat <remote>..HEAD`, read, with any commit whose diff does not
match its message stopped on the spot.

**The shape, for the record.** Neither error was caught by a test, a control or a gate, and neither
could have been: both are about what a commit *says* against what it *contains*, which no number
measures. They were caught by the person who made one of them reading his own history. That is the
category the precision entry at the top of this record already names — the assumptions the
mechanisms sit on — and it is the second entry in it.

**A note the D-CRO can add and the Chief cannot: the habit was safe for forty commits and stopped
being safe without a signal.** `git add -A` had been correct in that checkout all day, because one
seat was the only writer in it and every uncommitted change was that seat's own work, deliberately
made. It became wrong at the instant a subagent began writing files concurrently — and nothing
marked that instant. The command did not change, the checkout did not change, and the habit's
justification silently expired.

That is why the rule is staging by explicit path rather than "be careful with `-A`". A rule that
depends on noticing when your own preconditions lapse is care wearing a mechanism's clothes, which
this record already says about the deduplication rule that its author broke an hour after writing it.
The same shape, one level out: not a wrong belief about the data, a lapsed precondition about the
world the command runs in.

---

## Twenty-first: a gate that compared the column that agreed

**SWE-2's, caught on the laptop before a device was rented.** The multi-device gate said "one device
and two produce the same loss curve to tolerance". With the blocks sharded and the root replicated
outside any FSDP2 unit, the loss was bit-identical at step zero while the root's gradient was forty
per cent wrong; over six steps the loss drifted 8.83e-04, inside any tolerance anyone would have
written. Loss on identical weights and the same rows agrees by construction, and afterwards it is
one step behind the parameters. The gate would have passed the configuration it existed to catch,
and the damage would have surfaced as a run quietly optimising something else on rented hardware.

**The rule.** A gate compares the quantity nearest the mechanism it guards, never one downstream of
it: gradients and parameters for a training gate, residuals for a forward gate, the mask for a
masking gate. A downstream number is reported beside the gate and passes nothing on its own. And a
quantity that agrees by construction at the point of comparison (identical weights, same rows, step
zero) is not evidence, whatever tolerance it clears.

**Where it sits.** Beside the ninth and the eighteenth: a passing number produced by a comparison
that could not fail. This one is the cleaner instance, because both columns were real measurements
and the wrong one was the one the specification named.

---

## Twenty-second: two rules from the integration hour, and one reader that served two masters

**Both SWE-2's, from wiring the torch trainer into the pipeline.** First, `Trainer` chose MPS on
this laptop while the manifest said `cpu`, because the manifest recorded the request and not the
object; the device is now read back off the model. Second, the chunked loss, which reaches around
`forward` to avoid a 262,208-wide logit tensor, ran without the autocast that `accelerate` attaches
to `forward` and without the unshard hooks that FSDP2 attaches there, and each absence surfaced as
a separate bug. **The rule: anything that bypasses `forward` inherits none of what upstream attaches
to it and must supply it itself.** The §16.7 wrapper dissolves both cases at once by making the
chunked loss the forward that everything attaches to.

**The third member, found by following the fix to its end.** With the wrapper handed to `Trainer`,
`_save` branched on `isinstance(model, PreTrainedModel)`, found a wrapper, and wrote a bare state
dict with `inner.*` keys and no config, silently, producing an artefact nothing downstream could
load or name-match. So the rule is wider than `forward`: **upstream inspects the object it is
handed, and a wrapper changes both what runs and what it is.** A wrapper must delegate what
upstream branches on, and a test must assert the artefact is the kind of thing the next reader
expects, not that a file exists.

**The second rule, from two readings minutes apart.** `1115 passed, 1110 skipped` and `2211 passed,
14 skipped` on the same tree, both truthful: a box window was open for the first and the suite
correctly stood off every model-reaching test. **A suite reading is not a claim unless it carries
its skip count and its window state on the same line.** "Full suite green" without them is the
summary line of whichever run happened to be quoted.

**And the Chief's, found by SWE-2 from a worktree.** "No stage can load a registered checkpoint
from a worktree" was true for every process with `$AGENT_V2_BOX_STATE_DIR` set and false for every
process without it: `_resolve_checkpoint` read the primary checkout through `box_state_root`, whose
override exists so an isolated run cannot take the machine's lock, and a checkpoint that followed
the override resolved into scratch. Two things shared one reader; only one may be redirected. The
git-derived primary is now its own function and the checkpoint uses that. The shape is the eighth
entry's again: a mechanism built for one purpose, reused for another because it was there, carrying
a behaviour the second purpose never asked for.

---

## Twenty-third: a guard that passed the object it existed to catch, and a fixture that could not fail

**SWE-2's, on the torch train stage.** A guard meant to refuse the multimodal wrapper tested for
`.model` and `.lm_head`; the wrapper has both, so the guard passed it and the run failed later and
elsewhere. **A guard that passes the object it exists to catch is worse than none, because its
silence is read as evidence.** And nothing caught it because every fixture in the stream was built
from `Gemma3TextConfig`, which saves `architectures: [Gemma3ForCausalLM]`, a config shape no
registered checkpoint has: the suite exercised a stand-in that differed from the real thing in
exactly the way that mattered. **A synthetic fixture carries the real artefact's declared type and
key layout, or the path is untested by construction.** The Chief's own loader tests had the same
shape, a made-up wrapper type over a Llama text config, and were corrected the same evening with a
fixture mirrored from the snapshot's headers. It is the fourteenth entry's blind spot (the
uncached tokenizer) with a different artefact.

**One level out, the same evening.** The test that the stage's checkpoint reloads passed while
every weight came back freshly initialised, because the reader it used treats missing keys as a
warning. **An assertion that an artefact loads is worth nothing unless the loader fails closed**: a
tolerant reader turns a corrupted artefact into a passing test, which is the guard that passes
the object it exists to catch, applied to the reader instead of the guard.


**The family, named by SWE-2 once its third member was found.** The guard that passed the wrapper it
existed to catch; the reload assertion satisfied by a reader that treats missing weights as a
warning; and the `assert X == Y or True` in a record's producer, which parses as
`(comparison) or True` and has never tested anything (the twenty-sixth entry). None of these is an
absent check. Each is **a present check that is inert**: silence was read as evidence, and a reader
looking for a guard found one. That is the sharper form of both rules above and it supersedes them
as the thing to look for: not "is there a check" but "can this check fail, and has it".

---

## Twenty-fourth: the shared index, or why explicit-path `add` was not enough

**The Chief's, the same shape as the nineteenth, one mechanism deeper.** Commit `9ae4444` says it
pins upstream `jlens` in the cuda extra. It also deletes `pipeline/lens_fitting/upstream.py` and
`tests/test_lens_upstream.py`, 1,862 lines, and says nothing about them. The D-CRO had staged that
removal, step three of their announced move to `cuda-ws-d`, in the shared checkout's one index;
I added my two files by explicit path, as the nineteenth entry's rule requires, and then ran a
bare `git commit`, which commits everything staged. The rule against `add -A` protected against
staging someone else's changes myself. It did not protect against committing changes someone else
had staged, because `add` and `commit` are two gates and the rule covered one.

**The removal is the D-CRO's and intended**: both files are on `cuda-ws-d`, and `9ae4444`'s parent
has both, so nothing is lost. The message is wrong, and it is on origin, so this entry and the
naming commit that carries it are the forward fix; no history is rewritten.

**The rule, sharpened into a mechanism.** In a shared checkout, commit with a pathspec:
`git commit -- <paths>`, which takes only those paths whether or not anything else is staged, and
read `git status --short` before every commit, treating any staged entry that is not yours as a
stop. `git add` by explicit path stays; it was never the whole of the discipline.

**The message the removal was meant to carry, in the D-CRO's words, so it is on record beside the
commit that swallowed it:** "Move WS-D's adapter off the main line to `cuda-ws-d`, per plan §15.
`pipeline/lens_fitting/upstream.py` and `tests/test_lens_upstream.py` are CUDA-line work and the
main line takes none. Both are on `cuda-ws-d` at `5121083` with all four adversarial fixes and 35
tests. `local_llm_lab/upstream_ref.py` deliberately stays on main with its own
`tests/test_upstream_ref.py`: it is the single import seam that WS-A and WS-D share, it is not
backend code, and moving it would recreate the two-import-paths problem it was written to close.
The 880-line intermediate of the adapter first reached main at `4383fda` under a documentation
message, swept in by a `git add -A`. This is where it leaves."

**And the D-CRO's statement of the shape, which is the better one.** "`git add -A` was safe until a
subagent wrote concurrently; explicit-path `git add` was safe until a second seat committed from the
same index. Both times the command was unchanged, the checkout was unchanged, and only the world
around it moved. A rule that protects you *given* an assumption about who else is touching the
index is care wearing a mechanism's clothes; `git commit -- <paths>` does not depend on that
assumption, which is why it is the one that survives."

## Twenty-fifth: a merge that was already done, and a deletion waiting to be taken silently

The CRO asked me to merge `cuda-ws-d` into `cuda-migration` under their review, the way WS-A's seat
had been merged, and to confirm the merge-tree was clean against origin's tip first. The confirmation
is where this entry starts, because the merge does not exist.

`git merge-base cuda-ws-d origin/cuda-migration` returns `5121083`, which *is* `cuda-ws-d`'s tip. The
branch is zero commits ahead of the integration branch and forty-seven behind it. The merge-tree
writes `9d3fefa8ee`, which is `origin/cuda-migration`'s own tree, unchanged. There is nothing to
integrate because WS-D's adapter never travelled on `cuda-ws-d`: it was written on the main line
(`4383fda` swept the 880-line intermediate in, `805225c` finished it, `1912fa7` split out the seam),
and it reached `cuda-migration` through `c7cb0da`, an ordinary merge of main. The branch named after
the seat is a label on a commit the integration branch already contains.

The twenty-fourth entry says of the deletion at `9ae4444`: *"both files are on `cuda-ws-d`, and
`9ae4444`'s parent has both, so nothing is lost."* That sentence is true and it is the wrong
reassurance. The files are not preserved *by* `cuda-ws-d`; they are preserved by `cuda-migration`,
which carries them at the same blob hashes — `50cdbded86` for the adapter, `1c96991409` for its
suite — and which is the only branch where they are exercised. I ran that suite at
`origin/cuda-migration`'s tip in a detached worktree: 35 tests, 35 passed, nothing skipped, all four
adversarial fix markers present in the 975-line file. `tests/test_upstream_ref.py` passes on main,
3 tests. Every file is where the plan says it should be.

**The defect is in the direction nobody was checking.** `9ae4444` deleted the adapter and its suite
from main, deliberately and correctly, because the main line takes no CUDA code. The merge base of
main and `cuda-migration` is that same `5121083`, which still has both files. So relative to the
base, main deletes two files and `cuda-migration` does not touch them — and git resolves
delete-against-unmodified by taking the delete, with no conflict and no prompt. I simulated it:
`git merge-tree --write-tree origin/cuda-migration origin/codex/agent-v2-specs` reports zero
conflicts and writes a tree whose diff against `cuda-migration` is exactly

```
D  src/local_llm_lab/pipeline/lens_fitting/upstream.py
D  tests/test_lens_upstream.py
```

The next routine merge of main into the integration branch removes WS-D's adapter and its 35 tests
from the only branch that has them, and reports success while doing it. Nothing warns, because
nothing is wrong: both sides did what they meant to, and the merge did what a merge does.

**The fix belongs in the merge, not in a note.** When main is next merged into `cuda-migration`,
restore the two paths inside that merge commit rather than after it:

```bash
git merge --no-commit origin/codex/agent-v2-specs
git checkout HEAD -- src/local_llm_lab/pipeline/lens_fitting/upstream.py tests/test_lens_upstream.py
git commit
```

Doing it in the merge is what makes it a one-time cost. The merge commit records main's deletion as
merged while keeping the content, so the base moves past `5121083` and no later merge re-proposes it.
Restoring afterwards in a second commit leaves the deletion unmerged and the trap re-arms every time.

**The shape.** A branch that is an ancestor of its target looks exactly like a branch that has been
merged, and both the CRO and I read "the files are on `cuda-ws-d`" as "the work is on `cuda-ws-d`,
waiting". Asking what a merge *would do*, rather than whether it would conflict, is what turned a
routine confirmation into the finding: `merge-tree` reporting zero conflicts was the answer to the
question I was told to ask, and the diff of its written tree was the answer to the one that mattered.
A clean merge-tree means the merge is unambiguous. It does not mean the merge is harmless.

## Twenty-sixth: twelve benign closures, and the dead assertion found beside one of them

SWE-2 routed twelve `B023` warnings — a function defined in a loop closing over the loop variable —
across four files owned by other seats, with the reading that all twelve are benign, and the
explicit caveat that they had read for the closure question only and not for whether the scripts
compute what their records claim. That second question is the one that came to this seat.

**The closure reading is right, and I confirmed it independently rather than from the message.**
`ruff check --select B023` returns exactly twelve, in the four named files, at the lines and over
the variables SWE-2 listed. Every one is invoked inside the iteration that defines it: `g` in
`build_report.py:242` is consumed by the `"".join(...)` on the next line; `med_cos` in
`query_cosine.py:69` twice on the next line; `run_rest` in `cache_split_diagnostic.py:143` three
times at lines 150, 153 and 156, all before the loop turns; `turn_logp` in
`fixed_history_lens.py:263` three times on the line after it. None is stored, returned or deferred.
`B023` is about a call that outlives its iteration, and none of these does.

**The thing worth having is the one the closure question walked past.** `turn_logp` closes over
`n_p` and `ids_full`, and its correctness rests on an assumption the closure warning cannot see:
`n_p` is `len(tok(prompt_text))`, the prompt tokenized alone, while `ids_full` is
`tok(prompt_text + adapter_turn_t_raw)`, the two tokenized together. The slice `ids_full[n_p:]` is
the turn's own tokens **only if the concatenation does not re-tokenize across the seam**. Tokenizers
merge across a join routinely; when they do, the slice is off by one or more and the summed
log-probability is computed over a window that starts inside the prompt. These are the numbers the
record's attribution rests on: the adapter's wrong turn at −0.1 to −4.3 nats against the base's
correct turn at −12 to −83.

**Twenty lines above it there is a guard for exactly this, and it cannot fail.**
`fixed_history_lens.py:207` reads

```python
assert list(mtok.encode(p["prompt_text"])) == tok(p["prompt_text"], ...)["input_ids"][: p["n_prompt_tokens"]] or True
```

`==` binds tighter than `or`, so the whole expression is `(comparison) or True`, which is `True` for
every input. The assertion has never tested anything. It is the twenty-third entry's shape a second
time — a fixture that could not fail — and this time it sat in the run that produced a published
figure. Its intent also differs from the assumption above: it compares the *MLX* tokenizer against
the *HF* one, which is a specials-and-BOS question, not a seam question. So even alive it would have
guarded a neighbouring claim rather than this one, and it may well carry `or True` because it
tripped on a benign specials mismatch when it was written.

**So I measured the seam instead of arguing about the guard.** Tokenizer only, no weights, no lock:
for each of the eleven pairs in `run/fixed_history_lens.json`, compare `n_prompt_tokens` plus the
turn tokenized alone against the recorded `n_total_tokens`, and the same for the base turn against
`base_turn_tokens`.

```
11 adapter turns: 11 clean, 0 seam shifts
11 base turns:    11 clean, 0 inconsistent
```

Every seam is clean. `ids_full[n_p:]` is the turn's tokens in all twenty-two cases, so the
log-probability comparison in `ARM-A-DIVERGENCE-2026-09-07` is sound, and now sound by measurement
rather than by the assumption a dead assert was standing in for.

**What is not fixed, deliberately.** I have not edited the assertion. Making it live is a one-token
change and a bad unilateral one: if it carries `or True` because the two tokenizers disagree on
specials, turning it on breaks the script for a reason unrelated to anything it protects. The choice
belongs to the seat that owns `scripts/fixed_history_lens.py` — either delete it, or replace it with
the seam check above, which is the assumption the script actually depends on and which now has a
measured value to assert against. Leaving it as it stands is the one option that should not survive,
because a reader who greps for a guard finds one.

**The shape.** A lint class defines the question it asks, and answering it well is not the same as
answering the question the code raises. Twelve closures were benign and reading them was still worth
it, because the file that held the subtlest of them also held an inert guard over the assumption
that subtlety depended on. The warning was not the finding; it was the reason someone read the line.

---

## Twenty-seventh: a check that cannot pass

**The D-CRO's, in their own golden harness.** The gate that the two lenses under comparison differ
only in their estimator compared the declared ν blocks whole. Each estimator records its own knob
inside the block, and one records a backward-accumulation dtype the other has no backward pass to
accumulate in, so two correct fits of one corpus could never have satisfied it. **A check that
cannot pass is as useless as one that cannot fail**: the mirror image of the inert-guard family,
in the instrument rather than the subject. It was found only by building the real second operand,
because a fixture-shaped stand-in would have been written to satisfy the gate, which is the general
argument against mocking the thing a check exists to judge. Fixed to compare, key by key, what
decides the fitted quantity, with a test at each edge: a knob difference passes, a changed
selector or dtype still refuses.

## Twenty-eighth: an instruction that named a mechanism without naming its branch

**The Chief's.** SWE-2 was told to "reuse the `fetch-dictionary` mechanism in `device_setup.py`"
for the preflight's dictionary rows. The mechanism existed on main and had reached the integration
branch only with a forward merge made minutes after the instruction; SWE-2's tree was the
integration tip from before it, where the file held no such command. They searched, found nothing,
said so, and built one, and the two are textually mergeable and semantically a broken program: two
`fetch_dictionary` functions in one module and two `fetch-dictionary` subparsers in one `main`,
which `argparse` refuses at startup. The same instruction cited "the 12B re-render assertion in the
runbook", also on main only; SWE-2 reported it absent, correctly, from the tree they had.

**Two rules.** An instruction that names a mechanism or a sentence names the commit it is on, and
the seat merges that commit forward before building on it; the Chief checks the integration tip,
not main, before writing "reuse". And a merge that `merge-tree` calls clean is clean at the level of
lines, not of programs: when two branches add the same command, the merge is tested by running the
command, and the resolution is one implementation, chosen on its merits and named in the review.
Here the Chief's version stays, being on both branches and verifying every file, and takes from
SWE-2's the fetch-time digest sidecar and the offline check that the preflight rows need; the
preflight rows and the render row are SWE-2's and are rebased onto it.
