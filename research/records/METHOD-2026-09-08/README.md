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
