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

## The third got worse, and it is the sharpest finding of the day

`update-0028`'s prompt gives the path verbatim: *In `workspace/test/0028/config.ini`, change mode
from fast to audit.* Four expert steps: read, replace, re-read, finish.

Stage one's model read `/workspace/test/0028/config.ini` — the given path with a leading slash it
added itself — and never recovered. The corrected run reads `/test/0028/config.ini`, which drops a
component **and** adds the slash, and then issues that identical call twenty-three times running.

**Gemma does not copy a literal path. It regenerates one, and the regeneration is wrong.**
Everything downstream is a consequence of that single corrupted string, and no amount of ceiling
helps a model that cannot re-emit a path it was given.

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