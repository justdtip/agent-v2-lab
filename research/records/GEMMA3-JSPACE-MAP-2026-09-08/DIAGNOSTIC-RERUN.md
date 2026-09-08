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
to that note is `read_file`, twenty-three times. **There is no tool that creates a file.** The
model has formed an intention the toolset cannot express and emits an action that does not
implement its own stated plan.

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
