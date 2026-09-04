# Night report for the Director, 2026-09-05 (Chief AI Research Scientist)

Standing instruction for the night: get through as much of the interpretability as possible; all
interpretability lifts granted; the Head of Interpretability runs EXP-001 on the Chief's green.

## The result

**EXP-001 verdict, ratified: World A generalises to the hybrid. The D4 recipe is unchanged and
note discipline stays load-bearing.** On the Qwen3.5-4B base, as on the 3B, the hidden filename
does not survive the observation window: the hidden suffix's own probability never moves with
its context at any readout, on either block kind, across the workspace band. Every apparent
ordering effect is the already-read candidate moving. The decisive row (the model's own output)
is not significant when paired against the pre-registered null (10 v 4, p 0.18); the marginal
30/42 that looked like a memory signal is a test against a fair coin, the wrong null. The
positive control passes at named layers, with its sign flipping by depth and opposite to the
3B's, so controls are now worded by magnitude, not direction. The spot check shows the same
model at the same token emitting the suffix at p ≈ 1 when it is visible, so the null is
information, not capability. Readings: `under_review/EXP-001-RUN1-RESULT-READ-2026-09-05.md`
and `EXP-001-RUN2-RESULT-READ-2026-09-05.md` (runs 3 and 4 as its addendum). Record: #54.

## What landed (all gated on a full read of every hunk)

| commit | what |
| --- | --- |
| `bcfc6f9` | C7: provenance carries the resolved model identity in rollout and branch |
| `461b679` | EXP-001 slice: per-prompt cache (halves the run), both reduction axes, C3 to C6 |
| `f1230ac` | Preflight footprint calibration (schema 3; the probe gate no longer reads the footprint; the training gate refuses a skipped or failed footprint); both preflight artifacts regenerated bare and recorded by revision and digest |
| `4d4da3c` | R32 stage 2: chunkwise-parallel gated delta for training, fallback in the record |
| `2bb2761` | Instrument fix, five parts (period from the blocks; final-layer structural zero; two reader/writer mismatches; hard refusal on an underived family) |
| `927807b` | Records: R38, R37 amendment, spec edits, readings |

Pending, approved: the adapter-wrapped fixtures slice (#52), landing now.

## Three misses at the gates, one class, one ruling

The period walk failed on the real model because `mlx.nn.Module` is a `dict` subclass and a
Mapping-first accessor never sees plain attributes; its test used a plain fake. The R18a
precision block had been null in every probe artifact since 6f84217 because the reader looked
at the wrong level; its test used a hand-made record. A stage nothing drove through its own
entry point (#64). **R38** (wiring map §7): a reader is tested against the real writer; fixtures
through the real writer, class or view, never by hand; the reviewer checks the fixture's
provenance. #70 sweeps fourteen such sites in daylight. The Chief's own miss on 6f84217 is on
#59.

## Decisions for the Director

1. **B4 attempt 4** needs a training lift, and under T1 (#51) the 4B preflight must be
   regenerated with the arm's own chunk, chunkwise mode and longest trained row immediately
   before its gate. Stage 2 has removed the recurrence as the blocker; the footprint is the
   remaining question.
2. **The Head's #54 authorisation**: they hold their posts there for want of your word in their
   session; the Chief has posted on their behalf tonight.
3. **Daytime queue**: fixtures commit; P6 secondary condition (lifted; reading pre-registered on
   #26; the test of whether the note-representation failure generalises across families);
   #69/#72 (labelled control, paired test as the headline statistic, decomposition for every
   readout); #70; #71; #67; #63; #65.
