# R26 run logging, lanes A–D (issues #36–#39), review round 1, ratified (2026-09-05 04:30)

Reviewer: Claude (Chief). Evidence: the four work orders and the Chief's direct read of
`runlog.py:95-112,555-770` (JSON sanitiser, `TrainingHealth`), the full lane B diff of
`pipeline/cli.py`, and the log-open / guard / progress call sites in lanes C and D. The
independent R19 round was skipped on the Director's instruction for these lanes; accepted for
this ruling-driven, contract-first work, not as a precedent.

## Verdicts

| Lane | Issue | Verdict |
| --- | --- | --- |
| A core module | #36 | APPROVED TO COMMIT (first) |
| B training integration | #37 | APPROVED TO COMMIT with condition K1 |
| C P6, ablation, adapter-delta | #38 | APPROVED TO COMMIT |
| D state probe, assistant axis | #39 | APPROVED TO COMMIT |

## Lane A, as read

Rules are local and cheap as the docstring promises: non-finite loss raises `TrainingAborted`
after recording; spike and throughput compare against the median of the *prior* window only
and stay silent until it is full; memory flags once; the val-rising streak re-arms only after
a decrease; `on_finish` records `incomplete_run` as fatal without raising; verdict precedence
aborted > incomplete > warnings > healthy; `events.jsonl` is strict JSON. Correct.

## Lane B, as read

Thresholds and the state machine are built before any load; `metrics.jsonl` is written before
the rules run so an abort leaves the row; `train.log` stays verbatim through the nested tee;
`health.json` is written in `finally` on every path; an abort writes no provenance and exits
naming the flag; provenance carries the final verdict. Correct, with one gap:

**K1 (must land before commit).** `on_finish(final_checkpoint=(adapters / "adapters.safetensors").is_file())`
reads True for a **stale** file left by an earlier run in the same output directory, exactly
the class SPEC-002 §5 fixed in `checkpoint_dirs`. Require the file to be newer than the
stage's `started` time (or equivalently the `{iters:07d}_adapters.safetensors` save with a
post-start mtime); a fake-only test plants a stale file and expects `incomplete_run`.

Notes accepted as non-blocking: the unknown-`train.health`-key traceback; thresholds recorded
twice in provenance; the stash/pop hazard recorded (use `git show HEAD:path`).

## Lanes C and D, as read

`RunLog.open` precedes the GPU guard and the model load in every CLI; P6 reports per captured
case and per (layer, group) cell (`patch.py:1295,1357`); ablation per condition
(`adapter_delta.py:520`); capture per task and axis per role/trajectory. The Deputy's reversal
of the millisecond-loop progress in lane C was right: R26(g) means units of work, not loops.
Payloads unchanged. Correct.

## Gate wording

The lanes were dispatched before the Director's amendment. Code is unaffected; the gate is
R15 condition 8 (checklist A8): lanes A and B committed before any training lift. The Deputy's
board state should read that way.

## Commit order

A, then B (after K1), then C and D; one commit per lane referencing its issue; Deputy pushes.
The R26 review pair moves to `complete/` after the four commits.
