# ADAPTER-DEPTH-QUALITY-2026-09-08 — arm 1's three checkpoints on the divergence tasks

**Question (the Director's, through the Chief): which checkpoint of arm 1 is best?**

**Answer: 800 rows.** It passes 18 of the 19 tasks the divergence record used, against 13 at 400
and 18 at 1,200; it ties 1,200 on the count and wins on the kind of its single failure. The base
passes 15 of the same 19 and arm A's adapter passed 5.

| policy | passes /19 | base-pass, adapter-fail | adapter-pass, base-fail | locked in | integrity violations | turns |
|---|---|---|---|---|---|---|
| base (recorded, reused) | 15 | — | — | — | 39 in 3 trajectories | — |
| arm A @ 800 rows | 5 | 11 | 1 | 12 of 14 | — | — |
| arm 1 @ 400 rows | 13 | 4 | 2 | 0 of 6 | 28 in 5 | 93 |
| **arm 1 @ 800 rows** | **18** | **1** | **4** | **0 of 1** | **9 in 3** | **112** |
| arm 1 @ 1,200 rows | 18 | 1 | 4 | 1 of 1 | 21 in 1 | 133 |

Arm 1 is `configs/agent_v2e_qwen35_4b_top8.yaml`: arm A's recipe with `train.lora_layers: 8`, the
adapter restricted to the top 8 of 32 layers. It is the first adapter in this programme to beat
its own base on these tasks. Full depth at the same recipe produced one markedly worse.

## What was run, and what was not

Three evaluations, `--split test --limit 19`, greedy (`temperature` defaults to 0.0 in
`evaluate.py`), in the order the Chief set: 800, then 1,200, then 400. One window, 23:10Z to
23:28Z, about five minutes each — not the 2.6 hours estimated from arm A's 180-task run, because
arm A's adapter spent almost every task looping to the 24-step ceiling and arm 1's does not.

**The base was not rerun.** Its greedy results on these tasks are the first 19 trajectories of
`outputs/agent-v2/evals/base-test.json`, the same rows the divergence record used; reading them
back gives 15 of 19, which is the record's figure.

**The 19 tasks are the first 19 of the test split**, `test-read-0000-clean` through
`test-pointer_chain-0018-clean`, so `--limit 19` reproduces the divergence record's set exactly
rather than approximating it.

`mlx_lm.load_adapters` takes a **directory** holding `adapter_config.json` and
`adapters.safetensors`, never a bare checkpoint file; passing the file raises `NotADirectoryError`
on `<file>/adapter_config.json`. Each checkpoint is therefore presented as a directory under
`outputs/agent-v2e-qwen35-4b-top8/checkpoints/`, config copied and weights hard-linked. The
directory's name becomes `stage_eval`'s stem, so the three wrote to distinct `evals/*.json`
instead of overwriting one another.

## Why 800 and not 1,200

The counts tie at 18. The failures do not.

- **800's** one failure is `test-aggregate_report-0011-clean`, and it is not a reasoning failure:
  the model reads all six metrics, computes 276, writes `grand_total=276` into the report file,
  and then emits prose with no tool call where the `finish` should be. The task's work is done and
  its answer is correct; the trajectory has no repeated call and no loop.
- **1,200's** one failure is `test-update-0004-clean`, a verbatim re-issue of a call that had just
  succeeded, repeating to the 24-step ceiling. That is arm A's dominant defect — 5 of its 14
  failures and the shape of its 12 lock-ins — appearing in this adapter for the first time, and it
  carries all 21 of that checkpoint's integrity violations.

So the extra 400 rows do not buy a pass and do introduce the failure mode the depth restriction
was suppressing. 800 also finishes in fewer turns (112 against 133) and spreads its 9 integrity
violations across three trajectories rather than concentrating 21 in one.

## Defect classes, against arm A's six

Arm A's six were `malformed argument`, `re-issued call after success`, `path abbreviation`,
`wrong hop`, `state-note comparison error`, `wrong file`.

| checkpoint | classes seen |
|---|---|
| 400 | required tool unused ×3, truncated tool call ×2, malformed argument ×1 |
| 800 | truncated tool call ×1 |
| 1,200 | re-issued call after success ×1 |

**Two classes are new and are named rather than folded into arm A's six.** *Truncated tool call* is
a step that emitted no parseable action. *Required tool unused* is arithmetic done in the model's
head instead of through `calculate` — at 400 the model answers `852` where the answer is `568`
after a single read, and on one task reaches the right answer without ever calling the tool.

**Not one of the three checkpoints shows `path abbreviation`.** Every path they emit is rooted.
That class was 3 of arm A's 14 failures and it is gone.

## The classifier, and how far to trust it

`paired_tally.py` reduces arm A's markdown transcripts and arm 1's eval JSON to the same shape — a
list of (call, observation) pairs — so one rule runs over both. Applied to arm A it returns 5
passes, 11 base-pass/adapter-fail pairs and 12 of 14 failures locked in, which are the divergence
record's three headline figures, so the rule is checked against a hand reading before being used
on new data.

It agrees with arm A's hand labels on 13 of 14 tasks. The one it misses is
`test-aggregate_report-0011-clean`, which the record calls `wrong file`: the adapter skips
metric-4 for metric-5 at step 5, and detecting that needs the task's expected file list, which the
trajectory does not carry. Three of arm A's six are mechanical (unrooted path, non-arithmetic
expression, identical repeat after a good result); the other three need ground truth and are left
to a reader.

## Caveat on the margin

Nineteen tasks. The Wilson interval on 18 of 19 runs 0.754 to 0.991 and on the base's 15 of 19
runs 0.567 to 0.915, so **the margin over the base (18 against 15) is plausible, not established**.
The two policies disagree on five tasks, four the adapter wins and one the base wins, and the exact
paired test on those five gives p = 0.375. The margin over arm A
(18 against 5) is not in doubt. The full 180-task split on checkpoint 800 was ordered for exactly
this reason and is recorded separately when it lands.

## Files

- `evals/ckpt-*-test.json` — the three runs' trajectories, copied here because `outputs/` is not
  tracked.
- `paired_tally.py` — the reader and the rule; re-runs from a checkout with no `outputs/` tree.
- `paired_tally.out.txt`, `summary.json` — its output as run.
- `run-19.sh`, `run-full.sh` — the launching scripts, window announced and `end` trapped on exit.

---

## Appended: checkpoint 800 across the full 180-task test split

**159 of 180 against the base's 123.** The margin the nineteen tasks could only call plausible is
established here: 39 tasks the adapter wins, 3 the base wins, exact paired p = 5.6e-9, and the two
Wilson intervals no longer overlap (0.828 to 0.922 against 0.612 to 0.747).

Run 09:31 to 11:05 local, greedy, same harness, base reused from `outputs/agent-v2/evals/base-test.json`.

### Every family the base is weak at moves, and nothing regresses

| family | adapter | base | n |
|---|---|---|---|
| ledger_reconcile | 14 | 0 | 15 |
| batch_update | 15 | 1 | 15 |
| conditional_update | 5 | 1 | 15 |
| calculate | 15 | 13 | 15 |
| cross_reference | 15 | 13 | 15 |
| aggregate_report | 5 | 5 | 15 |
| read, search, synthesis, update, list, pointer_chain | 15 | 15 | 15 each |

All three of the base's wins are aggregate_report, which is also the only family the adapter fails
to improve. The six families the base already passes cleanly are untouched, so this is not a trade
of one capability for another.

### The training lowers every failure mode it was suspected of causing

| | base | arm 1 @ 800 |
|---|---|---|
| passes | 123 | 159 |
| failures | 57 | 21 |
| "no finish call" | 37 | 18 |
| exhausted | 32 | 7 |
| loop detected | 22 | 4 |
| integrity violations | 486 | 428 |
| total turns | 1,714 | 1,372 |

**A correction to the 19-task reading above.** That section named *truncated tool call* as a class
new to arm 1, because the 800 checkpoint's single failure was one. On 180 tasks it is the **base's
own dominant failure**, 37 of its 57, and the adapter halves it. The class was new to those
nineteen tasks, not new to the model, and the earlier sentence should be read with this beside it.

### What is left

Twenty-one failures, in two families: aggregate_report with 10 and conditional_update with 10, plus
one ledger_reconcile. Eighteen of the 21 are "no finish call". So the residue is not a reasoning
failure spread across the suite; it is one behaviour, in the two families that need multi-step
aggregation, and it is the behaviour the base is worst at.

### On the timings, which are not comparable

A Director-authorized lens diagnostic held the box alongside this run from 09:43:47 to 23:55Z,
under a written registration that capped it at 6 GiB and preserved this run's lock and window. The
per-task time went from a 23.3 s mean over the first 32 tasks to 135 s and 263 s for the two that
straddled its load, and recovered afterwards. **The verdicts are unaffected**, because decoding is
greedy and deterministic; the elapsed times in this run are not comparable with the 19-task run's,
and `full_split_tally.py` reports no wall clock for that reason.

---

## Appended: checkpoint 1,200 across the full split, and a correction to this record's answer

**175 of 180. The best checkpoint is 1,200, not 800, and the one-line answer above is wrong.**

| on 180 tasks | base | 800 rows | 1,200 rows |
|---|---|---|---|
| passes | 123 | 159 | **175** |
| adapter wins / base wins | — | 39 / 3 | **53 / 1** |
| exact paired p | — | 5.6e-9 | **6.1e-15** |
| Wilson 95% | 0.612–0.747 | 0.828–0.922 | 0.937–0.988 |
| loop failures | 20 | 2 | 2 |
| exhausted | 32 | 7 | 2 |
| tool errors | 273 | 18 | 17 |
| clean rate | 0.683 | 0.872 | 0.972 |
| turns taken | 1,714 | 1,372 | 1,334 |

1,200 beats 800 on 20 tasks and loses on 4. It fixes the two families 800 could not: aggregate
report 5 → 14 where 800 stayed at 5, and conditional update 1 → 15 where 800 reached only 5. Its
five failures are three ledger reconciliations, one aggregate report and `test-update-0004-clean`.

### Why the earlier answer was wrong

Both checkpoints passed 18 of the divergence 19, and the tie was broken on the kind of the single
failure: 1,200's was a verbatim re-issue run to the step ceiling, read here as arm A's dominant
pathology returning at the later checkpoint.

On 180 tasks the `update` family is 14 of 15 for 1,200, and that task is one of five failures in
the whole split. The lock-in was real and isolated, not a pathology spreading.

**The standard was available and was not applied.** The section above says the nineteen tasks
could not establish a *margin* at p = 0.375, and the section after it ranks two checkpoints on a
*single task's* failure mode, which is weaker evidence still. The honest answer at the time was
that the two were not separated.

Read back on the full-split files, the nineteen tasks give base 15, 800 18, 1,200 18 — exactly the
figures published from the shorter runs. Nothing is inconsistent; the set simply could not
separate the two checkpoints.

### What stands

Everything about the depth restriction. The top-8 adapter beats its base decisively at both
checkpoints, no family regresses at 800, and every failure mode the training was suspected of
causing is lower than the base's at both. And more training on the restricted depth kept improving
monotonically — exhaustion 7 → 2, clean rate 0.872 → 0.972, turns still falling — so arm 1 was
still descending at 1,200 rows.
