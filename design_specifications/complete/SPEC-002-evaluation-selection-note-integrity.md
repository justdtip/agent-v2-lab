# SPEC-002: Evaluation, checkpoint selection, and note-integrity diagnostics

> Read first: `01-IMPLEMENTER-BRIEFING.md` (standing rules, traps, hand-off) and `02-INTERFACE-AND-WIRING-MAP.md` (exact shared signatures, file ownership, implementation order, integration checks). Signatures in the wiring map override any looser wording here.

Status: COMPLETE (ratified 2026-09-05, review round 2; sanctioned deviations listed there). Author: Claude. Date: 2026-09-03.
Depends on: SPEC-001 for the `ModelSpec` plumbing only; §2 and §5 can be implemented first
against the current tree and run today over saved outputs. GPU: none for implementation and for
the retroactive analysis in §2.4; the new screen runs only inside `select`, which is gated.

## 0. Why

The v2c post-mortem (decision memo §2.5) shows every failure is a note-writing failure and that
the current instruments hid it: the 18-task screen scored 16/18 while two families sat at 0/15,
validation loss disagreed with the screen, and `failure_reasons` attributes a dropped value as
`wrong answer` or `repetition loop`. This spec makes note integrity a first-class, ground-truth
metric, rebalances selection, and hardens the verdict.

## 1. Selection screen

- Generator: `make_tasks(split, count, seed, *, difficulty=None)` gains an explicit difficulty
  override; when `None` the current mapping (`train 0, valid 1, test 2, else index % 2`) holds.
  Config `select.screen` replaces `select.limit`:

  ```yaml
  select:
    screen:
      - {split: valid,  difficulty: 1, per_family: {default: 1, long: 3}}
      - {split: valid2, difficulty: 2, per_family: {default: 1, long: 3}}
  ```
  where `long` = the six long-horizon families. This yields 2 × (6 + 18) = 48 tasks per
  checkpoint, about 12 minutes at the 3B model's measured 15 s per task, versus 4 minutes today.
  Split names are new (`valid2`), so isolation from train and test holds by construction; a test
  asserts zero prompt and path overlap across `train, valid, valid2, test` at production sizes.
- Score: family-balanced success (macro over families) first, then micro success, then clean
  rate, then valid-action rate, then **lower validation loss** parsed from the training log (or
  from `adapter_config`/metrics when mlx-lm exposes it), then earlier step. `selection.json`
  records every component, the Wilson 95% interval of micro success, and the val-loss/behaviour
  disagreement flag.
- `report` gains paired McNemar p-values between any two evaluations on the same task ids and
  Wilson intervals on every rate.

## 2. Note-integrity diagnostics

New module `src/local_llm_lab/pipeline/integrity.py` and CLI `agent-v2-integrity`.

### 2.1 Ground truth

For any trajectory the generator can rebuild its `Task` from `task_id` (split, family, index,
variant) plus the data seed. From `task.steps` and the simulator replay, derive per expert step
the **required carry set**: facts observed at step `i` that a later supervised step `j` uses and
that are hidden by windowing at `j` (the same relation `test_every_grounded_argument_is_derivable_from_its_context`
already computes for arguments, extended to values: approved amounts, metric values, loads,
`Next-Key`, worker queue entries, the running total, and file names in `pending:`).

### 2.2 Per-step checks on the policy's notes

| Check | Definition | Family extractors |
| --- | --- | --- |
| `verbatim_copy` | note identical to the previous note after whitespace normalisation | all |
| `value_drop` | a required-carry value that appears in no visible observation and not in this note | ledger, aggregate, conditional, batch, cross_reference |
| `premature_completion` | note matches a completion pattern while ground-truth remaining count > 0; patterns: `(full)`, `(final)`, `complete`, `all \d+ .* (read|inspected|applied|verified)`, `pending: none`, `Task complete` | all |
| `count_mismatch` | `(\d+) of (\d+)` where the second number differs from the true total, or the first differs from the true progress | listing-driven families |
| `stale_fact` | a stated value contradicts ground truth (e.g. `highest so far: service-0=97` when a higher load was already observed) | conditional, ledger, aggregate |
| `queue_loss` | the enumerated remaining set in the note is a strict subset of the true remaining set | batch, ledger, aggregate |

Each check yields `{step, kind, detail}`; a trajectory gets `integrity.first_violation` (earliest
by step), `integrity.counts`, and `integrity.clean` (no violations). The evaluator's
`failure_reason` order becomes: parse error, then **note integrity kind of the first violation**
(e.g. `value drop`, `premature completion`, `verbatim copy`), then repetition loop, budget
exhausted, verdict reason. `summarize` reports integrity counts per family and the fraction of
failures explained by an integrity violation.

### 2.3 Where it runs

Inside `evaluate_tasks` after each trajectory (cost: string matching only) and in `rollout` as an
additional keep filter (`integrity.clean` required, closing the outcome-only gap noted in the
pipeline review).

### 2.4 Retroactive analysis (no model, do now)

`agent-v2-integrity --eval outputs/agent-v2b/evals/runB-test180.json --eval outputs/agent-v2c/evals/best-adapter-test.json --output reports/note-integrity-B-vs-C.md`
rebuilds tasks by id, scores every saved trajectory, and writes per-family tables plus paired
flips. Acceptance: reproduces the decision-memo counts (15 verbatim copies in C cross_reference;
14 batch_update premature completions; 6 ledger value drops with the listed differences).

## 3. Verdict hardening (`pipeline/env.py`)

- `Simulator.verdict` compares every file not in `expected_files` with the task's initial
  `files`; any difference yields `unexpected file change: <name>`. Closes the `conditional_update`
  (throttle everything) and `batch_update` (`unmanaged.ini`) holes found in review.
- `required_tools` are satisfied only by calls that executed without error.
- Answer normalisation additionally strips one trailing `.` and surrounding backticks; the raw
  string is retained in the verdict for audit.
- `update` prompt phrasing 2 and all `batch_update` prompts are reworded so the expected answer
  is not verbatim in the prompt (mirror the `conditional_update` trick).

## 4. Runner and evaluator

- `detect_loop` rule 2 threshold: 6 same-tool errors → 4.
- `--stress` gets a unit test; add `--stress` and `--base` to the 180-task `all` stage.
- `Trajectory` records `thinking` and `think_tokens` (SPEC-001 §4) and `integrity`.
- Latency and token metrics unchanged; add `think_tokens_per_task`.

## 5. Known defects to fix (from the pipeline review)

| Location | Defect | Fix |
| --- | --- | --- |
| `cli.py::stage_select` | `max()` on empty checkpoint list | explicit `SystemExit` message |
| `cli.py::checkpoint_dirs` | stale `adapters.safetensors` reused across runs | compare SHA-256 of source and target; re-copy on mismatch; clear `checkpoints/` on `train` |
| `transcript.py` | `transcripts.jsonl` appended across reruns | truncate at `Transcript.start` of a new run (run id in the header line) |
| `branch.py` | `chosen` re-rendered canonically, `rejected` raw | use `step["raw"]` for `chosen` |
| `tasks.py::_wrong_path` | mangled path not checked for absence | guard like `_stale_guess` |
| `tasks.py::_wrong_path/_stale_path` | failing step keeps the note naming the correct path | rewrite the failing step's note to state the guessed path (SPEC-003 §2 owns the template text) |
| `tasks.py::applicable_variants` | probed at level 0 only | probe at the requested level; cache per `(family, level)` |
| `evaluate.py` | no test of `run_evaluation`, `--stress`, `stage_select`, `rollout`, `Transcript` | add the tests listed in the review |

## 6. Outputs

`evals/<label>-<split>.json` gains `integrity` per trajectory and per family, `wilson_95`, and
the resolved `ModelSpec`. `selection.json` gains component scores and intervals.
`reports/note-integrity-B-vs-C.md` is produced by §2.4.

## 7. Acceptance

- Retroactive analysis reproduces the memo's counts.
- On synthetic trajectories, each check fires exactly on its planted violation and nowhere else.
- The new screen's family balance is asserted by a test; selection is deterministic given the
  same evaluations.
- Verdict tests cover the three holes above.
