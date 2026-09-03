# SPEC-002 implementation report, part 3: selection

Status: implementation complete; independent review is pending until the reviewer returns.

## Scope

This change implements SPEC-002 section 1 only: deterministic family-balanced checkpoint
screen construction, difficulty-aware evaluation metadata, Wilson intervals, exact paired
McNemar comparisons, two-cell selector configuration, validation-loss parsing, and report
rendering. It does not modify note-integrity scoring, generator task rows, runner behavior, or
protected data/output/report artifacts. Step 0 behavior-preservingly moved the legacy empty
checkpoint regression from `tests/test_pipeline.py` to `tests/test_selection.py`.

## Changed paths and interfaces

| Path | Change |
| --- | --- |
| `configs/agent_v2*.yaml` | replaces legacy selection limit/split with exact `valid`/1 and `valid2`/2 screen cells |
| `pipeline/tasks.py` | exposes `LONG_HORIZON_FAMILIES` and a pure clean-task quota selector |
| `pipeline/evaluate.py` | adds `wilson`, `mcnemar`, exact rate counts/intervals, difficulty grouping, and screen-aware evaluation metadata |
| `pipeline/cli.py` | parses saved `train.log` validation losses and writes deterministic per-checkpoint selection components |
| `pipeline/report.py` | retains lightweight outcomes for safe pairing and renders supplied intervals/paired McNemar rows |
| `tests/test_tasks.py`, `test_evaluate.py`, `test_cli.py`, `test_selection.py`, `test_report.py` | focused contracts using fakes and temporary metadata only |
| `tests/test_pipeline.py` (Step 0 only) | removed the exact empty-checkpoint test after its behavior-preserving move to `tests/test_selection.py` |

Final line counts: configs 54/66/68; `tasks.py` 1,264; `evaluate.py` 482; `cli.py` 595;
`report.py` 137; `test_tasks.py` 152; `test_evaluate.py` 182; `test_cli.py` 20;
`test_selection.py` 147; and `test_report.py` 227.

Wiring-map rows touched are 2.4 (`Task` consumption only; frozen record and exact
`task_from_id` signature retained), 2.10 (evaluation summary and screen arguments), 2.11
(configured selector), plus integration check 9 (legacy/rich summary compatibility).

## Selection artifact schema

`selection.json` contains `selected_step`, `behavior_best_step`, `loss_best_step`, global
boolean `disagreement`, `model`, `data_seed`, exact `screen` definitions, and `checkpoints`.
Each checkpoint records its source adapter, `components` (family macro, micro, clean,
valid-action, exact task/success counts, and combined family counts), micro-success `wilson_95`,
and nullable `val_loss`. Ranking is macro success, micro success, clean rate, valid-action rate,
lower validation loss, then earlier step.

## Compatibility and decisions

- The generator remains version 2. The helper delegates to `make_tasks(..., perturb=False)` and
  does not modify makers, RNG derivation, task IDs, or rows; current hash-oracle tests pass.
- `stage_select(config, limit, quiet)` remains callable. A supplied `--limit` caps every cell;
  absent it retains the configured 24+24 screen.
- New summary rate records preserve existing rounded rates and note-integrity fields. Exact
  numerator/denominator values feed selection, while intervals use the standard uncorrected
  Wilson formula.
- McNemar only pairs non-empty identical per-trajectory `(task_id, difficulty)` sets with the
  same non-null data seed and split. Legacy records have no outcomes/metadata and therefore
  cannot be fabricated into pairs.
- The supplied Wilson expected value for 5/10 differs from the stated `z=1.96` formula by about
  3.4e-6. The formula governs: implementation returns `(0.2365895936, 0.7634104064)`.

## TDD and verification

RED checkpoints observed missing task selector symbols; missing Wilson/rate metadata/model-RNG
boundary; legacy config/selector fields; and interval/pair-less rendering. GREEN evidence:

- `tests/test_tasks.py`: 6 passed, including version-2 generator hash oracles.
- `tests/test_evaluate.py tests/test_integrity.py`: 41 passed.
- `tests/test_cli.py tests/test_selection.py tests/test_evaluate.py tests/test_tasks.py`: 14 passed.
- `tests/test_tasks.py tests/test_evaluate.py tests/test_cli.py tests/test_report.py tests/test_selection.py tests/test_integrity.py tests/test_runner.py`: 57 passed.
- Scoped Ruff over all changed pipeline/test modules: passed.
- `git diff --check` over owned source/config/test paths: passed.
- The banned-constant scan has one existing generator-range hit (`tasks.py:402`, `35`), not a
  model-layout literal and outside the new selector code; no introduced prohibited constants.

The requested strict C901 max-9 command reports two pre-existing out-of-scope functions:
`pipeline/cli.py::main` (17) and `pipeline/tasks.py::_wrong_old` (10). The fake-only full suite
could not collect locally: `.venv/bin/python -m pytest -q` aborts in `tests/test_jlens.py` before
test execution. Both conditions are recorded rather than hidden or fixed outside scope.

## Safety and pending review

No model, checkpoint, tokenizer, or training/evaluation/select stage was run. Test doubles used
temporary adapter directories and saved log text only. No protected `data/`, `outputs/`, or
`reports/` path was modified. The behavior-preserving extraction commit is
`8e7783977fc216568dc240ad9f426e2100c2ac59`; the selection implementation commit is
`72140033ef2f83badf0f8ba1ed3da0dfcb051eb5`. Observed-but-not-fixed items are the two existing C901
violations and the local full-suite collection abort above.

## Review-fix round 1

The formal screen ruling preserves exactly `{default: 1, long: 3}` per cell: `valid` at
difficulty 1 and `valid2` at difficulty 2, for 48 total tasks and combined short/long family
counts of 2/6. The screen is deliberately long-horizon weighted; family balance is the primary
macro score, not equal sample counts.

The accepted review fixes add fail-closed paired comparison identity (same non-null `data_seed`
and exact per-trajectory task-id/difficulty keys), truthful mixed-difficulty evaluation metadata,
the top-level integrity-clean Wilson interval in the nested report column, and restoration of
the extracted empty-checkpoint regression in `tests/test_selection.py`. RED and GREEN evidence
for all four focused regressions is recorded in the ignored task ledger. The round-1 fix commit
is `ab1c38df649bde1c705cb85c3504515efa301b5b`.

## Review-fix round 2

`load_summaries` now fails closed for an entire comparison cohort when any trajectory is
non-dict, ID-less, has an invalid difficulty, duplicates an identity, omits its verdict outcome,
or supplies a non-boolean outcome. It also accepts outcomes only when their unique exact
`(task_id, difficulty)` count equals a non-boolean, non-negative integer summary `tasks` value.
Matching cohorts remain pairable across different model/adapter metadata when their split,
non-null data seed, and exact per-task identities agree. RED/GREEN evidence is recorded in the
ignored task ledger; the commit hash is recorded by the subsequent documentation handoff.
