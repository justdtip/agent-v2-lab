# SPEC-002 implementation report, part 3: selection

Status: implementation complete; independent principal reviews approved, including the issue #7
provenance follow-up through review round 2 with no findings.

## Scope

This change implements SPEC-002 section 1 only: deterministic family-balanced checkpoint
screen construction, difficulty-aware evaluation metadata, Wilson intervals, exact paired
McNemar comparisons, two-cell selector configuration, validation-loss parsing, and report
rendering. It does not modify note-integrity scoring, generator task rows, runner behavior, or
protected data/output/report artifacts. R10 behavior-preservingly moved the legacy empty
checkpoint regression from `tests/test_pipeline.py` to `tests/test_selection.py`, then moved
the legacy report-table regression to `tests/test_report.py` and updated only its rich Wilson
interval expectations.

## Changed paths and interfaces

| Path | Change |
| --- | --- |
| `configs/agent_v2*.yaml` | replaces legacy selection limit/split with exact `valid`/1 and `valid2`/2 screen cells |
| `pipeline/tasks.py` | exposes `LONG_HORIZON_FAMILIES` and a pure clean-task quota selector |
| `pipeline/evaluate.py` | adds `wilson`, `mcnemar`, exact rate counts/intervals, difficulty grouping, and screen-aware evaluation metadata |
| `pipeline/cli.py` | parses saved `train.log` validation losses and writes deterministic per-checkpoint selection components |
| `pipeline/report.py` | retains lightweight outcomes for safe pairing and renders supplied intervals/paired McNemar rows |
| `tests/test_tasks.py`, `test_evaluate.py`, `test_cli.py`, `test_selection.py`, `test_report.py` | focused contracts using fakes and temporary metadata only |
| `tests/test_pipeline.py` (R10 extractions only) | removed the exact empty-checkpoint test to `tests/test_selection.py` and report-table test to `tests/test_report.py` |

Final line counts: configs 54/66/68; `tasks.py` 1,299; `evaluate.py` 482; `cli.py` 637;
`report.py` 138; `test_tasks.py` 265; `test_evaluate.py` 182; `test_cli.py` 169;
`test_selection.py` 147; and `test_report.py` 380.

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
`pipeline/cli.py::main` (17) and `pipeline/tasks.py::_wrong_old` (10). The earlier
`.venv/bin/python -m pytest -q` collection abort in `tests/test_jlens.py` is retained as
historical local-environment evidence. It is superseded for final verification by the successful
pinned fake-only `uv` gate recorded below.

## Safety and review status

No model, checkpoint, tokenizer, or training/evaluation/select stage was run. Test doubles used
temporary adapter directories and saved log text only. No protected `data/`, `outputs/`, or
`reports/` path was modified. The behavior-preserving extraction commit is
`8e7783977fc216568dc240ad9f426e2100c2ac59`; the selection implementation commit is
`72140033ef2f83badf0f8ba1ed3da0dfcb051eb5`. Observed-but-not-fixed items are the two existing C901
violations; the earlier `.venv` collection result is historical rather than the final gate.

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
ignored task ledger. The round-2 fix commit is
`87181f0c308a86577f3b5c5e939da6bb0096d25f`.

## Integration correction

The final R10 extraction moves `test_report_table_lists_runs_and_families` subtractively from
`tests/test_pipeline.py` to `tests/test_report.py`. It retains the run, family, and header
assertions while asserting rich summary cells: success, schema validity, and executable calls
show their required Wilson intervals. The full fake-only `uv run pytest -q` gate passes.
The integration correction commit is `2607e02db5d2ec4867cc09087c5e8a9cc78a0f56`.

## Review-fix round 3

Pairing now rejects every negative per-trajectory difficulty, including the `-1` unset
sentinel. The guard regressions use count-valid one-record cohorts where only the named
identity, split, seed, difficulty, or outcome property varies; malformed-record tests retain
the extra ID-less and non-dict cases to prove silent skipping cannot manufacture a pair.
The round-3 fix commit is `72b1e57d6514f6acb5904e4fa82c4b27799c39b0`.

## Review-fix round 4

The boolean `summary.tasks` guard now has a count-valid one-trajectory regression: without its
explicit boolean rejection, `len(outcomes) == True` would pair the cohorts. Missing and null
data-seed cohorts likewise retain otherwise matching non-empty outcomes, proving the non-null
seed guard is independently required. These are test-strengthening changes only. The round-4
commit is `67c52f36c85c3db8f302ee9992274b9712881c4a`; independent principal review approved this
round with no findings.

## Final controller verification

- Final authoritative fake-only gate: `uv run pytest -o addopts='' -q` — `355 passed, 1 xfailed
  in 4.75s`.
- Focused selection/neighbor set: 76 passed. Restored checkpoint plus pinned-hash nodes: 2
  passed. Scoped Ruff is clean; every exact task commit and scoped net-diff check is clean.
- Strict C901 remains at its pre-task-base values only: `pipeline/cli.py::main` = 17 and
  `pipeline/tasks.py::_wrong_old` = 10.
- Exact task commits: `8e7783977fc216568dc240ad9f426e2100c2ac59`,
  `72140033ef2f83badf0f8ba1ed3da0dfcb051eb5`,
  `ab1c38df649bde1c705cb85c3504515efa301b5b`,
  `87181f0c308a86577f3b5c5e939da6bb0096d25f`,
  `2607e02db5d2ec4867cc09087c5e8a9cc78a0f56`,
  `72b1e57d6514f6acb5904e4fa82c4b27799c39b0`, and
  `67c52f36c85c3db8f302ee9992274b9712881c4a`.

## Issue #7 `stage_select` provenance follow-up

Root formally released this bounded follow-up to `pipeline/cli.py` and `tests/test_cli.py` only.
Its fake-only RED regression showed that `stage_select` made no provenance call. The accepted
implementation writes `selection.json` first, then calls
`write_provenance(output, resolved=None, spec=load_model_spec(config["model"]),
extra={"stage": "select", "selection": json.loads(selection_path.read_text(encoding="utf-8"))})`;
the parsed durable JSON is therefore the exact provenance selection payload. Code commit:
`c73bfed74b5c565486dfe0da6ec82b25d5c6ff34`.

Review round 1 found one Medium test-oracle gap: the model-spec fake did not verify its lookup
argument. The mutation-resistant, test-only fix records and asserts the single exact
`config["model"]` lookup; commit `d510c265141ae920344622c0442b11bb5605ae98`. The same reviewer
approved review round 2 with no findings. Controller verification recorded 10 focused fake-only
tests passed, scoped Ruff and commit/path checks clean, and strict C901 limited to inherited
`pipeline/cli.py::main` = 17.

The full fake-only gate was not green: it reported `411 passed, 10 failed in 6.79s`. All ten
failures were the already documented external SPEC-003, probe, or protected-output failures
outside these two paths. Exact deselection confirmed `411 passed, 10 deselected in 5.58s`.
No model, checkpoint, tokenizer, probe, preflight, cache-equivalence, J-space workload, protected
artifact, or pending document was changed by this follow-up.
