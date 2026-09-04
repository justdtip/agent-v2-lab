# SPEC-002 evaluation, selection, and note integrity — review round 2 (retrospective), ratified (2026-09-05 01:40)

Reviewer: Claude (Chief). Evidence: a source-level conformance audit of every §1–§7 requirement
(111 tests green across `test_integrity.py`, `test_evaluate.py`, `test_pipeline.py`; fakes only),
the four implementation reports (parts 1–3, S4), round 1 (§3/§5 slice), and the Chief's direct
read of `integrity.py:200-295` (the six checks) and `evaluate.failure_reason`.

## Verdict: SPEC-002 is COMPLETE. All sections ratified; the spec and its reports move to complete/.

| Section | Status | Decisive evidence |
| --- | --- | --- |
| §1 selection screen | implemented | `tasks.py:59,96-169` difficulty override and `family_balanced_tasks`; `cli.py:466-575` components, Wilson, disagreement flag; `report.py` McNemar |
| §2 note integrity | implemented | `integrity.py:174-291`; six checks fire exactly once each on planted violations; `failure_reason` order parse → integrity kind → loop → exhausted → verdict; evaluator and rollout wired; `reports/note-integrity-B-vs-C.md` reproduces the memo (15 verbatim copies, 14 premature completions, 6 value drops) |
| §3 verdict hardening | implemented | `env.py:155,198,262-286` |
| §4 runner/evaluator | implemented | `runner.py:32,53-56`; `evaluate.py:23,137,345` |
| §5 eight defects | all fixed | cited per defect in the audit |
| §6/§7 outputs and acceptance | met | `evaluate.py:354,430-437`; tests named in the audit |

## Sanctioned deviations (not defects)

- Val-loss tie-break reads `metrics.jsonl` (R14), not the training log text.
- `Simulator`/`Verdict` tests remain in `tests/test_pipeline.py`; R10 applies when that lane
  is next touched.
- `GENERATOR_VERSION` is 4 by later specs; the SPEC-002 reports describe 2. Expected drift.

## Unverifiable without a model (carried to the B4 evaluation lift)

Live `--stress`, `think_tokens` on a real run, and val-loss selection on a real training run.

## Closes

Issues #2 and #7 (SPEC-002 items). Round 1's partial verdict is superseded.
