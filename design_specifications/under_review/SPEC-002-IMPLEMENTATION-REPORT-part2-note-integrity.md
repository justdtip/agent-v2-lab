# SPEC-002 implementation report, part 2: note integrity

Status: implemented for SPEC-002 §2 only. SPEC-002 and issue #2 remain open.

## Scope delivered

This change implements the model-free note-integrity vertical: deterministic task reconstruction,
ground-truth carry derivation, six trajectory checks, evaluation annotations and summaries,
rollout eligibility filtering, backward-compatible summary rendering, and a retroactive offline
B-vs-C report. It deliberately does not implement the later selection, verdict-hardening, runner
dataclass, or model-spec work in SPEC-002.

The claimed files at final verification were:

| Path | Final lines | Change |
| --- | ---: | --- |
| `src/local_llm_lab/pipeline/integrity.py` | 564 | new pure scoring and offline-report module |
| `src/local_llm_lab/pipeline/tasks.py` | 1,227 | difficulty override and exact task reconstruction |
| `src/local_llm_lab/pipeline/evaluate.py` | 350 | scoring, attribution, summary, and JSON bridge |
| `src/local_llm_lab/pipeline/rollout.py` | 209 | integrity-clean supervision filter |
| `src/local_llm_lab/pipeline/report.py` | 59 | backward-compatible integrity-clean column |
| `tests/test_integrity.py` | 582 | new isolated and artifact-backed coverage |
| `pyproject.toml` | 68 | `agent-v2-integrity` entry point |
| `reports/note-integrity-B-vs-C.md` | 95 | authorised offline analysis, 3,687 bytes |
| `docs/superpowers/plans/2026-09-04-spec-002-note-integrity.md` | 363 | execution plan and checklist |
| this report | measured at final verification | implementation evidence |

## Interfaces and wiring

Wiring-map §2.4 is implemented by retaining `difficulty`, adding its optional override, resolving a
single difficulty per generated task, recording it on the backward-compatible final `Task` field,
and adding `task_from_id`. Reconstruction parses from the right so hyphenated splits work, restores
the exact `v2:{seed}:{split}:{index}` RNG stream, calls one maker, applies the named variant from the
post-maker RNG state, and rejects malformed, unknown, or structurally impossible IDs with an error
that names the task. Existing IDs and generator hashes remain unchanged.

Wiring-map §2.9 is implemented with the exact frozen `Fact`, `Violation`, and `IntegrityReport`
records and the required public functions. Consumers are `evaluate.evaluate_tasks`,
`evaluate.failure_reason`, `rollout.collect_rollouts`, the offline CLI, and the new tests. The CLI
entry is registered in `pyproject.toml`.

The touched §4 wiring gap is intentionally bridged without taking ownership of SPEC-002 §4's future
`Trajectory` fields. `evaluate_tasks` and `collect_rollouts` attach dynamic `difficulty` and
`integrity` attributes after `run_task`; `write_report` merges them into copied serialised records.
`summarize` consumes the dynamic integrity record. `report.render` shows `integrity-clean` whenever
at least one new summary is present, prints `-` for legacy summaries in a mixed table, and preserves
the historical layout when every input is legacy. `run_evaluation` now records `data_seed`.

## Ground truth and checks

`required_carry` replays only deterministic expert actions in `Simulator`, stores non-finish
observations, applies the exact `keep_last` visibility boundary, and emits referenced hidden facts
with their producing action index. Extractors cover approved amounts (not held amounts), metric
values, service/load pairs, next keys, calculator totals, listed basenames, and normalised worker
manifest entries. Numeric references use digit boundaries and filenames use basenames.

Checks run once per policy step in the required order:

1. `verbatim_copy` compares adjacent whitespace-normalised notes.
2. `value_drop` requires hidden scalar facts unless the exact fact remains in a visible actual
   observation; path and worker enumeration belongs to `queue_loss`.
3. `premature_completion` compares completion claims with the canonical phase at the same valid
   index. Equivalent canonical phase-completion wording licenses generic `complete`; a completion
   after the canonical horizon is not premature.
4. `count_mismatch` compares every stated `x of y` pair with the same-index canonical pair.
5. `stale_fact` compares stated `approved`, `first half`, `second half`, and `highest so far`
   fields. A deletion remains `value_drop`; an introduced contradictory token is `stale_fact`.
6. `queue_loss` flags a strict subset of canonical basenames in `pending:` or `Remaining after
   this:` clauses.

Malformed saved records are scored without crashing. Violations are sorted by policy step and
check order; counts retain repeated step-level instances, while the retroactive report counts each
kind once per affected trajectory.

## Evaluation, rollout, and report behaviour

`failure_reason` orders parse error, first integrity kind with underscores rendered as spaces,
loop, exhaustion, then verifier reason. `summarize` reports clean/affected trajectory totals,
step-level violation totals, affected trajectories by kind, per-family clean rates and counts, and
a zero-safe failed-trajectory-only `failure_explained_rate`.

Rollouts retain all samples in diagnostics and pass every sample into summary metrics. Only an
outcome-successful, integrity-clean sample enters duplicate-signature selection and contributes
training rows. A task is solved for retained-supervision `pass_at_k` only if it has an eligible
candidate.

## Retroactive B-vs-C result

The authorised CLI read the two existing evaluation JSON files in place and generated
`reports/note-integrity-B-vs-C.md`. Input SHA-256 values were identical before and after:

- B: `ce0d8f4eade894389688eba250090bba24c8ef68b01c8ca7f0a87694c98ef630`
- C: `2cabb9a40a644d2dd5a6688b48dd769bd99af684fad4b07ed01507e426668ddc`

Outcome success was reconstructed from raw trajectory verdicts: cross-reference changed
14/15→0/15 and ledger reconcile 13/15→9/15. C failure counts were aggregate report 15, batch
update 14, conditional update 12, cross-reference 15, and ledger reconcile 6. All three memo
acceptance rows passed: C cross-reference `verbatim_copy=15`, C batch-update
`premature_completion=14`, and C ledger-reconcile `value_drop=6`. Overall integrity-clean counts
were B 122/180 and C 105/180; 34/35 B failures and 62/62 C failures carried a violation.

The only calibration adjustment was evidence-led and general: the sole apparent fifteenth C batch
premature completion was a successful 18-step trajectory that completed after the 14-step canonical
horizon. Its final completion is temporally valid; treating at-or-after-horizon completion as
licensed produced the specified 14 affected failed trajectories without task-ID or run-label
special-casing.

## TDD and verification evidence

RED evidence captured during implementation:

- task prerequisite selection: collection error importing missing `task_from_id` (exit 2);
- ground-truth API selection: `ModuleNotFoundError` for `pipeline.integrity` (exit 2);
- six-check selection: four failing assertions against the initial clean stub (exit 1), then
  root-cause corrections for action-argument masking and filename sentence boundaries;
- evaluator/report selection: five expected failures for absent annotations, attribution, summary,
  serialisation, and column (exit 1);
- rollout selection: the violating successful candidate incorrectly supplied the retained row
  (exit 1);
- CLI selection: entry-point registration was absent while the pure helper tests passed (exit 1).

GREEN evidence before final verification:

- task prerequisite selection: 12 passed;
- generator regression guard: `tests/test_tasks.py`, 4 passed;
- required-carry selection: 7 passed;
- six-check selection: 4 passed;
- evaluator/report selection: 5 passed;
- rollout selection: 1 passed;
- CLI and retroactive selection: 4 passed;
- combined `tests/test_integrity.py tests/test_tasks.py`: 40 passed;
- scoped Ruff: all checks passed.

The broader `tests/test_pipeline.py` run passed 83 tests and had one failure in
`test_jlens_map_averages_and_matches_manual_mean`: the test still calls the removed concurrent
`jlens_map(..., stats=...)` API. This is outside the claimed paths and was not altered here. The two
report compatibility failures first exposed by that run were fixed and their focused rerun passed
3 tests across old and new coverage.

The final command results, diff checks, banned-constant scan, commit, and status inspection are
recorded in the ignored task report.

## Deviations, ambiguities, and observed defects

- The wiring map's future `Task.difficulty` has no default and future `Trajectory` owns explicit
  fields. This bounded §2 implementation follows the task brief: a final `Task` default of `-1`
  preserves construction compatibility, and dynamic trajectory attributes bridge until §4.
- A table containing only legacy summaries retains its original columns; a mixed old/new table
  displays `integrity-clean` and `-` for legacy rows. This preserves existing callers while making
  the new metric visible.
- The offline report is 3,687 bytes, below R11's 1 MiB atomic-write threshold, so direct writing is
  permitted.
- The unrelated `jlens_map(..., stats=...)` regression remains observed but not fixed.

Pending specifications and protected `data/` and `outputs/` inputs were not modified. No model,
checkpoint, tokenizer, MLX, probe, preflight, or J-space workload ran. All test doubles were local
fakes and all artifact analysis was offline JSON/simulator work.
