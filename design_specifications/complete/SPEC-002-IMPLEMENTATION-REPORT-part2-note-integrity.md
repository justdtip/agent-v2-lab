# SPEC-002 implementation report, part 2: note integrity

Status: implemented for SPEC-002 §2 only. Independent review round 1 is **CHANGES REQUIRED**;
this remediation round is pending re-review. SPEC-002 and issue #2 remain open.

## Scope delivered

This change implements the model-free note-integrity vertical: deterministic task reconstruction,
ground-truth carry derivation, six trajectory checks, evaluation annotations and summaries,
rollout eligibility filtering, backward-compatible summary rendering, and a retroactive offline
B-vs-C report. The reviewed package also contains the B1 prerequisite that adds explicit defaulted
`difficulty` and `integrity` fields to `Trajectory`; this report does not claim the rest of the
later selection, verdict-hardening, runner, or model-spec work in SPEC-002.

The claimed files at final verification were:

| Path | Final lines | Change |
| --- | ---: | --- |
| `src/local_llm_lab/pipeline/integrity.py` | 712 | pure scoring, focused helpers, and offline reporting |
| `src/local_llm_lab/pipeline/tasks.py` | 1,226 | difficulty override and exact task reconstruction |
| `src/local_llm_lab/pipeline/evaluate.py` | 397 | scoring, attribution, and helper-factored summaries |
| `src/local_llm_lab/pipeline/rollout.py` | 209 | integrity-clean supervision filter |
| `src/local_llm_lab/pipeline/report.py` | 59 | backward-compatible integrity-clean column |
| `src/local_llm_lab/pipeline/runner.py` | 353 | B1 prerequisite: explicit defaulted trajectory fields |
| `tests/test_integrity.py` | 610 | isolated, integration, and artifact-backed coverage |
| `tests/test_runner.py` | 37 | B1 field round-trip and backward-compatibility coverage |
| `pyproject.toml` | 68 | `agent-v2-integrity` entry point |
| `reports/note-integrity-B-vs-C.md` | 95 | authorised offline analysis, 3,687 bytes |
| `docs/superpowers/plans/2026-09-04-spec-002-note-integrity.md` | 363 | execution plan and checklist |
| this report | 205 | implementation evidence |

## Interfaces and wiring

Wiring-map §2.4 is implemented by retaining `difficulty`, adding its optional override, resolving a
single difficulty per generated task, recording it on the backward-compatible final `Task` field,
and adding the exact normative signature
`task_from_id(task_id: str, seed: int, difficulty: int | None = None)`. Reconstruction requires an
explicit seed, accepts difficulty as the third positional-or-keyword argument, parses from the right
so hyphenated splits work, restores the exact `v2:{seed}:{split}:{index}` RNG stream, calls one
maker, applies the named variant from the post-maker RNG state, and rejects malformed, unknown, or
structurally impossible IDs with a task-named error. Existing IDs and generator hashes remain
unchanged.

Wiring-map §2.9 is implemented with the exact frozen `Fact`, `Violation`, and `IntegrityReport`
records and the required public functions. Consumers are `evaluate.evaluate_tasks`,
`evaluate.failure_reason`, `rollout.collect_rollouts`, the offline CLI, and the new tests. The CLI
entry is registered in `pyproject.toml`.

The reviewed B1 prerequisite closes the touched §4 wiring gap by declaring
`Trajectory.difficulty: int = -1` and `Trajectory.integrity: dict[str, Any]` explicitly; both fields
round-trip through `as_dict()` and `Trajectory(**record)`. `evaluate_tasks` and `collect_rollouts`
still assign resolved values after `run_task`, because the runner creates a trajectory before the
consumer performs integrity scoring. `write_report` explicitly copies those values into serialised
records; with B1 this is redundant for new trajectories but preserves the consumer boundary and old
test doubles without mutating the source. `summarize` consumes the explicit integrity record.
`report.render` shows `integrity-clean` whenever at least one new summary is present, prints `-` for
legacy summaries in a mixed table, and preserves the historical layout when every input is legacy.
`run_evaluation` records `data_seed`.

The offline reader treats exactly integer `difficulty=-1` as the B1 legacy/missing sentinel and
passes `None` to `task_from_id`, which uses the split mapping. Other negative values and non-integers
remain invalid. The compatibility cost is intentional conflation of an absent difficulty field and
a newly serialised default sentinel; neither carries a resolved difficulty, while real evaluations
continue to record a non-negative value.

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

Review remediation split fact parsing into scalar/listing/worker helpers, trajectory evaluation into
one helper per ordered check, summary aggregation into totals/failure/integrity/group helpers, and
Markdown rendering into section-row builders. Public interfaces, check order, classifications, and
the 3,687-byte report are unchanged. A strict maximum-complexity-9 C901 run passes the two files
containing the four review-target functions.

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

GREEN evidence before the first review package:

- task prerequisite selection: 12 passed;
- generator regression guard: `tests/test_tasks.py`, 4 passed;
- required-carry selection: 7 passed;
- six-check selection: 4 passed;
- evaluator/report selection: 5 passed;
- rollout selection: 1 passed;
- CLI and retroactive selection: 4 passed;
- combined `tests/test_integrity.py tests/test_tasks.py`: 40 passed;
- scoped Ruff: all checks passed.

N1 then strengthened `verbatim_copy` to assert the exact isolated dictionary. Its original copied
ledger note correctly double-fired `count_mismatch`; RED observed both kinds. Two adjacent neutral
invoice-reading notes now produce exactly `{"verbatim_copy": 1}` without production suppression.

B1 added the explicit trajectory fields. The resulting three RED failures exposed one obsolete
absence assertion and synthetic records containing the `difficulty=-1` sentinel. The write-report
test now asserts explicit-field round trips; a named sentinel test failed before the offline reader
normalised `-1`, then passed. The post-B1 MLX-free focused suite was 43 tests before this remediation.

This remediation added the normative signature test. RED was
`TypeError: task_from_id() takes from 1 to 2 positional arguments but 3 were given`; GREEN accepts
the third positional difficulty and rejects an omitted seed. The current focused set is 44 tests.
Behavior-preserving helper extraction retained all 44 tests throughout. Default scoped Ruff and
strict maximum-complexity-9 C901 checks pass. In-memory offline rendering remains byte-identical to
the committed report, and the protected input hashes remain unchanged.

The authoritative fake-only full-suite result supplied at HEAD `4631e78` is **302 passed, 0 failed,
0 skipped in 4.80 seconds**. This bounded remediation lane did not rerun the broad suite because its
prohibited real-MLX collection path is outside scope.

The final command results, diff checks, banned-constant scan, commit, and status inspection are
recorded in the ignored task report.

## Deviations, ambiguities, and observed defects

- `Task.difficulty=-1` and the B1 `Trajectory.difficulty=-1` preserve constructor compatibility.
  Resolved generated/evaluated tasks still record a non-negative difficulty, and offline analysis
  interprets only the exact integer sentinel as legacy/missing.
- A table containing only legacy summaries retains its original columns; a mixed old/new table
  displays `integrity-clean` and `-` for legacy rows. This preserves existing callers while making
  the new metric visible.
- The offline report is 3,687 bytes, below R11's 1 MiB atomic-write threshold, so direct writing is
  permitted.

## Review history and immutable packages

Independent review round 1 examined immutable package `review-a854b57..8d72622.diff`, SHA-256
`170e968342946373f499b23754bd773dd470283cf47ee91d5d73361631410751`, over committed range
`a854b577e66cb61a30cc7083a14dc0b5d13eaa0b..8d72622199732dc4d144bc3ba12ddc3f53fbf1b9`.
It included commits `ed9136a`, `fa8f201`, the relevant B1 runner/test-runner hunks from `fbfe714`,
and `8d72622`, while excluding unrelated probe/J-lens work. Verdict: **CHANGES REQUIRED** for the
signature, four complexity ceilings, and this report's stale B1/follow-up description.

This remediation addresses all three findings and is pending a new immutable package and independent
re-review. No approval is claimed here.

Pending specifications and protected `data/` and `outputs/` inputs were not modified. No model,
checkpoint, tokenizer, MLX, probe, preflight, or J-space workload ran. All test doubles were local
fakes and all artifact analysis was offline JSON/simulator work.
