# SPEC-002 implementation report, part 1

Date: 2026-09-03

## Reviewed slice

This report covers issue #2 review findings F1–F4 as one correction vertical following the
accepted SPEC-002 §3/§5 commits `fc38a9d` and `6dff90c`. Those accepted changes remain intact.

- F1: identify the current deterministic task generator as version 2 in dataset manifests and
  data-stage provenance, and pin the reference configuration's generated split hashes by
  generator version.
- F2: provide one transcript iterator that skips the `run_id` header and every other record
  without `task_id`.
- F3: make impossible `wrong_path` and `stale_path` construction fail with task-named errors
  instead of silently substituting another variant.
- F4: provide this partial-slice implementation report while leaving SPEC-002 pending.

## Files and callers

- `src/local_llm_lab/pipeline/tasks.py`: defines `GENERATOR_VERSION = 2` and raises task-named
  construction errors at impossible `_wrong_path` and `_stale_path` exits.
- `src/local_llm_lab/pipeline/data.py`: copies `GENERATOR_VERSION` into the top level of every
  returned and written dataset manifest.
- `src/local_llm_lab/pipeline/transcript.py`: exposes `iter_task_records(path)` for all future
  transcript consumers.
- `src/local_llm_lab/provenance.py`: adds the minimal deterministic SPEC-001-compatible
  `write_provenance(run_dir, *, resolved, spec, extra)` seam required by F1.
- `src/local_llm_lab/pipeline/cli.py`: `stage_data` is the changed caller; it writes provenance
  after dataset generation using registry metadata only, without resolving or loading weights.
- `tests/test_pipeline.py`: covers impossible recovery construction, manifest metadata,
  data-stage provenance, header-safe transcript iteration, and the version-keyed hash pin.
- `design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part1.md`: this hand-off
  record.
- `docs/superpowers/plans/2026-09-03-spec-002-review-corrections.md`: the reviewed execution plan.

## Decisions applied

- R5 assigns generator version 1 to commit `97d197c`, the last generator that reproduces run C.
  The current generator is version 2.
- Version 2's generator-owned task/expert rows are pinned to train
  `e7fa63ef9a2fe70b3563a23fa5421b4a6b11d11971802d2c4b6bce5a0a3c5d58`, valid
  `816543d1dee8299b2dbf514e93a2de480b69f84d6e8c53bda955c20c79f6213f`, and test
  `10042da5d9a4789f9a28fc6c0c9a8ea8efc59d090688f1e167c6de8d1de66877`. The oracle uses
  `configs/agent_v2c.yaml` for counts, seed, keep-last, and recovery repeats, with
  `chat_dir=None`.
- There is no cross-version run-C equality assertion. Row changes require an explicit generator
  version bump and a new current-version hash table entry.
- R6 is implemented by filtering all blank lines and all JSONL records lacking `task_id` through
  `iter_task_records`.
- Impossible recovery construction raises `RuntimeError` naming `task.task_id`; valid
  `wrong_path` and `stale_path` generation is unchanged.

## Verification

TDD recovery RED:

```text
uv run pytest -q tests/test_pipeline.py -k 'wrong_path or stale_path'
2 failed, 1 passed: both new tests failed with DID NOT RAISE RuntimeError.
```

TDD recovery GREEN:

```text
uv run pytest -q tests/test_pipeline.py -k 'wrong_path or stale_path'
3 passed.
```

TDD metadata/iterator RED:

```text
uv run pytest -q tests/test_pipeline.py -k 'generator_version or reference_generator or provenance or iter_task_records'
Collection failed because iter_task_records could not be imported.
```

TDD metadata/iterator GREEN:

```text
uv run pytest -q tests/test_pipeline.py -k 'generator_version or reference_generator or provenance or iter_task_records'
4 passed.
```

Final fake-only verification:

```text
uv run pytest -q tests/test_pipeline.py
87 passed (72 tests through 82%, then 15 through 100%); exit 0.

uv run ruff check src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/transcript.py src/local_llm_lab/pipeline/data.py src/local_llm_lab/pipeline/cli.py src/local_llm_lab/provenance.py tests/test_pipeline.py
All checks passed!; exit 0.

git diff --check
Exit 2 because of a pre-existing unrelated pending-spec modification:
design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md:470: new blank line at EOF.

git diff --cached --check
No output; exit 0.
```

The pending-spec whitespace belongs to work outside this task and was not changed. The staged
set contains exactly the eight intended product paths.

## Deliberately deferred

This part does not implement SPEC-002 §1 selection, §2 note integrity, §4 stress/loop settings,
§5's remaining evaluation/select/rollout tests, or §6 output schema. The pending specifications
remain in place and issue #2 remains open.

No real model or checkpoint was loaded. No real train, select, eval, rollout, branch, prefer,
preflight, or probe command ran.

## Fix round 1: hermetic reference hashes

Independent review found that the original hash test included `config["chat_replay"]`, which
resolves to untracked `data/chat_replay`. A clean checkout would silently omit those rows, while
local replay-data drift would be misclassified as a generator change.

The corrected test explicitly passes `chat_dir=None` and pins only deterministic rows owned by
`GENERATOR_VERSION`. It still loads `configs/agent_v2c.yaml` for task counts, seed, keep-last,
and recovery repeats. This guard intentionally does not detect replay-data drift and does not
preserve the complete locally mixed `agent_v2c` file hashes; replay provenance requires a
separate versioned input contract.

Fix Round 1 verification:

```text
uv run pytest -q tests/test_pipeline.py -k 'reference_generator'
RED with the old mixed hashes: 1 failed, showing all three generator-only digests.
GREEN with the new generator-only hashes: 1 passed.

uv run pytest -q tests/test_pipeline.py
87 passed; exit 0.

uv run ruff check src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/transcript.py src/local_llm_lab/pipeline/data.py src/local_llm_lab/pipeline/cli.py src/local_llm_lab/provenance.py tests/test_pipeline.py
All checks passed!; exit 0.

git diff --check -- tests/test_pipeline.py docs/superpowers/plans/2026-09-03-spec-002-review-corrections.md design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part1.md
No output; exit 0.
```

## R10 follow-up: configured replay oracle

R10 adds a second, conditional oracle for the shipped configuration's complete local mix while
preserving the hermetic generator-only oracle from Fix Round 1. The three complete
generator-version tests moved from `tests/test_pipeline.py` to dedicated `tests/test_tasks.py`:

- data-stage provenance records `GENERATOR_VERSION`;
- returned and persisted dataset manifests record `GENERATOR_VERSION`;
- the generator-only `agent_v2c` task/expert-row hashes remain pinned with `chat_dir=None`.

`tests/test_tasks.py` also pins the replay-inclusive hashes using the resolved
`config["chat_replay"]` directory and `config["chat_repeats"]`. It skips only when that configured
protected directory is absent. Both hash assertions instruct maintainers to bump
`GENERATOR_VERSION` and re-pin both oracles after an intentional row change. Each oracle writes
only to its independent pytest `tmp_path`.

### R10 RED/GREEN evidence

The replay-inclusive test was first wired with `chat_dir=None` while retaining the mixed-data
expectations:

```text
uv run pytest -q tests/test_tasks.py -k 'reference'
.F                                                                       [100%]
1 failed, 1 passed; exit 1.
```

The failure showed all three generator-only digests instead of the expected replay-inclusive
digests and included the actionable version-bump/two-repin message. After wiring
`chat_dir=config["chat_replay"]` and `chat_repeats=config["chat_repeats"]`, the same command
reported `.. [100%]` (2 passed; exit 0). Protected replay was present, so the absence-only skip
branch did not run in this checkout.

### R10 final verification

```text
uv run pytest --collect-only -q tests/test_tasks.py tests/test_pipeline.py
tests/test_pipeline.py: 84
tests/test_tasks.py: 4

uv run pytest -q tests/test_tasks.py
....                                                                     [100%]

uv run pytest -q tests/test_pipeline.py
........................................................................ [ 85%]
............                                                             [100%]

uv run ruff check tests/test_tasks.py tests/test_pipeline.py
All checks passed!

git diff --check -- tests/test_tasks.py tests/test_pipeline.py docs/superpowers/plans/2026-09-03-spec-002-review-corrections.md design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part1.md
No output; exit 0.
```

R10 changes tests and documentation only; production behavior is unchanged. The configured
protected replay files were read in place by the replay-inclusive test and were not copied,
manufactured, or modified. No model, tokenizer, or checkpoint was loaded, and no real MLX or
train/select/eval/rollout/branch/prefer/preflight/probe command ran.
