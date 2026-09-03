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
- Version 2 is pinned to train
  `ad660e83cd89958dcee9fba2ab1e53115d1fb079813ea694cd4b0e89530b3a79`, valid
  `d6dbc53573744751d74565a0de6ca5c6d381cba6b488ff6410194bf9b0d4e6d8`, and test
  `fc69b03fef8f423ee85a174214ad955fe3f4d324554217510b92f53435841ce3` for
  `configs/agent_v2c.yaml`.
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
