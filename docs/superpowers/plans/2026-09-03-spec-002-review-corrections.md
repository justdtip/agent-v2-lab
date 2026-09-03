# SPEC-002 Review Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and independently review issue #2 follow-ups F1–F4 under rulings R5–R6 without changing the accepted SPEC-002 §3/§5 behavior.

**Architecture:** Treat generator versioning as metadata around the deterministic task/data boundary: `tasks.py` owns the version, `data.py` copies it into every dataset manifest, and the data CLI writes the same value to run provenance through a small deterministic helper. Put transcript header filtering behind one reusable iterator, and make impossible recovery-variant construction fail at its source with task-specific errors. The R10 follow-up isolates the complete generator-version/hash-oracle test surface in `tests/test_tasks.py` and keeps separate generator-only and protected replay-inclusive guards.

**Tech Stack:** Python 3.13, pytest, YAML configuration, JSON/JSONL, GitHub CLI for issue evidence.

**Spec:** `design_specifications/pending/SPEC-002-evaluation-selection-note-integrity.md`, with binding rulings in `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` §7 and review findings in `design_specifications/under_review/SPEC-002-REVIEW-round1-2026-09-03.md`.

## Global Constraints

- `design_specifications/pending/` is authoritative and read-only.
- Commits `fc38a9d` and `6dff90c` are accepted and must not be reverted.
- `GENERATOR_VERSION = 1` identifies commit `97d197c`, the last generator that reproduces run C; the current generator is `GENERATOR_VERSION = 2`.
- Do not add a test requiring generator version 2 to reproduce run C. Pin the current generator's reference-config train/valid/test hashes so changed rows require a version bump.
- The version-2 generator-only reference uses `configs/agent_v2c.yaml` for counts, seed,
  keep-last, and recovery repeats, but passes `chat_dir=None` because chat replay is an external,
  untracked mix input. Its hashes are train
  `e7fa63ef9a2fe70b3563a23fa5421b4a6b11d11971802d2c4b6bce5a0a3c5d58`, valid
  `816543d1dee8299b2dbf514e93a2de480b69f84d6e8c53bda955c20c79f6213f`, and test
  `10042da5d9a4789f9a28fc6c0c9a8ea8efc59d090688f1e167c6de8d1de66877`.
- A second version-2 oracle uses the shipped `configs/agent_v2c.yaml` `chat_replay` path and
  `chat_repeats` value. When that protected directory is present, its mixed hashes are train
  `ad660e83cd89958dcee9fba2ab1e53115d1fb079813ea694cd4b0e89530b3a79`, valid
  `d6dbc53573744751d74565a0de6ca5c6d381cba6b488ff6410194bf9b0d4e6d8`, and test
  `fc69b03fef8f423ee85a174214ad955fe3f4d324554217510b92f53435841ce3`. Skip this oracle only
  when the configured replay directory is absent; never copy, manufacture, or modify replay data.
- Every transcript reader must skip the header record lacking `task_id` through `transcript.iter_task_records(path)`.
- Impossible `_wrong_path` and `_stale_path` construction must raise `RuntimeError` naming `task.task_id`; it must not silently change variants.
- No model/checkpoint loading and no real train/select/eval/rollout/branch/prefer/preflight/probe execution. Fake-only tests and deterministic data generation are allowed.
- Do not modify or regenerate tracked or untracked content under `data/`, `outputs/`, or `reports/`.
- Work directly on `codex/agent-v2-specs` in `/Users/daniel.tipton/Desktop/An app`; do not create or switch branches or worktrees.
- For R10, modify only the five paths in native task `01a06718-057a-7bb2-a62d-71f86e904d64`'s
  active Coordinator claim: `tests/test_pipeline.py`, `tests/test_tasks.py`, this plan, the part-one
  implementation report, and the SDD progress ledger. Skill-generated ignored SDD
  brief/report/review-package artifacts may live beside the claimed progress ledger.
- Leave issue #2 open because SPEC-002 §1, §2, §4, §5, and §6 remain outstanding.

---

### Task 1: Correct F1–F4 as one reviewed vertical

**Files:**

- Modify: `src/local_llm_lab/pipeline/tasks.py`
- Modify: `src/local_llm_lab/pipeline/transcript.py`
- Modify: `src/local_llm_lab/pipeline/data.py`
- Modify: `src/local_llm_lab/pipeline/cli.py`
- Create: `src/local_llm_lab/provenance.py`
- Modify: `tests/test_pipeline.py`
- Create: `design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part1.md`
- Include: `docs/superpowers/plans/2026-09-03-spec-002-review-corrections.md`
- Update: `.superpowers/sdd/2026-09-03-spec-002-review-corrections/progress.md`

**Interfaces:**

- Consumes: `write_dataset(output, counts, *, seed, keep_last, chat_dir, chat_repeats, recovery_repeats, extra_dirs) -> dict[str, Any]`, `load_model_spec(name_or_hf_id) -> ModelSpec`, and the existing transcript JSONL layout whose first record is `{"run_id": ...}`.
- Produces: `tasks.GENERATOR_VERSION: int = 2`; manifest top-level `generator_version`; deterministic `write_provenance(run_dir, *, resolved, spec, extra) -> Path` with top-level `generator_version`; and `iter_task_records(path: Path) -> Iterator[dict[str, Any]]` yielding only records containing `task_id`.
- Callers: `pipeline.cli.stage_data` writes data and then `outputs/<run>/provenance.json`; future transcript consumers call `iter_task_records` instead of parsing the header themselves.

- [ ] **Step 1: Replace the obsolete fallback expectation with failing task-specific error tests**

  In `tests/test_pipeline.py`, keep the existing guarded task that owns all four guessed filenames, but replace its transient-variant assertions with:

  ```python
  with pytest.raises(RuntimeError, match="guarded"):
      tasks._wrong_path(guarded, random.Random(0))
  ```

  Add a task with no viable stale-path candidate and assert `tasks._stale_path(...)` raises `RuntimeError` matching that task's id. The break caught is silent variant substitution; a return value of any variant must fail the tests.

- [ ] **Step 2: Run the fallback tests and verify RED**

  Run:

  ```bash
  uv run pytest -q tests/test_pipeline.py -k 'wrong_path or stale_path'
  ```

  Expected: the guarded wrong-path test fails because `_wrong_path` returns a transient task, and the stale-path test fails because `_stale_path` delegates to `_wrong_path`.

- [ ] **Step 3: Make the two impossible recovery paths fail loudly**

  In `tasks.py`, add the current generator constant near `FAMILIES`/`VARIANTS`:

  ```python
  GENERATOR_VERSION = 2
  ```

  Replace only the impossible construction fallbacks:

  ```python
  raise RuntimeError(f"{task.task_id}: could not construct a wrong_path variant")
  ```

  and:

  ```python
  raise RuntimeError(f"{task.task_id}: could not construct a stale_path variant")
  ```

  Keep valid `_wrong_path` and `_stale_path` generation unchanged.

- [ ] **Step 4: Run the focused recovery tests and verify GREEN**

  Run:

  ```bash
  uv run pytest -q tests/test_pipeline.py -k 'wrong_path or stale_path'
  ```

  Expected: all selected tests pass.

- [ ] **Step 5: Add failing manifest, provenance, transcript-iterator, and version-keyed hash tests**

  In `tests/test_pipeline.py`:

  - Import `GENERATOR_VERSION`, `iter_task_records`, `load_config`, `stage_data`, and `write_dataset` at their existing stable module seams.
  - Write a small temporary dataset and assert `manifest["generator_version"] == GENERATOR_VERSION` and the written `manifest.json` contains the same value.
  - Run `stage_data` with temporary `output` and `data` directories, tiny positive split counts, the registered `qwen25-coder-3b` name, and no chat or extra inputs. Assert `output/provenance.json` exists and its top-level `generator_version` equals `GENERATOR_VERSION`; this test must not invoke a loader.
  - Write a JSONL file containing a run header, two task records, a blank line, and a non-task metadata record. Assert `list(iter_task_records(path))` equals the two task records in order.
  - Generate the task/expert rows from `configs/agent_v2c.yaml` into `tmp_path`, explicitly
    passing `chat_dir=None`, then compare literal hashes through a version-keyed table:

  ```python
  expected = {
      2: {
          "train": "e7fa63ef9a2fe70b3563a23fa5421b4a6b11d11971802d2c4b6bce5a0a3c5d58",
          "valid": "816543d1dee8299b2dbf514e93a2de480b69f84d6e8c53bda955c20c79f6213f",
          "test": "10042da5d9a4789f9a28fc6c0c9a8ea8efc59d090688f1e167c6de8d1de66877",
      }
  }
  assert manifest["generator_version"] == GENERATOR_VERSION
  assert {split: info["sha256"] for split, info in manifest["splits"].items()} == expected[
      GENERATOR_VERSION
  ]
  ```

  The break caught is a task/expert-row-changing generator edit without an explicit version
  bump. Expectations are literal hashes derived independently before implementation; do not
  read expected hashes from `data/agent_v2c` or include untracked `data/chat_replay`. Replay-data
  drift and the full locally mixed dataset remain outside this generator-version oracle.

- [ ] **Step 6: Run the new metadata/iterator/hash tests and verify RED**

  Run:

  ```bash
  uv run pytest -q tests/test_pipeline.py -k 'generator_version or reference_generator or provenance or iter_task_records'
  ```

  Expected: collection or assertions fail because the new iterator/provenance API and manifest fields do not exist.

- [ ] **Step 7: Implement generator metadata and transcript filtering minimally**

  In `data.py`, import `GENERATOR_VERSION` and add a top-level manifest field:

  ```python
  "generator_version": GENERATOR_VERSION,
  ```

  In `transcript.py`, add the typed iterator:

  ```python
  from collections.abc import Iterator

  def iter_task_records(path: Path) -> Iterator[dict[str, Any]]:
      with path.open(encoding="utf-8") as handle:
          for line in handle:
              if not line.strip():
                  continue
              record = json.loads(line)
              if "task_id" in record:
                  yield record
  ```

  In `provenance.py`, implement only the deterministic write seam needed by F1 while matching the authoritative SPEC-001 signature. The payload must put `generator_version` at top level, serialize `resolved.as_dict()` when resolved is present or `dataclasses.asdict(spec)` otherwise, preserve caller extras under `extra`, use sorted JSON keys, create the run directory, and write one trailing newline:

  ```python
  def write_provenance(
      run_dir: Path,
      *,
      resolved: ResolvedSpec | None,
      spec: ModelSpec,
      extra: dict[str, Any],
  ) -> Path:
      payload = {
          "generator_version": GENERATOR_VERSION,
          "model": resolved.as_dict() if resolved is not None else asdict(spec),
          "extra": extra,
      }
      run_dir.mkdir(parents=True, exist_ok=True)
      target = run_dir / "provenance.json"
      target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
      return target
  ```

  Do not implement the unclaimed full provenance/hash/preflight surface from SPEC-001 Task 8.

- [ ] **Step 8: Wire data-stage provenance without loading a model**

  In `cli.py`, import `load_model_spec`, `GENERATOR_VERSION`, and `write_provenance`. After `write_dataset` returns, resolve only the registry declaration and write:

  ```python
  write_provenance(
      config["output"],
      resolved=None,
      spec=load_model_spec(config["model"]),
      extra={
          "stage": "data",
          "generator_version": GENERATOR_VERSION,
          "dataset_manifest": manifest,
      },
  )
  ```

  Loading YAML model metadata is allowed; importing or loading model weights/checkpoints is not.

- [ ] **Step 9: Run the new tests and verify GREEN**

  Run:

  ```bash
  uv run pytest -q tests/test_pipeline.py -k 'generator_version or reference_generator or provenance or iter_task_records'
  ```

  Expected: all selected tests pass.

- [ ] **Step 10: Write the part-one implementation report**

  Create `design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part1.md` with:

  - the exact reviewed slice and accepted commits `fc38a9d` and `6dff90c`;
  - every source/test file touched by F1–F4 and the callers changed;
  - decisions: R5 version 1/commit `97d197c`, current version 2 with the three pinned hashes, no cross-version run-C assertion, R6 header filtering, and task-named construction errors;
  - verification commands and results from this task;
  - explicit deferral of SPEC-002 §1 selection, §2 note integrity, §4 stress/loop settings, §5 remaining evaluation/select/rollout tests, and §6 output schema;
  - a statement that no real model/checkpoint or train/select/eval/rollout/branch/prefer/preflight/probe command ran;
  - a statement that the pending specifications remain in place and issue #2 remains open.

- [ ] **Step 11: Run the full fake-only verification set**

  Run:

  ```bash
  uv run pytest -q tests/test_pipeline.py
  uv run ruff check src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/transcript.py src/local_llm_lab/pipeline/data.py src/local_llm_lab/pipeline/cli.py src/local_llm_lab/provenance.py tests/test_pipeline.py
  git diff --check
  ```

  Expected: every command exits 0. Record exact counts/output in the SDD task report and the part-one implementation report.

- [ ] **Step 12: Inspect and commit only the claimed product paths**

  Confirm no path under `design_specifications/pending/`, `data/`, `outputs/`, or `reports/` changed. Inspect the staged names and diff, then commit with explicit pathspecs:

  ```bash
  git add -- src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/transcript.py src/local_llm_lab/pipeline/data.py src/local_llm_lab/pipeline/cli.py src/local_llm_lab/provenance.py tests/test_pipeline.py design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part1.md docs/superpowers/plans/2026-09-03-spec-002-review-corrections.md
  git diff --cached --name-only
  git diff --cached --check
  git commit -m "fix: address SPEC-002 review follow-ups"
  ```

  Do not stage the Coordinator board, SDD scratch artifacts, unrelated SPEC-004 work, or any pre-existing user/peer changes.

---

### Task 2: Split and extend the version-keyed hash oracles under R10

**Files:**

- Modify: `tests/test_pipeline.py`
- Create: `tests/test_tasks.py`
- Modify: `design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part1.md`
- Include: `docs/superpowers/plans/2026-09-03-spec-002-review-corrections.md`
- Update: `.superpowers/sdd/2026-09-03-spec-002-review-corrections/progress.md`

**Interfaces:**

- Consumes: `load_config(configs/agent_v2c.yaml)`, `stage_data`, and
  `write_dataset(..., chat_dir, chat_repeats, recovery_repeats)`.
- Produces: a dedicated `tests/test_tasks.py` containing the complete generator-version test
  surface, including both the generator-only and configured protected replay-inclusive hash
  oracles.
- Preserves: the generator-only hashes above, existing production behavior, and all protected
  replay data byte-for-byte.

- [x] **Step 1: Move only the generator-version tests and directly owned imports**

  Move these tests from `tests/test_pipeline.py` into the new `tests/test_tasks.py` without
  broadening the split:

  - `test_data_stage_writes_generator_version_provenance`
  - `test_dataset_manifest_records_generator_version`
  - the existing generator-only reference hash test

  Move or add only imports used by those tests. Keep `json` and `write_dataset` in
  `tests/test_pipeline.py` because unrelated tests still use them; remove its now-unused top-level
  `Path`, `load_config`, `stage_data`, and `GENERATOR_VERSION` imports.

- [x] **Step 2: Add the replay-inclusive oracle with deliberately wrong wiring and verify RED**

  Add a second test that loads `configs/agent_v2c.yaml`, skips only when the resolved configured
  `chat_replay` directory does not exist, and compares the three mixed hashes above through a
  `GENERATOR_VERSION`-keyed table. First call `write_dataset` with `chat_dir=None` while retaining
  the mixed expectations, then run:

  ```bash
  uv run pytest -q tests/test_tasks.py -k 'reference'
  ```

  Expected with protected replay present: the new test fails with all three generator-only
  digests differing from the mixed expectations. On a clean checkout without the configured
  protected directory, only the replay-inclusive case skips while the generator-only test passes.

- [x] **Step 3: Wire the configured replay input and verify GREEN**

  Pass `chat_dir=config["chat_replay"]`, `chat_repeats=config["chat_repeats"]`, and the existing
  seed, counts, keep-last, and recovery-repeat settings. Both oracles must write only to their
  independent pytest `tmp_path` output. Give both equality assertions a clear failure message
  instructing maintainers to bump `GENERATOR_VERSION` and re-pin both oracles when an intentional
  row change occurs. Run:

  ```bash
  uv run pytest -q tests/test_tasks.py -k 'reference'
  ```

  Expected with protected replay present: two passes. Expected when absent: one pass and one skip.

- [x] **Step 4: Verify collection, the focused module, the full pipeline module, and scoped Ruff**

  Run:

  ```bash
  uv run pytest --collect-only -q tests/test_tasks.py tests/test_pipeline.py
  uv run pytest -q tests/test_tasks.py
  uv run pytest -q tests/test_pipeline.py
  uv run ruff check tests/test_tasks.py tests/test_pipeline.py
  git diff --check -- tests/test_tasks.py tests/test_pipeline.py docs/superpowers/plans/2026-09-03-spec-002-review-corrections.md design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part1.md
  ```

  No command may load a model, tokenizer, checkpoint, or run real MLX execution.

- [x] **Step 5: Update evidence and commit the four tracked R10 paths**

  Add the R10 finding, RED/GREEN output, final verification counts, protected-directory skip
  semantics, and explicit no-model/no-data-mutation statement to the report and SDD ledger. Stage
  only the four tracked claim paths (the ledger is ignored), inspect the staged names and diff,
  then commit:

  ```bash
  git add -- tests/test_pipeline.py tests/test_tasks.py design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part1.md docs/superpowers/plans/2026-09-03-spec-002-review-corrections.md
  git diff --cached --name-only
  git diff --cached --check
  git commit -m "test: cover configured replay in generator hashes"
  ```

  Leave issue #2 open and do not stage or modify any protected data, pending specifications,
  Coordinator state, or unrelated user/peer work.

#### Task 2 independent review and closure

The independent R10 implementation review approved commit `ee07f6a` with no Critical,
Important, or Minor findings. Controller verification collected 84 tests from
`tests/test_pipeline.py` and 4 from `tests/test_tasks.py`, passed all 4 current task tests,
reported scoped Ruff clean, and found `git diff --check 6325d3a..ee07f6a` clean.

To isolate R10 from later branch changes, the controller exported exact commit `ee07f6a` with
`git archive` to `/private/tmp/spec002-r10.k3iEI4`; using `PYTHONPATH` against that archive,
`tests/test_pipeline.py` passed 84/84. The branch later advanced through unrelated commits
`b39d6ea` and `d498599`. At current HEAD, the pipeline module has one later concurrent failure:
`d498599` removed the old `jlens_map(..., stats=...)` API while the unchanged pipeline test still
calls it. That regression is outside R10 and does not alter its approved result.
