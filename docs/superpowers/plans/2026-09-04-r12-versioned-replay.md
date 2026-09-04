# R12 Versioned Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both retroactive analysis tools reconstruct the note templates named by a saved artifact while keeping every normal task-generation path on generator v4, restore v4 `row_labels` coverage, and remove Issue #13's duplicated registry literals from the saved-NPZ CLI contract.

**Architecture:** Keep `make_tasks` and `task_from_id` as HEAD-only generation APIs. Add one explicitly replay-only task reconstruction seam that selects the historical note templates for versions 1–4, then thread a resolved artifact version through integrity analysis and saved-NPZ reanalysis. New captures record `generator_version`; old unversioned artifacts require an explicit caller/CLI binding, and a recorded version cannot be overridden by a conflicting binding.

**Tech Stack:** Python 3.11+, dataclasses, NumPy NPZ metadata, argparse, pytest, Ruff.

**Spec:** `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` ruling R12 (ratified 2026-09-04 09:00), with R10 and R13 as binding adjacent rulings.

## Global Constraints

- R12: retroactive integrity and saved-NPZ reanalysis replay the artifact's recorded generator version; historical note-template functions stay in-repo behind a replay-only switch; normal training/data generation always uses HEAD templates.
- R12 legacy binding: an artifact with no recorded `generator_version` must receive an explicit version binding; never silently substitute HEAD. A conflicting explicit binding and recorded version is an error.
- Supported replay versions are exactly `1..GENERATOR_VERSION` (`GENERATOR_VERSION == 4` in this change); reject booleans, non-integers, versions below 1, and future versions with a named error.
- R10: new module tests live in `tests/test_tasks.py`, `tests/test_integrity.py`, and `tests/test_state_probe.py`. Issue #13 is a coordinator-assigned subtractive edit to the existing saved-NPZ contract in `tests/test_probes.py`; do not add unrelated coverage there.
- R13: do not request review or hand off until `uv run pytest -q` exits 0.
- Do not edit `design_specifications/pending/`, protected artifacts under `outputs/` or `data/`, or files outside the claimed paths.
- Do not load a model, tokenizer, MLX runtime, checkpoint, or probe artifact during implementation or verification.
- Preserve the current v4 generator hashes, task ids, prompts, actions, files, labels, and public HEAD-generation behavior. Do not bump `GENERATOR_VERSION` because no generated HEAD row changes.
- Use exactly one principal implementer with no subagents, followed by exactly one independent principal reviewer.

---

### Task 1: Replay historical notes and bind both offline consumers

**Files:**
- Modify: `src/local_llm_lab/pipeline/tasks.py`
- Modify: `src/local_llm_lab/pipeline/integrity.py`
- Modify: `src/local_llm_lab/probes/state_probe.py`
- Modify: `tests/test_tasks.py`
- Modify: `tests/test_integrity.py`
- Modify: `tests/test_state_probe.py`
- Modify: `tests/test_probes.py`
- Create: `design_specifications/under_review/R12-VERSIONED-REPLAY-IMPLEMENTATION-REPORT.md`

**Interfaces:**
- Consumes: `Task`, current private family makers and recovery-variant builders, `build_rows`, `ProbeDataset.meta`, evaluation `summary`, and R12's explicit legacy binding.
- Produces: `replay_task_from_id(task_id: str, seed: int, generator_version: int, difficulty: int | None = None) -> Task`; `_render_evaluations(paths: list[Path], seed: int, *, generator_version: int | None = None) -> str`; `reanalyse_dataset(..., generator_version: int | None = None) -> dict[str, Any]`; `--generator-version` on both retroactive CLIs; `generator_version` in newly built probe dataset/checkpoint metadata.

- [ ] **Step 1: Add failing replay-only task tests**

  In `tests/test_tasks.py`, import `replay_task_from_id` and add tests that:

  ```python
  current = task_from_id(task_id, seed, difficulty)
  replayed = replay_task_from_id(task_id, seed, 1, difficulty)
  assert [step.action for step in replayed.steps] == [step.action for step in current.steps]
  assert replayed.prompt == current.prompt
  assert replayed.steps[-1].thought == historical_v1_finish
  assert task_from_id(task_id, seed, difficulty) == current
  ```

  Cover versions 1, 2, 3, and 4 with literals copied from the corresponding repository revisions: v1 is commit `97d197c`, v2 is the generator at `67464c9`, v3 is `68e4dac`, and v4 is `83ca7e1`/HEAD. Include at least one simple family, every long family whose notes changed, and recovery rows covering the v1→v2 and v3→v4 injected-note differences. Assert invalid values (`True`, `0`, `5`, and a non-integer) raise `ValueError` naming `generator_version`. Assert `make_tasks(..., generator_version=...)` remains a `TypeError`, proving normal generation exposes no historical switch.

- [ ] **Step 2: Run the task replay tests and record RED**

  Run:

  ```bash
  uv run pytest -q tests/test_tasks.py -k 'replay or historical_generator'
  ```

  Expected: failure because `replay_task_from_id` and historical note selection do not yet exist.

- [ ] **Step 3: Implement the replay-only note switch in `tasks.py`**

  Keep the public HEAD path equivalent to:

  ```python
  def task_from_id(task_id: str, seed: int, difficulty: int | None = None) -> Task:
      return _task_from_id(task_id, seed, difficulty, note_version=GENERATOR_VERSION)

  def replay_task_from_id(
      task_id: str,
      seed: int,
      generator_version: int,
      difficulty: int | None = None,
  ) -> Task:
      version = _validate_generator_version(generator_version)
      return _task_from_id(task_id, seed, difficulty, note_version=version)
  ```

  Thread `note_version` explicitly through the private family-maker and recovery-variant seams; do not use mutable module state. Preserve historical *thought strings only* from the named revisions. Continue to use current task structure, files, actions, RNG, prompts, difficulty, and variant validation, as R12 names historical note-template replay rather than rollback of the generator. Version 1 and version 2 share clean-family notes; their only note differences are the historical wrong/stale-path injections shown by `git diff 97d197c ed9136a -- src/local_llm_lab/pipeline/tasks.py`. Version 3 is the Run-D note rewrite at `68e4dac`; version 4 includes the state-preserving corrections at `83ca7e1`.

- [ ] **Step 4: Make the task replay tests GREEN and preserve v4 hash oracles**

  Run:

  ```bash
  uv run pytest -q tests/test_tasks.py
  ```

  Expected: all pass, including both version-4 reference hash tests with their existing hashes.

- [ ] **Step 5: Add failing integrity version-binding tests**

  In `tests/test_integrity.py`, make `_write_evaluation` record `summary["generator_version"] = GENERATOR_VERSION` by default. Add tests for recorded-version replay, explicit binding of an unversioned legacy evaluation, rejection when neither exists, rejection of a binding that conflicts with a recorded version, and CLI forwarding of `--generator-version`. Pin the protected B/C contract without rewriting either file:

  ```python
  rendered = _render_evaluations(
      [run_b, run_c],
      20260902,
      generator_version=1,
  )
  ```

  Retain all existing memo rows and the exact three PASS checks.

- [ ] **Step 6: Run the integrity tests and record RED**

  Run:

  ```bash
  uv run pytest -q tests/test_integrity.py -k 'generator_version or retroactive_saved or cli_helper'
  ```

  Expected: the new binding tests fail and the protected memo contract still fails until the replay version reaches `task_from_id`.

- [ ] **Step 7: Thread the resolved version through integrity analysis**

  Add a small pure resolver in `integrity.py` with these semantics:

  ```python
  def _artifact_generator_version(
      recorded: object,
      explicit: int | None,
      *,
      source: Path,
  ) -> int:
      for label, value in (("recorded", recorded), ("explicit", explicit)):
          if value is not None and (type(value) is not int or not 1 <= value <= GENERATOR_VERSION):
              raise ValueError(f"{source}: invalid {label} generator_version {value!r}")
      if recorded is None and explicit is None:
          raise ValueError(f"{source}: artifact has no generator_version; bind one explicitly")
      if recorded is not None and explicit is not None and recorded != explicit:
          raise ValueError(
              f"{source}: recorded generator_version {recorded} conflicts with explicit {explicit}"
          )
      return int(recorded if recorded is not None else explicit)
  ```

  Read `generator_version` from the evaluation summary (and accept a top-level field if present for compatibility), resolve it against the keyword-only explicit binding, and call `replay_task_from_id`. Include the resolved version in the source/configuration table. Add `--generator-version` to `agent-v2-integrity`; it is an explicit legacy binding, not a fallback to HEAD.

- [ ] **Step 8: Make focused integrity tests GREEN**

  Run:

  ```bash
  uv run pytest -q tests/test_integrity.py
  ```

  Expected: all pass, including `test_retroactive_saved_evaluations_match_memo_contract`, with protected B/C bytes unchanged.

- [ ] **Step 9: Add failing saved-NPZ replay tests and restore v4 `row_labels` coverage**

  In `tests/test_state_probe.py`, add only fake/offline tests. Cover all of the following:

  - `build_label_dataset` and `build_probe_dataset` record `GENERATOR_VERSION` in consolidated metadata; capture checkpoint metadata/context also carries it.
  - `_regenerate_tasks` or `_offline_rows` uses a recorded historical version, accepts an explicit binding only for unversioned legacy metadata, and rejects missing/conflicting/unsupported bindings.
  - A tiny saved NPZ round-trip pins its version explicitly and reanalysis never calls model/tokenizer loaders.
  - `reanalyse_dataset(..., generator_version=...)` and the reanalyse CLI forward the binding and report the resolved version in result metadata.
  - Restore the deleted `row_labels` contracts against generator v4. Parse v4 `pending:`, `loads so far`/`highest so far`, `values so far`/`split after`, and batch `Inspected`/`Applied`/`verified` state from notes, then compare independently with structural `row_labels`; assert non-vacuous counts and `GENERATOR_VERSION == 4`. Keep the existing out-of-range, next-tool, phase-domain, and recovery-metadata coverage where it already lives; do not duplicate it.

- [ ] **Step 10: Run the state-probe tests and record RED**

  Run:

  ```bash
  uv run pytest -q tests/test_state_probe.py
  ```

  Expected: failures for missing version metadata/binding and missing restored note-oracle coverage setup.

- [ ] **Step 11: Implement saved-NPZ version recording and replay**

  Import `GENERATOR_VERSION` and `replay_task_from_id` in `state_probe.py`. Stamp new label datasets, captured per-task checkpoint metadata/context, and consolidated capture metadata with v4. Resolve recorded `dataset.meta["generator_version"]` against an optional explicit binding using the same missing/mismatch semantics as integrity. Refactor `_regenerate_tasks` to reconstruct each saved `task_id` directly with its saved per-row difficulty and the resolved replay version, rather than using `make_tasks` (which must stay HEAD-only). Thread the version through `_offline_rows` and `reanalyse_dataset`, report it under `results["metadata"]["generator_version"]`, and add `--generator-version` to `_main_reanalyse`.

- [ ] **Step 12: Make all owned-module tests GREEN**

  Resolve coordinator-assigned Issue #13 in the existing `test_reanalyse_cli_is_deterministic_and_never_calls_model_loading` only: replace duplicated literal registry values with `asdict(load_model_spec(dataset.meta["model"]))`, retain the independent exact `ModelSpec` key-set schema assertion, and parameterize the fake-only path over `qwen25-coder-3b` and `qwen35-4b`. Do not change production reanalysis behavior for this issue.

  Run:

  ```bash
  uv run pytest -q tests/test_tasks.py tests/test_integrity.py tests/test_state_probe.py tests/test_probes.py
  ```

  Expected: all pass.

- [ ] **Step 13: Run scoped quality and protected-path checks**

  Run:

  ```bash
  uv run ruff check src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/integrity.py src/local_llm_lab/probes/state_probe.py tests/test_tasks.py tests/test_integrity.py tests/test_state_probe.py tests/test_probes.py
  uv run ruff check --select C901 src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/integrity.py src/local_llm_lab/probes/state_probe.py
  git diff --check
  git diff --name-only -- outputs data design_specifications/pending
  ```

  Expected: all commands exit 0, and the final command prints only the pre-existing pending-spec modifications (no `outputs/` or `data/` paths and no new pending-spec edits from this task).

- [ ] **Step 14: Run the exact R13 full fake-only gate**

  Record exact HEAD, then run:

  ```bash
  git rev-parse HEAD
  uv run pytest -q
  ```

  Expected: exit code 0 with no failing node names. Do not request review while this command is red.

- [ ] **Step 15: Write the implementation report and commit only owned paths**

  Create `design_specifications/under_review/R12-VERSIONED-REPLAY-IMPLEMENTATION-REPORT.md`. Include focused test commands/results, scoped Ruff and C901 results, `git diff --check`, a no-model statement, protected-path confirmation, SDD ledger/reviewer status, and a section named exactly:

  ```markdown
  ## R13 full fake-only suite

  - HEAD: `<40-character commit>`
  - Command: `uv run pytest -q`
  - Exit code: `0`
  - Result: `<passed> passed, <failed> failed, <xfailed> xfailed`
  - Failing node names: none
  ```

  Commit only the explicitly owned paths; preserve all foreign dirty and staged state. The implementer returns `DONE`, commit ids, a one-line test summary, and concerns in the SDD task report.
