# SPEC-003 Run D Data Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement SPEC-003 sections 1 through 3 only: completion-safe Run D expert notes, the exact recovery recipe, explicit length-mixed split generation, and deterministic data/config plumbing.

**Architecture:** Keep task construction deterministic in `pipeline/tasks.py`, using the ratified SPEC-002 `Task.difficulty`, `task_from_id`, `completion_patterns`, and `required_carry` contracts as the ground-truth seams. Add a frozen `SplitSpec` to `pipeline/data.py`; generate and shuffle each named logical split with the existing seed derivations, concatenate logical splits by role for the mlx-lm `train/valid/test.jsonl` files, and include chat replay exactly once per role. `pipeline/cli.py` converts either the new `splits:` schema or the legacy `tasks:` schema to `SplitSpec` values before calling `write_dataset`.

**Tech Stack:** Python 3.13, frozen dataclasses, PyYAML, pytest, Ruff; simulator and tokenizer fakes only.

**Spec:** `design_specifications/pending/SPEC-003-run-d-data-recipe-and-cross-model-matrix.md`, sections 1 through 3 only.

## Global Constraints

- Read and obey `AGENTS.md`; pending `00-DECISION-MEMO-2026-09-03.md`, `01-IMPLEMENTER-BRIEFING.md`, and `02-INTERFACE-AND-WIRING-MAP.md` including R1-R11 and amendments; the ratified progress/fidelity review; SPEC-003; and GitHub issue #7 before writes.
- Sections 4 through 6 are excluded. Do not load a model, checkpoint, tokenizer, MLX, or run probes, preflight, training, selection, evaluation, cache-equivalence, or J-space.
- Write only the paths in this task's active Codex Coordinator claim. Treat `pipeline/integrity.py`, registry files, `protocol.py`, `provenance.py`, `tuner_data.py`, Task 5 paths, pending documents, `data/`, `outputs/`, and `reports/` as read-only.
- Preserve exact public signatures and completed SPEC-002 sections 1 and 2 behavior through `18c4bd9`. New tests belong only in `tests/test_tasks.py`, `tests/test_data.py`, `tests/test_cli.py`, and `tests/test_integrity.py`; never edit `tests/test_pipeline.py` or `tests/test_probes.py`.
- Bump `tasks.GENERATOR_VERSION` from 2 to 3 because note rows change. Update current-generator hash pins only after independently reproducible generation in pytest temporary directories; never write generated artifacts to repository `data/`, `outputs/`, or `reports/`.
- No note may match any `completion_patterns()` form except `pending: none`, and `pending: none` is legal only when the note's ground-truth remaining set is empty.
- `aggregate_report` uses one append-only `values so far:` list, includes the numeric `split after k of N` derived from the generated total in every read/calculation note, has no `(full)` marker, and carries all collected values on both sides of the split.
- `conditional_update` carries the complete append-only `loads so far:` table in every read/post-read decision note.
- `batch_update` uses `Inspected k of N` and `Applied k of N` progress with a ground-truth `pending:` list, retains `Next: worker-N.ini mode=old -> mode=new` in apply notes, and expresses phase exhaustion only as `pending: none`.
- Every `cross_reference` hop note names its one-based hop number and current key; each note before a search/read action carries the key consumed by that action, and consecutive hop notes differ.
- For `ledger_reconcile`, `cross_reference`, `conditional_update`, `batch_update`, and `aggregate_report`, every `Fact` returned by `required_carry(task, keep_last=2)` must appear in the note immediately preceding the action that needs it. Worker/path facts are checked with their normalised basename/value representation; numeric facts use digit boundaries.
- The unsupervised `wrong_path` and `stale_path` notes must state the exact guessed path used by their `read_file` call. Recovery repeat multipliers remain exactly `transient: 1`, `wrong_path: 2`, `unknown_tool: 2`, `stale_path: 6`, `failed_edit: 6` and are recorded in the manifest/provenance payload.
- Run D logical splits are exactly: `train` 240/difficulty 0/perturb true/role train; `train1` 120/1/true/train; `valid` 24/1/false/valid; `valid2` 24/2/false/valid; `test` 180/2/false/test; `test3` 60/3/false/test. The role outputs concatenate in declaration order after each logical split is shuffled with `random.Random(f"{seed}:{split}")`.
- Difficulty 3 uses the existing `level=3` arithmetic. Horizons for every family whose horizon is level-dependent must be strictly increasing from levels 0 to 3 for the same `(split, index, seed)`.
- Chat replay remains exactly one role-sized contribution: 240 rows in train, 48 in valid, and 60 in test when the configured protected replay directory exists; `train1`, `valid2`, and `test3` must not duplicate it. Extra training directories are included once in the train role.
- `configs/agent_v2d_qwen35_4b.yaml` differs from `configs/agent_v2d.yaml` only in `model` and `output`; `configs/agent_v2b_qwen35_4b.yaml` differs from `configs/agent_v2b.yaml` only in `model` and `output`. D3 uses `data/agent_v2d` and the six exact split declarations.
- `stage_data` must continue to call the existing SPEC-001 provenance seam with generator version, recovery multipliers, and the complete dataset manifest. Do not broaden into missing provenance work owned by SPEC-001; report any remaining dependency.
- Commit only explicit owned paths. Preserve unrelated dirty state and concurrent commits; do not switch branches/worktrees, push, close issues, or move the pending spec.

---

### Task 1: Run D generator, splits, configs, and deterministic evidence

**Files:**

- Modify: `src/local_llm_lab/pipeline/tasks.py`
- Modify: `src/local_llm_lab/pipeline/data.py`
- Modify: `src/local_llm_lab/pipeline/cli.py`
- Create: `configs/agent_v2d.yaml`
- Create: `configs/agent_v2d_qwen35_4b.yaml`
- Create: `configs/agent_v2b_qwen35_4b.yaml`
- Modify: `tests/test_tasks.py`
- Modify: `tests/test_data.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_integrity.py`
- Create: `design_specifications/under_review/SPEC-003-IMPLEMENTATION-REPORT.md`

**Interfaces:**

- Consumes: `Task.difficulty`, `make_tasks(..., *, perturb, difficulty)`, `task_from_id(...)`, `completion_patterns()`, `required_carry(task, *, keep_last)`, `write_provenance(...)`, and legacy `tasks:` configs at `18c4bd9`.
- Produces: `SplitSpec(count: int, difficulty: int | None = None, perturb: bool | None = None, role: Literal["train", "valid", "test"] = "train")`, role-aware `write_dataset(output, splits, *, seed, keep_last, chat_dir, chat_repeats, recovery_repeats, extra_dirs, tokenizer, spec)`, six-split Run D configs, generator version 3, and deterministic current-generator hashes.

- [ ] **Step 1: Add focused failing generator-invariant tests**

  In `tests/test_tasks.py`, add table-driven tests over difficulties 0, 1, 2, and 3 that independently assert: strictly increasing level-dependent horizons; aggregate `values so far:` plus the literal ground-truth `split after {len(values)//2} of {len(values)}`; full conditional `loads so far:` accumulation; batch `Inspected/Applied k of N`, exact pending worker sets, and apply-phase `Next:` queue heads; one-based cross-reference hop/key restatement; exact guessed-path agreement for every realised `wrong_path`/`stale_path`; and the generator-only hash oracle's new-version error before hashes are pinned.

  In `tests/test_integrity.py`, add one generator-wide completion test that scans every `Step.thought` across all families and levels with `completion_patterns()`, permits only the `pending: none` pattern, and independently derives whether the action-specific remaining set is empty. Add a required-carry test for exactly the five state-table families named in Global Constraints, using `required_carry(..., keep_last=2)` and literal fact matching rather than an implementation helper.

- [ ] **Step 2: Verify the new generator tests fail for the intended missing Run D behavior**

  Run:

  ```bash
  uv run pytest -o addopts='' -q tests/test_tasks.py tests/test_integrity.py
  ```

  Expected: assertions fail because generator version is 2, C's aggregate/conditional/batch wording remains, and existing notes contain completion forms; no collection/import error is acceptable.

- [ ] **Step 3: Implement the minimum completion-safe Run D task notes**

  In `pipeline/tasks.py`, set `GENERATOR_VERSION = 3`; retain RNG calls, task IDs, actions, files, variants, and signatures. Rewrite notes only. Replace all generic completion/finality wording with state plus `pending: none` when no action-specific item remains. Restore the three B-shaped long-family state structures exactly as constrained above, keep the batch apply queue head, and make cross-reference search/read notes carry one-based hop plus current key. Keep wrong/stale guessed path notes aligned with their call argument. Expose a pure `render_expert_note(task: Task, step_index: int) -> str` that bounds-checks and returns the canonical ground-truth note without duplicating templates, satisfying wiring-map gap 14 for the later patching consumer.

- [ ] **Step 4: Run generator/integrity tests to green and inspect the task-only diff**

  Run the Step 2 command again. Then run:

  ```bash
  uv run ruff check src/local_llm_lab/pipeline/tasks.py tests/test_tasks.py tests/test_integrity.py
  git diff --check -- src/local_llm_lab/pipeline/tasks.py tests/test_tasks.py tests/test_integrity.py
  ```

  Expected: all selected tests pass; Ruff and diff check exit 0.

- [ ] **Step 5: Add focused failing SplitSpec/data tests**

  In `tests/test_data.py`, construct six literal `SplitSpec` values and assert that `write_dataset`:

  - passes each logical split's literal difficulty and perturb value to real deterministic task generation;
  - emits role files `train.jsonl`, `valid.jsonl`, and `test.jsonl` whose expert task IDs are respectively `train+train1`, `valid+valid2`, and `test+test3`, with no cross-role IDs;
  - concatenates named splits in declaration order after split-specific seeded shuffles;
  - records `count`, `difficulty`, `perturb`, `role`, row counts, hashes, horizons, variants, and families for all six logical splits plus output hashes/counts;
  - applies recovery repeats to both train-role logical splits only;
  - adds fake chat rows exactly once to each matching role and fake extra rows exactly once to train;
  - produces byte-identical role files and manifest content across two separate `tmp_path` outputs.

  Keep old-config compatibility covered by converting its counts to explicit `SplitSpec` values at the CLI seam, not by changing its generated rows.

- [ ] **Step 6: Verify SplitSpec/data tests fail for the intended missing API**

  Run:

  ```bash
  uv run pytest -o addopts='' -q tests/test_data.py
  ```

  Expected: collection fails only on missing `SplitSpec`, or behavior assertions fail because named roles/splits are not implemented.

- [ ] **Step 7: Implement SplitSpec and deterministic role aggregation**

  Add the exact frozen `SplitSpec` dataclass and validate positive integer counts, non-negative optional difficulty, boolean-or-`None` perturb, and the three literal roles. Normalize generation around named logical splits. For each split, call:

  ```python
  make_tasks(
      split,
      split_spec.count,
      seed,
      perturb=split_spec.perturb,
      difficulty=split_spec.difficulty,
  )
  ```

  Build/optionally render expert rows and apply recovery repeats when `role == "train"`. Add each role's chat rows, and the train role's extras, only to the first declared logical chunk for that role before shuffling, preserving the existing one-split mix order. Shuffle every logical chunk with the unchanged `f"{seed}:{split}"` seed and concatenate chunks into their role in mapping order. Write only role JSONL outputs and `manifest.json`; all test generation stays under temporary directories.

- [ ] **Step 8: Run data tests to green and preserve rendering behavior**

  Run:

  ```bash
  uv run pytest -o addopts='' -q tests/test_data.py tests/test_tasks.py tests/test_integrity.py
  ```

  Expected: all pass, including Qwen2.5 fake-token rendering equivalence after updating the version assertion to 3.

- [ ] **Step 9: Add failing CLI/config tests**

  In `tests/test_cli.py`, assert `stage_data` converts both legacy `tasks:` and new `splits:` mappings to `SplitSpec`, rejects configurations defining both or neither, and passes the exact six D splits plus recovery multipliers to a fake `write_dataset`. Patch `write_dataset`, `write_provenance`, and `load_model_spec`; do not load any real model or write outside `tmp_path`.

  Assert the three shipped new YAML relationships by loading the mappings, removing only `model` and `output`, and comparing the remainder pairwise as stated in Global Constraints. Assert the six D split dictionaries literally.

- [ ] **Step 10: Verify CLI/config tests fail before plumbing/config creation**

  Run:

  ```bash
  uv run pytest -o addopts='' -q tests/test_cli.py
  ```

  Expected: failures name missing configs and absent `splits:` conversion; no model or pipeline stage runs.

- [ ] **Step 11: Implement CLI conversion and create exact configs**

  Add a small pure CLI helper that parses `splits:` entries to `SplitSpec`; for legacy `tasks:`, map each name/count to a spec with default difficulty/perturb and `role` equal to that standard split. `stage_data` passes the normalized mapping into `write_dataset` and retains its existing provenance call with the returned full manifest and `GENERATOR_VERSION`.

  Create D3 from Run C's non-data recipe with `output: outputs/agent-v2d`, `data: data/agent_v2d`, and the six literal split mappings. Copy D3 to D4 changing only `model: qwen35-4b` and `output: outputs/agent-v2d-qwen35-4b`. Copy the existing Run B config to B4 changing only `model: qwen35-4b` and `output: outputs/agent-v2b-qwen35-4b`.

- [ ] **Step 12: Run all focused tests and scoped lint**

  Run:

  ```bash
  uv run pytest -o addopts='' -q tests/test_tasks.py tests/test_data.py tests/test_cli.py tests/test_integrity.py
  uv run ruff check src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/data.py src/local_llm_lab/pipeline/cli.py tests/test_tasks.py tests/test_data.py tests/test_cli.py tests/test_integrity.py
  uv run ruff check --select C901 --config 'lint.mccabe.max-complexity=9' src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/data.py src/local_llm_lab/pipeline/cli.py
  ```

  Expected: focused pytest and normal Ruff pass. Strict C901 may report only the pre-task baseline `pipeline/cli.py::main` complexity 17 and `pipeline/tasks.py::_wrong_old` complexity 10; no new or increased violation is acceptable.

- [ ] **Step 13: Pin deterministic version-3 hashes with reviewable evidence**

  Run only the hash-oracle pytest nodes, which generate under `tmp_path`; capture the literal mismatches from the first run, independently regenerate in a second temporary pytest run, and confirm identical hashes before editing the version-3 tables. Pin both generator-only and configured-replay train/valid/test hashes in `tests/test_tasks.py`. Do not run the data CLI or write repository datasets.

- [ ] **Step 14: Write the implementation report**

  Create `design_specifications/under_review/SPEC-003-IMPLEMENTATION-REPORT.md` with: exact commits; changed paths and line counts; wiring rows 2.4, 2.5, 2.11 and gap 10; decisions; deviations; all RED/GREEN commands; old/new generator versions and literal old/new hash tables; deterministic double-generation evidence; recovery multipliers; provenance seam/dependency status; callers updated for changed signatures; strict C901 baseline comparison; banned-constant scan; full fake-only suite result; protected-path/no-model attestation; and observed-but-not-fixed items.

- [ ] **Step 15: Run final fresh verification and commit explicit owned paths**

  Run:

  ```bash
  uv run pytest -o addopts='' -q
  uv run pytest -o addopts='' -q tests/test_tasks.py tests/test_data.py tests/test_cli.py tests/test_integrity.py
  uv run ruff check src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/data.py src/local_llm_lab/pipeline/cli.py tests/test_tasks.py tests/test_data.py tests/test_cli.py tests/test_integrity.py
  uv run ruff check --select C901 --config 'lint.mccabe.max-complexity=9' src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/data.py src/local_llm_lab/pipeline/cli.py
  git diff --check -- src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/data.py src/local_llm_lab/pipeline/cli.py configs/agent_v2d.yaml configs/agent_v2d_qwen35_4b.yaml configs/agent_v2b_qwen35_4b.yaml tests/test_tasks.py tests/test_data.py tests/test_cli.py tests/test_integrity.py docs/superpowers/plans/2026-09-04-spec-003-run-d-data.md design_specifications/under_review/SPEC-003-IMPLEMENTATION-REPORT.md
  ```

  Inspect `git status --short`, preserve foreign staged/dirty paths, stage only the explicit owned paths, inspect `git diff --cached --stat` and `git diff --cached --check`, then commit the explicit paths with message `feat: implement SPEC-003 Run D data recipe`. Do not push.
