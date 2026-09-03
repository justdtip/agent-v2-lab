# SPEC-002 Section 1 Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement SPEC-002 section 1's deterministic difficulty-aware, 48-task family-balanced checkpoint screen and statistical evaluation/reporting contracts without running a model.

**Architecture:** Keep task generation deterministic and row-compatible, adding only a pure family-balanced task selector around the existing generator. Extend evaluation with pure Wilson/McNemar calculations and explicit difficulty metadata, then make `stage_select` aggregate two 24-task screen cells per checkpoint and rank checkpoint records lexicographically from behavior to loss to earlier step. Preserve the existing report table for legacy summaries while attaching intervals and paired comparisons when the richer evaluation records are available.

**Tech Stack:** Python 3.13, frozen dataclasses, PyYAML, pytest, Ruff; no new dependency.

**Spec:** `design_specifications/pending/SPEC-002-evaluation-selection-note-integrity.md` section 1, with exact signatures in `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` sections 2.4, 2.10, and 2.11, and rulings R5, R10, and R11 in section 7.

## Global Constraints

- No model, checkpoint, tokenizer, MLX, probe, preflight, cache-equivalence, or J-space workload may run; tests use fakes and saved lightweight metadata only.
- `make_tasks(split, count, seed, *, perturb=None, difficulty=None)` keeps the unchanged default mapping `train -> 0`, `valid -> 1`, `test -> 2`, and any other split -> `index % 2`; an explicit non-negative difficulty overrides that mapping.
- `Task` remains a frozen dataclass and `task_from_id(task_id: str, seed: int, difficulty: int | None = None) -> Task` remains the exact reconstruction signature.
- The screen is exactly two cells: `valid` at difficulty 1 and `valid2` at difficulty 2; each cell includes one task from every family and two additional tasks from each of the six long-horizon families, for 24 tasks per cell and 48 per checkpoint.
- Checkpoint ranking is deterministic in this order: higher family-macro success, higher micro success, higher clean rate, higher valid-action rate, lower validation loss, then earlier checkpoint step.
- `selection.json` records component values per checkpoint, the micro-success Wilson 95 percent interval, validation loss, a validation-loss/behavior disagreement flag, model, all screen difficulties, and data seed.
- Paired McNemar uses only identical matching task-id sets and the exact two-sided binomial test on discordant pairs.
- Every newly reported rate carries a Wilson 95 percent interval; legacy evaluation records remain readable and retain their historical table behavior.
- Ruling R5 applies: do not bump `GENERATOR_VERSION` or re-pin hashes unless generated rows actually change. This task must leave the pinned current-version train/valid/test hashes green.
- Ruling R10 applies: new selection/evaluation/report tests live in per-module files; `tests/test_pipeline.py` and `tests/test_probes.py` remain read-only unless their current owner releases an exact temporary boundary.
- Ruling R11 applies: metadata-sized JSON may write directly; no array or file over 1 MiB is created by this task.
- Preserve the accepted SPEC-002 section 2 note-integrity behavior through commit `67464c9`; do not edit `pipeline/integrity.py`, `tests/test_integrity.py`, rollout, runner, or protected `data/`, `outputs/`, and `reports/` paths.
- Commit only explicit owned paths. Preserve unrelated dirty state and do not switch branches/worktrees, push, close issues, or edit pending specifications and Deputy-owned notes/reviews/logs.

---

### Task 1: Difficulty-aware family screen, selection ranking, and statistical reporting

**Files:**

- Modify: `configs/agent_v2.yaml`
- Modify: `configs/agent_v2b.yaml`
- Modify: `configs/agent_v2c.yaml`
- Modify: `src/local_llm_lab/pipeline/tasks.py`
- Modify: `src/local_llm_lab/pipeline/evaluate.py`
- Modify: `src/local_llm_lab/pipeline/cli.py`
- Modify: `src/local_llm_lab/pipeline/report.py`
- Modify temporarily, then release from the board: `tests/test_pipeline.py`
- Modify: `tests/test_tasks.py`
- Create: `tests/test_evaluate.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_report.py`
- Create: `tests/test_selection.py`
- Create: `design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part3-selection.md`

**Interfaces:**

- Consumes: the current `Task`, `make_tasks`, `task_from_id`, `Trajectory`, `summarize`, `run_evaluation`, `stage_select`, `load_summaries`, and `render` seams; the real training-log line shape `Iter 300: Val loss 0.103, Val took 87.010s`.
- Produces: `LONG_HORIZON_FAMILIES`, a pure family-balanced task helper, `wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]`, `mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]`, difficulty-aware `run_evaluation`, richer summaries, deterministic checkpoint component records, and paired comparison rows in the report.
- Compatibility: `stage_select(config, limit, quiet) -> Path` stays callable with the existing signature. `--limit` may cap each configured screen cell for an explicit diagnostic run, but the default config uses the complete 24+24 family-balanced screen.

- [ ] **Step 0: Extract the one legacy selection test in a behavior-preserving commit**

  The goal coordinator has sent `RELEASED tests/test_pipeline.py`, and the board claim now temporarily includes that exact file. Move only `test_stage_select_refuses_empty_checkpoint_directory` from `tests/test_pipeline.py` to `tests/test_selection.py`, retaining the body and assertions byte-for-byte apart from imports required by the new module. Run the old node from its new location and the neighboring existing selection/config tests needed for collection, then commit only the two test files:

  ```bash
  uv run pytest -q tests/test_selection.py::test_stage_select_refuses_empty_checkpoint_directory
  git add -- tests/test_pipeline.py tests/test_selection.py
  git diff --cached --name-only
  git diff --cached --check
  git commit --only -m "test: isolate checkpoint selection coverage" -- tests/test_pipeline.py tests/test_selection.py
  ```

  Stop and report this preliminary commit to the controller before changing any production behavior. The controller will remove `tests/test_pipeline.py` from the board claim and resume this same principal implementer for Steps 1-20.

- [ ] **Step 1: Add task-contract and production-isolation tests**

  In `tests/test_tasks.py`, add literal assertions that:

  ```python
  assert [task.difficulty for task in make_tasks("valid2", 4)] == [0, 1, 0, 1]
  assert {task.difficulty for task in make_tasks("valid2", 48, difficulty=2)} == {2}
  assert task_from_id(task.task_id, 20260902, 2) == task
  ```

  Exercise the pure family-balanced helper for `valid`/1 and `valid2`/2, asserting 24 tasks per cell, counts 1 for the first six families and 3 for the six long-horizon families, all-clean variants, exact difficulties, deterministic equality on repetition, and no duplicate task IDs. Generate train 240 and test 180 plus those two screen cells and assert zero prompt and file-path overlap across all four split sets. Name the concrete mutations each test catches: wrong default mapping, ignored override, non-exact reconstruction, incorrect family quota, recovery leakage, nondeterminism, and split collision.

- [ ] **Step 2: Run task tests and capture RED**

  Run:

  ```bash
  uv run pytest -q tests/test_tasks.py
  ```

  Expected: the new family-balanced helper/import is absent while the existing generator-hash tests remain green. Record the command, exit code, and relevant failure in the implementation report.

- [ ] **Step 3: Implement the smallest task-selection seam**

  In `pipeline/tasks.py`, expose the six long-horizon families (`pointer_chain` through `aggregate_report`) and implement one pure helper that calls the existing `make_tasks` with `perturb=False` and an explicit difficulty, then deterministically selects the requested `default`/`long` family quotas. Validate non-negative integer quotas and reject unknown quota keys. Do not modify maker functions, RNG derivations, task IDs, task rows, `GENERATOR_VERSION`, or the exact public signatures already accepted in commit `67464c9`.

- [ ] **Step 4: Run task tests and capture GREEN**

  Run:

  ```bash
  uv run pytest -q tests/test_tasks.py
  ```

  Expected: all task, reconstruction, isolation, and current generator-hash tests pass without warnings.

- [ ] **Step 5: Add evaluation-statistics tests**

  In `tests/test_evaluate.py`, use literal hand-calculated cases:

  ```python
  assert wilson(0, 0) == (0.0, 0.0)
  assert wilson(5, 10) == pytest.approx((0.236593, 0.763407), abs=1e-6)
  assert mcnemar({"a": True, "b": True, "c": False},
                 {"a": False, "b": True, "c": True}) == {
      "tasks": 3, "a_only": 1, "b_only": 1, "discordant": 2, "p_value": 1.0
  }
  ```

  Cover asymmetric discordance with an independently calculated exact-binomial value, empty discordance, rejected mismatched task-id sets, and invalid counts. Build small real `Trajectory` values to assert that `summarize` reports Wilson intervals for success, clean, valid-action, schema-validity, executable-call, integrity-clean, every `by_family`/`by_variant`/`by_difficulty` success rate, and exposes exact rate numerators/denominators needed by selection. Add a fake-only `run_evaluation` test that patches the model boundary and verifies the explicit difficulty reaches generated tasks and appears in summary/per-trajectory JSON alongside `model` and `data_seed`.

- [ ] **Step 6: Run evaluation tests and capture RED**

  Run:

  ```bash
  uv run pytest -q tests/test_evaluate.py
  ```

  Expected: collection or assertions fail because `wilson`, `mcnemar`, difficulty-aware evaluation, and Wilson-bearing summaries are absent.

- [ ] **Step 7: Implement Wilson, exact McNemar, and metadata-rich summaries**

  In `pipeline/evaluate.py`:

  ```python
  def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
      if n < 0 or successes < 0 or successes > n:
          raise ValueError("successes must be between zero and n")
      if n == 0:
          return (0.0, 0.0)
      proportion = successes / n
      denominator = 1 + z * z / n
      centre = (proportion + z * z / (2 * n)) / denominator
      radius = z / denominator * math.sqrt(
          proportion * (1 - proportion) / n + z * z / (4 * n * n)
      )
      return (max(0.0, centre - radius), min(1.0, centre + radius))


  def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]:
      if not a or a.keys() != b.keys():
          raise ValueError("McNemar inputs must contain the same non-empty task ids")
      a_only = sum(bool(a[key]) and not bool(b[key]) for key in a)
      b_only = sum(bool(b[key]) and not bool(a[key]) for key in a)
      discordant = a_only + b_only
      tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
      p_value = 1.0 if discordant == 0 else min(1.0, 2 * tail / (2**discordant))
      return {
          "tasks": len(a),
          "a_only": a_only,
          "b_only": b_only,
          "discordant": discordant,
          "p_value": p_value,
      }
  ```

  Wilson validates `0 <= successes <= n`, returns `(0.0, 0.0)` for `n == 0`, uses the standard uncorrected Wilson score formula, clamps to `[0, 1]`, and rounds only at serialization boundaries. McNemar requires the same non-empty task-id set, counts `a_only` and `b_only`, and returns the exact two-sided binomial probability `min(1, 2 * sum(comb(d, i) for i in range(min(a_only, b_only) + 1)) / 2**d)` (or `1.0` for `d == 0`).

  Extend `summarize` with exact numerator/denominator fields and Wilson records without removing accepted note-integrity fields. Add `by_difficulty`. Extend `run_evaluation` with an optional `difficulty` and optional family quotas, pass them to the task helper/generator, and record `model`, `difficulty`, and `data_seed`; keep model loading behind the existing boundary so tests can fake it.

- [ ] **Step 8: Run evaluation tests and capture GREEN**

  Run:

  ```bash
  uv run pytest -q tests/test_evaluate.py tests/test_integrity.py
  ```

  Expected: the new statistics/metadata tests and the complete section-2 integrity tests pass without warnings.

- [ ] **Step 9: Add selection/config tests**

  In `tests/test_cli.py`, assert all three shipped configs contain exactly the two required screen cells and no legacy `select.limit`/`select.split` keys. In `tests/test_selection.py`, fake `checkpoint_dirs` and `run_evaluation` with complete summary dictionaries, then assert:

  - every checkpoint evaluates both screen cells with difficulty and quotas, totaling 48 tasks by default;
  - family macro is calculated from combined per-family counts, not the micro rate;
  - the exact ranking order is macro, micro, clean, valid actions, lower validation loss, earlier step;
  - real `train.log` syntax is parsed per checkpoint step and a missing loss ranks after a present tied loss;
  - `selection.json` contains `model`, `data_seed`, screen definitions/difficulties, each checkpoint's `components`, `wilson_95`, `val_loss`, and boolean `disagreement` plus the behavior-best and loss-best steps;
  - the copied best adapter is the deterministic winner and rerunning identical fakes produces identical selection JSON.

- [ ] **Step 10: Run selection tests and capture RED**

  Run:

  ```bash
  uv run pytest -q tests/test_cli.py tests/test_selection.py
  ```

  Expected: configs still use `split`/`limit`, and selection lacks screen aggregation, loss parsing, required ordering, component records, and metadata.

- [ ] **Step 11: Implement the configured screen and deterministic selector**

  Replace each config's `select` block with the exact two-cell `screen` list from SPEC-002. In `pipeline/cli.py`, parse validation losses only from saved lightweight metadata/log text, aggregate both cell summaries with exact counts, and rank with:

  ```python
  (family_macro_success, micro_success, clean_rate, valid_action_rate,
   -loss_or_infinity, -step)
  ```

  `--limit` remains a diagnostic cap applied deterministically within each cell and never changes the shipped default. Write metadata-sized `selection.json` directly under R11. Do not call any model or stage command in tests.

- [ ] **Step 12: Run selection tests and capture GREEN**

  Run:

  ```bash
  uv run pytest -q tests/test_cli.py tests/test_selection.py
  ```

  Expected: config and selection tests pass without warnings.

- [ ] **Step 13: Add report tests for intervals and paired comparisons**

  In `tests/test_report.py`, write two lightweight evaluation JSON fixtures with identical task IDs, split, and difficulty. Assert the rendered report includes Wilson intervals next to every new rate and an exact paired McNemar row with pair count, directional flips, discordant count, and p-value. Assert evaluations with different task-id sets, split, or difficulty are not paired. Retain explicit legacy-only and mixed-summary cases that prove section-2's historical column behavior is unchanged.

- [ ] **Step 14: Run report tests and capture RED**

  Run:

  ```bash
  uv run pytest -q tests/test_report.py
  ```

  Expected: interval and paired comparison assertions fail against the existing summary-only table.

- [ ] **Step 15: Implement report intervals and pairing**

  Extend `load_summaries` to retain only the lightweight task outcome/difficulty metadata required for pairing. Extend `render` so new records render their supplied Wilson intervals and matching evaluation pairs append an exact McNemar section; do not fabricate pairs for non-identical task-id sets or different split/difficulty metadata. Preserve the current no-summary message and legacy-only report layout.

- [ ] **Step 16: Run focused and compatibility tests**

  Run:

  ```bash
  uv run pytest -q tests/test_tasks.py tests/test_evaluate.py tests/test_cli.py tests/test_report.py tests/test_selection.py tests/test_integrity.py tests/test_runner.py
  ```

  Expected: all focused section-1 tests and section-2 compatibility tests pass without warnings.

- [ ] **Step 17: Write the implementation report**

  Create `design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part3-selection.md` with: exact scope; changed paths and line counts; interface-map rows touched; selection artifact schema; decisions and compatibility choices; RED/GREEN evidence; focused and broad verification; generator version/hash result; section-2 compatibility evidence; banned-constant scan; no-model/protected-path confirmation; exact commits; and observed-but-not-fixed items. State that independent review is pending until the reviewer returns.

- [ ] **Step 18: Run broad fake-only verification and static checks**

  Confirm no known active writer is changing a test/runtime surface used by the command, then run:

  ```bash
  uv run pytest -q
  uv run ruff check src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/evaluate.py src/local_llm_lab/pipeline/cli.py src/local_llm_lab/pipeline/report.py tests/test_tasks.py tests/test_evaluate.py tests/test_cli.py tests/test_report.py tests/test_selection.py
  uv run ruff check --select C901 --config 'lint.mccabe.max-complexity=9' src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/evaluate.py src/local_llm_lab/pipeline/cli.py src/local_llm_lab/pipeline/report.py
  git diff --check -- configs/agent_v2.yaml configs/agent_v2b.yaml configs/agent_v2c.yaml src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/evaluate.py src/local_llm_lab/pipeline/cli.py src/local_llm_lab/pipeline/report.py tests/test_tasks.py tests/test_evaluate.py tests/test_cli.py tests/test_report.py tests/test_selection.py docs/superpowers/plans/2026-09-04-spec-002-selection.md design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part3-selection.md
  ```

  The full suite must remain fake-only; if another active lane makes it unsafe or changes HEAD during the run, record the evidence and rerun from a stable observed HEAD.

- [ ] **Step 19: Inspect and commit explicit owned paths only**

  Re-read the board and Git status. Verify the staged names and patch, leave every foreign staged/dirty path untouched, then commit only the exact files changed by this task:

  ```bash
  git add -- configs/agent_v2.yaml configs/agent_v2b.yaml configs/agent_v2c.yaml src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/evaluate.py src/local_llm_lab/pipeline/cli.py src/local_llm_lab/pipeline/report.py tests/test_tasks.py tests/test_evaluate.py tests/test_cli.py tests/test_report.py tests/test_selection.py docs/superpowers/plans/2026-09-04-spec-002-selection.md design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part3-selection.md
  git diff --cached --name-only
  git diff --cached --check
  git commit --only -m "feat: add family-balanced checkpoint selection" -- configs/agent_v2.yaml configs/agent_v2b.yaml configs/agent_v2c.yaml src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/evaluate.py src/local_llm_lab/pipeline/cli.py src/local_llm_lab/pipeline/report.py tests/test_tasks.py tests/test_evaluate.py tests/test_cli.py tests/test_report.py tests/test_selection.py docs/superpowers/plans/2026-09-04-spec-002-selection.md design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part3-selection.md
  ```

- [ ] **Step 20: Self-review and report**

  Inspect the committed diff against every Global Constraint, update the implementation report before the commit if evidence is stale, and report `DONE`, exact commit(s), focused/full test totals, static-check results, concerns, and the report path. Do not dispatch a reviewer; the controller dispatches exactly one independent principal reviewer after the implementation report is complete.
