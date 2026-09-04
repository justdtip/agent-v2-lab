# SPEC-002 Section 4 Remainder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the fake-only runner/evaluator remainder of SPEC-002 section 4 with observable loop, stress, thinking-token, and integrity-summary coverage.

**Architecture:** Keep the existing runner/evaluator boundary: `evaluate_tasks` selects the deterministic stress fault tuple and passes it to `runner.run_task`; `runner.detect_loop` remains a pure post-hoc detector. Extend evaluator aggregation without changing latency or generated-token metrics, and preserve the existing structured integrity summary.

**Tech Stack:** Python 3.13, dataclasses, pytest, Ruff, uv

**Spec:** `design_specifications/pending/SPEC-002-evaluation-selection-note-integrity.md` section 4, under ratified rulings in `design_specifications/pending/01-IMPLEMENTER-BRIEFING.md`, `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md`, and `design_specifications/under_review/CODEX-COORDINATOR-NOTE-2026-09-04.md`

## Global Constraints

- Stay in `/Users/daniel.tipton/Desktop/An app` on the existing `codex/agent-v2-specs` branch; do not create or switch branches or worktrees.
- Modify only the seven paths in this task's Coordinator claim.
- Do not edit `src/local_llm_lab/pipeline/cli.py`; Lane D owns the 180-task `all`-stage `--stress`/`--base` wiring.
- Do not run model, checkpoint, training, selection, evaluation, rollout, branch, preference, preflight, probe, or stress commands; verification must use fakes only.
- Do not write under `data/`, `outputs/`, `reports/`, `design_specifications/pending/`, or `design_specifications/research/`.
- Preserve public signatures and the existing latency and generated-token metrics.
- Use strict RED-GREEN-REFACTOR evidence: every production behavior begins with a focused failing test.
- The implementation report must contain a section titled `R13 full fake-only suite` with the exact tested HEAD, command `uv run pytest -q`, exit code, counts, and failing node IDs; do not hand off unless it is green.
- Stage only explicit owned paths and preserve all foreign dirty and staged changes.
- One principal implementer owns this task, followed by one independent principal reviewer; neither role may delegate.

---

### Task 1: Runner and evaluator section 4 completion

**Files:**
- Modify: `src/local_llm_lab/pipeline/runner.py:29-35,312-346`
- Modify: `src/local_llm_lab/pipeline/evaluate.py:159-200,281-330`
- Test: `tests/test_runner.py`
- Test: `tests/test_evaluate.py`
- Create: `design_specifications/under_review/SPEC-002-S4-IMPLEMENTATION-REPORT.md`
- Create: `.superpowers/sdd/2026-09-04-spec-002-s4-remainder/task-1-report.md`

**Interfaces:**
- Consumes: `detect_loop(steps: list[dict[str, Any]]) -> bool`, `evaluate_tasks(..., stress: bool = False) -> list[Trajectory]`, `summarize(trajectories: list[Trajectory]) -> dict[str, Any]`, `Trajectory.think_tokens`, and `Trajectory.integrity`.
- Produces: rule 2 loop detection after four consecutive same-tool error observations; `summarize()["think_tokens"]` as the total thinking-token count; `summarize()["think_tokens_per_task"]` as the rounded mean across all trajectories; unchanged structured `summarize()["integrity"]`; fake-only proof that stress mode passes `STRESS_FAULTS` to `runner.run_task` and clean mode passes `None`.

- [x] **Step 1: Write the loop-threshold failing test**

  Add an observable test in `tests/test_runner.py` that builds four `read_file` actions with distinct `path` arguments and `ERROR` observations. Assert that the first three steps are not a loop and all four are a loop, so the identical-three-call rule cannot satisfy the test accidentally.

  ```python
  def test_detect_loop_flags_four_same_tool_errors_but_not_three() -> None:
      from local_llm_lab.pipeline.runner import detect_loop

      errors = [
          {
              "action": {"name": "read_file", "arguments": {"path": f"missing-{index}"}},
              "observation": f"ERROR missing-{index}",
          }
          for index in range(4)
      ]

      assert detect_loop(errors[:3]) is False
      assert detect_loop(errors) is True
  ```

- [x] **Step 2: Run the loop test and capture RED**

  Run:

  ```bash
  uv run pytest -q tests/test_runner.py::test_detect_loop_flags_four_same_tool_errors_but_not_three
  ```

  Expected: exit 1 because the current same-tool-error window is six.

- [x] **Step 3: Implement the four-error rule**

  Change `LOOP_SAME_TOOL_ERRORS` from `6` to `4` and update `detect_loop`'s rule-2 docstring from “last 6 calls” to “last 4 calls”. Do not change the identical-call or same-shape rules.

- [x] **Step 4: Run the loop test and capture GREEN**

  Run the Step 2 command again. Expected: exit 0 with one passing test.

- [x] **Step 5: Write the stress-boundary failing test**

  Add a parametrized unit test in `tests/test_evaluate.py`. Monkeypatch `evaluate.make_sampler`, `evaluate.run_task`, and `evaluate.check_trajectory`; use one in-memory `Task` and a fake `Trajectory`. Invoke `evaluate.evaluate_tasks` once with `stress=False` and once with `stress=True`, and independently assert that the captured `faults` argument is respectively `None` and `evaluate.STRESS_FAULTS`. No model API may be imported or executed.

  ```python
  @pytest.mark.parametrize(
      ("stress", "expected_faults"),
      [(False, None), (True, evaluate.STRESS_FAULTS)],
  )
  def test_evaluate_tasks_passes_stress_faults_to_runner(
      monkeypatch, stress: bool, expected_faults: object
  ) -> None:
      captured: dict[str, object] = {}
      task = Task(
          task_id="test-read-0000-clean",
          family="read",
          variant="clean",
          prompt="read a file",
          files={"a.txt": "x"},
          steps=(),
          expected_answer="x",
          required_tools=frozenset(),
      )

      monkeypatch.setattr(evaluate, "make_sampler", lambda _temperature: object())

      def fake_run_task(*_args, **kwargs):
          captured["faults"] = kwargs["faults"]
          return _trajectory(task.task_id, success=True, clean=True, difficulty=0)

      monkeypatch.setattr(evaluate, "run_task", fake_run_task)
      monkeypatch.setattr(
          evaluate,
          "check_trajectory",
          lambda *_args, **_kwargs: SimpleNamespace(
              as_dict=lambda: {"clean": True, "counts": {}}
          ),
      )

      evaluate.evaluate_tasks(
          object(), object(), [task], label="fake", stress=stress, quiet=True
      )

      assert captured["faults"] == expected_faults
  ```

  Import `SimpleNamespace` and `Task` in the test module. If this test is already green because the seam is correctly implemented, record it as characterization evidence and do not manufacture a production change.

- [x] **Step 6: Run the stress-boundary test**

  Run:

  ```bash
  uv run pytest -q tests/test_evaluate.py::test_evaluate_tasks_passes_stress_faults_to_runner
  ```

  Expected: two passing parametrized cases, or a genuine RED if the existing boundary differs. If RED, make the smallest change in `evaluate_tasks` needed to pass `STRESS_FAULTS` only in stress mode, then rerun to GREEN.

- [x] **Step 7: Write aggregate thinking and integrity failing assertions**

  Extend the `_trajectory` test helper with an explicit `think_tokens: int = 0` parameter. Add a summary test using two literal trajectories with thinking counts `3` and `8`; give the first a clean empty integrity record and the second `{"clean": False, "counts": {"value_drop": 2}}`. Assert independent literal expected values:

  ```python
  assert summary["think_tokens"] == 11
  assert summary["think_tokens_per_task"] == 5.5
  assert summary["integrity"] == {
      "clean_trajectories": 1,
      "clean_rate": 0.5,
      "affected_trajectories": 1,
      "violations": 2,
      "by_kind": {"value_drop": {"violations": 2, "affected_trajectories": 1}},
      "by_family": {
          "read": {
              "trajectories": 2,
              "clean_trajectories": 1,
              "clean_rate": 0.5,
              "violations": 2,
              "affected_trajectories": 1,
          }
      },
      "failed_trajectories": 1,
      "failed_with_violation": 1,
      "failure_explained_rate": 1.0,
  }
  ```

- [x] **Step 8: Run the aggregate test and capture RED**

  Run the new summary test by exact node ID. Expected: exit 1 because `think_tokens` and `think_tokens_per_task` are absent; the integrity literal must already match independently.

- [x] **Step 9: Implement aggregate thinking metrics**

  Add a `think_tokens` accumulator to `_trajectory_totals`, summing each trajectory's `think_tokens`. Add `think_tokens` and `think_tokens_per_task` to `summarize`; use the existing `_ratio(..., digits=2)` convention for a zero-safe per-task mean. Do not alter `generated_tokens`, `tokens_per_success`, or latency calculations.

- [x] **Step 10: Run focused GREEN tests**

  Run:

  ```bash
  uv run pytest -q tests/test_runner.py tests/test_evaluate.py
  ```

  Expected: all tests pass.

- [x] **Step 11: Run scoped static and diff checks**

  Run:

  ```bash
  uv run ruff check src/local_llm_lab/pipeline/runner.py src/local_llm_lab/pipeline/evaluate.py tests/test_runner.py tests/test_evaluate.py
  uv run ruff check --select C901 src/local_llm_lab/pipeline/runner.py src/local_llm_lab/pipeline/evaluate.py tests/test_runner.py tests/test_evaluate.py
  git diff --check
  ```

  Expected: every command exits 0.

- [x] **Step 12: Run the R13 full fake-only suite**

  Record the exact `git rev-parse HEAD`, then run exactly:

  ```bash
  uv run pytest -q
  ```

  Expected: exit 0. Record the exact count and an empty failing-node list. If any test fails, keep the task with the implementer and do not request review or hand off.

- [x] **Step 13: Write the implementation evidence**

  Create `design_specifications/under_review/SPEC-002-S4-IMPLEMENTATION-REPORT.md` and the SDD task report. Include scope and files, RED/GREEN node IDs and outputs, stress characterization result, focused pytest, Ruff, C901, `git diff --check`, exact diff-name checks, proof that no model/training/evaluation command ran, proof that no protected path changed, and a section titled exactly `R13 full fake-only suite` with tested HEAD, exact command, exit, counts, and failing nodes.

- [x] **Step 14: Stage and commit only owned paths**

  Inspect `git diff --name-only`, the protected-path diff, and the pre-existing index. Stage only these explicit paths:

  ```bash
  git add -- src/local_llm_lab/pipeline/runner.py src/local_llm_lab/pipeline/evaluate.py tests/test_runner.py tests/test_evaluate.py docs/superpowers/plans/2026-09-04-spec-002-s4-remainder.md design_specifications/under_review/SPEC-002-S4-IMPLEMENTATION-REPORT.md .superpowers/sdd/2026-09-04-spec-002-s4-remainder
  git diff --cached --name-only
  git diff --cached --check
  git commit -m "feat: complete SPEC-002 runner evaluation metrics"
  ```

  Expected: the staged list contains only the seven claimed path boundaries and the commit succeeds without absorbing foreign changes.
