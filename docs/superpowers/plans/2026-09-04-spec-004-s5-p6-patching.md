# SPEC-004 §5 P6 Patching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the gated, fake-tested P6 causal-patching CLI and exact activation replacement needed to localise dropped-value errors by residual layer and prompt position group.

**Architecture:** Extend the existing `InjectionHook` at its `ArchitectureView.run_block` seam so selected residual rows can be replaced exactly, without changing the established additive path or cache-offset accounting. Build `probes.patch` as a deterministic pipeline with pure seams for saved-evaluation selection, transcript replay, token grouping, greedy view-based generation, integrity scoring, control construction, aggregation, and JSON/Markdown rendering; the CLI alone performs policy loading and GPU guarding.

**Tech Stack:** Python 3.13, MLX arrays behind fake-compatible seams, argparse, JSON, dataclasses, pytest, Ruff.

**Spec:** `design_specifications/pending/SPEC-004-probe-program-revision.md` §5 and §7, with binding interfaces in `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` §2.13, §2.15, §2.16 and rulings R10, R11, R13.

## Global Constraints

- Work in the shared primary checkout on branch `codex/agent-v2-specs`; do not create or switch branches or worktrees.
- Modify only this task's active Coordinator claim: `src/local_llm_lab/probes/patch.py`, `tests/test_patch.py`, `src/local_llm_lab/probes/capture.py`, `tests/test_capture.py`, `pyproject.toml`, this plan, `design_specifications/under_review/SPEC-004-S5-IMPLEMENTATION-REPORT.md`, and this plan's ignored SDD workspace.
- Preserve all foreign working-tree and index changes. Stage and commit explicit claimed paths only; do not push.
- Do not edit any document under `design_specifications/pending/` or any file under `research/`.
- Do not write to `outputs/`, `data/`, or `reports/`. Unit tests use `tmp_path`; the real P6 CLI is implemented but never executed in this lane.
- Do not load a model, checkpoint, tokenizer, adapter, real probe workload, preflight, cache-equivalence workload, or J-space workload. All execution evidence is fake-only.
- R10: new capture tests remain in `tests/test_capture.py`; all new P6 tests go in `tests/test_patch.py`. Do not edit or duplicate tests in monolithic test files.
- R11: P6 JSON and Markdown contain aggregate metadata only, no arrays, and are expected to remain below 1 MiB; metadata-sized direct writes are permitted.
- R13: before review or hand-off, `uv run pytest -q` must exit 0. The implementation report must contain a section titled exactly `R13 full fake-only suite` recording exact HEAD, command, exit code, pass/fail/xfail counts, and every failing node (write `none` when there are none).
- P6 remains gated: implement and test on fakes, then stop. Every result records the resolved `ModelSpec`, the named control, the full command line, the deterministic seed, and Wilson 95% intervals over the task set.
- Compare only task IDs that pass under B and have a first `value_drop` under C, restricted to `aggregate_report` and `ledger_reconcile`; reconstruct tasks from the failing evaluation's recorded `data_seed` and trajectory `difficulty`.
- The counterfactual context is C's own transcript before the first dropped-value note with only the immediately preceding C note replaced by `tasks.render_expert_note(task, decision_step - 1)`.
- Position groups are named exactly `system_prompt`, `task_prompt`, `previous_notes`, `note_value_tokens`, `last_two_observations`, and `final_token`. Group positions are deterministic absolute token indices. `note_value_tokens` is a subset of the substituted canonical note and `final_token` is the prompt's last token.
- Each layer/group treatment patches counterfactual residual rows into the failing context. Controls are `unrelated_task` residuals and a seeded `random_positions` group of the same cardinality. A flip means the generated decision no longer receives a `value_drop` violation from `integrity.check_trajectory`.
- Residuals and replacement vectors stay float32; every manual forward uses `ArchitectureView.masks`, never `None`.

---

### Task 1: Deliver the complete P6 fake-tested vertical

**Files:**

- Modify: `src/local_llm_lab/probes/capture.py`
- Modify: `tests/test_capture.py`
- Create: `src/local_llm_lab/probes/patch.py`
- Create: `tests/test_patch.py`
- Modify: `pyproject.toml`
- Create: `design_specifications/under_review/SPEC-004-S5-IMPLEMENTATION-REPORT.md`
- Include: `docs/superpowers/plans/2026-09-04-spec-004-s5-p6-patching.md`
- Update, do not stage: `.superpowers/sdd/2026-09-04-spec-004-s5-p6-patching/progress.md` and `task-1-report.md`

**Interfaces:**

- Consumes: `ArchitectureView`, `capture_residuals`, `InjectionHook`, `load_policy`, `load_model_spec`, `resolve_policy`, `build_prompt`, `assistant_message`, `tool_message`, `parse_turn`, `strip_thinking`, `turn_is_complete`, `check_trajectory`, `task_from_id`, `render_expert_note`, and `evaluate.wilson`.
- Preserves the declared capture constructor exactly:

  ```python
  class InjectionHook:
      def __init__(
          self,
          view: ArchitectureView,
          layer: int,
          vector: Any,
          *,
          alpha: float = 1.0,
          from_position: int | None = None,
          at_positions: Sequence[int] | None = None,
          replace: bool = False,
          positions: str | int | tuple[str, int] | None = None,
      ) -> None: ...
  ```

- `replace=False` remains byte-for-behaviour additive: selected rows become `out + alpha * vector`. With `replace=True`, selected rows become the supplied source residual rows exactly; `alpha` is rejected unless it is the default `1.0`. A one-dimensional vector broadcasts to every selected row. A two-dimensional vector maps row-for-row to the ordered `at_positions`, must match their count and hidden width, and supports cached slices by retaining the absolute-position-to-source-row mapping. Non-selected rows, dtype, call counters, offsetless-cache accounting, and restoration-on-exit remain unchanged.
- Produces these stable P6 seams in `patch.py` (private helpers may support them):

  ```python
  POSITION_GROUPS = (
      "system_prompt",
      "task_prompt",
      "previous_notes",
      "note_value_tokens",
      "last_two_observations",
      "final_token",
  )

  @dataclass(frozen=True)
  class PatchCase:
      task: Task
      decision_step: int
      failing_steps: tuple[dict[str, Any], ...]

  def select_patch_cases(
      passing_payload: dict[str, Any],
      failing_payload: dict[str, Any],
      *,
      keep_last: int,
  ) -> list[PatchCase]: ...

  def position_groups(
      tokenizer: Any,
      prompt_ids: Sequence[int],
      *,
      system_text: str,
      task_text: str,
      previous_notes: Sequence[str],
      note_values: Sequence[str],
      observations: Sequence[str],
  ) -> dict[str, tuple[int, ...]]: ...

  def greedy_generate(
      view: ArchitectureView,
      tokenizer: Any,
      token_ids: Sequence[int],
      *,
      max_tokens: int,
  ) -> str: ...

  def run_patch_probe(
      model: Any,
      tokenizer: Any,
      cases: Sequence[PatchCase],
      *,
      spec: ModelSpec,
      resolved: ResolvedSpec,
      layers: Sequence[int],
      policy: str,
      keep_last: int,
      max_tokens: int,
      seed: int,
      command: Sequence[str],
  ) -> dict[str, Any]: ...

  def render_markdown(payload: dict[str, Any]) -> str: ...
  def main() -> None: ...
  ```

- `greedy_generate` uses `view.make_cache()`, `view.embed`, `view.masks`, `view.run_block`, `view.final_norm`, and `view.unembed`; it takes an argmax token at temperature zero and stops when `turn_is_complete` becomes true or the exact token budget is consumed. The prompt is forwarded once and later calls forward only the generated token, allowing the hook's absolute cache offsets to select prompt rows.
- `main` accepts required `--passing-eval`, `--failing-eval`, and `--output`; common `--model`, `--policy`, `--layers`, and `--allow-busy-gpu`; plus `--keep-last` (default 2), `--max-tokens` (default 200), and `--seed` (default 20260904). `--layers` accepts comma-separated residual indices in `[1, num_layers]` or fractions in `(0, 1]`, defaulting to `resolved.probe_layers`. The output directory receives `patch.json` and `patch.md`.

- [ ] **Step 1: Write exact-replacement tests and verify RED**

  Add tests that name these breaks:

  - a selected row whose original value is `1` and replacement is `7` becomes `7`, not `8`;
  - a two-row replacement maps distinct vectors to two ordered absolute positions across cached calls;
  - non-selected rows, output dtype, `calls`, `injected`, offsetless-cache tracking, and restoration after an exception are preserved;
  - hidden-width mismatch, two-dimensional row-count mismatch, and `replace=True, alpha != 1.0` raise clear `ValueError`s;
  - the existing additive tests remain unchanged and green.

  Run:

  ```bash
  uv run pytest -q tests/test_capture.py -k replace
  ```

  Expected: RED because `InjectionHook` raises `NotImplementedError` for `replace=True`.

- [ ] **Step 2: Implement the smallest exact-replacement branch and verify GREEN**

  Store `replace`, preserve ordered deduplicated `at_positions`, validate the replacement rank at construction, and validate row count/hidden width against the selected output at application. Build the selected local absolute positions once per intercepted call. Keep the current additive expression untouched for `replace=False`; for replacement use float32 while assigning source rows, then cast the complete result back to `out.dtype`.

  Re-run Step 1, then all of `tests/test_capture.py`. Expected: all capture tests pass.

- [ ] **Step 3: Write selection, replay, grouping, generation, scoring, and aggregation tests and verify RED**

  In `tests/test_patch.py`, use literal saved-evaluation dictionaries and small fake tasks/views/tokenizers. Cover:

  - selection intersects B-success and C-`value_drop` task IDs, keeps only the two required families, uses failing `data_seed`/`difficulty`, chooses the first dropped-value step, and rejects malformed/mismatched payloads;
  - replay changes only the immediately previous note to `render_expert_note(...)` and leaves system text, task text, older notes, actions, and observations identical;
  - a synthetic token stream yields all six exact group tuples, value-token positions are inside the canonical substituted note, last-two-observation positions exclude older observations, and missing/ambiguous token spans fail closed;
  - seeded random groups are deterministic, have the treatment group's exact cardinality, contain unique valid positions, and differ from the treatment positions when enough alternatives exist;
  - fake greedy generation calls the model-native mask seam for each forward, forwards the prompt once, then one token at a time, uses argmax, and honours both completion and token-budget stops;
  - a generated note is a flip only when `check_trajectory` no longer reports `value_drop` at the decision step; parse errors and remaining drops are non-flips;
  - every treatment cell has both named controls; task-level booleans aggregate to rates and Wilson intervals without counting multiple generations from one task as independent observations;
  - JSON-ready output records `resolved.as_dict()`, policy, layers, seed, complete command, group names, control names, selected task IDs, counts, rates, intervals, and no activation arrays; Markdown renders a layer×group treatment heat map with interval text and separate control tables.

  Run:

  ```bash
  uv run pytest -q tests/test_patch.py
  ```

  Expected: collection fails because `local_llm_lab.probes.patch` does not exist.

- [ ] **Step 4: Implement the pure P6 pipeline and verify GREEN**

  Load evaluation JSON through a validating helper. Reconstruct each selected task with `task_from_id`, replay C records before the failing decision through `assistant_message`/`tool_message`, and build both prompts with the selected registry `spec`. Locate content token spans deterministically in left-to-right order; do not silently guess around a missing or duplicated span.

  Capture full float32 residuals for failing/counterfactual prompts in one traversal each. Residual layer `L` maps to hook block `L - 1`. For a treatment, pass counterfactual rows for the source group into `InjectionHook(..., at_positions=failing_group, replace=True)` and generate from failing IDs. For `unrelated_task`, rotate the selected case list by one task and deterministically truncate/cycle source rows to the target cardinality. For `random_positions`, sample the same number of counterfactual source and failing target positions from valid non-treatment indices with `random.Random(f"{seed}:{task_id}:{layer}:{group}")`; fail closed if cardinality cannot be satisfied.

  Parse each generated turn, splice only its thought into a copy of the failing prefix at `decision_step`, and score with `check_trajectory`. Preserve one boolean per task/cell/control, then compute numerator, denominator, rate, and `wilson(numerator, denominator)` over tasks. Return only JSON-safe summaries and metadata.

  Re-run Step 3. Expected: all P6 unit tests pass with no model or checkpoint loading.

- [ ] **Step 5: Write CLI and entry-point tests and verify RED**

  Monkeypatch loading, policy resolution, GPU guard, and `run_patch_probe` so the tests cannot reach a checkpoint. Assert:

  - `agent-v2-probe-patch = "local_llm_lab.probes.patch:main"` resolves through installed project metadata;
  - both registry model name and policy name reach their resolvers, the resolved spec reaches the probe, and the output files contain the returned payload/Markdown;
  - invalid positive counts, malformed layers, empty selections, missing inputs, and invalid output shape exit through `parser.error` before model loading;
  - the captured command is `sys.argv` exactly and includes every caller-supplied flag.

  Run:

  ```bash
  uv run pytest -q tests/test_patch.py -k cli
  ```

  Expected: RED because the CLI wiring and entry point are absent.

- [ ] **Step 6: Wire the gated CLI and verify GREEN**

  Add exactly this script entry:

  ```toml
  agent-v2-probe-patch = "local_llm_lab.probes.patch:main"
  ```

  Validate paths and scalar arguments before `require_idle_gpu`. Load the registry declaration with `load_model_spec(args.model)`, resolve the adapter with `resolve_policy(args.policy)`, load `spec.hf_id`, create `ArchitectureView.from_model(model)`, resolve the model, parse layers against its depth, and call `run_patch_probe`. Write deterministic UTF-8 `patch.json` and `patch.md`, each ending in a newline. The implementation must not execute the CLI in this lane.

  Re-run Step 5 and all focused tests. Expected: all focused tests pass.

- [ ] **Step 7: Inspect, verify, report, and commit the owned slice**

  Run in this order:

  ```bash
  uv run pytest -q tests/test_capture.py tests/test_patch.py
  uv run ruff check src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py tests/test_capture.py tests/test_patch.py
  uv run ruff check --select C901 src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py
  python3 -m py_compile src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py tests/test_capture.py tests/test_patch.py
  git diff --check -- src/local_llm_lab/probes/capture.py src/local_llm_lab/probes/patch.py tests/test_capture.py tests/test_patch.py pyproject.toml docs/superpowers/plans/2026-09-04-spec-004-s5-p6-patching.md design_specifications/under_review/SPEC-004-S5-IMPLEMENTATION-REPORT.md
  uv run pytest -q
  ```

  If the full suite is red, do not request review or hand off; diagnose and fix only owned causes. Record foreign failures exactly and return `BLOCKED` without releasing the claim if no owned fix exists.

  Create `design_specifications/under_review/SPEC-004-S5-IMPLEMENTATION-REPORT.md` with scope, decisions, RED/GREEN evidence, focused/static/import evidence, no-model evidence, protected-path evidence, commit list, and the exact R13 section required by Global Constraints. Append equivalent concise evidence to the ignored SDD report/ledger. Inspect the owned diff and commit only explicit claimed tracked paths; never broad-stage.

