# SPEC-002 Note-Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and independently verify the complete offline SPEC-002 §2 note-integrity diagnostic, including deterministic task reconstruction, evaluator and rollout integration, and the retroactive run-B-versus-run-C report.

**Architecture:** Keep integrity scoring pure and model-free in a new `pipeline.integrity` module. Rebuild generator ground truth from `(task_id, data_seed, difficulty)`, derive hidden facts by replaying expert actions in the deterministic simulator, compare policy notes against that ground truth, and attach the resulting dictionaries at the evaluator and rollout seams without taking ownership of SPEC-002 §4's future `Trajectory` dataclass expansion. The CLI reads existing evaluation JSON as immutable input and writes one small Markdown report.

**Tech Stack:** Python 3.13, frozen dataclasses, regular expressions, JSON, argparse, pytest, Ruff.

**Spec:** `design_specifications/pending/SPEC-002-evaluation-selection-note-integrity.md` §2, with binding signatures in `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` §2.4, §2.9, and rulings R1–R11 in §7.

## Global Constraints

- Work in the shared primary checkout on branch `codex/agent-v2-specs`; do not create or switch branches or worktrees.
- Modify only this task's active Coordinator claim: `src/local_llm_lab/pipeline/integrity.py`, `src/local_llm_lab/pipeline/tasks.py`, `src/local_llm_lab/pipeline/evaluate.py`, `src/local_llm_lab/pipeline/rollout.py`, `src/local_llm_lab/pipeline/report.py`, `tests/test_integrity.py`, `pyproject.toml`, `reports/note-integrity-B-vs-C.md`, this plan, `design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part2-note-integrity.md`, and this plan's ignored SDD ledger.
- Do not edit any document under `design_specifications/pending/`.
- Treat everything under `outputs/` and `data/` as read-only. The only authorised `reports/` write is the new `reports/note-integrity-B-vs-C.md` file.
- Do not load a model, checkpoint, tokenizer, real MLX module, probe, preflight, or J-space workload. Use generator/simulator ground truth, saved JSON evaluation records, and pure fakes only.
- Under R10, every new test goes in `tests/test_integrity.py`; do not edit `tests/test_pipeline.py` or `tests/test_probes.py`.
- Under R11, metadata-sized output may write directly. The Markdown report is expected to remain far below 1 MiB; if it can exceed 1 MiB, use a same-directory temporary file, flush, `fsync`, and `os.replace`.
- `tasks.py` ownership is limited to `Task.difficulty`, `difficulty(..., override=...)`, `make_tasks(..., difficulty=...)`, deterministic `task_from_id`, and the smallest private parsing/helper code those exact interfaces need. Do not implement the §1 selection screen or change templates, variants, random derivations, generated rows, or `GENERATOR_VERSION`.
- Public signatures from wiring-map §2.4 and §2.9 are exact, including keyword-only markers.
- The offline CLI defaults missing legacy `data_seed` and per-trajectory `difficulty` to the historical `20260902` seed and the generator's split mapping; new evaluations record `data_seed` and top-level per-trajectory `difficulty`/`integrity`.
- Keep issue #2 open. Mirror material progress, review findings, and completion evidence there.
- Preserve all foreign working-tree and index changes. Stage and commit explicit claimed paths only.

---

### Task 1: Deliver the complete offline note-integrity vertical

**Files:**

- Create: `src/local_llm_lab/pipeline/integrity.py`
- Modify: `src/local_llm_lab/pipeline/tasks.py`
- Modify: `src/local_llm_lab/pipeline/evaluate.py`
- Modify: `src/local_llm_lab/pipeline/rollout.py`
- Modify: `src/local_llm_lab/pipeline/report.py`
- Create: `tests/test_integrity.py`
- Modify: `pyproject.toml`
- Create: `reports/note-integrity-B-vs-C.md`
- Create: `design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part2-note-integrity.md`
- Include: `docs/superpowers/plans/2026-09-04-spec-002-note-integrity.md`
- Update, do not stage: `.superpowers/sdd/2026-09-04-spec-002-note-integrity/progress.md`

**Interfaces:**

- Consumes: deterministic task makers and `Simulator`; trajectory steps shaped as `{"index", "thought", "action", "observation", "raw"}`; saved evaluation JSON shaped as `{"summary": ..., "trajectories": [...]}`.
- Produces exactly:

  ```python
  @dataclass(frozen=True)
  class Fact:
      kind: str
      value: str
      observed_at: int

  def required_carry(task: Task, *, keep_last: int) -> dict[int, frozenset[Fact]]: ...

  @dataclass(frozen=True)
  class Violation:
      step: int
      kind: str
      detail: str

  @dataclass(frozen=True)
  class IntegrityReport:
      violations: tuple[Violation, ...]
      first_violation: Violation | None
      counts: dict[str, int]
      clean: bool
      def as_dict(self) -> dict[str, Any]: ...

  def check_trajectory(
      task: Task,
      steps: list[dict[str, Any]],
      *,
      keep_last: int,
  ) -> IntegrityReport: ...

  def completion_patterns() -> tuple[re.Pattern, ...]: ...
  def main() -> None: ...
  ```

- Integrates: `evaluate_tasks` scores after `run_task`; `failure_reason` checks the first integrity violation after parse errors and before loop/exhaustion/verdict reasons; `summarize` reports integrity counts by family and the fraction of failed trajectories carrying a violation; `collect_rollouts` admits only successful, integrity-clean candidates; `report.render` shows the nested integrity-clean rate while remaining compatible with old summaries.

- [x] **Step 1: Write task difficulty/reconstruction tests and verify RED**

  Add tests that name these breaks:

  - `difficulty("test", 3, override=1)` must return `1`, while `override=None` preserves the existing split mapping.
  - `make_tasks("test", 24, difficulty=3)` gives every returned task `difficulty == 3` without changing its deterministic task IDs or action steps for the same level.
  - For `train`, `valid`, `test`, and a fresh split, rebuild representative clean and recovery tasks with `task_from_id(task.task_id, seed, difficulty=task.difficulty)` and assert equality with the originals.
  - Invalid IDs, unknown families/variants, and structurally impossible variant IDs raise a task-named `ValueError` rather than silently returning a different variant.

  Run:

  ```bash
  uv run pytest -q tests/test_integrity.py -k 'difficulty or task_from_id'
  ```

  Expected: collection fails because the new signatures and `task_from_id` do not exist.

- [x] **Step 2: Implement only the §2 task prerequisite and verify GREEN**

  In `tasks.py`, retain the public function name while avoiding parameter shadowing through a small private resolver:

  ```python
  def _difficulty_level(split: str, index: int, override: int | None) -> int:
      if override is not None:
          if override < 0:
              raise ValueError("difficulty must be non-negative")
          return override
      return {"train": 0, "valid": 1, "test": 2}.get(split, index % 2)

  def difficulty(split: str, index: int, override: int | None = None) -> int:
      return _difficulty_level(split, index, override)
  ```

  Add `difficulty: int = -1` as a backward-compatible final `Task` field. In `make_tasks`, resolve one `level`, use it for both the maker and `applicable_variants`, and set it on the final task with `replace`. Add `difficulty: int | None = None` after the existing keyword-only parameters.

  `task_from_id` must parse with `rsplit("-", 3)` so a split may contain hyphens, validate the family/index/variant, initialise the exact existing RNG string `f"v2:{seed}:{split}:{index}"`, call the matching maker once at the resolved level, apply the parsed variant with the same post-maker RNG state, reject realised-variant mismatch, and return a `replace`d task with the original ID and resolved difficulty. It must not call `make_tasks(..., count=index + 1)`.

  Re-run the Step 1 command. Expected: selected tests pass and the existing generator-only and configured-replay hash tests in `tests/test_tasks.py` remain unchanged.

- [x] **Step 3: Write dataclass, completion-pattern, and required-carry tests and verify RED**

  Add tests with literal expected dictionaries for `IntegrityReport.as_dict()`, including `first_violation=None`, sorted count keys, and ordered violation records. Assert `completion_patterns()` recognises exactly the six specified forms case-insensitively: `(full)`, `(final)`, `complete`, `all <n> ... read|inspected|applied|verified`, `pending: none`, and `Task complete`.

  Build expert trajectory records by replaying generated tasks through the real `Simulator` in a test-only helper. Assert `required_carry`:

  - validates `keep_last >= 0`;
  - carries hidden approved `amount` values for `ledger_reconcile` but not held amounts;
  - carries hidden `metric` values until their subtotal use;
  - carries the current highest `load` as `service-N=<value>` for `conditional_update`;
  - carries hidden `worker` entries and `path` basenames used by listing-driven notes;
  - uses `next_key` and calculator `total` facts when they are hidden and still referenced;
  - marks each fact with the expert action index that produced its observation.

  Run:

  ```bash
  uv run pytest -q tests/test_integrity.py -k 'report_as_dict or completion_patterns or required_carry'
  ```

  Expected: import/collection fails because `pipeline.integrity` does not exist.

- [x] **Step 4: Implement the pure ground-truth engine and verify GREEN**

  Create `integrity.py` with `from __future__ import annotations`, explicit `__all__`, the exact frozen dataclasses/signatures, and no import of MLX, model, tokenizer, or probe code.

  Replay expert steps through `Simulator.for_task(task)` and retain observations only for non-`finish` actions. Extract facts from deterministic observations as follows:

  - approved invoice `amount=<n>` → `Fact("amount", "<n>", step)`; held invoice amounts are not carry requirements;
  - metric file `value=<n>` → `Fact("metric", "<n>", step)`;
  - service file `name=service-N` plus `load=<n>` → `Fact("load", "service-N=<n>", step)`;
  - record `Next-Key: <value>` → `Fact("next_key", "<value>", step)`;
  - calculator `RESULT: <value>` → `Fact("total", "<value>", step)`;
  - `FILES:`/`MATCHES:` observations → one basename `Fact("path", basename, step)` per path;
  - batch manifest lines `<path>|mode=<old>->mode=<new>` → normalised `Fact("worker", "<basename> mode=<old> -> mode=<new>", step)`.

  For target expert step `j`, an observation is hidden exactly when it falls before the final `keep_last` prior non-finish observations. Keep an extracted fact in `required_carry[j]` only when its independently normalised value is referenced by that step's canonical note or action arguments. Return every expert step key with a `frozenset`, including empty sets. Numeric matching must use token boundaries so `12` cannot satisfy `112`; worker/path matching normalises whitespace and uses basenames.

  Re-run the Step 3 command and then:

  ```bash
  uv run pytest -q tests/test_integrity.py -k 'required_carry'
  ```

  Expected: all selected tests pass with no output noise.

- [x] **Step 5: Write one isolated RED test for each violation kind**

  Starting from real clean expert traces, make one note-only mutation per case and assert that the named kind fires exactly once while the unmodified trace is clean:

  | Break planted | Required detection |
  | --- | --- |
  | copy the previous normalised note into the next step | `verbatim_copy` |
  | delete one hidden required amount/metric/load/next-key/total value from its note | `value_drop` |
  | insert a completion pattern before the corresponding canonical phase/queue is complete | `premature_completion` |
  | replace either number in a canonical `x of y` phrase | `count_mismatch` |
  | replace a structured approved/first-half/second-half/highest-so-far value with a contradictory value | `stale_fact` |
  | remove one filename/worker entry from an otherwise exact enumerated remaining set | `queue_loss` |

  Also test that visible observations satisfy a value even when it is absent from the note, malformed/parse-error steps do not crash the scorer, duplicate patterns at one step produce one violation for that check, `first_violation` is the earliest by step with table-order tie-breaking, and counts are exact.

  Run:

  ```bash
  uv run pytest -q tests/test_integrity.py -k 'check_trajectory'
  ```

  Expected: assertions fail because `check_trajectory` does not yet implement the six checks.

- [x] **Step 6: Implement the six checks minimally and verify GREEN**

  Implement each table check once per policy step in this order: `verbatim_copy`, `value_drop`, `premature_completion`, `count_mismatch`, `stale_fact`, `queue_loss`. Use the stored policy `index` when valid, otherwise the list position, and compare phase-sensitive claims against the same-index canonical expert note/action:

  - copy: compare whitespace-normalised notes;
  - drop: inspect non-path/non-worker facts from `required_carry`; suppress a drop when the exact fact remains in the last `keep_last` actual observations;
  - completion: treat any public completion-pattern match as premature only when the corresponding pattern is not yet licensed by the canonical ground-truth note at that step;
  - count: compare every policy `x of y` pair with the canonical pair for that step;
  - stale: compare structured `approved:`, `first half:`, `second half:`, and `highest so far:` fields with the canonical field when the policy states it; missing required values remain `value_drop`, not stale;
  - queue: extract basenames from `pending:` and `Remaining after this:` clauses and flag only a strict subset of the canonical remaining set.

  Store concise factual details naming the missing, contradictory, or omitted values. Sort by `(step, check-order)`, compute counts from the complete tuple, set `first_violation` to the first item, and set `clean = not violations`.

  Re-run the Step 5 command, then run all pure engine tests. Expected: all selected tests pass.

- [x] **Step 7: Write evaluator/report integration tests and verify RED**

  With fake `Trajectory` objects and monkeypatched `make_sampler`/`run_task` boundaries, without importing real MLX, assert:

  - `evaluate_tasks` annotates every returned trajectory with `difficulty` and `integrity` after `run_task`;
  - `failure_reason` orders parse error first, then the first integrity kind rendered with spaces, then loop, exhaustion, and verdict reasons;
  - `summarize` includes nested `integrity` totals, affected-trajectory counts by kind, per-family counts/clean rates, and `failure_explained_rate` whose denominator is failed trajectories only;
  - `write_report` serialises top-level `difficulty` and `integrity` per trajectory even though the current `Trajectory.as_dict()` predates SPEC-002 §4;
  - `report.render` adds an integrity-clean column for new summaries and prints `-` for legacy summaries.

  Run:

  ```bash
  uv run pytest -q tests/test_integrity.py -k 'evaluate or failure_reason or summarize or write_report or report_render'
  ```

  Expected: focused assertions fail because the integration is absent.

- [x] **Step 8: Wire evaluator and report integration and verify GREEN**

  Import `check_trajectory` into `evaluate.py`. After each `run_task`, set dynamic attributes
  `trajectory.difficulty = task.difficulty` and
  `trajectory.integrity = check_trajectory(task, trajectory.steps, keep_last=keep_last).as_dict()`.
  Keep this bridge local until SPEC-002 §4 owns the dataclass fields. In `write_report`, merge
  those two attributes into each serialised trajectory record without mutating the source.

  Extend `failure_reason` and `summarize` exactly as Step 7 specifies. Use zero-safe rates and
  deterministic sorted dictionaries. In `run_evaluation`, add `summary["data_seed"] = seed`.
  In `report.render`, add one backward-compatible integrity-clean column read from the nested
  summary.

  Re-run the Step 7 command. Expected: all selected tests pass.

- [x] **Step 9: Write the rollout integrity-filter test and verify RED**

  Use fake `mlx`/`mlx.core` modules inserted only in the test process, a fake random seeder, and
  monkeypatched `run_task`/`trajectory_rows`. Return two outcome-successful trajectories for one
  task: one note-clean and one with a planted violation. Assert both remain in the diagnostic
  trajectory list, but only the integrity-clean candidate contributes training rows and
  `kept_rows`; assert the violating success cannot affect duplicate-signature selection.

  Run:

  ```bash
  uv run pytest -q tests/test_integrity.py -k 'rollout'
  ```

  Expected: the violating successful trajectory is incorrectly retained.

- [x] **Step 10: Require integrity cleanliness in rollout and verify GREEN**

  In `collect_rollouts`, score every trajectory immediately after `run_task`, attach the same
  dynamic `difficulty`/`integrity` attributes used by evaluation, and append to candidate
  rollouts only when both `trajectory.success` and `integrity.clean` are true. Keep every sampled
  trajectory in `all_trajectories` so diagnostics and pass@k remain auditable; define a task as
  solved for retained-supervision purposes only when it has at least one eligible candidate.

  Re-run the Step 9 command. Expected: the clean trajectory is retained and the violating one is
  excluded.

- [x] **Step 11: Write CLI and retroactive-report tests and verify RED**

  Add the `agent-v2-integrity = "local_llm_lab.pipeline.integrity:main"` entry point. Test a
  small temporary evaluation JSON end to end through a pure helper used by `main`, asserting
  deterministic Markdown with source, seed, keep-last, overall and per-family tables, violation
  counts by affected trajectory, integrity-clean paired flips, outcome paired flips, and the
  three memo acceptance rows.

  Add an artifact-dependent test that skips only when either saved evaluation is absent. It must
  read both files in place, write only to `tmp_path`, and assert these independently specified
  run-C affected-trajectory counts:

  ```python
  {
      ("cross_reference", "verbatim_copy"): 15,
      ("batch_update", "premature_completion"): 14,
      ("ledger_reconcile", "value_drop"): 6,
  }
  ```

  It must also assert the report reproduces the recorded success differences B→C:
  `cross_reference 14/15→0/15`, `ledger_reconcile 13/15→9/15`, and the complete C failure-family
  counts `aggregate_report=15`, `batch_update=14`, `conditional_update=12`,
  `cross_reference=15`, `ledger_reconcile=6`. These expected values are literals from the
  decision memo/saved-summary contract, never computed by the implementation under test.

  Run:

  ```bash
  uv run pytest -q tests/test_integrity.py -k 'cli or retroactive'
  ```

  Expected: the entry/helper/report assertions fail until the offline reporting path exists.

- [x] **Step 12: Implement the offline CLI, calibrate from causes, and generate the authorised report**

  `main` accepts repeatable required `--eval PATH`, required `--output PATH`, and optional
  `--seed` defaulting to `20260902`. For each input, use `summary.data_seed` when present or the
  CLI seed for legacy files; use `summary.keep_last` per evaluation; rebuild each task with
  `task_from_id(record["task_id"], seed, difficulty=record.get("difficulty"))`; never trust saved
  success summaries where the raw trajectory record can answer the question.

  Count a violation kind once per affected trajectory in the report even when the detailed
  `IntegrityReport.counts` contains repeated per-step instances. Pair evaluations by identical
  `task_id`; fail with a clear `ValueError` when the sets differ. Produce deterministic Markdown
  sections for sources/config, overall, per-family, violation kinds, integrity-clean flips,
  outcome flips, and memo acceptance checks. Never hard-code a PASS result: calculate observed
  counts, print expected/observed/status, and let the artifact-dependent test fail on mismatch.

  Run the Step 11 command. If a count differs, use systematic debugging: inspect the first
  mismatched task's ground-truth facts, canonical note, visible observations, policy note, and
  violations; adjust the extractor or semantic check only when that evidence identifies the
  root cause. Do not special-case task IDs, run labels, or expected aggregate counts.

  When green, generate exactly the authorised report:

  ```bash
  uv run agent-v2-integrity --eval outputs/agent-v2b/evals/runB-test180.json --eval outputs/agent-v2c/evals/best-adapter-test.json --output reports/note-integrity-B-vs-C.md
  ```

  Confirm the two evaluation inputs have no Git or filesystem-content changes and the report is
  below the R11 threshold.

- [x] **Step 13: Write the §2 implementation report**

  Create `design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part2-note-integrity.md`
  with: exact scope; files and line counts; every touched wiring-map §2.4/§2.9/§4 row and caller;
  task-rebuild compatibility decisions; each check's semantics; evaluator/rollout/report wiring;
  retroactive B-vs-C tables and acceptance counts; TDD RED/GREEN commands and results; banned-
  constant results; deviations/ambiguities; observed-not-fixed defects; and a statement that
  pending documents and protected inputs were not modified and no model/checkpoint/tokenizer/
  MLX/probe/preflight/J-space workload ran. Leave SPEC-002 and issue #2 open.

- [x] **Step 14: Run final focused verification and commit only reviewed claimed paths**

  Run:

  ```bash
  uv run pytest -q tests/test_integrity.py tests/test_tasks.py
  uv run ruff check src/local_llm_lab/pipeline/integrity.py src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/evaluate.py src/local_llm_lab/pipeline/rollout.py src/local_llm_lab/pipeline/report.py tests/test_integrity.py
  git diff --check -- src/local_llm_lab/pipeline/integrity.py src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/evaluate.py src/local_llm_lab/pipeline/rollout.py src/local_llm_lab/pipeline/report.py tests/test_integrity.py pyproject.toml reports/note-integrity-B-vs-C.md docs/superpowers/plans/2026-09-04-spec-002-note-integrity.md design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part2-note-integrity.md
  git diff -U0 ac45f7a1a5059ac5b2ee3bbe38dfc21a0e25df89 -- src/local_llm_lab/pipeline/integrity.py src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/evaluate.py src/local_llm_lab/pipeline/rollout.py src/local_llm_lab/pipeline/report.py | grep -E '^\+.*(model\.model\.layers|<\|im_end\|>|(^|[^0-9])36([^0-9]|$)|(^|[^0-9])2048([^0-9]|$)|(^|[^0-9])35([^0-9]|$))' || true
  ```

  Record exact exit codes and counts. Inspect `git status --short`, the current index, and the
  complete diff from the recorded base. Stage only the ten tracked claimed product/document
  paths; never stage the Coordinator board, ignored SDD workspace, pending documents, inputs, or
  foreign changes. Use an explicit path-limited commit so a foreign staged path cannot enter:

  ```bash
  git commit --only -m "feat: add SPEC-002 note integrity" -- src/local_llm_lab/pipeline/integrity.py src/local_llm_lab/pipeline/tasks.py src/local_llm_lab/pipeline/evaluate.py src/local_llm_lab/pipeline/rollout.py src/local_llm_lab/pipeline/report.py tests/test_integrity.py pyproject.toml reports/note-integrity-B-vs-C.md docs/superpowers/plans/2026-09-04-spec-002-note-integrity.md design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part2-note-integrity.md
  ```

  The controller will package the exact base-to-head diff for the independent principal review.
