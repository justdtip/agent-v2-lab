# Rollout Model-Aware Caller Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate only the rollout policy-loading caller so one active registry `ModelSpec` reaches the released evaluator loader and every rollout `run_task` prompt/render path, with mutation-sensitive fake-only verification and no compatibility fallback.

**Architecture:** Preserve `run_rollout`'s current `model_name: str` entry point for the out-of-scope pipeline CLI, resolve it once with `load_model_spec`, and pass that identical declaration to the released four-value `load_policy` contract. Make `collect_rollouts` require the returned `spec`, `ArchitectureView`, and `ResolvedSpec`, forward their exact identities to every `run_task`, and record resolved model metadata in the summary. The test harness replaces only model/runtime boundaries and fails if any context object is dropped, substituted, or reconstructed.

**Tech Stack:** Python 3.13, pytest, Ruff, existing `ModelSpec`/`ResolvedSpec`, `ArchitectureView`, and process-local fake `mlx`/`mlx_lm` boundaries.

**Spec:** `design_specifications/pending/SPEC-001-model-agnostic-backbone.md` §§1, 4, and 9; exact contracts and wiring come from `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` §§2.1, 2.3, 2.7, 2.10, 4.2, and rulings R10/R13. The Coordinator assignment narrows ownership to rollout only and makes native task `01a06844-b9b9-7bc2-bbf4-847fde9e8732`'s committed evaluator contract a hard dependency.

## Global Constraints

- Work in the existing primary checkout on branch `codex/agent-v2-specs`; do not create or switch branches or worktrees.
- Modify only `src/local_llm_lab/pipeline/rollout.py`, `tests/test_rollout.py`, this plan, `design_specifications/under_review/SPEC-001-ROLLOUT-MODEL-AWARE-CALLER-IMPLEMENTATION-REPORT.md`, and this plan's `.superpowers/sdd/2026-09-04-rollout-model-aware-load-policy/progress.md` ledger.
- Do not edit `pipeline/evaluate.py`, `pipeline/runner.py`, `pipeline/protocol.py`, `pipeline/cli.py`, monolithic tests, or any other caller. If an out-of-scope production caller or an R10 subtractive removal is required, record the exact path and notify the Coordinator instead of broadening.
- Do not load or execute a real model, checkpoint, tokenizer, adapter, rollout, evaluation, training, probe, preflight, cache-equivalence, or J-space workload. Tests use fakes only; importing a module must not cross a real model/runtime boundary.
- Do not guess or temporarily emulate the evaluator API. Before production edits, verify the upstream owner has released its claim, identify the committed release hash, and inspect the committed `load_policy` signature and return contract.
- The required upstream contract is the wiring-map §2.10 contract: `load_policy(spec: ModelSpec, adapter: Path | None, *, lazy: bool = False) -> tuple[Any, Any, ArchitectureView, ResolvedSpec]`. If the released contract differs, stop production work and report the mismatch because every implementation path would otherwise be a guess.
- No compatibility fallback is permitted in the migrated live path: `load_policy` receives a `ModelSpec`, and every `run_task` receives explicit `spec`, `view`, and `resolved` objects. Do not add a string/tuple-length fallback, default legacy declaration, or fake-only branch.
- Preserve rollout note-integrity filtering, de-duplication, task accounting, JSONL writes, cleanup, and the external `run_rollout(model_name=..., adapter=...)` call shape unless the committed upstream contract proves that impossible.
- Preserve every foreign dirty/staged path and concurrent commit. Stage and commit only explicit claimed paths; never use broad Git staging, reset, restore, stash, clean, merge, rebase, pull, or push.
- Follow strict TDD: add mutation-sensitive tests, confirm their expected RED against the old caller, make the smallest production change only after the upstream release, then confirm GREEN.
- R10: all new rollout tests live in `tests/test_rollout.py`. Do not copy the existing integrity-filter assertion from `tests/test_integrity.py`; if its old direct call becomes incompatible, report that exact subtractive move as a dependency.
- R13: before review or claim release, the exact `uv run pytest -q` command must exit 0 on a named committed HEAD. The implementation report must contain `## R13 full fake-only suite` with the exact commit, command, exit code, pass/fail/skip/xfail counts, and failing node IDs (`none` when green).

---

### Task 1: Prove and migrate the rollout model context

**Files:**

- Modify: `src/local_llm_lab/pipeline/rollout.py`
- Create: `tests/test_rollout.py`
- Create: `design_specifications/under_review/SPEC-001-ROLLOUT-MODEL-AWARE-CALLER-IMPLEMENTATION-REPORT.md`
- Include: `docs/superpowers/plans/2026-09-04-rollout-model-aware-load-policy.md`
- Update, do not stage: `.superpowers/sdd/2026-09-04-rollout-model-aware-load-policy/progress.md`

**Interfaces:**

- Consumes after dependency release: `load_model_spec(model_name: str) -> ModelSpec`; `load_policy(spec: ModelSpec, adapter: Path | None, *, lazy: bool = False) -> tuple[Any, Any, ArchitectureView, ResolvedSpec]`; `ResolvedSpec.as_dict() -> dict[str, Any]`; `run_task(..., *, spec: ModelSpec, view: ArchitectureView | None, resolved: ResolvedSpec | None, ...) -> Trajectory`.
- Produces: `collect_rollouts(model, tokenizer, tasks, *, spec: ModelSpec, view: ArchitectureView, resolved: ResolvedSpec, label, samples, temperature, keep_per_task=2, max_steps=24, max_tokens=200, keep_last=DEFAULT_KEEP_LAST, transcript_dir=None, quiet=False, seed=20260902) -> tuple[list[Trajectory], list[dict[str, Any]], dict[str, Any]]`.
- Preserves: `run_rollout` keeps required keyword-only `model_name: str` and `adapter: Path | None`; it resolves one declaration, calls the released loader once, forwards the exact four returned objects, writes `resolved.as_dict()` under summary key `model`, releases the model reference, and clears the cache once.

- [ ] **Step 1: Establish the fake-only baseline**

Run without model/runtime execution:

```bash
.venv/bin/pytest -q tests/test_integrity.py -k 'rollout'
.venv/bin/python -m py_compile src/local_llm_lab/pipeline/rollout.py
git status --short
git rev-parse HEAD
```

Expected: the existing rollout integrity-filter assertion passes, rollout compiles, and the status contains only already-known foreign paths.

- [ ] **Step 2: Write the mutation-sensitive fake harness**

Create `tests/test_rollout.py`. Use `load_model_spec("qwen35-4b")` as the non-legacy sentinel declaration. Install only a process-local fake `mlx.core` with `random.seed` and `clear_cache`, replace `make_sampler`, and make every fake reject missing or substituted identity objects.

The first test directly exercises `collect_rollouts` with one generated task and a fake trajectory. It must fail if production omits any of `spec`, `view`, or `resolved` when calling `run_task`:

```python
def test_collect_rollouts_forwards_active_model_context_to_every_task(monkeypatch) -> None:
    spec = load_model_spec("qwen35-4b")
    model, tokenizer, view, resolved = object(), object(), object(), object()
    task = make_tasks("rollout-model-context", 1, seed=17)[0]
    trajectory = Trajectory(task.task_id, task.family, task.variant, "fake", task.prompt)
    trajectory.verdict = {"success": False, "clean": False}
    seen = []

    def fake_run(actual_model, actual_tokenizer, actual_task, **kwargs):
        seen.append(
            (
                actual_model,
                actual_tokenizer,
                actual_task,
                kwargs["spec"],
                kwargs["view"],
                kwargs["resolved"],
            )
        )
        return trajectory

    monkeypatch.setattr(rollout, "run_task", fake_run)
    # Install the process-local mlx fake and deterministic sampler/integrity boundaries.
    rollout.collect_rollouts(
        model,
        tokenizer,
        [task],
        spec=spec,
        view=view,
        resolved=resolved,
        label="fake",
        samples=1,
        temperature=0.0,
        quiet=True,
        seed=17,
    )
    assert seen == [(model, tokenizer, task, spec, view, resolved)]
```

The second test exercises `run_rollout` with a temporary output directory. Its fake loader accepts only the exact registry `ModelSpec`, returns four identity sentinels, and its fake collector requires those identities. It asserts that the loader sees `spec`, never `model_name` or `spec.hf_id`, and that the returned summary records `resolved.as_dict()`:

```python
def test_run_rollout_loads_and_forwards_one_registry_spec(monkeypatch, tmp_path: Path) -> None:
    spec = load_model_spec("qwen35-4b")
    model, tokenizer, view = object(), object(), object()
    resolved = SimpleNamespace(as_dict=lambda: {"spec": {"name": spec.name}})
    seen = []

    monkeypatch.setattr(
        rollout,
        "load_model_spec",
        lambda name: seen.append(("registry", name)) or spec,
    )
    monkeypatch.setattr(
        rollout,
        "load_policy",
        lambda actual, adapter: seen.append(("loader", actual, adapter))
        or (model, tokenizer, view, resolved),
    )

    def fake_collect(actual_model, actual_tokenizer, tasks, **kwargs):
        seen.append(
            (
                "collect",
                actual_model,
                actual_tokenizer,
                kwargs["spec"],
                kwargs["view"],
                kwargs["resolved"],
            )
        )
        return [], [], {"pass_at_k": 0.0}

    monkeypatch.setattr(rollout, "collect_rollouts", fake_collect)
    monkeypatch.setattr(rollout, "make_tasks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(rollout, "write_jsonl", lambda *_args, **_kwargs: "digest")
    summary = rollout.run_rollout(
        model_name="qwen35-4b",
        adapter=None,
        label="fake",
        split="iter-model-context",
        limit=1,
        samples=1,
        temperature=0.0,
        output=tmp_path,
        transcript_dir=None,
        quiet=True,
    )
    assert seen[:2] == [("registry", "qwen35-4b"), ("loader", spec, None)]
    assert seen[2] == ("collect", model, tokenizer, spec, view, resolved)
    assert summary["model"] == resolved.as_dict()
```

Complete the fixture details without weakening these identity assertions. Do not assert only mock call counts; the test contract is the real rollout summary and exact context that the real `run_task` consumes.

- [ ] **Step 3: Run the harness and verify RED for the old caller**

Run:

```bash
.venv/bin/pytest -q tests/test_rollout.py
```

Expected RED: `collect_rollouts` rejects the required `spec`/`view`/`resolved` keywords and `run_rollout` passes a string to the old two-value loader. Record the exact failing nodes and messages. If a test errors for a malformed fake rather than the old production boundary, repair the fake and rerun until the expected production gap is demonstrated.

- [ ] **Step 4: Stop at the upstream dependency gate**

After RED, do not edit `rollout.py`. Update the Coordinator claim to `blocked`, retain the same exact paths, and set `blockedBy` to native task `01a06844-b9b9-7bc2-bbf4-847fde9e8732`. Send at most one `DEPENDENCY` notice naming `src/local_llm_lab/pipeline/evaluate.py::load_policy`. The implementer report must return `BLOCKED` with the RED evidence and leave the harness uncommitted.

- [ ] **Step 5: Verify and consume the committed upstream contract**

Resume only after a `RELEASED` notice or a fresh canonical board listing proves the upstream claim no longer owns `evaluate.py`. Record the release commit, then inspect the committed function with:

```bash
upstream_commit=$(git log -n 1 --format=%H -- src/local_llm_lab/pipeline/evaluate.py)
git show "${upstream_commit}:src/local_llm_lab/pipeline/evaluate.py"
```

Confirm it accepts a `ModelSpec`, optional adapter and keyword-only `lazy`, returns `(model, tokenizer, view, resolved)`, loads `spec.hf_id`, and resolves the same `spec`. If any part differs from the normative contract, remain blocked and report the exact mismatch.

- [ ] **Step 6: Implement the minimal rollout migration**

In `rollout.py`, add type-only imports for `ArchitectureView`, `ModelSpec`, and `ResolvedSpec` where runtime imports are unnecessary, plus the runtime `load_model_spec` import used by `run_rollout`.

Make the `collect_rollouts` context required and pass it through unchanged:

```python
def collect_rollouts(
    model: Any,
    tokenizer: Any,
    tasks: list[Task],
    *,
    spec: ModelSpec,
    view: ArchitectureView,
    resolved: ResolvedSpec,
    label: str,
    # existing options unchanged
) -> tuple[list[Trajectory], list[dict[str, Any]], dict[str, Any]]:
    ...
    trajectory = run_task(
        model,
        tokenizer,
        task,
        sampler=sampler,
        spec=spec,
        view=view,
        resolved=resolved,
        # existing arguments unchanged
    )
```

At the `run_rollout` boundary, resolve exactly once and use the released four-value contract:

```python
spec = load_model_spec(model_name)
model, tokenizer, view, resolved = load_policy(spec, adapter)
trajectories, rows, summary = collect_rollouts(
    model,
    tokenizer,
    tasks,
    spec=spec,
    view=view,
    resolved=resolved,
    # existing arguments unchanged
)
summary.update({"model": resolved.as_dict(), ...})
```

Do not add fallback unpacking, `hasattr`-based compatibility, a default spec, or a string loader path. Preserve model cleanup and cache clearing exactly once after output writes.

- [ ] **Step 7: Verify GREEN, existing rollout behavior, and caller boundaries**

Run:

```bash
.venv/bin/pytest -q tests/test_rollout.py
.venv/bin/pytest -q tests/test_integrity.py -k 'rollout'
.venv/bin/ruff check src/local_llm_lab/pipeline/rollout.py tests/test_rollout.py
.venv/bin/ruff check --select C901 --config 'lint.mccabe.max-complexity=9' src/local_llm_lab/pipeline/rollout.py
.venv/bin/python -m py_compile src/local_llm_lab/pipeline/rollout.py tests/test_rollout.py
git diff --check -- src/local_llm_lab/pipeline/rollout.py tests/test_rollout.py
```

Expected: both new mutation-sensitive tests pass, the existing integrity behavior remains covered, and static checks exit 0. If `tests/test_integrity.py::test_rollout_retains_only_integrity_clean_successes` fails only because its direct call lacks the now-required model context, do not duplicate or edit it; send one dependency notice requesting the exact R10 subtractive move and remain blocked until the Coordinator assigns that path.

- [ ] **Step 8: Scan production callers and write the implementation report**

Search all production calls to `collect_rollouts`, `run_rollout`, and `load_policy`. The report must list every caller whose committed arguments no longer match, including exact file and line. Do not fix callers outside `rollout.py`.

Create `design_specifications/under_review/SPEC-001-ROLLOUT-MODEL-AWARE-CALLER-IMPLEMENTATION-REPORT.md` with: exact scope and line counts; upstream release commit and inspected signature; identity-flow diagram in prose; RED/GREEN node IDs and outputs; mutation rationale; preserved integrity/de-duplication/output/cleanup behavior; resolved summary schema; caller scan; R10 disposition; no-model evidence; changed-path evidence; review status; and the exact R13 section required below.

- [ ] **Step 9: Commit the bounded source/test/report/plan paths**

Inspect the index before staging. Stage only the four tracked deliverables; the ignored SDD ledger and task report stay unstaged:

```bash
git add -- src/local_llm_lab/pipeline/rollout.py tests/test_rollout.py docs/superpowers/plans/2026-09-04-rollout-model-aware-load-policy.md design_specifications/under_review/SPEC-001-ROLLOUT-MODEL-AWARE-CALLER-IMPLEMENTATION-REPORT.md
git diff --cached --name-status
git diff --cached --check
git commit --only -m "refactor: migrate rollout policy loading" -- src/local_llm_lab/pipeline/rollout.py tests/test_rollout.py docs/superpowers/plans/2026-09-04-rollout-model-aware-load-policy.md design_specifications/under_review/SPEC-001-ROLLOUT-MODEL-AWARE-CALLER-IMPLEMENTATION-REPORT.md
```

Do not request review yet.

- [ ] **Step 10: Run and record the exact R13 full fake-only gate**

At the committed integrated HEAD, run:

```bash
uv run pytest -q
uv run pytest -q -o addopts='' --tb=no
```

The first command is the binding R13 gate; the second makes the pass/fail/skip/xfail count explicit without changing the tested set. If either is red, record every failing node and retain the active claim. Do not request review, report completion, add a compatibility fallback, or release until the exact full suite is green.

Update the report under the exact heading `## R13 full fake-only suite` and make one report-only commit if the pre-gate report did not already contain the final committed-HEAD evidence. Copy the literal output of `git rev-parse HEAD`, the literal command `uv run pytest -q`, exit code `0`, the complete pass/fail/skip/xfail counts, and `Failing node IDs: none`; do not use symbolic or placeholder values.

```markdown
## R13 full fake-only suite

- Tested commit: the literal 40-character commit printed by `git rev-parse HEAD`
- Command: `uv run pytest -q`
- Exit code: `0`
- Counts: the literal pytest pass/fail/skip/xfail totals
- Failing node IDs: `none`
```

- [ ] **Step 11: Package and independently review the complete change**

Generate one immutable base-to-head review package. The single independent reviewer must inspect the plan, task brief, implementer report, implementation report, exact diff, upstream contract consumption, mutation sensitivity, R10/R13 evidence, out-of-scope caller list, and no-model constraint. Critical/Important findings return verbatim to the same implementer for up to five bounded fix/re-review rounds; Minor findings are recorded in the ledger.

After a clean review, rerun the focused tests, scoped Ruff/C901, compile/diff checks, and exact R13 suite against the reviewed commit. Append the SDD completion line with the literal seven-character base and head SHAs plus `review clean`, report the final range/verdict/evidence to the Coordinator, and release this claim as completed.
