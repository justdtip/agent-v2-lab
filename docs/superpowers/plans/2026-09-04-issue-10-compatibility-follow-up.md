# Issue #10 Compatibility Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the two remaining Issue #10 compatibility-mode paths by threading the selected `ModelSpec` through branch prompting and migrating evaluation policy loading to the registry-backed, resolved-spec contract.

**Architecture:** Keep prompt rendering and model loading as separate seams. Branch mining resolves one active model declaration and passes that same object through its seed trajectory, branch prompts, and continuations. Evaluation owns the model-loading boundary: it accepts a `ModelSpec`, loads `spec.hf_id`, returns the model, tokenizer, architecture view, and resolved declaration, then passes those objects into every in-scope task run and records resolved metadata.

**Tech Stack:** Python 3.13, pytest, Ruff, existing `ModelSpec`/`ResolvedSpec`, `ArchitectureView`, `mlx-lm` behind fakes, and the registry-backed probe policy resolver.

**Spec:** `design_specifications/pending/SPEC-001-model-agnostic-backbone.md`; exact signatures come from `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` §2.3 and §2.10. The Coordinator's reopened GitHub Issue #10 assignment narrows this plan's file ownership.

## Global Constraints

- Work in the existing primary checkout on branch `codex/agent-v2-specs`; coordinated tasks must not create or switch branches or worktrees.
- Modify only `src/local_llm_lab/pipeline/branch.py`, `tests/test_branch.py`, `src/local_llm_lab/pipeline/evaluate.py`, `tests/test_evaluate.py`, this plan, `design_specifications/under_review/GITHUB-ISSUE-10-COMPATIBILITY-FOLLOWUP-REPORT.md`, and this plan's `.superpowers/sdd/2026-09-04-issue-10-compatibility-follow-up` workspace.
- Do not modify `src/local_llm_lab/probes/policies.py`; its active owner must release before Task 2 consumes `resolve_policy(name: str, spec: ModelSpec) -> Path | None`.
- Never load or execute a real model, checkpoint, tokenizer, adapter, evaluation, branch-mining CLI, probe, preflight, cache-equivalence, training, or J-space workload. All model-boundary tests use fakes.
- Preserve foreign dirty/staged files and commits. Stage and commit only explicit task-owned paths; never use broad Git staging, reset, restore, stash, clean, merge, rebase, pull, or push.
- Follow strict TDD: add a mutation-sensitive failing test, confirm the expected RED, write the smallest production change, then confirm GREEN.
- R10: run the changed module's per-module tests before each task review. R13: `uv run pytest -q` must exit 0 on the integrated committed state before handoff or release.
- The implementation report must contain `## R13 full fake-only suite` with the exact tested commit, exact command, exit code, pass/fail/xfail counts, and failing node IDs.
- Out-of-scope `evaluate.load_policy` consumers are coordination dependencies, not permission to broaden this plan. Record every exact path in the report.

---

### Task 1: Thread the active model declaration through branch prompts

**Files:**

- Modify: `src/local_llm_lab/pipeline/branch.py`
- Modify: `tests/test_branch.py`
- Create: `design_specifications/under_review/GITHUB-ISSUE-10-COMPATIBILITY-FOLLOWUP-REPORT.md`

**Interfaces:**

- Consumes: `ModelSpec`, `load_model_spec(model_name)`, the keyword-only `spec` argument of `build_prompt`, and the keyword-only `spec` argument of `run_task`.
- Produces: a required keyword-only `spec: ModelSpec` parameter on `_continue` and `mine_pairs`, with the selected declaration forwarded to every prompt-rendering path.
- Preserves: the current keyword-only `run_branch_mining` parameters, including `model_name: str` and `adapter: Path | None`, remain source-compatible for the out-of-scope `pipeline/cli.py` caller during this task.

- [ ] **Step 1: Write failing branch forwarding tests**

Add mutation-sensitive tests that use `load_model_spec("qwen35-4b")` as a non-legacy sentinel.

```python
def test_continue_forwards_the_active_spec_to_prompt_rendering(monkeypatch) -> None:
    from types import SimpleNamespace

    from local_llm_lab.agent_protocol import Action
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import branch
    from local_llm_lab.pipeline.tasks import Task

    spec = load_model_spec("qwen35-4b")
    seen = []
    monkeypatch.setattr(branch, "build_prompt", lambda *_a, **kw: seen.append(kw["spec"]) or "p")
    monkeypatch.setattr(branch, "generate_turn", lambda *_a: "raw")
    monkeypatch.setattr(
        branch,
        "parse_turn",
        lambda _raw: SimpleNamespace(thought="done", action=Action("finish", {"answer": "yes"})),
    )

    class FakeSimulator:
        def execute(self, _action):
            return ""

        def verdict(self):
            return SimpleNamespace(success=True)

    task = Task("valid-read-0000-clean", "read", "clean", "p", {}, (), "yes", frozenset())
    assert branch._continue(
        object(), object(), task, [], FakeSimulator(), spec=spec, sampler=object(),
        remaining_steps=1, max_tokens=8, keep_last=2,
    ) is True
    assert seen == [spec]


def test_mine_pairs_forwards_the_active_spec_to_seed_and_branch_paths(monkeypatch) -> None:
    from types import SimpleNamespace

    from local_llm_lab.agent_protocol import Action
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import branch
    from local_llm_lab.pipeline.tasks import Task

    spec = load_model_spec("qwen35-4b")
    seen = {"run_task": [], "build_prompt": [], "continue": []}
    task = Task(
        "valid-read-0000-clean", "read", "clean", "p", {"x": "yes"}, (), "yes", frozenset()
    )
    trajectory = SimpleNamespace(
        success=True,
        steps=[{
            "thought": "done", "raw": "chosen",
            "action": {"name": "finish", "arguments": {"answer": "yes"}},
        }],
    )
    monkeypatch.setattr(
        branch, "run_task",
        lambda *_a, **kw: seen["run_task"].append(kw["spec"]) or trajectory,
    )
    monkeypatch.setattr(branch, "make_sampler", lambda _temperature: object())
    monkeypatch.setattr(
        branch, "build_prompt",
        lambda *_a, **kw: seen["build_prompt"].append(kw["spec"]) or "prompt",
    )
    monkeypatch.setattr(branch, "generate_turn", lambda *_a: "rejected")
    monkeypatch.setattr(
        branch, "parse_turn",
        lambda _raw: SimpleNamespace(thought="read", action=Action("read_file", {"path": "x"})),
    )
    monkeypatch.setattr(
        branch, "_continue",
        lambda *_a, **kw: seen["continue"].append(kw["spec"]) or False,
    )

    branch.mine_pairs(
        object(), object(), task, spec=spec, branches=1, temperature=1.0,
        max_steps=2, max_tokens=8, keep_last=2,
    )
    assert seen["run_task"] == [spec]
    assert seen["build_prompt"] and set(seen["build_prompt"]) == {spec}
    assert seen["continue"] == [spec]
```

The fixtures must force a non-`finish` alternative through `_continue`; deleting `spec=` from the seed `run_task`, either direct `build_prompt`, or `_continue` must fail at least one test.

- [ ] **Step 2: Run the tests and confirm the compatibility path is detected**

Run:

```bash
uv run pytest -q tests/test_branch.py -k 'active_spec or active_model'
```

Expected RED: `_continue` and `mine_pairs` reject the new `spec` keyword, or the recording fake observes a missing/`None` spec.

- [ ] **Step 3: Implement the smallest complete forwarding change**

In `branch.py`, add `spec: ModelSpec` to the keyword-only section of both existing function
signatures, then make these exact call-site changes:

```python
from local_llm_lab.models import ModelSpec, load_model_spec

prompt = build_prompt(tokenizer, messages, spec=spec, keep_last=keep_last)
seed_trajectory = run_task(
    model, tokenizer, task, sampler=greedy, spec=spec, label="seed",
    max_steps=max_steps, max_tokens=max_tokens, keep_last=keep_last, transcript=transcript,
)
success = _continue(
    model, tokenizer, task, branch_messages, branch_sim, spec=spec, sampler=greedy,
    remaining_steps=max_steps - point - 1, max_tokens=max_tokens, keep_last=keep_last,
)
```

Inside `run_branch_mining`, resolve `spec = load_model_spec(model_name)` once, continue calling the pre-migration loader with `spec.hf_id`, and pass `spec=spec` to `mine_pairs`. Do not change the public `run_branch_mining` signature in Task 1.

- [ ] **Step 4: Verify Task 1**

Run:

```bash
uv run pytest -q tests/test_branch.py
uv run ruff check src/local_llm_lab/pipeline/branch.py tests/test_branch.py
uv run ruff check --select C901 src/local_llm_lab/pipeline/branch.py
python3 -m py_compile src/local_llm_lab/pipeline/branch.py tests/test_branch.py
git diff --check -- src/local_llm_lab/pipeline/branch.py tests/test_branch.py
```

- [ ] **Step 5: Commit Task 1 and record evidence**

Stage only `branch.py`, `test_branch.py`, and the report. The report records the Task 1 RED/GREEN commands, exact output, changed behavior, and that no real workload ran.

```bash
git add -- src/local_llm_lab/pipeline/branch.py tests/test_branch.py design_specifications/under_review/GITHUB-ISSUE-10-COMPATIBILITY-FOLLOWUP-REPORT.md
git commit -m "fix: forward model spec through branch mining"
```

---

### Task 2: Migrate evaluation to resolved model and registry policy plumbing

**Dependency gate:** Before editing, re-list the canonical board and verify the explicit `RELEASED` contract notice from the owner of `src/local_llm_lab/probes/policies.py`. The owner released commit `e53bd810d5204541efb56ebd5b9c1682120c9cbf` with `resolve_policy(name: str, spec: ModelSpec) -> Path | None` and model-scoped `policy_names(spec)` while retaining its broader caller-migration claim; re-read that committed file and report before implementation.

**Files:**

- Modify: `src/local_llm_lab/pipeline/evaluate.py`
- Modify: `tests/test_evaluate.py`
- Modify: `src/local_llm_lab/pipeline/branch.py`
- Modify: `tests/test_branch.py`
- Modify: `design_specifications/under_review/GITHUB-ISSUE-10-COMPATIBILITY-FOLLOWUP-REPORT.md`

**Interfaces:**

- Consumes: released `resolve_policy(name, spec)`, `ArchitectureView.from_model(model)`, and `ModelSpec.resolve(model, tokenizer)`.
- Produces: wiring-map §2.10 `load_policy(spec: ModelSpec, adapter: Path | None, *, lazy: bool = False) -> tuple[Any, Any, ArchitectureView, ResolvedSpec]`; `run_evaluation` replaces `model_name: str` with required keyword-only `spec: ModelSpec` while preserving its other current keyword-only parameters.
- Preserves: direct adapter paths remain supported through the released resolver; no string-to-default-spec compatibility branch remains inside `load_policy`, `evaluate_tasks`, or `run_evaluation`.

- [ ] **Step 1: Write failing loader, runner, metadata, and registry-policy tests**

Use fake objects and a fake `mlx_lm.load`; never import or load a checkpoint.

```python
def test_load_policy_uses_spec_and_returns_view_and_resolved(monkeypatch, tmp_path) -> None:
    import sys
    from types import SimpleNamespace

    from local_llm_lab import arch
    from local_llm_lab.pipeline import evaluate

    model, tokenizer, view = object(), object(), object()
    resolved = SimpleNamespace(as_dict=lambda: {"spec": {"name": "fake"}})
    seen = []

    class FakeSpec:
        hf_id = "fake/hf"

        def resolve(self, actual_model, actual_tokenizer):
            assert (actual_model, actual_tokenizer) == (model, tokenizer)
            return resolved

    def fake_load(model_id, *, adapter_path, lazy):
        seen.append((model_id, adapter_path, lazy))
        return model, tokenizer

    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=fake_load))
    monkeypatch.setattr(
        arch.ArchitectureView, "from_model", classmethod(lambda _cls, actual: view)
    )
    spec = FakeSpec()
    model, tokenizer, actual_view, actual_resolved = evaluate.load_policy(
        spec, tmp_path, lazy=True
    )
    assert seen == [("fake/hf", str(tmp_path.resolve()), True)]
    assert (actual_view, actual_resolved) == (view, resolved)


def test_evaluate_tasks_forwards_required_model_metadata_to_runner(monkeypatch) -> None:
    from types import SimpleNamespace

    model, tokenizer, spec, view, resolved = object(), object(), object(), object(), object()
    task = Task(
        "valid-read-0000-clean", "read", "clean", "p", {}, (), "yes", frozenset()
    )
    trajectory = _trajectory(task.task_id, success=True, clean=True, difficulty=0)
    seen = []
    monkeypatch.setattr(evaluate, "make_sampler", lambda _temperature: object())
    monkeypatch.setattr(
        evaluate, "run_task",
        lambda *_a, **kw: seen.append((kw["spec"], kw["view"], kw["resolved"])) or trajectory,
    )
    monkeypatch.setattr(
        evaluate,
        "check_trajectory",
        lambda *_a, **_kw: SimpleNamespace(as_dict=lambda: {"clean": True, "counts": {}}),
    )
    evaluate.evaluate_tasks(
        model, tokenizer, [task], label="fake", spec=spec, view=view, resolved=resolved, quiet=True,
    )
    assert seen == [(spec, view, resolved)]


def test_run_evaluation_records_resolved_model_metadata(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    model, tokenizer, view = object(), object(), object()
    spec = SimpleNamespace(hf_id="fake/hf")
    resolved = SimpleNamespace(as_dict=lambda: {"spec": {"name": "fake"}})
    monkeypatch.setattr(evaluate, "load_policy", lambda *_a, **_kw: (model, tokenizer, view, resolved))
    monkeypatch.setattr(evaluate, "make_tasks", lambda *_a, **_kw: [])
    monkeypatch.setattr(evaluate, "evaluate_tasks", lambda *_a, **_kw: [])
    monkeypatch.setattr(evaluate, "_seed_model_rng", lambda _seed: None)
    monkeypatch.setattr(evaluate, "_clear_model_cache", lambda: None)
    summary = evaluate.run_evaluation(
        spec=spec, adapter=None, label="fake", split="valid", limit=1,
        output=tmp_path / "eval.json", transcript_dir=None, quiet=True,
    )
    assert summary["model"] == resolved.as_dict()
```

Add a CLI test that passes `--model qwen35-4b --policy base`, asserts `resolve_policy("base", spec)` runs before `run_evaluation`, and asserts the exact same `spec` object reaches `run_evaluation`. Unknown registry policy errors must occur before the model loader seam.

```python
def test_main_resolves_registry_policy_before_evaluation(monkeypatch, tmp_path) -> None:
    import sys

    spec, adapter, seen = object(), tmp_path / "adapter", []
    monkeypatch.setattr(evaluate, "load_model_spec", lambda name: seen.append(("spec", name)) or spec)
    monkeypatch.setattr(
        evaluate, "resolve_policy",
        lambda name, actual: seen.append(("policy", name, actual)) or adapter,
    )
    monkeypatch.setattr(
        evaluate, "run_evaluation",
        lambda **kwargs: seen.append(("run", kwargs["spec"], kwargs["adapter"])),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["agent-v2-eval", "--model", "qwen35-4b", "--policy", "base", "--output", str(tmp_path / "eval.json")],
    )
    evaluate.main()
    assert seen == [
        ("spec", "qwen35-4b"),
        ("policy", "base", spec),
        ("run", spec, adapter),
    ]
```

Update branch fakes to require `load_policy(spec, adapter)` and four-value unpacking; assert the same `spec` object reaches `mine_pairs`.

- [ ] **Step 2: Run focused tests and confirm the old loader contract fails**

Run:

```bash
uv run pytest -q tests/test_evaluate.py tests/test_branch.py -k 'load_policy or model_metadata or registry_policy or active_spec'
```

Expected RED: the old two-argument string loader returns only two values, evaluation omits model metadata when calling `run_task`, and the evaluation CLI has no registry policy resolver path.

- [ ] **Step 3: Implement the exact four-value loader**

In `evaluate.py`:

```python
def load_policy(
    spec: ModelSpec,
    adapter: Path | None,
    *,
    lazy: bool = False,
) -> tuple[Any, Any, ArchitectureView, ResolvedSpec]:
    configure_local_cache()
    from mlx_lm import load

    model, tokenizer = load(
        spec.hf_id,
        adapter_path=None if adapter is None else str(adapter.resolve()),
        lazy=lazy,
    )
    view = ArchitectureView.from_model(model)
    resolved = spec.resolve(model, tokenizer)
    return model, tokenizer, view, resolved
```

Make `evaluate_tasks` require keyword-only `spec`, `view`, and `resolved` and pass all three to `run_task`. Make `run_evaluation` require `spec`, call and unpack the four-value loader, pass metadata to `evaluate_tasks`, and write `resolved.as_dict()` as the evaluation summary's `model` value.

- [ ] **Step 4: Resolve the evaluator CLI policy through the released registry API**

Replace the evaluator CLI's `--adapter` input with `--policy`, defaulting to `base`. Before any loader call:

```python
spec = load_model_spec(args.model)
try:
    adapter = resolve_policy(args.policy, spec)
except ValueError as error:
    parser.error(str(error))
```

Then call `run_evaluation` with `spec=spec`, `adapter=adapter`, and every existing evaluation argument. The label defaults to `base` for the base policy and otherwise to the policy name/path basename. No omitted-spec compatibility mode is permitted.

In `branch.py`, retain the existing public `run_branch_mining` parameters `model_name: str` and `adapter: Path | None` for the out-of-scope pipeline caller, but replace its loader call with four-value unpacking from `load_policy(spec, adapter)`. Pass the same `spec` object onward.

- [ ] **Step 5: Record out-of-scope consumers without editing them**

The implementation report must list these production consumers as exact coordination follow-ups:

```text
research/cache_equivalence.py
src/local_llm_lab/pipeline/rollout.py
src/local_llm_lab/pipeline/jlens.py
src/local_llm_lab/probes/adapter_delta.py
src/local_llm_lab/probes/assistant_axis.py
src/local_llm_lab/probes/patch.py
src/local_llm_lab/probes/state_probe.py
```

It must also note that `src/local_llm_lab/pipeline/cli.py` and `src/local_llm_lab/probes/adapter_delta.py` call `run_evaluation` and require coordinated migration to the exact §2.10 signature. Do not edit their source or tests in this plan.

- [ ] **Step 6: Verify Task 2 and the integrated fake-only suite**

Run:

```bash
uv run pytest -q tests/test_evaluate.py
uv run pytest -q tests/test_branch.py
uv run ruff check src/local_llm_lab/pipeline/evaluate.py tests/test_evaluate.py src/local_llm_lab/pipeline/branch.py tests/test_branch.py
uv run ruff check --select C901 src/local_llm_lab/pipeline/evaluate.py src/local_llm_lab/pipeline/branch.py
python3 -m py_compile src/local_llm_lab/pipeline/evaluate.py tests/test_evaluate.py src/local_llm_lab/pipeline/branch.py tests/test_branch.py
git diff --check -- src/local_llm_lab/pipeline/evaluate.py tests/test_evaluate.py src/local_llm_lab/pipeline/branch.py tests/test_branch.py
uv run pytest -q
uv run pytest -q -o addopts='' --tb=no
```

The exact R13 command must exit 0. If out-of-scope consumers fail from the signature migration, do not add a compatibility fallback and do not edit them; retain the claim, record the exact failing nodes, and wait for coordinated consumer migration.

- [ ] **Step 7: Commit source/tests, then commit final evidence separately**

Commit only the four owned source/test files after the focused and R13 gates are green:

```bash
git add -- src/local_llm_lab/pipeline/evaluate.py tests/test_evaluate.py src/local_llm_lab/pipeline/branch.py tests/test_branch.py
git commit -m "refactor: migrate evaluation policy loading"
```

Update the report's exact `## R13 full fake-only suite` evidence against that committed integrated code state, then commit only the report:

```bash
git add -- design_specifications/under_review/GITHUB-ISSUE-10-COMPATIBILITY-FOLLOWUP-REPORT.md
git commit -m "docs: record issue 10 compatibility verification"
```

- [ ] **Step 8: Publish the reviewed contracts to routed caller owners**

After the independent task review and controller verification are clean, the controller sends one
bounded `RELEASED` notice per routed owner, naming the committed signatures and no transcript or
test output:

```text
01a066e6-99ae-7ef3-8955-37419c5b6689: pipeline/jlens.py, probes/assistant_axis.py, probes/patch.py, probes/state_probe.py, probes/adapter_delta.py
01a06748-e8c7-7232-9062-e5c9c92e8f56: research/cache_equivalence.py
01a06718-057a-7bb2-a62d-71f86e904d64: pipeline/rollout.py
01a06858-d818-70c1-bde8-9803b0a892bd: pipeline/cli.py
```

The released boundary is the exact committed `load_policy(spec, adapter, *, lazy=False)` four-value
return and `run_evaluation(spec=...)` keyword contract. These notices grant no authority and do not
change this plan's path claim.
