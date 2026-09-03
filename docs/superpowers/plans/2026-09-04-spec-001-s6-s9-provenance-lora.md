# SPEC-001 §6/§9 Provenance and LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete deterministic repository/run provenance, remove the CLI's hard-coded LoRA
targets in favor of architecture-resolved targets, and finish the 180-task `all` stage's
`--base`/`--stress` wiring without executing a model.

**Architecture:** `provenance.py` owns deterministic source/package/Git metadata and serializes
one stable JSON schema. `cli.py` supplies stage-specific payloads, resolves the effective
training `ModelSpec` before generating mlx-lm configuration, and delegates target discovery to
`ModelSpec.resolve`/`ArchitectureView.lora_targets`. The existing stage functions remain the
public orchestration seams; tests replace model/runtime boundaries with fakes.

**Tech Stack:** Python 3.11+, pytest, PyYAML, `importlib.metadata`, Git subprocesses, existing
`ModelSpec`/`ResolvedSpec`/`ArchitectureView`, and existing pipeline stage functions.

**Spec:** `design_specifications/pending/SPEC-001-model-agnostic-backbone.md` §§6 and 9;
`design_specifications/pending/SPEC-002-evaluation-selection-note-integrity.md` §4;
`design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` rulings R3/R10/R11/R13; and
`design_specifications/under_review/CODEX-COORDINATOR-NOTE-2026-09-04.md` Lane D.

## Global Constraints

- Shared primary checkout and current `codex/agent-v2-specs` branch only; never switch branches
  or worktrees, and stage/commit only explicit owned paths while preserving foreign changes.
- Exactly one principal implementer performs the extraction and behavior tasks, spawns no
  helpers/reviewer, and the controller dispatches exactly one independent principal reviewer.
- No model, checkpoint, tokenizer, training, evaluation, rollout, probe, preflight,
  cache-equivalence, or J-space workload may run. Tests use fakes and temporary directories.
- Do not write `outputs/`, `data/`, `reports/`, model registries/config YAML, or any document in
  `design_specifications/pending/`.
- R10: new tests live in `tests/test_cli.py` or `tests/test_provenance.py`. The two directly
  relevant legacy CLI tests move subtractively from `tests/test_pipeline.py` before behavior
  changes, under the temporary formal claim amendment.
- R11: `provenance.json` is metadata-sized and may use a direct deterministic write.
- R13: before requesting the final behavior review, record exact HEAD and run
  `uv run pytest -o addopts='' -q`; hand-off is allowed only with exit 0 and exact counts/nodes
  recorded in the implementation report and SDD ledger.
- Existing `stage_select` provenance at `c73bfed` is preserved. This lane adds the missing
  `stage_train`, `stage_eval`, and `stage_rollout` calls only.
- No compatibility fallback may retain the old hard-coded seven-key list. Missing
  `train.lora_keys` means policy `"auto"`; an explicit configured list remains exact and is
  validated by the architecture view.

---

### Task 0: R10 extraction boundary

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: existing `lora_config` and `stage_train` behavior before this plan changes either.
- Produces: per-module ownership for the exact legacy tests
  `test_lora_config_reflects_grad_checkpoint_and_resume` and
  `test_stage_train_clears_only_its_checkpoint_directory`.

- [ ] **Step 1: Move the two exact tests subtractively**

Copy the complete two function definitions from `tests/test_pipeline.py` into
`tests/test_cli.py`, removing only those same definitions from the monolith. Preserve bodies,
assertions, literals, and local imports; make only the import adjustment required for collection.

- [ ] **Step 2: Prove collection and behavior from the new module**

Run:

```bash
uv run pytest -o addopts='' -q \
  tests/test_cli.py::test_lora_config_reflects_grad_checkpoint_and_resume \
  tests/test_cli.py::test_stage_train_clears_only_its_checkpoint_directory
```

Expected: `2 passed`; the old node IDs no longer collect from `tests/test_pipeline.py`.

- [ ] **Step 3: Verify an AST-equivalent/subtractive move**

Compare the pre-move functions with the new definitions and confirm no assertion or setup
behavior changed. Run scoped Ruff and `git diff --check` on the two test files.

- [ ] **Step 4: Commit the extraction only**

```bash
git add tests/test_cli.py tests/test_pipeline.py
git commit -m "test: isolate training CLI coverage" -- tests/test_cli.py tests/test_pipeline.py
```

Stop here. The controller obtains independent review and removes `tests/test_pipeline.py` from
the active claim before any expectation or production edit.

---

### Task 1: Deterministic provenance core

**Files:**
- Modify: `src/local_llm_lab/provenance.py`
- Create: `tests/test_provenance.py`

**Interfaces:**
- Produces: `source_tree_hashes(root: Path) -> dict[str, str]`.
- Preserves: `write_provenance(run_dir: Path, *, resolved: ResolvedSpec | None,
  spec: ModelSpec, extra: dict[str, Any]) -> Path`.

- [ ] **Step 1: Write source-tree hash tests**

Create a temporary tree containing `src/pkg/a.py`, `src/pkg/ignored.txt`,
`configs/model.yaml`, `configs/nested/run.json`, `uv.lock`, and an unrelated file. Hand-compute
SHA-256 literals with `hashlib.sha256(bytes).hexdigest()` in the test fixture and assert the
result contains, in lexicographic repository-relative POSIX order, only both Python/config files
and `uv.lock`.

- [ ] **Step 2: Verify RED**

Run `uv run pytest -o addopts='' -q tests/test_provenance.py` and confirm import/attribute failure
for missing `source_tree_hashes`, not a fixture error.

- [ ] **Step 3: Implement `source_tree_hashes` minimally**

Hash regular files under `src/**/*.py`, every regular file under `configs/**`, and `uv.lock` when
present. Keys are relative POSIX paths; candidate paths and returned keys are deterministic.

- [ ] **Step 4: Write the deterministic provenance-schema test**

Monkeypatch the module's project root, package-version lookup, Git metadata helper, and
`sys.argv`. Call `write_provenance` twice without changing inputs and assert byte-identical JSON
with this exact top-level shape:

```python
{
    "command": ["agent-pipeline", "data"],
    "extra": {"stage": "data"},
    "generator_version": GENERATOR_VERSION,
    "git": {
        "branch": "codex/agent-v2-specs",
        "commit": "a" * 40,
        "dirty_patch_sha256": "b" * 64,
    },
    "model": expected_model_mapping,
    "packages": {
        "mlx": "1.0",
        "mlx-lm": "2.0",
        "numpy": "3.0",
        "transformers": "4.0",
    },
    "source_tree_hashes": expected_hashes,
}
```

Also cover `resolved.as_dict()` taking precedence over `asdict(spec)`.

- [ ] **Step 5: Verify RED, then implement the schema**

Use `importlib.metadata.version`, returning `None` only for `PackageNotFoundError`. Git metadata
records `git rev-parse HEAD`, `git branch --show-current`, and SHA-256 of
`git diff --binary HEAD --` bytes from `PROJECT_ROOT`. Serialize with sorted keys, indentation,
and one trailing newline. Re-run `tests/test_provenance.py` GREEN.

- [ ] **Step 6: Commit the provenance core**

```bash
git add src/local_llm_lab/provenance.py tests/test_provenance.py
git commit -m "feat: record deterministic run provenance" -- \
  src/local_llm_lab/provenance.py tests/test_provenance.py
```

---

### Task 2: Architecture-resolved LoRA configuration and training provenance

**Files:**
- Modify: `src/local_llm_lab/pipeline/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `_resolve_training_spec(config: dict[str, Any]) -> ResolvedSpec`.
- Changes: `lora_config(config, resolved, iters=None, resume_from=None) -> dict[str, Any]` now
  requires the resolved spec that supplies `spec.hf_id`, `num_layers`, and `lora_keys`.
- Preserves: `stage_train(config, iters, resume_from=None) -> None`.

- [ ] **Step 1: Write RED tests for automatic and explicit key policies**

Use a real lightweight `ModelSpec` value and monkeypatch the lazy base-model loader plus
`ModelSpec.resolve`. Assert absent `train.lora_keys` gives the effective spec policy `"auto"`;
an explicit list becomes an exact tuple in original order. Assert effective rank/scale/dropout
come from `config["train"]`, and resolution receives the registry `hf_id`, never its short name.

- [ ] **Step 2: Write RED tests for generated mlx-lm config**

Construct a literal fake `ResolvedSpec` with non-36 `num_layers` and hybrid target paths. Assert
`lora_config` writes `model=resolved.spec.hf_id`, the resolved layer count, a fresh list of exact
resolved keys, and unchanged training/resume settings. Mutations back to the seven static keys,
`config["model"]`, or `train["num_layers"]` must fail.

- [ ] **Step 3: Verify RED, then implement minimal resolution**

Delete `LORA_KEYS`. Normalize a configured list to a tuple, default the policy to `"auto"`, and
build an effective frozen `ModelSpec`/`LoraSpec` with `dataclasses.replace`. Lazily load the base
from the spec's `hf_id`, call the effective spec's existing `resolve(model, tokenizer)` (the
architecture-view-owned target resolver), and clear only the model cache after resolution.

- [ ] **Step 4: Add training provenance RED coverage and implementation**

Extend the moved fake `stage_train` test so it supplies a fake resolved spec, proves the written
YAML contains its keys/layer count, and captures exactly one post-success call:

```python
write_provenance(
    config["output"],
    resolved=resolved,
    spec=resolved.spec,
    extra={"stage": "train", "training_config": parsed_lora_yaml},
)
```

The provenance call must not occur when the subprocess exits nonzero. Implement only enough to
make these tests pass and print the resolved target count/keys without loading a model in tests.

- [ ] **Step 5: Run focused GREEN checks and commit**

Run the named training/LoRA nodes plus all of `tests/test_cli.py`, scoped Ruff, strict C901, and
`git diff --check`. Commit only `cli.py` and `test_cli.py`.

---

### Task 3: Evaluation/rollout provenance and 180-task `all` wiring

**Files:**
- Modify: `src/local_llm_lab/pipeline/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Preserves: existing `stage_eval` and `stage_rollout` signatures.
- Changes CLI only: `all` gains `--base`, `--stress`, and `--limit` default `180`.

- [ ] **Step 1: Write stage provenance RED tests**

For `stage_eval`, fake two policies (`base` and an explicit best adapter), return two literal
summaries, and assert one provenance call after both evaluations with
`extra={"stage": "eval", "evaluations": summaries}`. For `stage_rollout`, fake a literal summary
and assert one post-run call with `extra={"stage": "rollout", "summary": summary}`. Both calls
use `resolved=None` and the exact `load_model_spec(config["model"])` result, matching the already
accepted `stage_select` boundary without a second model load.

- [ ] **Step 2: Verify RED, then add the three-stage calls**

Collect evaluation summaries in policy order and emit provenance once after the loop. Capture
the rollout summary and emit once after `run_rollout`. Preserve all existing output paths,
transcript setup, and return behavior.

- [ ] **Step 3: Write `all` CLI wiring RED tests**

Fake every stage and invoke `main()` with a temporary config. Assert the default `all` call
selects the best adapter and evaluates exactly 180 tasks with `base=False`, `stress=False`.
Assert `all --base --stress` passes the explicit best adapter with `base=True`, `stress=True`,
so `stage_eval` evaluates base plus best in one ordered call. An explicit `--limit 17` overrides
180.

- [ ] **Step 4: Implement minimal parser/orchestration changes**

Add the two boolean flags and `default=180` limit. Replace the two hard-coded evaluation calls
with one call whose adapter is `config["output"] / "best-adapter"` and whose `base`, `stress`,
and `limit` values come from the parsed `all` arguments.

- [ ] **Step 5: Focused and R13 verification**

Run `tests/test_provenance.py tests/test_cli.py tests/test_selection.py`, scoped Ruff, strict
C901 max 9 with baseline comparison, and exact diff checks. Then record exact HEAD and run:

```bash
uv run pytest -o addopts='' -q
```

Do not request review if exit code is nonzero. Diagnose any failure and fix only Lane D failures;
for a foreign active-lane failure, retain ownership and coordinate until the integrated suite is
green. Record exit code, pass/fail/xfail counts, and failing node IDs in both report and ledger.

- [ ] **Step 6: Write implementation evidence and commit owned artifacts**

Create `design_specifications/under_review/SPEC-001-S6-S9-IMPLEMENTATION-REPORT.md` with the
requirement-to-evidence map, exact commits, R13 result, focused/static results, provenance schema,
LoRA compatibility decision, all-stage semantics, no-model evidence, and residual risks. Commit
the final `cli.py`/`test_cli.py` changes plus plan/report as explicit owned paths; keep the ignored
SDD ledger uncommitted.
