# SPEC-001 §10 Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the fake-tested SPEC-001 §10 preflight report and the pre-load train/eval gate, then leave the reviewed CLI ready for this task's separately authorised, serialized `qwen35-4b` execution.

**Architecture:** `pipeline/preflight.py` owns model inspection, deterministic report persistence, cached Hugging Face revision discovery, and central artifact validation. Its inspection and execution APIs expose narrow injection seams so unit tests use fake models, tokenizers, views, JVPs, clocks, and memory counters; the real loader is imported only inside the preflight execution path and is called with `lazy=True`. `pipeline/cli.py` remains a thin adapter: it adds `preflight --model`, calls the central guard before train/select/eval model loading, and propagates one explicit `--skip-preflight-check` override.

**Tech Stack:** Python 3.13, MLX/MLX-LM (lazy runtime imports), pytest, Ruff, deterministic JSON.

**Spec:** `design_specifications/pending/SPEC-001-model-agnostic-backbone.md` §10, interpreted with `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` §§7–8 and the exact Coordinator assignment for task `01a06728-508f-7980-b9f5-e3fb0da01dd0`.

## Global Constraints

- Keep `design_specifications/pending/` and `research/` read-only.
- All automated tests inject fakes and must never load or download a checkpoint.
- The report path is exactly `outputs/preflight/<registered-model-name>.json`; JSON uses stable keys, sorted mappings, indentation, and a trailing newline.
- The preflight is the only new model-loading stage and calls `mlx_lm.load(spec.hf_id, lazy=True)` through a late import.
- Inspect and report: architecture counts by layer kind, hidden size, vocabulary size, tied embeddings, 64-token residual equivalence, a middle-layer forward JVP with finite-difference fallback, a sample prompt and token count for each thinking mode, explicit LoRA keys and trainable parameter count, native cache kinds/strategy, and the specified memory estimate with its budget result.
- Memory estimate is the loaded parameter-tree byte size plus `batch_size * max_seq_length * hidden_size * num_layers * 4` float32 activation bytes; compare its GiB total with `ModelSpec.memory_budget_gib`.
- Pre-load validation accepts only a passed artifact whose registered name, `hf_id`, and cached `refs/main` snapshot revision equal the current model declaration/cache. Missing, malformed, failed, or stale evidence raises `SystemExit` before a loader is called. `--skip-preflight-check` is the only override.
- Guard train, behavioral selection, and eval call sites within this claim. Record probe guarding as a sequenced follow-up; do not edit any probe module.
- Preserve existing user and peer changes; touch only claimed files, stage explicit paths, and do not use broad Git operations.
- Apply R10: preflight module tests live in `tests/test_preflight.py`; CLI wiring tests live in `tests/test_cli.py`.
- Apply R11: this small JSON metadata artifact is written directly, not via array/NPZ atomic-write machinery.
- Apply R13: before review, run focused tests, Ruff on touched Python files, inspect the scoped diff, and run exact `uv run pytest -q`, recording HEAD, exit code, pass count, and collected node count.
- This implementation phase is fake-only. The real command is executed only by this native task after independent approval and after acquiring the exclusive `model-execution` board action.

---

### Task 1: Fake-testable preflight report, central guard, and CLI wiring

**Files:**

- Create: `src/local_llm_lab/pipeline/preflight.py`
- Create: `tests/test_preflight.py`
- Modify: `src/local_llm_lab/pipeline/cli.py`
- Modify: `tests/test_cli.py`
- Create: `design_specifications/under_review/SPEC-001-S10-PREFLIGHT-IMPLEMENTATION-REPORT.md`

**Interfaces:**

- Consumes: `ModelSpec`, `ResolvedSpec`, `ArchitectureView`, `jacobian_vector_product`, `PROJECT_ROOT`, and the registered model configs.
- Produces: one public preflight executor accepting a model name and injectable loader/output root; one central `require_preflight(ModelSpec, *, skip: bool = False, ...)` guard; deterministic report data/persistence helpers; and CLI stages/flags that delegate to these functions.
- The artifact must expose stable top-level identity and status fields so the guard does not infer them from display text: schema version, model name, `hf_id`, snapshot revision, and overall passed status.

- [ ] **Step 1: Write failing pure/report tests in `tests/test_preflight.py`.**

  Cover a fully fake model/tokenizer/view inspection without calling the default loader. Assert exact architecture layer counts, dimensions, tied status, native cache class names, resolved cache strategy, 64-token residual equivalence result, middle layer, forward JVP method and finite result, all four thinking-mode prompt/token-count entries, exact LoRA paths/count, exact memory formula/budget outcome, and stable JSON bytes/path across two identical fake runs. Include a forward-JVP failure/non-finite case that selects the finite-difference fallback.

- [ ] **Step 2: Run the new tests and capture RED evidence.**

  Run: `uv run pytest -q tests/test_preflight.py`

  Expected: collection/import failure because `pipeline/preflight.py` does not exist.

- [ ] **Step 3: Implement the smallest coherent preflight core.**

  Keep MLX, MLX-LM, and Hugging Face runtime imports inside functions. Build exactly 64 deterministic token IDs from a fixed prompt; compare the view's manually traversed final residual with the text module's native final residual and record max absolute error plus a strict pass result. Probe the middle residual with an all-ones float32 tangent; try `jacobian_vector_product(..., method="forward")`, fall back only on a raised exception or non-finite output, and fail if the fallback is non-finite. Render one fixed user message for `unsupported`, `off`, `inference`, and `trained`, preserving the model's chat-template kwargs while setting/removing `enable_thinking` appropriately, and store both prompt and token count. Count parameter-tree bytes without eagerly evaluating arrays, apply the exact activation formula, and emit byte/GiB components and `within_budget`.

  Derive the current remote revision from `${HF_HOME}/hub/models--<org>--<repo>/refs/main` after `configure_local_cache()`; if `ModelSpec.resolve()` did not expose a revision, bind this cache revision into the resolved report. Report native cache entry type names without attempting generation. Write JSON directly with `json.dumps(..., indent=2, sort_keys=True) + "\n"`.

- [ ] **Step 4: Run focused report tests and capture GREEN evidence.**

  Run: `uv run pytest -q tests/test_preflight.py`

  Expected: all tests pass with no checkpoint load and pristine output.

- [ ] **Step 5: Add failing stale/missing guard tests.**

  In `tests/test_preflight.py`, cover missing artifact, malformed JSON, wrong registered name, wrong `hf_id`, absent cached revision, stale revision, failed preflight status, failed memory budget, current valid evidence, and `skip=True`. Assert failures occur before a supplied sentinel loader/action can run.

- [ ] **Step 6: Run the focused guard tests and capture RED evidence.**

  Run: `uv run pytest -q tests/test_preflight.py -k 'require_preflight or artifact'`

  Expected: new tests fail because the guard is absent/incomplete.

- [ ] **Step 7: Implement strict central pre-load validation.**

  Validate the direct artifact path for `spec.name`; reject absent/unreadable/non-object records and exact identity/revision/status mismatches with actionable `SystemExit` messages naming `agent-pipeline preflight --model <name>` and, where appropriate, `--skip-preflight-check`. Return the validated mapping for callers/tests. `skip=True` returns without reading cache or artifact.

- [ ] **Step 8: Run all module tests and capture GREEN evidence.**

  Run: `uv run pytest -q tests/test_preflight.py`

  Expected: all pass, no model/checkpoint load.

- [ ] **Step 9: Add failing CLI wiring tests in `tests/test_cli.py`.**

  Cover `preflight --model qwen35-4b` dispatch without loading the YAML run config; `train`, `select`, `eval`, and `all` calling the guard before their first model-loading stage; `--skip-preflight-check` propagation; and existing stage-call signatures. Monkeypatch the executor/guard/stages so no checkpoint is loaded.

- [ ] **Step 10: Run the CLI tests and capture RED evidence.**

  Run: `uv run pytest -q tests/test_cli.py -k 'preflight'`

  Expected: failures because the parser and guard wiring do not yet exist.

- [ ] **Step 11: Wire the CLI seam minimally.**

  Add `preflight --model <name>`. Dispatch it before `load_config()` so it depends only on the registry model argument. Add `--skip-preflight-check` to `train`, `select`, `eval`, and `all`; call one helper that loads the declared `ModelSpec` and invokes `require_preflight` before `_resolve_training_spec`, `run_evaluation`, or the selection loop. In `all`, avoid duplicate validation while ensuring the single guard precedes training. Preserve every unrelated stage and its existing behavior.

- [ ] **Step 12: Run focused combined tests and Ruff.**

  Run: `uv run pytest -q tests/test_preflight.py tests/test_cli.py`

  Run: `uv run ruff check src/local_llm_lab/pipeline/preflight.py src/local_llm_lab/pipeline/cli.py tests/test_preflight.py tests/test_cli.py`

  Expected: all tests and lint pass with pristine output.

- [ ] **Step 13: Write the under-review implementation report.**

  Record the approved scope; artifact contract; guard order; fake-only RED/GREEN evidence; R10/R11/R13 rulings; focused test/lint evidence; baseline HEAD `65e93fbb91e0e077413d13b4c9be905f7c08eee8` with 498 passing tests; the exact files changed; and the deliberate follow-up that probe commands remain unwired until their owning lane releases them. Reserve a clearly labeled section for the controller's later serialized `qwen35-4b` execution evidence.

- [ ] **Step 14: Inspect the exact scoped diff and run R13.**

  Run: `git diff --check -- src/local_llm_lab/pipeline/preflight.py src/local_llm_lab/pipeline/cli.py tests/test_preflight.py tests/test_cli.py design_specifications/under_review/SPEC-001-S10-PREFLIGHT-IMPLEMENTATION-REPORT.md`

  Run exact: `uv run pytest -q`

  Record immediately before the run: `git rev-parse HEAD` and `uv run pytest --collect-only -q` node count. Record the exact exit code and passing-test count from the full run in the report.

- [ ] **Step 15: Self-review, commit only the task files, and report.**

  Re-read the diff for eager imports/loads, swallowed JVP failures, nondeterministic values, non-pre-load guards, accidental probe edits, and foreign changes. Stage only the five implementation/report files listed above; verify staged names and patch; commit with a narrow message. Do not run the real preflight command.

---

## Controller-only completion sequence

After Task 1 has both independent spec compliance and quality approval, the controller performs the final whole-diff review/verification required by SDD. Then this exact native task re-lists the board, adds exclusive action `model-execution` to its current claim, and immediately runs exactly `uv run agent-pipeline preflight --model qwen35-4b`. No training, evaluation, probe, push, or second model command is permitted. Capture wall time, report memory estimate/observed MLX peak where available, JVP method, cache kinds/strategy, local cache footprint, and zero external API cost; release the exclusive action promptly; update only the claimed implementation report with execution evidence; verify the named artifact and final scoped tree; then release the task claim.
