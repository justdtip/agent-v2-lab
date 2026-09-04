# Issue 15 Residual Equivalence Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose and correct the `qwen35-4b` preflight residual-equivalence failure without changing the float32 `ArchitectureView` contract or overwriting the failed incident artifact.

**Architecture:** Treat the existing failed artifact as immutable incident evidence. Reproduce the suspected BF16-versus-FP32 control-path split with a small hybrid fake, then make the preflight compare the manual float32 traversal with the model-native traversal seeded by the same float32 embedding. Use the specification's `1e-5` absolute tolerance, write failed evidence before returning a nonzero status, independently review all offline work, and permit at most one serialized real-model retry to a separate output root after the reviewer approves the exact command.

**Tech Stack:** Python 3.13, MLX/MLX-LM, pytest, Ruff, deterministic JSON, Markdown incident reporting

**Spec:** `design_specifications/pending/SPEC-001-model-agnostic-backbone.md` §§2, 10-12 and GitHub issue #15 assignment from the goal Coordinator

## Global Constraints

- `ArchitectureView.embed(ids)` and `ArchitectureView.run_block(...)` continue returning float32; activations and tangents remain float32 and weights remain quantised.
- The residual contract compares the view's pre-unembedding final residual with the text module's own block/mask/final-norm path to absolute tolerance `1e-5` on exactly 64 tokens.
- `outputs/preflight/qwen35-4b.json` is immutable incident evidence. Its bytes must retain SHA-256 `499efee17d6ba76f389ff975dbc227d7e569d5e4836963be4260fcafba9d5679` throughout this task.
- `require_preflight` remains fail-closed for missing, malformed, stale, unsupported-schema, or `passed: false` artifacts. No skip/bypass path is added or exercised.
- A failed `agent-pipeline preflight` writes its deterministic evidence artifact and exits nonzero.
- Offline implementation and review must not load a checkpoint, tokenizer, weights, or execute model inference. Tests use only local fakes.
- Do not run training, evaluation, rollout, probe, cache attestation, push, publish, or any destructive command.
- At most one real `qwen35-4b` retry is allowed. Before it, the reviewer must approve the exact command, the Coordinator board must be re-listed, and this task must acquire exclusive `model-execution`.
- The retry must write to a distinct issue-15 evidence directory, never the canonical `outputs/preflight/qwen35-4b.json` path.
- One principal implementer owns the code/report work and one independent principal reviewer gates it.

---

### Task 1: Reproduce, fix, review, and document the precision-path mismatch

**Files:**
- Inspect, modify only if the regression proves it necessary: `src/local_llm_lab/arch.py`
- Modify: `src/local_llm_lab/pipeline/preflight.py`
- Create: `tests/test_arch.py`
- Modify: `tests/test_preflight.py`
- Create and later append reviewed retry evidence: `design_specifications/under_review/GITHUB-ISSUE-15-RESIDUAL-EQUIVALENCE-DIAGNOSIS.md`

**Interfaces:**
- Consumes: `ArchitectureView.embed(ids)`, `masks(h, cache)`, `run_block(i, h, masks, cache_i)`, `final_norm(h)`, and a Qwen-style `text_module(ids, cache=None, input_embeddings=None)`.
- Produces: `_residual_equivalence(view, ids, array_api) -> dict[str, Any]` with `max_abs_error`, `absolute_tolerance`, `passed`, and `token_count`; `run_preflight(...) -> Path` on success and `SystemExit` after writing evidence on failure.

- [ ] **Step 1: Freeze the incident and write the offline trace before editing product code**

  Record the current HEAD, `git status --short`, and SHA-256 of `outputs/preflight/qwen35-4b.json`. In the diagnosis report, trace the installed Qwen3.5 path against `ArchitectureView` in a table with these exact rows: embedding dtype, attention/SSM mask construction, cache/no-cache behavior, per-kind block selection, residual index convention, per-block dtype handling, and final norm. Cite the local installed source and cached `config.json`/safetensors metadata. The first hypothesis must be exactly: the Qwen checkpoint's BF16 native embedding path and the view's FP32-promoted path are different numerical programs, so the current oracle can accumulate a large difference even when masks, block order, residual indices, and final norm are structurally correct. Do not propose or implement a second hypothesis unless the RED test falsifies this one.

- [ ] **Step 2: Add the mutation-sensitive hybrid fake regression**

  Create `tests/test_arch.py` with a four-block fake whose `is_linear` sequence is `(True, True, True, False)`, whose attention and SSM helpers return distinct sentinels, whose embedding output is low precision, and whose nonlinear block arithmetic makes FP32 seeding observably differ from the default low-precision native path. The fake text module must accept `input_embeddings=None` exactly like installed Qwen2/Qwen3/Qwen3.5.

  The regression must establish all three facts, not merely one:

  ```python
  view = ArchitectureView.from_model(model)
  embedded = view.embed(ids)
  result = _residual_equivalence(view, ids, mx)

  assert embedded.dtype == mx.float32
  assert default_native_error > 1e-5  # the fixture is mutation-sensitive
  assert result == {
      "absolute_tolerance": 1e-5,
      "max_abs_error": 0.0,
      "passed": True,
      "token_count": 64,
  }
  ```

  The fake must record and assert that both native and manual traversals use no cache, one attention mask, one SSM mask, three linear-attention blocks, one attention block, residual index 4 before the final norm, and the same final norm. If removing the `input_embeddings=` seed from the eventual production fix would not fail this test, strengthen the fixture before proceeding.

- [ ] **Step 3: Run the single regression and capture RED**

  Run:

  ```bash
  .venv/bin/python -m pytest -q tests/test_arch.py::test_hybrid_float32_equivalence_uses_the_same_embedding_seed
  ```

  Expected: FAIL because the current `_native_final_residual` calls `text_module(ids)` without the view's float32 embedding. Record command, exit status, and the concise assertion failure in the diagnosis report. If the test does not fail for that reason, stop product edits and return `BLOCKED` with the falsifying evidence.

- [ ] **Step 4: Make the minimal apples-to-apples residual comparison**

  In `src/local_llm_lab/pipeline/preflight.py`, add:

  ```python
  _RESIDUAL_ABSOLUTE_TOLERANCE = 1e-5
  ```

  Preserve the first float32 embedding before the manual loop and pass that exact array to the native text-module path:

  ```python
  embedded = view.embed(ids)
  manual = embedded
  masks = view.masks(manual, None)
  for index in range(view.num_layers):
      manual = view.run_block(index, manual, masks, None)
  manual = view.final_norm(manual)
  native = _native_final_residual(view, ids, embedded)
  error = _scalar(array_api.max(array_api.abs(manual - native)))
  return {
      "absolute_tolerance": _RESIDUAL_ABSOLUTE_TOLERANCE,
      "max_abs_error": error,
      "passed": error <= _RESIDUAL_ABSOLUTE_TOLERANCE,
      "token_count": 64,
  }
  ```

  Change `_native_final_residual` to accept `input_embeddings` and call:

  ```python
  native = view.text_module(ids, input_embeddings=input_embeddings)
  ```

  Keep its existing `last_hidden_state`/tuple/direct-array normalization. Do not relax `ArchitectureView`'s float32 contract and do not change block, mask, cache, residual-index, or norm behavior unless the RED fixture independently proves such a change is required.

- [ ] **Step 5: Prove failed preflights preserve evidence and exit nonzero**

  In `tests/test_preflight.py`, update the successful report expectation to include `"absolute_tolerance": 1e-5`. Add a mismatching fake view and a test named `test_failed_preflight_writes_evidence_then_exits_nonzero` which invokes `run_preflight(..., output_root=tmp_path)` under `pytest.raises(SystemExit, match="preflight failed")`, then reads `tmp_path / "fake-model.json"` and asserts `passed is False`, residual `passed is False`, and the mismatch/tolerance fields are present.

  In `run_preflight`, always call `write_report` first. Return the resulting path only when `report["passed"] is True`; otherwise raise `SystemExit` whose message names the failed model and the evidence path. Do not modify `pipeline/cli.py`: its existing direct call will naturally exit nonzero.

- [ ] **Step 6: Run GREEN, focused compatibility checks, and lint**

  Run and record exact exits/counts:

  ```bash
  .venv/bin/python -m pytest -q tests/test_arch.py tests/test_preflight.py
  .venv/bin/python -m pytest -q tests/test_probes.py -k 'architecture_view'
  .venv/bin/ruff check src/local_llm_lab/arch.py src/local_llm_lab/pipeline/preflight.py tests/test_arch.py tests/test_preflight.py
  ```

  Re-run the incident SHA-256 check and assert it still equals `499efee17d6ba76f389ff975dbc227d7e569d5e4836963be4260fcafba9d5679`.

- [ ] **Step 7: Complete the offline diagnosis report and commit**

  The report must contain: scope and exclusions; immutable incident path/hash and original `2.5591506958007812` error; installed/cached-source evidence; the seven-row path trace; the single hypothesis; RED/GREEN/lint commands with HEADs, exits, and test counts; the exact `1e-5` tolerance rationale from SPEC-001 §2; the failed-command exit decision; changed files; and an explicit statement that no weights/model inference occurred. Commit only the claimed source, tests, and report.

- [ ] **Step 8: Independent offline review and exact retry authorization**

  The reviewer must issue both a spec-compliance verdict and a code-quality verdict. In addition, it must approve or reject this exact proposed retry shape: first verify that `outputs/preflight/issue-15-retry/qwen35-4b.json` does not exist, acquire `model-execution`, run exactly the following command once, and release the action immediately:

  ```bash
  .venv/bin/python -c 'from pathlib import Path; from local_llm_lab.pipeline.preflight import run_preflight; run_preflight("qwen35-4b", output_root=Path("outputs/preflight/issue-15-retry"))'
  ```

  Approval must confirm the command cannot resolve to or overwrite `outputs/preflight/qwen35-4b.json`; a failed retry is expected to exit nonzero only after writing its separate evidence file.

- [ ] **Step 9: Controller-only serialized retry, if approved**

  The controller re-lists the board, records HEAD/status/canonical hash, acquires exclusive `model-execution`, runs the approved command exactly once, records command/exit/output evidence, releases the action immediately, hashes both canonical and retry artifacts, and confirms the canonical hash is unchanged. No second real-model command is allowed regardless of result.

- [ ] **Step 10: Append retry evidence, re-review, and run R13 verification**

  The same principal implementer appends the controller-provided lock revisions, exact command, HEAD, exit, residual error/tolerance/pass status, artifact paths/hashes, and canonical immutability result to the diagnosis report, then commits the report-only update. The same independent reviewer performs a scoped re-review of that append. After approval, the controller records final HEAD and runs:

  ```bash
  .venv/bin/python -m pytest --collect-only -q
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check src tests
  ```

  Record exit codes and exact collected/passed counts in the final execution record. Reconfirm `require_preflight` rejects the preserved canonical `passed: false` artifact without loading a model, and reconfirm its SHA-256.
