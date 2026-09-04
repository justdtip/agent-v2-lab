# Issue 15 Residual Equivalence and Gate Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Falsify or confirm the BF16-versus-FP32 explanation for the `qwen35-4b` residual failure, correct the evidence criterion only if controls support it, and expose a consumer-specific preflight gate without overwriting the incident artifact.

**Architecture:** Preserve the briefing-mandated FP32 `ArchitectureView` path and add an explicitly diagnostic native-dtype traversal plus dtype-aware error instrumentation. Independently review that instrumentation and the consumer gate, run two serialized controls, then let the same implementer apply the measured conclusion and the same reviewer gate it before one protected Qwen3.5 retry. Failed artifacts remain evidence and return nonzero; canonical incident bytes never change.

**Tech Stack:** Python 3.13, MLX/MLX-LM, pytest, Ruff, deterministic JSON, Markdown incident reporting

**Spec:** `design_specifications/pending/SPEC-001-model-agnostic-backbone.md` §§2, 10-12; the user-ratified Deputy/Coordinator comments on GitHub issues #15 and #16

## Global Constraints

- Keep `ArchitectureView.embed(ids)` and `ArchitectureView.run_block(...)` float32. Activations and tangents remain float32 and weights remain quantised; do not remove or weaken those casts.
- Treat BF16-versus-FP32 as the leading hypothesis, not a conclusion. The Qwen3.5 native-dtype manual-loop control must try to falsify it before the equivalence pass criterion changes.
- The regression fake is genuinely BF16 and hybrid: three linear-attention blocks and one attention block, with distinct native masks and no cache.
- Record both maximum absolute error and scale-normalized maximum relative error. Proposed tolerance is derived before controls from the reference/native dtype: `relative_tolerance = 2 * finfo(reference.dtype).eps` and `absolute_tolerance = relative_tolerance * max(reference_scale, finfo(reference.dtype).tiny)`, where `reference_scale = max(abs(reference))`. Do not tune either threshold after observing a model result.
- The pre-fix instrumentation may report `within_tolerance`, but `_residual_equivalence(...)["passed"]` remains exact equality until both controls support the hypothesis and the fix is independently reviewed.
- `outputs/preflight/qwen35-4b.json` is immutable incident evidence. Its bytes retain SHA-256 `499efee17d6ba76f389ff975dbc227d7e569d5e4836963be4260fcafba9d5679` throughout.
- `require_preflight` defaults to the view/probe predicate and remains fail-closed. The training predicate requires schema, identity, revision, memory, prompt-rendering, and LoRA evidence but deliberately does not depend on residual/JVP. No CLI call-site changes belong to this task.
- A failed preflight writes deterministic evidence and then exits nonzero.
- Offline work does not load checkpoints, tokenizers, weights, or execute real-model inference. Tests use only local fakes.
- Do not run training, evaluation, rollout, probe capture, cache attestation, push, publish, or destructive commands. Do not reclaim `pipeline/cli.py` or `tests/test_cli.py`.
- Model processes are serialized and never resident concurrently. Before each control/retry: independently approve the exact command, re-list the board, acquire exclusive `model-execution`, run exactly once, end the process, release promptly, and re-hash the canonical artifact.
- Authorized real-model executions are exactly two pre-fix controls (Qwen3.5 native-dtype manual loop; Qwen2.5 BF16 preflight) and one post-fix Qwen3.5 preflight retry. No second attempt of any command is allowed.
- Control outputs live only under `outputs/preflight/issue-15-controls/`; retry output lives only under `outputs/preflight/issue-15-retry/`. All targets must be absent before their one write.
- Use one principal implementer for all code/report phases and one independent principal reviewer for every gate.

---

### Task 1: Instrument, control, fix, and document residual equivalence and consumer gates

**Files:**
- Modify: `src/local_llm_lab/arch.py`
- Modify: `src/local_llm_lab/pipeline/preflight.py`
- Create: `tests/test_arch.py`
- Modify: `tests/test_preflight.py`
- Create and append measured evidence: `design_specifications/under_review/GITHUB-ISSUE-15-RESIDUAL-EQUIVALENCE-DIAGNOSIS.md`

**Interfaces:**
- Produces `ArchitectureView.diagnostic_native_final_residual(ids) -> array`: the raw embedding, native masks, direct blocks, and final norm with no FP32 promotion and no cache. It is diagnostic-only; all existing view methods retain their contracts.
- Produces `run_residual_control(model_name, *, output_path, ...) -> Path`: fake-testable, one-model diagnostic JSON comparing FP32 manual, native-dtype manual, and native reference residuals.
- Produces `_residual_metrics(actual, reference, *, array_api) -> dict[str, Any]`: `max_abs_error`, `max_relative_error`, `reference_dtype`, `reference_epsilon`, `reference_scale`, `relative_tolerance`, `absolute_tolerance`, and `within_tolerance`.
- Produces `require_preflight(spec, *, consumer: Literal["view", "training"] = "view", ...)`: common identity/revision/schema validation followed by consumer-specific evidence validation.
- Preserves `run_preflight(...) -> Path` on success and changes failure to `SystemExit` only after the artifact is written.

- [ ] **Step 1: Freeze evidence and trace both paths before product edits**

  Record HEAD, `git status --short`, the canonical artifact hash, the installed MLX-LM Qwen3.5/Qwen2 source, cached Qwen3.5 config, and safetensors metadata. Begin the diagnosis report with a table whose exact rows are: embedding dtype, attention/SSM masks, cache/no-cache, per-kind blocks, residual indices, per-block dtype boundary, final norm. Record the leading hypothesis exactly as a hypothesis and bank the canonical successful evidence: finite-difference JVP finite at layer 16, 12 LoRA target suffixes / 32,464,896 trainable parameters, 32 layers (24 linear-attention / 8 attention), 3.8451762199401855 GiB within 22 GiB, and cache strategy `none` via unverified equivalence.

- [ ] **Step 2: Add the BF16 hybrid RED regression for native-dtype diagnostics**

  In `tests/test_arch.py`, use a real `mx.bfloat16` embedding and four dtype-sensitive fake blocks with `is_linear == (True, True, True, False)`. The fake text module has Qwen's `__call__(ids, cache=None, input_embeddings=None)` shape, builds one attention and one SSM mask per traversal, uses no cache, and records input/output dtype, block kind/order, residual index 4, and one final norm.

  Add `test_hybrid_native_diagnostic_preserves_bfloat16_and_matches_reference`. It computes the normal FP32 view residual, the fake's default BF16 reference, and `view.diagnostic_native_final_residual(ids)`, then asserts:

  ```python
  assert view.embed(ids).dtype == mx.float32
  assert fp32_residual.dtype == mx.float32
  assert native_reference.dtype == mx.bfloat16
  assert native_manual.dtype == mx.bfloat16
  assert float(mx.max(mx.abs(fp32_residual - native_reference)).item()) > 0.0
  assert float(mx.max(mx.abs(native_manual - native_reference)).item()) == 0.0
  ```

  Mutation sensitivity: deleting the diagnostic method, routing it through `embed`/`run_block`/`final_norm`, using the wrong kind mask, introducing a cache, changing block order, or skipping/doubling norm must fail the test.

- [ ] **Step 3: Capture RED and implement only the diagnostic traversal**

  Run:

  ```bash
  .venv/bin/python -m pytest -q tests/test_arch.py::test_hybrid_native_diagnostic_preserves_bfloat16_and_matches_reference
  ```

  Expect a missing-method failure. Record command, HEAD, exit, and concise failure. Implement `diagnostic_native_final_residual` in `arch.py` using the same token-id normalization as `embed`, raw `text_module.embed_tokens`, `masks(h, None)`, direct block invocation with `masks[layer_kind(index)]` and `cache=None`, then raw `text_module.norm`. Do not call the existing FP32-promoting `embed`, `run_block`, or `final_norm` methods.

- [ ] **Step 4: Add dtype-derived relative/absolute metric instrumentation**

  Add fake-only tests for `_residual_metrics` using BF16 and FP32 references. Compute:

  ```python
  info = array_api.finfo(reference.dtype)
  scale = _scalar(array_api.max(array_api.abs(reference)))
  relative_tolerance = 2.0 * float(info.eps)
  absolute_tolerance = relative_tolerance * max(scale, float(info.tiny))
  max_abs_error = _scalar(array_api.max(array_api.abs(actual - reference)))
  max_relative_error = max_abs_error / max(scale, float(info.tiny))
  within_tolerance = max_abs_error <= absolute_tolerance
  ```

  Tests prove the values change with reference dtype and scale and reject a post-hoc hard-coded `1e-5`. Extend `_residual_equivalence` to serialize these fields for FP32-manual versus native-reference, but at this phase keep `passed` equal to exact equality and serialize `criterion: "exact_pre_control"`.

- [ ] **Step 5: Add and fake-test the one-model residual control writer**

  Add `run_residual_control` with injected loader/spec-loader/view-factory/revision-reader/array-api seams. It loads lazily, builds the exact 64-token prompt, computes three residuals—normal FP32 view loop, `diagnostic_native_final_residual`, and `text_module(ids)` native reference—and writes deterministic JSON to an explicit `output_path`. The JSON contains schema/model/HF/revision/token identity, FP32-manual-versus-native metrics, and native-manual-versus-native metrics. It never calls JVP, LoRA discovery, cache creation, memory estimation, prompt-mode rendering, or another model load.

- [ ] **Step 6: Add the consumer-specific gate API from issue #16**

  Add a `consumer` keyword to `require_preflight`, defaulting to `"view"`. Common checks remain schema version, registered model name, HF id, and current cached revision.

  The training predicate requires all of:

  ```python
  memory["within_budget"] is True
  [entry["mode"] for entry in thinking_prompts] == ["unsupported", "off", "inference", "trained"]
  all(isinstance(entry["prompt"], str) and entry["prompt"] for entry in thinking_prompts)
  all(isinstance(entry["token_count"], int) and entry["token_count"] > 0 for entry in thinking_prompts)
  isinstance(lora["keys"], list) and lora["keys"] and all(nonempty strings)
  isinstance(lora["trainable_parameters"], int) and lora["trainable_parameters"] > 0
  ```

  The view predicate requires every training predicate plus `residual_equivalence["passed"] is True`, `jvp["finite"] is True`, and top-level `passed is True`. Unknown consumers fail before any action. Add tests that the preserved shape of a `passed: false` artifact can satisfy `consumer="training"` while default/`consumer="view"` rejects it, and parameterize malformed memory/rendering/LoRA/residual/JVP cases. Preserve skip semantics and action-after-validation ordering. Do not touch CLI call sites.

- [ ] **Step 7: Make failed preflights write evidence then exit nonzero**

  Add `test_failed_preflight_writes_evidence_then_exits_nonzero`. Use an injected mismatching fake, expect `SystemExit` containing `preflight failed`, then parse its `tmp_path` artifact and assert its failed metrics were written. In `run_preflight`, store `path = write_report(...)`, return it only when `report["passed"] is True`, otherwise raise `SystemExit` naming model and path.

- [ ] **Step 8: Run offline GREEN/lint, complete the pre-control report, and commit**

  Run and record exact exits/counts:

  ```bash
  .venv/bin/python -m pytest -q tests/test_arch.py tests/test_preflight.py
  .venv/bin/python -m pytest -q tests/test_probes.py -k 'architecture_view'
  .venv/bin/ruff check src/local_llm_lab/arch.py src/local_llm_lab/pipeline/preflight.py tests/test_arch.py tests/test_preflight.py
  ```

  Re-hash the incident artifact. The report records the trace, single hypothesis, BF16 RED/GREEN, metrics/formula, gate API, failed-exit decision, exact commands/HEADs/exits/counts, changed files, and that no model/weights ran. Commit only claimed source/tests/report.

- [ ] **Step 9: Independent pre-control review and exact two-command authorization**

  The reviewer issues spec-compliance and code-quality verdicts, actively tries to falsify the hypothesis/test fixture, checks the tolerance is dtype/scale-derived but not yet used as the pass criterion, and approves or rejects these exact absent-target commands separately:

  ```bash
  .venv/bin/python -c 'from pathlib import Path; from local_llm_lab.pipeline.preflight import run_residual_control; run_residual_control("qwen35-4b", output_path=Path("outputs/preflight/issue-15-controls/qwen35-native-loop.json"))'
  .venv/bin/python -c 'from pathlib import Path; from local_llm_lab.pipeline.preflight import run_preflight; run_preflight("qwen25-coder-3b", output_root=Path("outputs/preflight/issue-15-controls"))'
  ```

- [ ] **Step 10: Controller-only serialized controls**

  For each approved command: verify target absence; re-list the board; record HEAD/status/canonical hash; acquire `model-execution`; run that command exactly once; ensure the process ends; release promptly; record exit/output/artifact hash; and reconfirm the canonical hash. Release between commands when practical. Never keep Qwen3.5 and Qwen2.5 resident concurrently. If Qwen3.5 native-manual-versus-native does not collapse within its dtype-derived tolerance, the hypothesis is falsified: do not apply the tolerance fix or run the Qwen3.5 retry; return to one-hypothesis-at-a-time offline diagnosis.

- [ ] **Step 11: Apply the measured criterion only if both controls support it**

  Give the same principal implementer the two immutable control artifacts. If Qwen3.5 native-manual-versus-native is within tolerance and Qwen2.5 shows the same class of FP32-versus-BF16 divergence, change `_residual_equivalence` to set `passed = metrics["within_tolerance"]` and `criterion = "reference_dtype_scaled"`. Do not change the formula. Update fake expectations, append both controls and the conclusion to the diagnosis report, and commit. The same independent reviewer performs a scoped re-review.

- [ ] **Step 12: R13 before the one Qwen3.5 retry**

  After review approval, record HEAD and run:

  ```bash
  .venv/bin/python -m pytest --collect-only -q
  .venv/bin/python -m pytest -q
  .venv/bin/ruff check src tests
  ```

  Record exact exits and collected/passed counts; confirm the canonical hash and all three execution targets (two controls present, retry absent).

- [ ] **Step 13: Independently authorize and run one protected Qwen3.5 retry**

  The reviewer approves or rejects this exact command and confirms it cannot overwrite the canonical artifact:

  ```bash
  .venv/bin/python -c 'from pathlib import Path; from local_llm_lab.pipeline.preflight import run_preflight; run_preflight("qwen35-4b", output_root=Path("outputs/preflight/issue-15-retry"))'
  ```

  If approved, the controller verifies target absence, re-lists/acquires `model-execution`, runs once, ends the process, releases immediately, and records command/HEAD/exit/counts/tolerance/evidence/hash. No second retry is allowed.

- [ ] **Step 14: Final report append, scoped review, and fail-closed checks**

  The same implementer appends the retry result, every lock transition, all artifact hashes, the gate API contract for issue #14 P1, and the explicit no-bypass/no-extra-execution statement; then commits only the report. The same reviewer re-reviews that append. The controller confirms the preserved canonical artifact still fails default/view `require_preflight` without loading a model and records whether its training predicate passes. Reconfirm canonical SHA-256 and notify issue #14 P1 of the released `require_preflight(..., consumer="training")` contract without editing its CLI files.
