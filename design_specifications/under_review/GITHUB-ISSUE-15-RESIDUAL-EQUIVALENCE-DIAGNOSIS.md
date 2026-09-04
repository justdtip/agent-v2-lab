# GITHUB-ISSUE-15 residual-equivalence diagnosis (pre-control)

This report records the offline implementation, with no model, weight, tokenizer, or preflight execution.

## Scope and frozen evidence

The revised-task base was `bd9caa8e21cce565bb66c624567e90301f30790d`. The canonical incident
artifact is `outputs/preflight/qwen35-4b.json`, SHA-256
`499efee17d6ba76f389ff975dbc227d7e569d5e4836963be4260fcafba9d5679`, with original maximum
absolute residual error `2.5591506958007812`. The worktree initially had foreign heartbeat-log
changes and untracked `data/`; neither was modified.

Cached Qwen3.5 revision `32f3e8ecf65426fc3306969496342d504bfa13f3` is under
`.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/`. Its `config.json` declares
BF16 text dtype, 32 layers, hidden size 2560, full-attention interval 4, and the three-linear/one-
full pattern. The safetensors index reports `total_size: 3034147328`; the cache tree records
`model.safetensors` size 3034300695 and LFS SHA-256
`5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db`. This was metadata-only
inspection; no tensor was materialized.

Installed `.venv/lib/python3.13/site-packages/mlx_lm/models/qwen3_5.py:254-275` shows a native
raw embedding, both masks, `is_linear` mask selection, no-cache list, and one final norm. The
installed Qwen2/Qwen3 loops expose the same optional `input_embeddings` signature.

Banked canonical evidence: finite-difference JVP finite at layer 16; 12 LoRA target suffixes /
32,464,896 trainable parameters; 32 layers (24 linear-attention / 8 attention); memory
3.8451762199401855 GiB within 22 GiB; cache strategy `none` from unverified equivalence.

## Path trace and hypothesis

| Path aspect | FP32 view loop | Native and diagnostic loop |
| --- | --- | --- |
| embedding dtype | `embed()` promotes to FP32 | raw embedding is BF16 |
| attention/SSM masks | one helper mask of each kind | same helpers and two masks |
| cache/no-cache | masks and blocks use `None` | native none list; diagnostic `None` blocks |
| per-kind blocks | `run_block` uses `is_linear` | SSM for linear; attention otherwise |
| residual indices | layer 0; index 4 pre-norm in fake | same four-block traversal |
| per-block dtype boundary | every result promoted FP32 | diagnostic preserves native dtype |
| final norm | norm output promoted FP32 | one raw `text_module.norm` |

Leading hypothesis: **the Qwen checkpoint's BF16 native embedding path and the view's FP32-promoted
path are different numerical programs, so the current oracle can accumulate a large difference even
when masks, block order, residual indices, and final norm are structurally correct.** It remains a
hypothesis until controller-only controls measure both paths.

## TDD and implementation

RED at base `bd9caa8e21cce565bb66c624567e90301f30790d`:

```text
.venv/bin/python -m pytest -q tests/test_arch.py::test_hybrid_native_diagnostic_preserves_bfloat16_and_matches_reference
exit 1: AttributeError: 'ArchitectureView' object has no attribute 'diagnostic_native_final_residual'
```

The genuine-MLX fake has four dtype-sensitive blocks with `is_linear == (True, True, True, False)`.
It asserts one attention mask, one SSM mask, no cache, three SSM then one attention block, residual
index 4, one norm, and BF16 input/output. FP32 output differs from the BF16 reference while the raw
diagnostic loop exactly matches it. Removal, use of FP32 helpers, wrong masks/order, caching, or a
changed norm count fails the test.

GREEN:

```text
.venv/bin/python -m pytest -q tests/test_arch.py::test_hybrid_native_diagnostic_preserves_bfloat16_and_matches_reference
exit 0: 1 passed
```

`_residual_metrics` records dtype, epsilon, scale, absolute/relative error, and:

```text
relative_tolerance = 2 * finfo(reference.dtype).eps
absolute_tolerance = relative_tolerance * max(reference_scale, finfo(reference.dtype).tiny)
max_relative_error = max_abs_error / max(reference_scale, finfo(reference.dtype).tiny)
```

Exact equality remains the pass criterion: `criterion: "exact_pre_control"` and zero error. The
fake BF16/FP32 metric test proves the derived budget depends on reference precision and scale, not a
hard-coded `1e-5`.

`run_residual_control` writes a fake-tested, one-lazy-load comparison of FP32 manual,
native-dtype manual, and native reference residuals for the exact 64-token identity. It does not
call JVP, LoRA discovery, cache creation, memory estimation, prompt rendering, or a second loader.

`require_preflight` defaults to `consumer="view"`. Common identity/revision checks now require
memory, ordered nonempty thinking prompts, and nonempty LoRA evidence. `training` can accept that
evidence while view equivalence fails; `view` additionally requires passed residual equivalence,
finite JVP, and top-level `passed`. Unknown consumers fail before action; known `skip` calls still
short-circuit reads. Failed `run_preflight` calls write evidence before nonzero exit.

## Offline verification

The metrics RED was an exit-1 collection failure for missing `_residual_metrics`; focused preflight
GREEN then passed 25 tests. Required checks at shared HEAD
`6fa50d86756930a1f03ca0a1d5c2d87be00ec16c`:

```text
.venv/bin/python -m pytest -q tests/test_arch.py tests/test_preflight.py
exit 0: 26 passed

.venv/bin/python -m pytest -q tests/test_probes.py -k 'architecture_view'
exit 0: 13 passed

.venv/bin/ruff check src/local_llm_lab/arch.py src/local_llm_lab/pipeline/preflight.py tests/test_arch.py tests/test_preflight.py
exit 0: All checks passed!
```

The probe command initially could not collect inside the sandbox because its existing MLX fakes
could not see Metal; the outside-sandbox rerun above used no model assets and passed. The canonical
artifact was rehashed after verification and remains
`499efee17d6ba76f389ff975dbc227d7e569d5e4836963be4260fcafba9d5679`.

Changed files are `src/local_llm_lab/arch.py`, `src/local_llm_lab/pipeline/preflight.py`,
`tests/test_arch.py`, and `tests/test_preflight.py`. No checkpoint inference or model/weight/tokenizer
load occurred. The implementation stops before all control, review, and retry steps.

## Fix round 1: terminal non-finite JVP evidence

Review found that `_jvp_result` previously raised before `run_preflight` built and wrote a report
when both forward and finite-difference JVPs were non-finite. The smallest fix returns
`{"finite": False, "layer": layer, "method": "finite_difference"}` instead. `run_preflight`
then completes its normal report, writes it, and raises its existing `preflight failed` exit.

RED at shared HEAD `a87124590417d5272dea119ced9ef2e5b5811f37`:

```text
.venv/bin/python -m pytest -q tests/test_preflight.py::test_nonfinite_jvp_writes_failed_evidence_then_exits_nonzero
exit 1: expected preflight failed; got preflight JVP is non-finite after finite-difference fallback
```

The fake-only regression makes both JVP methods non-finite and asserts nonzero exit after the
artifact exists with `jvp.finite is False`. GREEN evidence:

```text
.venv/bin/python -m pytest -q tests/test_preflight.py
exit 0: 26 passed

.venv/bin/python -m pytest -q tests/test_arch.py tests/test_preflight.py
exit 0: 27 passed

.venv/bin/ruff check src/local_llm_lab/arch.py src/local_llm_lab/pipeline/preflight.py tests/test_arch.py tests/test_preflight.py
exit 0: All checks passed!
```

No model, checkpoint, tokenizer, preflight, or control command was run. Canonical artifact SHA-256
remains `499efee17d6ba76f389ff975dbc227d7e569d5e4836963be4260fcafba9d5679`.
