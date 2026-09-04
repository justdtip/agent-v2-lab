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

## Runtime fix round 2: MLX finfo floor

The controller verified the Qwen3.5 native-loop target was absent, acquired `model-execution` at
claim revision 3, and ran exactly once:

```text
.venv/bin/python -c 'from pathlib import Path; from local_llm_lab.pipeline.preflight import run_residual_control; run_residual_control("qwen35-4b", output_path=Path("outputs/preflight/issue-15-controls/qwen35-native-loop.json"))'
```

It exited 1 before artifact write in `_residual_metrics` with
`AttributeError: 'finfo' object has no attribute 'tiny'`; the target remains absent. The controller
released `model-execution` at claim revision 4. The canonical artifact remained SHA-256
`499efee17d6ba76f389ff975dbc227d7e569d5e4836963be4260fcafba9d5679`.

The installed MLX finfo contract uses `.smallest_normal`, not NumPy's `.tiny`. RED changed the
metric fake to expose only `.smallest_normal` and ran:

```text
.venv/bin/python -m pytest -q tests/test_preflight.py::test_residual_metrics_follow_the_reference_dtype_and_scale
exit 1: AttributeError: 'types.SimpleNamespace' object has no attribute 'tiny'
```

The minimal fix substitutes `info.smallest_normal` for the scale floor in both the absolute and
relative formula through their shared `floor`; `2 * epsilon`, exact pre-control equality, gates,
and execution boundaries are unchanged. GREEN at shared HEAD
`d17feb316b8b3c9aace0b8597ad342f0aa8a56ca`:

```text
.venv/bin/python -m pytest -q tests/test_preflight.py
exit 0: 26 passed

.venv/bin/python -m pytest -q tests/test_arch.py tests/test_preflight.py
exit 0: 27 passed

.venv/bin/ruff check src/local_llm_lab/arch.py src/local_llm_lab/pipeline/preflight.py tests/test_arch.py tests/test_preflight.py
exit 0: All checks passed!
```

No model, checkpoint, tokenizer, preflight, control, or retry was run during this remediation.

## Step 11: structural RMS roundoff criterion

The two immutable control artifacts are preserved and rehashed: Qwen3.5 native-loop
`9a5bc0faa30c0c9bd256fa60fc0076afc38bcf651d69a052c039aecb666bfe73` and Qwen2.5 preflight
`c02011a36a02158e8f35d70e19cc52516c9766d99f751abf2a27a6cfbd65db17`. The canonical artifact
remains `499efee17d6ba76f389ff975dbc227d7e569d5e4836963be4260fcafba9d5679`.

Controller evidence: Qwen3.5 first acquired revision 3 and released revision 4, exiting 1 before
output on the MLX `.tiny` defect. The corrected control acquired revision 5 at `fd01f5b`, exited 0,
wrote the native-loop artifact, and released revision 6. It measured native-manual versus native
exactly zero and FP32 versus BF16 absolute `2.5591506958007812`, relative
`0.054741191354027406`, scale `46.75`, epsilon `0.0078125`. Qwen2.5 acquired revision 7 at
`e53bd810d5204541efb56ebd5b9c1682120c9cbf`, exited 1 only after persisting its report, and
released revision 8. It measured FP32 versus FP16 absolute `0.4695625305175781`, relative
`0.0032807862394241267`, scale `143.125`, epsilon `0.0009765625`.

Both controls reject the two-epsilon proposal: Qwen3.5 budget `0.015625` / `0.73046875`; Qwen2.5
budget `0.001953125` / `0.279541015625`. No multiplier is fitted to observed errors. The structural
replacement is `rounding_steps = 2 * num_layers + 1`,
`relative_tolerance = sqrt(rounding_steps) * reference_epsilon`, and absolute tolerance is that
budget times `max(reference_scale, smallest_normal)`. Predeclared Qwen3.5 budgets are 65 steps,
`0.06298638865858242` relative / `2.944613669788728` absolute; Qwen2.5 uses 73 steps,
`0.00834375365753665` / `1.194199742234933`.

RED at `770f7d516412be3d676caf615dba10680bb23d62` added `num_layers` metrics tests:

```text
.venv/bin/python -m pytest -q tests/test_preflight.py::test_residual_metrics_follow_the_reference_dtype_and_scale tests/test_preflight.py::test_residual_metrics_uses_rms_roundoff_boundaries tests/test_preflight.py::test_residual_metrics_rejects_invalid_layer_counts
exit 1: _residual_metrics() got an unexpected keyword argument 'num_layers'
```

The implementation validates positive non-bool integer counts, serializes `rounding_steps`, passes
`view.num_layers` into residual equivalence and both control metrics, switches the criterion to
`reference_dtype_rms_roundoff`, and uses `within_tolerance` for pass. FP32 view methods, gates, and
write-before-exit behavior are unchanged.

GREEN at the same shared HEAD:

```text
.venv/bin/python -m pytest -q tests/test_preflight.py
exit 0: 32 passed

.venv/bin/python -m pytest -q tests/test_arch.py tests/test_preflight.py
exit 0: 33 passed

.venv/bin/python -m pytest -q tests/test_probes.py -k 'architecture_view'
exit 0: 13 passed

.venv/bin/ruff check src/local_llm_lab/arch.py src/local_llm_lab/pipeline/preflight.py tests/test_arch.py tests/test_preflight.py
exit 0: All checks passed!
```

No model, preflight, control, or retry command was run in Step 11.
