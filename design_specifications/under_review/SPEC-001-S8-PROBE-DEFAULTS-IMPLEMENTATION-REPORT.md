# SPEC-001 §8 probe-defaults implementation report

Date: 2026-09-04
Task base: `6fa50d86756930a1f03ca0a1d5c2d87be00ec16c`
Mode: fake/offline implementation only

## Status

The model-aware policy and layer-selection implementation is complete through its focused,
adjacent, repository-rule, Ruff, C901, and diff gates. The exact committed-HEAD R13 result is
recorded in the final section after the implementation commit.

No checkpoint, tokenizer, model, probe, J-lens computation, inference, training, evaluation,
preflight, or agent/probe/pipeline CLI was run. Pytest invoked CLI `main()` functions only with
loaders, GPU guards, architecture views, and compute seams replaced by fakes.

## Observable behavior

- Policy names now come from the selected `ModelSpec`: the implicit `base` maps to `None`, qwen25
  retains registry policies A/B/C, qwen35 exposes no invented adapter names, and an explicit
  existing directory remains accepted as an absolute resolved path.
- `LayerSelection` records source, requested lexical tokens, fractions, resolved integer indices,
  and actual decoder depth. Its `as_dict()` output is JSON-safe and has exactly those five keys.
- Integer-looking tokens are one-based indices. Decimal/exponent tokens are fractions in
  `(0, 1]`; consequently lexical `1` is index 1 while `1.0` and `1e0` mean full depth.
- Registry defaults use only `ModelSpec.probe_layer_fractions`. At 32 layers they resolve to
  `[5, 11, 16, 21, 27, 32]`. Resolved collisions retain the first token and fraction.
- State probe, axis build, J-lens, and patching retain integer `layers` for existing compute APIs
  and add `layer_selection` metadata to their persisted results.
- Malformed layer syntax fails before GPU/model-loading seams. Bounds are checked against
  `ArchitectureView.num_layers` before probe, J-lens, axis, or patch computation.

## Policy and layer API

`src/local_llm_lab/probes/policies.py` now owns:

- `policy_names(spec) -> tuple[str, ...]`
- `resolve_policy(name, spec) -> Path | None`
- `validate_layer_syntax(raw) -> None`
- `resolve_layers(raw, spec, num_layers) -> LayerSelection`
- frozen `LayerSelection.as_dict() -> dict[str, object]`

The removed `POLICY_ADAPTERS`/`POLICY_NAMES` globals no longer allow one model's adapters to leak
into another model. Configured policy paths remain project-root-relative; no registry was edited.

## Caller compatibility decisions

- State probe capture stores selection metadata in checkpoint context, checkpoint shards, the
  combined NPZ metadata, and the result JSON's existing `meta` mapping. A reused modern artifact
  preserves recorded selection metadata when `--layers` is omitted. A legacy artifact preserves
  saved indices with `source: legacy-artifact` and null requested/fraction/depth values. Fractions
  against a legacy artifact fail closed because its depth is unavailable.
- Axis changes apply only to `build --layers`. `project --layer` remains the explicit single layer
  selected from the axis file. Both subcommands resolve their policy against the selected spec.
- J-lens uses `pipeline.evaluate.load_policy(spec.hf_id, adapter)` rather than a direct MLX loader.
  New `--policy` is registry-backed; `--adapter PATH` remains a deprecated explicit-directory
  alias, and simultaneous use is rejected.
- Patch deletes its private duplicated layer parser/validator, resolves policy with the selected
  spec, passes tuple integer indices to `run_patch_probe`, and adds selection metadata afterward.
- The historical state-probe lifecycle test now uses valid one-based layer 1 instead of layer 0.

## TDD RED/GREEN evidence

Pure seam RED, before `policies.py` implementation:

```text
uv run pytest -q tests/test_policies.py
```

Exit 2 during collection: `ImportError: cannot import name 'policy_names'`; expected because the
registry-aware API did not exist. After the minimal pure implementation, the same command exited
0 with 24 passed.

Caller/repository-rule RED, before caller production edits:

```text
uv run pytest -q tests/test_state_probe.py tests/test_probes_axis.py tests/test_jlens.py tests/test_patch.py tests/test_repository_rules.py
```

Exit 1: 16 failed / 53 passed. Failures named the absent architecture-depth seams, removed global
constants still imported by callers, spec-free resolver calls, hard-coded defaults, missing
metadata, malformed syntax reaching the GPU guard, direct J-lens loading, depth overflow reaching
`probe_layers`, missing policy/adapter conflict behavior, duplicated patch parsing, and the AST
rule finding the three literal defaults.

Caller GREEN: the same command exited 0 with 69 passed. The combined pure/caller command exited 0
with 93 passed. Adjacent `tests/test_probes.py tests/test_adapter_delta.py` initially exposed one
historical layer-0 fixture; after adapting it to the binding one-based contract, it exited 0 with
119 passed.

## Requirement-to-test map and mutation review

| Required break | Test evidence |
| --- | --- |
| Global policies leak into qwen35 | `test_registry_policy_resolution_is_per_model_and_preserves_explicit_directories` |
| 32-layer mapping changes to a 36-layer literal | `test_qwen35_default_layers_resolve_without_dtype_or_model_runtime` plus state/axis/J-lens caller tests |
| `1.0` is treated as index 1 | `test_cli_layer_tokens_preserve_lexical_kind_and_first_resolved_occurrence` |
| Collision order changes | the `0.5,16,0.500` and `0.01,01` literal cases in the same test |
| Invalid syntax/depth reaches runtime | policy invalid tables and the state/axis/J-lens/patch validation tests |
| Resolver ignores selected spec | state, axis project/build, J-lens, and patch fake resolver assertions |
| A caller omits selection metadata | state, axis, J-lens, and patch JSON/diagnostics assertions |
| A literal numeric parser default returns | `test_probe_layer_defaults_come_from_runtime_registry_metadata` |

The mutation review confirms each named test fails for the corresponding wrong depth, lexical
branch, spec argument, omitted metadata write, validation reordering, or AST default mutation.

## Static and adjacent verification

These commands all exited 0:

```text
uv run pytest -q tests/test_policies.py tests/test_state_probe.py tests/test_probes_axis.py tests/test_jlens.py tests/test_patch.py tests/test_repository_rules.py
uv run pytest -q tests/test_probes.py tests/test_adapter_delta.py
uv run ruff check src/local_llm_lab/probes/policies.py src/local_llm_lab/probes/state_probe.py src/local_llm_lab/probes/assistant_axis.py src/local_llm_lab/probes/patch.py src/local_llm_lab/pipeline/jlens.py tests/test_policies.py tests/test_state_probe.py tests/test_probes_axis.py tests/test_patch.py tests/test_jlens.py tests/test_repository_rules.py tests/test_probes.py
uv run ruff check --select C901 src/local_llm_lab/probes/policies.py src/local_llm_lab/probes/state_probe.py src/local_llm_lab/probes/assistant_axis.py src/local_llm_lab/probes/patch.py src/local_llm_lab/pipeline/jlens.py
git diff --check
```

Ruff printed `All checks passed!`; diff check was silent. Read-only baseline C901 checks showed
that `build_probe_dataset`, `_analyse_cohort`, state `main`, and J-lens `main` already exceeded
the threshold at the task base. Narrow function-level C901 annotations document that inherited
orchestration debt; the new `_reuse_layer_selection` helper was reduced below the threshold.

## Files and scope protection

Implementation paths are the five source modules, six existing test modules, new
`tests/test_policies.py`, and this report. The legacy resolver test was removed subtractively from
`tests/test_probes.py`; the approved plan was not edited.

Initial status already contained foreign changes in pending specifications, `arch.py`,
`preflight.py`, their tests, the heartbeat report, and `data/`. This task neither staged nor edited
those paths. It did not modify `outputs/`, `data/`, `reports/`, `research/`, pending documents,
`arch.py`, `preflight.py`, either Qwen3.5 registry file, or create `pipeline/policies.py`. Exact
commit path inspection is the final protected-path proof.

The earlier real preflight found layer 35 invalid for the 32-layer model. This task fixes the
source of that default but did not rerun preflight or any model-backed operation.

## R13 committed-HEAD gate

Pending the exact post-implementation-commit `uv run pytest -q` run.
