# SPEC-001 S8 Probe Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace model-specific probe layer defaults and the global adapter-policy table with registry-backed, depth-validated selections whose outputs record both requested fractions and resolved indices.

**Architecture:** `local_llm_lab.probes.policies` remains the single policy-wiring owner and gains a small pure layer-selection value object plus syntax/depth resolution helpers. Probe and J-lens CLIs validate layer syntax before any model-loading boundary, resolve indices only after actual decoder depth is available, pass the selected registry spec into policy resolution, and persist reproducible layer-selection metadata while preserving existing integer layer lists for consumers.

**Tech Stack:** Python 3.13, frozen dataclasses, argparse, pathlib, pytest, Ruff; numpy/MLX-free policy and layer-selection fakes.

**Spec:** `design_specifications/pending/SPEC-001-model-agnostic-backbone.md` §8, constrained by `design_specifications/pending/01-IMPLEMENTER-BRIEFING.md` and `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` §2.15/§4/R10/R13.

## Global Constraints

- No checkpoint, tokenizer, MLX model, probe, J-lens, inference, training, evaluation, preflight, or agent CLI execution. Tests use pure or existing fakes only.
- Do not modify `outputs/`, `data/`, `reports/`, `research/*.md`, pending design documents, `src/local_llm_lab/arch.py`, or `src/local_llm_lab/pipeline/preflight.py`.
- Do not create `src/local_llm_lab/pipeline/policies.py`; the existing `src/local_llm_lab/probes/policies.py` owns this wiring.
- Do not add policy paths to either Qwen3.5 registry file while its `policies` mapping is empty.
- Preserve the `base` policy as `None`, named qwen25 policy paths relative to the project root, and explicit existing adapter directories as resolved absolute paths.
- `--layers` accepts comma-separated integer indices or decimal fractions in `(0, 1]`; lexical `1` is index 1 and lexical `1.0` is the full-depth fraction. Defaults come only from `ModelSpec.probe_layer_fractions`.
- Fractions resolve with `max(1, round(fraction * num_layers))`; every resolved index must be within `[1, num_layers]`; preserve first occurrence order while deduplicating collisions.
- Every newly resolved output retains its existing integer `layers` representation and also records a JSON-safe layer-selection mapping containing source, requested tokens, fractions, indices, and actual `num_layers`. Legacy reused captures that predate this metadata retain their saved indices and explicitly record unavailable depth/fraction fields instead of inventing them.
- Validate malformed layer syntax before GPU guards or model loaders. Validate upper bounds immediately after actual depth is available and before probe/J-lens execution.
- Follow R10: new policy coverage lives in `tests/test_policies.py`; caller coverage stays in the corresponding existing module test file. Move the legacy policy resolver test out of `tests/test_probes.py` subtractively.
- Follow R11: these metadata-sized JSON writes may remain direct.
- Follow R13: hand off only with exact `uv run pytest -q` evidence at a named HEAD, including exit code, pass/fail/xfail counts, and failing node IDs.
- Use one principal implementer and one independent principal reviewer. Neither may dispatch helpers or additional reviewers. In this one-task plan, the task review is also the whole-change final review; the same reviewer performs any scoped re-review.
- Work in the existing shared checkout and branch. Preserve foreign dirty/staged/untracked work; stage and commit only the explicit files in this plan; do not push, merge, rebase, stash, reset, restore, clean, or switch branches/worktrees.

---

### Task 1: Registry-backed policies and reproducible layer selections

**Files:**
- Modify: `src/local_llm_lab/probes/policies.py`
- Modify: `src/local_llm_lab/probes/state_probe.py`
- Modify: `src/local_llm_lab/probes/assistant_axis.py`
- Modify: `src/local_llm_lab/probes/patch.py`
- Modify: `src/local_llm_lab/pipeline/jlens.py`
- Create: `tests/test_policies.py`
- Modify: `tests/test_state_probe.py`
- Modify: `tests/test_probes_axis.py`
- Modify: `tests/test_patch.py`
- Modify: `tests/test_jlens.py`
- Modify: `tests/test_probes.py`
- Modify: `tests/test_repository_rules.py`
- Create: `design_specifications/under_review/SPEC-001-S8-PROBE-DEFAULTS-IMPLEMENTATION-REPORT.md`
- Keep: `docs/superpowers/plans/2026-09-04-spec-001-s8-probe-defaults.md`
- Keep: `.superpowers/sdd/2026-09-04-spec-001-s8-probe-defaults/`

**Interfaces:**
- Consumes: `ModelSpec.policies`, `ModelSpec.probe_layer_fractions`, `ArchitectureView.num_layers`, and the existing `load_policy(model_name, adapter_path)` model-loading seam.
- Produces: `policy_names(spec) -> tuple[str, ...]`, `resolve_policy(name, spec) -> Path | None`, `validate_layer_syntax(raw) -> None`, `resolve_layers(raw, spec, num_layers) -> LayerSelection`, and `LayerSelection.as_dict() -> dict[str, object]`.
- `LayerSelection.as_dict()` returns exactly the JSON-safe keys `source`, `requested`, `fractions`, `indices`, and `num_layers`. `source` is `"registry-default"` when `raw is None` and `"cli"` otherwise.
- Existing downstream APIs continue receiving `list[int]` or `tuple[int, ...]` layer indices; metadata is additive.

- [ ] **Step 1: Add failing pure policy and 32-layer selection tests**

Create `tests/test_policies.py`. First move the existing `test_resolve_policy_maps_names_and_rejects_unknown` behavior out of `tests/test_probes.py`, then strengthen it with literal expectations:

```python
def test_registry_policy_resolution_is_per_model_and_preserves_explicit_directories(tmp_path):
    qwen25 = load_model_spec("qwen25-coder-3b")
    qwen35 = load_model_spec("qwen35-4b")
    assert policy_names(qwen25) == ("base", "A", "B", "C")
    assert policy_names(qwen35) == ("base",)
    assert resolve_policy("base", qwen35) is None
    assert resolve_policy("B", qwen25).as_posix().endswith("outputs/agent-v2b/best-adapter")
    assert resolve_policy(str(tmp_path), qwen35) == tmp_path.resolve()
    with pytest.raises(ValueError, match="unknown policy"):
        resolve_policy("B", qwen35)


def test_qwen35_default_layers_resolve_without_dtype_or_model_runtime():
    selection = resolve_layers(None, load_model_spec("qwen35-4b"), 32)
    assert selection.indices == (5, 11, 16, 21, 27, 32)
    assert selection.fractions == (0.167, 0.333, 0.5, 0.667, 0.833, 1.0)
    assert selection.as_dict() == {
        "source": "registry-default",
        "requested": ["0.167", "0.333", "0.5", "0.667", "0.833", "1.0"],
        "fractions": [0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
        "indices": [5, 11, 16, 21, 27, 32],
        "num_layers": 32,
    }
```

Add table-driven cases proving lexical `1`/`1.0`, mixed indices/fractions, first-occurrence deduplication, malformed tokens (`""`, commas with empty cells, zero, negatives, non-finite values, fractional values above one), invalid `num_layers`, and out-of-range explicit indices. Expected indices and fractions must be hand-derived literals rather than computed by the production helper.

- [ ] **Step 2: Run the new policy tests to verify RED**

Run: `uv run pytest -q tests/test_policies.py`

Expected: FAIL because the registry-aware policy/layer-selection interfaces do not yet exist. Record the exact command, exit code, relevant failure, and why it is expected in the implementation report.

- [ ] **Step 3: Implement the minimal shared policy and layer-selection seam**

In `src/local_llm_lab/probes/policies.py`, remove the global `POLICY_ADAPTERS` table. Add frozen `LayerSelection` and the four functions named in Interfaces. Parse integer-looking tokens as indices and decimal/exponent tokens as fractions; require finite values. For explicit integer selections, record `index / num_layers` as the corresponding fraction; for fractional selections, record the requested fraction. When multiple tokens resolve to the same index, retain the first token/fraction only.

Policy resolution must follow this precedence: exact `"base"` -> `None`; a key in `spec.policies` -> project-root-relative configured path; an existing directory supplied explicitly -> absolute resolved path; otherwise raise `ValueError` naming `policy_names(spec)`. Do not inspect or manufacture paths outside `ModelSpec.policies`.

- [ ] **Step 4: Run the policy tests to verify GREEN**

Run: `uv run pytest -q tests/test_policies.py`

Expected: PASS with pristine output.

- [ ] **Step 5: Add failing caller and repository-rule tests**

In each existing per-module test file, add a fake-only test that names the production break it catches:

- `tests/test_state_probe.py`: a qwen35-like fake depth of 32 makes an omitted `--layers` dispatch `[5, 11, 16, 21, 27, 32]`, passes the selected spec to `resolve_policy`, and stores the exact layer-selection mapping in capture/result metadata before any probe seam is invoked.
- `tests/test_probes_axis.py`: `_build` resolves the same default indices, uses the selected spec for named/explicit policy resolution, and writes the mapping into diagnostics; malformed syntax must stop before the GPU guard/loader.
- `tests/test_jlens.py`: the J-lens CLI defaults at depth 32, accepts a mixed explicit request such as `"1,0.5,1.0"`, validates depth before `probe_layers`, and records both fractions and indices in JSON. If `--policy` is introduced to satisfy the common CLI contract, keep `--adapter` as a deprecated explicit-directory alias and reject using both.
- `tests/test_patch.py`: replace the private duplicated parser expectations with the shared selection contract, verify `resolve_policy` receives the selected spec, and verify patch JSON gains the exact selection mapping without changing `run_patch_probe`'s integer `layers` contract.
- `tests/test_repository_rules.py`: extend the AST scanner so a probe/J-lens `add_argument("--layers", default="2,4,6")` or `add_argument("--readout-layers", default="2,4,6")` literal numeric list is reported, while `default=None`, unrelated numeric strings, and runtime-computed defaults are not.

Run the focused caller tests before production changes:

`uv run pytest -q tests/test_state_probe.py tests/test_probes_axis.py tests/test_jlens.py tests/test_patch.py tests/test_repository_rules.py`

Expected: FAIL specifically on the current global resolver signature, hard-coded state/axis/J-lens defaults, missing metadata, duplicated patch parsing, and missing AST rule.

- [ ] **Step 6: Migrate every affected caller minimally**

Apply the following flow in each CLI: load the registry `ModelSpec` from `--model`; call `validate_layer_syntax` before any GPU guard/model loader; resolve the adapter with `resolve_policy(args.policy, spec)` where that CLI has a policy; after loading the fake/real model boundary, obtain actual depth from `ArchitectureView.from_model(model).num_layers`; call `resolve_layers`; pass `list(selection.indices)` to existing computation; attach `selection.as_dict()` to output metadata.

Specific compatibility requirements:

- `state_probe.py`: change the `--layers` default to `None`. New captures store `dataset.meta["layer_selection"]`; result JSON also exposes that mapping through its existing metadata. A reused artifact with existing `layer_selection` reuses it when no explicit layers are requested. A legacy reused artifact retains its saved integer layers and records `source: "legacy-artifact"`, those indices, and `null` for unavailable depth/fraction values rather than guessing them; explicit fractional requests against such an artifact fail closed.
- `assistant_axis.py`: change only the build subcommand's `--layers` default; the project subcommand's explicit single `--layer` continues to select a layer present in its axis file. Add `layer_selection` to build diagnostics. Both build and project resolve policies from their selected model spec.
- `pipeline/jlens.py`: change `--layers` to default `None`; resolve and validate after `ArchitectureView` exposes depth; add `layer_selection` beside the existing integer `layers` field. Use registry-backed `--policy` through `pipeline.evaluate.load_policy`; preserve `--adapter PATH` as an explicit-directory compatibility alias and reject simultaneous `--policy`/`--adapter`.
- `patch.py`: import the shared helpers, delete `_parse_layers` and `_validate_layer_syntax`, pass `spec` to `resolve_policy`, and add `layer_selection` to the final payload after `run_patch_probe` returns.
- `state_probe.py` and `assistant_axis.py`: remove imports/help text tied to a global `POLICY_NAMES`; help text says policy names come from the selected model or an explicit directory.

- [ ] **Step 7: Verify caller GREEN and run mutation checks**

Run:

`uv run pytest -q tests/test_policies.py tests/test_state_probe.py tests/test_probes_axis.py tests/test_jlens.py tests/test_patch.py tests/test_repository_rules.py`

Then mentally or temporarily mutate each of these and confirm a named test would fail: 32 -> 36 depth mapping, `1.0` treated as index 1, policy lookup ignores the selected spec, a caller omits metadata, validation occurs after the loader, or a CLI literal numeric list is restored. Record the test names covering those breaks.

- [ ] **Step 8: Run adjacent fake-only tests and static checks**

Run:

```bash
uv run pytest -q tests/test_probes.py tests/test_adapter_delta.py
uv run ruff check src/local_llm_lab/probes/policies.py src/local_llm_lab/probes/state_probe.py src/local_llm_lab/probes/assistant_axis.py src/local_llm_lab/probes/patch.py src/local_llm_lab/pipeline/jlens.py tests/test_policies.py tests/test_state_probe.py tests/test_probes_axis.py tests/test_patch.py tests/test_jlens.py tests/test_repository_rules.py tests/test_probes.py
uv run ruff check --select C901 src/local_llm_lab/probes/policies.py src/local_llm_lab/probes/state_probe.py src/local_llm_lab/probes/assistant_axis.py src/local_llm_lab/probes/patch.py src/local_llm_lab/pipeline/jlens.py
git diff --check
```

Expected: all commands exit 0; no model/checkpoint/tokenizer/probe/J-lens execution occurs.

- [ ] **Step 9: Write the implementation report and commit exact paths**

Create `design_specifications/under_review/SPEC-001-S8-PROBE-DEFAULTS-IMPLEMENTATION-REPORT.md` with: observable behavior, files changed, policy/layer API, explicit compatibility decisions, TDD RED/GREEN evidence, fake-only assertion, protected-path assertion, repository-rule evidence, and a requirement-to-test table. Record that the real preflight found layer 35 invalid for the 32-layer model, but do not rerun it.

Before staging, run `git status --short` and `git diff --cached --name-only`. Stage only the files listed in this Task. Inspect `git diff --cached --stat` and `git diff --cached --check`, then commit with:

`git commit -m "Implement model-aware probe defaults" -- <each exact staged path>`

Do not include foreign staged/untracked work.

- [ ] **Step 10: Run the exact R13 full-suite gate at the committed HEAD**

Run exactly: `uv run pytest -q`

Record in the report: full `git rev-parse HEAD`, exact command, exit code, passed/failed/xfailed counts, and every failing node ID (or `none`). If red, diagnose whether failures belong to this task; do not request review or hand off until the shared HEAD is green. Commit the evidence-only report amendment as an exact-path commit.

- [ ] **Step 11: Self-review and report**

Inspect the complete task range and confirm: no claimed file was omitted; no Qwen3.5 policy path was invented; no model-specific layer literal remains in the target CLIs; no protected or pending-document path appears; every resolver caller passes the selected spec; output JSON preserves integer layers and adds reproducible selection metadata. Write the full implementer report to the SDD report path and return the short SDD status contract.
