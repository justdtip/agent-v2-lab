# R18a gate report: preflight residual equivalence gates on the native-dtype comparison

Implementer slice: rulings R13, R16, R18/R18a, R19 (wiring map §7). Files owned and changed:
`src/local_llm_lab/pipeline/preflight.py` and `tests/test_preflight.py`. No other file touched;
nothing committed (working tree only, awaiting the Chief's gate). No model was loaded or run;
every test runs on the numpy fakes. Revised after the independent R19 review: the reviewer's
AND-condition is now implemented (see §1 and §5.1).

## 0. Standing hypothesis for the authorised rerun — read this before interpreting a failure

This gate encodes a hypothesis that no fake can test: **that mlx's fused native forward and
the view's stepwise block loop are bitwise-identical (or within one rounding floor AND
Frobenius relative ≤ 1e-4) on the real bfloat16 hybrid.** If they are not — if kernel fusion
or scheduling introduces distributed reassociation error above 1e-4 Frobenius-relative — this
gate will fail a **healthy** model at the authorised rerun. That failure mode is deliberate:
it is loud, fail-closed, and the artifact preserves both metric blocks
(`max_abs_error`, `frobenius_relative_error`, and the derived tolerances) so the Chief can
decide with data whether the bound, the loop, or the model is at fault. A rerun failure is
therefore the hypothesis test running to completion, not a surprise; do not loosen the gate
in anticipation of it.

## 1. What gates now

- `run_preflight`'s `residual_equivalence` gates on **`native_manual_vs_native`**: the view's
  manual block loop run in the model's native dtype (`view.diagnostic_native_final_residual`,
  `arch.py:146-159`) against the model's own native forward (`_native_final_residual`,
  `preflight.py:388-394`).
- The gate is a **conjunction of two conditions**, `_native_gate_passed`
  (`preflight.py:294-306`), applied to the native block at `preflight.py:334`:
  1. the auditable derived elementwise floor from `_residual_metrics`
     (`preflight.py:347-387`): `relative_tolerance = sqrt(2*num_layers+1) * eps(reference
     dtype)` — and because both sides now share the native dtype, that epsilon is the shared
     dtype's;
  2. `frobenius_relative_error <= _NATIVE_FROBENIUS_TOLERANCE` (`1e-4`, `preflight.py:291`),
     the ruling's Frobenius-relative bound.
  The reviewer's dominance analysis motivates the AND: at a 64x2560 bfloat16 geometry the
  derived max-abs floor alone passes distributed relative error up to ~6.3e-2 — the signature
  of a wrong norm weight, mis-scaled residual, or mask defect — while the 1e-4 bound fails it;
  failing the floor implies failing the bound, so the conjunction strictly tightens and loses
  nothing.
- Gate wiring downstream is unchanged in shape: `report["passed"]` at `preflight.py:147-149`
  and `_view_evidence_passed` (`preflight.py:518-527`) still read `residual_equivalence.passed`.
- Criterion string in the artifact: `"native_dtype_rms_roundoff_and_frobenius_relative"`
  (`preflight.py:314`), with the bound itself recorded as `frobenius_relative_tolerance`
  (`preflight.py:323`) so the artifact states its own criterion numerically.

## 2. What is reported, never gated

- `fp32_manual_vs_native` (the float32 manual loop of `_manual_final_residual`,
  `preflight.py:339-345`, against the same native forward) is kept in the artifact with its
  full `_residual_metrics` block plus `"gates": false` and a purpose string labelling it as
  the precision-gap measurement for the capture-dtype decision (`preflight.py:315-322`).
  It feeds the R18(c) refit question and can fail its own tolerance without affecting
  `passed` — proven by `test_fp32_precision_gap_is_reported_but_never_gates`
  (`tests/test_preflight.py:654-681`): the `_MismatchingView` fake (native paths agree,
  float32 path off by 0.5) passes preflight while the fp32 block records
  `within_tolerance: false`.
- The failure test fails for the right reason: `_NativeDivergentView`
  (`tests/test_preflight.py:121-125`) diverges only in the native manual loop, and
  `test_failed_preflight_writes_evidence_then_exits_nonzero`
  (`tests/test_preflight.py:620-651`) asserts the native block fails, the fp32 block is
  clean, and the evidence artifact survives the nonzero exit.
- The distributed-defect catch is demonstrated at unit level by
  `test_distributed_error_passes_the_derived_floor_but_fails_the_frobenius_gate`
  (`tests/test_preflight.py:457-488`, the reviewer's cited example reworked and renamed):
  a 0.2% distributed error on a bfloat16 reference passes the derived floor
  (`within_tolerance: true`) at `frobenius_relative_error ≈ 0.002` — twenty times over the
  ruling's threshold — and `_native_gate_passed` correctly fails it, while an exact-zero
  error passes both conditions.

## 3. Schema change (version 1 → 2)

- `_SCHEMA_VERSION = 2` (`preflight.py:24`), written into both artifacts and enforced by
  `require_preflight` (`preflight.py:230`), so every stale v1 artifact under
  `outputs/preflight/` is rejected as unsupported until the gated real rerun regenerates it.
- New `residual_equivalence` shape (asserted whole-dict in
  `test_run_preflight_writes_stable_complete_fake_report`, `tests/test_preflight.py:328-363`):
  `criterion`, `frobenius_relative_tolerance`, `gated_comparison: "native_manual_vs_native"`,
  the two metric blocks `fp32_manual_vs_native` / `native_manual_vs_native` each carrying
  `gates` and `purpose`, `passed`, `token_count`. A reader of `outputs/preflight/<model>.json`
  sees the full criterion, its numeric bound, and which block gated without reading source.
- `_residual_metrics` gains `frobenius_relative_error` (`preflight.py:365-376`), the measure
  R18a names ("Frobenius relative plus elementwise maxima"); norms are accumulated in float32
  so a low-precision reference cannot corrupt its own error measure.
- `run_residual_control` (`preflight.py:156-204`) is unchanged in shape beyond this: its two
  top-level blocks now come from the shared `_residual_comparisons` helper
  (`preflight.py:270-288`, called at `preflight.py:185-190`), each gaining only the
  `frobenius_relative_error` field, and its `schema_version` moves with the shared constant.
  The key-set guard in `test_run_residual_control_writes_only_the_three_residual_comparisons`
  (`tests/test_preflight.py:560-574`) pins the control blocks to exactly the ten metric keys —
  no preflight-only `gates`/`purpose` labels leak in. The recording-view ordering test
  (`tests/test_preflight.py:576-617`) passes untouched, so the standing per-base control
  works as before.
- `cli.py` needed no change and was not touched: `cli.py:742` calls `run_preflight(args.model)`
  and `cli.py:52` calls `require_preflight`; neither reads the artifact JSON directly, and
  `test_cli.py:21/61/91` monkeypatch both entry points.

## 4. New production-configuration coverage

`test_run_preflight_defaults_use_the_loader_view_and_resolved_spec`
(`tests/test_preflight.py:684-727`) drives `run_preflight` with `view_factory` and `resolver`
omitted — the previously unexercised default branches at `preflight.py:91-94` and exactly the
configuration of the sole production caller (`cli.py:742`). It proves, on fakes only:

- the loader's own view instance does the residual work (`_RecordingPreflightView`,
  `tests/test_preflight.py:128-148`, records `embed`, `run_block:0..3`, `final_norm`,
  `native_manual`), with `ArchitectureView.from_model` monkeypatched to fail if any second
  view were built;
- the loader's resolved spec is used untranslated: `snapshot_revision` comes from
  `resolved.snapshot_revision` (`preflight.py:95`) with a `revision_reader` that fails the
  test if consulted, and `cache`/`lora` fields land from the resolved spec;
- the artifact passes and records `gated_comparison: "native_manual_vs_native"`.

## 5. Decisions taken under ambiguity (briefing rule 1.10)

1. **RESOLVED by R19 review: the gate is now derived-floor AND Frobenius ≤ 1e-4.** The first
   round gated on the derived tolerance alone per the task directive and flagged the deviation
   from R18a's literal bound; the independent reviewer's numeric analysis (derived floor ~630x
   more permissive against distributed deviations at this geometry, with strict dominance of
   the 1e-4 bound over the floor's failure set) settled it against the derived tolerance
   alone. Implemented as §1 describes; both metrics remain in the artifact unchanged.
2. **`token_count` is `int(ids.shape[1])`** (`preflight.py:335`) instead of the literal 64;
   same value, no hard-coded prompt length.
3. **Frobenius norms in float32.** Native-dtype accumulation of a sum of squares would round
   in the reference's own precision; the diagnostic is promoted before the norm
   (`preflight.py:363-373`).
4. **Control schema version moves with the shared constant.** Its metric blocks did change
   (the R18a-required Frobenius field), so a version that says otherwise would lie; nothing in
   the repository validates the control's `schema_version` (grep: only `preflight.py` and
   `tests/test_preflight.py` reference it).

## 6. Verification

- `uv run pytest`: **627 passed in 8.72s** (R13; full-suite green at hand-off, count current
  as of this revision — the tree includes parallel slices' tests). The preflight module alone:
  36 passed.
- Banned-constant grep over my diff (R17 substring pass included;
  `tests/test_repository_rules.py`: 10 passed): no `36`, `2048`, `35`, end-of-turn literal, or
  `model.model.layers` added. `1e-4` is R18a's ruled tolerance, held in one named constant
  (`preflight.py:291`), not a model constant. One hit for projection-name substrings: the new
  default-branches test asserts the fake fixture round-trip
  `["layers.0.q_proj", "layers.3.down_proj"]` (`tests/test_preflight.py:718-721`), the same
  literals the existing fixture and stable-report test already carry
  (`tests/test_preflight.py:100-107,365-368`) — fixture data, not a model constant.
- Signatures: `run_preflight`, `run_residual_control`, `require_preflight`, `write_report`,
  `artifact_path`, `cached_revision` all unchanged; no caller updates required.

## 7. Unverifiable without a real run (gated)

- The §0 hypothesis: whether the real hybrid's fused forward and the stepwise loop are
  bitwise-identical, and if not, whether their divergence sits under the 1e-4
  Frobenius-relative bound. The artifact will record `max_abs_error` and
  `frobenius_relative_error` either way.
- The actual float32 precision-gap magnitude per base (the capture_dtype input).
- Regeneration of `outputs/preflight/<model>.json` at schema version 2: every existing v1
  artifact is now rejected by `require_preflight` by design, so the preflight rerun must
  happen under its separate authorisation before any model-loading stage passes the guard.

## 8. Observed, not fixed

- None within scope. The parallel working-tree edits to `probes/patch.py`, `tests/test_patch.py`
  and the heartbeat log were present before this slice started and were left alone.
