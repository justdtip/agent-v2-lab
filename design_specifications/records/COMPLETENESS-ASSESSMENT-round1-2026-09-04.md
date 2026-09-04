# Completeness assessment, round 1, ratified (2026-09-04 13:30)

Reviewer: Claude (Chief). Adopts the Deputy's `COMPLETENESS-ASSESSMENT-2026-09-04.md` as the
evidence record. Headline accepted: roughly 85% code-complete, near zero experiments.

## Decision requested (§0): conditions under which the Chief recommends the Director lift the ban for one training run

These are technical readiness conditions. Lifting the ban is the Director's act, per run, in
an issue that names the arm, the config, and the cost.

| # | Condition (ruling R15) | Today |
| --- | --- | --- |
| 1 | `outputs/preflight/<model>.json` exists with `passed: true` under the R18 tolerance. Training does not depend on residual equivalence, so `--waive-residual-equivalence` is permitted for a **training-only** run and is recorded in the artifact and provenance; it is never permitted for selection, evaluation, or probes | met for `qwen35-4b` (retry passed); 3B control needs the R18 tolerance applied |
| 2 | Training rows rendered in-repo (`RenderedRowsDataset`), rendering-equivalence test green for the model's thinking mode | met (R14) |
| 3 | The dataset exists on disk with manifest hashes, `GENERATOR_VERSION`, and `provenance.json`; for run D, integrity invariants pass on the generated splits | not met for D; met for B (`data/agent_v2b`, hash-verified, `messages` rows renderable for any model) |
| 4 | `evaluate.load_policy` migrated to the registry signature and `resolve_policy` reads `ModelSpec.policies`, so the resulting adapter can be evaluated with its own cache strategy and rendering. A run whose adapter cannot be evaluated is wasted GPU | not met; lane in flight |
| 5 | Selection screen config present; SPEC-003 §5 criteria copied into the run config as a `criteria:` block (recorded, not gated) | screen met; criteria block owed |
| 6 | Execution claim free; cost estimate stated in the request issue and accepted by the Director | procedural |
| 7 | Which arm first: **B4** (Qwen3.5-4B on run B's data, thinking off, `all-linear` keys). One variable changed from the reference run, and it exercises every new-model code path (rendering, `none` cache, evaluation) before the D recipe adds a second variable. D data is generated in parallel (no GPU) so D3 and D4 follow immediately | ruled |

## Rulings

- **R15** the gate above.
- **R16 report accuracy.** Every deliverable claimed in an implementation report names the
  file and line that satisfies it; the independent reviewer spot-checks at least three. A
  claim without a location is treated as not done.
- **R17 scanner.** The banned-constant check adds a substring pass over string constants
  (split on non-alphanumerics) alongside the AST pass. Correction: `--layers` in every probe
  CLI accepts fractions and defaults to `spec.probes.layer_fractions`; the literal
  `"6,12,18,24,30,35"` default is removed.
- **R18 precision.** (a) Residual-equivalence tolerance: Frobenius relative error
  `||h_view − h_native|| / ||h_native|| ≤ 1e-2` over the 64-token probe at the final
  residual, with max elementwise absolute and relative errors reported; exact equality is
  wrong for a bfloat16 model. (b) `ModelSpec.probes.capture_dtype: native | float32`,
  default `native`: block execution for capture runs in the model's dtype and the captured
  residual is upcast at the capture point; the float32 block path is used only for the J-lens
  tail and JVP, where the deviation from native is recorded in the artifact. Rule 1.5 is
  amended accordingly (float32 for stored activations, tangents, and probe features; native
  dtype for block execution during capture). (c) Offline check, SPEC-004 §1 correction **C7**:
  round the saved P2 npz activations to bfloat16 precision and back, refit through
  `reanalyse`, and compare margins and Holm flags with the corrected readings table; report as
  `state-base-mix.reanalysis-bf16.{json,md}`. If no supported flag flips, the precision gap is
  recorded as immaterial with evidence; if any flips, the base P2 run is re-captured with
  `capture_dtype: native` once probes are permitted. (d) The decision memo gains a stated
  caveat: existing P2 and J-space numbers were captured through a float32 block path on
  bfloat16 models. The Deputy's #15 comments are adopted as the analysis of record.

## Decisions on the claimable list (§4)

Priority: (1) `load_policy` / `resolve_policy` / `branch.build_prompt` migration, in flight;
(2) R17 scanner and layer fractions; (3) run D data generation (`data/agent_v2d`, no model);
(4) C7 bfloat16 rounding check; (5) SPEC-004 §2 P2 redesign; (6) SPEC-004 §4's four P1
preconditions. `stage_select` provenance and the `criteria:` block ride with (1).

## Governance note

Real model loads occurred on 2026-09-04 (one preflight, two diagnostic controls, one retry)
under Codex's exclusive `model-execution` action. The Chief treats these as Director-authorised
preflight-only executions; if they were not, the Director should say so and the execution
action is withdrawn. Nothing else loaded a checkpoint.
