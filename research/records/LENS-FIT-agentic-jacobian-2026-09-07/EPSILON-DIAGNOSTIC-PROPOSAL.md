# Proposed diagnostic01 — epsilon stability and cache-path agreement

Status: **awaiting the Director's ruling; not implemented or run**. 7 September2026. Responds to CHECK-01-INCONCLUSIVE.md and the immutable check-01/self-check.json. Original fit plan and acceptance outcomes remain unchanged.

## Exact proposed experiment

Use the same checkpoint snapshot, frozen corpus row0,439-token prompt, position438, layers1/16/31,16 random unit directions from seed20260904 and batch8 as check-01. No new prompt selection. Per layer, compute full uncached reference responses at four preselected multipliers of the original epsilon:1/8,1/4,1/2,1. Compute both restore and broadcast responses at the two preselected endpoints1/8 and1.

Compare each adjacent reference pair as D(s) against D(s/2), retaining D(s) as the tolerance denominator, exactly as the original check. Keep stability atol0.003/rtol0.03. Compare each cached endpoint to the same-scale uncached reference independently, with self atol0.0003/rtol0.003, even when the reference is unstable. That independent descriptive comparison must not bypass require_self_check or create a passing production proof.

For every scale/direction and comparison retain actual epsilon, response norms, maximum and RMS error, failed-coordinate count, total-coordinate count and the original threshold. Record nonfinite values as explicit failures, not JSON NaN. Include source/snapshot/plan hashes, input IDs and direction identity, exact forward-token accounting, per-layer/scale progress, elapsed time and measured MLX peak. Stream and materialise each result; no retained device graph over the scale grid or activation dataset.

## What the result could resolve

If decreasing epsilon improves adjacent-reference stability, finite-step curvature is a candidate explanation. If it worsens, numerical cancellation or rounding is a candidate. Endpoint cache/reference errors show whether the two paths introduce an additional discrepancy. These are diagnostic patterns, not proofs of cause. A passing pair alone cannot establish the correct derivative; this small ladder does not locate a stable region above the original epsilon or below1/8, and does not prove every standard-basis direction is stable.

No automated epsilon selection, refit, benchmark, map artifact, production self-check pass or scientific profile follows. Production still uses the originally prescribed epsilon rule. Any replacement step rule requires a separate explicit method amendment and fresh preregistration/self-checks. The three failures and six inconclusive outcomes in check-01 stay on record.

## Cost, resources and slot

Let R be one full-reference measurement and C both cache modes. The original derivative work was2R+C; this proposal is4R+2C, approximately twice that work, before load/preparation/recording differences. Check-01 was observed finished about173seconds after launch, so roughly6minutes is a planning estimate, not a measured runtime guarantee. Declare an8-minute execution budget with orderly stopping between calls; retain any partial evidence as incomplete.

Retain the inspected8GiB envelope and0.6working-set cap. Shapes and batch width are unchanged; no new larger-cache claim. Per-workload checks and kernel-critical pressure supervision remain. Announce a fresh exclusive slot only after the ruling and queue handoff, run through the normal primary lock, and append its end. Issue88 owns the next available block following Codex's overall end line; no slot is held for this proposal.

## Independent pre-proposal review

GPT-6 Astra read the saved result and confirmed its interpretation: three stability failures make all six cache checks inconclusive, with no recorded cached/reference error. It recommended this four-scale/two-endpoint experiment as the first diagnostic. An initially considered seven-scale/every-mode grid would cost between3.5 and7times the original derivative work, not necessarily3.5times; the proposed12-minute prediction for that broader grid was unsupported and is not adopted. No diagnostic implementation, tests or model calls were made in preparing this proposal.

**Requested ruling:** approve this bounded diagnostic-only experiment while preserving the original production epsilon rule and tolerances.
