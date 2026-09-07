# Diagnostic01 completed — cache paths agree; original derivative step is unstable

8 September 2026 local. Source5f7a202, numerical implementation0725738. The Director explicitly authorized clearing the stale issue88 window. Immediately before removal, the former owner PIDs20550/20551 were absent, the primary model lock was absent and the R45(b) mapped-MLX inventory was empty. Only the exact authorized nonce was removed. The reviewed wrapper acquired a fresh window, loaded under the normal primary model lock, supervised its own child and closed its window on exit. No foreign process was stopped.

## Result and interpretation

All12 cached/reference comparisons pass the original coordinatewise0.0003+0.003*abs(reference) bound: both restore and broadcast, at1/8 and1 of the original epsilon, at layers1/16/31. These are tolerance-based derivative comparisons, not bit-identical trajectory-equivalence gates. They argue against a restore-specific fault as the cause of this instrument failure within the tested prompt/directions.

Only2 of9 reference-stability comparisons pass: layer31 at1/2 versus1/4 and1/4 versus1/8. Layers1 and16 still fail at the smallest pair; layer16 has just one failed coordinate, which remains a failure under the preregistered all-coordinate rule. Each cell below is maximum absolute response difference, with failed coordinates in parentheses, out of40,960.

| Layer | D(1) vs D(1/2) | D(1/2) vs D(1/4) | D(1/4) vs D(1/8) |
| --- | --- | --- | --- |
| 1 | 0.601956844 (33,776) | 0.166157722 (18,039) | 0.0426683426 (1,198) |
| 16 | 0.110102773 (25,048) | 0.0281713009 (2,767) | 0.00723493099 (1) |
| 31 | 0.0168907195 (11) | 0.00470891595 (0) | 0.00119856 (0) |

Maximum and RMS discrepancies decrease at every layer as the step shrinks. Maximum discrepancies shrink roughly fourfold per halving, consistent with finite-step truncation/nonlinearity. That supports the step-size explanation; it does not prove the exact derivative or certify arbitrary directions. Cache/reference differences increase at the smaller endpoint, so numerical rounding/cancellation cannot be dismissed outside the sampled range. At the original1-versus1/2 pair, all three maximum errors exactly reproduce check-01.

The full-primal/selected-position norm ratios are23.0460,22.5137 and23.8759 for layers1/16/31. The original unit-direction perturbation therefore has about23% of the selected position's residual norm. These are measured ratios; a square-root-of-length approximation is not needed. No new production epsilon was selected, no production proof was created and no fit followed. Requirements§15's separate per-position plateau sweep remains necessary before a revised Jacobian self-check. Regression is separately permitted by§15 and was not run in this diagnostic window.

## Execution and evidence audit

The diagnostic completed in174.795 seconds including its CLI preflight/load; wrapper lifetime175.019 seconds. Measured MLX process peak4,156,089,896 bytes =3.87066GiB, below the inspected8GiB bound and the0.6-working-set cap. All seven kernel-pressure samples were normal. Exit0, owned child exited, window closed, primary lock absent; a fresh external R45(b) inventory after exit was empty.

All24 raw response arrays and21 comparisons are saved. The offline audit recomputed hashes, per-direction norms and all-coordinate comparison statistics; validated the ledger chain; and reconciled9,792 actual decoder-block calls and2,766,528 materialized block-tokens to the independent before-run expectation. No scheduled calls remain unconfirmed. A block-token is one batch-row position supplied to one decoder block, not an emitted token or whole-model pass. The original capture ForwardLedger is unchanged.

DIAGNOSTIC-01-AUDIT.json contains the audit and full comparison summaries; diagnostic-01/ contains the raw responses, directions, progress, provenance and hash-chained ledger. DIAGNOSTIC-01-AUDIT-NOTE.md records a corrected offline norm-reduction mismatch at1.8e-15; no model response or production bound was changed. DIAGNOSTIC-01-AUDIT-SOURCE.txt preserves the audit implementation.

Independent GPT-6 Astra interpretation review agrees: the tested cache paths pass,7/9 stability pairs fail, the scale trend supports finite-step truncation as an explanation without proving cause, and production acceptance remains false. The reviewer read the saved audit but did not independently recompute the raw hashes/chain; those were recomputed by the root task's separate offline audit.
