# Diagnostic01 — approved registration before measurements

7 September 2026. The Director approved EPSILON-DIAGNOSTIC-PROPOSAL.md in this task ("Approved", then "yes"). That immutable proposal remains a historical proposal; this record establishes its approved status. No diagnostic measurement has been made when this file is written.

## Scope and relationship to the new ruling

Run exactly the proposal's four full-sequence epsilon multipliers and two cached endpoints. Primary commit2576ce9 subsequently records requirements§15, authorizing a different, per-position plateau sweep and allowing regression independently. This diagnostic measures the originally proposed scale ladder and cache discrepancy; it does not implement or substitute for the§15 sweep. Neither an original-scale passing comparison nor a smaller error selects a production step. The original check-01 and plan stay immutable. The new ruling's causal explanation is a hypothesis to test: the norm ratio need not equal the square root of sequence length when residual magnitudes vary across positions.

## Frozen measurement

Original plan hash9fa35bb395d8c424b982d817f966204dc39bdb8d31f28ed9b9dd716d33dbc558; source row0 train-read-0000-clean,439 IDs, position438, layers1/16/31,16 unit directions seeded20260904, batch8. Original full-primal epsilon rule unchanged. Uncached reference at multipliers1/8,1/4,1/2,1. Both restore and broadcast at1/8 and1. Actual epsilon depends on each tangent's float32 norm and is recorded per direction.

Adjacent reference pairs compare D(s) to D(s/2), with D(s) supplying the denominator: error <=0.003+0.03*abs(D(s)). Cached endpoints compare to the same-scale reference independently under0.0003+0.003*abs(reference), even where stability fails. All coordinates must pass for a descriptive threshold pass. Maximum and RMS error, failed/total and nonfinite counts, and per-direction response norms are retained. Nonfinite evidence fails explicitly and JSON never emits NaN or infinity. Raw CPU response files allow independent recomputation; no activation corpus or retained graph across scales.

## Exact work accounting

This experiment calls partial decoder tails, not token generation or the capture runner. A single observing view records actual run_block invocations with block index, batch width, sequence length, phase and scale. A block-token is one row position supplied to one decoder block; summing these is deliberately not called emitted tokens or full-model tokens. Lazy graph construction and materialization are distinguished. Existing native evaluation boundaries stay unchanged; bookkeeping must not add per-block evaluation or alter numerical scheduling. A completed workload confirms only calls actually covered by its evaluation; interrupted partial work remains explicitly unconfirmed. Hash-chained ledger rows are the source of totals, not a parallel estimate. Capture's existing ForwardLedger and production replay are unchanged.

## Budget and launch

The execution budget is480 seconds including CLI preflight and load, with checks between calls; an in-flight native call is not interruptible by this deadline. Retain partial evidence and mark incomplete if the budget expires. Approximate planning estimate remains6 minutes, not a guarantee. Inspected8GiB bound, actual0.6-working-set cap, allocator cache0, existing per-workload memory checks and measured MLX peak apply. No benchmark, fit, validation or profile follows automatically.

Primary issue95 has now landed (1f3dd37, subsequent test fixc31bc49). The operational launcher will use primary runlock.announce_window/end_window and export its nonce to the child, with source hashes/revisions recorded for both the primary launcher and worktree diagnostic. The child uses the existing normal primary model lock. Before announcement, the launching process reads the live heartbeat and checks the window, lock and R45(b) mapped-process inventory. A current foreign holder is not cleared or interrupted. R46 kernel-pressure supervision continues through the owned child's exit; an end line closes this single diagnostic window. No other MLX-reaching suite or checkpoint launch belongs in that window.
