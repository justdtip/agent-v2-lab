# State programme record

Decoding mode: `greedy` — estimand: context-averaged outcome frequency over the task distribution (one outcome per context) *(manifest.json)*
Fixture run: `True` *(manifest.json)*

**Rate**: 31297358.09546 episodes/hour, measured over 60 episodes in mode `greedy` *(rate.json)*
**Pilot rows**: 60 *(pilot/rows.jsonl)*

## Pre-registration *(preregistration.json)*

Retained diagnostics: D1, D4, D5, D10 (M = 4)
Dropped: D2 — the state does not reach it: bound on |contrast| is 0.0000 ≤ 0 (lower 0.0000, upper 0.0000); D3 — the state does not reach it: bound on |contrast| is 0.0000 ≤ 0 (lower 0.0000, upper 0.0000); D6 — the state does not reach it: bound on |contrast| is 0.0000 ≤ 0 (lower 0.0000, upper 0.0000); D7 — not reached: one arm produced no scorable transcript; D8 — not reached: one arm produced no scorable transcript; D9 — not reached: one arm produced no scorable transcript

| tolerance | value | derivation |
|---|---:|---|
| `d_min` | 1.00000 | min over retained diagnostics of the (1 - alpha) bootstrap bound on |contrast| |
| `epsilon_sub` | 0.25000 | d_min / 4 |
| `epsilon_perp` | 0.16667 | 2 * epsilon_sub / 3 |
| `epsilon_reuse` | 0.25000 | epsilon_sub |
| `epsilon_pred` | 0.00000 | split-half log-loss gap of the pilot's fitted belief update |
| `epsilon_dyn` | 0.00000 | split-half log-loss gap of the pilot's fitted belief update |
| `n` | 82 | ceil(ln(2M/alpha) / epsilon_sub^2) |

Budget: within budget — within budget at the derived tolerance

**Main rows**: 60 *(main/rows.jsonl)*

## Estimands *(estimands.json)*

| estimand | distance | tolerance | passes |
|---|---:|---:|---|
| dynamic | — | 0.00000 | not measured (no scorable row) |
| predictive | 0.00000 | 0.00000 | untestable (degenerate tolerance) |
| relation | 0.00000 | 0.25000 | True |
| reuse | 0.00000 | 0.25000 | True |
| specificity | 0.00000 | 0.16667 | True |
| substitution | 0.00000 | 0.25000 | True |

Level: J-space vectors (diagnostic means); never a coefficient table

The pilot is not a result and its numbers are not findings. Every figure above names the file it was read from.
