# Offline note-integrity comparison

## Sources and configuration

| run | source | data seed | keep-last | tasks |
| --- | --- | --- | --- | --- |
| runB-180 | outputs/agent-v2b/evals/runB-test180.json | 20260902 | 2 | 180 |
| best-adapter | outputs/agent-v2c/evals/best-adapter-test.json | 20260902 | 2 | 180 |

## Overall

| run | outcome success | integrity clean | affected trajectories | failed with violation |
| --- | --- | --- | --- | --- |
| runB-180 | 145/180 | 122/180 | 58 | 34/35 |
| best-adapter | 118/180 | 105/180 | 75 | 62/62 |

## Per-family

| run | family | outcome success | failures | integrity clean |
| --- | --- | --- | --- | --- |
| runB-180 | aggregate_report | 0/15 | 15 | 0/15 |
| runB-180 | batch_update | 1/15 | 14 | 0/15 |
| runB-180 | calculate | 15/15 | 0 | 15/15 |
| runB-180 | conditional_update | 12/15 | 3 | 0/15 |
| runB-180 | cross_reference | 14/15 | 1 | 15/15 |
| runB-180 | ledger_reconcile | 13/15 | 2 | 2/15 |
| runB-180 | list | 15/15 | 0 | 15/15 |
| runB-180 | pointer_chain | 15/15 | 0 | 15/15 |
| runB-180 | read | 15/15 | 0 | 15/15 |
| runB-180 | search | 15/15 | 0 | 15/15 |
| runB-180 | synthesis | 15/15 | 0 | 15/15 |
| runB-180 | update | 15/15 | 0 | 15/15 |
| best-adapter | aggregate_report | 0/15 | 15 | 0/15 |
| best-adapter | batch_update | 1/15 | 14 | 0/15 |
| best-adapter | calculate | 15/15 | 0 | 15/15 |
| best-adapter | conditional_update | 3/15 | 12 | 0/15 |
| best-adapter | cross_reference | 0/15 | 15 | 0/15 |
| best-adapter | ledger_reconcile | 9/15 | 6 | 0/15 |
| best-adapter | list | 15/15 | 0 | 15/15 |
| best-adapter | pointer_chain | 15/15 | 0 | 15/15 |
| best-adapter | read | 15/15 | 0 | 15/15 |
| best-adapter | search | 15/15 | 0 | 15/15 |
| best-adapter | synthesis | 15/15 | 0 | 15/15 |
| best-adapter | update | 15/15 | 0 | 15/15 |

## Violation kinds (affected trajectories)

| run | family | violation | affected trajectories |
| --- | --- | --- | --- |
| runB-180 | aggregate_report | value_drop | 15 |
| runB-180 | batch_update | count_mismatch | 15 |
| runB-180 | batch_update | premature_completion | 15 |
| runB-180 | conditional_update | premature_completion | 15 |
| runB-180 | conditional_update | value_drop | 1 |
| runB-180 | ledger_reconcile | count_mismatch | 13 |
| runB-180 | ledger_reconcile | value_drop | 2 |
| best-adapter | aggregate_report | premature_completion | 13 |
| best-adapter | aggregate_report | stale_fact | 15 |
| best-adapter | aggregate_report | value_drop | 15 |
| best-adapter | aggregate_report | verbatim_copy | 6 |
| best-adapter | batch_update | premature_completion | 14 |
| best-adapter | batch_update | queue_loss | 6 |
| best-adapter | conditional_update | count_mismatch | 15 |
| best-adapter | conditional_update | premature_completion | 15 |
| best-adapter | conditional_update | stale_fact | 8 |
| best-adapter | conditional_update | value_drop | 9 |
| best-adapter | cross_reference | verbatim_copy | 15 |
| best-adapter | ledger_reconcile | count_mismatch | 15 |
| best-adapter | ledger_reconcile | value_drop | 6 |

## Integrity-clean paired flips

| transition | tasks |
| --- | --- |
| clean → clean | 105 |
| clean → affected | 17 |
| affected → clean | 0 |
| affected → affected | 58 |

## Outcome paired flips

| transition | tasks |
| --- | --- |
| success → success | 116 |
| success → failure | 29 |
| failure → success | 2 |
| failure → failure | 33 |

## Memo acceptance checks

| family | violation | expected | observed | status |
| --- | --- | --- | --- | --- |
| cross_reference | verbatim_copy | 15 | 15 | PASS |
| batch_update | premature_completion | 14 | 14 | PASS |
| ledger_reconcile | value_drop | 6 | 6 | PASS |
