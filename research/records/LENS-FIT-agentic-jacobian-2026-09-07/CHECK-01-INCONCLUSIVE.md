# Checkpoint self-check01 — inconclusive, fitting stopped

7 September2026. Sourceb0470fb, plan9fa35bb395d8c424b982d817f966204dc39bdb8d31f28ed9b9dd716d33dbc558, exact files and dependencies bound by launch/run records. The process exited2 via the existing proof gate. Neither original thresholds nor epsilon rule changed after results.

| Layer | Epsilon | Maximum epsilon/half response difference | Reference stability | Restore and broadcast |
| --- | --- | --- | --- | --- |
| 1 | 0.5059779739 | 0.6019568443 | fail | inconclusive |
| 16 | 1.838570404 | 0.1101027727 | fail | inconclusive |
| 31 | 8.900575562 | 0.01689071953 | fail | inconclusive |

These are directional finite-difference responses in final pre-norm residual units, not logits or output-token divergence. The reference-stability criterion is coordinatewise0.003+0.03*abs(reference); the recorded function verdict is fail. Because reference stability failed, response_agreement reports each cache comparison inconclusive with max_error null. Therefore these records cannot say that either cache path disagreed or agreed numerically with a stable reference. The six configured comparisons did run, with16 unique directions and actual restore width1/broadcast width8.

This is an instrument failure to establish a stable derivative at the prescribed step size. Possible causes include finite-step nonlinearity or numerical cancellation/roundoff; the present data do not distinguish them. A tolerance increase would not answer that question. The fixed full-primal epsilon rule remains the production rule until a Director amendment. No fit, benchmark, map validation, native pilot identity, prose capture or scientific profile was run after the stop.

Operational evidence: check-01-supervision.jsonl carries normal kernel-pressure samples and the owned PID88320. Launch11:47:17.416UTC, observed gone11:50:10.167UTC, so elapsed execution is at most172.752s; no exact end timestamp or final MLX peak was emitted by the self-check stage. Its internal per-workload8GiB checks did not raise. Do not replace the missing MLX peak with OS RSS. The following diagnostic proposal requires explicit per-layer/scale progress, final elapsed and measured MLX peak in its artifact. Primary lock and mapped-MLX inventory were empty after exit.

The overall exclusive window is closed in primary and issue88 may proceed. Regression calibration04 remains successful; scientific acceptance remains incomplete. Next action needs the Director's method ruling on the separate epsilon-sweep proposal.

Timing precision note:172.752s above is the interval between the launch timestamp and the later supervision timestamp, with process inspection immediately following that timestamp. It is an observational duration of about173s, not an exact process-exit timer. The diagnostic cost uses it only as an estimate and carries a separate stopping budget.
