# Resource calibration04 passed — 7 September 2026

Source452f75c. Granted exclusive handoff in primary5a17860. Fresh launch inventory clear; ordinary loader held the primary model lock. This was the first successful checkpoint load by this task. The existing registration was unchanged. JSONL and launch metadata bind corpus, exact snapshot, source and dependency versions. Process exited0; lock and mapped-MLX inventory clear afterwards.

| Prefix tokens | MLX peak GiB | Working-set share | Before-call bound GiB | Seconds per sequence |
| --- | --- | --- | --- | --- |
| 256 | 6.1844 | 0.3482 | 10.0000 | 0.8595 |
| 512 | 6.1698 | 0.3474 | 10.0000 | 1.2568 |
| 1024 | 6.4286 | 0.3620 | 8.9823 | 2.4933 |
| 2044 | 6.8275 | 0.3844 | 9.9988 | 5.0013 |

Actual recommended working set 19069665280bytes (17.7600GiB);0.6cap 10.6560GiB. All measured sizes stayed below their registered before-call bounds and cap. Allocator cache0. Kernel pressure was1(normal) before launch, during the first sizes and after exit; no intervention. These byte observations do not establish a buffer-count bound.

One-layer fixed-grid solve 0.7636s, peak 5.4887GiB against before-call bound 5.6229GiB. No quality values were retained. The frozen518-sequence agentic fit projects to 1401.060s (23.35minutes), excluding loading and artifact serialization. This is a projection, not the required actual under-hour acceptance. The fit remains unrun; no map, lens sidecar or scientific result was created.

Next ordered stage is the separately registered Jacobian self-check, then batch8-first benchmark. All subsequent calls retain the exclusive announcement/normal-lock discipline.
