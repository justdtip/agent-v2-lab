
---

## Note, appended 2026-09-08: arm A's recorded peak is 9.726 GiB, not 10.444

`outputs/agent-v2e-qwen35-4b/health.json` records `peak_memory_gb: 10.443645016`, and the field
name is honest: `mlx_lm/tuner/trainer.py:341` computes it as `mx.get_peak_memory() / 1e9`, so it
is **gigabytes**. Every budget in this repository is in gibibytes — the registry declares
`budget_gib`, the device reports a recommended working set of 17.76 GiB, and `preflight` resolves
its gate in gibibytes.

**Converted, arm A peaked at 9.726 GiB, which is 0.548 of the working set.** Anyone reading the
recorded 10.444 against a gibibyte figure is 7 percent high; issue 93 is the fix in the code, and
this note is so the number is right where the run itself is described.
