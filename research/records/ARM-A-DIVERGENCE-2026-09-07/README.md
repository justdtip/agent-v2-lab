
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

## Verification note, 2026-09-09

The fixed-history log-probability figures in §4 rest on `ids_full[n_p:]` being the recorded turn's
own tokens, which holds only if concatenating prompt and turn does not re-tokenize at the seam. The
assertion in `scripts/fixed_history_lens.py:207` that appears to guard a neighbouring form of this
is inert (`... or True` is always true) and has never tested anything. The seam itself was measured
directly on 2026-09-09 from `run/fixed_history_lens.json` and the tokenizer alone: **all 11 adapter
turns and all 11 base turns tokenize cleanly at the seam**, so the slice is correct in every pair and
the adapter-versus-base log-probability comparison stands. Recorded as the twenty-sixth entry of
`research/records/METHOD-2026-09-08/README.md`, which also says why the assertion was reported
rather than repaired.
