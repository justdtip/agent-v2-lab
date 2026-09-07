# Native live-lens infrastructure verification — 7 September 2026

The native capture path passes on the cached, unadapted Qwen3.5-4B checkpoint. This is an infrastructure result; it does not certify adapter readability or complete the ten-episode pilot.

## Evidence

- `acceptance-v1.json`: five turns and 190 emitted tokens, 200 native forward calls. Every forward logit hash and every generated turn matched with capture enabled. Prefill, decode, and the observation-window transition were exercised. Zero injection with rebuilt state and same-shape final-layer identity both had max absolute error **0**. Peak MLX allocation was **3.930 GiB**, under the declared 12 GiB window. The native comparison wraps the model only to retain its outputs; it uses no internal capture hooks.
- `short-capture-{1,2}.jsonl`: byte-identical full records, including source top-k readings, sampled attention rows, head readings, rolling-distribution ranks at horizons 1/4/8, and hashes of native outputs. Native fixed-history replay also matched exactly.
- `reconstruct.py`: independently reconstructs all seven retained audit rows with **zero error**, using the stored attention weights and top-k readings. Run with the worktree source first on `PYTHONPATH`. `audit-reconstruction.json` retains the first check, whose algebraically equivalent multiplication/division order differed by at most 6.94e-18; the reconstruction script uses the specified overlap fraction before weighting. No record was replaced.
- `population.json`: the recorded selection plus the floor yields **23 primary heads**; **26 sensitivity heads** are frozen separately in the same manifest. File SHA-256: `10424db1237671ee484ca0310b10cc7f86b5ebe4b616ff2a4adab035382ab6da`. The manifest was written before these checkpoint tests.
- `tests-v2.txt`: **207 passed, one pre-existing repository-guard failure**. The failure flags two pre-existing 2048-token row-cap configs as forbidden model constants. It was independently reproduced in the primary checkout. The 183 focused live-lens, runner, architecture, model-registry, and CLI tests pass. The other 24 passing tests are repository guards.
- Targeted Ruff checks and `git diff --check` pass. The pre-existing unused `render` import/name collision in `pipeline/cli.py` was not included in that clean lint claim.

## Measured cost

`benchmark-v1.json` retains all observations: one warmup per mode, three measured repetitions with rotated order, a nine-token prefix and eight fixed decode forwards. Every mode matched native forward hashes. Timings include identical output-hash checks. These are short-prefix measurements, not estimates for a whole episode or a long context.

| Mode | Median seconds | Relative to native |
| --- | ---: | ---: |
| Native | 0.1555 | 1.00× |
| Ten residual reads | 0.1812 | 1.17× |
| Lens reading and full-vocabulary ranking | 2.1156 | 13.60× |
| Attention rows, 80 heads | 0.1955 | 1.26× |
| Attention and written vectors, 80 heads | 0.2120 | 1.36× |
| Full record | 4.2024 | 27.02× |

The separate five-turn episode took 15.62 seconds native and 17.26 seconds with residual/head hooks. That comparison excludes lens unembedding and cannot stand in for full-record cost. Full capture remains expensive and needs a longer-context benchmark before a pilot cost estimate.

## Scope and limits

The source adds scoped native observers in the architecture compatibility boundary; optional runner capture; immutable instrument loading; append-only hash-chained records; exact native replay; injection safeguards; and three independent lens-verdict fields. Only a failed map-agreement check requests a refit. No concept or output-divergence value can trigger it.

Training declaration checks run before the trainer import for the model with the recorded launch envelope: explicit chunkwise recurrence, chunk/row/batch envelope, explicit batch iteration units and complete gradient-accumulation groups, and a finite bounded allocator cache. Future launches need `train.iters_unit: batches`. Existing primary configs and the trained adapter were not changed. Prompt-token caps still use the existing tokenizer-aware dataset loader; a byte cache limit does not guarantee a Metal buffer-count limit.

Attention capture observes the last query of each native forward, including every decode step; source readings cover its full prefix. The current path requires ordinary KV caches and the resolved no-reuse strategy. Final-layer identity preserves both dtype and tensor shape. Hash-only records support exact replay; nonzero tolerances require explicitly retained logits. Head vector sums differ from the native low-precision projected sum by up to **0.0625 absolute** in the episode; this is retained as a descriptive reconstruction diagnostic, with no acceptance bound claimed.

Still outstanding: a numerical acceptance bound for same-position nonzero injection against the older offline path; the layer-20 adapter Jacobian sampling/averaging protocol and prespecified bound; held-out concept contrasts and the elicited-report harness; token-level span labels and matched controls; the ten-episode pilot and viewer. The finite derivative comparison helper currently accepts prepared responses; an adapter measurement driver is not yet implemented. The relevant scientific decisions must be supplied before their measurements. No adapter calibration, fitting, grid, factorial or whole-episode ablation ran.

## Reproducibility and review

`protocol.json` and `benchmark-protocol.json` precede the corresponding measurements. The scripts use the primary checkout's process-scoped model lock, even though they import this worktree's source. Launch/completion entries were appended to the shared heartbeat log; both model processes exited and released the lock. The hosted n1000 lens is verified against SHA-256 `381c089dcffead8147ee91f944496f468cce2c7d593e0a1b17230745055aea12`.

`source-manifest.json` identifies the final source by file hashes. The default capture path used for acceptance is unchanged by the later attention-only benchmark switch; all native fixture checks were rerun, and the final benchmark verifies both switch values on the checkpoint. The patch remains in the isolated worktree pending Chief review; no implementation hunk has landed in the shared checkout.

## Guard addition, 7 September 2026, 21:05 (Chief, landing issue 89's rule)

`acceptance-v1.py` and `benchmark-v1.py` ran every statement on import — no `__main__` block —
with the checkpoint path absolute and the primary model-run lock taken on line 13, so `python
benchmark-v1.py` loaded the 4B model. Both now carry the standard refusal guard at the top; the
guard is the only change (`acceptance-v1.py` e6adb67786088fae8b193701bf12383b52163042402d17051eabac9e5d665299 → 5040ef280d63975b19c551f1749db47c7896484944c9d927cdef5ce97dc0aaa3; `benchmark-v1.py` 614b8483f34e17f3d5db968b6b69dc4a6fb1e3b69ad7840adf95e1d0acb29f0b → 55a4fd8f6a6738ece3cbc2a0c91f7a703e1cf96d0735d604c9715adf06079669).
`source-manifest.json` pins source files, not these scripts.
