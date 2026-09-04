# B4 start fixes (issue #47): dataset protocol, split loading, health on every exit path — review round 1, ratified (2026-09-05 09:05)

Reviewer: Claude (Chief). Evidence: the work order and direct reads of the three diffs
(`tuner_data.py`, `cli.py`, `runlog.py`) against the installed library
(`mlx_lm/tuner/datasets.py:158-172` `CacheDataset`, `trainer.py:102-143` `iterate_batches`).

## Verdict: APPROVED TO COMMIT. B4 attempt 3 may start on commit.

- **Protocol.** `CacheDataset.itemlen` returns `len(self._data[idx])` and `__getitem__` calls
  `self._data.process(self._data[idx])`; `iterate_batches` sorts by `itemlen` and unpacks the
  processed pair. `RenderedRow.__len__` as the token count and `process` as a pure lookup
  returning `(tokens, offset)` are exactly right, and the tests now drive the library's real
  wrapper and iterator. Tokenisation stays eager, so the fail-closed length check is unmoved.
- **Split loading.** The train stage loads only the splits it trains on; the 15 rendered
  test rows over 2,688 tokens under the Qwen3.5 tokenizer never reach a trainer (evaluation
  runs tasks, not rendered rows). The ceiling revert is correct: the recipe stays run B's.
- **Health.** `_finish_health` runs on the normal path before provenance and again in
  `finally`; `on_finish` is idempotent; a loader failure now reads `status: error`,
  `verdict: incomplete`, one fatal flag at zero iterations. This is the #35 amendment as
  ratified.

## Ruling R31: library-protocol tests use the library's real classes

Three defects reached a live run before any test: the adapter wrapper (view walker), the
dataset protocol (`CacheDataset`), and the health path. Each was a seam between our code and
mlx-lm that fakes had stubbed on both sides. From now on, any test of a seam with mlx-lm
(dataset protocol, trainer callback, cache classes, LoRA layer types, tokenizer boundary
behaviour) drives the **library's real class** on that side with fakes only for weights and
compute, as this slice's tests do. Recorded in wiring map §7.

## After commit

B4 attempt 3 with the health record from iteration one; then the P6 ledger rerun under R27
(#46). Attempts 1 and 2 stay on disk as records; their `health.json` files are the evidence
for the amendment.
