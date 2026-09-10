# CPU calibration after the WS-A review

Written before any full-checkpoint load. This is a resource and precision measurement; it does
not release the graph-once estimator. The initial seam evidence remains in its original files.

## Material and execution

- Official local `google/gemma-3-4b-it` snapshot
  `093f9f388b31de276ce2de164bdc2081324b9767`. CPU, stored bf16 text weights only, eager attention,
  torch 2.14.0, one thread, deterministic algorithms, no autograd and no MLX imports.
- Hugging Face performs checkpoint key conversion. The loader audits all text keys, shapes,
  restored tying and the exact excluded vision/projector key set. A tiny serialized fixture
  verifies every loaded tensor against its original value and a missing-weight control refuses.
- `calibration-token-ids.json` freezes the first 1,400 IDs of turn 3 in the first lexicographic
  complete pilot record having a sufficiently long turn. The existing reader validated its hash
  chain. The original turn contains 1,430 IDs. The source and selected JSON carry SHA256 hashes.
- A live `runlock run` wrapper owns the window; the calibration verifies that holder and its nonce,
  then takes the model lock before loading. It never clears another process's lock or window.
- The explicit R47 cap stays **10.656005859375 GiB**. The new device helper's physical-RAM CPU budget
  is not used to increase it. Projection for this one-sequence calibration is **9.8 GiB**: text
  weights 7.2276, simultaneous loop/native residuals 0.7010, native logits 0.6838, plus roughly
  1.19 GiB for runtime, observation and attention intermediates/allocator retention. This is an
  estimate to test, not an established peak. The promoted 2.5006 GiB output head is never invoked;
  no float32 full-sequence unembedding is included.
- Measure fresh-process RSS and lifetime peak after loading and during complete all-layer
  comparisons at 64 and 1,400 IDs. Stop if a measured peak exceeds the cap. Earlier measurements
  are saved before subsequent work. Full shard hashes are computed without bulk buffering.

## What the numbers mean

The normal comparison is the view's float32-promoted loop against native bf16 capture. Report
per-layer maximum absolute error and the explicitly defined descriptive ratio
`max(abs(loop-native)) / max(abs(native))`. Also report the model's rotary tables evaluated with
float32 output against those observed from its native bf16 path, and the mask-dispatch control.

The small-model diagnostic already shows why this cannot silently count as the former float32
identity check: at both lengths the all-float32 paths agree exactly, while the mixed reference
shows approximately 0.6–1.1% relative discrepancies. These are three-block random-model findings,
not measurements of Gemma 4B. The discrepancy includes arithmetic rounding throughout the native
blocks, not just rotary-table rounding. The 1e-3 registered bound is not relaxed.

Gate 1 checks the loaded model against its own config. Gate 2 records measurements, without
claiming full acceptance while the reference precision and additional control requirements remain
unresolved. Gates 3–4 and the estimator remain unexecuted after an unaccepted gate 2. The hook-site
and entry-transform controls are named as unexecuted, never silently credited. No CUDA/MPS runs.

## Reproduction

Run `cpu_gates.py --help` for the explicit arguments. Execution requires `--execute`, the local
snapshot, frozen token JSON, a new output path, the above cap/projection and the exact current
source commit; a stale declared commit refuses. The external window names the projected peak.
Exit 1 can mean a completed calibration with acceptance still incomplete; read `status`, `gates`
and `error` in the JSON. It never means a passed acceptance suite.
