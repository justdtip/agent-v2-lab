# Position coverage amendment — before capture or fitting

Director review, 2026-09-08: increase the transcript fit context from 2,048 to **2,816 tokens**
to cover the supplied audited map maximum of **position 2,749**. This supersedes the context in
`TRANSCRIPT-REGISTRATION-v2.json`; that unrun registration remains unchanged. The new frozen plan
is `TRANSCRIPT-REGISTRATION-v3.json`. Both precisions still use one corpus and the unchanged ridge
estimator. No new model run has occurred.

## Finding, technique and implementation

**Current finding.** The hosted lens was fitted at positions 16–126. Using the maximum 2,749
supplied in the Director's audit, the ratio of read endpoint to hosted fit endpoint is **21.82**.
The prior proposed 2,048-token context permits indices 0–2,047, giving a ratio of **1.343**.
The new context permits indices 0–2,815, which contains the reported endpoint. These are coordinate
endpoint ratios, not measured error factors or evidence that a new fit has improved a reading.
The newer 2,749 endpoint is attributed to the Director's supplied audit; this source-only amendment
did not independently scan the large map records. Bind the corresponding map record at result time.

**Technique.** Measure positional support using actual scored fit positions. A configured capacity,
an unscored context token, a held-out position or a source transcript offset does not establish that
the regression was fitted at a particular replay position. Record the histogram, its endpoint,
the count at or above the required endpoint, and the fraction above the prior 2,047 boundary.

**Implementation.** The v3 registration binds the requested maximum 2,749, the historical reference
maximum 2,047 and its source. Capture carries that request into its hash chain. The corpus reader
binds it back to the capture and recomputes the coverage from the frozen score masks. If the
**fit-scored replay maximum is below 2,749**, status is `ruling_required` and fitting is refused.
Input and held maxima are reported separately. No extra generation, resampling, padding or BOS
insertion is used to make the gate pass. Counts and distributions remain necessary even if the
endpoint passes: a single far-position sample is not evidence of dense coverage or fit quality.

## Why the position coordinates are comparable

The map under `cache_strategy: none` starts each generation turn without a reused prefix from the
previous turn. Prefill chunks and decoding within that turn still use cache state, and their
positions advance as `cache.offset + local_index`; they do not restart at zero on each chunk.
The regression starts every bounded replay window at zero. Both therefore measure positions from
the beginning of a freshly constructed context, with within-turn map offsets recorded explicitly.

The coordinate comparison does **not** establish identical contexts, masks, forward partitions,
floating-point schedules, BOS composition or activations. A truncated replay can remove earlier
history and shift a source token to a lower position. Original captured offsets and actual replay
positions remain separate in the manifest for this reason. The fresh-window replay disclosure
continues to apply.

A change in between-turn cache strategy requires renewed position/context verification. The
current capture path rejects turn-cache reuse explicitly, so it cannot silently inherit a future
registry default. Covering this observed maximum also does not cover future reads above 2,815.

## Historical resource proposal — withdrawn under R47

The calculation below is retained to explain the superseded proposal. Comparing it with the
registry's 22 GiB limit was wrong: the applicable ordinary threshold is 10.656 GiB. Its assumed
full-vocabulary fitting buffers were also unsupported by the native residual path. Neither this
21 GiB proposal nor the 14.5 GiB initial calibration or 16 GiB capture proposal authorizes a run.
[R47-MEMORY-CORRECTION.md](R47-MEMORY-CORRECTION.md) supersedes their resource interpretation.

The measured native bf16 128-token fit peaked at **12.4436 GiB**. The revised scalar estimate uses
the same registered dimensions: residual width 2,560, vocabulary 262,208, 34 layers, eight attention
heads and FFN width 10,240. The scored block is half a context: **1,408 positions**, not 1,024.

| Added term beyond the baseline | GiB |
|---|---:|
| 34 FP32 residual streams, growth from 128 to 2,816 | 0.871582 |
| Masked FP32 rows, 1,408 positions | 0.456543 |
| Two FP32 attention buffers, growth from 128 | 0.471680 |
| Two FFN buffers, growth from 128 | 0.205078 |
| Two vocabulary-logit buffers, growth from 128 | 5.251282 |
| **Baseline plus these terms** | **19.699765** |

The withdrawn declaration was **21 GiB**, leaving 1.300235 GiB arithmetic margin. Its comparison
with the 22 GiB registry budget did not satisfy R47. For scale, an extra BF16 gathered-row copy is
0.228271 GiB and sixteen
FP32 solve matrices are 0.390625 GiB. Their simultaneous allocation is not asserted. The fixed
fit/held statistics, 3.222656 GiB, are already present in the measured baseline and are not added
twice. This is a source estimate, not a measured bound; native calibration can still refuse it.

For each full window, token-dependent terms increase 37.5% and quadratic attention terms 89.06%
versus 2,048. The larger scored block can reduce the number of windows, so neither ratio predicts
total elapsed time. Actual corpus shape and measured calibration determine the run projection.

Capture was estimated at **16 GiB** for its 8,192-token prompt bound and library prefill schedule.
Initial 256/512 calibration was estimated at **14.5 GiB**. Both exceed R47 and are withdrawn.
The ladder's existing next-size checks use 0.6 of the device working set; the earlier prose
incorrectly described that as a 22 GiB cap. The correction adds protection before checkpoint
loading and requires matching evidence for a full fit.

## Source verification

113 affected integration checks passed with MLX imports blocked in parent and subprocesses. They
include the 2,748/2,749 fit-scored boundary, long-input and held-only controls, rehashed-target
tampering, and legacy corpus readback. Ruff and diff checks passed; independent source review found
no actionable issue. `POSITION-COVERAGE-VERIFICATION.json` records source hashes and the validated
v3 registration. Actual positional support and peak memory are still unmeasured.
