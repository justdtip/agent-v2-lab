# R47 memory correction handoff

Branch: `codex/gemma-transcript-bridge`.
Incremental base: `adcd43d0d95a276ba38a87ccd60d5f3c9a916f3c`.
Payload commit: `773503683c136f027b2ca2e176d288bb6892b5cb`.
Patch: `R47-MEMORY.patch`.
Patch SHA-256: `bd83f94ac2cbe7024d24bc314273b73f8f8bea7e8f33f151b83592c6b68dd61e`.

The patch contains the source correction and its records. It excludes this handoff and its own
patch file. Reverse application against the payload tree passes `git apply --reverse --check`.
On the matching base, review and apply with `git apply --check R47-MEMORY.patch` followed by
`git apply R47-MEMORY.patch`; do not apply over unrelated edits without checking the diff.

The 21 GiB proposal is withdrawn. R47's current ordinary threshold is **10.656 GiB**. The fitter
already uses batch one. It retained both fit/held statistics and had a native collector ownership
cycle. The patch clears residual ownership on success and failure and offers explicit split-spill
statistics, reducing their simultaneous size from **3.223 to 1.611 GiB** during accumulation.
It keeps within-split FP32 accumulation order, masks, forwards and ridge objective unchanged.
The bf16 weight files are approximately **7.228 GiB**, a file-size proxy rather than measured RAM.

Full regression fitting now requires successful, workload- and implementation-bound memory
qualification before checkpoint loading. Calibration repeats four updates per split, checks each
shape projection before execution, and records phase allocation. Runtime checks preserve the R47
cap. MLX allocation counters and analytical host reserves are not whole-process RSS measurements
or protection against the first unexpectedly large allocation.

**244 affected integration checks passed with real MLX imports blocked**, including subprocesses.
Ruff and whitespace checks passed. See `R47-SOURCE-VERIFICATION-v2.json` for exact file hashes and
checks. Native numerical equivalence and the new real-checkpoint peak remain unmeasured.

`R47-DIAGNOSTIC-PLAN-v2.json` freezes the next step: batch one, 128 tokens, four fit plus four held
forwards, split-spill, no solve or lens artifact, with a **10.5 GiB engineering projection**. This
uses the old four-fit-row peak plus an explicit allowance; it is not a measured or guaranteed
peak and qualifies no larger workload. Check its frozen hashes before invocation. No model run
or new machine window was launched for this correction. Wait for the Deputy's actual release,
announce a live owned short window, read its holder status back, then run the diagnostic.

Transcript capture has a pre-load scalar projection check, **not** a capture-specific measured
qualification validator. Its withdrawn 16 GiB projection cannot launch; capture needs a separate
honest bound or calibration. Existing corpus concentration, span, precision and actual scored
position coverage gates remain in force. See `R47-MEMORY-CORRECTION.md` for the finding, transferable
technique, implementation and limitations, and `WINDOW-REQUEST.md` for subsequent sequencing.
