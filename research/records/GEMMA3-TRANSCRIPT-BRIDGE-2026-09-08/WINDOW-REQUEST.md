# Queued request — not an active machine window

Task 1 remains the critical path. Predecessor: the Deputy's all-layer stage-two map, observed
holding PID 37593 at 2026-09-08T11:22:02Z with a 330-minute declaration. An expected end is not a
release. No window has been announced or claimed by this work; wait for the actual handoff.

## Serial sequence and projected peaks

1. Generate the three registered training cohorts through evaluation with explicit no reuse,
   24 steps and 200 emitted tokens per turn. The capture ledger counts successful forwards and
   records generated spans as tokens arrive. **Capture projection: 16 GiB**, conservative and
   unmeasured, with its weight/cache/logit/attention terms in `TRANSCRIPT-REGISTRATION-v2.json`.
   Cap prompts at 8,192 tokens and refuse oversized prompts before a forward; do not truncate them
   to make the projection fit. Capture each cohort in its own process and fresh artifact paths.
2. Freeze the corpus using the exact capture records. Inspect largest-episode share, identical-call
   concentration, four generated spans, input-role counts, absolute-position distributions and
   BOS composition. If concentration exceeds one third, stop for the Director. No resampling or
   downweighting without a ruling.
3. Native masked resource calibration, all layers. **Initial 256/512 bound: 14.5 GiB.** Use suffix
   windows whose scored tokens remain from the registered source row; an empty prefix mask is not
   a valid calibration. Measure and project larger shapes before running them.
4. BF16 and 4-bit regression fits on exactly the same frozen corpus, serially. **Provisional
   2,048-token peak: 18.5 GiB**, derived from the 12.44 GiB native 128-token fit plus the changed
   residual/logit/attention/mask buffers. This is not the measured launch allowance: calibration
   must produce that before fitting under the 22 GiB registry resource limit. Select duration
   from actual captured positions and measured calibration, then announce it.
5. Paired per-layer precision comparison and records. Fit outputs must identify both exact
   checkpoint snapshots, the same corpus hash and masks, native residual source and R57 identity.
6. A1 convention diagnostic and benchmark in a separate CPU window if needed. **Projection:
   6 GiB**, mostly the 2.69 GB FP32 readout, decoder products and bounded independent-reference
   scratch. Start with the raw identity baseline; the norm-convention diagnostic remains distinct.
   Measure 16 features before projecting the full readout, which now includes a direct pass as well
   as the lensed pass. The old 150-second estimate is not a measurement of this implementation.

## Window ownership

All invocations use the primary box-state directory
`/Users/daniel.tipton/Desktop/An app`, even from the home worktree. `runlock run` records its own
long-lived wrapper PID and closes the announcement in `finally`. If using `announce` directly,
pass `--holder-pid` for a process that outlives the command. Read `runlock status` and confirm the
holder says **running** before starting. The launch driver's own ownership/readback check runs
before checkpoint access and again before accepting completion.

No weights are downloaded for the transcript run: both converted checkpoints already exist.
Native MLX tests are machine work too and stay out of another seat's window.
