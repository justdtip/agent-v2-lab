# Deferred short calibration request — no active machine window

The Director's R47 correction supersedes the earlier 16 / 14.5 / 21 GiB proposals. They are all
above **0.6 of the recommended working set**, currently **10.656 GiB**, and cannot be used as
launch allowances. The registry's 22 GiB budget does not override that threshold. Historical
`TRANSCRIPT-REGISTRATION-v3.json` remains an unlaunched protocol record; its memory numbers are
withdrawn. See [R47-MEMORY-CORRECTION.md](R47-MEMORY-CORRECTION.md).

The Deputy's stage-two map is the predecessor. The last observed declaration belongs to PID
37593, opened at 2026-09-08T11:22:02Z and extended to 420 minutes. Neither an estimated finish
time nor an empty filtered process search is a handoff. No window has been announced or claimed
by this task. Model loads and native MLX tests wait for the actual release.

## First machine work

Request a short diagnostic window, starting with one window in flight. The native residual API
already enforces batch one; increasing a nonexistent batch setting is not the remedy. The
corrected candidate clears residual ownership and streams fit/held statistics separately.

Start with a fresh checkpoint load and a 128-token, batch-one diagnostic. Record load peak and
post-load active/cache bytes, then forward/statistics/release phase allocation. The previous
first 128-token sequence peaked at 9.658591 GiB, while repeated sequences later reached
12.443628 GiB. A first-row result alone cannot certify the absence of retention over a corpus.
Exercise both split slots and repeated rows, with no quality outputs or lens writing.

The narrow invocation is frozen in `R47-DIAGNOSTIC-PLAN-v2.json`: **10.5 GiB projected peak**, only
for eight 128-token forwards using `split_spill`, without solving or writing a lens. The first
four fit rows of the old run reached 9.960718 GiB. Retaining the inspected 128-to-512 tensor
allowance, despite keeping this diagnostic at 128, gives 10.104273 GiB and leaves 0.395727 GiB
for spill staging and unmeasured workspace. A single FP32 staging matrix is 0.024414 GiB.

This is an evidence-based engineering projection, not a measured peak or a guaranteed upper
bound. The old first held row reached 11.352926 GiB with both splits resident; the new bound
depends on the tested split-release behavior and cannot authorize the legacy memory strategy.
Before invocation, compare the frozen source and corpus hashes, confirm the actual handoff and
live owned window, and recheck the current device threshold. Larger shapes require new evidence.
A measured breach is a breach, not successful pre-launch protection.

The regular fitting preflight still requires an honest initial bound and checks it before
checkpoint loading. Each subsequent token length must pass a projection before its forward.
Full fitting additionally requires matching successful calibration evidence, including the
unchanged model, corpus, residual source, storage strategy, source files and scored/input
workload. A window by itself cannot waive the cap.

## Work after qualification

1. Qualify transcript capture separately before its 72-task generation. The ledger materializes
   logits, unlike residual fitting; a fit calibration does not certify capture memory. Keep
   explicit no reuse, 24 steps, 200 generated tokens, and the registered token limits. Do not
   shorten prompts or resample tasks to make a resource bound pass without recording the change.
2. Freeze the exact corpus and enforce concentration, span and position gates. In particular,
   actual FIT SCORED replay positions must reach 2,749; long inputs and held-only coverage do
   not qualify. Fit windows remain capped at 2,816 and carry their original offsets and BOS facts.
3. Calibrate the complete native masked split-spill fitting path separately for bf16 and 4-bit,
   on the frozen workload. Include solve, disk transfer and artifact validation allowances.
4. Run the two qualified fits serially, then the paired per-layer precision comparison. Announce
   duration from the real corpus and measured timings. The earlier eight-hour estimate is a
   planning estimate, not a measured duration of the revised implementation.
5. A1 remains separate, with its own CPU window and measured feature-batch timing. This memory
   correction does not alter the bridge's identity gate or normalization-convention diagnostic.

## Window ownership

Use the primary box-state directory `/Users/daniel.tipton/Desktop/An app`, including from the
home worktree. A `runlock run` wrapper owns its lifetime and records closure in `finally`.
If announcing directly, pass `--holder-pid` for a process that outlives the command. Read
`runlock status` back and confirm the holder is running before starting. Ownership is checked
before device queries or checkpoint access. No foreign lock is cleared and no foreign process
is stopped.
