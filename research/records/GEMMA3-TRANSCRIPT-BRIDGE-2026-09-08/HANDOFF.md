# Transcript and bridge source handoff

Payload commit: `f566f34`, branch `codex/gemma-transcript-bridge`.
Base: `2edad6e57a28b335b3b820a83931294af21c7418` from `codex/agent-v2-specs`.
Worktree: `/Users/daniel.tipton/worktrees/gemma-transcript-bridge`.

## Delivered and checked

The training-transcript capture/freeze path, native masked fitting and calibration, paired-precision
identity checks, four generated-token spans, original/replay position histograms, BOS policy and
concentration gates are implemented. Both source prerequisites are pinned before generation.
The complete implementation contains 37 changed/new files, including tests and frozen records.

The A1 path authenticates the exact decoder and shipped examples, runs the direct identity baseline
first, preserves its compatibility failure as evidence, and measures direct-versus-J readouts.
Labels remain explicitly unlabelled. CPU benchmarking now measures both passes and serialization.

The Director-authorized final normalization vector is acquired, source-shard hash verified and
committed as `final-norm-weights.npz`. It contains the stored vector and mathematical FP32 `1+w`
gain. It does not contain the input-dependent RMS denominator or assert bit equivalence to a
native low-precision RMSNorm kernel. The source audit distinguishes Google's tutorial formula from
the model's channel gain; the actual shipped examples generator remains unidentified.

190 integrated checks passed with MLX imports blocked in both parent and subprocesses; Ruff and
diff checks passed. `SOURCE-VERIFICATION.json` binds evidence to source hashes. Independent source
review found no remaining actionable issue after the benchmark/sequence corrections.

## Complete patch, including new files (R58)

The exact command used, from the worktree root:

```sh
git diff --binary --full-index 2edad6e57a28b335b3b820a83931294af21c7418 f566f34 -- . > research/records/GEMMA3-TRANSCRIPT-BRIDGE-2026-09-08/IMPLEMENTATION.patch
```

Both endpoints are commits, so new files are included; this is not an untracked working-tree diff.
Patch SHA256: `d6ee11b033e1c65d99e36b75ab3e975b82efefbc1b75efd544addf9ea546722d`.
Size: 354,342 bytes. `git apply --check` passed against the primary tree. This handoff and the patch
file itself are delivery wrappers added after the payload commit; they do not recursively appear
inside their own payload.

## Remaining scientific work

No new rollout or model fit ran. The Deputy's map still held a **running** window on final readback.
Task 1 generation is the critical path once that seat releases the machine; `WINDOW-REQUEST.md`
and `TRANSCRIPT-REGISTRATION-v2.json` give the exact sequence and projected peaks. Calibration,
concentration acceptance and both native fits must precede claims about transcript lenses.

A1's real direct overlap, norm-convention diagnosis, full readout and resource measurements are
unrun. The registered convention diagnosis is separate from the unchanged raw-W gate. A2 waits
for the transcript lens. No new box window was opened, no foreign lock was cleared, and no process
belonging to another seat was stopped. The shared source tree was only read and checked.
