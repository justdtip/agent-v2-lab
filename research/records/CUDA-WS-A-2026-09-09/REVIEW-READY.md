# WS-A is ready for source review

The review corrections and CPU calibration harness are ready for review on
`codex/cuda-torch-seam`. Verified source is
`ac8a91eade9f37dcaf053db667a88b395bf52b14`; the complete test evidence and handoff
are committed at `1265542`. This readiness commit changes documentation only.

## Reviewable result

- All torch seam imports resolve the pinned upstream implementation through the
  shared loader, with explicit missing-clone and conflicting-clone behaviour.
- `input_device` reads actual embedding placement without moving parameters.
- Evidence is refreshed on torch 2.14.0: **2,203 passed, 10 skipped**, with the
  missing historical datasets/results identified. The earlier scanner failure
  and its successful correction are both retained.
- Duplicate patch artifacts are removed; commits are the source authority.
- The CPU calibration uses audited official bf16 text weights, fixed token IDs,
  a declared source commit, an owned window/model lock and the unchanged
  **10.656005859375 GiB** cap. Its interruption and loader paths have tests.

The handoff is `README.md`; `review-verification.json` binds completed test runs
and source hashes. The original historical raw pytest output is preserved,
including its whitespace, rather than rewritten as clean evidence.

## Execution still in progress at handoff

Calibration retry02 began at 2026-09-09 01:42:09 UTC under wrapper PID 77000 and
model PID 77007. At the review snapshot, both 64-token and 1,400-token normal
comparisons had completed; the long rotary and mask-control work was ongoing.
The observed lifetime peak through the 1,400-token normal comparison was
**6.8061676025390625 GiB**. This is a partial measurement, not the final peak.

The largest per-layer descriptive ratio `max(abs(loop-native))/max(abs(native))`
was **0.012385945924570863** at 64 tokens and **0.06901261613175676** at 1,400.
The native reference computes in bf16 while the loop promotes blocks to float32.
These numbers therefore combine numerical precision differences with any seam
error; they do not establish that the seam is equivalent or defective. The
registered 1e-3 bound has not been relaxed.

The evolving `cpu-calibration-02.json` and `.log` are deliberately not included
in this commit while the process is writing them. Its wrapper owns cleanup;
no other seat's process, lock or window may be cleared. Final results will be
preserved in a later evidence commit. If the run is still active at the follow-up,
inspect it without importing MLX or loading another checkpoint.

## Pending acceptance and review decision

Gate 1 passed against the actual loaded config. Gate 2 is measured but not
accepted: reference precision and the additional hook-site/entry-transform
control requirements still need resolution. Gates 3–4, CUDA/MPS and the
graph-once estimator remain **unexecuted**. The estimator must not start before
the four ordered checkpoint gates are accepted. The Q5 capture-dtype ruling is
also not assumed.

The reviewer should distinguish source readiness from checkpoint acceptance.
A matching-precision native reference would separate seam agreement from the
precision experiment; this is a recommendation for a ruling, not a convention
implemented by this handoff.

## Requested follow-up

Fifteen minutes after this readiness commit, check committed review notes and
Git for review results and the next authorized work. Read the current migration
plan and workstream ownership before acting; preserve other seats' changes.
