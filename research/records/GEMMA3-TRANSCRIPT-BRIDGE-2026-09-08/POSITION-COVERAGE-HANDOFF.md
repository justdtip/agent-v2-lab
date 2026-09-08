# Position-coverage source handoff

Apply this amendment after the original payload at049a2dd, or use the updated branch
`codex/gemma-transcript-bridge`. Payload commit: `72019e1` (source change `82d4417`).

The exact patch command, from the worktree root (R58):

```sh
git diff --binary --full-index 049a2dd 72019e1 -- . > research/records/GEMMA3-TRANSCRIPT-BRIDGE-2026-09-08/POSITION-COVERAGE.patch
```

Both endpoints are committed; new files are included. Reverse apply-check against the payload
tree passed. This wrapper and patch file do not recursively include themselves.

Patch SHA256: `9d68b9993e3130cc1b512aecf3920d166b48c217bc2fe622fd751bcc64e95fc7`.

The reviewed plan is2,816tokens, with a hard actualFIT-scored replay position2,749 gate,
1,408 maximum scored positions per window, and a provisional21GiB fit projection subject to
measured calibration. `POSITION-COVERAGE-AMENDMENT.md` records the coordinate comparison,
resource calculation and remaining limits. `TRANSCRIPT-REGISTRATION-v3.json` is the launch plan;
earlier unrun plans remain unchanged.

113 affected integration checks pass with native imports blocked. No model run occurred; the
Deputy's window still reported a running holder. Position coverage and fit quality are unmeasured.
