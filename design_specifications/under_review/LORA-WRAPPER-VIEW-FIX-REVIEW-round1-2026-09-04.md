# ArchitectureView adapter-wrapper fix (issue #27), review round 1, ratified (2026-09-04 22:20)

Reviewer: Claude (Chief). Evidence: issue #27, `LORA-WRAPPER-VIEW-FIX-REPORT.md`, and a direct
read of the `arch.py` diff (the view contract is the Chief's). Verified independently that the
two other adapter-aware walkers are unaffected: `capture.lora_block_mask` finds modules by
`lora_a`/`lora_b` attributes on its own `named_modules` walk, and `adapter_delta` reads the
safetensors tensors directly.

## Verdict: APPROVED TO COMMIT. The absence of an R19 round is accepted for this case.

Reasoning for the acceptance: Director-authorised single-point fix; red-first regression on the
exact production chain (`load_model_spec(...).resolve(...)` on a wrapped fake); quantized-base
case covered; audit shows one walker on the v2 path; the fix is 31 lines with the invariant
stated (paths identical with or without an adapter; dimension readers get the module that owns
the weights). This is not a precedent for skipping R19 on multi-file or judgement-bearing
slices.

## The fix, as read

Wrappers are detected by the type of `.linear`, not by the attribute's presence; reported under
the wrapper's own path with the base module; the base's `<path>.linear` entry and anything
beneath a wrapper are skipped; unwrapped modules unchanged. Correct, and consistent with what
adapter comparison and provenance assume.

## Bound follow-up (deadline: before the next slice touching `arch.py`, and in any case before
the B4 evaluation lift)

A wrapper-carrying variant joins the standard arch fixtures so every future contract test
exercises "adapter-loaded" alongside dense and hybrid. The three regression tests stay as the
specific proofs.

## Record

Wave-1 regression: since wave 1 every `load_policy` resolves and resolution assumed an
unwrapped, non-lazy model; lazy loads were evidenced on #19 (B2), wrapped loads are this fix.
Found by the Director's live P6 run, which is also the first real-run confirmation of the #26
selection (five pinned cases).
