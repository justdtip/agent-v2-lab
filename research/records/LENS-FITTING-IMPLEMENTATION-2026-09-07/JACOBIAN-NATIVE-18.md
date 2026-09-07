# Jacobian final-source native verification

Recorded 2026-09-07T20:19:20+10:00. This appends to JACOBIAN-REVIEW-17.md; that earlier source-review record is unchanged.

The original implementer, GPT-6 Astra, ran a verification-only follow-up on commit 1534a9f16a07180d740f2b547fb5c762f161eee5. Fresh escalated inventory showed no primary model lock and no process mapping MLX. From this persistent worktree, with PYTHONPATH=src and the primary virtualenv: `python -m pytest -q -p no:cacheprovider tests/test_lens_jacobian.py`. Result: 36 passed, exit 0. This includes 30 pure tests and 6 native tiny-model cases, including actual batch widths 32 and 64 and the analytic 150-position fit/held validation. The worker checked owned source/test files against the named commit with git diff --exit-code.

No source edits, checkpoint load, runtime tests, lock clearing or foreign process intervention. The test process exited. These tests close the pending tiny-model verification of the two independently reviewed source fixes. They do not supply checkpoint self-check tolerances, timing, convergence, map agreement or a fitted scientific instrument.
