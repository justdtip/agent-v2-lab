# Corpus review and final verification — 7 September 2026

Scope: requirements §3.1/§10 only. Implementation commit `31ec13c`, review fix `016c5f6`; primary cache/rules incorporated at merge `53343a3` through `ac1b700`.

Independent GPT-6 Astra review requested two bounded changes, both medium priority/high confidence:

1. The prose reader accepted a rehashed second chunk labelled step 999 because token_start was only checked against step_index. It now reconstructs ordered source/chunk identities from declared per-source token counts, validates the corresponding division/remainder and reconciles global tails and sequence counts. This validates format/accounting without claiming that hashes are signatures or retokenizing sources in the reader.
2. Stored offsets lacked the typed/bounded/monotonic validation used for fresh tokenization. Encoding and reading now share `_validate_offsets`; negative, reversed, out-of-text, boolean and malformed offsets are covered by regressions. Legitimate duplicate offsets remain permitted.

The fix also makes `pilot_exclusion.applies_to = agentic_task_ids` explicit, so authorised WikiText validation is not described as excluded. The exact historical schema-1 rule remains readable; no frozen manifest was rewritten.

Scoped re-review of `5945118..016c5f6`: **pass**, both findings resolved, no new breakage identified. Reviewer inspected source, tests and the implementation report; execution below is the controller's separate fresh evidence.

Final checks on `016c5f6`: **81 tests passed in 4.33s**, Ruff clean, `git diff --check` clean. Both real corpora were read successfully with the exact before/after manifest file hashes unchanged. See `CORPUS-VERIFY-02.json` and `corpus-final-check-01.log` through `03.log`. The named tests' transitive top-level import closure was inspected before execution and contains no MLX; the corpus-read process also asserted that no MLX package was imported. No checkpoint or new corpus download was involved in final verification.

No regression solve, Jacobian fit, native self-check, replay identity check or scientific profile is claimed. The ridge scaling ruling and direct authorisation for the rejected GitHub backup push remain pending. The earlier README and build record are retained unchanged as dated history; cache integration is now satisfied for future replay source work.
