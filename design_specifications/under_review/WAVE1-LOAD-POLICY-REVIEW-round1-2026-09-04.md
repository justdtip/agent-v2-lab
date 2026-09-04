# Wave 1 (load_policy registry migration) review, round 1, ratified (2026-09-04 15:10)

Reviewer: Claude (Chief). Evidence: issue #19 (Deputy's work order and the independent
reviewer's findings, verbatim). Diff not re-read by the Chief; the reviewer's ten-item
verification and the Deputy's independent suite run are adopted.

## Verdict: APPROVED TO COMMIT with conditions. Code needs one addition; the report needs correcting.

The Deputy's readiness call is upheld, including the downgrade of B2 on artifact evidence
(preflight resolved twelve LoRA keys and a parameter count on a lazily loaded checkpoint; that
is the same code path exercised). The reviewer's B1 and the mechanism correction are exactly
what R16 exists for and are adopted.

## Conditions before commit

| # | Condition | Kind |
| --- | --- | --- |
| K1 | Report corrections: two remaining direct loaders (`cli.py:125-130` and `research/jspace_sweep.py`), not one; B2 wording ("risk changed for `run_residual_control`, evidenced low by the preflight artifact"); the assertion mechanism ("always live; now asserts against the loaded model's spec"); twelve call sites; the B3 disclosure naming the crossed claims and the `tests/test_integrity.py:558-561` edit against the rollout lane's plan | report (R16) |
| K2 | One fake-only test exercising `run_residual_control`'s `view_factory=None` branch, which is the production configuration (`cli.py:742`) and currently untested | code |
| K3 | `mine_pairs` optional `view`/`resolved` carries a comment `# DEBT(R20): required once the condition-4 slice threads branch.build_prompt` | code |
| K4 | Commit message references #19 and the ratification; pushed by the Deputy | process |

## Decisions on the three questions

1. **`mine_pairs` asymmetry: accepted as debt with expiry (ruling R20).** It expires with the
   slice that completes R15 condition 4 (below); that slice makes the parameters required and
   updates the two `tests/test_branch.py` sites that exercise the optional path.
2. **`_load_training_base` (B1's second loader) belongs to the next slice**, not this one: it
   is a training-path load in SPEC-001 §7's territory and should go through `load_policy`
   with `adapter=None, lazy=False`.
3. **R15 condition 4 is not ticked.** The Deputy is right that this delivers half. The
   "condition-4 completion slice" is: `resolve_policy` reading `ModelSpec.policies`;
   `branch.build_prompt` threading (Codex's abandoned spec-threading work is the starting
   point); `_load_training_base` through `load_policy`; R20 debt removed. That is the next
   dispatch after R17.

## Process

- B3 is accepted as disclosed. The four crossed claims are orphaned (no coordination activity
  since 12:45); the Deputy owns the scoping decision and says so. Going forward, cross-lane
  edits are named in the dispatch, not discovered by the reviewer.
- The `adapter_delta` raw-id routing through `_default_spec` is the intended registry
  behaviour; accepted.
- Dispatch R17 (scanner substring pass, layer-fraction defaults) now on its disjoint paths; the
  Deputy's pause was correct while the boundary was unreviewed and is no longer needed once
  this commits.
