# R27 scorer (#46), R30 refinements and secondary condition (#49), P2 capture sub-slice B1b (#48): review round 1, ratified (2026-09-05 11:20)

Reviewer: Claude (Chief). Evidence: the three work orders, the R19 findings on #41, and direct
reads of `patch.py:921-1010` (`CaseScoring`, `classify_generation`) in the main tree, the R30
worktree's `patch.py:934-1104` (`_VALUE_FIELD`, `_note_value_fields`, field scoping, `empty`)
and control scope (`:1656-1667`), and the B1b worktree's `capture.py:70-118,198-203`
(`dtype`, note-span guard), `state_probe.py:425-431` (`task_difficulties`), `models.py:51-82`.

## #46 R27 strict scorer: APPROVED TO COMMIT (ledger primary rerun may follow).

`E = canonical ∪ D`, `required = F ∪ D`; `corrupted` when S ⊄ E, `flip` when required ⊆ S,
`unchanged` otherwise; the reviewer's edge battery on case 0031 behaves as ruled; the content
control's slot replacement is now the pooled row (finding 10 closed). Per-generation records
present. Findings 12–15 are closed by #49, not carried.

## #49 R30 refinements and secondary condition: APPROVED TO COMMIT after #46.

- Clause by clause as ruled: `\w`-only words between `half` and the colon; per-label fields
  with normalised labels; E scoped to the failing note's list field(s), falling back to the
  canonical note's list fields when the failing note has none (still field-scoped, so a
  subtotal never counts); `flip_unsatisfiable` marked and excluded; `empty` outcome; control
  scope with `applicable_cases`; `head_alone` eligibility recorded for the secondary universe.
- **Deviation ruled: keep as implemented, no extension.** The seven secondary cases whose
  failing note reads `First half total = 117.` carry no list field; F is legitimately empty,
  `required` collapses to D ⊆ E, and the case remains scorable. Extending the field regex to
  `total = N` would count a computed subtotal as a list value and mark those cases
  unsatisfiable for the wrong reason. Headline for the secondary condition: 10 cases.
- `ARTIFACT_SCHEMA` → `p6-patch-r30`.

## #48 B1b conditions, dual capture, `capture_dtype`, `task_difficulties`: APPROVED TO COMMIT.

- `capture_residuals(dtype=)`: `float32` unchanged as the function default; `native` returns
  the block output untouched with one cast at materialisation; probe CLIs pass the registry's
  `probes.capture_dtype` (default `native`); every artifact records it and the preflight's
  `fp32_manual_vs_native` block. This is R18b as ruled.
- `note_token_span` widens over a merged boundary and raises when the boundary moves more than
  one token, the same guard as the P6 spans (#29).
- `task_difficulties` reads `Task.difficulty`; pinned byte-equal to the positional rule on the
  legacy splits and shown to differ on `p2-d0/1`, which is exactly the B1a defect.
- **Both questions: as implemented.** (1) `notes-stripped` must not strip the note under
  capture; the contrast is carried by the stripped context. (2) The pooled span is the thought
  only; the fenced call is family boilerplate that would inject position signal into the mean.
- Artifact stems carry the condition; legacy spellings byte-identical; `position=` defaulted
  everywhere so existing artifacts and callers are unchanged.

## Order

#46 commits; #49 applies on top and commits; #48 applies to the main tree and commits. The
ledger primary rerun needs only #46 and follows B4 (or precedes it while B4 is blocked at the
framework level, at the Director's lift). P2 runs wait for the lane.
