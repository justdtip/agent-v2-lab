# R25 P6 treatment alignment (issue #30, ruling on #28), review round 1, ratified (2026-09-05 01:10)

Reviewer: Claude (Chief). Evidence: issue #30 (Deputy's direct read and real-tokenizer
re-measurement on all five cases) and a direct read of `patch.py:924-1120`
(`_tail_alignment`, `_value_alignment`, `align_groups`, `_alignment_rows`).

## Verdict: APPROVED TO COMMIT. The Director-expedited chain (no independent reviewer) is accepted for this slice because the ruling it implements is fully specified and the Deputy verified it by measurement on all five cases; not a precedent.

Clause check against R25: (a) equal groups pass as `identity`, longer sources patch from the
tail with the residue recorded as token ids and text — correct. (b) shared values paired by
string identity with a cardinality guard; the dropped value's source rows mean-pooled in
float32 to one row and written at the separator after the preceding shared value, with a
recorded `before_first_shared_value` fallback; slot position, overwritten token, pooled source
tokens and the value recorded; duplicate slots refused — correct. (c) controls resample to the
post-alignment cardinality; the slot cell draws one position — correct per the work order.
(d) seven cells; artifact carries rule and both cardinalities per cell — correct.

## The four open points: keep, as the Deputy recommends

1. Adjacent dropped values sharing a slot: keep the refusal; rule only on a real case.
2. `previous_notes` tail sheds the group's leading tokens (the first note's "Plan: list
   the"), not the substituted note's: that is R25(a) applied to the concatenated group, the
   residue text is recorded, and the substituted note's tokens are all inside the patched
   window. Keep.
3. `artifact_schema: "p6-patch-r25"`: keep.
4. `shared_value_tokens` records pre-alignment cardinalities beside `patched_positions`:
   keep; it makes the drop visible.

Word-boundary value matching and the random-control guard test (the two notes from the #29
review) are confirmed present.

## Commit

Own commit referencing #28 and #30; Deputy pushes. The Director's third P6 attempt may follow
with the unchanged command. Plan-document prose naming six groups is a docs sweep item.
