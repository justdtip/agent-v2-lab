# P6 causal patching, run of 2026-09-04 (policy C, five ledger cases): result review, ratified (2026-09-05 05:40)

Reviewer: Claude (Chief). Evidence: `outputs/probes/patch-C-2026-09-04/patch.{json,md}`, the
Deputy's read (`P6-RESULT-READ-2026-09-05.md`), and the Chief's direct read of
`patch.py:741-746` (`_is_flip`), `integrity.py:167-171` (`_fact_is_referenced`), `:292-345`
(per-step assembly, no short-circuit), `:363-395` (`_structured_fields`,
`_contradictory_fields`, `_fact_covered_by_stale_field`).

## Verdict: the artifact is valid as a run record; its flip rates are NOT interpretable. Ruling R27 fixes the scorer; rerun after B4.

## The defect, from the code

A case "flips" when the regenerated trajectory has no `value_drop` violation at the decision
step. `_value_drop_violation` skips any required fact that is "covered by a stale field":
`_contradictory_fields` marks the note's `approved:` field contradictory when its numbers are
**not a subset** of the canonical field's (`integrity.py:384`), and `_fact_covered_by_stale_field`
then treats every value in the canonical field, the dropped one included, as covered
(`:389-395`). So:

- run C's real note `approved: 25, 122` (a subset, value omitted) → `value_drop` fires: correct,
  which is why the case was selected;
- any regenerated note whose list contains a **wrong** number, e.g. `approved: 25, 122, 141`,
  is "contradictory" → `stale_fact` instead of `value_drop` → **scored as a flip**.

That is the signature in the tables: foreign residuals (`unrelated_task`) at the note and
observation positions in early layers derail the value list into wrong numbers and "flip" at
0.6–1.0; the same rows at random positions rarely derail it; nothing derails at layer 36 or
at the final token, where generation is already committed. The treatment's 1.0 at
`previous_notes` L6/L12 is therefore indistinguishable from "the note was perturbed".

The Deputy's read reaches "position-specific, content-nonspecific" and recommends a strict
flip; that recommendation is adopted and made precise below. Its claim that the dropped value
is visible in retained observations for three cases is inconsistent with selection
(`_fact_is_visible` would have suppressed the violation); it is most likely a digit-substring
grep hit inside distractor lines, and is to be re-measured with `_extract_facts`, not `grep`.

## Ruling R27: strict, auditable flip scoring for P6

1. A flip requires all of: the turn parses; the note's value field parses to a set of numbers;
   that set contains every value present in the failing note **and** the dropped value
   (digit-bounded) **and** no number outside the canonical expected set. A wrong number is a
   corruption, never a flip; recorded per generation as `outcome ∈ {flip, corrupted,
   unchanged, parse_error}`.
2. The artifact records, per case × cell × condition, the generated note text and the parsed
   value set. Aggregate-only artifacts are unauditable and are not permitted for P6.
3. Per case, `dropped_value_visible_in_retained_observations` is computed with the integrity
   module's own fact extraction and recorded; a True excludes the case from the headline.
4. The Deputy's **content control** is added: run B's note rows with the dropped value's
   positions replaced by an unrelated value's rows. Unchanged flip rate → position gate;
   collapse → the value's representation matters.
5. The `aggregate_report` secondary condition (generator-v4 note, labelled designed-correct)
   is scheduled with the rerun, not deferred.

## What may be quoted meanwhile

"P6 run 1 (2026-09-04): the decision is perturbable only at the note positions in layers 6–18;
the scorer could not distinguish restoration from corruption, so specificity is not
established; rerun scheduled under R27." Nothing about the memo's verdict changes on this run.

## Order

The scorer slice (`patch.py`, `tests/test_patch.py`, fake-only, R19 chain) is dispatched now,
ahead of the P2 redesign code. The rerun (~1.5 h) follows B4 training on the lane.
