# P6 causal patching, rerun under R27 (2026-09-05, policy C, five ledger cases): result review, ratified (2026-09-05 23:30)

Reviewer: Claude (Chief). Evidence: `outputs/probes/patch-C-r27-2026-09-05/patch.{json,md}`,
`run.log` (1 h 19 min, logged under R26, schema `p6-patch-r30`), and the per-generation
records (690 generations, every note text on file). Five scoring-version-stable cases, none
excluded for visibility. Strict R27 scoring; three controls.

## The result

| Cell (treatment) | L6 | L12 | L18 | L24 | L30 | L36 |
| --- | --- | --- | --- | --- | --- | --- |
| `previous_notes` (B's note-region residuals into C) | 4/5 flip, 1 empty | 4/5 flip | 4/5 flip | 0 (5 unchanged) | 0 | 0 |
| every other cell | 0 | 0 | 0 | 0 | 0 | 0 |

Controls: `unrelated_task` 0/5 flips everywhere (its early-layer outcomes are `corrupted`,
which the old scorer had counted as flips); `random_positions` 0 everywhere;
`content_swap` at `previous_notes` 0 flips and **5/5 corrupted at L6**.

## What the generated notes show (verbatim from the artifact)

- Treatment at L6–18 writes the dropped value back in its place:
  case 0127 `approved: 141, 120, 89, 23` (89 was the dropped value); case 0163
  `approved: 155, 168, 32, 143`; case 0175 `approved: 27, 54, 49`. Restoration, not
  perturbation. The single L6 non-flip (`empty`) restored 85 but dropped the field label, and
  the strict scorer correctly refused to count it.
- Content swap at L6 writes the **foreign** number in the dropped value's slot:
  `approved: 25, 122, 88, 55` where 85 was dropped and 88 is the swapped-in value; `141, 120,
  10`; `86, 98, 33, 48`; `155, 168, 4`; `27, 88, 49`. The rows at the value's positions carry
  the value's identity, and C's later layers write whatever identity those rows carry.
- At L24 and beyond every treatment is `unchanged`: C reproduces its own dropped-value note.
  The transfer from the note positions to the writing position is complete by layer 18.
- `dropped_value_slot` (one pooled row at the separator) yields digit fusion at L6/12
  (`approved: 25, 1228, 55`, `141, 1208, 23`): the injected identity attaches to the preceding
  number rather than becoming a list item. A value's representation is distributed over its
  tokens and the separators around it; a single pooled row is not a list entry.

## Reading, against SPEC-004 §5's pre-registered decision rule

Flips come from patching the note-value region at early and middle layers, are
position-specific (random positions do nothing), content-specific (swapping the value's rows
changes the number written), and absent from late layers and the final token. **The error is
in reading and representing the note's value list at the note positions in layers ≤ 18, not
in the computation that follows**: given B's early-layer representation of the list, C's own
later layers write the correct note. That is the write-side half of the regimen verdict, now
with strict scoring and three controls at zero. It also says why run C's templates failed:
they changed what the early layers had to encode, and the encoding did not survive
extrapolation.

## Limits, stated

Five cases, one family (`ledger_reconcile`); a rate of 0.8 carries a Wilson interval of
[0.38, 0.96]. The `aggregate_report` secondary condition (ten headline cases under R30, HEAD
note as the designed-correct counterfactual) is the next run and the only way to extend the
claim to the family that matters most. Indicative and consistent; not yet decisive across
families.

## Quotable

"P6 (R27 rerun, 2026-09-05): patching run B's note-region residuals into run C at layers
6–18 restores the dropped value in 4 of 5 ledger cases with all three controls at zero;
swapping only the value's rows makes C write the swapped number. The failure is in the
early-layer representation of the note's list, which data reaches; the downstream computation
is intact."

## Bookkeeping

Probes checklist B3 ticks as "run complete, reading ratified". The decision memo gains a P6
entry. The secondary condition run follows on the lane (about 1.5 h at ten cases).
