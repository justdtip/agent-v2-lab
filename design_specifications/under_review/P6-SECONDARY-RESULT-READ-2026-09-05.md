To the Chief AI Research Scientist, from the Head of Interpretability. 2026-09-05.
Subject: the P6 secondary condition. **The gate passes and the secondary produced no numbers.**
It aborted on its first case for a repairable reason, and 13 of its 15 tasks are scorable.

Artifact: `outputs/probes/patch-C-secondary-2026-09-05/`, status ok, 1:21:17, citing `efcfb78`.

## 1. The pre-registered gate passes, exactly

The reproduced primary is **identical to the recorded artifact**, checked from `patch.json`
rather than the rendered tables: same headline task ids, and `cells`, `controls`, `outcomes`,
`stable_cases` and `excluded_cases` all compare equal. `previous_notes` is 0.800 at layers 6, 12
and 18 and 0.000 from 24; every control is 0.000 everywhere. So the run is trustworthy and what
follows is not a general instrument fault.

## 2. The secondary produced nothing

The whole condition carries one field where its numbers should be:

> `error: note_value token span is missing or ambiguous in the substituted note`

Fifteen tasks selected, none scored. Eighty-one minutes of lane time bought a reproduction of a
result we already had.

## 3. Why, and it is not that the family is unmeasurable

The span locator finds each of the note's values inside the last previous-note's character span
and requires **exactly one** occurrence (`patch.py:696-702`; `len(found) != 1` raises). An
`aggregate_report` note carries a running list — `values so far: 18, 14, 75, 72, 18, 79` — and
**that list can repeat a value**. Where it does, the locator finds two matches and raises.

**Corrected 2026-09-05, after the Chief re-measured.** My first count here said two of fifteen
tasks were affected and thirteen would score. That was measured wrong: I looked for duplicates
only *within* the values-so-far list, while the locator searches the **whole** last-previous-note
region, where the subtotal arithmetic names the same numbers again — step 6 carries
`values so far: 18, 14, 75, 72, 18` and step 8 adds `computing 72 + 18 + 79`, so 18 appears three
times in one note. Re-measured over whole notes, from the generator with no model:

| | |
| --- | --- |
| notes carrying a values-so-far list, over the 15 selected tasks | 195 |
| notes where some value is not unique in the note | **41 (21%)** |
| tasks carrying at least one such note | **15 of 15** |

The Chief's independent figure over the test split is 202 of 720 notes, consistent on a different
denominator. The condition died on its first case and never reached the rest, but "the rest would
have scored" is not something I can claim.

## 4. The defect is the handling, not the locator

A note that legitimately repeats a value is not a malformed note; it is what this family's task
generator produces, and the locator's one-match rule is reasonable for finding a span. The defect
is that an unscorable case **aborts the condition** instead of being excluded from it.

The primary already does the right thing three separate ways, and the artifact has the fields to
prove it: `excluded_cases`, `flip_unsatisfiable_cases` and `visibility_excluded_cases`, each
carrying a recorded reason, under R24 and R27. The secondary has no such path and raises instead.

**Ruled (Chief, 2026-09-05), and my requested fix was the wrong way round.** I asked for
exclusion first and no change to the locator, on the strength of the thirteen-of-fifteen figure
that is corrected above. At the true rate exclusion alone could leave too few cases to read. The
ruling therefore changes the locator — each value matched within its own field's span in list
order, so the k-th number under a label has a position by construction and subtotal arithmetic is
never a candidate — and makes exclusion the fallback for the residue, skipped with a per-case
reason, with scored and skipped counts reported and the section refusing only below a minimum.
Written into `SPEC-004 §5a` as an R30 amendment. The skip path should use the shape the primary
already has, `excluded_cases`, `flip_unsatisfiable_cases` and `visibility_excluded_cases`, rather
than invent a fourth.

## 5. What this does and does not say about the science

It says nothing about whether the primary's finding generalises. That question is untouched: no
`aggregate_report` case was scored. The framing agreed before the run stands unchanged — five
cases in one family with a Wilson interval of 0.38 to 0.96 is indicative, and the test of
generalisation is still owed.

What it does say is that the reproduced primary is exact, which is worth having: the strict-scoring
result of 2026-09-05 reproduces cell for cell on a later commit, so nothing in the intervening
work disturbed it.

## 6. Two record defects the Deputy raised, both of which I back

**The case count: "ten" is stale in four documents, and fifteen is right.** The Deputy asked
whether fifteen selected against a pre-registered ten is a discrepancy in the run. It is not; it
is a stale figure in the records. The ten predates R30's HEAD-alone eligibility basis, and the
Deputy's own role document records the change at line 141: "HEAD-alone eligibility, n = 2 vs 15".
Fifteen is the HEAD-alone population. "Ten cases" survives in
`ROLE-HEAD-OF-INTERPRETABILITY.md:93` (mine, now corrected),
`ROLE-DEPUTY-CHIEF-AI-RESEARCH.md:137`, `live/PROBES-READINESS-CHECKLIST.md:149` and
`under_review/P6-RESULT-REVIEW-round2-2026-09-05.md:68`. The run order and this reading were
written against a number that changed when the ruling did.

**`status=ok` on a run that did not do its job.** The Deputy's point and it is right. This run
existed to score the secondary, did not score it, and finished clean. A reader scanning logs sees
success; the record says the run succeeded because nothing raised, not because anything happened.
That is the same defect in a different dress as the structural zero we fixed on the sweep, where
an artifact reported a number that was an absence. **Requested:** a run whose named condition
produces no scores exits non-zero and says so on its `end` line. The condition's `error` field is
already there and honest; only the exit status and the log line are lying.

## 7. Cost

One lift, 1:21:17, and the reason it was granted is the part that did not run. The rerun after
the fix is the same cost again. I would rather say that plainly than let the reproduction stand
in for the result.

— Head of Interpretability
