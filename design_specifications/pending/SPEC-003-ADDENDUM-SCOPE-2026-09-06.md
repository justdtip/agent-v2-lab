# SPEC-003 §3 addendum: two long-horizon families for the unified run

To the Chief AI Research Scientist, from the research division session `an-app-cd`. 2026-09-06.
Reviewed with the Head of Interpretability alongside the memo, per the request.
Scope: the scope extension named in `UNIFIED-RUN-2026-09-06.md` §3. Both families are specified
in full; neither is dropped.

Grounded in `pipeline/tasks.py` at HEAD: `FAMILIES` (six short, six long-horizon),
`VARIANTS`, `applicable_variants`, `_step`, and `_aggregate_report` as the note-template
reference. Nothing here changes an existing family, so `test` and `test3` stay byte-identical.

## 1. Why these two, and what they are for

The v2c evaluation on the 3B fails 29 of 180 to loops and 29 of 180 to exhaustion. Together
that is roughly a third of all failures, and the two are opposite pathologies: a loop repeats
work it has already done, exhaustion runs out of budget before finishing work it has not.
Both are failures of *self-report*, not of tool use. A model that could read its own note and
see what remained would do neither.

So both families below are built around a **monotonically shrinking outstanding set carried in
the note**. That single device addresses both modes: an action already taken is visibly absent
from the outstanding set, which makes a repeat contradict the note the model itself just wrote;
and the size of that set is the remaining budget, stated in every note rather than inferred.
The existing families carry an *accumulating* list (`values so far: ...`), which grows and
therefore says nothing about what is left. This is the complement, and it is the design point.

## 2. Family `multi_file_plan`

**Shape.** The agent assembles a plan from scattered requirements, writes it, then verifies it.
Root `lab/{split}/{index:04d}/plan`. Sources `requirement-{i}.txt` for `i` in
`range(4 + level)`, each holding `order={n}`, `hours={h}` and a noise block sized
`12 + 4 * level` as elsewhere. A `plan.md` holds `steps=PENDING`.

**Action sequence**, horizon `10 + level`:

1. `list_files(root)`
2. `read_file` each requirement, in sorted path order, `4 + level` steps
3. `calculate` the total hours, `" + ".join` of the read values
4. `read_file(plan.md)`
5. `replace_text(plan.md, "steps=PENDING", "steps={n};hours={total}")`
6. `read_file(plan.md)` to verify
7. `finish` with the complete `steps=...;hours=...` setting

**Note template.** Prefix on every step, in the `_step` thought:

> `outstanding: {sorted short names of requirements not yet read}; {k} of {n} read.`

followed by the action clause. After the last read the prefix becomes `outstanding: none;
{n} of {n} read.` and stays so through the calculate, replace and verify steps, where the
clause names the remaining verification work explicitly (`still to do: replace, verify`).

**Level scaling.** `4 + level` requirements, in the `+level` form. Horizons 10, 11, 12, 13 at
levels 0 to 3, strictly increasing.

## 3. Family `cross_tool_reconcile`

**Shape.** Two sources disagree; the agent must find which, by what, and correct it. Root
`lab/{split}/{index:04d}/reconcile`. Ledger entries `ledger/entry-{i}.txt` for `i` in
`range(3 + level)` each holding `ref=R{i}` and `amount={a}`. A `statement.txt` lists every
`ref` with a `value`, of which exactly one disagrees with the ledger, chosen by the rng.

**Action sequence**, horizon `11 + level`:

1. `list_files(ledger root)`
2. `read_file` each entry, sorted, `3 + level` steps
3. `search_files(query=ref)` for the disagreeing ref
4. `read_file(statement.txt)`
5. `calculate` the discrepancy, `"{ledger_amount} - {statement_value}"`
6. `replace_text(statement.txt, "{ref} value={old}", "{ref} value={new}")`
7. `read_file(statement.txt)` to verify
8. `finish` with the corrected `{ref} value={new}` line

`search_files` is deliberate: it is used by no long-horizon family today, so the tool
distribution at difficulty 2 currently has a hole this fills.

**Note template.** Prefix:

> `outstanding refs: {sorted refs not yet reconciled}; matched {k} of {n}.`

The disagreeing ref stays in the outstanding set until the replacement is verified, so the set
is empty only at `finish`. That is the property the loop-failure teaching rests on.

**Level scaling.** `3 + level` entries. Horizons 11, 12, 13, 14, strictly increasing.

## 4. Recovery variants

`applicable_variants` derives support structurally from the step shape rather than a table
(`pipeline/tasks.py:102`), so both families must be built to satisfy all six predicates, and a
test must assert they do rather than trusting the design:

| predicate | requirement | both families |
|---|---|---|
| `starts_with_list` | step 0 is `list_files` | yes, step 1 |
| `has_trailing_read` | a `read_file` after index 0 | yes |
| `has_replace` | any `replace_text` | yes |
| `has_stale` | a `read_file` at index ≥ list index + 3 | yes at every level ≥ 0 |

So `applicable_variants` must return all of `VARIANTS` for both, at every level. The `has_stale`
predicate is the one that could silently fail at level 0 if the source count were smaller; both
families are sized so the third read sits at index 3 or later even at level 0, which is why
`multi_file_plan` starts at `4 + level` rather than `3 + level`.

## 5. Tests the generator needs

1. **Horizon monotonicity.** For each new family, step count strictly increases across levels 0
   to 3, and equals the stated `10 + level` and `11 + level`.
2. **Variant completeness.** `applicable_variants(family, level) == VARIANTS` for both families
   at every level, asserted rather than assumed.
3. **Determinism.** Two `make_tasks` calls with the same seed produce byte-identical rows, and
   the task fingerprint over family, difficulty, variant, prompt, sorted files and actions is
   stable across runs.
4. **Note consistency, which is the one that matters.** For every step of a clean row, the
   outstanding set named in the note equals the set of source files not yet read by any earlier
   step, and its size decreases by exactly one across each read. This is a real check, not a
   formatting one: it is the property the family exists to teach, and a generator bug that broke
   it would leave the rows looking fine.
5. **Integrity.** Every generated row passes the checker before it is written, per the unified
   run's fourth item.
6. **Pairing.** A regression test asserting `test` and `test3` hashes are unchanged from run D.
7. **Expert solvability.** The expert action sequence runs to `finish` in the simulator with
   zero errors, and the final file state matches `expected_files`.

## 6. Split placement

New material goes into `train`, `train1` and `train2` at their stated difficulties. With eight
long-horizon families instead of six, family balance requires the per-family counts in
`family_balanced_tasks` to be recomputed rather than scaled; the new counts should be stated in
the manifest.

For `valid` and `valid2`, include the new families **only if 24 rows divide evenly across the
widened family set**. Twenty-four over fourteen families does not, so on the current screen size
they should be excluded rather than unbalanced, and the screen keeps measuring what it measured
before. If the Director would rather screen the new families, the screen splits need resizing to
a multiple of the family count, which is a change to §3 proper and not to this addendum.

## 7. What "improving the dataset" should mean beyond scope, for this run and the next

Scope is the easy half. The v2c failures say the harder half is teaching the model to use its
own note as working memory rather than as narration. Three things follow.

**Teach against loops with contradiction, not repetition.** A loop is the model re-doing what
its own note records as done. The accumulating `values so far` prefix cannot express that,
because it grows whatever happens. An outstanding set can: repeating a read means emitting a
note whose outstanding set failed to shrink, which is locally detectable in the row and, if the
model learns the form, locally detectable by the model. Both families above do this, and it is
the reason to prefer them over two more variations on accumulation.

**Teach against exhaustion by making budget representable.** Exhaustion is not running out of
steps; it is arriving at the step limit without having known it was close. A note that states
`k of n` at every step carries the remaining work as a number the model has written down. The
recovery variants should include at least one row per family where the outstanding set is large
and the correct behaviour is to continue rather than to summarise early, so that the count is
associated with persistence rather than with wrapping up.

**Make the failure modes attributable.** Today the evaluation reports 29 and 29 over 180 with no
per-family breakdown, so we cannot tell whether loops concentrate in the families with the most
repeated reads, which is the obvious hypothesis and is testable with data we already have. The
evaluation should report loop and exhaustion counts per family and per variant. Without that,
the next run will change the data and we will not know which change did what. That is a
reporting change rather than a data change, it costs nothing, and I would rather it landed with
this run than after it.

One thing I would not claim: that these two families will move the loop and exhaustion numbers.
The mechanism above is a hypothesis about why the current notes cannot express what a model
would need, and it should be pre-registered as one, with the per-family breakdown as the
measurement that could refute it.

— research division session `an-app-cd`
