# R25 — P6 treatment alignment slice

Date: 2026-09-04. Base: `ac9c27a` ("Locate P6 position groups over the windowed prompt with
boundary-merge repair (issue #29)") on `codex/agent-v2-specs`. Worktree:
`/Users/daniel.tipton/Desktop/An app/.claude/worktrees/agent-ae5774554efd587eb` (detached at
`ac9c27a` + this slice, uncommitted). Files changed: **only**
`src/local_llm_lab/probes/patch.py` (+396/-19) and `tests/test_patch.py` (+678/-35).
`capture.py` was **not** touched. No model weights loaded; one tokenizer load in the
skip-guarded smoke test. Nothing written under `outputs/`, `data/`, `reports/`, or
`design_specifications/pending/`.

R25's canonical text is issue #28's last comment, read verbatim; the wiring map's §7 copy is at
`design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md:583-598`. Governing documents
read: `01-IMPLEMENTER-BRIEFING.md` (rule 1.5 float32 `:...` §1; rule 1.7 banned constants; trap
6 §4), `02-INTERFACE-AND-WIRING-MAP.md:535-598` (R19, R22, R23, R24, R25),
`SPEC-004-probe-program-revision.md` §5, `capture.py:248-258` (the `InjectionHook` constraint
R25 is grounded in), `.codex/coordination/CURRENT.md`,
`.superpowers/sdd/2026-09-04-spec-004-s5-p6-patching/`, and
`under_review/P6-POSITION-GROUPS-FIX-REPORT.md`.

---

## 1. What changed and why

`InjectionHook(replace=True)` maps source row *i* to `at_positions[i]` and refuses unequal
counts (`src/local_llm_lab/probes/capture.py:248-258`, the `replacement row count must match
ordered at_positions` check). On all five selected cases the counterfactual note is one value
longer than the failing note, so `previous_notes` and `note_value_tokens` had unequal
cardinality and the run aborted at the pre-slice guard
(`patch.py:964` on `ac9c27a`, kept at `:1298` as a defensive post-alignment invariant: `treatment source and target group cardinality must match`).

The slice adds a **pairwise alignment seam** between "resolve groups on one prompt" and "inject".
`position_groups` still resolves the six per-prompt groups; a new `align_groups` matches the
counterfactual (source) prompt's groups onto the failing (target) prompt's and produces the
seven R25 cells.

| clause | implementation | file:line |
| --- | --- | --- |
| (a) tail alignment for any unequal group | `_tail_alignment` | `patch.py:924-951` |
| (a) leading residue recorded as `unpatched_source_tokens` (count + token ids + text) | `_residue_record`, `GroupAlignment.residue`, `GroupAlignment.record()` | `patch.py:912-921`, `:869`, `:881-900` |
| (a) equal-cardinality groups unchanged (`rule="identity"`, empty residue) | `_tail_alignment` split==0 branch | `patch.py:940-951` |
| (b) `shared_value_tokens`, paired by string identity via `_note_values` | `_value_alignment` shared/dropped partition | `patch.py:970-971` |
| (b) equal cardinality by construction — asserted | per-value length check raising a named error | `patch.py:977-983` |
| (b) `dropped_value_slot`: mean-pool source rows to ONE row | `_alignment_rows` | `patch.py:1112-1128` |
| (b) float32 pooling (briefing rule 1.5); pooled row shaped like one residual row | `mx.mean(... .astype(mx.float32), axis=0)` then `mx.stack` | `patch.py:1124-1128` |
| (b) slot = separator immediately after the preceding shared value | `_value_alignment` | `patch.py:1018-1020` |
| (b) fallback when the dropped value would have been first | `before_first_shared_value` branch | `patch.py:1013-1017` |
| (b) record slot position, overwritten token (id + text), pooled source tokens, dropped value, and which rule fired | `slots` entries | `patch.py:1036-1049` |
| (b) several dropped values patched in one injection, in source order | `pooled`/`positions` lists, one row per slot | `patch.py:1005-1053` |
| (c) controls resample to post-alignment cardinality | `_random_control_pair(cardinality=len(target))` with `target = alignment.target_positions` | `patch.py:1293-1295`, `:1313-1323` |
| (c) treatment source exclusion set covers pooled rows | `GroupAlignment.treatment_source_positions` | `patch.py:872-879`, used at `:1317` |
| (c) `random_positions` draws one non-treatment position per dropped value | falls out of `cardinality=len(target_positions)` | `patch.py:1319` |
| (c) refusal kept when the pool cannot satisfy | `_random_control_pair` untouched | `patch.py:1157-1173` |
| (d) `POSITION_GROUPS` = seven cells | module constant | `patch.py:50-66` |
| (d) heat map / R24 record / five-case caveat unchanged in structure | `render_markdown` unchanged except one appended section | `patch.py:1379-1413` |
| (d) every cell's artifact entry records rule, both cardinalities, residue and slot details | `cells[...]["alignment"]` per task, and `cases[i]["alignment"]` | `patch.py:1333-1340`, `:1264-1271` |
| schema field | `ARTIFACT_SCHEMA = "p6-patch-r25"`, emitted as `artifact_schema` | `patch.py:68-69`, `:1346` |

Supporting changes:

- `PROMPT_GROUPS` (`patch.py:40-48`) names the six groups a tokenizer can resolve on **one**
  prompt. `position_groups` and `_groups_for` are stated against it; `POSITION_GROUPS` is now
  strictly the seven cells, because `shared_value_tokens` and `dropped_value_slot` do not exist
  until both prompts are seen.
- `position_groups` gains a `value_spans=` out-param (`patch.py:522`, filled at `:579-581`)
  giving each note value its own token positions in note order — the input the identity pairing
  needs. `_groups_for` now returns `(groups, repairs, value_spans)` (`patch.py:814-846`).
- `_alignment_table` renders the new "## Alignment (R25)" section of `patch.md`
  (`patch.py:1400-1404`, `:1415-1445`).
- `_decoded` (`patch.py:902-909`) is a guarded `tokenizer.decode`, so a tokenizer fake without
  `decode` records token **ids** and an empty text rather than crashing the artifact.

### Coordinator addition 1 — word-boundary value matching (issue #29)

`position_groups` located a note value with `region.find(value)` and refused when the substring
occurred twice; `32` inside `132` was therefore an "ambiguous" abort. `_value_pattern`
(`patch.py:449-458`) matches with lookarounds excluding a neighbouring digit, decimal point or
sign — `(?<![0-9.\-])…(?![0-9.])` — and `position_groups` requires exactly one match
(`patch.py:563-566`). A genuine duplicate still matches twice and is still refused.
Test: `test_note_values_match_on_word_boundaries_not_substrings`
(`tests/test_patch.py:2438-2455`), which also asserts the duplicate refusal survives.

### Coordinator addition 2 — random-control pool arithmetic

`_random_control_pair` itself is unchanged (`patch.py:1157-1173`); only its inputs move to
post-alignment values. `test_random_control_pool_arithmetic_is_pinned_for_every_cell`
(`tests/test_patch.py:2468-2531`) recomputes the draw independently from
`random.Random(f"{seed}:{task_id}:{layer}:{group}")` for **all seven cells** on the smallest
fakes, asserts disjointness from the treatment on both sides, asserts
`dropped_value_slot` draws exactly one position per dropped value, and re-pins the
`random control candidate pool cannot satisfy treatment cardinality` refusal.
`tests/test_patch.py` was run bare three times: exit 0, 0, 0.

### Coordinator addition 3 — `_token_span`

Untouched. The slice patch applies cleanly onto the committed `ac9c27a` (verified below), and
the guard and `_ParityTokenizer` test that commit added are intact
(`patch.py:487-494`, `tests/test_patch.py:2189-2222`).

---

## 2. The slot rule, spelled out, on the real case 0031

Rendered note structure (`pipeline/tasks.py:1034`, `_join` at `:637-638`): the ledger note lists
approved amounts as `", ".join(values)` inside `approved: <list>;`, so between two adjacent
values stands `", "` and after the last one stands `";"`. The rule is stated in **token** terms,
which is what the injection needs:

- **preceding shared value** = the last value, in the *counterfactual note's* order, that occurs
  before the dropped value and is also present in the failing note.
- **separator position** = `target_positions_of(preceding_shared_value)[-1] + 1` — the single
  token right after that value's last token in the failing prompt.
- **fallback** (dropped value would have been first, no shared value precedes it) =
  `target_positions_of(first_shared_value)[0] - 1`; the record's `slot_rule` says which fired
  (`after_preceding_shared_value` / `before_first_shared_value`).

Measured on `test-ledger_reconcile-0031-clean` with the cached 3B tokenizer only (no weights),
decision step 7, failing prompt 2287 tokens, counterfactual 2291:

- failing note: `… approved: 25, 122; held (skip): 144, 123. …` → values `25` @ (1713, 1714),
  `122` @ (1717, 1718, 1719).
- counterfactual note: `… approved: 25, 122, 85; …` → adds `85` @ (1722, 1723).
- shared = `["25", "122"]`; dropped = `["85"]`; preceding shared value = `122`.
- **slot = 1719 + 1 = 1720**, whose target token id is `26`, decoding to **`";"`** — exactly the
  separator that would have become `", "` had 85 been listed.
- pooled source = mean of the counterfactual residual rows at 1722 and 1723 (`85`), one row,
  float32.

The seven cells on that case (source, target, patched, residue):

| cell | rule | source | target | patched | residue |
| --- | --- | ---: | ---: | ---: | ---: |
| system_prompt | identity | 359 | 359 | 359 | 0 |
| task_prompt | identity | 69 | 69 | 69 | 0 |
| previous_notes | tail | 663 | 659 | 659 | **4** (`"Plan: list the"`) |
| shared_value_tokens | shared_value_identity | 7 | **5** | 5 | 0 |
| dropped_value_slot | dropped_value_slot | **2** | 1 | 1 | 0 |
| last_two_observations | identity | 967 | 967 | 967 | 0 |
| final_token | identity | 1 | 1 | 1 | 0 |

Note on the brief's phrasing: the task brief said "the shared values (expect 5) and the dropped
values (expect 2)". Read as **token** counts those are exactly right (5 shared value tokens, 2
pooled source tokens for the one dropped value). Read as **value** counts they are not: case
0031 has 2 shared values and 1 dropped value. The tests pin both readings explicitly
(`tests/test_patch.py:2113-2159`).

All five selected cases align; `previous_notes` is `tail` on every one, every case has exactly
one dropped value with `slot_rule == after_preceding_shared_value`, and the overwritten
separator is `";"` on four cases and `","` on `…-0175-clean` (where the dropped value `54` sat
between `27` and `49`). The smoke test asserts that distribution
(`tests/test_patch.py:2166-2201`).

The residue on `previous_notes` is `"Plan: list the"` — the **leading** tokens of the *first*
note in the concatenated group, not the extra value. That is the intended consequence of R25(a)
("the decision-adjacent context is the note's end"): the four surplus source tokens are shed
from the front of the block, so the substituted note's own tokens keep near-identical absolute
offsets. Worth the Chief's eye: it means the `previous_notes` cell patches a source block whose
first note is truncated by four tokens.

---

## 3. Red-first evidence

The thirteen new tests were written and run against `ac9c27a` **before** any change to
`patch.py`. Command: `uv run pytest -p no:cacheprovider tests/test_patch.py -k "align or seven
or slot or pooled or tail or shared or dropped or word_boundaries or control_pool"` →
**13 failed, 45 deselected**. Failure text per test:

| test | failure on the fixed-but-unaligned state |
| --- | --- |
| `test_run_patch_probe_aligns_the_unequal_note_groups_end_to_end` | `ValueError: treatment source and target group cardinality must match` (`patch.py:964`) |
| `test_pooled_dropped_value_row_is_the_float32_mean_of_the_source_rows` | same, `patch.py:964` |
| `test_controls_resample_to_the_post_alignment_cardinality` | same, `patch.py:964` |
| `test_note_values_match_on_word_boundaries_not_substrings` | `ValueError: note_value token span is missing or ambiguous in the substituted note` (`patch.py:529`) — the issue-#29 abort, reproduced |
| `test_position_groups_names_seven_cells_including_the_split_value_cells` | `AssertionError` on the six-cell `POSITION_GROUPS` |
| `test_alignment_refuses_a_source_group_shorter_than_the_target` | `AssertionError: Regex pattern did not match` (no refusal raised at all) |
| `test_alignment_refuses_when_no_value_was_dropped` | `AssertionError: Regex pattern did not match` |
| the six remaining unit tests (`tail`, `shared`, `slot`, `first-value`, `two-slots`, `control_pool`) | `ValueError: not enough values to unpack (expected 3, got 2)` from `_groups_for`, i.e. the value-span input R25 needs did not exist |

Three of the thirteen produce the exact live error text the Director hit; the remaining ten
fail because the seam R25 prescribes was absent, which is the honest pre-slice signal.

New tests (`tests/test_patch.py`), all fake-only (R10) except the skip-guarded smoke test:

| requirement | test | lines |
| --- | --- | --- |
| seven cells; six per-prompt groups unchanged | `test_position_groups_names_seven_cells_including_the_split_value_cells` | `:2293-2316` |
| (a) tail residue count and content; identity groups untouched | `test_tail_alignment_takes_the_source_tail_and_records_the_leading_residue` | `:2318-2347` |
| (b) identity pairing when source order differs (`["57","62","41"]` vs `["41","57"]`) | `test_shared_value_tokens_align_by_string_identity_when_order_differs` | `:2349-2374` |
| (b) slot rule + full record | `test_dropped_value_slot_is_the_separator_after_the_preceding_shared_value` | `:2376-2403` |
| (b) first-value fallback and its recorded rule | `test_dropped_first_value_uses_the_separator_before_the_first_shared_value` | `:2405-2417` |
| (b) two dropped values, two distinct slots, source order | `test_two_dropped_values_get_two_slots_in_source_order` | `:2419-2436` |
| word-boundary values (issue #29) + duplicate still refused | `test_note_values_match_on_word_boundaries_not_substrings` | `:2438-2455` |
| refusals: source shorter than target; no dropped value | `test_alignment_refuses_a_source_group_shorter_than_the_target`, `test_alignment_refuses_when_no_value_was_dropped` | `:2457-2466` |
| (c) control pool arithmetic pinned for all seven cells | `test_random_control_pool_arithmetic_is_pinned_for_every_cell` | `:2468-2531` |
| end-to-end through `run_patch_probe`; seven cells in the aggregate; cell + case artifact entries; JSON-safe; markdown section | `test_run_patch_probe_aligns_the_unequal_note_groups_end_to_end` | `:2595-2623` |
| pooled row = float32 mean of its source rows, one row per slot | `test_pooled_dropped_value_row_is_the_float32_mean_of_the_source_rows` | `:2625-2639` |
| (c) controls at post-alignment cardinality, both dropped values | `test_controls_resample_to_the_post_alignment_cardinality` | `:2641-2668` |

Fakes: `_value_messages` (`:2225-2250`) builds a replayed ledger case whose last note approves
exactly the given values (earlier notes fixed, so the two message lists differ only in the
substituted note — the R22b construction); `_aligned` (`:2252-2291`) runs the real
`build_prompt` / `window_messages` / `position_groups` / `align_groups` over
`_MergingTokenizer`; `_alignment_probe` (`:2534-2592`) drives `run_patch_probe` with a recording
`InjectionHook` and residual rows carrying their own position index, so a pooled row is
checkable arithmetically.

Existing tests updated for the new signatures and cell set:
`test_position_groups_are_exact_and_fail_closed` (`:756`, `PROMPT_GROUPS`),
`test_groups_for_extracts_canonical_family_value_tokens` (`:813-822`, three-tuple + per-value
spans assertion), `test_groups_for_uses_the_windowed_view_and_char_offsets_with_a_merging_tokenizer`
(`:1885-1891`), `test_groups_for_windows_with_the_threaded_keep_last` (`:1978-1980`),
`test_patch_probe_routes_every_group_and_control_with_exact_trace` (`:1232-1505`, rewritten:
fakes `_groups_for` instead of `position_groups`, and the whole injection trace is now written
out against the hand-computed R25 alignment for the seven cells),
`test_patch_probe_rejects_unequal_treatment_group_cardinality` → renamed
`test_patch_probe_refuses_a_source_group_shorter_than_the_target` (`:1507`, because unequal is
now *aligned*, and only a shorter source is refused), `_probe_fixture` (`:1610-1620`),
`test_real_tokenizer_resolves_the_six_groups_on_the_first_selected_case` → renamed
`test_real_tokenizer_resolves_and_aligns_the_seven_cells_on_the_first_case` (`:2062-2201`).

---

## 4. Verification

- `uv run pytest -p no:cacheprovider`, run bare, twice: **exit status 0 both times, 666 passed,
  1 skipped** (baseline `ac9c27a` in this worktree: 652 passed, 2 skipped → 654 collected; this
  slice adds 13 tests → 667 collected). The one skip is
  `tests/test_tasks.py:254` — `data/chat_replay` is absent from the isolated worktree; it is
  environmental, pre-existing, and unrelated to this slice (I was instructed not to write under
  `data/`). The `test_patch.py` smoke test **ran** (not skipped): I temporarily symlinked
  `outputs/` and `.cache/huggingface` at the main tree read-only so the protected evaluations
  and the cached tokenizer resolved; both symlinks were removed before the patch was produced,
  and neither is in the diff.
- `tests/test_patch.py` alone, bare, three times: exit 0, 0, 0 (58 passed each).
- Banned-constant scanner: `uv run pytest tests/test_repository_rules.py` → 10 passed. Grep over
  `patch.py` for `20260902|torch|<|im_end|>|\b36\b|\b35\b|\b2048\b|Qwen|qwen` returns only the
  pre-existing registry **name** `--model default="qwen25-coder-3b"` (`patch.py:1482`), which is
  a registry key, not a model constant. No seed literal was added: `--seed` keeps its CLI
  default and every new draw goes through the existing `f"{seed}:{task_id}:{layer}:{group}"`
  derivation. No `torch`, no new dependency.
- Lint: `uv run ruff check` reports **20 errors on `ac9c27a` and 20 after this slice** — the two
  `B905` findings the slice introduced were fixed; the pre-existing 20 (and the pre-existing
  `ruff format` drift on both files) are untouched, per briefing §6 "do not reformat files you
  are not changing".
- Slice patch verified: cloned the worktree to a scratch checkout of `ac9c27a` and ran
  `git apply --check R25-alignment.patch` → **applies cleanly**; applying it reproduces
  `396/19` + `678/35`.

---

## 5. Deviations from R25 and decisions taken (briefing rule 1.10)

Flagged explicitly; none contradicts R25, all are points R25 left open.

1. **Source shorter than target is refused, not cycled.** R25(a) says "the last |target| source
   positions patch the target", which presupposes `|source| >= |target|`. When it is not, the
   only alternatives are `_match_rows`-style cycling (invents rows) or truncating the target
   (silently drops positions). I fail closed with
   `"<group>: treatment source group has N positions, fewer than the target's M; tail alignment
   cannot match cardinality"` (`patch.py:934-939`). Does not fire on any of the five cases.
2. **Two dropped values resolving to the same slot are refused.** R25 says each slot gets its own
   pooled row but does not say what happens when two dropped values share a preceding shared
   value (adjacent drops). `InjectionHook` de-duplicates `at_positions`, so writing two rows to
   one position is not expressible. I raise a named error (`patch.py:1026-1031`). Does not fire
   on any of the five cases (each has exactly one dropped value). **Open question for the Chief:
   if the real run ever produces adjacent drops, should they pool into one row at one slot, or
   should the second take the next separator?**
3. **A case with no shared value at all is refused** (`patch.py:972-976`): the slot rule is
   defined only relative to a shared value, and both R25 branches need one.
4. **A case with no dropped value is refused** (`patch.py:995-999`), because
   `dropped_value_slot` would be an empty cell and `_groups_for`'s "at least one token" rule
   applies to every cell. Selection guarantees a value drop, so this is a fail-closed guard.
5. **Cell ordering of `shared_value_tokens`.** Pairs are emitted in the *counterfactual note's*
   value order (`patch.py:986-987`), not sorted by target position, because that is the order
   the "preceding shared value" rule is defined against. The pairing is what the hook needs;
   ordering is not otherwise load-bearing.
6. **Artifact schema field.** The payload carried no schema or version field, so nothing was
   bumped. Per the instruction not to invent versioning beyond a field name I added exactly one
   name — `artifact_schema: "p6-patch-r25"` (`patch.py:68-69`, `:1346`) — with no comparison,
   parsing or migration logic anywhere. If the Chief prefers no new field, deleting those two
   lines is the whole change.
7. **"Update `aggregate_report`"** in the task brief: `aggregate_report` is a *task family* name
   in this codebase, not a function. I read the clause as the artifact aggregate and updated
   `payload["groups"]` (already derived from `POSITION_GROUPS`, `patch.py:1354`), the per-cell
   records, and `render_markdown`'s tables. `aggregate_task_flips` needed no change. Grep
   confirms `POSITION_GROUPS` / `note_value_tokens` have no consumer outside `patch.py` and
   `tests/test_patch.py` (`docs/superpowers/plans/2026-09-04-spec-004-s5-p6-patching.md:27,71-75`
   names the old six groups in prose; it is a plan document and I did not edit it — **observed,
   not fixed**).
8. **`capture.py` untouched.** The pooled cell composes with the existing hook exactly as the
   brief preferred: a group is positions + rows, and a pooled row is one row, so
   `at_positions=(slot,)` with a `(1, d)` array is already expressible.
9. **The brief's case-0031 numbers** ("shared values 5, dropped values 2") are token counts, not
   value counts; see §2. Reported rather than silently reconciled.

## 6. Observed, not fixed

- `docs/superpowers/plans/2026-09-04-spec-004-s5-p6-patching.md:27` and `:71-75` still describe
  the six-group `POSITION_GROUPS`. Stale after R25.
- `tests/test_tasks.py:254` skips in any isolated worktree because `data/` is not checked out.
  Not a defect of this slice, but it means "0 skipped" is unreachable from a worktree.
- `render_markdown`'s new alignment table repeats identical rows across layers if a reader
  expects per-layer variation; alignment is layer-independent by construction, so the table is
  keyed by (task, cell) only. Stated here so the Chief can rule on the cell-level duplication in
  `cells[layer:group]["alignment"]`, which is the literal reading of R25(d) ("every cell records
  the alignment rule and cardinalities") and costs five small dicts per cell.

## 7. Open questions for the Chief

1. Adjacent dropped values (deviation 2) — pool into one slot, or advance to the next separator?
2. `previous_notes` tail alignment sheds the **first note's** leading four tokens
   (`"Plan: list the"`), not the substituted note's. Confirm that is the intended reading of
   "the decision-adjacent context is the note's end" for a *concatenated* multi-note group, or
   should the tail be taken per note?
3. Keep or drop the `artifact_schema` field (deviation 6).
4. `shared_value_tokens` source cardinality is recorded as the whole `note_value_tokens` count
   (7 on case 0031), not the shared subset (5). That is what makes the drop visible in the
   artifact; confirm it is the cardinality R25(d) wants recorded, or should it be the
   post-alignment 5/5?
