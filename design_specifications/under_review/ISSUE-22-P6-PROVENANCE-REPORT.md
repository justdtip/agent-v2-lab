# Issue 22 — P6 provenance micro-slice (R22), rounds 1–3

Date: 2026-09-04. Implementer slice covering `src/local_llm_lab/probes/patch.py` and
`tests/test_patch.py` only. No commit; working tree only, pending R19 review and the
Chief's gate on issue 22. The saved evaluations under `outputs/` were read but never
written. Round 2 adapted the tool to the pre-schema evaluations (missing `integrity`/
`difficulty`); round 3 version-bound that recomputation through the R12 replay path and
restricted selection to SPEC-004 §5's case universe. Line numbers in the round 1–3
sections cite the working tree as re-verified after round 3; the K1 section below
inserted ~180 lines into `patch.py` (everything from `_VALUE_DROP_DETAIL` onward shifted
by up to +140) and cites the tree after K1. `_recomputed_value_drop` and
`_value_drop_step` named in round 3 became `_recomputed_judgement` and `_saved_judgement`
(K1 §1).

## R22a — `--data-seed` override (round 1)

- CLI flag: `src/local_llm_lab/probes/patch.py:835-840`.
- Resolution in `_records` (`patch.py:59-90`): evaluation field wins when present
  (`patch.py:84-90`, source `evaluation`); a present field that disagrees with the
  override errors naming both values (`patch.py:86-89`); an absent field uses the
  override with source `flag`, and no override fails closed (`patch.py:77-82`);
  non-integer field or override are errors (`patch.py:68-69`, `patch.py:84-85`).
- The artifact records `data_seeds` per input file as `{"data_seed": <int>,
  "data_seed_source": "evaluation"|"flag"}` (built at `patch.py:204-207`, written at
  `patch.py:897`).
- Seed-literal scope (R16 correction, twice-flagged): the earlier sentence "appears
  nowhere in source" was wrong — the pipeline's own `make_tasks` default carries the
  seed at `src/local_llm_lab/pipeline/tasks.py:99`. The accurate claim, scoped to the
  owned files as originally required: `20260902` appears nowhere in
  `src/local_llm_lab/probes/patch.py` (grep count 0) and exactly once in
  `tests/test_patch.py:535`, as fixture data in the real-files pin — the run's recorded
  configuration, not a constant the tool uses — per the slice's explicit allowance for
  fixture data in tests.

## R22b — counterfactual note provenance (round 1)

- `counterfactual_note` (`src/local_llm_lab/probes/patch.py:440-484`):
  - PRIMARY: the passing run's saved note at step `decision_step - 1` from the loaded
    evaluation, used when the passing trajectory's actions (name and arguments, via
    `_action`, `patch.py:430-437`) equal the failing run's at every step before the
    decision (`patch.py:469-475`); basis recorded verbatim (`patch.py:481-483`).
  - FALLBACK: `render_expert_note` at HEAD, tagged `generator_v<GENERATOR_VERSION>`
    (`patch.py:456-460`), with the reason recorded: steps unavailable
    (`patch.py:462-463`), prefix too short (`patch.py:464-467`), missing action
    (`patch.py:472-473`), divergence at a named step (`patch.py:474-475`), no saved
    note text (`patch.py:476-478`).
- `PatchCase.passing_steps` (`patch.py:56`), populated from the passing record's steps
  when well formed (`patch.py:263-268`).
- `replay_counterfactual` returns `(failing, counterfactual, provenance)`
  (`patch.py:487-513`).
- Artifact: per-case `{task_id, decision_step, counterfactual_source,
  counterfactual_basis}` under `payload["cases"]` plus the count summary
  (`patch.py:694-706`, `patch.py:782-783`); markdown section (`patch.py:805-808`).

## Round 2 — eligibility recomputation for pre-schema evaluations

Both saved evaluations predate per-trajectory `integrity` and `difficulty` (verified
read-only: step records carry `action/index/observation/raw/thought`; trajectories
carry `task_id`, `verdict`, `family`, `variant` — no `integrity`, `difficulty`, or
`data_seed`), so selection found zero cases. The tool adapts, never the evidence:

- `_derived_difficulty` (`src/local_llm_lab/probes/patch.py:93-104`): when the field
  is absent, difficulty derives from the task id's split and index via the pipeline's
  `difficulty(split, index)` (`src/local_llm_lab/pipeline/tasks.py:59-61`; the flagged
  off-by-one was re-verified against the live tree with `sed -n '57,62p' | cat -n` —
  `def difficulty` sits at line 59, so 59-61 stands). Fail-closed behaviour, as
  directed: a task id whose difficulty cannot be derived raises a named `ValueError`
  (`patch.py:100-104`) and aborts the entire selection via `parser.error` — a
  deliberate whole-run abort on malformed evidence, not a silent per-record skip.
- Saved values win by field presence (`"difficulty" in record`, `patch.py:231`;
  `"integrity" in record`, `patch.py:245`); recomputation provably never runs when the
  fields exist (poisoned-checker test, `tests/test_patch.py:330-367`).
- Artifact `eligibility` block, mirroring `data_seeds` (`patch.py:271-298`, written at
  `patch.py:898`): `passing.eligibility_source: "evaluation"` (the passing input
  contributes only saved task ids, verdicts, and steps); `failing` carries
  `eligibility_source: "evaluation"|"recomputed"|"mixed"` plus per-field consulted
  counts (`integrity`/`difficulty`), tallied at `patch.py:244`, `247`, `257`.

## Round 3 — version-bound recomputation and the §5 universe

The round-2 review proved the unbound recompute selected 8 artifacts: judged under the
failing run's own generation (v1 templates), 8 of 13 cases have no value drop — they
were v4-template-style judgements about v1-era notes, and exactly the 8 whose C
verdict passed. Both independent protections are now in:

1. **Version binding (R12).** The recompute builds its judging task through
   `replay_task_from_id` (`src/local_llm_lab/pipeline/tasks.py:180-195`), never
   `task_from_id` at HEAD: `patch.py:255-256`, with `_recomputed_value_drop` documented
   as requiring the version-bound task (`patch.py:141-153`). The binding resolves in
   `_generator_version_binding` (`patch.py:106-139`): the evaluation's recorded version
   wins (`summary.generator_version`, else top-level; `patch.py:121-124`), an explicit
   `--generator-version` flag (`patch.py:841-849`, threaded at `patch.py:867`) is used
   only when nothing is recorded, a conflict errors naming both values
   (`patch.py:134-138`), and when a recomputation is needed with no version from either
   source the tool fails closed with a named error rather than defaulting to HEAD
   (`patch.py:248-253`). The artifact's eligibility block records the version used AND
   the basis for it: `recomputed_generator_version`, `generator_version_source:
   "evaluation"|"flag"`, and `generator_version_basis` (`patch.py:284-291`).
   `PatchCase.task` itself remains the HEAD task — R22b's generator fallback is
   defined at HEAD and the task structure is version-invariant; only the eligibility
   judgement is replayed.
2. **§5 universe filter.** Selection now requires the failing trajectory's verdict to
   be an actual failure — SPEC-004 §5's "task ids that pass under B and fail under C"
   — via the gate at `patch.py:221-229` (`verdict.get("success") is not False` skips;
   a missing or malformed verdict is not provably failing and is excluded). No
   exploratory flag was added: value-drop-but-passed cases are simply outside the
   default universe.

### Decisions the Chief must ratify at the gate

- **The v1 attribution.** Neither the failing evaluation nor `data/agent_v2c`'s
  manifest records a generator version, so the retry's `--generator-version 1` binding
  rests on R22's own rationale (the in-repo v1 templates are run C's rewrite) and the
  v1≡v2 / v3≡v4 template structure. The tool records the mechanical basis (explicit
  flag binding, evaluation records none) in the artifact; the scientific attribution
  of v1 to run C is the Chief's to ratify or reverse.
- **The R22 extension** (carried from round 2): eligibility recomputation itself
  extends R22's letter under the adapt-not-edit principle.
- **Remaining HEAD seam, for awareness:** flip scoring at probe time (`_is_flip`,
  `patch.py:516-521`, via `_score_patch`, `patch.py:637-664`) still judges patched
  generations with `check_trajectory` over the HEAD task. For the five selected cases the
  drop exists under both v1 and v4 judgements, so scoring is coherent here; whether
  that seam should also be version-bound is left to a ruling, not smuggled in.

## Tests (per-module, R10; fakes, fixtures, and read-only evidence; no model execution)

- Round 1: seed matrix (`tests/test_patch.py:86-137`), saved-note preference
  (`test_patch.py:821-873`), fallback matrix (`test_patch.py:875-889`; unavailable
  steps at `test_patch.py:788-792`), per-case artifact provenance
  (`test_patch.py:1205-1223`), markdown summary (`test_patch.py:1295-1310`).
- Round 2: recompute-both with sources recorded (`test_patch.py:158-203`), saved
  fields win with recomputation provably not invoked (`test_patch.py:330-367`), mixed
  sources (`test_patch.py:370-428`), dry fixture shaped like the real files
  (`test_patch.py:431-508`).
- Round 3:
  - The 8-artifact exclusion pinned: a case whose drop exists only under HEAD — the
    bound-version replay finds no violation — is NOT selected, and the checker is
    asserted to receive the replayed task, never the HEAD task
    (`tests/test_patch.py:206-233`; the positive twin at `test_patch.py:158-203`
    asserts the same identity on the selecting path).
  - Version binding matrix: recorded-in-evaluation used with source and basis
    recorded, conflict naming both values, fail-closed named error when a recompute
    needs a version and none exists, non-integer binding rejected
    (`tests/test_patch.py:236-284`).
  - Verdict filter pinned: dropped-but-succeeded and verdict-less trajectories
    excluded, verdict-failed selected (`tests/test_patch.py:287-327`).
  - Real-files pin (skipped cleanly when the untracked evidence is absent):
    the exact retry selection yields **5 cases with decision steps [6, 7, 7, 7, 7]**,
    flag-sourced seed, v1-bound recomputed eligibility, and all five counterfactual
    notes from `passing_transcript` (`tests/test_patch.py:511-550`).
  - CLI threads `--generator-version` and writes `eligibility` into `patch.json`
    (`tests/test_patch.py:916-1016`, asserts at `test_patch.py:1007-1008`).

## Suite (R13)

`uv run pytest` — **639 passed** at the round-3 hand-off; **647 passed, exit status 0**
after K1 (run bare, status checked directly, not through a pipe);
`uv run pytest tests/test_patch.py` — 33 passed at round 3, 38 after K1.

## Director's retry — re-verified against the real files

```
uv run agent-v2-probe-patch \
  --passing-eval outputs/agent-v2b/evals/runB-test180.json \
  --failing-eval outputs/agent-v2c/evals/best-adapter-test.json \
  --policy C --model qwen25-coder-3b \
  --data-seed 20260902 \
  --generator-version 1 \
  --output outputs/probes/patch-C-2026-09-04
```

The selection stage of this exact command was executed read-only against the real
saved evaluations (no model load, no GPU): **5 eligible cases**
(`test-ledger_reconcile-0031/0127/0139/0163/0175-clean`), decision steps
7/7/7/7/6 — matching the reviewer's triple convergence (v1 replay, §5 verdict
universe, and SPEC-004's stated case count) — with `data_seed_source: "flag"`,
`eligibility_source: "recomputed"` (5/5 both fields),
`recomputed_generator_version: 1` (`generator_version_source: "flag"`), and all five
counterfactual notes sourced from `passing_transcript`. Without `--generator-version`
the tool now refuses with the named R12 error instead of judging at HEAD. 5 ≥ 2
satisfies the probe's unrelated-task-control minimum.

## K1 — ruling R24, `scoring_version_stable` (the Chief's commit condition)

Condition from `P6-ELIGIBILITY-REVIEW-round1-2026-09-04.md`: flip scoring (`_is_flip`,
`src/local_llm_lab/probes/patch.py:622-627`) judges the patched trajectory with
`check_trajectory` over `PatchCase.task`, built at HEAD (`task_from_id`, `patch.py:332`),
while eligibility is judged under the bound replay version. R24 admits that seam per case
only when the decision step AND the dropped-value set are identical under both.

### 1. Both judgements computed and recorded per case

- `DropJudgement` (`patch.py:52-63`): `decision_step`, `dropped_values` (a tuple, or
  `None` when the judging source records no values), `judged_under`.
- `scoring_version_stable(bound, head)` (`patch.py:66-75`): true iff both decision steps
  are set and equal AND both value sets are known and equal as sets. Unknown values on
  either side can never be "identical", so they fail closed to unstable.
- Value extraction (`patch.py:200-262`): `_dropped_values` parses the
  `"missing required values: …"` detail that `integrity._value_drop_violation` writes
  (`src/local_llm_lab/pipeline/integrity.py:238-254`); `_judgement` takes the earliest
  `value_drop` from either `Violation` objects or the saved dicts of an evaluation's
  `integrity` block (`patch.py:222-238`); `_recomputed_judgement` (`patch.py:241-253`)
  runs `check_trajectory` over a given task — the bound replay for eligibility, the HEAD
  task for the R24 check; `_saved_judgement` (`patch.py:256-262`) reads a saved block.
- `select_patch_cases`: the bound judgement is the one that selects the case — the saved
  block (`patch.py:339`) or the v-bound replay (`patch.py:349-354`, unchanged R12 path);
  after the drop is confirmed the HEAD judgement is computed over exactly the task and
  steps `_is_flip` will score (`patch.py:361-365`) and both ride on the case
  (`PatchCase.bound_judgement`/`head_judgement`, `patch.py:83-85`, appended at
  `patch.py:372-376`). The selection stage still loads no weights.
- Artifact fields per case (`PatchCase.scoring_record`, `patch.py:97-108`):
  `scoring_version_stable`, `decision_step_bound`, `decision_step_head`,
  `dropped_values_bound`, `dropped_values_head`, `judged_under_bound`,
  `judged_under_head` — the two judgements sit beside the flag so a reader sees why a
  case is unstable. `PatchCase.scoring_version_stable` (`patch.py:87-95`) raises on a
  case that carries no judgements rather than guessing.

### 2. Headline over stable cases only; unstable listed separately

- `run_patch_probe` splits the cases (`patch.py:799-800`); the unrelated-task control
  minimum now counts stable cases and names both counts in its error
  (`patch.py:801-805`); unstable cases get their counterfactual provenance and full
  scoring record under `excluded_cases` with `excluded_reason:
  scoring_version_unstable` (`patch.py:807-822`) and are never prepared, captured, or
  scored — only stable cases enter `prepared` and the cells (`patch.py:824`).
- Payload (`patch.py:904-909`): `scoring_generator_version` (HEAD's
  `GENERATOR_VERSION`), top-level counts `stable_cases` / `unstable_cases`,
  `headline_task_ids` (stable), `cases` (stable records), `excluded_cases` (unstable
  records); `selected_task_ids` still lists every selected case.
- Markdown (`render_markdown`, `patch.py:915-957`): the heat map is preceded by
  "Headline over N scoring-version-stable case(s); M unstable case(s) excluded (R24)"
  (`patch.py:923-931`); a "Scoring version stability (R24)" section tables the headline
  cases and, under its own "Unstable cases (excluded from the headline)" heading, the
  excluded ones or "None." (`patch.py:949-956`, `_stability_table` `patch.py:960-974`).

### 3. Measured on the real files (selection only, read-only, no model load)

Director's command, parsed by `main()` and driven through selection with the
policy-resolution seam stubbed to abort before the GPU guard: 5 cases, **5 stable, 0
unstable**. Per case (`decision_step_bound` / `decision_step_head`, dropped values bound
/ HEAD; `judged_under` = `generator_v1 replay` / `generator_v4 HEAD`):

| task | stable | decision step v1 / HEAD | dropped values v1 / HEAD |
|---|---|---|---|
| test-ledger_reconcile-0031-clean | true | 7 / 7 | 85 / 85 |
| test-ledger_reconcile-0127-clean | true | 7 / 7 | 89 / 89 |
| test-ledger_reconcile-0139-clean | true | 7 / 7 | 100 / 100 |
| test-ledger_reconcile-0163-clean | true | 7 / 7 | 32 / 32 |
| test-ledger_reconcile-0175-clean | true | 6 / 6 | 54 / 54 |

This matches the round-2 review's finding (decision steps identical under v1 and v4); it
is now a recorded fact per case, not a claim. The real-files pin asserts exactly these
values (`tests/test_patch.py:697-720`) beside the existing five-case assertions
(`test_patch.py:667-696`), skip-guarded on the protected evaluations as before. The
seed and version literals remain fixture data in that test only (`20260902`: 0 hits in
`patch.py`, 1 in `test_patch.py`; the version is used from `GENERATOR_VERSION` in source,
never as a literal).

### 4. Tests (fakes only; no model execution)

- Stable fixture and a HEAD/bound decision-step disagreement (unstable) and a
  dropped-value disagreement (unstable), with the recorded judgements exposing the
  reason (`tests/test_patch.py:502-551`); the fail-closed matrix — unknown values,
  no drop, set-vs-order comparison, a case with no judgements raising
  (`test_patch.py:554-569`).
- `run_patch_probe`: an unstable case is excluded from every cell (denominators 2 of 3
  cases, only the two stable cases captured), listed under `excluded_cases` with its
  full record, counts 2/1 (`test_patch.py:1550-1592`); fewer than two stable cases is
  refused naming both counts (`test_patch.py:1595-1609`); shared fake seams at
  `test_patch.py:1482-1547`.
- Markdown: headline sentence, stability table, the unstable row only below the
  "Unstable cases" heading, "None." when there are none (`test_patch.py:1612-1650`);
  the pre-K1 minimal payload still renders (`test_patch.py:1650-1665`).
- Existing tests updated for the new seam rather than weakened: the eligibility
  recompute test now asserts the checker sees the replayed task for eligibility and the
  HEAD task for R24, in that order, and the resulting judgements (`test_patch.py:204-255`);
  the "recomputation never runs for saved fields" poison moved from `check_trajectory`
  (which the R24 HEAD judgement legitimately calls) to `replay_task_from_id`, and the
  test now asserts the single checker call is over the case's own HEAD task and that a
  saved `detail` supplies the bound values (`test_patch.py:383-438`); a saved block
  without `detail` yields unknown bound values and therefore an unstable case
  (`test_patch.py:60-120`); the dry fixture at HEAD/HEAD asserts stability and that the
  real checker's detail parses to the removed value on both sides
  (`test_patch.py:572-654`, tail); `run_patch_probe` fixtures carry stable judgements
  and assert the new payload keys (`test_patch.py:1222-1479`).

### 5. Decisions taken (briefing rule 1.10)

- Unstable cases are listed, not run: the headline cells and every control run over
  stable cases only, and an unstable case costs no generation. R24 asks for them to be
  listed separately, and scoring them at HEAD is precisely what the ruling withholds.
- Unknown dropped values fail closed. A saved `integrity` block whose violations lack the
  `detail` field (none of the real files hit this path — both are pre-schema and
  recomputed) yields `dropped_values_bound: null` and an unstable case; the record says
  why. Nothing is inferred from a missing field.
- The dropped-value set is read from the checker's detail string rather than by
  re-deriving `Fact` sets through `integrity`'s private helpers; the dry fixture pins the
  round trip against the real checker so a format change there fails loudly here.
- The HEAD judgement runs in `select_patch_cases` (one stage, the case carries its
  stability, the selection-only pin sees it without weights) rather than in a separate
  annotation pass that `run_patch_probe` would then have to trust.

### 6. Interpretation (the Chief's notes, carried into the experiment's reading)

- **Five cases.** The universe is small because `aggregate_report` fails under both B
  and C, so only `ledger_reconcile` supplies pass/fail pairs with an empirically passing
  note. The heat map will be indicative, not decisive; Wilson intervals over five tasks
  will be wide. It must be reported as such, and the artifact's `stable_cases: 5` is the
  denominator behind every headline cell.
- **Bound follow-up — NOT implemented here, labelled for the record:** a secondary P6
  condition for the `aggregate_report` failures using the generator-v4 note as the
  counterfactual (no passing run exists for that family). It tests a designed-correct
  note rather than an empirically passing one and must be labelled so wherever it is
  reported; it is the only way to put the family that matters most under the patching
  lens.

### 7. Suite and command (R13)

`uv run pytest` run bare: **647 passed, exit status 0** (8.94s). K1-only delta:
`patch.py` +181/−40, `tests/test_patch.py` +314/−8. Nothing under `outputs/`, `data/`,
`reports/`, or `design_specifications/pending/` was written; the Director's command is
unchanged and still selects exactly five cases:

```
uv run agent-v2-probe-patch \
  --passing-eval outputs/agent-v2b/evals/runB-test180.json \
  --failing-eval outputs/agent-v2c/evals/best-adapter-test.json \
  --policy C --model qwen25-coder-3b \
  --data-seed 20260902 \
  --generator-version 1 \
  --output outputs/probes/patch-C-2026-09-04
```
