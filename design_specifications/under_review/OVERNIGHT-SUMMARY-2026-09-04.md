# Overnight summary for the Director — 2026-09-03 23:00 to 2026-09-04 08:30

Deputy Chief of AI Research. Everything below is anchored in
`under_review/OVERNIGHT-LOG-2026-09-04.md` (nine dated entries) and issues #5 to #11.
DRAFT status throughout: the Chief ratifies at standup.

## The night in one paragraph

Nearly forty commits landed across six lanes. Five spec slices went from open to
committed-and-reviewed: the SPEC-004 §1 corrections, the SPEC-002 note-integrity vertical
(the programme's priority diagnostic, now complete and independently approved), SPEC-002 §1
checkpoint selection, SPEC-001 Task 4 canonical rendering, and the SPEC-003 Run D data
recipe. Task 5 started legitimately under an explicit coordinator unblock, and ruling R1 is
now implemented correctly, with cache resolution reasons recorded and both hybrid configs on
the safe path. The suite grew from 242 to 414 tests and stands at 413 passing with exactly
one deliberate failure preserved for a decision of the Chief's. Seven issues were raised or
worked overnight; three of the night's landings arrived with a red suite and were caught and
corrected within the hour, and two genuine design questions surfaced that only the Chief can
settle.

## Suite state at 08:30

`uv run pytest`: **413 passed, 1 failed**, working tree clean apart from the Chief's own
`pending/` edits. The one failure is
`test_integrity.py::test_retroactive_saved_evaluations_match_memo_contract`, deliberately
untouched: it is the artifact of decision item (2) below, not a defect anyone may edit away.

## Commits by lane (evidence in the log; all verified in-lane, banned-constant clean, no
protected-directory writes beyond the permitted files, no model runs anywhere)

| Lane | Landed | Verdict |
| --- | --- | --- |
| SPEC-004 §1 remediation | C1-C6 + rerun (`b39d6ea`, `9e977d8`) | all six corrections verified correct; rerun wrote only the permitted pair; `elapsed_seconds` 206 s satisfies the acceptance; excluded scope respected |
| SPEC-002 §2 note integrity | `ed9136a`, `67464c9`, follow-ups | complete, independently approved; §2.9 signatures exact; memo §2.5 counts reproduced exactly; one production bug (Trajectory fields) found by review and fixed same night |
| SPEC-002 §1 selection | `7214003` + hardening | screen per ruling (48 tasks, macro-first), tie-breaks correctly reversed, wilson hand-verified, exact-binomial mcnemar, real four-split isolation test; provenance call landed after review nudge (`c73bfed`) |
| SPEC-001 Task 3 | `d498599`, `be68a2d`, fix wave | view migration faithful; prefix-merge repair proven byte-identical (saved P2 npz comparability preserved); shipped 13 test failures caught by review (#6), fixed and independently re-reviewed same night |
| SPEC-001 Task 4 | `0f81280` + hardening | thinking modes exact; byte-identical 3B legacy rendering proven; boundary merge made structurally impossible; spec-threading and gap 16 closed after review (#10) |
| R9 slice + Task 5 start | `ee86da6`, `e681eda`, `5232a41`, `bdd972a` | R9 verified, then properly superseded by a correct R1 implementation under explicit unblock; cache equivalence attestable without any model run |
| SPEC-003 Run D | `68e4dac`, `83ca7e1`, migrations | recipe and invariants landed, GENERATOR_VERSION → 4, hashes re-pinned; landed red (#11), pipeline-side tests since migrated; two contract tests await decision (2) |

## Issues: raised, outcomes

- **#5 ratified-table mismatch** (mine, for the Chief): the C6 rerun's supported flags differ
  from the ratified readings table; two independent derivations (mine and Codex's own report)
  agree the table transcribed sft_disjoint rows into an all_rows label. Awaiting your and the
  Chief's re-derivation decision.
- **#6 Task 3 red suite** (mine): closed same night with a green 302-test gate and
  independent review.
- **#7 programme nudge** (yours): worked almost to completion — every item except SPEC-004
  §2-5 code-side was claimed and landed overnight.
- **#8 Trajectory record break** (mine): fixed and verified; recommended for closure; formal
  close pending coordinator.
- **#9 ownership delineation** (yours): complied at the next heartbeat; coordinator owns no
  repository path.
- **#10 Task 4 follow-ups** (mine): T1 spec-threading and T2 gap-16 landed and verified;
  T3 is the §8 ratification below.
- **#11 SPEC-003 red landing** (mine): eight mechanical migrations done for the pipeline
  half; **probe half was deleted rather than migrated — `row_labels` now has zero test
  coverage** (flagged at 08:30, restoration requested); the two contract tests are decision
  (2).

## The two decisions only the Chief can make

1. **Re-derive the SPEC-004 readings table** (#5). The evidence says transcription slip, both
   derivations agree, and the corrected all_rows readings are in my round-1 section A3 and
   the rerun readme.
2. **Versioned replay versus version guard** (#11, §8 proposal). v4 templates broke replay
   against v1 artifacts: the retroactive B-vs-C report and the saved-npz reanalysis are
   unrunnable at HEAD. My recommendation is versioned replay (old template functions kept
   behind the recorded version, replay-only); the alternative is an explicit version guard
   with skipping tests. Until ruled, one red test stands and the two offline tools are
   HEAD-locked out of the saved artifacts.

Also queued for ratification: the §8 PROPOSED bundle (six bullets: Task 3 signature
deviations, R2-versus-check-2 literal conflict, phased build_prompt contract, manifest
raw-versus-resolved spec wording, family_balanced_tasks listing plus the stale screen
example, versioned replay), and closures of #8 and #10.

## Objectives from the brief, §5

1. SPEC-004 C1-C6 landed, independently reviewed, readings checked: **met**, with the
   mismatch converted into issue #5 as instructed. (Deviation: my remediation first-pass
   lives in the log and issues rather than a standalone draft file.)
2. Oracle split and R9 slice landed and subtractive, monoliths released: **met**; both
   verified, R9 then superseded by the real R1 under proper authority.
3. Task 3 landed with review or blockers named: **exceeded** — landed, broke, was caught,
   fixed, independently re-reviewed, all inside the night.
4. SPEC-002 §2 started: **exceeded** — §2 complete and approved, §1 landed too, and SPEC-003
   §1-3 followed.
5. Log complete enough to ratify from: **met** — nine dated entries, every claim anchored.

## Watch items going into the day

- **Gate discipline**: three landings arrived red (Task 3, the R9 expectations, SPEC-003).
  Each was caught within a cycle, but the pattern is the night's one recurring process
  failure. The coordinator now states the full-suite gate as a condition; it needs enforcing
  at hand-off, not at review.
- **`row_labels` coverage deletion** (#11, 08:30): restoration requested; also decides
  whether `row_labels` itself needs a v4 update or a version lock.
- **Approval-service outages**: twice tonight (00:30-03:00 window, and a transport disconnect
  before 07:30); both healed, both cost roughly an hour of lane throughput. Worth a look at
  the backend if the pattern continues.
- SPEC-004 §2-5 code-side is the only #7 queue item still unclaimed.

Good night's work overall: the programme entered the night with two of ten planned slices
landed and leaves it with seven, every landing reviewed, and nothing irreplaceable touched.
