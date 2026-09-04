# Document moves out of under_review/ (Chief's decision, 2026-09-05)

Basis: two read-only classifications of all 47 files against ratification posts and the branch
history. Execution: file moves are the Chief's domain; the commit is the Deputy's, referencing
this file. Nothing here changes any spec or ruling.

## New directories

- `complete/` — approved implementations (report + its ratified review round, moved as a pair).
- `live/` — instruments re-derived every commit; not "awaiting approval", never "complete".
- `records/` — closed logs, superseded drafts, one-time notes, blocked attestations.
  Rulings do not live here: wiring map §7 is the durable rulings register.

## A. Move to complete/ (7 pairs, 14 files; all Chief-ratified, conditions closed, commits on branch)

| Pair | Issue | Commit |
| --- | --- | --- |
| LOAD-POLICY-MIGRATION-REPORT.md + WAVE1-LOAD-POLICY-REVIEW-round1-2026-09-04.md | #19 | a2f003c |
| R17-SCANNER-REPORT.md + R17-SCANNER-REVIEW-round1-2026-09-04.md | #21 | afc0905 |
| R18A-GATE-REPORT.md + R18A-GATE-REVIEW-round1-2026-09-04.md | #23 | ed88c96 |
| R21-RENDER-GUARD-REPORT.md + R21-RENDER-GUARD-REVIEW-round1-2026-09-04.md | #24 | 94c920e |
| ISSUE-22-P6-PROVENANCE-REPORT.md + P6-ELIGIBILITY-REVIEW-round1-2026-09-04.md | #22/#26 | 07c6657 |
| LORA-WRAPPER-VIEW-FIX-REPORT.md + LORA-WRAPPER-VIEW-FIX-REVIEW-round1-2026-09-04.md | #27 | 89dfb56 |
| P6-POSITION-GROUPS-FIX-REPORT.md + P6-POSITION-GROUPS-REVIEW-round1-2026-09-04.md | #29 | ac9c27a (the two doc files are untracked; commit them in the same move) |

## B. Move to live/ (5 files)

TRAINING-READINESS-CHECKLIST.md, PROBES-READINESS-CHECKLIST.md (both ratified; headers restored),
HEARTBEAT-LOG-2026-09-04.md, B4-TRAINING-LIFT-REQUEST-2026-09-04.md,
P6-LIFT-REQUEST-2026-09-04.md (marked FINAL but not executed; re-date on execution).

## C. Move to records/ (12 files)

OVERNIGHT-LOG-2026-09-04.md, OVERNIGHT-SUMMARY-2026-09-04.md, CODEX-OVERNIGHT-NOTE-2026-09-04.md,
CODEX-COORDINATOR-NOTE-2026-09-04.md, CHIEF-WORKFLOW-NOTE-2026-09-04.md,
CACHE-EQUIVALENCE-ATTESTATION-QWEN35-4B-2026-09-04.txt, CACHE-EQUIVALENCE-ATTESTATION-REPORT-2026-09-04.md
(blocked run; registry `equivalence_verified` stays null), PROGRESS-AND-FIDELITY-REVIEW-draft-2026-09-03.md
+ PROGRESS-AND-FIDELITY-REVIEW-round1-2026-09-03.md (programme review, not an implementation),
COMPLETENESS-ASSESSMENT-2026-09-04.md + COMPLETENESS-ASSESSMENT-round1-2026-09-04.md (same),
SPEC-004-s1-REVIEW-draft-2026-09-03.md (superseded by its round-1 file, which stays with its cycle).

## D. Stay in under_review/ (16 files): committed but never Chief-ratified

These landed under Codex's own review or the Deputy's "principal review" before the R19 chain
existed, or their cycle is explicitly incomplete. They need a **retrospective ratification pass**
(one issue per spec; the Deputy's completeness assessment is the evidence base):

- SPEC-001: S6-S9, S8-PROBE-DEFAULTS, S10-PREFLIGHT reports; R12-VERSIONED-REPLAY report.
- SPEC-002: parts 1, 2 (note integrity, header still says "changes required"), 3 (selection),
  S4; SPEC-002-REVIEW-round1 (partial verdict, stays with them).
- SPEC-003: implementation report (ruling request outstanding).
- SPEC-004: S3 block ablation, S5 P6 patching, s1 remediation report + s1 REVIEW-round1
  (remediation not re-ratified after C1-C6).
- GITHUB-ISSUE-10 follow-up report (#10 still open); GITHUB-ISSUE-15 diagnosis (superseded
  in practice by R18a; ratify as record when SPEC-001 is closed).

## E. Specs in pending/

None moves. A spec leaves pending/ only when every section is implemented and ratified; the
retrospective pass in D decides SPEC-002 and SPEC-004 §1/§3/§5 first.
