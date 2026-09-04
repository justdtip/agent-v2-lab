# SPEC-003 §1–§3 (run D data recipe), review round 1 (retrospective), ratified (2026-09-05 02:05)

Reviewer: Claude (Chief). Evidence: source-level audit (291 tests green across the task, data,
probe, adapter-delta, patch, and integrity modules; fakes only) and the Chief's direct read of
`tests/test_integrity.py:79-94` (the completion-assertion invariant).

## Verdict: §1, §2, §3 and the §7 configs are COMPLETE and ratified. The spec stays in pending/ because §4–§5 are gated runs.

| Item | Evidence |
| --- | --- |
| §1.1–1.3 B templates, queue head, numeric split point, no `(full)` | `tasks.py:1170-1395` |
| §1.4 no completion assertion except `pending: none` on an empty queue, every note, levels 0–3 | `test_run_d_generator_uses_pending_none_only_for_empty_action_queues` (`test_integrity.py:79`), `completion_patterns()` shared with `integrity.py` |
| §1.5–1.7 `k of N` from listing/manifest; hop notes; required-carry invariant over five families | `tasks.py:488-490,1220-1300`; `test_integrity.py:107` |
| §2 guessed-path notes; multipliers unchanged and recorded | `tasks.py:504-544`; `configs/agent_v2d.yaml:18-22` |
| §3 explicit difficulty, six splits, level 3, version-pinned hashes (R5 supersedes byte-for-byte) | `configs/agent_v2d.yaml`; `test_tasks.py` pinned-hash tests |
| §7 three configs | present |

Sanctioned: the named §1.2 test was replaced by a broader successor during the v4 migration.
Open (lane work, not conformance): `data/agent_v2d` has not been generated; it needs no model
and is training-gate condition 3 for the D arms.
