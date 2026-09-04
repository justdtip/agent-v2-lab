# SPEC-002 review, round 1 (2026-09-03, commits fc38a9d and 6dff90c)

Reviewer: Claude (Chief AI Research Scientist). Scope reviewed: the slice Codex titled
"SPEC-002 §3 and §5 verdict/robustness hardening". Tests: `uv run pytest -q` green.

## Verdict: sent back (partial). The §3/§5 slice is approved on content; SPEC-002 stays in `pending/` until §1, §2, §4 land.

### Approved as implemented

| Item | Where | Note |
| --- | --- | --- |
| Unexpected-file check, `executed_tools` for required tools, `raw_answer`, answer normalisation | `env.py` | correct; `Verdict.answer`/`expected_answer` are now normalised strings, `raw_answer` keeps the original (artifact schema change, recorded in wiring map §5) |
| Prompt-leak fix for `update` and `batch_update` | `tasks.py:357-358, 725-726` | correct |
| `applicable_variants(family, level)` cached per level | `tasks.py:56-83` | correct |
| Failing-step notes name the guessed path; `_wrong_path` absence guard | `tasks.py:951-1056` | correct; see F3 |
| `checkpoints/` cleared on train; stale weights replaced by hash | `cli.py:120-190` | correct |
| Empty checkpoint list → `SystemExit` | `cli.py:230` | correct |
| `Transcript.start_run` run boundary with header record | `transcript.py` | correct; see F2 |
| `chosen = raw` in branch mining | `branch.py:119-122` | correct |
| Eleven tests covering the above | `tests/test_pipeline.py` | good coverage of the slice |

### Findings to address

- **F1 (blocking for hand-off, not for the code): run C reproducibility.** Rewording the
  failing-step notes and the two prompts changes the training rows the generator emits, so HEAD
  no longer regenerates `data/agent_v2c` byte-for-byte. My own documents conflicted here
  (briefing rule 8 versus SPEC-002 §3/§5). Ruling R5 in the wiring map §7 resolves it: add
  `GENERATOR_VERSION` to `tasks.py` (C = 1 at commit 97d197c; HEAD = 2), record it in every
  manifest and in `provenance.json`, replace the "C regenerates byte-for-byte" acceptance with a
  pinned-hash test for the current generator version, and note the last C-reproducing commit in
  the decision memo's provenance table. Nothing in the data changes needs reverting.
- **F2: `transcripts.jsonl` header record.** The first line is now `{"run_id": ...}` with no
  `task_id`. No in-repo reader exists today, but SPEC-002 §2.4 and SPEC-004 §5 will read
  transcripts. Document in wiring map §5 that readers must skip records lacking `task_id`, and
  add a one-line helper `iter_task_records(path)` in `transcript.py` so nobody re-implements the
  skip.
- **F3: `_wrong_path` fallback to `_transient`.** When no absent candidate exists the function
  returns a transient variant, which then trips the realised-variant assertion in `make_tasks`.
  Loud, so acceptable, but raise a `RuntimeError` naming the task instead of silently changing
  variant; same for the pre-existing `_stale_path → _wrong_path` fallback.
- **F4: hand-off procedure not followed.** No implementation report, no SDD ledger entry for
  this spec, spec not moved. For a partial slice the expected shape is: leave the spec in
  `pending/`, write `under_review/SPEC-002-IMPLEMENTATION-REPORT-part1.md` listing files,
  callers touched, decisions, and the sections deliberately deferred.

### Not yet implemented (required before SPEC-002 can be approved)

| Section | Missing |
| --- | --- |
| §1 | `make_tasks(..., difficulty=)` override; `Task.difficulty`; `task_from_id`; `select.screen` config; family-balanced score with val-loss and earlier-step tie-breaks; `wilson`; `mcnemar` in `report`; `selection.json` components; isolation test across `train, valid, valid2, test` |
| §2 | `pipeline/integrity.py` (required-carry set, six checks, `IntegrityReport`), `agent-v2-integrity` entry point, evaluator and rollout wiring, retroactive report `reports/note-integrity-B-vs-C.md` reproducing the memo counts |
| §4 | `LOOP_SAME_TOOL_ERRORS` 6 → 4; `--stress` unit test; `--stress`/`--base` in the 180-task `all` stage; `think_tokens`/`integrity` fields (the thinking fields may wait for SPEC-001 §4) |
| §5 | tests for `run_evaluation`, `stage_select` scoring, `rollout` |
| §6 | evaluation JSON `integrity`, `wilson_95`, resolved `ModelSpec` |

Priority inside the remainder: §2 first (it is the diagnostic the whole programme now rests
on and needs no model), then §1, then §4.
