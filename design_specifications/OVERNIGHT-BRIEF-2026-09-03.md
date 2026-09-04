# Overnight brief for the Deputy (2026-09-03, ~23:00 to morning)

From: the Chief. Authority delegated by the Director for the night: hourly check-ins and
raising GitHub issues as required (Codex's monitor assigns issues to lanes automatically).
Everything else in your role document stands unchanged: no commits, no model runs, no rulings,
no approvals, no edits in `pending/` beyond §8 proposals and the briefing §3 anchor table.

## 1. Situation at handover

Four Codex lanes are active or queued, all bounded and consistent with rulings R7 to R11:

| Lane | Work | State |
| --- | --- | --- |
| SPEC-004 remediation | C1 to C6, remediation tests moved to `tests/test_reanalysis.py`, gap-16 rename | in flight, dirty tree |
| Generator oracle split | oracle tests to `tests/test_tasks.py`, actionable re-pin failure, chat-inclusive hash (`ee07f6a` may already cover part) | in flight |
| Issue #1 / R9 safety slice | hybrid configs to `cache.strategy: none`, `TEMPORARY until R1` comment, `xfail(strict=True)`, tests to `tests/test_models.py` | queued behind the SPEC-004 lane's release of `tests/test_probes.py` |
| SPEC-001 Task 3 | capture, J-lens, adapter-delta, J-space, GPU guard on the view | running, claims neither monolithic test file |

Full Task 5 stays blocked until Codex's head explicitly unblocks it; that is their gate, not
ours. Issues #1 to #4 are open; #4 mirrors UUID-to-checkbox ownership.

## 2. Hourly check-in: what to do each hour

Keep each cycle cheap (role doc §8). In order:

1. **Progress survey.** `git log --oneline <last-seen>..HEAD`, `git status --short`,
   `.codex/coordination/CURRENT.md`, `gh issue list`. Map each new commit to its lane and spec
   section. Update your role doc §9 and append one dated entry (3 to 6 lines) to
   `under_review/OVERNIGHT-LOG-2026-09-04.md`: commits landed, lanes moved, anything raised.
2. **Code review, first pass, per landed commit.** Draft review file per role doc §5 if the
   commit completes a lane; otherwise fold findings into the log. Checks, in order:
   signatures against wiring map §2; the lane stayed inside its claimed paths; banned-constant
   grep on added lines; no protected-directory writes; no model loads in tests;
   `uv run pytest -q` green; report and SDD ledger entry present at lane completion.
3. **Fidelity audit, delta only.** Do not redo the full audit; check the night's diffs against
   the eight rules (briefing §1) and the specific corrections on issues #3 and #4. Confirm
   R10 moves are subtractive from the monoliths (definitions removed, not duplicated) and that
   collection still passes.
4. **Anything else you judge necessary**, within role bounds.

## 3. Specific things to verify tonight, per lane

- **SPEC-004 remediation:** C1 per-cell `n_test`/`n_cells` and `n/a (n=…)` suppression; C2
  Holm-support definition in the markdown; C3 `elapsed_seconds` + data-stage ModelSpec; C4
  cohort labels and by-difficulty table; C5 report + ledger entry; C6 rerun wrote only the
  two reanalysis files (check with `find outputs -newermt`). Compare the rerun's readings
  against the ratified table in `SPEC-004-s1-REVIEW-round1`. **If the recomputed Holm results
  disagree with the ratified table, that is a finding to report, not an error to reconcile**:
  Codex has been told to preserve computed outputs and report mismatches honestly. Quote both
  numbers in the log and raise an issue titled "SPEC-004 §1: ratified-table mismatch" if any
  supported/unsupported flag flips.
- **Oracle split:** `ee07f6a` claims chat-inclusive coverage; verify the chat-inclusive case
  pins real hashes and skips only when `data/chat_replay` is absent, and that a version bump
  produces the actionable message, not `KeyError`.
- **R9 slice:** both hybrid YAMLs read `strategy: none`; the `models.py` branch carries the
  TEMPORARY comment; the snapshot test is `xfail(strict=True, reason="R1")` and the suite
  treats it as expected-fail, not pass.
- **Task 3:** this is the highest-risk lane. Signatures against wiring map §2.13 and §2.14
  exactly; `_causal_mask` gone from `jlens.py` with masks from `view.masks`; `_distribution`
  and `_encode` promoted to public names with `jspace_sweep.py` and `adapter_delta.py`
  updated (§4 gap: callers!); `InjectionHook` wraps `run_block` and tracks the offset for
  `ArraysCache`; the J-lens JVP records `jvp_method`; the GPU guard added to `agent-v2-jlens`;
  hybrid-fake tests present; the saved-artifact regression test still passes. Watch
  specifically for capture-side changes to `response_mean_activations`' prefix repair: any
  change there invalidates comparability with the saved P2 npz and must be flagged.

## 4. Escalation to the Chief (do not decide these yourself)

Wake the thread (log prominently + raise an issue labelled for the Chief) if any of:
a supported/unsupported flip against the ratified SPEC-004 table; any commit touching
`outputs/` beyond the two reanalysis files, `data/`, `reports/`, or `research/`; any model or
tokenizer load outside a fake; a signature landing that contradicts wiring map §2; two lanes
writing the same path; Task 5 resolver/runner work starting without the head's unblocking
message; or a proposed change to a ruling. Everything else waits for morning with your
recommendation in the log.

## 5. Objectives by morning (what "a good night" looks like)

1. SPEC-004 C1 to C6 landed, independently reviewed by Codex's reviewer, your first-pass
   review drafted, readings confirmed or a mismatch issue filed.
2. Oracle split landed and `tests/test_pipeline.py` released; R9 slice landed and
   `tests/test_probes.py` released; both verified subtractive.
3. Task 3 either landed with your draft review ready for my ratification, or its blockers
   named precisely in the log.
4. SPEC-002 §2 (note integrity) started by Codex once seams free; if no lane has picked it up
   by ~03:00, raise an issue: "SPEC-002 §2 is the priority lane per issues #2/#4; seams are
   now free" so the monitor assigns it.
5. The overnight log complete enough that I can ratify reviews at morning standup without
   re-deriving anything.

Keep drafts DRAFT. Good watch.
