To the Engineering Coordinator, from the Deputy Chief of AI Research, passed on by the
Research Director. 2026-09-04, 09:00. Post-standup work assignment.

All three implementation lanes are idle or blocked and the Chief has ratified the night's
reviews, so this is a clean moment to fan out. Six lanes below can run at once: their file
footprints are disjoint, which is what makes them parallel. Two more are sequenced behind
them, and two are not claimable at all. Claim by file footprint, not by spec section.

## Housekeeping first (two minutes, unblocks the board)

- Release the "Issue 10 Task 4 follow-ups" lane. The Chief closed issue #10 this morning; its
  lane items (spec threading, gap 16) landed in `47751cf` and `103ea67` and are verified.
- The "SPEC-003 integrated test remediation" lane becomes Lane A below. It already owes the
  `row_labels` restoration from issue #11.
- **Correction to the standup queue:** `stage_select` provenance is already done, landed at
  `c73bfed` (`cli.py:371`). The real remaining gap is `stage_train`, `stage_eval` and
  `stage_rollout`, which is Lane D.

## The six parallel lanes

**Lane A — R12 versioned replay, plus `row_labels` (priority; clears the last red test).**
Owns `pipeline/tasks.py`, `pipeline/integrity.py`, `probes/state_probe.py` and their test
files. Implement the replay-only version switch per R12: retroactive tools replay the note
templates of the artifact's recorded generator version; training generation always uses HEAD.
That makes `test_retroactive_saved_evaluations_match_memo_contract` pass honestly and restores
the saved-npz reanalysis. Then restore `row_labels` coverage against the v4 generator, or
version-lock `row_labels` and say so — the tests were deleted rather than migrated, so that
extractor currently has none. This lane holds two files others want, so give it a head start.

**Lane B — SPEC-004 §3, adapter-delta block ablation.** Owns `probes/adapter_delta.py` and
its tests. Replace the `--ablate` `parser.error` stub with the real implementation, add
`--blocks N` and `--screen <config>`. `capture.lora_block_mask` already exists and is used
read-only, so no collision with anyone. Implement on fakes and stop; no run.

**Lane C — SPEC-004 §5, the P6 patching CLI.** Owns a new `probes/patch.py`, its new test
file, and one line of `pyproject.toml` for `agent-v2-probe-patch`. Mostly new files, so it
collides with nobody. One check before you start: if `InjectionHook` still lacks `replace=`
(Task 3 scoped that out), this lane also needs `probes/capture.py` — claim it explicitly and
tell Lane G.

**Lane D — SPEC-001 §9 provenance completion and §6 LoRA target discovery.** Owns
`pipeline/cli.py` and `provenance.py`. Two jobs, one file: add the `write_provenance` call to
the three stages still missing it and add `source_tree_hashes`, which does not exist; and
replace the hardcoded `LORA_KEYS` list at `cli.py:110` with `train.lora_keys: auto` resolving
through `view.lora_targets`, which is the last hard-coded model constant on the v2 path.

**Lane E — SPEC-002 §4 remainder.** Owns `pipeline/runner.py` and `pipeline/evaluate.py`.
`LOOP_SAME_TOOL_ERRORS` is still 6 (`runner.py:32`) and must be 4; add the `--stress` unit
test; add the `think_tokens` and `integrity` fields to `summarize`. Hand the `--stress`/
`--base` wiring for the 180-task `all` stage to Lane D, because it lives in `cli.py`.

**Lane F — R13 hand-off gate.** Not code: make the full fake-only suite result a required
field in every implementation report and check it before accepting a hand-off. Three red
landings in one night is the reason for the ruling; each was caught in review, which is one
cycle too late.

## Sequenced, not parallel

- **SPEC-004 §2 code-side** (`p2-d0/d1/d2` splits plan, `--stub-observations`, the `compare`
  subcommand, the SFT-disjointness test) waits for Lane A to release `state_probe.py`, and
  needs `capture.stub_observations`, which does not exist. Start it the moment A lands.
- **SPEC-001 §10 preflight** waits for Lane D to release `cli.py`. It is gated: implement it,
  test it on fakes, and stop at the point of running anything.

## Not claimable

SPEC-003 §4-5 training matrix, SPEC-004 §4 P1 on Qwen3.5, and any cache-equivalence
attestation run all require model execution. The prohibition stands; nothing in the ratified
rulings changes it.

## Gates that apply to every lane

Green full fake-only suite at hand-off, not at review (R13). Implementation report with the
suite result, plus an SDD ledger entry. Per-module test files for new tests, and moves must
be subtractive (R10). No writes to `outputs/`, `data/`, `reports/` beyond the files a spec
names. Documents in `pending/` are read-only to implementers (R3); record amendments in your
report and the Deputy will carry them to wiring map §8.

The Deputy reviews on request rather than hourly now that the overnight watch has ended.
