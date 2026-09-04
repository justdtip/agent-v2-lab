# Five lanes at the gate (issue #58): review round 1, ratified (2026-09-05)

Reviewer: Claude (Chief). Evidence: the brief's line references read in the review worktree
`.claude/worktrees/review-lanes` at `abf6f03` + all lane patches (suite 1053 passed, 8 skipped).

## Verdicts: all five APPROVED TO COMMIT, in the brief's landing order, subject to the sequencing ruling below.

| Lane | Read | Verdict |
| --- | --- | --- |
| 1 run D data and config recipe | four config diffs (dead `num_layers: 36` removed; D4 gets batch 1 × 4 and `gated_delta_chunk: 64`); `tests/test_tasks.py:406-440` regenerates run D byte for byte against the committed manifest, skipping cleanly when the data is absent | approved |
| 2 probe CLI preflight and provenance (#33 closure) | `require_preflight` before the GPU guard and the load (`patch.py:2334-2337`); float32 block copied from the artifact (`patch.py:1983`, `state_probe.py:932-950`); `adapter_base_identity` / `_refuse_mixed_bases` (`adapter_delta.py:90-150`): unknown never refuses, two known and different values refuse | approved |
| 3 rollout and branch hardening | `spec.chat.end_of_turn` replaces the import-time constant (`branch.py:146-160`); R21 guard plus a stage manifest so the guard is no longer inert (`rollout.py:24,225`) — this closes the K2 I set on #24 | approved |
| 4 legacy writer guards and atomic manifests | `chat_replay.py:155-166` guards before the teacher loads, against a default that pointed at a protected dataset; `write_text_atomic` for both manifests (`data.py:380,442`) | approved |
| 5 pipeline run logs and the R23 basis | `_generator_version_basis` (`integrity.py:493-512`) in `patch.py`'s shape, reachable only after a bound version; `incomplete_run` as an end-event field on evaluate/prefer/integrity (`evaluate.py:520`) | approved |

## Rulings

- **Sequencing (R37).** Lane 2 lands now: it requires preflight evidence, and both artifacts on
  disk are valid at schema 2, so nothing is blocked. The preflight schema-3 slice (#51 lane 2,
  after calibration) is committed only as one operation with the regeneration of both
  `outputs/preflight/*.json` on the lane: the Director's approval of that commit is the lift
  for the two short regeneration runs (3B, 4B), and no other stage runs between them. Until
  then the schema-3 slice stays uncommitted. The board carries the pair as one item.
- **R26 amended (vocabulary).** `incomplete_run` on a non-training stage (evaluation,
  integrity, preference) is an end-event summary field meaning "the stage exited before
  writing its report"; it carries no health verdict and no `TrainingHealth`. One word, one
  meaning: stopped short. Lane 5's usage is ratified as the definition.

## Delivery of the run D data

Regenerate in the main tree through the sanctioned `data` and `render` stages and verify the
manifest digest equals the reviewed one
(`6d8f3c7036c43934bd74dcbb4b082ff480d399802b7149a08a8ed9707b2069ac`); do not move files. The
regeneration test in lane 1 then runs rather than skips, and determinism is proven in the same
act. A digest mismatch is a finding, not something to reconcile.

## Note

The state-probe comparability lane returns for re-review under the #57 ruling; EXP-001's
sweep prerequisites return once the kind-matched layer family is derived (the Deputy's catch).

---

## Round 1b (2026-09-05): full-diff read, after the Director asked whether the implementations had been read

The first pass read only the brief's named lines. This pass read every source hunk of the
21-file diff in `.claude/worktrees/review-lanes` (1,402 insertions). Findings:

| Lane | Full-read result | Change to verdict |
| --- | --- | --- |
| 1 configs + regeneration test | as described | none |
| 2 probe CLI preflight/provenance | `adapter_base_identity` reads `adapter_config.json` then `provenance.json`, unknown never refuses; gates precede loads in both `adapter_delta` arms and both `assistant_axis` entry points; `save_axis` made atomic (R11); `capture_dtype` threaded through `build_axis` with the float32 path byte-identical; provenance written by every CLI | none |
| 3 rollout/branch | `mine_pairs` now requires `view`/`resolved` (R20 limb discharged); `end_of_turn` from the spec; guards before any work. **Two gaps:** (K1) the new `_write_stage_manifest` in both modules writes the R21 sentinel with a plain `write_text`, while this same cycle made every other manifest atomic through `runlog.write_text_atomic` — use it; (K2) `rollout.py`'s module entry point writes no `provenance.json` (branch's does; the CLI path does) — add the same `write_provenance` call | approved **with K1, K2** |
| 4 legacy writers | guards before the teacher load; honest `DEBT(R21)` on the non-atomic legacy row writes | none |
| 5 pipeline run logs, R23 basis | `run_evaluation` and `run_prefer` open the log before seeding and loading and close it on every path; `_read_pairs` rewrite is equivalent; `_generator_version_basis` reachable only after binding | none |

**K3 (commit hygiene, blocking).** Every `state_probe.py` hunk in the review worktree is the
state-probe comparability lane (R29 addendum, R35 block, `generator_version_basis` in the
reanalysis), which the brief itself says is under correction on #57 and not part of this
gate. The five commits must exclude `src/local_llm_lab/probes/state_probe.py` and the
matching `tests/test_state_probe.py` hunks; each lane commit's `git diff --stat` is checked
against its declared file list before push.

Non-blocking follow-ups recorded: evaluations share `outputs/<run>/evals/` so their
`run.log`/`events.jsonl` interleave across evaluations (start/end events keep them
parseable; a per-evaluation name suffix is the fix); `git_tree_dirty` sits in `integrity.py`
only because `runlog.py` belonged to another lane, and moves beside `git_commit` when
`runlog.py` is next touched; `patch.py`, `adapter_delta.py` and `assistant_axis.py` import the
5,000-line `state_probe` module for two helpers (`preflight_precision_block`,
`spec_capture_dtype`), which belong in `preflight.py`/`policies.py`.

Verdict after the full read: lanes 1, 2, 4, 5 APPROVED TO COMMIT; lane 3 APPROVED after K1
and K2; all subject to K3. The hold is lifted on those terms.
