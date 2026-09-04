# Chief's review: the six committed lanes (issue #59), round 1, 2026-09-05

Reviewer: Chief AI Research Scientist. Basis: the source hunks of all six commits, read in full
(`8fdffea`, `6f84217`, `30424d6`, `1d2da93`, `1953493`, `d0adfc6`), the tests that pin them, and a
bare run of the three suites the two previously unread lanes touch (`tests/test_jlens.py`,
`tests/test_jspace_sweep.py`, `tests/test_state_probe.py`: 157 passed).

## What was read, and when

- Four lanes (`8fdffea`, `6f84217`, `30424d6`, `1d2da93`) were read in full in the review worktree
  before landing (FIVE-LANES-REVIEW round 1b). They landed as read: the only file that differs
  between that worktree and `d0adfc6` is `pipeline/protocol.py`, and that difference belongs to an
  uncommitted slice (the dead `tools=` seam), not to these commits.
- Two lanes (`1953493`, `d0adfc6`) had **not** been read by the Chief before they were committed.
  Issue #59 states every lane was "gated by the CRO on a full read of every source hunk"; for
  these two that was not so. They are read now, below. Both pass.
- Every commit's file list matches its declared lane. The state-probe comparability hunks landed
  separately from the pipeline lanes, as required (K3).

## `1953493`: state-probe comparability (#40, #57)

**Approved as committed.** Findings:

1. The partition. The reanalysis corrects one family per cohort spanning every (target, layer)
   cell (`_cohort_holm_families`, keyed on the single pooled scope). `compare` corrects one family
   per (cohort, scope) (`_compare_holm_families`); its pooled scope is the reanalysis family
   exactly, and each within-difficulty scope is its own family, which the reanalysis does not
   produce. Both collapse a cell to the larger of its two control p-values before Holm
   (`_compare_cell_holm_p` mirrors `_analyse_cohort`'s `max(p_position, p_surface)`). This is the
   #57 ruling, correctly applied and correctly extended to the scopes only `compare` has.
2. **Record correction.** Issue #59 says the correction "went further than the ruling by making
   both tools call one shared partition function". That is not what landed: there are two
   parallel functions, held equal by
   `test_compare_and_the_reanalysis_form_the_same_holm_families`, which compares the groupings
   as sets of cells. The implementation is acceptable; the record must describe it accurately.
3. Two-sided bootstrap p for a margin difference, `2(1+tail)/(n+1)` capped at one; R8's row floor
   applied to the smaller side; an ineligible cell keeps its interval flag, reports no Holm flag
   and names the reason; `holm_supported` is strictly additional to `supported`. All correct.
4. R35: nine dimensions (R29's five plus policy, derivative method, prompt rendering, estimator
   variant) recorded whole with both sides' values and a disposition; refusals raise with the
   block attached; the markdown carries the same table. Correct.
5. Minor: `generator_version_basis` is now duplicated between `patch.py` and `state_probe.py`
   with identical text. Consolidate when either is next touched.

## `d0adfc6`: EXP-001 sweep prerequisites (#54)

**Approved as committed**, with pre-run conditions C1 and C2 below. Findings against the five
blocking amendments and the addendum:

- **B1/B2, rendering.** `select_cases` renders through `build_prompt` with the registry's
  template kwargs and `keep_last` set to the row's own tool count, so the already-windowed row is
  not re-windowed. `build_prompt` asserts the generation suffix, which on a thinking-off model is
  the closed think block, before the forced prefix is appended. The distribution is therefore
  read outside the think block. Correct.
- **B3, readouts.** `jlens_readouts` takes one JVP per (context, source) and reduces it three
  ways: `self` at the source, `future` summed over strictly later positions, `all` their sum and
  primary. `jlens_map` is now the self-only limiting case of the same function, so the historical
  contract survives. Correct.
- **A1, empty window.** `EmptyFutureWindowError` is raised whenever `future` or `all` is
  requested and the window is empty; a self-only call never raises. Default sources 0.25/0.5/0.75
  and corpus length 128 make the window non-empty by construction. Correct.
- **B5, kind-matched family.** `hybrid_period` reads `full_attention_interval` from the loaded
  model's own configuration; the grid is the multiples of the period, filtered against the view's
  own `layer_kind` (which returns the same `attention`/`linear_attention` strings the module
  compares against); each in-band layer takes the nearest grid layer, lower index on a tie; an
  in-band layer already on the grid takes none; the dense 3B returns the selection unchanged with
  the reason recorded. For the 4B: fractions round to 5, 11, 16, 21, 27, 32; the band
  (round(32/3)=11 to round(5·32/6)=27, final layer excluded) is 11, 16, 21, 27; partners 12, 20,
  28; family 5, 11, 12, 16, 20, 21, 27, 28, 32. That is the pre-registered nine-layer list.
- **R34/R35 blocks.** The conformance block names the index convention, the family with period
  and roles, sources, positions read per readout, corpus size and length, the window medians,
  the JVP method with its source, the capture-dtype request against what the view delivered, and
  the self-only variant. The comparability block carries the R35 coordinates. Correct.
- **R26.** The run log opens with the identity before the preflight gate, the GPU guard and the
  model load; one progress line per probe point; the empty-window error is logged before the CLI
  exits; provenance written. Correct.
- **JVP method** fails closed without a flag or a preflight record. Correct.
- **Statistics.** Exact two-sided binomial sign test; standard Holm step-down; one family per
  readout across every swept layer, partners included. Correct. Logit-lens rows and the model's
  own output are outside any family and render with a dash; the markdown should say the dash
  means "not in a Holm family" (minor, C6).

### Conditions

- **C1 (before the EXP-001 lift, halves the run).** `run_sweep` computes the mismatched null for
  case *i* by re-running every J-lens map on case *i+1*'s prompt, which is exactly the matched
  computation for case *i+1*. Every JVP in the run is done twice. Cache the per-prompt
  distributions (or the candidate probabilities for both pairs) and read the null from the
  cache. Add a test on the fake view that the cached and uncached `results` are identical.
- **C2 (before the EXP-001 lift, Head of Interpretability to answer).** `future` and `all` are
  sums over output positions, not per-sample means. With sources at 0.25/0.5/0.75 of a 128-token
  context the windows are roughly 96, 64 and 32, so the summed estimate weights the earliest
  source about three times the latest relative to a mean. The docstring's "up to a scale the
  sign test ignores" holds per sample, not across samples. The block declares "summed", so R34
  is met; the Head must confirm the sum is the intended estimator, or change it to the mean,
  **before** the run. Changing it after would be a variant change under R34.
- **C3.** The comparability block records `keep_last=None`, but the rendered context was
  windowed by `build_rows` (keep_last 2, built into the rows) and then not re-windowed. Record
  the row windowing and the no-rewindow rule in the prompt-rendering coordinate.
- **C4.** `test_kind_matched_family_reproduces_the_4b_sweep_from_its_own_configuration` derives
  its expectation from the same registry and library the code reads, so a change to the registry
  fractions passes silently; its docstring claims the opposite. Add a literal assertion that the
  `qwen35-4b` default family is (5, 11, 12, 16, 20, 21, 27, 28, 32).
- **C5.** `research/jspace_sweep.py` is now a wrapper and says "the historical path keeps
  working". It runs a different variant (three readouts, interior sources, 128-token corpus,
  derived family, registry rendering) and will not reproduce the World A table. State in the
  wrapper and in `research/jspace_probe.md` that the recorded numbers reproduce only from the
  script at the parent of `d0adfc6`, named by commit.
- **C6.** Markdown: label the dash in the Holm column.

C3 to C6 are follow-ups, not blockers. C1 and C2 gate the EXP-001 lift request.

## The three Director decisions: the Chief's recommendation

1. **Run-D data.** Regenerate in the main tree through the `data` and `render` stages under the
   R21 write boundary, and compare the manifest digest with the reviewed one (`6d8f3c70…`). A
   match proves determinism and delivers the data in one act; a mismatch is a finding to be
   filed, not a choice to be made. No model, no lift, no lane. Deputy executes.
2. **Schema bump and regeneration.** R37 governs: the preflight schema-3 slice lands only
   together with the regenerated artifacts it invalidates. The slice stays at the Chief's gate
   until a "preflight regeneration" lift is granted; inside that lift the Deputy lands the slice
   and runs the preflight for both registered models (`qwen25-coder-3b`, `qwen35-4b`), and the
   code and the regenerated `outputs/preflight/*.json` are committed together. The preflight is
   minutes of model load and a JVP check. Nothing else may use the lane between the two.
3. **Lane order.** Recommend against the queued order. Every probe is blocked until decision 2
   is executed, so preflight regeneration goes first. Then the R32 stage-2 memory and step-time
   probe, because it is short and decides whether B4 attempt 4 exists at all. Then EXP-001's
   runs on the 4B base and the 3B comparator, once C1 and C2 are closed, because they gate the
   J-space question on the new base. The P6 secondary condition last: the primary condition
   already delivered the regimen-bound verdict, and the secondary refines a result no decision
   is waiting on.

## Verdict

All six commits stand. No revert, no correction to committed code required before the next lane.
Open items: C1 to C6 above, the record correction on the partition function, and
`rollout-branch-hardening` still in correction on K1 and K2.

## Addendum: the seventh lane, `382e2f8` (rollout and branch hardening, issues #24 and #58)

Landed after the six, together with two records commits (`4df386e`: heartbeat log, wiring map
carrying the Chief's own R29 addendum 2, R37 and R26 amendment text, and the five-lanes review;
`82bb61d`: the Deputy role file). Records only; the wiring-map edit is the Chief's text.

**Approved as committed.** Read in full against the round 1b conditions:

- **K1 met.** Both `_write_stage_manifest` helpers write through `runlog.write_text_atomic`, with
  the reason stated: the manifest is the R21 guard's own sentinel.
- **K2 met.** The rollout entry point now writes provenance, as the branch entry point does.
- Both entry points guard the output directory before loading anything, open the run log with
  the identity before the model load, pass the log's progress callback into the inner loop (one
  line per task), and log completion with the manifest and provenance paths. The `mine_pairs`
  seam now requires `view` and `resolved` and the end-of-turn token comes from the spec.
- The inner functions write `summary.json`, not `manifest.json`, so the new stage manifest
  overwrites nothing. The resolved model identity (`resolved.as_dict()`) reaches both the stage
  manifest and the provenance file inside the embedded summary.
- **C7 (follow-up, non-blocking).** Both entry points call `write_provenance` with
  `resolved=None`, so the provenance file's top-level `model` is the static spec rather than the
  resolved identity the probe CLIs record there; the resolved identity is present only under
  `extra.summary.model`. Return the resolved spec from the inner functions (or read it back
  from the summary) and pass it through, so every stage's provenance has the same shape.
- Suites `tests/test_rollout.py`, `tests/test_branch.py`, `tests/test_protocol.py`: 38 passed.
