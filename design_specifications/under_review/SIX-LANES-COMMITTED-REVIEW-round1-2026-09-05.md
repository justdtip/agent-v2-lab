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

## Gate: C7 provenance shape (rollout and branch), 2026-09-05 evening

Worktree `agent-a58330251f8da8762`, based on `c741b28` (one records-only commit behind the
head). Four files dirty, the four owned. Read in full.

**Approved to commit.**

- `run_rollout` and `run_branch_mining` take a keyword-only `on_resolved` sink, defaulting to
  `None`, and fire it immediately after `load_policy`, before any task is generated. The return
  shape is unchanged, so the two keyword-only call sites in `cli.py` are untouched. `Callable`
  and `ResolvedSpec` were already imported in both modules.
- Both entry points collect the sink into a list and pass its first element to
  `write_provenance`, so the provenance file's top-level `model` is now the resolved identity,
  matching the summary one level down and the shape the probe CLIs write.
- The accepted deviation: an unfired sink degrades to the old shape (`None`) rather than
  raising. Ruled acceptable: in production the sink cannot be unfired, raising there would kill
  a run whose dataset is already on disk, and the two ends are pinned by tests (the real
  function fires the sink; the entry points' provenance carries the resolved identity).
- Tests: 21 passed bare in the worktree; ruff clean on the four files.
- Follow-up, agreed as a separate issue: three `cli.py` sites still pass `resolved=None`
  (`stage_select`, `stage_eval`, `stage_rollout`). The third is a one-line pass-through now
  that `run_rollout` takes the sink; the first two need the same sink upstream in their helper.

## Gate: EXP-001 slice C1 and C3 to C6 (issues #61, #62), 2026-09-05 night

Worktree `agent-a0845fe6ae2bd578d`, based on `54e0796` (one commit behind head; the commit
between is C7, disjoint files). Six files dirty, the six owned. Read in full.

**Approved to commit.**

- **C1.** `run_sweep` gathers, per prompt, every token id it must be scored against (its own
  case's pair and its predecessor's through the rotation), scores each prompt once keyed on the
  prompt string, and re-keys by candidate label for the artifact. The matched computation's
  statistics are the ones recorded, as before. The test spies on the map entry point with a
  token-dependent unembedding: sixteen maps uncached, eight cached, results and per-case blocks
  identical. Exact saving, no number moves.
- **C2, second half.** The conformance block carries `reduction` with `within_a_sample`,
  `across_samples` and a definition of a sample; the docstrings and the markdown name both
  axes. The claim that the across-sample division is removed by the final RMS norm before the
  unembedding is correct, which is why only the within-sample axis was ever a choice.
- **C3.** The prompt-rendering coordinate now carries `row_keep_last`, `rewindowed` and a
  windowing rule. The sweep records `row_keep_last` as the protocol default, which is
  `build_rows`' actual default (verified: `data.py:148`), and `rewindowed=False`.
- **C4.** The literal nine-layer list (5, 11, 12, 16, 20, 21, 27, 28, 32), the partners and the
  pairs are pinned beside the derivation test, whose docstring now states correctly that it is
  blind to a change in its sources.
- **C5.** The wrapper says it keeps the path and not the numbers; `research/jspace_probe.md`
  carries the reproduction callout naming commit `1953493`. The markdown edit is within C5's
  named scope.
- **C6.** The Holm column is labelled and the dash explained.
- Two suites: 53 passed in the worktree; ruff clean on all six files.
- **Follow-up, agreed as its own issue.** `agent-v2-jlens` records `keep_last=None` meaning
  "the default applied", while the sweep's `None` means "each case passed its own row's tool
  count". The new coordinates disambiguate; `jlens` should record the effective default.

## Gate: preflight footprint calibration (#51), round 1, 2026-09-05 night

Worktree `agent-a8745dfdde8711c26`, based on the wip commit `7991927` (whose preflight and CLI
modules are byte-identical to the main tree's uncommitted ones). Three files dirty. Read in
full against the branch head, so the review covers the schema-3 slice and the calibration
together, which is what the commit will carry.

**Verified.**
- The envelope: the floor is a least-squares line over the four no-recurrence points lifted to
  its binding point; each recurrence form is fitted on the residual over the floor in its
  declared shape (proportional, affine, flat) and lifted again; the chunked envelope no longer
  reads the chunk. I evaluated all thirteen points myself: every estimate sits at or above its
  measurement, including the three OOM lower bounds, and the predicted verdict at a 17.76 GiB
  budget with 10% headroom equals the measured verdict at every point. Zero mismatches. The
  chunked 1591-token case is refused on its own measurement (16.93 × 1.1 > budget), which is
  the correct reading of the headroom rule, not a weakening of it.
- The selector: `_select_recurrence_mode` validates the configured name first, then floor for a
  backbone with no recurrence, then unrolled when no chunk is configured, else the calibrated
  form for the name. That is `cli._training_backbone`'s order. Both halves of the falsification
  are addressed.
- Single-observation caveats on `unrolled` and `chunkwise`; coefficients and points recorded in
  the artifact; domain departures (batch size, checkpointing) named, not scaled for.
- Suites: 119 passed in the worktree. The two ruff findings are the pre-existing `render`
  shadow in `cli.py` (#64), not this slice's.

**Two conditions on this commit (K4, K5), then it lands with the bare regenerated artifacts.**
- **K4.** `require_preflight(consumer="training")` accepts a skipped footprint: the bare
  artifact's block carries `passed: true`, and `_training_evidence_passed` checks only
  `passed`. So the moment tonight's artifacts exist, the training gate would pass a footprint
  that was never computed. `_training_evidence_passed` must return False when the block is
  `skipped` (or `refused`), with a test that the same bare artifact is accepted for the `view`
  consumer and rejected for `training` with a reason naming the missing row count. Tonight's
  probes use the `view` consumer and are unaffected.
- **K5.** `_training_backbone` maps `chunkwise` to `install_chunkwise_gated_delta`, which does
  not exist in the tree this commit produces (R32 stage 2 is unlanded). An arm configuring
  chunkwise would fail with an AttributeError at train start, after the gate. Resolve the
  installer with a default and raise a clear error naming the missing function and R32 stage 2,
  with a test using a fake `training` module that lacks it.

**Condition on the next training lift (T1), recorded on #51, not on this commit.** The
footprint lives in a per-model artifact but is a per-arm question. Before a training gate
reads it, either the artifact is regenerated with the arm's own chunk, mode, batch size and
longest trained row immediately before the gate under the same lift, or
`require_preflight(consumer="training")` compares those inputs against the arm and refuses a
mismatch and any non-empty `calibration_domain_departures`. The Deputy's recommendation to
defer the domain question is accepted; it is folded into T1.

**Commit scope.** Only: `pipeline/preflight.py`, `pipeline/cli.py`, `tests/test_preflight.py`,
`tests/test_models.py`, and the comment-only R32(b) lines in `configs/models/*.yaml`, plus the
two regenerated `outputs/preflight/*.json`. Not: `configs/agent_v2b_qwen35_4b.yaml`,
`training/__init__.py`, `training/gated_delta_chunkwise.py`, `tests/test_gated_delta_chunkwise.py`
(R32 stage 2), `tests/test_arch.py`, `tests/test_probes.py`, `tests/test_cli.py` (other slices).

**K4 revised (same night).** As written, K4 would have rejected the view consumer: the footprint check runs for every consumer (`:563`), footprint `passed` is a conjunct of `report["passed"]` (`:470`) on which `run_preflight` exits (`:475`), and the view evidence inherits it (`:1186`). Revised: footprint leaves `report["passed"]`; the training consumer requires the block present, not skipped, not refused, passed; the view consumer ignores it. Scope widened by `tests/test_cli.py` (one inert stage-2 line rides along, named in the commit message).

**Round 2 (same night): K4 and K5 approved; the calibration commit is cleared.** Read the
delta alone (`fix-k4-k5-mine-only.patch`) against the gated calibration. `report["passed"]` is
now residual, finite JVP and memory within budget; `run_preflight` raises only on that; the
footprint check moved out of `_training_evidence_passed` into `_footprint_rejection`, consulted
only under the training consumer, which rejects a missing block, a refused form (read before
`skipped`, deliberately), a skipped block naming the missing row count, and a failed estimate.
The view consumer reads none of it. `_training_backbone` resolves the installer with a default
and raises a `ValueError` naming the function and R32 stage 2, with no fallback. Tests: the
bare artifact passes view and stops training; failed, refused and absent blocks likewise; the
fixture's default footprint is computed-and-passing so the malformed-evidence cases cannot
pass vacuously; the K5 guard is tested against a fake training module. 84 passed in the two
suites. Implementer decisions accepted: three tests encoding the abolished expectation
rewritten; a refused form no longer fails the command. Follow-up, not this commit: one line of
operator output when `run_preflight` writes a failed or refused footprint and exits zero.
Commit: the seven files plus `tests/test_cli.py`, with both regenerated artifacts named by
snapshot revision; the Chief verifies the artifacts on disk before sending the green.

## Gate: R32 stage 2, chunkwise gated delta rule, round 1 (algebra and wiring), 2026-09-05 night

Main tree, uncommitted: `training/gated_delta_chunkwise.py`, its test file, the `training`
package export, `configs/agent_v2b_qwen35_4b.yaml` (mode `chunkwise`). Commit held under the
lane freeze. Read in full.

**Algebra: correct, checked on my own derivation.** From the library's step (decay, delta
update, post-update read; `gated_delta.py:127-168`, verified) the state recurrence is
`S_t = g_t S_{t-1}(I - β_t k_t k_tᵀ) + β_t v_t k_tᵀ`; normalising by the cumulative gate gives
a rank-one update per token whose coefficients are gate ratios `G_t/G_s ≤ 1`, so no `1/G`
appears. The code's four lines match: the UT-transform matrix is `tril(β_i D_ij (KKᵀ)_ij, -1)`;
the pseudo-values solve `(I+A)` against `β⊙V - β⊙G⊙(K S_inᵀ)`; the read is
`G⊙(Q S_inᵀ) + (D_incl ⊙ QKᵀ) Ũ`, inclusive so a token sees its own update as the reference
does; the outgoing state is `G_C S_in + (Ũ ⊙ G_C/G)ᵀ K`. The blockwise unit-lower inverse is
exact (`Inv ← Inv - mask_s ⊙ Inv L Inv` writes only the (2,1) block, which equals
`-Inv22 L21 Inv11`), and I traced chunk six through its three levels. The gate floor at the
smallest normal is right. The padding step (unit gate, zero key, value, query and beta) is the
identity on the state and yields zero outputs, sliced off. The head repeat and the float32
policy mirror the reference (`gated_delta.py:240-244`, verified). The installer's patch point
is stage 1's and reaches only the `use_kernel=False` branch (`gated_delta.py:282`, verified).
The even-chunk restriction is a measured MLX Metal defect, characterised by process counts and
error magnitudes, not asserted.

**Mask fallback: not live on the trainer's path.** `create_ssm_mask` (`models/base.py:58-61`)
returns `None` without a cache, so the full model in training mode hands no mask to the
recurrence. The real-library test drives `GatedDeltaNet` directly and so does not cover the
model's own forward; that is a coverage gap, closed under K6.

**Conditions on the commit.**
- **K6 (the Deputy's first gap, adopted).** `fallback_counts()` is read by nothing but tests,
  and the train stage records the configured mode, not the form that ran. After the
  `_training_backbone` block, `stage_train` reads the counts and writes them into
  `health.json`, the provenance and the run log; a non-empty count on a chunkwise arm is a
  logged warning naming the reason. Add one test that drives the real `qwen3_5` `Model`
  forward in training mode under the installer with a tiny configuration and asserts
  `fallback_counts() == {}`, so the model path (mask included) is covered, not only the layer.
- **K7 (the Deputy's second gap).** `chunkwise_state_bytes` has no consumer: the calibrated
  envelope (f1230ac) reads no state shape. Remove it and its two tests; a wiring-map §8 note
  records that stage 2's analytic figure was superseded by the fitted envelope. If the
  implementer prefers to keep it, its docstring must say it is a reference figure the preflight
  does not read.

**Arithmetic still to be confirmed** by running the test file (small tensors, Metal, since the
odd-chunk defect is Metal-specific) in the gap between EXP-001 runs; the Deputy runs it with a
captured exit code and reports the worst-error figures. The commit lands after the last
EXP-001 artifact is written, per the freeze.

**Round 2 (same night): K6 and K7 approved; R32 stage 2 cleared to commit after the last
EXP-001 artifact.** Worktree `stage2` on `c349739`; delta read in full. K6 landed in a better
shape than either option offered: `_training_backbone` yields a report and fills it in its own
`finally`, binding the counter only on the chunkwise branch, so `stage_train` never touches the
global; three distinguishable values (counts, "not applicable: checkpointed", "not applicable:
no recurrence installed"); recorded in `health.json` (and so in provenance under `health`) and
on the run log, a non-empty count warning; written from the stage's `finally`, so a run that
died mid-training records the recurrence it was running. The model-forward test proves the mask
fallback dormant in training through the real `qwen3_5` `Model`, with a control that forces a
mask and counts one fallback per linear layer. K7: the analytic state-bytes function, its two
tests, both re-exports and its three private constants are gone; stage 1's own constant and
`training_state_bytes` are untouched; wiring-map §8 carries the supersession note (PROPOSED,
ratified here; to be folded into the rulings when the commit lands). Arithmetic on Metal
(Deputy, exit code captured): 66 passed; worst errors forward 2.0e-05, carried state 2.0e-06,
gradients 3.9e-04, which pass on the relative term of the declared tolerances, as reassociated
float32 sums should; recorded as relative agreement, not absolute. Full suite on Metal 1232
passed. Four device-only CPU failures reproduced in the untouched tree and filed as #67.
Implementer decisions accepted: one home for the fallback record; pre-existing ruff findings
left alone; a dict-or-string value so not-applicable cannot read as an empty count.

## Gate: EXP-001 instrument fix, five parts (issues #54, #68), 2026-09-05 ~05:15

Worktree `wt-hybrid-period` off `c349739`; eight files; disjoint from stage 2. Read in full.
**Approved to commit.**
- Period: `hybrid_period` now inverts `is_linear` over the view's own blocks (first attention
  block's index plus one), with the configuration walk as a cross-check that raises on
  disagreement; the source names which route answered. `_member` reads the attribute first
  and falls back to the Mapping, with the trap documented at the site. My reproduction on a
  tiny real `qwen3_5` through the real `ArchitectureView` returns 4 and derives the expected
  partners. Dense backbones return `None` with a stated reason.
- Accessor audit: `preflight._module_candidates` returned after the Mapping branch, so plain
  attributes were never yielded (a second live instance, surviving only because the
  recurrence module is a registered submodule); fixed. Six sites marked safe with reasons,
  including `_parameter_tree_bytes`, where Mapping-first is correct because it counts
  parameter buffers.
- #68: `NoTailBlocksError` (a subclass of the empty-window error, named apart) raises when
  `future`/`all` is requested at `L == num_layers`; both CLIs compute only `self` and the logit
  lens there; the exclusion, its reason and its Holm effect are recorded in the conformance
  block; the Holm families are built from the layers scored. The jlens CLI's mirrored
  `jlens_top_k` carries a per-layer `primary_readout` so the final-layer cell is self-describing.
- Hard refusal: the sweep refuses a decoder with both block kinds that derived no partners,
  naming the period read and `--layers` as the way past.
- Precision block: read from `residual_equivalence.fp32_manual_vs_native` with a top-level
  fallback; the reader-versus-writer diff found this the only key at the wrong level.
- R38 applied: every new fixture is the real writer (`_preflight`), the real class
  (`nn.Module`, a real tiny model) or the real view; the precision test pins the writer's
  shape and the reader's behaviour in one, so it goes red if either moves. The old blockless
  fake is kept under a name saying what it covers.
- My run: 128 passed across the four suites in the worktree, ruff clean; the Deputy's full
  suite 1177 passed (pre-stage-2 base). Citation for the reruns: the fix commit.

## Gate: adapter-wrapped standard fixtures (#52), 2026-09-05 ~05:30

Main tree, two files (`tests/test_arch.py`, `tests/test_probes.py`), read in full.
**Approved to commit after the last EXP-001 artifact.**
- `wrap_with_adapter` runs `mlx_lm.tuner.utils.linear_to_lora_layers` over the fake's text
  module with production's argument shape (keys, rank, scale, dropout), so the fixture holds
  the library's own `LoRALinear` wrappers, and it raises naming the keys if nothing was
  wrapped. `make_fake(factory, variant)` gives every standard fake a bare and a wrapped form;
  eleven structural tests are parametrised over both.
- `make_quantized_dense_fake` is quantized by `nn.quantize` (group 64, 4 bits) into real
  `QuantizedLinear` modules; it is wider than the other fakes because `mx.quantize` accepts
  only group sizes 32/64/128, stated in its docstring. Wrapped, it is the shape an adapter load
  produces over the real 4-bit base, exercised through the view's block walk, dimension reader
  and residual capture.
- The #27 blindness is pinned: the view's discovery unwraps `.linear` at `arch.py:318-320`,
  and the wrapped-variant tests go red without it. Path-keyed consumers see the same paths;
  dimension readers get the module that owns the weights.
- Noted, not a defect: `lora_b` initialises to zero, so wrapped and bare are numerically
  identical and the exact residual equality holds; the variants test structure, not
  arithmetic. R38 in shape before R38 was written. Deputy's run: 126 passed on the CPU device.

## Gate: R38 sweep, slice one (#70), 2026-09-05 morning

Worktree `wt-r38-slice1` at `9ec46d9`; five files. Read in full; my run of the three suites
passed (104 passed, 1 skipped), ruff clean. **Approved to commit; no freeze in force.**
- Three fixtures rebuilt through real writers: `write_provenance` and mlx-lm's own
  `save_config` (through `lora_config` and `_effective_lora_args`) for the adapter tests;
  `write_report` plus `summarize` plus the new `evaluation_metadata` seam for the integrity and
  report tests; real `ResolvedSpec` over a real registry declaration. Each key is then moved
  and the test confirmed red, or the silent disappearance asserted where the reader skips.
- **One reader wrong:** `adapter_base_identity` read `model.spec.hf_id` only; `write_provenance`
  writes a flat `asdict(spec)` block when a stage holds no `ResolvedSpec`, and four such stages
  overwrite the run directory the reader consults. Fixed, nested first and flat as fallback.
  The precision-block defect again, one reader over.
- Refuted lead corrected: no provenance under `outputs/` carries a non-null
  `snapshot_revision` (`models._snapshot_revision` finds none on MLX objects), so a `None`
  revision is diagnostic of nothing.
- Two assertions changed, both hand-fixture artifacts: the Wilson interval for one of two is
  9 to 91 percent (verified), and the mixed-integrity layout is rendered for the first time.
- `evaluation_metadata` extracted as a writer seam: accepted; pure, no behaviour change.

**Rulings on the three open items.** (1) `generator_version` is written by nobody:
write it, in `evaluation_metadata`, and in the patch artifact; readers prefer the recorded
value, refuse a conflicting flag, and keep failing closed when it is absent (legacy files still
need the flag). (2) `data_seed`'s silent fallback in `_analyse_evaluation`: never silent; the
analysis record carries `seed_source` naming the caller's seed and the missing field, and the
run log warns. Likewise `report.load_summaries` reports what it skipped and why rather than
dropping rows silently. (3) The seam stands. Items 1 and 2 are slice two of #70.

## Gate: R38 sweep, slice two (#70), 2026-09-05 morning

Worktree `wt-r38-slice2` at `990b43c`; ten files. Read in full; my run of the six touched suites
green; ruff unchanged from baseline. **Approved to commit.**
- A1 landed: `evaluation_metadata` writes `generator_version`; the integrity and patch readers
  prefer the recorded value, refuse a conflicting flag naming both values and the artifact,
  and fail closed when absent. The patch artifact now carries
  `evaluation_generator_version` and its source in the eligibility block of both sections,
  unconditionally; the pre-existing `recomputed_generator_version` fired only against legacy
  files. A2 landed: `seed_source` in the analysis record and the report, a run-log warning
  when the caller's seed was used; `report.load_summaries` names every skipped file and why
  on stderr instead of dropping rows.
- **One reader wrong, worse than slice one's:** `probes/patch.py` read `data_seed` at the
  payload's top level while the writer puts it in the summary, so every real artifact took the
  field-absent branch, the flag's value was used with source "flag", and the R22a conflict
  check never ran. Fixed, summary first and top level second. Latent for the queued P6
  secondary (both inputs record no seed at either level), live for the next evaluation the
  writer produces.
- Site `data.py:413`: the Deputy's premise (that it feeds the R21 guard) was wrong and the
  implementer corrected it; the guard fires on the output manifest's presence first. A real
  write-ordering hazard next door is #73, ruled below.
- Site `assistant_axis.py:1023`: the seam is well shaped (the reader touches trajectories
  only, so it needs the runner seam, not `evaluation_metadata`); both hand fixtures replaced by
  `write_report` output, and a renamed writer key now fails twelve tests where it failed none.
  Two soft keys pinned rather than tightened: a missing action key silently truncates a
  trajectory at that turn. Ruled below.
- Cumulative: six readers audited, two wrong.

**Rulings.** (#73) The manifest is the commit point: data files first, the manifest last and
atomic. A directory without a manifest is unprotected by design and re-runnable, which is the
right property for a crashed render. What must not happen is a reader consuming it: every
consumer of a dataset directory (`render`, the training data loader, the probes' data paths)
requires the manifest and refuses without it. Slice three. (Truncation) A step whose action key
is missing in an artifact our own writer produced is a malformed record and raises; the only
tolerated layout is the legacy `<tool_call>` form the parser already accepts by name. Slice
three, with the pinning test inverted.

## Gate: R38 sweep, slice three (#70, #73), 2026-09-05 late morning

Worktree `wt-r38-slice3` at `efcfb78`; thirteen files. Read in full; my run of the seven touched
suites green. **Approved to commit.**
- #73 landed on the read side: `require_dataset_manifest` refuses a dataset directory without
  a manifest, naming it, and is called by `render_dataset` (source), `_role_chat_rows` (chat
  and extra dirs), `load_rendered_splits` (the seam where a dataset becomes weights),
  `build_expanded_data`, `train_expanded` and the assistant-axis prompt loader. The write
  guard still leaves such a directory re-runnable. Verified: all eleven pipeline dataset
  directories on disk carry manifests; `data/sft` and `data/rewards` do not and are read only
  by the legacy `train_sft`/`train_grpo` scripts and `configs/lora.yaml`, outside the guarded
  readers; if ever fed to one they refuse by design. The Deputy's premise that the manifests
  were not already atomic was refuted (they are); the non-atomic sites named in `data.py`'s
  own DEBT block are #74, health.json first, agreed.
- Part B landed as refined: a step with `parse_error` ends the trajectory; an executed step
  missing `action`, `thought` or `observation` raises naming file, trajectory and index.
  Verified by the Deputy over 6,503 step records on disk: exactly two key shapes. The legacy
  `tool_call` clause was correctly refuted (a parse question inside a valid step).
- `selection.json` now written atomically; provenance records what was persisted, and the
  read-back is load-bearing (Wilson tuples normalise to JSON lists).
- Reported, ruled below: `_selection_components` scores `family_macro_success` at zero when
  `by_family` is absent, so the first ranking key falls silently through to micro success
  while the criterion string still says macro-first; and a dead legacy fallback names counts
  no writer has emitted.
- Seven fixtures gained a hand-made empty `manifest.json`; the guard reads presence only.
  Accepted, with the presence-only contract stated in the guard's docstring.
- Cumulative: nine readers audited, two wrong.

**Rulings (slice four).** `summarize` always writes `by_family`, so a summary without it is
malformed: `_selection_components` raises rather than scoring zero, and the criterion string
can never disagree with the comparison made. The dead legacy fallback is removed. Slice four
also takes the remaining five of the fourteen sites.

## Gate: R38 sweep, slice four (#70), 2026-09-05 late morning

Worktree `wt-r38-slice4` at `8420542`; nine files. Read in full; my run of the four non-MLX
touched suites green (the MLX one rides the Deputy's Metal run in the gap). **Approved, with two
small items to land in the same commit, and the commit held for the Metal run.**
- The enumeration was short by one: fifteen readers, not fourteen (`cli._validation_losses`
  reading `metrics.jsonl`). The Deputy found and stated it; the table needed auditing too.
- Two more readers wrong, both of the second form (a fallback for a legacy shape that never
  existed, whose default is the value the field actually holds): `baseline_reanalysis_parameters`
  fell back to fit constants for baselines "written before the fit block", but the writer
  (`f90578b`, 2026-09-03) precedes the reader (`eca116b`, 2026-09-04) and the only baseline on
  disk carries the block; and `ExpansionSpec.load` read `initialization` and `version` with
  dataclass defaults that are the only values ever written. Both now require the keys.
  `generator_version` stays soft in the baseline reader, correctly: the ratified baseline
  records none.
- `_selection_components` raises without `by_family` (ruling A1) and refuses `valid_actions`
  without `rate_counts` rather than naming a dead legacy pair; the two flat fallbacks real
  artifacts resolve against are kept and pinned. `_validation_losses` pinned to the training
  writer's record shape. `require_dataset_manifest` states its presence-only contract.
- **Ratified:** the fit-constants tightening, made by the implementer on A2's reasoning outside
  the ruling's named site; same evidence, same rule.
- **Add before commit:** `ExpansionSpec.load` validates `version` against the versions this
  code applies (currently 1) and refuses others by name; a field whose job is to say "not this
  recipe" must be checked, not only present. One line and one test.
- **The closing summary** across the four slices is the sweep's deliverable and must be a file
  the record keeps (`under_review/R38-AUDIT-SUMMARY-2026-09-05.md` or the closing #70 comment
  carried by the records commit), not only a commit message.
- Totals on the Deputy's count: fifteen readers, four wrong; two fixtures already built from
  the real writer still hid a defect, found only by moving the key.

**Landed:** slice four as `ba6435b` (nine files, declared scope; `SUPPORTED_VERSIONS = (1,)` check in `ExpansionSpec`; full suite on Metal 1362 passed). #70 and #75 closed. Audit summary ratified; R38 amended twice (soft defaults; confirm-or-refute binds the officers).
