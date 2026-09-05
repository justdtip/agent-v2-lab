# SPEC-004: Probe program revision (P1, P2, P5, P6, and offline re-analysis)

> Read first: `01-IMPLEMENTER-BRIEFING.md` (standing rules, traps, hand-off) and `02-INTERFACE-AND-WIRING-MAP.md` (exact shared signatures, file ownership, implementation order, integration checks). Signatures in the wiring map override any looser wording here.

Status: pending. Author: Claude. Date: 2026-09-03.
Depends on: §1 depends on nothing and runs today; §2-5 depend on SPEC-001. GPU: §1 none; §2-5
gated. Supersedes the execution order in `research/representation_probes.md` §11.

## 0. What changes and why

The hardened P2 base run is complete and, read with its position controls, already answers the
read-side question (decision memo §2.2). What remains uncertain is the write side (why the
policy drops values and asserts completion), and that is where the remaining probe budget goes.
P1 is closed on the 3B coder base for lack of any persona expression. P5 item 5 and P6 are
implemented because both are cheap and both reuse primitives that already exist in `capture.py`
(`lora_block_mask`, `InjectionHook`) and are currently imported by nothing.

## 1. Offline re-analysis of the saved hardened run (no model)

New CLI `agent-v2-probe-state reanalyse --input outputs/probes/state-corrected-hardened-20260903T172549/state-base-mix.npz --output <dir>`:

1. **Uncertainty.** Bootstrap over task ids (1,000 resamples) for every probe, position baseline,
   and margin; five split seeds; report median and 2.5/97.5 percentiles. A margin whose interval
   excludes zero is "supported".
2. **Surface-feature baseline.** Fit the same ridge/logistic on non-activation features: prompt
   token count, last-note token count, number of digits and of commas in the last note, family
   one-hot, step index. Report margin over this baseline next to the position margin; a probe must
   beat both.
3. **SFT-row exclusion.** Recompute everything with the `train` split rows dropped (labels and
   activations are in the npz; task ids carry the split), so the base result becomes comparable
   with future adapter runs on SFT-disjoint rows.
4. **Target re-coding** from regenerated ground truth (deterministic, no model):
   `running_max` → binary `is_new_max` (value just read exceeds the running maximum) and ordinal
   `rank_of_last_read`; `prev_error` → `hidden_error` (an error occurred at a step whose
   observation is now stubbed); `pending_count` also as ordinal classes 0-6.
5. **Within-position by difficulty** and per family, with intervals.
6. **Multiple comparisons.** Report the number of (target, layer) cells tested and Holm-adjusted
   support flags.

Output `state-base-mix.reanalysis.{json,md}` with the same table layout as the original plus an
`intervals` block; a README paragraph states which conclusions survived.

## 2. P2 redesign (gated runs)

- Splits: `p2-d0`, `p2-d1`, `p2-d2` (120 tasks each, explicit difficulty, perturbed at 0 and 1,
  clean at 2), never overlapping any SFT split; a test asserts disjointness from every training
  split of every run in `configs/`.
- Conditions per policy: `intact`, `notes-stripped`, `observations-stubbed` (`keep_last = 0`),
  `both`. `--strip` gains `--stub-observations`.
- Policies: `base`, `B`, `D3` (SPEC-003), later `qwen35-4b` base and `D4`. Adapters are compared
  only within the same base.
- Capture positions: last prompt token (as now) and mean over the teacher-forced note tokens of
  the expert target (write-side view: does the residual while writing carry the value about to
  be dropped). Both stored; layer set from `spec.probes.layer_fractions`.
- Analysis: everything in §1 plus a paired cross-policy comparison (`compare` subcommand) that
  bootstraps the difference of margins between two result files on identical task splits and
  writes one table per target.
- Cost: about 65 minutes per (policy, condition) on the 3B model at the measured rate; the four
  conditions for base and B are 8.7 hours. Run `intact` and `both` first (the decisive pair),
  the rest only if the pair is ambiguous.

## 3. P5 item 5: layer-block ablation (gated, cheap)

`agent-v2-probe-delta --ablate --adapter <dir> --blocks 6` evaluates the SPEC-002 screen with the
adapter zeroed everywhere except one block of `num_layers / blocks` consecutive layers
(`capture.lora_block_mask`), plus the full adapter and the empty adapter as anchors. Output:
success per family per kept block with Wilson intervals, and the block whose removal costs most.
Cost: 8 screens × 12 minutes ≈ 1.6 hours per adapter on the 3B model. Run on B and D3; repeat on
the 4B adapters to see whether behaviour localises differently in a hybrid stack.

## 4. P1 closure and conditional reopening

- Write `outputs/probes/axis-corrected/CLOSED.md` recording the persona-expression counts from
  the 288 saved rollouts (best role 3/8; 19 of 24 roles at 0/8) and the decision that the coder
  base has no usable persona space. No further P1 work on `qwen25-coder-3b`.
- After SPEC-001 `preflight` on `qwen35-4b`: one `build` (about 20 minutes). Proceed to `project`
  only if the fail-closed gate passes and `pc1_cosine_abs ≥ 0.5` at a middle layer. Fix before
  that run: give the default persona the same 8 prompts as the roles (matched design), balance
  exemplars across high/low/neutral groups, and disable the same-policy judge (heuristic only).
  Steering remains unimplemented until a passing axis exists.

## 5. P6: causal patching at the dropped-value decision (gated)

The saved evaluations give 29 task ids that pass under B and fail under C. For each
`aggregate_report` and `ledger_reconcile` id among them, the failing decision is the note that
drops a value. Implement `agent-v2-probe-patch`:

1. Replay C's own transcript to the step before the drop; build the counterfactual context by
   substituting a B-style note (generator-rendered from ground truth) for C's previous note, so
   the two contexts differ only in the note text.
2. Capture residuals at every layer for both contexts; define position groups: system prompt,
   task prompt, previous notes, the substituted note's value tokens, last two observations, final
   token.
3. Patch group × layer from the counterfactual into the failing context (replace, not add; an
   `InjectionHook` mode `replace=True`) and greedily generate the note; score whether the dropped
   value now appears (integrity check `value_drop` from SPEC-002).
4. Controls: patch from an unrelated task's context; patch random position groups of the same
   size.

Output: a layer × group heat map of flip rates with intervals over the task set. Cost: about
6 groups × 6 layers × 20 tasks × 2.5 s ≈ 30 minutes on the 3B model. Decision rule as in the
design document §9: flips from the note-value positions at early or middle layers say the error
is in reading and representing the list; flips only from late layers at the final token point at
the computation itself.

### 5a. R30 amendment: locating note values in a family whose values repeat (2026-09-05)

**Occasioned by:** the `aggregate_report` secondary condition ran for 1:21:17 and scored nothing
(`outputs/probes/patch-C-secondary-2026-09-05/`). Its `secondary` block carries an `error` where
the scores belong: *note_value token span is missing or ambiguous in the substituted note*. The
primary reproduction in the same run is byte-identical to the recorded artifact, so the
instrument is sound and only this path is affected.

**Cause.** `patch.py:695-702` locates each note value by searching the **whole** last-previous-note
region for a unique match, and raises when the count is not one. That rule is right for
`ledger_reconcile`, where a value occurs once, and wrong for `aggregate_report`, where repetition
*is* the data: a note carries a running list (`values so far: 18, 14, 75, 72, 18`) and then names
the same numbers again in its subtotal arithmetic (`computing 72 + 18 + 79`). Measured offline
with no model: the Chief finds 202 of 720 notes over the test split refused as ambiguous; over the
fifteen tasks this run selected I find 41 of 195 notes (21%) ambiguous, and **all fifteen tasks
carry at least one**. `run_patch_probe` raises on the first such case and the CLI catches it at
section level (`patch.py:2416`), so one unlocatable case abandons the whole section.

**Ruled (Chief, 2026-09-05; written here because the scorer is the Head of Interpretability's
domain).**

1. **Locate by field occurrence, not by uniqueness over the region.** Each note value is matched
   within its own field's span (the `_VALUE_FIELD` match) in list order, so the k-th number under
   a label has one position by construction. Subtotal arithmetic outside a list field is never a
   match candidate. R25's `value_spans` keeps its meaning.
2. **Skip, never abandon.** A case whose note still cannot be located is skipped with its reason
   recorded per case, and the section scores the rest, reporting scored and skipped counts. The
   section refuses only if fewer than the minimum remain.
3. **Two numbers, five and fifteen (Chief, 2026-09-05, correcting #26).** Score **every
   placeable case of the fifteen**; **refuse only below five** scored cases, the primary's own
   count; report scored and skipped counts **out of fifteen**, with the reason per skip, and the
   interval over the scored count. **Fifteen** is the population R30's HEAD-alone basis selects
   and is what this run selected.

   *There is no "ten".* The figure entered #26 by transcription from a size estimate in the run
   plan and was then read as a designed count. It predates R30 and never described the
   population. #26 is corrected. This is the third figure today that was right when written and
   outlived its ruling, and all three survived in prose rather than in code, where nothing
   recomputes them.
4. **An unscored section is not a clean run.** A run whose secondary section is unscored does not
   end `status=ok`: the `end` event and the health block carry the unscored section, and the
   markdown says so at the top.

**Why (1) rather than (2) alone, corrected against my own first reading.** I initially reported
that thirteen of fifteen cases would score under exclusion alone, having measured duplicates only
*within* the values-so-far list. The locator searches the whole note, where the arithmetic repeats
values, and on that measure every one of the fifteen tasks carries an ambiguous note. Exclusion
alone could therefore leave too few cases to read, which is why the locator changes and exclusion
becomes the fallback for the residue.

## 6. Deferred

P3 (sense of being on track) and P4 (concept injection) wait for a policy from SPEC-003 and for
rollouts to be permitted; P3's rollout collection becomes the natural first use of the fixed
rollout stage. The rollout and DPO stages inherit the SPEC-002 integrity filter and the
`chosen = raw` fix before any use.

## 7. Acceptance

- §1 runs to completion on the saved npz in under ten minutes on CPU and its tables carry
  intervals for every number.
- §2-5 have unit tests on the fake models for: split disjointness, the two stripping modes, the
  paired bootstrap, block-mask coverage (every layer in exactly one block), the `replace` mode of
  the hook, and the P6 position grouping on a synthetic context.
- Every result file records the resolved `ModelSpec`, the control it was reported against, and
  the command line.
