# Probe readiness checklist

**RATIFIED by the Chief, 2026-09-04 16:00, on the Director's relay (the ratification text was lost in the 2026-09-05 regeneration; header restored; no section follows).** Companion to `TRAINING-READINESS-CHECKLIST.md`, same
conventions: updated by the Deputy after every commit, an item ticks only on a commit, a
file:line, or a checksummed artifact — re-measured, not re-read (Chief's standard, #21). Covers when probes may run and what the probe programme
(SPEC-004) still needs. Training's arm 1 has its own list; the one shared item is marked.

As of: `0b5227e` (scorer `0862c01`, R30 `68c1990`, B1b `1604f38`, B5 `eca116b`, B1c `5e1f3b2`,
R26 logging, position groups, R25), suite **1016 passed, exit 0** bare (combined tree).
Updated: 2026-09-05 evening by the Deputy — third currency audit applied (one anchor, one
closed issue, historical B4 anchors labelled, one ruling recorded, one filename). Earlier: after
the R26 commits. Every probe run now writes
`run.log` + `events.jsonl` beside its artifact and reports per outer unit (R26 g); a probe lift
request cites those files by path and SHA-256. Exception: the third P6 attempt was launched
before the R26 commits and ran the pre-instrumentation code to completion, so its artifact
(B3) has no `run.log`/`events.jsonl`.

## The one distinction that matters

**The hold applied to Qwen3.5, not to probes as such.** The 3B base and the B/C adapters were
never held: block ablation and P6 patching are code-complete, take their inputs from artifacts
already on disk, and are gated only by the single-execution rule and a per-run authorisation.
The Qwen3.5 hold stood until the preflight passed under R18a — it now has (A2) — because
every probe runs decoder blocks by hand through the view, and the view was unverified on that
architecture until then.

## Part A — lifting the Qwen3.5 probe hold

- [x] A1. **MET — R18a gate rewiring, committed `ed88c96` (issue #23).** *Shared with training
  checklist item A1.1.* `run_preflight` gates on `native_manual_vs_native` (derived floor AND
  Frobenius ≤ 1e-4; `preflight.py:390-436` in the current tree — the R32 lane-2 edits shift
  this file; cite `_native_gate_passed`/`_residual_equivalence` by name); the FP32 comparison is reported, never gated.
- [x] A2. **MET — the Qwen3.5 probe hold is lifted.** `outputs/preflight/qwen35-4b.json`
  `passed: true` under R18a at 15:20 (native max_abs 0.0, Frobenius 0.0; SHA-256
  `8a6f4298…`), with the 3B control passing identically at 15:18. Terms of #15/#16 satisfied.
- [x] A3. **Probe CLIs are model-aware** — layers from the registry with fraction support
  (`e53bd81`; `policies.py:96-99`, resolution recorded with `source`), per-model adapter
  resolution (`policies.py:46,53-54`), spec threading (`47751cf`), loader migration (wave 1,
  `a2f003c`; Part 0 of the training list is committed).
- [x] A4. **CLOSED by the C7 refit (`eca116b`, #44; issue #15 closed): supported set unchanged across 96 cells, max margin change 0.0054 — the caveat is bounded as immaterial.** Original text: probe
  activations are FP32 by rule 1.5 while the model deploys BF16 — pre-existing, applies to the
  3B equally, surfaced on issue #15. The C7 refit (round FP32 activations in the saved npz to
  BF16, refit through the reanalysis pipeline, compare Holm flags) is offline, needs no model,
  and would bound the effect with evidence. *Ruled — item 4 of R18 (the C7 refit clause; no separate "R18(c)" heading exists in the wiring map) on issue #15 — ruled; an offline implementer slice (no model), unassigned — corrected by the Chief 2026-09-05. Issue #15 is CLOSED (at `eca116b`): the preflight question by A2, the C7 item by B5.*

## Standing note, 2026-09-04 ~19:30 — two blockers found by live P6 attempts; one fixed and committed, one fixed in the tree, one ruling requested

The Director's live P6 run found that every adapter-loaded path (B2 block ablation, B3 P6,
adapter evaluation, rollout) crashed at `spec.resolve` since `a2f003c`: the view could not see
mlx-lm's `LoRALinear` wrappers. Fixed in `arch.py` by the Deputy under direct authorisation,
regression-tested on wrapped fakes, suite 642 passed; **committed `89dfb56` after the Chief's direct
read (#27)**. Base-only paths (both preflights, the render) were never affected. B2 is
unblocked. Bound follow-up from #27: an adapter-wrapped variant in the standard arch fixtures,
due before the B4 evaluation lift — not yet dispatched.

- [ ] **Bound follow-up (#27), not a Section A gate:** an adapter-wrapped variant in the standard
  arch fixtures (`tests/test_arch.py`), due before the B4 EVALUATION lift. **Dispatched
  2026-09-05 evening** (fake-only; R31 form: the shared fakes wrapped with mlx-lm's own
  `linear_to_lora_layers`, existing view tests parametrised over bare/wrapped; proof against
  the pre-fix `arch.py` in a scratch module).

The Director's second P6 attempt (after `07c6657`) then crashed in `position_groups`
("missing token span for observation"): `_groups_for` extracted observations from the full
message list while the prompt was rendered windowed (`keep_last=2`), and the two verbatim
observations tokenise differently in context (tool-response boundary merge, briefing trap 6).
**Fixed and committed `ac9c27a`** (#29; `P6-POSITION-GROUPS-FIX-REPORT.md`):
groups over windowed messages, char-span location with boundary-merge repair recorded per
group; the Deputy's independent drive on the real tokenizer matches the implementer's table
exactly on all five cases; the Chief's condition (a boundary moving more than one token raises) is in. **Behind it, verified on real data: the
counterfactual note is longer than the failing note by construction (663 vs 659 note tokens;
7 vs 5 value tokens — the extras are the dropped values) and the patcher requires equal
source/target cardinality. Ruled as R25 on issue #28** (tail alignment with residue
recorded; `note_value_tokens` → `shared_value_tokens` + `dropped_value_slot`; controls resample
post-alignment; seven cells). **The R25 slice is committed `036e62c` (#30, Chief-ratified).** The third attempt completed
(17:49–19:00 local, 2026-09-04); artifact and read under B3.

## Part B — SPEC-004 sections, in the spec's own priority

- [ ] B1. **§2 P2 redesign — 0%, and the spec calls it "where the remaining probe budget
  goes".** Nothing exists: `capture.stub_observations`, `--stub-observations`, the
  `p2-d0/p2-d1/p2-d2` splits plan, the `compare` subcommand, dual-position capture, the
  SFT-disjointness test. All fake-only code, claimable now. *Chief-approved as three sub-slices (#40; R28/R29 ruled). **B1a committed `2d9445c`** (#43): `P2_SPLITS`, `make_p2_tasks`, `task_fingerprint`, six-config
  sweep regenerated from tables, zero collisions at full size. **B1c committed `5e1f3b2`** (#45; Holm across the comparison cells is a follow-up line);
  **B1b committed `1604f38`** (#48: conditions, `layer_{L}_note_mean` with the widening span,
  R18b `capture_dtype` native by default, `task_difficulties` fix). The P2 code is complete;
  runs wait for the lane (65 min per policy-condition on the 3B).*
- [ ] B2. **§3 block ablation run** — code and tests complete (`adapter_delta.py:416-524`,
  reviewed twice); no `ablation.json` exists anywhere under `outputs/`. A 3B run on the B and C
  adapters: ~30 min per screen, execution authorisation only.
- [ ] B3. **§5 P6 patching run — run 1 artifact on disk; rerun under R27 pending** — the #22 slice is committed (`07c6657`: `--data-seed`
  override used only when the field is absent, R22 note provenance per case, R23 generator-v1
  binding, R24 `scoring_version_stable`). K1 closed on real data: five cases selected, five
  stable, decision steps 7/7/7/7/6, dropped values 85/89/100/32/54. Lift request sent
  (`P6-LIFT-REQUEST-2026-09-04.md`, approved by the Director). **Two live attempts failed on
  code, not on inputs** — see the standing note: wrapper visibility (fixed, `89dfb56`),
  position groups (fixed, `ac9c27a`), and the R25 alignment slice (fixed, `036e62c`). Nothing known stands between the code and the run.
  Bound secondary condition from #26's ratification: `aggregate_report` failures with the
  generator-v4 note as counterfactual, labelled designed-correct — after the primary run.
  **RUN COMPLETE (third attempt, Director-run, finished 2026-09-04 19:00 local):**
  `outputs/probes/patch-C-2026-09-04/patch.json` (SHA-256
  `34542df54136b66b4a2bff4d0396a55275626556ac2042d98fdc759a40146aa4`) and `patch.md`
  (`b740e17dd62e5f9dce1d7915bb8d17185e1d5d20eef99696b9dee605ad793fc6`); schema `p6-patch-r25`;
  five stable cases, none excluded; six layers × seven cells = 42; alignment table identical
  to the pre-run measurement (residue 4/4/5/4/4; slots on ";" ×4 and ","). Ran under the
  pre-R26 code: no `run.log`. **Chief's verdict (`under_review/P6-RESULT-REVIEW-round1-2026-09-05.md`,
  R27):** valid as a run record; its flip rates are NOT interpretable — the scorer counted any
  regenerated note with a wrong number as a flip through the stale-field coverage path
  (`integrity.py:384-395`), which is the whole unrelated-control signal. Quotable meanwhile:
  "the decision is perturbable only at the note positions in layers 6–18; specificity is not
  established; rerun scheduled under R27." The memo's verdict does not move. **R27 scorer slice
  (#41) dispatched**: strict per-generation outcome, every generated note recorded, visibility
  computed with the integrity module's extraction, a content control, and the `aggregate_report`
  secondary condition scheduled with the rerun (~1.5 h on the lane after B4 training).
  **Scorer committed `0862c01` (#46) and R30 refinements `68c1990` (#49, schema `p6-patch-r30`).**
  **The ledger primary rerun is RUNNING** (Director-started 2026-09-05 ~21:38, own monitor;
  `outputs/probes/patch-C-r27-2026-09-05/`, 42 cells, ~1 h 20). Run 1's artifact stays as a record.
- [ ] B4. **§4 P1 on Qwen3.5 — four pre-registered code fixes first.** *Original text
  (pre-`3867cc6`, anchors historical):* `CLOSED.md` absent; the matched-design flaw (default
  persona got the full `prompts` list while roles got `prompts[:role_prompts]`); the exemplar
  imbalance — presence: 0 of 6 high, 8 of 8 low, 6 of 10 neutral roles carried one; `--judge`
  defaulting to on. *Now:* `CLOSED.md` present; both sides use `shared_prompts`
  (`assistant_axis.py:595-620`); `ROLE_EXEMPLARS` empty; `use_model_judge` default False (`:635`). Then a ~20-minute gated run, additionally behind A2. *Fixes and closure record **committed `3867cc6`** (#42, Chief-ratified): `CLOSED.md` written by the new `close` subcommand
  (`outputs/probes/axis-corrected/CLOSED.md`, SHA-256 `fb66141c…`; measured best role 3 of 8,
  21 of 24 roles silent vs the spec's expected 19 — reported, not reconciled); matched prompts,
  exemplars dropped from all 24 roles (ruling 1), judge default off. Ticks on commit; the 4B
  `build` waits for the lane after B4 training.*
- [x] B5. **C7 BF16 refit — DONE, committed `eca116b` (#44)** — same item as A4's experiment, ruled as item 4 of R18; listed here
  because its output is a probe-programme artifact (a sensitivity bound on every P2 margin). *Implemented; real CPU run done (3 min 24 s): **supported set unchanged** — all 96
  (analysis, target, layer) cells keep both support flags, max absolute margin change 0.0054;
  rounding altered every stored activation at Frobenius-relative 1.5e-3 to 1.8e-3 per layer.
  Artifacts `outputs/probes/state-corrected-hardened-20260903T172549/refit-bf16/`
  (`refit-comparison.json` SHA-256 `a99bcd59…`, `state-base-mix.reanalysis-bf16.json`
  `fd0b0d83…`). Ruled (R23 addendum, ratified on #44, `135c48d`): the pre-versioning capture
  is bound to generator v2 — `--no-round` at v2 reproduces the baseline byte for byte (192
  margin deltas exactly 0.0); v4 flips 7 cells.*

## Explicitly not on this list

P3, P4, rollout-based probes and DPO (SPEC-004 §6 defers them until a policy from the training
matrix exists); the training arm itself (own checklist); the ratified-table history (settled —
R12's versioned replay landed at `d94a9f0` and the retroactive tools work against saved
artifacts again).
