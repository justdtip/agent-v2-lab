# Probe readiness checklist

**RATIFIED by the Chief, 2026-09-04 16:00, on the Director's relay (the ratification text was lost in the 2026-09-05 regeneration; header restored; no section follows).** Companion to `TRAINING-READINESS-CHECKLIST.md`, same
conventions: updated by the Deputy after every commit, an item ticks only on a commit, a
file:line, or a checksummed artifact — re-measured, not re-read (Chief's standard, #21). Covers when probes may run and what the probe programme
(SPEC-004) still needs. Training's arm 1 has its own list; the one shared item is marked.

As of: `2405598` (R26 run logging committed for every probe CLI: P6/ablation/delta `0a79788`,
state/axis `2405598`; position-groups fix `ac9c27a`; R25 `036e62c`), suite **754 passed, exit 0** bare.
Updated: 2026-09-05 by the Deputy (after the R26 commits). Every probe run now writes
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
  Frobenius ≤ 1e-4; `preflight.py:291-330`); the FP32 comparison is reported, never gated.
- [x] A2. **MET — the Qwen3.5 probe hold is lifted.** `outputs/preflight/qwen35-4b.json`
  `passed: true` under R18a at 15:20 (native max_abs 0.0, Frobenius 0.0; SHA-256
  `8a6f4298…`), with the 3B control passing identically at 15:18. Terms of #15/#16 satisfied.
- [x] A3. **Probe CLIs are model-aware** — layers from the registry with fraction support
  (`e53bd81`; `policies.py:96-99`, resolution recorded with `source`), per-model adapter
  resolution (`policies.py:46,53-54`), spec threading (`47751cf`), loader migration (wave 1,
  `a2f003c`; Part 0 of the training list is committed).
- [ ] A4. **Interpretation caveat, Chief's decision, does not gate execution:** probe
  activations are FP32 by rule 1.5 while the model deploys BF16 — pre-existing, applies to the
  3B equally, surfaced on issue #15. The C7 refit (round FP32 activations in the saved npz to
  BF16, refit through the reanalysis pipeline, compare Holm flags) is offline, needs no model,
  and would bound the effect with evidence. *Ruled — item 4 of R18 (the C7 refit clause; no separate "R18(c)" heading exists in the wiring map) on issue #15 — ruled; an offline implementer slice (no model), unassigned — corrected by the Chief 2026-09-05. Issue #15 stays open for this C7 item only; its preflight question is closed by A2. Planned as slice B5 (#40).*

## Standing note, 2026-09-04 ~19:30 — two blockers found by live P6 attempts; one fixed and committed, one fixed in the tree, one ruling requested

The Director's live P6 run found that every adapter-loaded path (B2 block ablation, B3 P6,
adapter evaluation, rollout) crashed at `spec.resolve` since `a2f003c`: the view could not see
mlx-lm's `LoRALinear` wrappers. Fixed in `arch.py` by the Deputy under direct authorisation,
regression-tested on wrapped fakes, suite 642 passed; **committed `89dfb56` after the Chief's direct
read (#27)**. Base-only paths (both preflights, the render) were never affected. B2 is
unblocked. Bound follow-up from #27: an adapter-wrapped variant in the standard arch fixtures,
due before the B4 evaluation lift — not yet dispatched.

- [ ] **Bound follow-up (#27), not a Section A gate:** an adapter-wrapped variant in the standard
  arch fixtures (`tests/test_arch.py`), due before the B4 EVALUATION lift. Unassigned.

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
  SFT-disjointness test. All fake-only code, claimable now. *Planned in three sub-slices
  (splits + disjointness; conditions + dual capture; `compare`) in the Section B brief at the
  Chief's review (#40); `state_probe.py` work sequenced after the B5 refit lands in the same file.*
- [ ] B2. **§3 block ablation run** — code and tests complete (`adapter_delta.py:416-524`,
  reviewed twice); no `ablation.json` exists anywhere under `outputs/`. A 3B run on the B and C
  adapters: ~30 min per screen, execution authorisation only.
- [x] B3. **§5 P6 patching run — artifact on disk** — the #22 slice is committed (`07c6657`: `--data-seed`
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
  pre-R26 code: no `run.log`. Interpretation is the Deputy's read → Chief (see the log).
  The secondary condition remains the bound follow-up.
- [ ] B4. **§4 P1 on Qwen3.5 — four pre-registered code fixes first**, none applied:
  `outputs/probes/axis-corrected/CLOSED.md` absent (288 saved rollouts on disk); the
  matched-design flaw (default persona gets the full `prompts` list, `assistant_axis.py:600-609`,
  roles get `prompts[:role_prompts]`, `:610-623`); the exemplar imbalance — presence, not only
  the 6/8/10 role counts at `:106-107`: 0 of 6 high, 8 of 8 low, 6 of 10 neutral roles carry
  one; `--judge` defaulting to on (`assistant_axis.py:1378-1383`; `build_axis_run` default at
  `:639`). Then a ~20-minute gated run, additionally behind A2. *Planned: Section B brief
  `under_review/PROBES-SECTION-B-PLAN-2026-09-05.md`, at the Chief's review (#40).*
- [ ] B5. **C7 BF16 refit** — same item as A4's experiment, ruled as item 4 of R18; listed here
  because its output is a probe-programme artifact (a sensitivity bound on every P2 margin). *Planned as the first Section B slice — fully offline (the saved npz is float32 on six
  layers; the baseline reanalysis carries `holm_supported` per target/layer) — in the Section B
  brief at the Chief's review (#40).*

## Explicitly not on this list

P3, P4, rollout-based probes and DPO (SPEC-004 §6 defers them until a policy from the training
matrix exists); the training arm itself (own checklist); the ratified-table history (settled —
R12's versioned replay landed at `d94a9f0` and the retroactive tools work against saved
artifacts again).
