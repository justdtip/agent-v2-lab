# Training readiness checklist

**RATIFIED by the Chief, 2026-09-04 16:00, on the Director's relay (the ratification text was lost in the 2026-09-05 regeneration; header restored; no section follows).** After ratification this document is updated by the
Deputy after every commit: the header's as-of line moves, and any item whose evidence changed
gets its line re-derived, never assumed — re-measured, not re-read (Chief's standard, #21). An item is ticked only when its evidence line cites a
commit, a file:line, or an artifact with a checksum. Structure follows ruling R15; nothing here
adds to or relaxes it.

As of: `0b5227e`, suite **1016 passed, exit 0** bare (combined tree with lane 2 uncommitted) (green on every bare run since; earlier note: nine consecutive green runs including
`tests/test_patch.py` alone and reversed file order; two auditors each saw one transient red in
that file during an implementer's edit window, not reproduced since).
Updated: 2026-09-05 evening by the Deputy — third currency audit applied (three anchors
re-derived; every artifact, figure and status re-measured current). Earlier: after the four R26 commits `c1f7d51`…`2405598` (A8 met; carried
by docs commits `a9c1535`/`e6500ba`), then re-audited line by line (second audit).

## Part 0 — gate mechanics (before anything else can tick)

- [x] **Commit wave 1 + its gate conditions** — committed `a2f003c` (2026-09-04 14:15),
  condition review APPROVED TO COMMIT with conditions K1–K4, all satisfied per the commit
  message (`complete/WAVE1-LOAD-POLICY-REVIEW-round1-2026-09-04.md`), suite 602 at commit.
- [x] **Scanner slice through its own gate** — ratified #21, committed `afc0905`, suite 602.

## Part A — first arm B4, the seven R15 conditions

**A1. Preflight `passed: true` under R18a** — **MET** (A1.2) — *but R32(b)/(d) add the device
working-set budget (17.8 GiB, not the registry's 22) and a calibrated training-footprint gate
with 10% headroom (lane 2, uncommitted pending calibration from the probe); the 4B artifact
must be regenerated under schema 3 when lane 2 lands, before any 4B stage.*
- [x] A1.1 **MET — committed `ed88c96` (issue #23).** `run_preflight` gates on
  `native_manual_vs_native` (criterion `native_dtype_rms_roundoff_and_frobenius_relative`,
  derived floor AND Frobenius ≤ 1e-4; `preflight.py:390-436` in the current tree — the R32
  lane-2 edits shift this file; cite `_native_gate_passed`/`_residual_equivalence` by name); `fp32_manual_vs_native` is
  reported, never gated. The override-branch test is included.
- [x] A1.2 **MET.** Both preflights run by the Director under the R18a gate (schema 2,
  criterion `native_dtype_rms_roundoff_and_frobenius_relative`): `outputs/preflight/qwen35-4b.json`
  `passed: true`, native max_abs **0.0**, Frobenius **0.0** (SHA-256
  `8a6f429809eebdc11d65202e3c9ebeea703b578ad43300c0318b776222464a79`, 15:20); the 3B control
  `outputs/preflight/qwen25-coder-3b.json` `passed: true`, native 0.0/0.0 (SHA-256
  `c536353043eb5d0eb53dcb8ecd1eb275616b398ad089bdd7c3b08cdccae36e04`, 15:18). The standing
  hypothesis resolved: fused forward and stepwise loop are bitwise-identical on both bf16
  models. fp32 gap measured, never gated: 3B Frobenius 4.2e-3, 4B 2.8e-2.
- [ ] A1.3 **SUPERSEDES A1.2's artifacts, not its finding (2026-09-05).** The two artifacts
  A1.2 cites are `schema_version: 2`; the calibrated preflight module requires 3, so both were
  regenerated on the night of 2026-09-05 and A1.2's SHA-256 digests no longer match anything on
  disk. The **finding** stands unchanged: the fused forward and the stepwise loop are
  bitwise-identical on both models, and that is a property of the models, not of the artifact
  format. Regenerated bare, one at a time on an idle lane, both `passed: true`:

  | model | snapshot revision | SHA-256 | JVP |
  |---|---|---|---|
  | `qwen35-4b` | `32f3e8ecf65426fc3306969496342d504bfa13f3` | `455aa0698266c628fe3fdb4eef8d50869cacf26b842b6b1818b832b14f5b8afb` | `finite_difference`, layer 16 |
  | `qwen25-coder-3b` | `3dd939c621c08e5753d5b89f35a2642cd83b98ca` | `4a4862c44bad17f741ca7d897ef5ec17d3a13d0b07ee7f98f2e3373a2117e965` | `forward`, layer 18 |

  `outputs/` is gitignored (`.gitignore:12`), so these artifacts are identified here by digest
  and revision rather than by a committed path, as A1.2's pair was.
- [ ] A1.4 **A bare preflight no longer satisfies a training run (2026-09-05, R32(d) as ruled
  with the calibration).** `require_preflight(consumer="training")` now requires the
  `training_footprint` block to be present, computed, not refused, and passed. A block that was
  **skipped** for want of a row count records `passed: true` inside itself and still bars a
  training run, because passing on it would clear the gate on a number nobody has. So
  `agent-pipeline preflight --model <name>` on its own is **no longer sufficient** for any arm's
  training gate: the arm's preflight must pass `--data <dir>` or `--max-row-tokens <int>`, and
  `--gated-delta-chunk` where the arm configures one.

  The probe side is deliberately unaffected. `consumer="view"` does not read the footprint at
  all, and `report["passed"]` no longer carries it, so a model whose training footprint misses
  its headroom still produces a usable artifact and is still probeable. A failed footprint is
  recorded rather than fatal; the preflight command exits zero and the training consumer is the
  one that refuses. Before B4 attempt 4 or either D arm, regenerate that arm's preflight **with
  the row count** and check `training_footprint.passed` in the artifact rather than the
  command's exit status.

**A2. In-repo rendered rows, rendering-equivalence green** — **MET** (R14).
Evidence: `cli.py:47,614` consumes `load_rendered_splits` (import, call site); byte-identical migration tests
`tests/test_data.py:252,271`; committed at `a871245`.

**A3. B4 dataset on disk with manifest hashes, `GENERATOR_VERSION`, `provenance.json`.**
- [x] A3.1 **MET — committed `94c920e` (issue #24, R21).** `guard_dataset_write` at
  `data.py:78`, `render_dataset` at `data.py:377`, B4 config repointed with `source_rows:`;
  chat-replay rows render through the `pre-expansion-policy-replay` source.
- [x] A3.2 **MET.** Rendered by the Director at 15:21: `data/agent_v2b-qwen35-4b/` with
  `train/valid/test.jsonl`, `manifest.json` (SHA-256
  `8fa5899fb43fea0ca757c4b926242b62c8b7b4100c6e03dc9e6a1736dfff5dd0`; records source dir,
  source manifest SHA `f6784deb…`, per-split source SHAs, `generator_version: null` for
  pre-versioning B, rendering thinking-off with `enable_thinking: false`) and `provenance.json`
  (SHA-256 `679520a2452d63893b898f0e7342eae2b127c5dc34d48c551b1f6f1a7467aa7a`).

**A4. Model-aware loading, four parts.**
- [x] A4.1 `load_policy` on the registry signature — **committed `a2f003c`**. Evidence:
  `evaluate.py:72-79`, twelve call sites verified by two independent reviews.
- [x] A4.2 `resolve_policy` reads `ModelSpec.policies` — committed `e53bd81`.
  Evidence: `policies.py:46,53-54`.
- [x] A4.3 `branch.build_prompt` threaded — **committed `a2f003c`** (Codex's completed work,
  preserved). Evidence: `branch.py:55,135`.
- [ ] A4.4 `_load_training_base` through `load_policy(adapter=None, lazy=False)` — the
  Chief-defined "condition-4 completion slice", which also expires `DEBT(R20)` and makes
  `mine_pairs` view/resolved required. *Defined, unassigned.* Today `cli.py:187-191` (`_load_training_base`) still
  calls `mlx_lm.load` directly and `branch.py:75` carries the `DEBT(R20)` marker. Related, not
  gating: `train.num_layers: 36` in the B4 config is stale (the 3B count) and never read —
  `lora_config` writes `resolved.num_layers` (32); one-line cleanup across the cross-model configs.

**A5. Screen present; criteria recorded.**
- [x] A5.1 `select.screen` in the B4 config — `configs/agent_v2b_qwen35_4b.yaml:44-47`.
- [x] A5.2 **MET** — `criteria:` blocks recorded verbatim from SPEC-003 §5 in all three arm configs (B4 vs B, D3 vs B, D4 vs D3), committed `7dd251d` (issue #18); recorded, never used for selection.

**A6. Execution claim free; cost stated and accepted.**
- [ ] A6.1 **Lane held by the Director's P6 rerun (2026-09-05 evening); condition 6 accepted**
  (Director: "Go"). Three attempts failed before iteration one: 1–2 on our code (fixed
  `9d5828c`, #47); 3 on Metal OOM at the first optimiser step — the library's training-mode
  gated-delta recurrence is an unrolled per-token loop (R32, #50). **Lane 1 of the remedy is
  committed `0b5227e`** (chunked checkpointed recurrence; batch 1 × accumulation 4; chunk 64).
  Attempt 4 gates (Chief, #51): the 2,257-token row steps under the device working set with
  10% headroom at the chosen chunk, AND the clean per-step time projects 400 iterations at
  ≤ 3 h; otherwise stage 2 first. The probe runs after P6 releases the lane. The ceiling
  change was reverted (`112648b`); attempts 1–3 kept on disk as records. Previously: free since the Director's third P6 attempt completed
  (17:49–19:00 local, 2026-09-04, artifact `outputs/probes/patch-C-2026-09-04/`); earlier
  execution today: preflights 15:18/15:20, P6 attempts ~17:00 and ~18:30, all Director-run.
  No live agent holds `model-execution`. A P6 run and a training run cannot overlap.
- [ ] A6.2 Cost accepted by the Director. Stated: ~100 min training (400 iters at the measured
  1.4×), ~70 min for the 180-task evaluation, memory 3.85 GiB against 22 (preflight artifact).
  The request is drafted — `B4-TRAINING-LIFT-REQUEST-2026-09-04.md`, command
  `uv run agent-pipeline --config configs/agent_v2b_qwen35_4b.yaml train` verified against
  `--help` — and is held until P6 completes, because the two runs share the single lane.

**A7. First arm is B4** — **MET** by R15 itself: Qwen3.5-4B on run B rows, thinking off,
`all-linear` keys.

**Bound follow-up (not a condition):** wave 1 broke every adapter-loaded path (the view could
not see `LoRALinear` wrappers); fixed `89dfb56` (issue #27). #27 binds an adapter-wrapped
variant in the standard arch fixtures **before the B4 evaluation lift** — dispatched
2026-09-05 evening (fake-only, R31 form).

**A8. Run logging in place (R26, Director's amendment 2026-09-05 03:40).** Training does not
start until lane A (`src/local_llm_lab/runlog.py`) and lane B (training integration:
`run.log`, `events.jsonl`, `health.json` on every exit path, `non_finite_loss` and
`incomplete_run` aborts, `train.health:` config) are committed through the R19 chain with
fake-only tests green. Evidence when ticked: the two commit hashes and the test names.
- [x] A8.1 **MET — lane A committed `c1f7d51`** (#36): `src/local_llm_lab/runlog.py`, 70 fake-only
  tests in `tests/test_runlog.py` (`test_events_stay_strict_json_when_a_logged_field_is_not_finite`
  among them), scanner green.
- [x] A8.2 **MET — lane B committed `88ecac0`** (#37): `stage_train` opens the `RunLog` before the
  base model loads (`cli.py`, `with RunLog.open(output, name="train", …)`), so the record
  starts before iteration one; `health.json` on every exit path; `non_finite_loss` aborts with
  no provenance; `incomplete_run` on a short run or a missing final checkpoint, with the
  Chief's condition that the checkpoint be newer than the run's start
  (`test_stage_train_ignores_a_stale_checkpoint_from_an_earlier_run`); eleven tests in
  `tests/test_cli.py` (`test_stage_train_keeps_the_metrics_stream_byte_identical`,
  `test_stage_train_aborts_on_a_non_finite_loss_without_writing_provenance`, …).

**The lift itself:** when A1–A6 and A8 tick (A7 is definitional), the Deputy posts the request on issue #18 with artifact
paths, SHA-256s, the recompute command, and one evidence line per condition; the Director lifts
per run. (Chief's standard, 2026-09-04.)

## Part C — arm D4: Qwen3.5-4B on the run D recipe (Director's request, 2026-09-05 late)

Same seven R15 conditions plus condition 8 (logging). Evidence lines are D4's own; nothing is
inherited from B4 except the base model's preflight.

- [x] **C1. MET — run D generated and integrity-checked.** Regenerated in the main tree through
  the sanctioned data stage 2026-09-05 and verified against the reviewed artifact: manifest
  SHA-256 `6d8f3c7036c43934bd74dcbb4b082ff480d399802b7149a08a8ed9707b2069ac`, every split byte
  for byte identical to the lane's. Original text: `agent-pipeline data --config configs/agent_v2d.yaml` →
  `data/agent_v2d` (splits train/train1/valid/valid2/test/test3, manifest, provenance,
  `GENERATOR_VERSION`); integrity invariants pass on the generated splits. No model.
  Decision recorded: chat replay kept as-is for D3/D4 (comparability with B); fixing it is
  a separate variable (SPEC-003 amendment pending).
- [x] **C2. MET — rendered for the 4B** (`data/agent_v2d-qwen35-4b`, 46/47 MB); every rendered
  split byte for byte identical to the reviewed artifact, the manifest differing only in the
  recorded source directory, which is the path it came from. Original text: `agent-pipeline render --source data/agent_v2d --output
  data/agent_v2d-qwen35-4b --model qwen35-4b` (tokenizer only); manifest hashes recorded.
  Note the longest training row now comes from `train1` (difficulty 1): expect ~2,460 tokens
  with completion, longer than B4's 2,257.
- [ ] **C3. D4 config carries the R32 memory settings:** `batch_size: 1`,
  `grad_accumulation_steps: 4`, `gated_delta_chunk: 64` (as `agent_v2b_qwen35_4b.yaml`);
  recipe numbers otherwise unchanged; `criteria:` block present (D4 versus D3).
- [ ] **C4. R32 probe covers D's longest row.** The lane probe must include the longest
  `train`/`train1` row of `data/agent_v2d-qwen35-4b` (not only B4's 2,257) at batch 1 across
  chunk 32/64/128, reporting peak and clean step time. Gate: steps under the working set
  with 10% headroom, and 400 iterations project at ≤ 3 hours; otherwise stage 2 first.
- [ ] **C5. Preflight artifact valid for the schema in force** (regenerated after lane 2
  lands; schema 2 acceptable before that per the #51 ratification).
- [ ] **C6. D3 has run (recommended, not required).** D4's criterion is paired against D3;
  without D3 the only comparison is against run B, which changes two variables (base and
  data). D3 costs ~2 h on the dense path and needs none of the R32 machinery.
- [ ] **C7. Director's lift** with the evidence format: data manifest hashes, probe numbers,
  config hash, preflight artifact hash, cost (expect ~1.4× B4's estimate at the longer rows).

## Part B — from one arm to the full matrix

- [ ] B1. Run D data generated: `data/agent_v2d` with manifest + provenance (issue #17;
  generator v4 is correct here — D *is* the new recipe). No model needed. *Unowned since the
  Codex lane died.*
- [ ] B2. Run D hash pin: the oracle (`tests/test_tasks.py`) covers only the run C config;
  extend to `agent_v2d.yaml` per R5.
- [ ] B3. Integrity invariants pass on the generated D data (R15's condition-3 clause for D).
- [ ] B4-arm. D3 (3B + D data) and D4 (Qwen3.5-4B + D data): configs exist
  (`configs/agent_v2d.yaml`, `configs/agent_v2d_qwen35_4b.yaml`); each run needs its own
  Director lift with the same evidence format.
- [ ] B5. D4-think: no config and no thinking-mode override mechanism exists yet (completeness
  assessment, SPEC-003 §4). *Deferred by the spec's own priority; conditional on D4.*
- [ ] B6. First end-to-end Qwen3.5 evaluation exercises the migrated loader against real
  weights — implicitly part of the B4 arm's eval; called out because it is the first live use
  of wave 1's code and anything found there feeds back into this list.

## Explicitly not on this list

The Qwen3.5 probe hold (lifted — both preflights passed under R18a, probes checklist A2); the FP32-versus-BF16
measurement-fidelity question and the C7 refit (issue #15, affects probe interpretation, not
training); git-lfs for `data/` (Director's open decision; the verified 3.6 GB backup of
2026-09-04 removes the single-copy risk meanwhile).
