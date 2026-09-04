# Probes checklist Section B — readiness assessment and implementation brief

**Deputy → Chief, 2026-09-05, for review before dispatch.** Requested by the Director: audit
the checklists' currency (separate auditors, corrections applied on their return), assess
Section B of `live/PROBES-READINESS-CHECKLIST.md`, and if any part can start, plan it and
brief it. Governing text: SPEC-004 §1, §2, §4, §7 (`pending/SPEC-004-probe-program-revision.md`),
R18 item 4 (C7) and R26 (wiring map §7), issue #15's Chief comment defining C7.

## 1. Readiness of each Section B item

| item | what it needs | lane? | can start now? |
| --- | --- | --- | --- |
| B1 §2 P2 redesign | code (splits, disjointness test, `--stub-observations`, dual capture positions, `compare`), then gated runs of ~65 min per (policy, condition) on the 3B | runs yes; code no | **code yes**, fake-only |
| B2 §3 block ablation run | execution only (code complete, `ablation.json` absent) | yes, ~30 min per screen | after P6 releases the lane; Director's ordering vs B4 training |
| B3 §5 P6 run | in flight (1 h 00 at this writing, no artifact yet) | occupying | — |
| B4 §4 P1 closure + fixes | offline record from 288 saved rollouts; three fake-only code fixes; then a ~20 min gated `build` on the 4B | code no; run yes | **record and fixes yes** |
| B5 C7 BF16 refit | offline numpy on the saved npz through the existing reanalysis; no model | no | **yes** |

Three slices can start now without the lane and without touching anything a live run uses.
The lane order after P6 is the Director's: B4 training (~100 min + ~70 min eval) is first by
the Director's own preference for a health record from iteration one; B2 (30 min) and the P1
`build` (20 min) fit either before or after it; the P2 runs (8.7 h for the decisive pair on
two policies) come last and only after their code lands.

## 2. Slice B5 — C7 BF16 refit (offline; smallest; recommended first)

**Definition (issue #15, Chief; R18 item 4):** the saved P2 npz holds FP32 activations. Round
them to BF16 precision and back, refit through the existing reanalysis pipeline, and compare
the margins and Holm flags against the ratified table. Unchanged supported set → a measured
bound on the precision gap, recorded as immaterial with evidence; any flipped flag → a finding
about the fragility of those margins.

**Inputs on disk (read-only):** `outputs/probes/state-corrected-hardened-20260903T172549/`
`state-base-mix.npz` (108 MB; keys `layer_{6,12,18,24,30,35}` each `(2367, 2048)` float32;
`label_*` targets; `task_ids`, `family`, `step_index`, `difficulty`, `meta`) and the baseline
`state-base-mix.reanalysis.json` (schema with `analyses.{all_rows,sft_disjoint}.targets.
<target>.layers.<layer>.{holm_adjusted_p, interval_supported, holm_supported}` and a `readme`
naming the surviving set: pending_count at 12/18/24; phase at 30/35; hidden_error at all six;
next_tool at 30/35; first_bucket_count at 6/12/18/30).

**Work:** new subcommand `agent-v2-probe-state refit-bf16 --input <npz> --baseline <reanalysis.json> --output <dir>`
in `probes/state_probe.py`: load the npz; round every `layer_*` array through bfloat16 and
back to float32 (`mx.array(x).astype(mx.bfloat16).astype(mx.float32)` or a numpy
bit-truncation helper — one implementation, tested against the other on a small array; no
model); run `reanalyse_dataset` with the SAME `split_seeds`, `bootstrap_resamples`,
`data_seed`, `generator_version` the baseline's `metadata` records; write
`state-base-mix.reanalysis-bf16.{json,md}` plus `refit-comparison.{json,md}`: per (analysis,
target, layer) the baseline and refit margins, both support flags, and the flip set; a
one-paragraph verdict. Determinism: the same seeds must reproduce the baseline exactly when
rounding is disabled (`--no-round` control, asserted in a test on a fake npz).

**Invariants:** no model execution; read-only under `outputs/` except the new output dir;
fake-only tests (`tests/test_state_probe.py`: rounding helper exactness on known bf16 values,
`--no-round` reproduces a baseline bit-for-bit, flip-set computed correctly on a synthetic
pair); R26(g) progress per analysis/target; RunLog with input SHA-256s in the identity.
Runtime on CPU: SPEC-004 §7 bounds the reanalysis at ten minutes; the refit is two of them.

**Gate to run:** none beyond the Chief's ratification of the slice; the Director need not
lift execution because nothing executes.

## 3. Slice B4 — P1 closure record and the three pre-registered fixes (offline + fake-only)

**Closure record (SPEC-004 §4):** write `outputs/probes/axis-corrected/CLOSED.md` from the
288 saved rollouts (`rollouts-base.jsonl`: fields `exemplar, prompt, rendered_prompt,
response, role`; 25 roles, the default assistant with 96 rows and 24 roles with 8 each),
scoring each with `heuristic_expression_score` (`assistant_axis.py:422`) at the ratified
`min_expression = 0.34` (`:638`), reporting per-role counts of rollouts at or above threshold
and the decision the spec pre-registers ("the coder base has no usable persona space; no
further P1 work on qwen25-coder-3b"). The spec's expected counts (best role 3/8; 19 of 24 at
0/8) are re-measured, not copied; a discrepancy is reported, not reconciled. Produced by a
small script under `scripts/` or a `close` subcommand — Chief's preference; the record cites
the rollouts file SHA-256 and the heuristic's file:line.

**Three code fixes, fake-only, each red-first (anchors from the code map, re-read by the Deputy):**
(1) matched design — `collect_rollouts` gives the default assistant the full `prompts` list
(`assistant_axis.py:600-609`) while each role gets `prompts[:role_prompts]` (`:610-623`,
`role_prompts` default 8 at `:587`, `--prompts` default 96 at `:1365`); the fix gives the
default persona the same eight prompts. (2) exemplar balance — the 24 roles are six high, eight
low, ten neutral (`:106-107`, `:384-392`), and exemplar PRESENCE is what is unbalanced: none of
the six high roles passes an exemplar, all eight low roles do, six of ten neutral do
(`_role(...)` sixth argument; `ROLE_EXEMPLARS` built only when present, `:387-389`;
`rollout_role` prepends it). **Ruling requested:** balance by giving every role an exemplar
(authoring six high + four neutral) or by dropping exemplars from all — the Deputy recommends
dropping them, so the axis derives from the system prompts alone and the rollouts carry one
fewer confound; either way a test asserts equal presence across the three groups. (3) judge
off — `--judge` is `BooleanOptionalAction, default=True` at `:1378-1383` (the checklist's
`:1346-1350` anchor is stale; `:1346` is `def main`), `build_axis_run(use_model_judge=True)` at
`:639` while `build_axis` defaults to `False` at `:749`; the fix makes both default off and
records `diagnostics["model_judge"]` (`:665`) as before. Tests pin the prompt-count equality,
the exemplar balance, and the defaults. No payload key changes; the `build` and `project`
CLIs keep their flags. Files: `probes/assistant_axis.py`, `tests/test_probes_axis.py`, plus
the closure script.

**Gate to run:** the ~20-minute `build` on qwen35-4b waits for the lane and a Director lift;
proceed to `project` only if `axis_verdict` passes (`:709-738`: `pc1_cosine_abs ≥ 0.5` and
`split_half_cosine ≥ 0.9` at `:719`).

## 4. Slice B1 — §2 P2 redesign code (fake-only; the largest; three sub-slices)

**What exists today (code map, Deputy-verified anchors):** `build_probe_dataset`
(`state_probe.py:381-397`) replays each task with `build_rows(task, keep_last=keep_last)`
(`:464`), renders with `build_prompt(..., keep_last=len(context))` (`:465-468`; the messages are
already windowed) and captures ONLY the last prompt token (`capture_residuals(...,
positions="last")`, `:471`; `capture.py:43-44`). Observations are stubbed by
`window_messages` (`protocol.py:157-173`) with `hidden_observation` text (`:152-154`);
`keep_last=0` already stubs every observation — no flag exposes it (`--keep-last` at
`state_probe.py:2891` defaults to 2). `--strip` (`:2879`) removes note state fields only. Splits
come from `--splits` / `--mix-difficulty` with `MIX_PLAN = (("train", None), ("p2mix", None),
("test", False))` (`:106`); `make_tasks` already accepts `perturb` and `difficulty`
(`tasks.py:96-103`); no `p2-d0/1/2` split exists. The only SFT-disjointness logic is the
`train-` prefix cohort in `reanalyse_dataset` (`:2190-2215`, R7); no test compares against
`configs/`. `main()` dispatches on `sys.argv[1] == "reanalyse"` (`:2843-2846`). npz schema:
`layer_{L}`, `label_*`, `task_ids`, `family`, `step_index`, `difficulty`, `meta`
(`save_dataset` `:725-749`, `load_dataset` `:752-775`). The token-range mean primitive exists
in `response_mean_activations` (`capture.py:101-149`, used by P1 only).

**B1a — splits and disjointness (`pipeline/tasks.py`, `probes/state_probe.py` plan, tests).**
Add `p2-d0`, `p2-d1`, `p2-d2`: 120 tasks each, explicit difficulty 0/1/2, perturbed at 0 and
1, clean at 2, drawn under the same `data_seed` discipline as every other split; a
`P2_PLAN` beside `MIX_PLAN` and `--p2` (or `--splits p2-d0,p2-d1,p2-d2`) selects them. The
disjointness test (SPEC-004 §2, §7) asserts against EVERY training split of EVERY config in
`configs/` that names a `data:` directory: by task id (trivially distinct by prefix) AND by
content — a task fingerprint (family + generated inputs, ids and split names excluded) must
not appear among the config's `train/valid/test` rows on disk, and for configs whose data is
absent (`agent_v2d`), among the rows regenerated from the config's seed. **Ruling requested:**
the fingerprint definition (the Deputy proposes SHA-256 of the task's serialised inputs with
`task_id`/split stripped) and whether absent-data configs are regenerated in the test or
skipped with a recorded reason.

**B1b — conditions and dual capture (`probes/state_probe.py`, `probes/capture.py`, tests;
sequenced after B5 lands in the same file).** (i) `--stub-observations` sets `keep_last=0` for
the capture and records the condition name in `meta["condition"]` ∈ {`intact`,
`notes-stripped`, `observations-stubbed`, `both`} from the (`--strip`, `--stub-observations`)
pair; the artifact stem carries it. (ii) Dual capture: keep the last-prompt-token capture and
add the mean over the teacher-forced note tokens of the expert target (the row's target
assistant message): render prompt + note, run once, mean-pool residuals over the note-token
positions via `response_mean_activations`' range logic; store as `layer_{L}_note` beside
`layer_{L}`, `meta["capture_positions"] = ["last", "note_mean"]`; `load_dataset` reads both
when present; `fit_probes` and `reanalyse_dataset` take `position="last"|"note_mean"` (default
`last`, so every existing artifact and test is unchanged). Layer set from the registry's
`probe_layer_fractions` (`models.py:53`; the spec's `spec.probes.layer_fractions` name does not
exist — noted, not changed). Cost: one extra forward per row over the note tokens.

**B1c — `compare` (`probes/state_probe.py`, tests).** `agent-v2-probe-state compare` bootstraps
the DIFFERENCE of margins between two policies on identical task splits and writes one table
per target. Pairing needs per-task quantities: **ruling requested** between (a) `reanalyse`
gaining an optional per-task sidecar (per split seed, per task: prediction and truth) that
`compare` pairs on, refusing result files without it, or (b) `compare` taking the two npz
files and recomputing both fits on shared splits (twice the reanalysis cost, no schema change).
The Deputy recommends (a): it keeps `compare` cheap and makes every future result file
comparable.

**Invariants for all of B1:** fake-only tests (`_planted_dataset`, `_offline_reanalysis_dataset`,
the `_Probe*` fakes in `tests/test_probes.py`); no execution; no writes under `outputs/` or
`data/`; existing artifacts load unchanged; existing tests untouched except where a signature
gains a defaulted argument; RunLog progress per task (R26 g) already present from lane D;
R16 reports. Acceptance per SPEC-004 §7: unit tests for split disjointness, the two stripping
modes, and the paired bootstrap.

**Runs (after code, after the lane frees):** `intact` and `both` first for `base` and `B` on
the 3B — the decisive pair, ~65 min per (policy, condition) — then the rest only if ambiguous.

**Observed, outside this plan:** R18's `ModelSpec.probes.capture_dtype` is not implemented
(`grep capture_dtype` hits only a test docstring); every capture is float32 by `capture.py:96`
and `state_probe.py:596-604`. B5 measures the consequence; implementing the switch is a
separate slice if the refit says it matters.

## 5. Proposed lanes and order

1. **Now, in parallel, disjoint files:** B5 (`state_probe.py` refit subcommand + its tests) and
   B4 (`assistant_axis.py` fixes + closure script + tests). B5 and B1 both touch
   `state_probe.py`, so B1's `state_probe.py` work is sequenced after B5 commits; B1's
   `capture.py`/`protocol.py`/`tasks.py` parts can start alongside.
2. Each slice: implementer → Deputy's direct review → Chief's gate (the Director's expedited
   chain, unless the Chief asks for the R19 round on B1, which is the one large enough to
   deserve it).
3. Runs, in the Director's lane order after P6: B4 training; then B2 (30 min), P1 `build`
   (20 min), and the P2 decisive pair when its code has landed.

---

## Chief's decisions (2026-09-05 05:40)

1. **Order:** (i) P6 scorer slice under R27 (new, ahead of everything: the run already paid
   for is uninterpretable without it); (ii) B5 C7 refit as specified, with `--no-round`
   control; (iii) B4 closure record and fixes; (iv) B1 in its three sub-slices. B2 and the P1
   build wait for the lane after B4 training.
2. **B4 ruling:** drop exemplars from all roles (the Deputy's recommendation); the axis
   derives from system prompts alone; a test asserts no role passes an exemplar. Closure via a
   `close` subcommand under `agent-v2-probe-axis` (RunLog, input SHA-256 in identity), not a
   script under `scripts/`.
3. **B5:** approved as written. Progress per analysis/target per R26(g).
4. **B1:** approved in principle; the P2 splits are `p2-d0/1/2` selected by `--p2`; the
   disjointness test compares against every `configs/*.yaml` training split; the observation
   stub condition is `--keep-last 0` exposed as `--stub-observations` (alias, recorded);
   dual capture positions store `layer_{L}` (last token) and `layer_{L}_note_mean`. Bring the
   sub-slice briefs for gating one at a time.
