# Where the data is: a primer for Codex, for analysis tasks

**Chief, 2026-09-10, on the Director's request.** The Director will hand you analysis tasks over the
programme's data. This says where every kind of data lives, what shape it has, how to read it,
and the rules an analysis record follows. It names nothing from memory: every path below was
listed from the tree today, and where a path is on one branch and not another, it says which.

## 1. The three places, and what is tracked

| where | tracked? | what |
|---|---|---|
| `research/records/<TOPIC>-<DATE>/` | tracked | every experiment record: a `README.md` written from the files beside it, plus the JSON, JSONL, npz, logs and scripts that produced it. **This is the data that is in git.** |
| `data/<corpus>/` in the **primary checkout** (`/Users/daniel.tipton/Desktop/An app/data/`) | untracked, ignored | the task corpora: `train.jsonl`, `valid.jsonl`, `test.jsonl`, `manifest.json`, sometimes `provenance.json`. A worktree does not have them; read them at the primary's absolute path. |
| `models/`, `.cache/huggingface/`, `outputs/` in the primary checkout | ignored | checkpoints and lens archives; the hub cache with the official Gemma snapshots and one dictionary layer; run outputs, transfer archives, box-state files. |

Two branches hold records. **Main** (`codex/agent-v2-specs`) has the laptop-era records. The
**integration branch** (`cuda-migration`) has everything on main plus the CUDA-line records; merge
`origin/cuda-migration` into your branch to see them, or read them with `git show
origin/cuda-migration:<path>` without checking out.

## 2. The records, by what they contain

**The Gemma 3 map and lens line** (main):

| record | what to read |
|---|---|
| `GEMMA3-JSPACE-MAP-2026-09-08/` (13 files) | the J-space map on the 4B: foreknowledge grids by depth and horizon, faceted by span; `DIAGNOSTIC-RERUN.md`; the pre-registration it answers is `design_specifications/pending/GEMMA3-MAP-PREREGISTRATION-2026-09-08.md` |
| `GEMMA3-MAP-PRIMARY-2026-09-09/` | the primary comparison stratified by token history, and its reversal |
| `GEMMA3-LENS-2026-09-08/`, `GEMMA3-FITTING-2026-09-08/` (12), `GEMMA3-REGRESSION-2026-09-08/` (19) | the hosted 4B lens as converted, its validation preparation, and the native bf16 regression registration; archives under `models/jlens/` (`gemma-3-4b-it_jacobian_lens.npz` with its `.json` sidecar carrying `npz_sha256` and the model identity) |
| `LIVE-LENS-PILOT-2026-09-07/` (31) | the fifteen pilot episodes with per-position lens captures, one JSONL per episode (`agentic-d<depth>-<family>-<index>.jsonl`, up to 15 MB; `.gz` twins); the stage-two capture records the acceptance kit reads |
| `LIVE-LENS-INFRASTRUCTURE-2026-09-07/` (19), `LENS-FITTING-IMPLEMENTATION-2026-09-07/` (47), `LENS-FIT-agentic-*-2026-09-07/` | how the capture path and the fitting were built and verified; registrations |
| `GEMMA3-CONFIDENCE-2026-09-08/`, `GEMMA3-OBSERVATION-ROLE-2026-09-08/`, `GEMMA3-PORT-ACCEPTANCE-2026-09-08/`, `GEMMA3-WEIGHTS-2026-09-08/` | confidence under a false premise; the observation-role decision; the architecture-view port's acceptance and the two defects it caught; the weight conversion |
| `CONFOUND-REGISTER-2026-09-08/` | every confound found in the Gemma stage-one run, the map and the regression fit, with its disposition |
| `METHOD-2026-09-08/` | the method record: thirty numbered entries of what went wrong and the rule each became; read it before writing any claim |

**Training, adapters, cost and memory** (main):

| record | what to read |
|---|---|
| `ADAPTER-DEPTH-ARM1-2026-09-08/`, `ADAPTER-DEPTH-QUALITY-2026-09-08/` (14), `DEPTH-WHY-2026-09-08/` | the top-8 adapter arm: its checkpoints on the divergence tasks, and why depth mattered |
| `ARM-A-DIVERGENCE-2026-09-07/` (28) | arm A's divergence: `descriptive.json`, `build_report.py`, the HTML report |
| `TRAIN-COST-2026-09-05/` (65), `TRAIN-EFFICIENCY-2026-09-05/`, `CTX-EFFICIENCY-2026-09-05/`, `ACCUMULATION-2026-09-08/`, `RECURRENCE-EXPONENT-2026-09-07/`, `HISTORY-CACHE-2026-09-07/` (13) | one training step per row length; memory per token; context efficiency; the accumulation gap; the recurrence exponent sweep; the history cache's acceptance (`base-acceptance.json`, `adapter-acceptance.json`, 4.7 MB each) |
| `QUANT-GAP-2026-09-05/` (19), `jlens-hosted-qwen35-4b-2026-09-05/` (52), `RETRIEVAL-CHANNELS-2026-09-06/` (18), `WP12-BROADCAST-HEADS-2026-09-06/` (56), `ATTENTION-TERM-2026-09-08/` | the Qwen-era lens work: the hosted lens on bf16 against 4-bit; read-weight measurements; the recurrent and attention channels on retrieval text; which heads mediate entry to J-space; the attention-score term |
| `INERT-GUARD-SWEEP-2026-09-09/` | the sweep of the code for guards that cannot fail |

**The CUDA line and the state programme** (integration branch only):

| record | what to read |
|---|---|
| `CUDA-WS-A-2026-09-09/` (65) | the torch seam on the real 4B: CPU calibrations (`cpu-calibration-02.json`), the three device runs (`device-gates-01..03.json`, per-unit stores, logs), `calibration-token-ids.json`, the precision diagnosis, the first-hour checklist. Numbers are `{value, basis}` objects |
| `CUDA-WS-C-2026-09-09/` (13) | the training port: the multi-device gradient gate, the smoke train, the padding finding |
| `SAE-J-BRIDGE-2026-09-09/` (9) | Stage A on the real layer-17 dictionary: `a1/layer18.json`, `a1/layer18_top10.npz` (top-10 tokens per feature through the lens), `a1/control_sweep.json` (the overlap curve at every layer), the shipped examples npz; `a1_readout.py` and `control_sweep.py` reproduce them |
| `SAE-DECODER-INTERVENTION-2026-09-10/` (41) | your own wrapper's evidence: `counterexample.json`, acceptance logs and XML, `VERIFICATION.json` |
| `STATE-PROGRAMME-FIXTURE-2026-09-10/` | the run script's fixture record: `manifest.json`, `rate.json`, `pilot/rows.jsonl`, `preregistration.json` (sealed), `main/rows.jsonl`, `estimands.json`; every number a fixture's, as the README says |
| `WSB-DEVICE-CHECKLIST-2026-09-09/`, `WSD-DEVICE-CHECKLIST-2026-09-09/`, `CUDA-WS-D-2026-09-09/` | the device checklists with the number each gate must produce; device results land beside them as the seats commit |

Device records arrive on the integration branch as each seat commits them; the card's live files
are not reachable from the laptop.

## 3. The corpora under `data/`

Each corpus is three JSONL splits and a manifest. A row is `{prompt, completion, messages,
metadata}`; `metadata` carries `task_id` (`<split>-<family>-<index>[-<variant>]`), `family`,
`variant`, `step`, `source` (`expert` or a replay source), `recovery`, `difficulty`, `perturb`.
The current corpus is `agent_v2e` (source rows) and `agent_v2e-gemma3-4b` (rendered under the
Gemma template: 6,685 train, 381 valid, 1,841 test; its manifest records `source.sha256` per
split and `outputs.<split>.sha256`, and the device reproduced those digests under the 12B entry).
`agent_v2e-qwen35-4b*` are the Qwen renders at three caps; `agent_v2b..d` are earlier
generations; `chat_replay` is **protected replay data** and, with the saved evaluations, is never
regenerated or rewritten. `data/agent_sft`, `complex_agent_sft`, `expanded_agent_sft*`, `sft`,
`rewards` are the pre-v2 corpora.

The task generator is `src/local_llm_lab/pipeline/tasks.py` (twelve families in `FAMILIES`,
recovery variants in `VARIANTS`, plus the state programme's `existence` and relation makers
outside the cycle); the environment is `pipeline/env.py`; the render is `pipeline/data.py`.

## 4. Models and lenses

Registry entries are `configs/models/*.yaml` (`gemma3-4b`, `gemma3-4b-bf16`, `gemma3-4b-cuda-bf16`,
`gemma3-12b-cuda-bf16`, `qwen35-4b`, `qwen35-9b`, `qwen25-coder-3b`), read with
`local_llm_lab.models.load_model_spec(name)`; `base_of_artifact(x)` resolves a name, hub id or
path to the base model every identity check compares. Local conversions are `models/gemma-3-4b-it-bf16`
and `-4bit` (MLX format, refused by the torch loader by design); the official snapshots are in
`.cache/huggingface/hub/`; the Gemma Scope 2 dictionary layer 17 of the 4B is there too, with
its verified digest recorded under `.cache/huggingface/dictionaries/`. Lens archives:
`models/jlens/*.npz` keyed `J<N>` for block `N`, which this repository's capture convention calls
layer `N + 1`; load them with `LensMaps.load` from `pipeline/live_lens/instruments.py`, which
checks the sidecar's digest and identity.

## 5. How to read the formats

- **Value-basis JSON.** Scalars in device and gate records are `{"value": x, "basis": "measured-here" | "laptop-basis" | "expected" | "shared-card"}`. Read the basis before the value; a number without one is a bug in the record, not a convenience for you.
- **JSONL rows.** One JSON object per line; the state programme's rows carry `arm`, `condition`, `scores` (`D1`–`D10`, `None` when unreached, which is not zero), `falsified`, `resume_key`. Use `local_llm_lab.pipeline.state_programme.record.read_rows`, which refuses a partial line by number.
- **npz.** `numpy.load(path, allow_pickle=False)`; arrays are named (`top_tokens`, `top_scores`, `J17`, …).
- **Golden episodes.** `research/acceptance/golden_trajectories.py` (integration branch) has `load_episodes(records_dir)` for the fifteen pilot episodes; `tolerance.py` and `readout_tolerance.py` hold the gate arithmetic.
- **Manifests.** `manifest.json` at the root of every corpus and record: digests, seeds, the registry name, the generator version. `require_dataset_manifest(path)` in `pipeline/data.py` is the check the train stage runs.

## 6. Rules for an analysis record

1. **Never modify a record.** Records are append-only evidence. An analysis is a new directory,
   `research/records/<TOPIC>-<DATE>/`, with the script that made it, its outputs, and a README
   written from those outputs and nothing else, naming the source record and its commit.
2. **Every number carries its basis**, and a figure computed from a laptop record is labelled as
   the laptop's; a figure from a device record as the device's; they are not averaged together.
3. **Layer numbering.** Block `N`'s output is layer `N + 1` here; lens archive key `J<N>` is layer
   `N + 1`; a dictionary hook `model.layers.N.output` reads layer `N + 1`. `sae_bridge.layer_for_hook`
   is the one place the arithmetic lives; use it rather than adding one by hand.
4. **Family constants stay out of scripts under `src/`, `research/*.py` and `scripts/`.** The rules
   test derives a denylist from the registry (`34`, `2560`, `262208`, `<end_of_turn>` for Gemma;
   `36`, `2048`, `<|im_end|>` for Qwen); scripts under `research/records/` are excluded from the
   walk, which is where an analysis script belongs.
5. **No model loads on the laptop for analysis.** Reading files needs no window; a model load
   does (`runlock run`), and the Director's rule is that runs go to the device. If an analysis needs
   a forward pass, say so and stop.
6. **Protected data** (`data/chat_replay`, the saved evaluations the tests call protected) is read
   only, and its rows are never quoted into a record beyond what a finding needs.
7. **Fixture numbers are not findings.** The state programme's committed record, the wrapper's
   counterexample and every acceptance log are instrument evidence; the README of each says so.
8. **Commit by pathspec** on your branch, push for review, and give the Director the sha and the
   record path. A record you cannot reproduce from its own script is not finished.
