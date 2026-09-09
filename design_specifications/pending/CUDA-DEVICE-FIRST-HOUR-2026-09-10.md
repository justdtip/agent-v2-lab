# The device's first hour

The Director's ruling (plan §16.12): all subsequent runs use the rented GPU; the laptop is for
tests; bugs found only on the device are expected, resolved there, recorded there, and their tests
come back to the tree. This is the one document that hour runs from, in execution order, each item
with the number it must produce and the laptop figure it is compared against. Every seat's record
carries its own section in the same shape; this page points at them and fixes the order.

## 0. What the device needs before anything runs — `lab-device`

One command, five parts, in this order. `uv run lab-device --help` lists them; each writes what it
found rather than what it was asked.

| step | command | what it does |
|---|---|---|
| clone | `git clone … && git checkout cuda-migration` | the integration branch carries every workstream |
| install | `uv sync --extra cuda` | torch ≥ 2.14, transformers 5.16.1, accelerate, peft, trl, safetensors and the pinned upstream `jlens` (git, 581d398); MLX is not installed and must not be |
| token | `uv run lab-device login` | prompts for the Hugging Face token (hidden) and hands it to `huggingface_hub`'s own store under the project cache; this code never keeps it. Do not copy the cache directory between machines; log in on each |
| weights | `uv run lab-device fetch --dry-run`, then `uv run lab-device fetch [--mode inference\|lora\|full] [ids…]` | for each id (default: Gemma 3 4B, 12B, 27B; Qwen3.5 4B, 9B — Gemma 3 has no 9B) reads the hub's metadata, prints whether it fits the device's R47 budget at 2, 2.5 and 16 bytes per parameter, downloads those that fit under the chosen mode, and refuses an MLX conversion by its format. The registry entry for the 4B is `gemma3-4b-cuda-bf16`; new sizes get entries and a `family` row in the rule test before they are run |
| data | on the laptop `uv run lab-device pack-data data/agent_v2e-gemma3-4b`; on the device `uv run lab-device verify-data <archive> --dest data` | the task corpus rendered under the Gemma template (6,685 train, 381 valid, 1,841 test rows; split digests d7feef2e…, 53339592…, 6518f957…). The 4B, 12B and 27B tokenizers are byte-identical by the hub's hashes, so this one render serves all three; the device re-render under `gemma3-12b-cuda-bf16` (`agent-pipeline render --source data/agent_v2e --output data/agent_v2e-gemma3-12b --model gemma3-12b-cuda-bf16`) must reproduce those digests, and that is asserted, not assumed |
| dictionaries | `uv run lab-device fetch-dictionary google/gemma-scope-2-4b-it --layer 17 --layer 21` (and the 12B repository likewise) | Gemma Scope 2 residual dictionaries, `resid_post_all` at every layer, 16k width, about 336 MB per 4B layer and 504 MB per 12B layer; gated, so after `login`. Not on the laptop: the caches hold only their config files. The bridge order (`SAE-J-BRIDGE-ORDER-2026-09-08.md`, third amendment) names the layers the map makes interesting; the config's hook string is verified against our layer convention (block N's output is our layer N+1) by a test, not a comment |
| preflight | `LLL_BACKEND=torch uv run lab-device preflight --json outputs/preflight.json --data data/<dataset>` | environment (versions, upstream commit, MLX absent), backend and CUDA devices with memory, determinism pinned before the first CUDA use and read back, every torch-loadable registry checkpoint in the cache with its format and text bytes and a feasibility verdict, the dataset's manifest and splits, git HEAD and tree state, the box window. Every row carries a basis; exit is non-zero on any FAIL and names it. Gate 1 does not start on a failed preflight |

Environment for every run: `LLL_BACKEND=torch`, `LLL_DEVICE=cuda` (or `cuda:N`), and `device.pin(seed, attention="eager")`
called before the first CUDA use in every process, so `CUBLAS_WORKSPACE_CONFIG` is set before
cuBLAS initialises; `device.describe()` printed into every record. Every run under
`runlock run --seat <seat> --purpose … --minutes … -- <command>`. The suite: `uv run pytest`, and
expect "16 test files not collected: MLX is not installed on this box", which is correct and not a
pass of those files.

## 1. WS-A — the view on the real checkpoint, gates 1–4 (Codex)

`research/records/CUDA-WS-A-2026-09-09/cpu_gates.py --execute --checkpoint <snapshot> --token-ids
calibration-token-ids.json --output <record> --cap-gib <device cap> --projected-peak-gib <basis>
--source-commit <hash>`, adapted for the device per the WS-A order's review amendment:

| what | must produce | laptop basis |
|---|---:|---:|
| seam exactness: bf16 loop through `_block` against bf16 native, 64 and 1,400 tokens | exactly 0 | fixture: float32 loop vs native 0.0 |
| the cross-precision floor: promoted loop against bf16 native | declared, not gated | 1.24% at 64 tokens; rotary rounding 2^-9 |
| controls at 1,400: mask dispatch, hook-site off-by-one, entry-transform omission | each fails outside the floor | fixture mask control 21% vs 1% floor |
| float32-loaded control (14.5 GiB fits here) | loop vs native under the 1e-3 bound | fixture 0.0 |
| peak memory | measured, beside the projection | 9.8 GiB projected on CPU |

Then §6.3, the graph-once estimator, written and fixture-tested on the laptop, run here.

## 2. WS-B — trajectories, readout, the tolerance gate (SWE-1)

The WS-B checklist (`research/records/WSB-DEVICE-CHECKLIST-2026-09-09/README.md`) is this section
in full, in SWE-1's order. **The number most likely to be misread in the hour:** the bookkeeping
join reports 5,245 of 5,245 and looks like the strongest figure in the record; it is evidence of
nothing except that the reader and the writer share a position convention, and it is labelled
bookkeeping in the checklist, the kit's output and the manifest. It is not a backend result.
Gate 6's producing side is not ported, so it reports unavailable whatever the input, and the
smoke pass says so rather than implying the gate is covered.

`scripts/tolerance_baseline.py --records <stage-two records> --json <out>` over all fifteen
episodes (no `--episode`), then `scripts/acceptance_gates.py --model gemma3-4b-cuda-bf16`:

| what | must produce | laptop basis |
|---|---:|---:|
| argmax agreement, every episode, flips with probabilities and the confident count | flips only where precision puts them; 0 at P ≥ 0.99 with the count stated | `calculate-0158`: 98 of 103, 5 flips in the 25 unconfident, 0 of 78 confident; the eleven short episodes as reported |
| the four long episodes (`update-0028` first, 2,607-position turn) | the same signature beyond the 1,024 window | none: unrun on the laptop, the point of the gate |
| within-backend determinism: `calculate-0158` twice on the device | identical flip positions | same on CPU |
| gates 5 and 6 | PASS with their measured bands | kit prints them |
| peak memory | measured | 7.88 GiB against 7.56 projected on CPU |

Cache strategies stay refused until each reproduces the `none` trajectories byte for byte here.

## 3. WS-C — training, sharded (SWE-2)

The WS-C record (`research/records/CUDA-WS-C-2026-09-09/WS-C-RECORD.md`) carries this section in
full; its order is the one to run, cheapest and most diagnostic first, so an hour that goes wrong
goes wrong early and for a nameable reason. The device arms compare GPU against GPU, so every local
figure is the expected magnitude and never a bit target: kernels and reduction order differ, and an
identical number would be a coincidence.

| step | must produce | laptop basis | a mismatch means |
|---|---:|---:|---|
| 1. the environment says what it is (seconds, no model) | the determinism block | the CPU block, every field but `device` matching | `pin()` ran after something touched the device, and every later number inherits it |
| 2. the sharded path under NCCL (minutes, tiny checkpoint) | per-parameter gradient deviation at step zero | 1.24e-07, same script | above ~1e-5 is the port, not arithmetic; first check both arms consumed the same rows. **Never accept a loss-curve agreement in its place**: the broken configuration gave a bit-identical loss beside a gradient wrong by 1.67 relative |
| 3. the memory arithmetic meets the device (one step at the cap, real checkpoint) | measured peak | 16 bytes per parameter, 68.8 GB at 4.30B; logits 2.82 GB unchunked against 0.54 GB at chunk 512 | above: the trainable slice is larger than intended; far below: the freezing opened less than intended |
| 4. the reshard rung (`reshard_after_forward` both ways) | the rung, measured | none, and none claimed | this step creates the rung |
| 5. arm 1 at both checkpoints, full fine-tuning, the 180-task split | both scores beside their MLX counterparts | 175 of 180 at 1,200; 159 at 800; full-split passes is the pre-registered criterion | the four moving variables are stated so a difference is not read as a failed reproduction |

## 4. WS-D — the lens un-port's golden test (D-CRO)

The finite-difference lens against upstream's exact-autograd lens on the same corpus and
checkpoint, both float32 on disk, through `fit_upstream_jacobian` on `cuda-ws-d` (5121083, 35
tests, all four adversarial fixes). The D-CRO's device checklist is
`research/records/WSD-DEVICE-CHECKLIST-2026-09-09/` and is this section in full.

| what | must produce | laptop basis |
|---|---:|---:|
| FD versus exact on a **declared subset** of rows and positions, the exact side re-run on the same subset, per layer | the residual, with the epsilon and the storage floor declared; no threshold | tiny decoder: 3.6e-3 worst layer at epsilon scale 0.01, halving the step divides it by 3.9, transposed 250x worse |
| exactness, upstream exact re-run on the same rows | exactly zero | fixture: zero |
| the three controls, transposed, layer-shifted, wrong-corpus | each disagrees by more than the candidate | fixture: refused otherwise |
| the orientation check through upstream's transport | passes on the written artefact | fixture |

## 5. Rules of the hour, learned on the laptop tonight

- **Every tool writes each result as it completes and flushes.** SWE-1's runner accumulated
  eleven episodes and wrote once at the end; an interrupt after ten minutes recovered nothing.
  On the device that is a paid hour that cannot say what failed or where. A tool that writes once
  at the end is not allowed in the hour.
- **Resumable, keyed on what was measured.** A gate's record counts only for the same source
  commit, the same checkpoint sha and the same `device.describe()`; the kit skips a gate whose
  record matches and re-runs one whose record does not, saying why.
- **Smoke pass before full pass.** Every gate on its smallest input first, so an hour that dies at
  minute fifty has touched every gate once.
- **Provenance as fields, per cell and not per row.** Every number carries
  `basis: measured-here | laptop-basis | expected`; every record's head carries `device.describe()`,
  the checkpoint sha and the source commit. On one row the halves differ: "0 flips at P ≥ 0.99"
  is measured on the device while "of 78 confident positions" comes from the MLX recording and
  is not re-measured there; a reader who cannot see which is which takes the row as one
  measurement (SWE-1). A field without a kind is refused.
- **Resume keys on the tree that ran**, not on time: the working tree's content including
  untracked files, the checkpoint sha, the device reading and the gate's input; a modified tree
  resumes only its own records, and the refusal names every field that differs.
- **Only a pass resumes.** A failure or an unavailable is a thing to try again, not to inherit; the
  store keeps every result so the history is readable, and resume reads only the successes.
- **Stop at the first failing gate** with what it saw against what it expected.
- **A projection is a basis.** Every laptop figure above is what the device number is compared
  against, never what it is expected to equal; and a laptop figure taken on a shared box is not a
  basis for anything (a 667 s idle projection ran past 630 s at load 6.4).

## 6. What will probably break first, so it is not a surprise

- `device.py`'s CUDA branches have never executed; every one is tested only through stubs.
- `hf_text` loads on the CPU and moves, doubling the transient; the `device_map` rung is not taken.
- `pin()` must run before any CUDA use in every process, including `torchrun` workers.
- NCCL reduction order is not `gloo`'s; the gates compare per parameter, not by norm, for that reason.
- The acceptance kit's `--model` needs an announced window; on this box that is `runlock run`.
- Anything named in a laptop projection is a basis, not a number; every peak is re-measured.

When one of these breaks, the fix is made on the device, the record says so, and its test comes back.
