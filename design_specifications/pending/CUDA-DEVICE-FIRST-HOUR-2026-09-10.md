# The device's first hour

The Director's ruling (plan §16.12): all subsequent runs use the rented GPU; the laptop is for
tests; bugs found only on the device are expected, resolved there, recorded there, and their tests
come back to the tree. This is the one document that hour runs from, in execution order, each item
with the number it must produce and the laptop figure it is compared against. Every seat's record
carries its own section in the same shape; this page points at them and fixes the order.

## 0. What the device needs before anything runs

| item | how | who |
|---|---|---|
| the repository at `cuda-migration` (WS-A, WS-B merged; WS-C and WS-D merge in before the hour) | `git clone`, `git checkout cuda-migration` | Chief merges, seats push |
| the environment | `uv sync --extra cuda` installs torch ≥ 2.14, transformers 5.16.1, accelerate, peft, trl, safetensors and the pinned upstream `jlens` (git, 581d398); MLX is not installed and must not be | Chief (pinned tonight) |
| the checkpoint | `google/gemma-3-4b-it` into the primary checkout's `.cache/huggingface` (`configure_local_cache` sets `HF_HOME` there): the Director's token, or a copy of the laptop's snapshot directory; the repository's `models/` conversions are **not** needed and are refused by the torch loader by name | Director |
| the registry entry | `gemma3-4b-cuda-bf16`, `hf_id: google/gemma-3-4b-it`; nothing to edit | on main |
| the records and calibration inputs | in git: the stage-two golden records, `research/acceptance/MANIFEST.md`, Codex's `calibration-token-ids.json` | on the branches |
| the rendered training data for arm 1 | gitignored; copied from the laptop | Director / SWE-2 |
| the box discipline | `runlock` works unchanged on Linux (`ps -Ao`, `ru_maxrss` in KB handled); every run under `runlock run --seat <seat> --purpose ... --minutes ... -- <command>` | all |
| the suite | `uv run pytest` — expect the line "16 test files not collected: MLX is not installed on this box"; that is correct and not a pass of those files | all |

Environment for every run: `LLL_BACKEND=torch`, `LLL_DEVICE=cuda` (or `cuda:N`), and `device.pin(seed, attention="eager")`
called before the first CUDA use in every process, so `CUBLAS_WORKSPACE_CONFIG` is set before
cuBLAS initialises; `device.describe()` printed into every record.

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

`torchrun --nproc_per_node=<N> research/records/CUDA-WS-C-2026-09-09/stage_agreement.py --mode
fsdp2 ...` against `--mode plain`, then the stage on the real recipe:

| what | must produce | laptop basis |
|---|---:|---:|
| joined FSDP2 gate under NCCL, N ranks | per-parameter gradient agreement at step zero to float32 epsilon; values after N steps | 1.24e-07 gradients, 0.0 values, two `gloo` CPU processes |
| arm 1's recipe at both checkpoints, 800 and 1,200, full fine-tuning, scored on the 180-task split | both scores beside their MLX counterparts; full-split passes is the pre-registered criterion | MLX: 159 and 175 of 180 |
| the R60(c) memory measurement; `reshard_after_forward` as a rung | measured, then chosen | none |
| every departure from the MLX recipe | in the run manifest | listed in the WS-C record |

## 4. WS-D — the lens un-port's golden test (D-CRO)

The finite-difference lens against upstream's exact-autograd lens on the same corpus and
checkpoint, both float32 on disk, through `fit_upstream_jacobian`; the command and its record are
named in the WS-D record when `cuda-ws-d` merges.

| what | must produce | laptop basis |
|---|---:|---:|
| FD versus exact, per layer, at the fit's context length | the residual, with the storage floor declared | adapter fixture tests only |
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
- **Provenance as fields.** Every number carries `basis: measured-here | laptop-basis | expected`;
  every record's head carries `device.describe()`, the checkpoint sha and the source commit.
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
