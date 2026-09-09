# WS-C: full fine-tuning under FSDP, multi-GPU when present, optional

**From the Chief, 2026-09-09.** Read the plan §5 WS-C, §8.3, §10.2, §12.2. The Director has ruled
that fine-tuning updates the model's own weights, not adapters, and that a training run may claim
every device present. Develop on CPU torch against a tiny model for the loop's correctness; the
memory and multi-device behaviour are measured on the remote and nowhere else, per R60(c).

## Reuse before invention

The training stack is not ours to write: HF `transformers` for the model, `accelerate` for FSDP,
HF `Trainer` or a minimal loop. **What is ours and is preserved unchanged in intent:**
`depth_expansion.py` (adding layers and warming their output — a scientific mechanism),
`train_expanded.py`'s clipped `StableAdamW` (a few lines on `torch.optim.AdamW`), the checkpoint and
validation cadence, `require_dataset_manifest`, and the config schema in `configs/training/*.yaml`.
**Deleted for Gemma:** `training/gated_delta_chunked.py`, `gated_delta_chunkwise.py` (614 lines of
Qwen3.5 DeltaNet recurrence; a dense model has no such state).

## What you build

**1. `train_full.py`** (or the torch branch of `train_expanded.py` — keep the existing CLI and config
names; a new name is only for a new thing). Full-parameter fine-tuning: bf16 weights, fp32 optimiser
states, gradient checkpointing on by default, the clipped optimiser, the cadence, the manifest
requirement. **FSDP under `accelerate`** shards parameters, gradients and optimiser states across
the devices a window names; **world size 1 is the single-device path with no flag**. CPU-offloaded
optimiser states are the fallback rung when a device is short (plan §10.2).

**2. `depth_expansion.py` on torch.** Same operations, same config, same provenance record.

**3. `checkpoint_delta.py` from `adapter_delta.py`.** Per-module relative perturbation `‖ΔW‖/‖W‖`
in quadrature — the depth-why quantity — over full weight differences between a fine-tuned
checkpoint and its `base`. Same outputs, same record format, so the depth-why record's comparison
code reads it.

**4. Registry.** A fine-tuned checkpoint is a new `hf_checkpoint` entry with `base` and a
`training_lineage` step, per R57; the lens identity by lineage holds.

## Memory, measured before projected

Full fine-tuning costs roughly `params × 12` bytes before activations — ~48 GB at 4B parameters,
which fits one 80 GB device with checkpointing and does not fit 24 GB; 27B needs several devices.
**Do not project from a configuration** (the mistake of the stage-two declaration and the 21 GiB
fit): one step at the largest sequence cap the run will use, on the device it will use, then the
micro-batch chosen under the R47 fraction, and the window declaration carrying the measurement.

## Confounds and findings this stream resolves or re-measures

- **The programme's most consequential training finding was measured on adapters**: training the
  lower layers destroyed the model, top-8 beat the base (18 of 19 against 15), arm A's top-8 slice
  improved a broken model and degraded a healthy one. **Re-measure it under full fine-tuning first**
  — per-layer freezing is the full-FT analogue of `lora_layers`.
- The 2.4 MiB/token and ~6k-row ceiling were unified-memory numbers; both are re-measured.
- `adapter_delta`'s cross-adapter cosine in the rank-r core has no full-FT analogue; the quadrature
  quantity does, and the record says which.

## Golden tests

- A 40-row smoke train on CPU with a tiny model runs the loop, the cadence, the checkpoint and the
  manifest end to end.
- On the remote: one device and two devices produce the same loss curve to tolerance on the same
  rows, and `checkpoint_delta` reads the output.
- Arm 1's recipe (train top-8 layers only) reproduces its divergence result on the same checkpoint
  under full fine-tuning, or the difference is recorded as the finding.

## Budget

~180 new; ~40 edited in `depth_expansion`; `gated_delta_*` deleted; `train_grpo.py` deferred pending
the survey's ruling.

---

## Corrections from the survey (plan §13), which supersede anything above they contradict

Read plan §13 in full. The items below are the ones that change this order.
- `iters` in every training config counts **micro-batches**, not optimizer steps (`iters_unit:
  batches`); 1,200 at accumulation 4 is 300 HF steps. No test catches this; you do.
- mlx-lm's batch order (length-sort, fixed batches, permutation from numpy's global state, seeded
  at `cli.py:282-326`), padding to `1 + 32·ceil(L/32)`, and **unweighted** accumulation where HF is
  token-weighted: reproduce or **record the departure**, never silently.
- The `[mlx]` path keeps LoRA, where `scale` is literal and PEFT's is `alpha/r`; irrelevant to full
  fine-tuning, relevant if anyone compares against an MLX adapter record.
- `adapter_delta`'s exact SVD via QR of the rank-r factors has no full-FT analogue; `checkpoint_delta`
  takes the full SVD of the materialised difference and the record says so.


## The Research Division's answers (plan §14) supersede the above where they conflict

Read plan §14 and `CUDA-MIGRATION-RESEARCH-BRIEF-ANSWERS-2026-09-09.md` in full.
- **The memory was understated.** Sixteen bytes per parameter under AdamW with bf16 mixed
  precision: **68.8 GB total for 4B**, 438.9 GB for 27B. One 80 GB device is marginal for 4B.
  **Fused or chunked linear cross-entropy before the first run** — the 262k-vocabulary logits are the
  first thing to cut.
- **FSDP2** (`fsdp_version: 2`), rank-0 loading, `Trainer` owning checkpointing and
  `num_items_in_batch`. **No world-size-1 FSDP for development**: it silently zeros some gradients
  (pytorch #144045). The single-device path is plain training; FSDP is the multi-device path. Your
  golden test's "one device and two devices agree" compares plain against FSDP2, and says so.
- The `1 + weight` RMSNorm stores its weights as zeros; a live trap under mixed precision.
