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
names; a new name is only for a new thing). Full-parameter fine-tuning: fp32 master weights and fp32 optimiser
states with bf16 compute (amendment A1), gradient checkpointing on by default, the clipped optimiser, the cadence, the manifest
requirement. **FSDP under `accelerate`** shards parameters, gradients and optimiser states across
the devices a window names; **world size 1 is the single-device path with no flag**. CPU-offloaded
optimiser states are the fallback rung when a device is short (plan §10.2).

**2. `depth_expansion.py` on torch.** Same operations, same config, same provenance record.

**3. `checkpoint_delta.py`, scaffolded on `adapter_delta.py`, computing the depth-why quantity as
`research/records/DEPTH-WHY-2026-09-08/adapter_geometry.py` computes it (amendment A2).** Per-*layer*
relative perturbation with numerator and denominator aggregated separately in quadrature over the
layer's adapted modules, `sqrt(Σ_m ‖ΔW_m‖²) / sqrt(Σ_m ‖W_m‖²)`, over full weight differences between
a fine-tuned checkpoint and its `base`. **Not** per-module ratios averaged or summed in quadrature:
that inflates by roughly the square root of the module count, the mistake that script's author made
and recorded at its line 7. The delta record has no reader anywhere in the tree; `checkpoint_delta`
writes the same per-layer table the script's `cols` dict holds, so the two records read side by side
and no shared reader is required.

**4. Registry.** A fine-tuned checkpoint is a new registry entry whose `hf_id` (the field's real
name; `hf_checkpoint` exists nowhere in the tree, amendment A3) sits beside `base` and a
`training_lineage` step, per R57; the lens identity by lineage holds. **WS-E adds registry entries,
not WS-C**: `tests/test_pipeline.py` pins the model list literally and moves in the same commit.

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
- One device and two devices agree **per parameter** on the same rows: gradients at step zero and
  parameters after N steps to float32 epsilon, as a max relative deviation per parameter; the loss
  curve is reported beside them and passes nothing on its own (amendment A6). Answerable on CPU
  under `gloo` before any device is rented; `checkpoint_delta` reads the output.
- Arm 1's recipe (train top-8 layers only) reproduces **both** its checkpoints, 800 and 1,200, under
  full fine-tuning, each scored on the full 180-task split beside its MLX counterpart, or the
  difference is recorded as the finding. The criterion is pre-registered here, before any
  re-measurement (amendment A4): full-split passes select; validation loss monitors.

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

## Amendments, 2026-09-09 evening — from SWE-2's eleven-agent mapping of the tree

Four of SWE-2's findings changed this order's text above; the changes are in place and this section
says what moved and why, so the order does not silently read as if it had always been right.

**A1. Optimiser precision.** The order said "bf16 weights, fp32 optimiser states" as if the second
followed from the first. It does not: `torch.optim.AdamW` allocates its moments in the parameter's
dtype, so a bf16 trainable parameter gets a bf16 second moment, nothing warns, and the loss still
falls while small gradients round away in eight mantissa bits. The clipped optimiser the order asks
to preserve *is* fp32 moments, and the only way to have them is an fp32 master parameter: FSDP mixed
precision (fp32 sharded parameters, bf16 compute) or autocast over fp32 weights. The loop now
**refuses a bf16 trainable parameter** rather than accommodating one. Sixteen bytes per parameter
already assumed this; the words did not.

**A2. The `checkpoint_delta` quantity.** The order named per-module ratios in quadrature. That is the
inflated quantity the depth-why script warns against in its own header. The reference is the script,
not the probe module the order named; and the "comparison code that reads the record" does not
exist. Same table, side by side, is the requirement.

**A3. `hf_checkpoint` → `hf_id`.** A field name I invented; the Chief's error. The sixth registry
entry (`gemma3-4b-cuda-bf16`) is WS-E's and already on main; it landed without moving the literal
pin in `tests/test_pipeline.py`, so main was red on that test until the same evening. Fixed there.

**A4. Which checkpoint won.** `ADAPTER-DEPTH-ARM1` selected 800 on validation loss;
`ADAPTER-DEPTH-QUALITY` scored the full split and found 1,200 at 175 of 180 against 800's 159
(p = 6.1e-15), and said the one-line answer was wrong, without the earlier record pointing at it.
Ruling: full-split passes is the pre-registered selection criterion from here on; validation loss on
48 rows is a monitoring signal and demonstrably a poor selector for this recipe (it rose twelvefold
between 800 and 1,200 while the task score rose). WS-C reproduces both checkpoints and scores both.
The ARM1 record carries the same amendment.

**A5. The two rule-test failures on `cuda-ws-c` are not WS-C's.** `cuda-ws-c` is thirteen commits
behind main, one of them `d6dc41f`, the refusal guard on `GEMMA3-REGRESSION-2026-09-08/run_native.py`.
Merge main; they clear.

**A6. The gate compared the wrong column, and the chunked loss belongs inside the root unit.**
SWE-2's two-device arm on CPU showed the loss bit-identical at step zero while the root unit's
gradient was forty per cent wrong, because a root left outside any FSDP2 unit is never reduced. The
golden test above now requires per-parameter gradient agreement at step zero and per-parameter value
agreement after N steps; the loss is reported and passes nothing. And the chunked cross-entropy,
which reads `lm_head.weight` directly, moves inside the root unit's forward through a thin wrapper
module that is sharded as the root, so FSDP2 owns every parameter and no second reduction path
exists. Plan §16.7 carries the tables and the reasoning; the hand-reduced arm stays in the record as
the negative control.
