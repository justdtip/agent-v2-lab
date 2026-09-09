# WS-A: the device's first hour

Status: **prepared on fixtures; every CUDA measurement below is unexecuted**. The authority is the
three-task order in [the WS-A brief](../../../design_specifications/pending/CUDA-WS-A-CODEX-2026-09-09.md#next-instruction-2026-09-10--three-tasks-that-need-no-device-in-this-order),
introduced at `6ab3318`, and the [shared first-hour runbook](../../../design_specifications/pending/CUDA-DEVICE-FIRST-HOUR-2026-09-10.md).
Graph-once source is `08a87fb`; the device gate script is `68b1ede`. No further model-scale laptop
execution is authorised. This checklist does not grant a device window or change Q5's production
capture-dtype ruling, which remains with the Director.

## Finding, technique, implementation

**Finding.** The laptop established that comparing a promoted float32 path with native bf16
execution includes a precision discrepancy, so that comparison alone cannot identify a seam defect.
A tiny same-precision reference agrees exactly. Separately, one saved forward graph with batched
cotangents reproduced the sequential Jacobian exactly on the tested fixtures under both attention
kernels. Its local timing result is recorded below; CUDA speed and graph memory remain unmeasured.

**Technique, independent of this stack.** First compare the instrumented path with the model's own
forward using identical weights, arithmetic dtype, tokens and positions. Measure a separate
cross-precision arm to expose its floor; deliberately corrupt three independent boundaries to
check that the instrument detects errors beyond that floor. Re-run with all weights in the higher
precision to separate arithmetic effects from implementation differences. For the Jacobian, reuse
one forward graph and apply batches of output cotangents, comparing every source-layer result with
the sequential estimator before measuring time. Measure peak memory separately against the
existing estimator that replicates the forward batch at the same cotangent batch size.

**Implementation.** `cpu_gates.py` uses the shared text loader and device helpers with
`TorchArchitectureView`; `torch_jacobian.jacobian_for_prompt_vjp` uses upstream's `ActivationRecorder`
and batched `torch.autograd.grad` over one forward graph.

## Ordered device checks

Run serially under an owned window. Stop at the first failing check, preserving completed phases.
The gate script performs the bf16 short and long arms before releasing that model and loading the
float32 control. Its final `pass` covers structure, residual seam and float32 control only;
it explicitly leaves gates 3–4 unexecuted.

| Order and check | Number or record required on the device | Laptop basis | What a mismatch means |
|---|---|---|---|
| 0. Provenance and admission, before loading | Source commit and fingerprint; official snapshot SHA256 map; fixed-token input identity; runtime/device description; seed and eager pinning before CUDA initialisation. No MLX or vision parameters. | Official local Gemma snapshot and torch 2.14 CPU evidence; fixtures mirror the registered snapshot's declared type and key layout. | Wrong material, loader path or runtime invalidates the comparison. Refuse rather than silently converting a different checkpoint or using a stale resume record. |
| 1. Budget, before loading; resources at every phase | Declare worst device peak for **both** bf16 and full-float32 loads, plus a separate host RSS cap. Both caps must respect the shared R47 budgets. Record current and peak device allocation and host RSS, including load and cleanup. | The old 9.8 GiB projection and 7.238632 GiB measured CPU peak covered the earlier bf16-resident calibration only. They do not cover this float32-loaded run. | A projection below required tensor storage is invalid. A measured excess stops the run. Investigate loader transients, retained tensors and workspace; do not increase a cap implicitly. |
| 2. Gate 1: structure, each loaded precision | Layer count, width, vocabulary and attention-family assertions pass; loader dtype/device, exact text-key audit and tied weights are recorded. | Structure accepted on the old CPU checkpoint; current loader/shape path passes snapshot-shaped fixtures. | A model-layout or checkpoint-conversion mismatch; residual numbers are uninterpretable until fixed. |
| 3. Gate 2 smoke: 64 fixed tokens, bf16 | Native-dtype loop versus native capture: **maximum absolute error exactly 0 at every site**, with both dtypes bf16. Record the promoted-float32 arm and rotary discrepancy separately. All controls measured; the short mask control must also be exactly 0. | Earlier mixed-precision discrepancy 1.2386%; rotary maximum absolute difference 2^-9. These are descriptive measurements, not an allowed native-dtype error. | Nonzero native error identifies a seam/arithmetic mismatch within one backend. A short mask difference is unexpected because the prompt does not exceed the sliding window. A changed cross-precision floor alone does not fail this gate. |
| 4. Gate 2 long: 1,400 fixed tokens, bf16 | Native-dtype error remains **exactly 0**. Each of mask dispatch, hook-site off-by-one and entry-transform omission must have maximum norm-relative error **strictly above** the measured global promoted-versus-native floor. Record every layer, not only the maximum. | Earlier mixed-precision discrepancy 6.9013%; old mask-control maximum 53.9638%. The old run did not execute the other two controls. | Native error only beyond the window points first to mask/position dispatch. A control inside the floor means insufficient discrimination; it cannot certify the seam and cannot justify a relaxed bound. |
| 5. Full-float32 control: 64, then 1,400 tokens | All weights and both compared residuals float32; per-site maximum norm-relative error **≤ 1e-3** at both lengths. Record a fresh load peak. | Tiny same-weight float32 loop versus float32 native: exactly 0 at both lengths. Full-checkpoint float32 control was unexecuted on the laptop. | An excess survives removal of bf16 arithmetic and requires investigation of the port. It cannot be excused by the bf16 floor. |
| 6. Graph-once correctness, smallest fixture first | Eager and SDPA: compare every source block 0–4 to target block 5 at 64 tokens and width 64 against upstream sequential; each result within float32 epsilon. Record forward-count evidence and hook cleanup. | Maximum absolute and norm-relative errors were 0 for every tested layer, both kernels. | An estimator, cotangent-selection or backend-autograd defect. Do not benchmark a numerically different estimator. |
| 7. Graph-once timing, same device against itself | Warm both paths; 48 tokens, width 64, source blocks 1 and 3 to target 4, cotangent batch 16 versus upstream `dim_batch=1`. Both timing arms already use one forward; this measures batching the backward work. Record all three end-to-end samples and best sequential/batched ratio for eager and SDPA; required ratio ≥ 1. | Ratios 2.9757963 and 2.69233879 on an active/shared laptop. **No device performance basis.** | A ratio below 1 fails the specified fixture timing check. Inspect batching/kernel overhead on that device; CPU ratios are not GPU targets. |
| 8. Graph memory at the intended context | Announce context, dtype, source/target layers, cotangent batch and projection before running. Compare peak device allocation and host RSS with upstream's replicated batched forward at **matched token IDs, context, source/target layers, dtype and cotangent batch**, not the sequential timing baseline. Record one forward with input batch 1 for graph-once, versus upstream's replicated input batch. Start with a bounded rung and increase only within both caps. | **None: device graph memory has never executed.** Both timing arms already save a single-example graph; their ratio does not measure the memory claim against the replicated-batch implementation. | More than one forward per prompt violates graph reuse. Excess memory requires identifying tape, cotangent and gradient storage; one graph does not imply memory independent of batch size. |
| 9. Gates 3–4 and WS-B handoff | First a smoke decode, then the required golden-episode coverage: capture on/off identity; each final residual's native readout versus that forward's returned logits, with **prefill and decode errors separated**. Establish the checkpoint's measured per-row band; test both `skew_after_prefill` and `corrupt_residual` controls. | Tiny readout maximum 1.49e-8; **no measured full-checkpoint native-readout band**. WS-B's 98/103 teacher-forced argmax agreement measures a different comparison. | Capture changes generation: capture-path defect. Residual readout fails within its own backend: indexing, hook or readout defect. A missing band or incomplete golden coverage leaves the gate unexecuted/incomplete, never passed by borrowing an argmax rate or a tolerance. |

The existing Jacobian tests are CPU fixtures. Repeating that suite on a GPU host alone is not a
CUDA Jacobian measurement: the model and input tensors for rows 6–8 must actually use the selected
CUDA device, and the resulting record must say so. Rows 8–9 have no claimed device result or
complete device-run record yet. WS-B's native-readout work supplies the missing band and golden
coverage; keep that handoff explicit.

## Laptop evidence and box state

Every historical number in this table has basis `laptop-basis`; it is not a device prediction.
**None of these measurements establishes continuous host idleness.** Correctness observations
remain useful on a shared box; shared timing is not a performance basis.

| Evidence and source | Observed result | Was the box idle? |
|---|---|---|
| [Real CPU calibration](cpu-calibration-02-summary.json), `ac8a91e` | Mixed-precision maximum norm-relative error 0.0123859459 at 64 tokens and 0.0690126161 at 1,400; rotary max absolute 0.001953125 = 2^-9 at both. Status: calibrated, acceptance incomplete. | An owned model window is recorded; host idleness is **unrecorded**. |
| Same run's resources | Peak RSS 7.238632 GiB; elapsed 1,785.2 s; historical cap 10.656006 GiB. Lock released and window closed. | Owned window, host idleness unrecorded. Elapsed time is not an idle or CUDA runtime baseline. |
| [Pre-run arithmetic](CPU-CALIBRATION-PLAN.md), before that CPU run | 9.8 GiB projection for bf16-resident text weights, promoted block execution and its retained outputs. No full-float32 model or graph tape. | **Not applicable:** an analytical projection, not a run. It is superseded as a budget for the new device control. |
| [Tiny precision decomposition](precision-diagnosis.json), `0431e00` | Same bf16-rounded weights: all-float32 loop/native error 0 at 64 and 1,400 tokens. Promoted loop versus bf16 native about 0.596–1.050%; versus float32 native at most 0.072325%. The historical `bf16_loop` field denotes the promoted loop on bf16-stored weights. | **Unknown**; no idleness evidence. These are correctness comparisons, not timing claims. |
| [Native-dtype fixture verification](native-dtype-verification.json), `79d2567` | 140 passed, 0 skipped, 34.209 s; real native-dtype checkpoint gate unexecuted. | **Shared:** WS-B held the model window. No performance basis. |
| [Integrated fixtures](integrated-fixtures-02.log), source through `68b1ede`; [box snapshot](integrated-fixtures-02-box-state.json) | 261 passed, 0 skipped, 47.17 s. Includes the graph exactness and warmed timing results in rows 6–7; graph source `08a87fb`. | **Active, not established idle:** pre-run CPU 11.7%; load averages 5.53/5.83/6.22 on 12 logical CPUs; no lock/window. A snapshot is not a continuous idleness record. |

The historical floor includes rounding across the native blocks, not just rotary-table rounding.
Its magnitude can change with kernels and reduction order. Record the new device floor beside the
old one; retain the exact-zero same-dtype gate and the separately ruled float32 bound.

## Future launch and recovery

This is a template with deliberately unset values, **not an executed command**. Replace every
placeholder with the rented device's reviewed budget, material and exact checked-out commit.
The projection must include the full-float32 load; do not substitute the historical 9.8 GiB value.

```sh
LLL_BACKEND=torch LLL_DEVICE='<cuda:N>' uv run python -m local_llm_lab.runlock run \
  --seat codex-ws-a --purpose '<checks, host/device caps and worst projection>' \
  --minutes '<announced minutes>' -- \
  uv run python research/records/CUDA-WS-A-2026-09-09/cpu_gates.py \
  --execute --device '<cuda:N>' --checkpoint '<official local snapshot>' \
  --token-ids research/records/CUDA-WS-A-2026-09-09/calibration-token-ids.json \
  --output '<new device record.json>' --source-commit '<exact current commit>' \
  --cap-gib '<device cap>' --host-cap-gib '<host RSS cap>' \
  --projected-peak-gib '<worst projection including float32>' --seed 0 --threads 1
```

The wrapper owns and closes the announced window; the gate script checks its live holder and nonce
before acquiring the model lock. It does not clear foreign locks. Real checkpoints require CUDA.
`--fixture-cpu` is an explicit, size-bounded exception for tiny fixtures, not a laptop override.
The shared loader loads via host memory and then moves to the device: the host budget must cover
that path as well as device tensor storage. R47 applies to both budgets.

Each report's head records the source identity, checkpoint hashes and device/runtime identity.
Numeric scalars are serialized as `{value, basis}` with basis `measured-here`, `laptop-basis` or
`expected`. Phases are atomically written as they complete, including results before a later
failure. Inspect `acceptance_scope`, individual gates and `error`, not only the top-level status.

Resume by adding `--resume` with the same output path and identities. A prior attempt is archived;
completed units are reused only after identity, completeness, dtype, acceptance and original
resource checks pass. The current attempt's memory remains separate from original
`resumed_resources`: a skipped load is not a new low-memory checkpoint measurement. Changed
source, checkpoint, inputs or runtime cannot inherit an earlier pass. A later failure must retain
the completed evidence and its original error, and cleanup must release only this run's resources.
