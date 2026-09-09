# WS-A review corrections and CPU calibration

**Source review passed:** the edit round through `e0a05cf` was accepted and merged
into `cuda-migration` at `6106ea8`; the dated ruling is in the WS-A order at
`cd8055d`. [REVIEW-READY.md](REVIEW-READY.md) preserves the earlier handoff state.

Latest verified source: `ac8a91eade9f37dcaf053db667a88b395bf52b14`, on
`codex/cuda-torch-seam`. The source includes the committed shared upstream loader and device shim.

**2,203 tests passed, 10 skipped** in the final full suite plus calibration fixtures on torch
2.14.0. The skipped checks require absent historical datasets/saved results. Native MLX library
checks ran in a separate full-suite process; no checkpoint was loaded by those tests. The focused
CPU suites blocked real MLX imports. `review-verification.json` binds source hashes and each run,
including the initial scanner failure and the successful rerun; original evidence is preserved.

The three review edits are complete: the view, capture and acceptance helpers all reach upstream
through `load_upstream`; missing references skip with a reason and conflicting copies refuse;
the evidence was rerun on torch 2.14; and the duplicate patch/manifest files were removed.
`input_device` now reports actual embedding placement without moving the model. The architecture
constant guard follows the existing policy into `arch_base.py` while still scanning `arch_torch.py`.

## Finding and technique

The small fp32 model still has exactly zero residual-source disagreement at 64 and 1,400 tokens,
with the mask control biting only beyond the window. Final readout maximum error is unchanged at
`1.4901161193847656e-08`. These are small-model measurements, not checkpoint acceptance.

The revised bf16 loading proposal needs a reference-precision distinction. On a random three-block
fixture using exactly the same bf16-rounded weights in both copies, the float32 loop/native paths
agree exactly. The promoted loop versus **native bf16** differs by about 0.6–1.1% under the stated
descriptive maximum-norm ratio. Against the **float32 native** copy its largest ratio is below
0.073%. See `precision-diagnosis.json`. This is not evidence of a defect in Gemma 4B: it demonstrates
that a comparison mixing block arithmetic includes more than rotary-table rounding.

The transferable technique is to hold stored weight values fixed, change computation precision
independently, and compare each hand-run path to its matching native reference before attributing
a difference to the instrumentation. These measurements do not choose Q5 or relax the 1e-3 bound.

## Real-checkpoint execution

The earlier memory question is superseded by plan §16.1: store the CPU weights in bf16, promote
per block, and measure within the existing 10.656 GiB cap. `CPU-CALIBRATION-PLAN.md` was written
before loading; it fixes a 9.8 GiB projection, one sequence, source-validated pilot token IDs,
native HF text-only loading and a running owned window/model lock. Its loader and interruption
paths have fixture tests and independent source review. Reports update atomically and retain the
final memory reading even after failure.

The first attempt, `cpu-calibration-01.json`, was refused **before any torch or checkpoint load**:
another seat's pytest process had MLX mapped. Its gate statuses are all unexecuted. The wrapper
closed its own window; no foreign process, lock or window was changed. Further attempt records,
if present, carry their own source commit and observed status.

Retry02 completed in **1,785.2 seconds** (29.75 minutes), with a fresh-process peak of
**7.2386322021484375 GiB**, below the 9.8 GiB projection and 10.656 GiB cap. The
wrapper and model exited, and both owned lock and window were confirmed closed.
`cpu-calibration-02-summary.json` binds the final raw JSON and log by SHA256.

The largest mixed-precision residual ratio was **1.2386%** at 64 tokens and
**6.9013%** at 1,400. Replacing sliding masks with the observed global mask left
all per-layer maximum errors unchanged at 64; at 1,400 it changed them at every
block output and reached **53.9638%**. Rotary-table rounding was **2^-9** at both
lengths. These measurements hold stored weights and token IDs fixed; the loop
uses float32 block arithmetic while the native reference uses bf16. Thus the
normal difference is a precision-confounded measurement, not a seam acceptance.
A future all-layer bf16 comparison must establish exactness separately.

The review at `cd8055d` supplies that next gate: native-dtype loop versus native
dtype, exactly zero; cross-precision comparisons remain descriptive, with no
post hoc tolerance. Hook-site and entry-transform controls are still unexecuted
in retry02. Its gate 1 passed; gate 2 was measured but not accepted; gates 3–4
remain unexecuted. The evidence predates that new gate implementation.

The graph-once estimator remains unstarted pending accepted checkpoint gates 1–4. CUDA/MPS and
full float32-loaded checkpoint execution are unexecuted. The following original handoff is retained
as history; its larger-host question was answered by §16.1.

---

# Original handoff at a639cae (superseded where noted)

Source: `a639cae`, branch `codex/cuda-torch-seam`, based on the requested `2b7905f`.
The fetch of `origin cuda-migration` reported no such remote ref; the exact requested commit was
already in the local repository and is the base. The primary checkout was not changed.

**85 CPU/backend-neutral checks passed**, with real MLX imports blocked. Ruff and whitespace checks
passed. `verification.json` binds the source, tests, dependency and measurement evidence. The first
commit, `82d7149`, extracted the shared base; the moved policy/index/cache/span contracts remain
AST-identical. The native MLX regression suite is unexecuted, not reported green.

## Finding, transferable technique, implementation

The migration exposed a numerical contract that a method-signature port misses: MLX promotes a
float32 activation multiplied by bf16 weights; torch Linear refuses that mixture. A CPU regression
reproduced the dtype error. The torch hand-run path now uses the library's stateless functional call
with just the current module's parameters/buffers promoted. Native weights retain their dtype and
parameter identity; native capture keeps native dtype. This is tested on a tiny CPU bf16 model in
addition to the float32 development gates. It is not CUDA evidence or a checkpoint precision change.

Observation must also inherit **cached entry positions**, not just masks. Observing entry at zero
and masks at the live offset failed a controlled position-dependent entry test. The entry and block
argument bundles now come from one native observation using a copy of the supplied cache. Detached
clones supply the cache copy's tensor values, allowing an ordinary eval model whose cache carries a
gradient graph; the caller's cache and graph are not replaced. Observation is a real additional
text forward, with autograd disabled, and the cache copy is an explicit resource cost.

Implementation reuses upstream `anthropics/jacobian-lens` at
`581d398613e5602a5af361e1c34d3a92ea82ba8e` (Apache-2.0):

- `TorchArchitectureView` delegates discovery to `_find_layout`, with the prior repository paths
  only as fallback. Shared validation and LoRA traversal live in `arch_base.py`; public call
  signatures match the MLX view. Native norm/readout delegates to `HFLensModel.unembed` without
  invoking its constructor's parameter-freezing/tokenizer changes.
- The view observes native block arguments, including masks, rotary embeddings and position IDs.
  Hand-running reuses those arguments. Boolean masks retain their Boolean meaning. It supports
  the DynamicCache adapter and upstream crop semantics; a sliding cache does not promise rollback.
  Unported recurrent backbones and head/span capture are refused explicitly.
- `TorchCapture` subclasses `ActivationRecorder` and delegates block recording/lifecycle to it.
  It adds entry/kwargs/output hooks and the existing sink protocol. `intervene(layer, position, fn)`
  replaces a native vector; addition and interchange are functions on the same seam. Tests cover
  gradients, absolute cached positions, direct model calls, repeated decoding and cleanup on errors.
- `research/acceptance/torch_seam.py` supplies structural and readout probe functions for the shared
  acceptance runner. It does not load checkpoints, select devices or choose acceptance thresholds.

The index test calls upstream `_check_layer_indices`: upstream block outputs 0–32 and target 33
map to repository residuals 1–33 and target 34. Entry residual 0 is separate. A second native HF
layout (GPT-2, including tuple block outputs) also has exact residual-source agreement in the CPU test.

## Executed evidence, in gate order

1. **Full-size structural discovery passed on meta tensors**, from the official Gemma checkpoint's
   config: 34 layers, width 2,560, vocabulary 262,208, and the configured sliding/global spans.
   No checkpoint tensor data was loaded for this check.
2. **Small random HF Gemma, CPU float32/eager:** residual-source maximum error is exactly 0.0 at
   64 and 1,400 tokens. The mask-dispatch negative control is 0.0 at 64 and reaches 0.212559253 at
   1,400. The model has three blocks and width 16; these are not Gemma 4B residual measurements.
3. **Same development model:** final-layer readout top-1 matches all three emitted greedy decisions.
4. **Same development model:** the row-wise readout maximum is 1.4901161193847656e-08. Both controls
   are detected: skew only after prefill, and corrupted residual. Missing readout reports `None`;
   a performed zero-error comparison reports `0.0`. This distribution does not set the checkpoint
   or CUDA tolerance.

`development-results.json` carries the individual figures, versions and structural report.
`measure_development.py` reproduces this small-model evidence and the metadata/header audit without
loading weights. The default isolated runtime is torch 2.9.1, Transformers 5.16.1, CPU, one thread,
deterministic algorithms and eager attention. The shared environment was not changed.

## Why the actual checkpoint gates have not run

Header-only inspection finds **3,880,263,168 text parameters: 14.455107 GiB in float32**. The full
multimodal checkpoint is 16.019044 GiB before activations, logits or loading copies. Both exceed the
previous R47 cap of **10.656006 GiB**. No local exception or different CPU threshold has been assumed.

Torch is eager: `native_residuals` currently executes the model head for every input position.
Unlike the old lazy MLX path, discarded vocabulary logits are still allocated. Per-layer diagnostic
promotion, observation-cache copies and full output logits belong in the real resource measurement.

The official weights are local, but the MLX text-only conversion is not assumed to be a drop-in HF
checkpoint. Its historical wrapper config and key names need the registry/loader's verified mapping.
WS-E owns that loader. The model is never downloaded or loaded by this workstream's gate primitives.

The sampled stage-two records provide frozen input/emitted IDs, logit hashes, argmax and readout
errors. They do not provide full logits or residual arrays in the inspected events. Hashes cannot
establish a cross-backend maximum-logit-error distribution. That is a separate evidence requirement
for the cross-backend harness; it does not prevent within-backend readout identity once the CPU load
is authorized and feasible.

**Full-checkpoint gates 2–4, CUDA/MPS, the estimator rewrite, sharding and the 2,816-token resource
measurement remain unexecuted.** The estimator has not been started: the brief explicitly puts it
after the four real gates. The next required decision is the CPU memory envelope/window or a larger
host for those gates. No new model lock or machine window was taken.

One historical attribution should not transfer into a new finding: the earlier 21 GiB regression
proposal was an incorrect, unexecuted allocation projection. This workstream did not demonstrate
that it was caused by an autograd graph. Upstream's replicated Jacobian graph is a separate source
fact and the future graph-once rewrite addresses that estimator.
