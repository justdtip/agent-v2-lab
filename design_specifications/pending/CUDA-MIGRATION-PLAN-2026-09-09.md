# CUDA migration: survey, segmentation, and the path forward

**Chief, 2026-09-09, on the Director's pivot.** The Director's terms: support arbitrary model
parameter counts; use several GPUs for training without requiring them; write as few new lines as
possible and reuse what exists; never at the cost of fidelity; and treat the cloned
`anthropics/jacobian-lens` as something to **refine, extend, optimise, and make higher resolution**,
not merely to call. Five seats: the Chief (plan, review, acceptance), the D-CRO, Codex, and two
engineers. The hardest pieces go to Codex through the Director.

The seams below were read personally (`arch.py` in full, the capture path's call sites, the runner's
decode loop, the regression fitter, the MLX Jacobian's structure against upstream's estimator, the
installed HF Gemma 3 layer signature, the test doubles, the training entry, and the state
derivation). A parallel survey on Opus is refining the per-subsystem line counts; where its numbers
land they replace the estimates here.

---

## 1. The seam, ruled: a torch view slots under the existing interface with no signature change

`src/local_llm_lab/arch.py` (1,120 lines) is the one compatibility boundary, and its design already
does the right thing: it discovers the decoder **structurally** — a text module owning
`embed_tokens`, `layers`, `norm` at `model`, `model.model`, or `model.language_model.model`, and an
`lm_head` found by shape — and never consults a model-type string (`arch.py:22-24, 73-91`). HF
models have exactly that structure, `language_model.model` included for `Gemma3ForConditionalGeneration`.
**Discovery needs no new code.**

MLX appears only inside method bodies, in four kinds:

| leak | where | torch counterpart | size |
|---|---|---|---|
| `mx.array(ids)`, `.astype(mx.float32)` on every return | `embed`, `run_block`, `final_norm`, `unembed`, `residuals` | `torch.as_tensor`, `.float()` | mechanical |
| the block call `block(h, mask=, cache=)` | `run_block`, `_observe_forward` | `layer(h, attention_mask=, position_embeddings=, past_key_values=)` — **HF layers take precomputed rotary `position_embeddings`; MLX blocks compute rope internally** | the one real divergence |
| the per-block cache list with `.offset`, `.state`, `.is_trimmable()` | `make_cache`, `cache_trimmable`, `_cache_offset`, `cached_logits` | HF `DynamicCache` is **one object**, `get_seq_length()`, `crop()` | an adapter |
| `_Watch`/`Tap` proxies replacing `module.layers[i]` | `_observe_forward`, `NativeCapture` | `register_forward_pre_hook` / `register_forward_hook` on `model.layers[i]` — **MLX has no hooks; torch does natively** | net deletion |

**Ruling.** A `TorchArchitectureView` implements the same public surface — `embed`, `masks`,
`run_block`, `final_norm`, `unembed`, `native_readout`, `make_cache`, `cache_trimmable`,
`residuals`, `native_residuals`, `residual_source_agreement`, `cached_logits`, `tail`, `layer_kind`,
`attention_span`, `lora_targets`, `num_layers`, `hidden_size` — **with no signature change**. The
`position_embeddings` divergence is absorbed inside the view: `masks()` returns, per block, an opaque
bundle `{attention_mask, position_embeddings}` captured from the model's own forward, and
`run_block` unpacks it. Callers already treat the mask value as opaque (`run_block(index, h, masks,
cache_i)`); the only code that inspects mask values is inside `arch.py` itself.

The principle to carry over unchanged is the one the MLX view learned expensively: **observe, never
reconstruct.** `_observe_forward` runs the model's own forward with the blocks proxied to record what
each was handed (`arch.py:461-509`), because Gemma applies a `sqrt(hidden)` entry scale and builds two
masks and dispatches on the block index, and a hand-reconstruction was wrong by a factor of fifty and
then by 95% above the window. On torch the same observation is a forward pre-hook on every layer
capturing `hidden_states`, `attention_mask` and `position_embeddings`. HF Gemma 3 builds
`causal_mask_mapping = {"full_attention", "sliding_attention"}` and `position_embeddings[layer_type]`
and dispatches on `config.layer_types[i]` (`modeling_gemma3.py`, text model forward) — the same
two-mask structure, so `attention_span` reads `config.layer_types` directly.

**What becomes simpler.** `NativeCapture` (`arch.py:806-1120`, ~310 lines) exists because MLX cannot
hook a forward. On torch it is a context manager registering pre- and post-hooks on `layers[i]` and
removing them on exit. The `_ACTIVE_CAPTURES` guard, `Tap`, `ModelTap`, the `replace`/`_restore`
bookkeeping all go. The injection seam (`arch.py:1046-1057`, `out.at[0, local].add(direction)`)
becomes a forward hook returning a modified output — and because a hook can **replace** as well as
add, the interchange interventions the state derivation asks for (§7.1: `h ← h + D(z' − z)` on a
frozen support; §11.3: repeated clamps across turns) are the same mechanism with a different delta.
That is the payoff of the migration for the science, and it is why the torch view's hook API is
designed for replacement from the start rather than retrofitted.

**What is dead on Gemma and stays MLX-only or deleted.** Head capture (Qwen3NextAttention-only,
`arch.py:1030`, reads `cache.state[0]` and is unsafe under rotation). `hidden_spans` span masking
(refuses on a dense backbone, `arch.py:207-213`). The `linear_attention` layer kind and the DeltaNet
LoRA suffixes. None is ported.

---

## 2. Un-porting: the MLX code that was itself a port, and goes back to its reference

**The Jacobian estimator.** `pipeline/lens_fitting/jacobian.py` (957 lines) is a **finite-difference**
estimator — `unit_directions`, `finite_difference_steps`, `reference_responses`, `cached_responses`,
`full_jacobian`, `self_check` — built because MLX lacked the reverse-mode machinery. Upstream
`jlens/fitting.py::jacobian_for_prompt` is the exact estimator: one forward with the graph retained,
`ceil(d_model/dim_batch)` backward passes with one-hot cotangents at every valid target position,
source-averaged (`fitting.py`, the loop after `valid_position_mask`). The MLX divergence was a
platform limitation and carries a fourth error term (truncation) that §7.3 of the derivation does not
even list. **On CUDA the finite-difference machinery is deleted — roughly 450 lines — and replaced by
an adapter of about 80 that calls upstream and writes our artefact.** What survives from
`jacobian.py` is ours and has no upstream counterpart: `make_plan`/`freeze_plan`/`read_plan` (the
pre-registered fit), `Convergence`, `memory_gate`/`time_gate`/`WorkloadMemoryGuard` (R47), and
`write_record`. **Upstream computes; our scaffolding records and gates.**

One difference is scientific and is kept as an *option*: the MLX estimator was per-position
(`prepare_position(view, ids, layer, position)`), which is what makes a position-resolved lens
possible. Upstream gets the same by masking its cotangent to a position band, so the option costs a
selector, not an estimator.

**The lens artefact.** Upstream `JacobianLens.save/load` is the `.pt` we already convert from; our
converter produced the npz, the sidecar and the identity block (`GEMMA3-LENS-2026-09-08/sidecar.json`)
and validated the layer convention against upstream's source yesterday. The adapter is the converter,
already written.

**LoRA training.** `train_expanded.py` (331 lines) drives `mlx_lm.tuner.trainer.train` with a custom
clipped `StableAdamW`, gradient checkpointing, and `depth_expansion` (adding layers, warming their
output). On torch this is HF + PEFT with the same three things preserved as ours: the clipped
optimiser (a few lines on `torch.optim.AdamW`), the checkpoint/val cadence, and **`depth_expansion`**,
which is a scientific mechanism and must survive intact.

**Generation.** `generate_turn_with_count` (`runner.py`) wraps `mlx_lm.stream_generate` and stops on
`</tool_call>` ids or when `turn_is_complete` says the fenced call has closed. The stop logic needs a
per-token decision, which HF `generate`'s stopping criteria make awkward; a hand-rolled greedy loop of
about fifty lines is cleaner and is what produced the golden trajectories' semantics anyway.

---

## 3. The scientific divergences that must be preserved

Missing one of these is the fidelity failure the Director forbade.

1. **Residual convention.** Layer L is the output of block L−1; layer 34 is the final residual before
   the norm; lens map `J[L−1]` reads layer L (`instruments.py:276`, verified against upstream
   `_check_layer_indices` at `44ad800`). Both the hook path and the hand-run path honour it.
2. **Observe the entry residual and the masks; never reconstruct them.** On HF the entry scale lives
   inside `embed_tokens` (`Gemma3TextScaledWordEmbedding`), so reconstruction would happen to be right
   on Gemma — the principle is kept anyway because the next family will not be Gemma.
3. **Hand-run residuals are float32** (`run_block` casts; "diagnostic dtype promotion"). Native-path
   residuals keep native dtype until read.
4. **`residual_source_agreement`**: the hand-run loop against the model's own forward, per layer, with
   a negative control that breaks the mask dispatch. It found the entry-scale omission and the
   boolean-mask cast. It is the port's acceptance evidence and it runs on CUDA before anything else.
5. **The readout gate**: `readout.logits(residual, 34)` recomputed through norm-and-unembed against
   the model's returned logits, recorded on every forward, with two negative controls (skew after
   prefill; corrupt the residual). On MLX decode it reads exactly 0.0. On CUDA bf16 it will not; the
   gate becomes a tolerance, **set from the measured distribution on the golden episodes and not
   chosen**, and the controls must still fail.
6. **Greedy determinism.** Three runs produced byte-identical trajectories. CUDA needs
   `torch.use_deterministic_algorithms(True)`, fixed attention kernel selection, and the same dtype
   path, or the golden tests will read as defects.
7. **The regression fit's arithmetic**: float32 sufficient sums, `alpha · mean_diag(XᵀX)` penalty,
   fixed grid, first-wins ties, small negative SSE preserved as cancellation (`regression.py`). Ports
   line for line.
8. **The lens's ν is declared** (derivation §3.2): endpoint, position weighting, pair weighting, and
   corpus, in the sidecar. Upstream's default ν is target-summed, source-averaged over positions 16 to
   `seq_len−1`; every lens we fit records its own.

---

## 4. Reuse ledger

Estimates from the reads; the survey's counts replace them where they differ.

| subsystem | lines | keep | edit | replace with upstream | new | defer / delete |
|---|---:|---:|---:|---:|---:|---:|
| scientific contract: protocol, env, tasks, records, spans, instruments, integrity, corpus, profiles, artifacts | ~7,000 | all | 0 | — | 0 | 0 |
| `arch.py` → `arch.py` (MLX, kept) + `arch_torch.py` | 1,120 | 1,120 | ~40 (extract shared discovery) | — | **~450** | head capture, hidden_spans not ported |
| `live_lens/session.py` | 523 | ~490 | ~30 (eight `mx` sites → view ops) | — | ~10 | 0 |
| `runner.py`, `rollout.py`, `evaluate.py` | ~1,750 | ~1,400 | ~40 | — | **~60** (greedy loop) | trim/snapshot/history caches (~300) deferred |
| `lens_fitting/regression.py` | 219 | ~170 | ~45 (mx→torch 1:1) | — | 0 | 0 |
| `lens_fitting/jacobian.py` | 957 | ~400 (plan, gates, records) | ~30 | adapter → `jlens.fitting` | **997** (built; budgeted ~80 — §13.5 correction) | **~450 deleted** |
| `lens_fitting/replay.py`, `runtime.py`, `prose.py`, `validation.py` | ~1,600 | most | ~60 | — | ~20 | 0 |
| `pipeline/jlens.py` | 1,987 | — | — | — | 0 | **defer** (probe-era, superseded; survey to confirm) |
| training: `train_expanded.py`, `depth_expansion.py` | ~600 | depth_expansion | ~40 | PEFT + accelerate | **~180** | `gated_delta_*` (614, DeltaNet) **kept under `[mlx]`** — ruling 5 keeps the recurrence, the torch path refuses recurrent backbones, and §16.3 forbids Gemma-scoped deletions (was "deleted for Gemma") |
| box discipline: `runlock.py`, `preflight.py`, `cli.py` | 4,460 | ~4,300 | ~30 | — | **~60** (device backend) | 0 |
| probes | 15,742 | — | — | — | ~200 (patch core on the new seam) | **~15,500 deferred** |
| registry + `pyproject` | — | — | ~30 | — | 0 | 0 |
| tests | 46,623 | 23 files as-is | 25 files parametrised | — | torch doubles ~400 | 0 |
| **golden acceptance harness** | — | reuse `residual_source_agreement`, the readout gate, `_logit_hash` | — | — | **~250** | 0 |
| **lens extensions (§6)** | — | — | — | on upstream | **~400** | 0 |

**Order of magnitude: roughly 2,000 new lines, roughly 1,000 deleted, roughly 40,000 kept.** The
new-line count is dominated by the torch view, the training adapter, the doubles, and the harness;
the science-bearing code — the contract, the records, the fitter's arithmetic, the plan and gate
scaffolding — is untouched or edited mechanically.

---

## 5. Workstreams for five seats

Every stream names the files it owns, the interface it provides and consumes so the others can start
without it, its golden test, and its line budget. Two land first because everything else is measured
against them: **the seam** and **the harness**.

### WS-A. The torch seam — Codex, via the Director. Hardest, first.

**Builds on upstream directly** (the Director's instruction): `jlens/hooks.py::ActivationRecorder` is
extended into `TorchCapture`, and discovery delegates to `jlens/hf.py::_find_layout`, with our
`_TEXT_MODULE_PATHS` as the fallback. **Owns.** New `src/local_llm_lab/arch_torch.py`; a small extraction of the backend-neutral parts of
`arch.py` (structural discovery, index validation, `lora_targets` over a linear-type tuple) into a
shared base both views inherit.

**Provides.** `TorchArchitectureView` with the surface in §1, signatures unchanged; `TorchCapture`
(hook-based) with the sink protocol `residual(layer, offset, h)` / `output(offset, ids, logits)`
unchanged, plus an intervention API that generalises injection: `intervene(layer, position, fn)` where
`fn: h → h'`, covering add, replace, and interchange; a cache adapter presenting HF's single cache as
the per-block list the repo expects (`offset`, `get_seq_length`, `crop`).

**Consumes.** An HF model loaded by the registry (WS-E) with `device_map`.

**Golden test.** `residual_source_agreement` at 64 and 1,400 tokens with the mask-dispatch negative
control, reproduced on CUDA — zero where MLX read zero, and the control non-zero at the long length
only. Then the layer-34 identity check and the readout gate with both negative controls.

**Budget.** ~450 new, ~40 edited. **Why Codex:** it is the piece where a plausible-looking result
can be wrong at every layer — the boolean-mask cast was invisible at every test length — and Codex
ported this view once already and knows where it bit.

### WS-B. Generation, the `none` cache path, and the golden trajectory harness — Engineer A

**Owns.** `runner.py::generate_turn_with_count` (torch branch), a `GreedyLoop` of ~50 lines with the
existing stop semantics, `make_turn_cache` for `cache_strategy: none` only, and the harness
`research/acceptance/golden_trajectories.py`.

**Provides.** Token-for-token reproduction of stage two's 15 episodes from their recorded
`prompt_ids`, with the first divergence point as the diagnostic; the readout comparison at layer 34
(argmax agreement rate, max-abs logit error, with quantisation attributed by running the MLX 4-bit
and CUDA bf16 side by side on the same prefix).

**Consumes.** WS-A's view. Can start on the harness and the loop against a stub view immediately.

**Golden test.** All 15 trajectories byte-identical; layer-34 argmax agreement ≥ the measured
MLX-4-bit-versus-bf16 baseline; the tolerance written from the distribution, not chosen.

**Budget.** ~60 new in the runner, ~250 in the harness. Deferred: `trim`, `snapshot`, `history`
caches — the golden records never exercised them.

### WS-C. Training on HF + PEFT, multi-GPU optional — Engineer B

**Owns.** `train_expanded.py` torch branch; `depth_expansion.py` kept and adapted; the clipped
optimiser; the checkpoint/val cadence; `adapter_delta.py`'s reader for PEFT adapters.

**Provides.** The same CLI, config files (`configs/training/*.yaml`) and adapter provenance record;
`device_map="auto"` for model parallelism across whatever GPUs exist, `accelerate` for data
parallelism when more than one is present, **degrading to one GPU with no flag**.

**Consumes.** The registry's HF checkpoint (WS-E). Independent of WS-A except through `lora_targets`,
which the shared base provides.

**Golden test.** A 40-row smoke train on one GPU and on two agrees **per parameter**: gradients at
step zero and parameters after N steps to float32 epsilon, with the loss curve reported beside them
and passing nothing on its own (§16.7); and `checkpoint_delta` reads the output; then arm 1's recipe (top-8 layers) reproduces the
18-of-19 divergence result on the same checkpoint.

**Budget.** ~180 new. **Deleted:** `training/gated_delta_chunked.py`, `gated_delta_chunkwise.py`
(614 lines, Qwen3.5 DeltaNet recurrence; dead on a dense model). `train_grpo.py`: survey to rule.

**Re-measure before any projection.** The 2.4 MiB/token training memory and the ~6k-row ceiling were
MLX unified-memory numbers; on CUDA with gradient checkpointing they are different numbers and
R60(c) applies — a declaration carries its measurement, taken at the largest context the run will
reach.

### WS-D. The lens: un-port to upstream, then extend it — D-CRO

**Owns.** `lens_fitting/jacobian.py` (delete the finite-difference machinery, keep the scaffolding),
the ~80-line adapter to `jlens.fitting.jacobian_for_prompt`, `regression.py` torch port, and the
extension programme in §6.

**Provides.** A fitted lens in our artefact format with identity block and declared ν; the
transcript-length fit that Codex's task 1 specified, now without the 21 GiB problem (§6.3).

**Consumes.** WS-A's view for `native_residuals` (regression) and an HF model handle for upstream
(Jacobian). Upstream's `HFLensModel` (`jlens/hf.py`) does its own structural discovery and can share
the model object with our view.

**Golden test.** Fit the hosted lens's own recipe (WikiText, 128 tokens, 546 prompts, bf16) on CUDA
and compare to `neuronpedia/jacobian-lens` per layer: cosine and relative difference at every layer,
against the numbers already in `GEMMA3-REGRESSION-2026-09-08`. Agreement validates the un-port; the
residual disagreement is the estimator difference, measured for the first time.

**Budget.** ~80 adapter (built at 997 lines, 574 of them code; accepted 2026-09-09 evening, §13.5 correction) + ~45 regression edits + ~400 extensions; ~450 deleted.

### WS-E. Registry, backend selector, box discipline, tests, records — Chief

**Owns.** `configs/models/*.yaml` (a `backend:` field and an `hf_checkpoint:` beside `hf_id`,
honouring R57 and the Lens Ontology ruling: lineage, not path); `pyproject.toml` extras `[mlx]` and
`[cuda]` so the laptop path survives; a `device.py` with two implementations of the six calls the box
discipline makes (`working_set`, `peak`, `reset_peak`, `clear_cache`, `set_cache_limit`, `device_info`)
and the multi-GPU form of each; the test strategy (23 backend-agnostic files untouched; 25 MLX-bound
files parametrised over `{mlx, torch}` with the same doubles; conftest's window-held skip generalised
to "which device holds a window"); review of every other stream's acceptance; the records.

**Budget.** ~30 registry + ~60 device + ~400 doubles + conftest edits.

### Deferred, explicitly

Probes (`state_probe`, `assistant_axis`, `state_swap`, `jspace_sweep`, `capture`, `power`,
`read_share`) — 15,500 lines built for the hybrid architecture; not one has run on Gemma; the injection
seam is the one intervention path the programme uses and it moves to WS-A. `patch.py`'s
donor-difference core (~200 lines) migrates onto the new `intervene` API when the fixed-point
experiment is ordered. `pipeline/jlens.py` (1,987) pending the survey's ruling on whether it is
superseded by `lens_fitting/`. The three cache-reuse strategies. Head capture. Span masking.

---

## 6. The lens programme: refine, extend, optimise, higher resolution

Each item is small on a torch codebase we now hold the reference for, and each is motivated by a
result from the last two days rather than by taste.

### 6.1 Refine: declare ν, and separate the three errors

Every lens artefact records its endpoint (target layer), its position weighting, its pair weighting,
its corpus, its precision and its estimator, in the sidecar (derivation §3.2, §7.3). The hosted lens's
sidecar gains these retroactively from upstream's defaults, which are now known. A lens whose ν is not
declared cannot be compared with another, and yesterday's comparison of Codex's fit against the
hosted lens was made without knowing they differed on position composition.

### 6.2 Higher resolution, position: a lens per position band

Upstream masks its cotangent to `valid_position_mask`; generalise the mask to a **band selector**.
Fit `J_L^{(b)}` for bands such as 16–126, 127–512, 513–1,024, 1,025–2,048, 2,049–2,816 on the
transcript corpus. This is the direct answer to the binding limitation on every number the map
publishes — the 21.8x extrapolation — and it makes amendment 9's extrapolation test computable for the
first time (it was structurally uncomputable on the hosted lens: no agentic reading is in range). It
also makes the window-conditioned secondary comparison interpretable, because a band above 1,024 is a
lens that has seen the two attention kinds differ. **~60 lines on upstream.**

### 6.3 Optimise: hold the graph once, and shard prompts across GPUs

Upstream replicates the prompt `dim_batch` times to batch cotangents and retains the whole graph
(`fitting.py`: `input_ids.expand(dim_batch, -1)`, `retain_graph=True`). At 128 tokens and
`dim_batch=128` that is ~168 MB per layer; at 2,816 tokens it is ~3.7 GB per layer, ~125 GB for the
stack — which is why Neuronpedia fitted at 128 on a 179 GB card, and why the transcript-length fit
projected 21 GiB on a 24 GiB box. Two changes: (a) run the forward **once**, unreplicated, and batch
the cotangents with `torch.func.vjp` + `vmap`, so memory is one copy of the activations plus one
cotangent batch; (b) shard **prompts** across GPUs — each computes its full `J` for its prompts and
`JacobianLens.merge` (upstream, already written) is the reduction. Single-GPU is the one-shard case,
which satisfies "without strictly requiring it". **~120 lines, and it is the item that makes 6.2
runnable at all.** The MLX `benchmark`/`memory_gate` scaffolding projects it per R47 before it runs.

### 6.4 Higher resolution, span: a lens per token class

The map found call arguments held one to two orders of magnitude further from the output than the
skeleton carrying them, closing at layer 24, and the hosted lens was fitted on prose that is nearer
in-domain on notes than on calls. Fit `J_L^{(span)}` for `note`, `call_skeleton`, `call_argument`,
`chat_prose` from the span labels the recorder already emits — the same cotangent selector as 6.2 with
a different mask. **~30 lines** once 6.2 exists.

### 6.5 Higher resolution, depth: sub-block residuals and multi-target lenses

Upstream hooks `layers[i]` outputs; hook `self_attn` outputs too and fit at the post-attention
residual, giving 68 read points on Gemma instead of 34. The commitment at layer 23-to-24 that the
fixed point and the corpus-wide transition both show becomes attributable to attention or to the
MLP of block 24, which today's data cannot say. Separately, upstream's `target_layer` argument
already permits `J_{L→M}` for `M < final`; loop it to get depth-to-depth transport, which the
commitment analyses want directly. **~80 lines.**

### 6.6 Variance-aware: the second moment beside the mean

Derivation §3.4: `E[KᵀK] = JᵀJ + E[ΔᵀΔ]`, so a small averaged effect does not imply a small
context-specific one. Accumulate `E[KᵀK]` per layer alongside `J` (one extra `d×d` accumulator,
computed from the same gradients), and report per-layer `‖E[ΔᵀΔ]‖ / ‖JᵀJ‖` as the lens's
**self-reported cancellation**. A layer where that ratio is large is a layer where the averaged lens
is least trustworthy — which is disclosure the map has never been able to make. **~40 lines.**

### 6.7 The bridge and the state programme

Stage A of the SAE–J bridge (`SAE-J-BRIDGE-ORDER`) is CPU linear algebra and unaffected. The state
derivation's §§7–13 need interchange interventions, repeated clamps, retention with carrier controls
(replay identical tokens while resetting or exchanging retained state per layer), and branched
continuations from a checkpointed state. WS-A's `intervene(layer, position, fn)` and the HF cache
object make the first three straightforward; `branch.py` (419 lines, kept) is the fourth. These are
ordered after the migration validates, not as part of it — but the seam is designed for them now so
they are not a second migration.

---

## 7. The acceptance gate

No CUDA number enters a record until, in this order:

1. **Structural**: the torch view discovers Gemma 3 4B and reports 34 layers, 2,560 hidden, spans
   `sliding`/`global` matching `config.layer_types`, residual convention pinned.
2. **`residual_source_agreement`** at 64 and 1,400 tokens: hand-run against native, zero at both,
   and the mask-dispatch negative control non-zero at 1,400 only.
3. **Layer-34 identity**: top-1 of the native softmax is the emitted token, every decode step.
4. **Readout gate** with both negative controls passing on CUDA, tolerance set from the measured
   bf16 distribution on the golden episodes and recorded with its basis.
5. **Golden trajectories**: 15 episodes token-for-token from recorded `prompt_ids`.
6. **Golden lens reads**: rank rows at every layer for the 4,801 emitted tokens through the hosted
   lens, agreement reported with quantisation attributed.
7. **Lens un-port**: the hosted recipe refitted on CUDA and compared per layer.

Steps 1–4 gate WS-A; 5 gates WS-B; 6 gates the map on CUDA; 7 gates WS-D. Training (WS-C) gates on
its own smoke test and on `adapter_delta` reading its output.

---

## 8. Policy questions, ruled by the Director (2026-09-09)

The Director answered these in the document. His words are quoted; the consequence for the plan
follows each.

1. **Locks.** *"1 model per device, training specifically on both if possible."* — **Ruled: one
   model per device.** A window names the device set it holds; a training run may claim every
   device present under one window; the suite's skip gate is keyed on the devices a window names,
   not on the box.

2. **Precision.** *"bf16. I want resolution."* — **Ruled: bf16 is the CUDA path, everywhere.** No
   4-bit on CUDA. The MLX-4-bit-versus-CUDA-bf16 difference is measured once on the golden episodes
   and recorded as the comparability bound for every number that crosses the two.

3. **Training.** *"When we fine-tune, we will also be fine-tuning the actual model itself, rather
   than applying LoRA adapters."* — **Ruled: full fine-tuning, not LoRA.** WS-C is rewritten below
   (§12.2). The depth-expansion machinery is preserved; `adapter_delta.py` becomes a
   checkpoint-delta reader over full weight differences, and the depth-why geometry (per-module
   relative perturbation in quadrature) generalises to it unchanged.

4. **Probes.** Not addressed; the recommendation stands — deferred except `patch.py`'s intervention
   core, until a probe is ordered on Gemma.

5. **The MLX path.** *"Yes. We keep it so we can keep doing experiments locally if needed. This is a
   separate branch that will eventually be merged with the main one. I want to open a new branch
   with every line of code so that we can reuse what we already have where applicable, and to
   avoid introducing new conventions when we have existing implementations."* — **Ruled: MLX kept;
   `cuda-migration` carries every line and merges back; reuse before invention.** A new convention
   where an existing implementation exists is a defect in review.

6. **The branch.** *"Agreed."* — `cuda-migration` from `2b7cba1`, which now also carries the
   regression baseline (§11.2).

7. **Where the work happens.** *"The idea is for us to implement this locally, then pull it onto
   the remote server with the GPUs when we're ready. This is to minimize costs associated with
   renting the server without using the thing we're renting it for. Because of this, I would like
   us to be ultra-diligent because we won't be able to load cuda for testing until we actually
   have the device. I don't expect a perfect implementation that immediately works for the first
   time when we actually get the new device. I am, however, hoping for us to be able to
   efficiently diagnose and resolve any issues we run into, because compute is not exactly
   cheap."* — **Ruled: local-first, CPU torch as the development environment, and the first hour
   on rented hardware is diagnostic by design.** §12.1.

8. **Sequencing.** *"Fire off the other sessions' tasks, then provide me what I should give
   codex. Make any necessary preparations beforehand."* — Done in this order; the seat orders and
   Codex's brief are the companion documents to this plan.

## 9. Sequence

| week | lands | unblocks |
|---|---|---|
| 1 | WS-E registry + backend selector + device shim; WS-B harness against a stub; WS-A view through gate step 2 | everything |
| 1–2 | WS-A through gate step 4; WS-B greedy loop through step 5 | the map on CUDA; WS-D |
| 2 | WS-D un-port through step 7; WS-C smoke on one and two GPUs | the extensions; training runs |
| 2–3 | WS-D 6.2 + 6.3 (position bands at transcript length, sharded); WS-C arm-1 reproduction | the extrapolation test; a map read through a matched lens |
| 3 | 6.4–6.6; `patch.py` core on `intervene`; the fixed-point experiment ordered | the state programme |

**Codex through the Director:** WS-A in full, and 6.3 (the `vjp`/`vmap` rewrite and the sharded
merge), which are the two places where a plausible number can be wrong in a way no control on the
number reveals.

---

## 10. Size- and memory-adaptive by construction, because the 80 GB card is not guaranteed

The Director may have an 80 GB GPU and may not. The design must therefore be **agnostic to layer
count and parameter count**, and must **pivot to less memory without a redesign**. Three rules, and
the machinery for each already exists in some form.

### 10.1 Nothing in source names a dimension

The view reads `num_layers`, `hidden_size` and `vocab_size` from the model (`arch.py:53-57`); the
global-attention layers are read from the model's own dispatch, never from a period; lens artefacts
key maps by layer index; the regression accumulators are sized from `view.hidden_size`. **WS-E's
first check is a grep for hardcoded dimensions in `src/`**, and any hit is a defect to fix before
the branch takes code. Records and configs may name a model's dimensions — that is provenance — but
no code path may assume them.

### 10.2 Every memory-bearing choice is measured on the target device and then chosen, not set

The R47 machinery in `jacobian.py` (`memory_gate`, `WorkloadMemoryGuard`, `benchmark`) projects a
workload and **stops** if the projection exceeds 0.6 of the working set. On CUDA it is extended to
**choose**: measure one unit of work at the largest context the run will reach (R60(c)), then pick
the largest batch that fits under the R47 fraction of `torch.cuda.mem_get_info()` on the device the
run will use. The free variables, by workstream:

| workstream | the knob | measured unit | fallback ladder if the projection does not fit |
|---|---|---|---|
| WS-D Jacobian (6.3) | cotangent batch (`dim_batch`) | one prompt, one backward, at the fit's context length | halve `dim_batch` → fit **source-layer bands** (0–k, k–N) with the graph held from the band's first layer, upstream's `start_graph_at` → accumulate on CPU |
| WS-D regression | prompts in flight | one window through `native_residuals` | one window at a time (the fitter already accumulates row by row) → accumulators on CPU (`d×d` float32, never differentiated) |
| WS-C training | micro-batch, sequence cap, checkpointing | one step at the sequence cap | gradient checkpointing on → micro-batch 1 with accumulation → `device_map="auto"` with `max_memory` and **CPU offload**, which runs anything at the cost of speed |
| WS-B capture | none: one sequence, no reuse | the longest golden episode (2,749 positions) | the map runs on any device that holds the model; if it does not, `device_map` shards the model |
| all | the model itself | load | bf16 on one device → `device_map="auto"` across devices → 8-bit/4-bit only as a **declared** precision change with the comparability bound re-measured |

**The declaration carries the measurement and the choice.** A run's window announces the device,
the measured unit cost, the chosen batch and the projected peak, so a miss is diagnosable as a
method error and not bad luck — which is the lesson of the stage-two memory declaration and the
21 GiB projection, both from yesterday.

### 10.3 Arbitrary parameter count: what scales and what does not

For a model with `N` layers and hidden `d`, the objects that grow: the model (`device_map` handles
it), the retained graph for the Jacobian (source-layer bands bound it to a fraction of `N`), the
accumulators (`2 × N × d² × 4` bytes; ~14 GB at Gemma 3 27B's 61 layers and 5,376 hidden — fine on
80 GB, CPU-resident otherwise), and the lens artefact itself (`N × d² × 2` bytes at float16;
~3.5 GB at 27B). Nothing else scales with the model. The records, the contract, the harness and the
scaffolding are size-free.

**A larger model is therefore a registry entry and a re-measurement, not a code change**, and a
smaller GPU is a different rung on the same ladder. That is the property the Director asked for,
and it is cheaper to build in now than to retrofit after the first out-of-memory on a 27B fit.

---

## 11. Two dependencies the D-CRO found, a correction to 6.6, and the pre-hardware bridge

### 11.1 6.6 is verified sound, and it has a cost the plan did not state

The D-CRO checked the second-moment extension mechanically rather than accepting it. Upstream
materialises `grad` as `[dim_batch, seq_len, d_model]` before averaging over positions, and the sum
over output dimensions **decomposes across passes** — `(KᵀK)[i,j] = Σ_d K[d,i]K[d,j]` with each pass
owning a disjoint set of `d` — so one `einsum('bpi,bpj->ij', g, g)` accumulated per pass gives the
exact second moment with one `d×d` accumulator per layer. Forty lines is right.

**What it costs, now in the §10.2 table where it belongs:** it scales as `n_positions × d_model³`.
At the hosted recipe's 111 valid positions that is 3.7 TFLOP per prompt, about a minute over 546
prompts at ~50 TFLOPS; at transcript length it is 94 TFLOP per prompt, about seventeen minutes.
Memory is **+0.81 GiB** for 33 fp32 accumulators, and it lands on exactly the budget `dim_batch` is
chosen against.

**And the expectation must be declared, for the same reason ν is.** `E[KᵀK] = JᵀJ + E[ΔᵀΔ]` is exact,
but *what the expectation ranges over* — source position within a prompt, prompt within the corpus,
or both pooled — changes what the cancellation ratio claims. Within-prompt and between-prompt
variance are different statements about a lens's trustworthiness, and "self-reported cancellation"
reads as the pooled one. **The artefact records which, and reports the within-prompt and
between-prompt terms separately**, fixed before anything is computed.

### 11.2 The golden baseline is not on the branch

`GEMMA3-REGRESSION-2026-09-08` — the per-layer comparison against the hosted lens that WS-D's golden
test is defined against — exists only on `codex/gemma-lens-fitting`, four commits ahead of the tree
at `0fc04f9`, never merged. Neither `main` nor `cuda-migration` holds it. **It lands on
`cuda-migration` before WS-D has a baseline to be golden against**; the Chief is bringing it over.

### 11.3 There is no torch on this machine, and that is a bridge rather than a wall

`import torch` fails in the venv. So nothing in WS-A, WS-B or WS-D's adapter can be **executed** here
as written, and the D-CRO's rule stands: anything handed over before it can run is marked
*unexecuted*, and today has shown what unexecuted claims cost.

But torch runs on this machine without a GPU — CPU, and MPS on Apple silicon — and the bf16 Gemma
checkpoint is on disk. **That is the pre-hardware execution environment for the seam.** Gate steps
1 through 4 (structural discovery, `residual_source_agreement` with its negative control, the
layer-34 identity, the readout gate with both controls) and the lens un-port can all run on CPU in
**float32**, which is a *cleaner* validation of the code paths than CUDA bf16 would be: it removes
kernel selection and precision from the comparison against MLX, so a disagreement is a defect and
not a rounding question. CUDA then becomes a device change on validated code, and the bf16-versus-fp32
difference is measured once as a known quantity rather than confounded with the port.

**So the device abstraction covers `{mlx, cpu, mps, cuda}` from the start**, not `{mlx, cuda}`.
Installing a CPU/MPS torch into the venv is a change to the environment every seat shares and is the
Director's call; it is one command and it is the single action that converts the plan from
unexecutable-until-hardware into executable today.

### 11.4 Sequencing, corrected on the D-CRO's recommendation

Stage two's records are the entire evidence base for every §6 item, and the outstanding analyses on
them — the stratified reads, the truncation outcome, the verdict reason vectors — need no torch, no
GPU and no migration. They are also exactly the things that quietly do not get finished when a
repository pivots. **The D-CRO closes them first and hands WS-D a complete evidence base.** WS-A
starts on CPU torch the day it is installed; the two run in parallel because they are different
seats, and neither waits on the other.

---

## 12. Local-first, and what the Director's rulings change

### 12.1 CPU torch is the development environment; the remote's first hour is diagnostic by design

Nothing CUDA can run until the device is rented, and the device costs money while it idles. So:

- **Every workstream develops and validates on CPU torch with float32 arithmetic**, against the MLX
  golden records, on this machine — weights loaded in the checkpoint's bf16 and promoted per block
  by the view, not loaded as float32, which does not fit the R47 cap (§16.1). What passes on CPU is recorded — gate by gate, at a commit — in a
  manifest the branch carries, so that on the remote a failure is attributable to *the device
  change* and nothing else.
- **A remote diagnosis kit ships with the branch**: `scripts/acceptance_gates.py`, the seven §7
  gates as one script that runs in minutes on a fresh device and stops at the first failing gate
  with the number it saw and the number it expected. The first hour on rented hardware runs that,
  not an experiment. Determinism settings (`torch.use_deterministic_algorithms`, attention kernel
  pinned, dtype path recorded) are set by `device.py` and printed by the kit before the first gate.
- **Expected differences are pre-declared, not discovered**: the CPU-fp32-versus-CUDA-bf16 delta on
  the readout gate and the golden lens reads is projected from the MLX-4-bit-versus-CPU-fp32 delta
  already measured locally, and a remote result outside that band is a defect, not a rounding
  question.
- **Ultra-diligence, operationalised**: no CUDA-path line lands without a CPU test that exercises
  it; every device-dependent branch (`if device.type == "cuda"`) has a CPU test that proves the
  other arm; and every memory-bearing knob is measured on the remote before the run that depends on
  it (§10.2), with the declaration carrying the measurement.

### 12.2 WS-C rewritten: full fine-tuning, multi-GPU when present, optional

The Director rules that fine-tuning updates the model's own weights, not adapters. That changes the
memory model and the tooling, and it simplifies the provenance.

**Memory.** Full fine-tuning in bf16 with fp32 Adam states costs roughly `params × (2 + 2 + 8)`
bytes plus activations: about **48 GB for 4B parameters** before activations, which fits one 80 GB
device with gradient checkpointing and does not fit 24 GB; Gemma 3 27B does not fit one 80 GB device
at all and needs sharding across several. So the training stream is **FSDP under `accelerate`**,
which shards parameters, gradients and optimiser states across whatever devices exist and runs
unsharded on one — "multi-GPU when present, not required" is the same code path with a different
world size. CPU-offloaded optimiser states are the fallback rung for a device short of memory
(§10.2's ladder). None of this is written by us: `accelerate` configures it and HF's `Trainer` or
a minimal loop drives it.

**What is ours and is preserved.** `depth_expansion.py` (adding layers and warming their output);
the clipped optimiser; the checkpoint and validation cadence; the dataset manifest requirement; the
training config schema in `configs/training/`. These move onto the torch loop as they are.

**Provenance.** A full-fine-tuned checkpoint is a new `hf_checkpoint` with `base` and a
`training_lineage` entry per R57; the identity a lens reads by lineage still holds. `adapter_delta.py`
becomes `checkpoint_delta.py`: per-module relative perturbation `‖ΔW‖/‖W‖` in quadrature, the same
quantity the depth-why record computed on LoRA, now over full weights. The top-8-layer result
(training the lower layers destroyed the model) is the first thing to re-measure under full
fine-tuning, because it is the programme's most consequential training finding and it was measured
on adapters.

**The `[mlx]` path keeps LoRA**, because that is what runs on the laptop and what every existing
training record was made with. The two are different experiments and the registry says which.

### 12.3 Seat orders and Codex's brief

Companion documents, each naming files, interfaces, golden tests and budgets from this plan:

- `CUDA-WS-A-CODEX-2026-09-09.md` — the torch seam and the graph-once estimator rewrite, for the
  Director to relay.
- `CUDA-WS-B-ORDER-2026-09-09.md` — generation, the `none` cache path, the golden harness.
- `CUDA-WS-C-ORDER-2026-09-09.md` — full fine-tuning under FSDP, multi-GPU optional.
- `CUDA-WS-D-ORDER-2026-09-09.md` — the lens un-port and the six extensions, for the D-CRO.
- WS-E is the Chief's and needs no order.

---

## 13. What the survey corrected, verified, and what it left for the Director

Thirty-one Opus agents — nine readers, a verifier on each reader's two most consequential claims, a
planner — with eleven of fourteen adversarial claims refuted and the planner re-verifying the
load-bearing ones against the installed `transformers` 5.16.1 and the records. Everything below
marked **V** the planner read; the rest I checked against my own reads of the same files. Where the
survey and this plan disagreed, the survey was right in every case below.

### 13.1 The seam: four corrections

- **The per-block bundle is `(mask, (cos, sin), position_ids)`** and the `(cos, sin)` pair is per
  **layer type** (**V** `modeling_gemma3.py:558-561`), so one rope pair would give five of six Gemma
  blocks the wrong base. `cache_position` is not required (**V** zero occurrences).
- **`cache.layers[i]` is already the per-block handle** (**V** `cache_utils.py:1271, 1306`), with
  `get_seq_length()` and `crop()`; the adapter is ~5 lines, not 40. Two traps: `crop` takes a
  **negative** count and returns nothing where `trim_prompt_cache` returned the count `runner.py:135`
  tests; and `.state` carries keys and values but **not position** on either backend — which means
  **`SnapshotCache` restore is already broken on Gemma's 29 rotating layers today, on MLX**
  (`runner.py:187`). An existing defect to fix, not a fidelity target.
- **`NativeCapture.__enter__` imports `Qwen3NextAttention` unconditionally** (**V** `arch.py:974-977`),
  so the injection seam cannot be entered on any model without mlx_lm's Qwen module. Conditional on
  head capture. And the HF model returns `CausalLMOutputWithPast` and takes `past_key_values=`, a
  calling-convention leak at `arch.py:965-966` and `session.py:410`.
- **`residuals()` is a float32-per-block recomputation, not a capture** (**V** `arch.py:313, 457,
  465`), and `native_residuals` is the true capture that **no probe consumes**. Hooks reproduce the
  native path; substituting them silently changes what every probe and the Jacobian's `pre_norm_tail`
  see. **That is a decision, Q5 below, not a port.** Two additions to the contract: `view.input_device`
  (placement is the caller's; `from_model` never moves a model) and `view.vocab_size`.

### 13.2 Things this plan had wrong

- **The three cache strategies are not deferrable.** `history` is `qwen35-4b`'s declared default with
  equivalence recorded; `trim` is the default model's. With `cache.layers` the shim is ~35 lines. WS-B
  carries them.
- **The registry does not gain a `backend` field.** Identity is `base` + `training`;
  `load_model_spec` matches on `hf_id`; one entry naming two artefacts is what that split was invented
  to prevent, and it would change the `registry_sha256` committed in records. **One entry per CUDA
  artefact, zero parser lines, the five existing YAMLs byte-identical.**
- **Packaging has a blocker before torch is considered**: `transformers`, `numpy`, `pyyaml`,
  `huggingface_hub`, `safetensors` are imported at module scope in nineteen files and arrive
  **transitively through `mlx-lm`**. Promote them first; then extras `[mlx]` / `[cuda]` — extras and
  not markers, because MLX wheels are macOS-arm64-only and a Linux box cannot resolve them.
- **Emitted tokens in the golden corpus are 5,245, not 4,801** (the smaller figure was agentic-only).
  Corpus, verified: 5,439 `forward`, 5,245 `emitted`, 94 `begin_turn`, 534,990 `rank`, 114,621
  `reading`.
- **The un-port changes the estimator's ν and must say so.** Our MLX fit uses the same-position
  reduction (requirements §2); upstream sums over targets. `profiles.py` already keeps `jacobian`
  distinct from `hosted-jacobian`. WS-D's adapter supports both through the selector and **declares
  which**; the lens fitted under upstream's default is not the same estimator as the one under ours.
- **The finite-difference saving is one factor of ~15.5x**, all from the layer loop, not 16–33x; and
  the step rule had a local precedent (`jlens.py:1029-1031`). Whether FD stays or autograd replaces it
  is **Q3**, not an implementer's call.
- The reuse ledger: the survey's counts replace mine below.

### 13.3 Divergences this plan did not list (survey §3, numbered as there)

**10** EOS is never appended by the MLX loop and three downstream sites rely on it; HF `generate`
appends it. **11** EOS comes from the config's id set `[1, 106]`, never `tokenizer.eos_token_id`, or
every turn becomes a 200-token runaway. **12** The `</tool_call>` stop set is a no-op on Gemma (unk id
3) and stays byte-for-byte. **13** **The prefill partition at 2,048 is part of the record**
(`NATIVE_PREFILL_STEP_SIZE`, asserted by `ForwardLedger.validate` every forward); HF prefills in one
chunk, so chunk explicitly or the forward rows change shape under an identical trajectory. **14**
**TF32 must be off** — `torch.backends.cuda.matmul.allow_tf32 = False` /
`set_float32_matmul_precision("highest")` — or every fp32 unembed and lens matmul is silently
re-quantised to ten mantissa bits; nothing in the repository says so because MLX has no TF32. **15**
R18: native-precision capture, one float32 cast at pooling — in tension with 13.1's fourth item; Q5.
**20** LoRA `scale` is literal in MLX and `alpha/r` in PEFT (`scale: 32.0` at `r=16` is
`lora_alpha=512`); matters to the `[mlx]` path only now. **21** `iters` counts micro-batches. **22**
mlx-lm's batch order, padding to `1 + 32·ceil(L/32)`, and **unweighted** accumulation where HF is
token-weighted: reproduce or record the departure. **23** The observation re-roling convention must
not be replaced by a torch template's tool role; that changes what the model was shown. **25**
`_resolve_checkpoint` stays confined to `models/` paths; torch entries name HF ids.

### 13.4 The acceptance gate, as the survey specified it — adopted over §7

- **G-0, no model, first hour, blocks everything**: `read_record` and `validate_events` over all
  fifteen episodes on the new box; plus one MLX-versus-MLX self-replay on the laptop with
  `logits_sha256` armed, to prove the harness is not what changed.
- **G-1, unblocks everything**: `residual_source_agreement` at 64 and 1,400 tokens. MLX baseline: gate
  0.0 at both; control 0.0 at 64 and 5.3125 at 1,400. CUDA criterion pre-registered: **≤ 1e-3
  relative at every layer in bf16**, the control exceeding it by two orders at 1,400. **Three**
  negative controls, each shown to bite at 1,400 and not at 64 — mask dispatch, hook-site off-by-one,
  entry-transform omission — and **the controlled number must not be bit-identical to the
  uncontrolled one**, which is how R56(e) was found.
- **G-2, teacher-forced golden trajectories** via `replay_record` with `logits_sha256` **dropped**
  from the key (it cannot match across backends) and `argmax` armed; **and** the second enforcement
  site at `profiles.py:306-315`, which compares a measured float, must be exempted or it fails.
  Attribution from the data: at layer 34 all 5,245 emitted tokens are rank 1, 80.06% at P ≥ 0.99,
  2.19% below 0.5. **A single argmax flip at P ≥ 0.99 fails the run outright**; quantisation cannot
  move it, so it is a mask, position, entry or norm defect. Then a free-running greedy replay of the
  three chat episodes, because teacher forcing hides accumulation.
- **G-3, the readout gate, decode and prefill reported separately**: MLX baseline is all 5,339 decode
  forwards at exactly 0.0 and the 100 prefill forwards at 0.375–0.75. A gate that does not split them
  fails on a fact already true. **Logit distance needs no sidecar**: at layer 34 the rank rows'
  probabilities are the model's own softmax, so `log(p_a/p_b) = z_a − z_b` exactly, ~9,391 gaps.
- **G-4, golden lens reads, paired**: per-token Δrank and McNemar on the rank-1 indicator; the
  layer-convention off-by-one is six to ten sigma per layer and catches itself on episode one.
- **G-5, multi-GPU, a separate arm never folded in**: G-1 to G-4 again under each parallelism with
  its own bands, provenance recording device count, dtype, deterministic flags, TF32 state, pinned
  attention backend and all-reduce order. **Byte-identical trajectories are a within-backend,
  fixed-topology property**; a sharded matmul reduces in a different order. The one-versus-W-device
  arm passes on per-parameter gradient and parameter agreement, never on the loss (§16.7).
- Determinism on CUDA: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  TF32 off, a pinned SDPA backend.

### 13.5 The reuse ledger, the survey's, replacing §4

In-scope source 26,062 of 44,739 lines. **New 1,473 — 5.7% of in-scope — against 22,833 kept, 1,338
edited, 695 replaced by upstream, 18,677 deferred, 1,196 deleted. Net source delta −418.** Seventeen
lines kept, edited or replaced for every new line. Tests are a further ~1,450 new. The per-subsystem
table is in the survey record and its counts are the ones to hold seats to.

**Correction, 2026-09-09 evening — the WS-D adapter.** Budgeted at ~80 lines, built at 997 (574
code, 247 docstring, 39 comment, 137 blank), read in full by the Chief. Of the code, ~120 lines are
the estimator adapter proper (load the clone, index conventions, the per-row loop, convert, write),
~130 the §6.1 ν declaration and its reader, which the WS-D order required on top of a budget that
only ever covered the estimator call, ~70 the §6.2 selector seam pulled forward, ~75 the orientation
check through upstream's own transport, ~50 measurement-versus-declaration guards, the rest refusal
messages. The estimator is imported, never copied; nothing re-derives upstream. Accepted, and the
budget inconsistency was the plan's. The headline moves: **new ~2,390 — 9.2% of in-scope**; net
source delta **+499** rather than −418; about ten lines kept, edited or replaced per new line rather
than seventeen; tests a further ~2,100 with WS-D's 650.

### 13.6 Seat assignment: where the survey and the plan differ, and what stands

The survey put lens fitting on an engineer and training on Codex in a later phase; this plan put the
lens on the D-CRO — who owns the lens science and closed the analyses every extension rests on — and
training on Engineer 2. **The plan's assignment stands, having been dispatched**, with the survey's
Phase 0 (packaging and registry) taken by the Chief now, and its G-0 to G-5 adopted as the harness
Engineer 1 builds to the Chief's specification.

### 13.7 Four decisions for the Director, from the survey

- **Q1, refined.** One lock per **device** (by UUID, not index — `CUDA_VISIBLE_DEVICES` makes indices
  lie), one window per **box**; a fully sharded run announces the window **and** takes every device's
  lock. Host RAM is the window's business, not a fifth lock.
- **Q2, sharpened.** bf16 stands, and the acceptance test becomes three-tier: a **fresh MLX-bf16
  stage-two run** on `gemma-3-4b-it-bf16` as the Tier-1 comparator, so the divergence under test is
  kernel reduction order and not a quantisation scheme; G-1 to G-4 as Tier 2; and **the existing
  4-bit rows demoted explicitly from acceptance test to sanity reference** — argmax, ids, shapes,
  counts only — so no one later cites a 4-bit-versus-bf16 rank delta as a finding. The programme's
  own `QUANT-GAP-2026-09-05` measured that boundary as the worst-agreeing readout.
- **Q3.** Finite differences or exact autograd for the Jacobian: defer to the pre-registered c-sweep
  (§15 of the fitting requirements). A layer with no plateau is instrument-limited and autograd is
  the remedy; a plateau keeps the current estimator. Either way the same-position reduction is
  preserved and declared, and the fitted lens stays distinct from the hosted one in `profiles.py`.
- **Q5.** The residual capture dtype. Move to the native path, declare it as a **measurement change**
  in the record, re-run rather than re-record R18a's manual-versus-native comparison, and accept that
  the `--residual-source manual` arm changes meaning. The alternative — hooks that reproduce an fp32
  recomputation nobody chose — costs several hundred unbudgeted lines to preserve an artefact of one
  cast. **Taken in the open, not inherited.**

### 13.8 Phase 0 landed, and two things every seat must know about it

**The venv is managed by `uv`, and there is no `pip` in it.** The install commands are:

```bash
uv sync --extra mlx            # the laptop: exactly what the venv holds today
uv sync --extra cuda           # the GPU box
uv sync --extra mlx --extra cuda   # the laptop during the migration, CPU torch beside MLX
```

**A plain `uv sync` now removes MLX from the laptop**, because `mlx-lm` and `mlx-tune` moved from
base dependencies to the `[mlx]` extra (§13.2). That is the intended layout and it is a trap for
anyone who types the old habit. The five dependencies that arrived transitively — `transformers`,
`numpy`, `pyyaml`, `huggingface_hub`, `safetensors` — are direct now, at the installed versions as
floors, so the tree imports on a box with neither extra.

**The registry has six entries.** `gemma3-4b-cuda-bf16` is a copy of the bf16 entry whose `hf_id`
is the upstream repo id; the five existing files are byte-identical, so every `registry_sha256` in a
committed record still verifies.

**A latent behaviour, pre-existing and unchanged:** `base_of_artifact` resolves a registry name, an
absolute checkpoint path, or a base id to the base, and passes anything else through untouched by
design. The *relative* form `models/gemma-3-4b-it-4bit` is "anything else", because `hf_id` is
resolved to an absolute path at load. A record that names a checkpoint by its relative path will not
resolve through it. Not a defect this migration introduced; worth knowing before someone writes a
comparison that assumes it does.

---

## 14. The Research Division's answers, and what they change

`CUDA-MIGRATION-RESEARCH-BRIEF-ANSWERS-2026-09-09.md`, five questions answered with sources and
versions, measured on CPU with torch 2.14.0 and transformers 5.16.1 on this machine. Two verdicts
are "does not work, do this instead", and one of those invalidates a written test. In their order.

### 14.1 The golden trajectory test splits in two. G-2 as written is invalid.

**Reproducing MLX recordings on CUDA token-for-token is not achievable**, and the plan's acceptance
test assumed it. Divergence is not gradual: one flipped argmax then complete separation, and over
2,000 to 2,700 tokens the probability of zero flips is effectively zero. bf16 carries eight mantissa
bits against a 262,208-entry vocabulary, so exact top-1 ties occur and tie-breaking has differed
between CPU and CUDA. Nobody gates on cross-backend token identity.

**What is achievable**: same device, same process, batch one, fixed shapes — CUDA-versus-CUDA
token-identical replay, because the forward has no atomic adds. **Byte-identical trajectories are a
within-backend, fixed-topology property**, which the survey's determinism note had already said.

**G-2 becomes two tests.** (a) **CUDA-versus-CUDA exact replay**, fifteen of fifteen, bit-identical,
within one build and device. (b) **MLX-versus-CUDA as a tolerance comparison**: teacher-forced
argmax agreement with the survey's attribution rule (a flip at P ≥ 0.99 fails outright), the
divergence-index distribution with a floor, top-5 Jaccard, and per-step KL percentiles against the
recorded layer-34 distributions. The free-running cross-backend token match is struck.

### 14.2 `attn_implementation="eager"` on the replay path — for determinism, and for that alone

**Corrected by the Research Division at `c0e4233`: one of the two reasons below was a warm-up
artefact and is withdrawn.** Best-of-three steady-state timing shows batched Jacobian rows about
**1.5x faster than sequential under both implementations** — `sdpa` 9.2 ms against 13.6, `eager`
9.9 against 15.1; re-run here, 1.66x and 1.61x — and the missing batching rule costs a one-time
first-call penalty of 22.9 ms, nothing per call after. The batching argument for `eager` collapses;
the determinism argument stands alone and is unaffected. Two justifications described as converging
did not converge. The test that caught it was written best-of-three, which is the argument for
landing a measurement as a test rather than a script.

One flag, now **one** reason. Under `sdpa`, Gemma 3 runs **two attention kernels in one
forward** — sliding layers always carry an explicit 4D mask and cannot take the `is_causal` fast
path, full layers receive `None` and can — which is a dispatch split inside every forward. And the
efficient-attention backward has **no batching rule** (pytorch #117016, open; its fix unmerged), — which the corrected measurement shows costs a first-call warm-up only, not throughput; **withdrawn as a reason for `eager`**. `flash_attention_2` is a hard error under `torch.func`.
`torch.use_deterministic_algorithms(True)` is mostly a backward-pass list and does nothing about
reduction order; replay is batch one, unpadded, identically chunked.

### 14.3 §6.3 works, with a different mechanism and a restated memory claim

Use **`torch.autograd.grad(..., is_grads_batched=True)`** rather than `vjp` + `vmap` directly: same
batching backend, but it keeps leaf and `requires_grad_` semantics so upstream's `ActivationRecorder`
works unchanged. Forward hooks are fine; `register_full_backward_hook` is a hard error; never pass
`output_hidden_states=True`. The saving is real and is on the forward tape only — activations saved
once instead of `dim_batch` times — paid for with transient gradient buffers, which is why `jacrev`
exposes `chunk_size`. **The test is a timing ratio**, batched against sequential on a small fixture,
failing below one; a warning-based assertion would not fire on this path.

**And the memory comparison in §6.3 was against a straw man.** Upstream fits at **128 tokens by
default** (`fit(max_seq_len=128, skip_first=16)`, `encode(max_length=max_seq_len)`): ~168 MB per
layer at `dim_batch=128`. The 3.7 GB per layer was *our* extrapolation to 2,816 tokens, which upstream
never does. The memory problem is created by our departure, not by upstream's design.

### 14.4 The 128-token fit window is upstream's default, and transcript-length fitting is a departure

The estimator truncates every prompt to 128 tokens, drops the first 16 and the last one, and fits on
at most 111 positions per prompt; the published lenses were fitted so at `n_prompts=1000`. A position
selector exists **only at readout** (`JacobianLens.apply(positions=)`), never at fit. `merge` is an
`n_prompts`-weighted mean. Upstream has one commit, is unmaintained, has no PRs merged, and
Neuronpedia's runner is not public, so **no diff against the hosted recipe is possible beyond its
recorded flags**.

**Consequence, stated rather than inherited.** WS-D's golden test refits at 128 tokens and *is*
like-for-like. §6.2's position bands at transcript length are a **different estimator by
construction** — the same fact this plan recorded yesterday as the 21.8x extrapolation, now seen from
the fitting side — and they need their own justification (which they have: no agentic reading is in
range) and their own memory model (§10.2). The two are never compared as if they were one.

### 14.5 The cache: offsets from the API, never from tensor shapes, and recording before rewind

`layer.keys.shape[-2]` reads 7 where the absolute position is 12 on every sliding layer. Offsets come
from `get_seq_length()` and `get_mask_sizes()`. `crop()` on a sliding layer at or past the window
**raises** unless `activate_past_recording()` was called before the window filled; with recording
active, negative arguments work and positive ones raise. So any strategy that rewinds — `trim`,
`snapshot`, `history` — calls `activate_past_recording()` on every sliding layer at construction.

### 14.6 Training: the memory was understated, and world-size-1 FSDP is not the single-device path

AdamW under bf16 mixed precision is **16 bytes per parameter** sharded, not twelve: Gemma 3 4B at
4.30B parameters is **68.8 GB total**, and 27B is **438.9 GB total, 54.9 GB per device at eight
ranks**, before activations and before the gathered `embed_tokens` root unit (2.82 GB at 27B). So
**one 80 GB device is marginal for 4B full fine-tuning**, not comfortable, and the 262,208-vocabulary
logits are the first thing to cut: **fused or chunked linear cross-entropy before the first run**,
not after the first out-of-memory.

Use **FSDP2** (`fsdp_version: 2`; FSDP1 is deprecated from torch 2.11), load on rank 0 rather than
meta-device init, and let `Trainer` own activation checkpointing and the token-normalised
accumulation (`num_items_in_batch`) rather than reimplementing them. **Do not develop under
world-size-1 FSDP**: FSDP2 at world size one silently zeros gradients for some parameter shapes in
non-root units (pytorch #144045). So §12.2's "world size 1 is the same code path" is withdrawn: the
single-device path is plain training, and FSDP is the multi-device path. The `1 + weight` RMSNorm is
a live trap under mixed precision, since its weights are stored as zeros.

### 14.7 What is measured and what is not

Everything above marked measured was measured on **CPU**, on this machine, with torch 2.14.0 — which
means torch exists here in some environment, and §11.3's bridge is not hypothetical. Nothing is
verified on CUDA. The Research Division's scripts can be moved into the tree as tests, and the
timing-ratio test of 14.3 and the two-kernel dispatch check of 14.2 should be.

### 14.8 A fourth upstream default, the sharpest in its class, and two process rules from the same hour

**Upstream fits at 128 tokens and reads out at 512, by default, silently** (D-CRO, verified against
the clone at `4383fda`). `HFLensModel.encode` defaults to `max_length=512`; `JacobianLens.apply`
defaults to `max_seq_len=512`; `jacobian_for_prompt` passes its own `max_seq_len=128` into `encode`.
A caller running `fit(...)` then `apply(...)` at their defaults applies a lens four times outside the
length it was fitted at, and neither signature warns them. That is this programme's 21.8x arriving
from upstream's own defaults, and it is the specific trap the instruction to use upstream directly
was most likely to spring. **The adapter passes lengths explicitly at both ends and relies on
neither default.** Upstream sets no `attn_implementation` anywhere in `jlens/`, so `eager` is ours
to set; and there is no fit-time position selector — `jacobian_for_prompt` takes only `skip_first` —
so the selector of §6.2 is ours to build.

**§6.2, restated more sharply than "a declared departure".** Because 128 is definitional, a
transcript-length band fit is **not a higher-resolution version of the hosted lens; it is a different
estimator that happens to share an implementation.** The recovery gate — a band covering all positions
reproduces the plain fit — proves the *selector* is faithful. It cannot prove the *lengths* are
comparable. Those are different guarantees, only the first is testable, and the two lenses never
appear in one table without that sentence.

**Process rule, recorded against the Chief.** Landing Codex's lens-fitting branch brought
`run_native.py` into `research/records/` with no refusal guard: it takes the model-run lock and loads
a 4B checkpoint if invoked. I ran the tests near the diff and reported them green. The rule tests that
scan the tree for unguarded launchers are not "tests near the diff"; the full suite would have caught
it and the D-CRO's did. **A merge that brings scripts into the tree is followed by the full suite.**
The guard was added in the established form; the record was not touched; the rule was not relaxed.
It is the first time the guard rule caught a script arriving from another seat, which is what it was
written for.

**Coordination rule, from the manifest race.** Two seats edited `pyproject.toml` within one hour —
`uv add torch` at `d6dc41f`, Phase 0's restructure at `7eef9f8` — and only commit order decided the
outcome. It resolved correctly because the second edit's rewrite subsumed the first, which is luck.
**The manifest and the lockfile are one seat's files at a time, announced like a window.**

**The environment, for the record.** Torch 2.14 with MPS is in the venv beside MLX; the state
`uv sync --extra mlx --extra cuda` produces. **A plain `uv sync` strips MLX** and would break every
MLX record, replay and suite run on this machine; the D-CRO has surfaced that to the Director
directly, since the failure would look like a broken repository rather than a missing extra. The
full suite with torch present: 2,123 passed, nothing skipped — `transformers` takes different code
paths with torch importable, and they are clean.

### 14.9 Two things from landing the measurements as tests

`tests/torch/` holds the batching-ratio, cache-trap and mask-dispatch tests, both attention
implementations asserted faster than sequential because both are. **Writing the timing best-of-three
is what exposed the warm-up artefact** that had passed as a finding in a report; a script would have
published it. The general rule: **a measurement that will be relied on is landed as a test, so it is
re-run on every version rather than believed once.**

A trap for whoever extends that directory: **`tests/torch/__init__.py` must not exist.** A package
named `torch` under `tests/` shadows the real one and breaks collection with an error that looks
nothing like the cause. The directory is deliberately a namespace.

For the record: torch was installed at 10:04:06 and `accelerate` at 10:07:41, before the Research
Division read the brief, and pinned at 10:21:08 by the Chief. The Director installed it.

---

## 15. Branch topology and two seam rulings, settled by the first status reports

### 15.1 `cuda-migration` is the integration branch; the main line takes no CUDA code

Settled when Engineer 1's worktree made it concrete. **Seat branches** — `cuda-migration` (WS-B),
`codex/cuda-torch-seam` (WS-A), `cuda-ws-c` (WS-C), `cuda-ws-d` (WS-D, once step 1 lands) — carry
each stream's code. **`cuda-migration` is the integration branch**: seat branches merge into it when
their gates pass, and it merges the main line in regularly. **The main line** (`codex/agent-v2-specs`)
carries the plan, the orders, the records, Phase 0's packaging and registry, and the Research
Division's measurement tests — what every seat needs regardless of backend — and takes no CUDA code
directly. It merges `cuda-migration` back on the Director's word. A seat working in the main checkout
commits only its own files by explicit path, and the Chief's commits do the same, so that
uncommitted work in a shared checkout is never swept in.

### 15.2 The generation loop produces from the model's logits; the readout is compared, never substituted

Engineer 1's first loop generated from `view.native_readout(hidden)` so that a generated token and a
captured one came from one readout by construction. The record was written the other way round. On
MLX, `NativeCapture.__call__` handed **the model's own logits** to `sink.output`, the `argmax` rows
were computed from those, and the readout gate then recomputed `native_readout(h)` at layer 34 and
**compared** it, recording the error on every forward (divergence §3.6). Inverting that makes the
readout the producer and the gate a comparison against the thing that generated the token, which
silently changes what the golden records mean on the tolerance half of G-2. **Ruled: the loop calls
`logits = generation_model(ids, cache)` through WS-A's capture wrapper, which owns the backend
difference — HF returns `CausalLMOutputWithPast` and takes `past_key_values=` — and never calls
`native_readout`; the capture session does, in the gate, as today.**

### 15.3 WS-B's first status, for the record

`research/acceptance/golden_trajectories.py` loads all fifteen records with hash chains verified;
every one of 5,245 emitted tokens equals its forward's argmax, the record's own oracle; the agentic
subset is 4,801 and both figures are right. The stop rule is one `_consume_stream` fed by both
backends; `torch_greedy_stream` is ~50 lines under it. **No torch reproduction has happened**; the
replay generator reads its answers from the record and passes by construction, prints that in a
banner, and a `--self-test` plants token flips and requires divergence at the planted index. That is
the discipline: a harness that passes trivially and says so, with a check that proves the comparison
can fail. Two rule-test failures on that branch are the guard the D-CRO added at `d6dc41f`, which the
branch predates; a merge from the main line resolves them.

## 16. Evening amendments, 2026-09-09

### 16.1 The CPU gates load bf16, promote per block, and declare the rope dtype

Codex's WS-A record measured what §12.1 asked for and found it does not fit: the text weights alone
are 14.46 GiB in float32 against R47's 10.66 GiB cap, before activations or logits. The ruling is
not a larger host. `google/gemma-3-4b-it` is stored in bf16, so a float32 load is an exact upcast of
the same values, and the view's `_call_promoted` performs that upcast per block at call time: the
hand-run loop's matmuls at bf16 loading are **the same arithmetic on the same numbers** as at float32
loading, with 7.2 GiB resident instead of 14.5 and a ~0.4 GiB transient per block. What is *not* the
same is the natively observed bundle: the rotary tables and masks come out of the native forward in
the model's dtype, so a bf16-loaded gate runs float32 arithmetic over bf16-rounded `cos`/`sin`,
where the MLX golden computed its rope in float32. That difference is real, small (bf16 rounds at
2^-8), and must be declared rather than absorbed: the gate record reports, for its sequence length,
the maximum deviation between the model's rotary module evaluated in float32 and the observed
tables, as one number beside the residual agreement. Two more constraints from the same arithmetic:
the fp32 `unembed` promotes the tied 262,208 × 2,560 head (2.7 GB) per call, so gates unembed scored
rows only, through `cached_logits` or `native_readout`, never a full sequence; and the projection
(7.2 + 0.4 + 0.7 GiB of native bf16 logits at 1,400 tokens ≈ 8.5 GiB) is measured before it is
believed, per §10.2, with MLX not loaded. The float32-loaded comparison is a control for the GPU
host, where 14.5 GiB is nothing. Daniel's Q5 (capture dtype) is adjacent and still his.

### 16.2 One seam for reaching upstream

`load_upstream()` (WS-D) is the only way `jlens` enters an interpreter: it takes `$JLENS_PATH` or the
reference clone, puts it first on `sys.path`, refuses a second jlens the interpreter already holds,
and records the commit in ν. WS-A's `arch_torch.py` and `torch_capture.py` imported `jlens` at module
top level with nothing declaring it; in the shared venv both test files ERROR at collection and abort
the run, and beside a pip-installed jlens the seam would correctly refuse them. Ruling: the D-CRO
relocates the seam to a top-level `upstream_ref.py` (no lens-fitting imports; re-exported where it
was), Codex imports through it, `UpstreamUnavailable` also subclasses `ImportError` so a missing clone
is a skip with a message and never a collection error, and the `[cuda]` extra pins
`jlens @ git+https://github.com/anthropics/jacobian-lens@581d398…` for a fresh box, with
`_clone_commit` reading the install's `direct_url.json`. Never vendored, as §2 says.

### 16.3 Gemma 3 is the constraint, not the target

Daniel, this evening: larger models are expected as soon as the hardware allows; Gemma 3 4B is what
the laptop can hold, not a preference; the representations in larger models are the interesting
ones. Consequence: model-agnosticism is the deliverable, not a courtesy. The WS-A view already
discovers through upstream's `_find_layout` with our paths as fallback, reads `attention_span` from
`config.layer_types`, and refuses what it cannot represent instead of mislabelling it; the
acceptance kit takes any HF causal LM. §10 is the main path, since a 27B or 70B model never fits
the laptop and "local-first" is a development discipline for the 4B only. Every ruling from here
asks whether it holds for a model nobody has named.

### 16.4 Worktrees and the shared venv

The venv's editable install resolves `local_llm_lab` to the main checkout's `src/`. In any seat
worktree, without `PYTHONPATH=<worktree>/src`, the tests exercise main's code and new modules do not
import. Confirmed in `cuda-ws-c`. Every seat runs with that variable set, and a record's "tests
passed" names the source path they resolved to.

### 16.5 `device.py` landed, and the gated-delta deletion is withdrawn

`src/local_llm_lab/device.py` is the WS-E device shim: `backend()`, `select()`, `pin()` (deterministic
algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8` before the first CUDA use, TF32 off, cuDNN benchmark
off, the seed), `describe()` (read back from the runtime, never echoed from the arguments; says
`UNPINNED` until `pin()` runs; imports torch only as the backend and mlx never), the six box calls
two ways with a multi-device `"all"` form, and `budget()` for the R47 fraction of what the device
grants. SWE-1's kit reads `describe()`; WS-C wires the pipeline `train` stage's torch branch behind
`backend()`. The gated-delta deletion the ledger listed is withdrawn: the `[mlx]` extra keeps the
recurrence (ruling 5), the torch view refuses recurrent backbones, and §16.3 rules out anything
Gemma-scoped. `tests/test_import_tree.py` (SWE-2's guard) is on main so every seat's suite asserts
which tree it is testing.

### 16.6 WS-A review

Read in full by the Chief: `arch.py` diff (a pure extraction, AST-identical contracts, the MLX
suite green on the branch), `arch_base.py`, `arch_torch.py`, `torch_capture.py`, the record and the
acceptance primitives; 79 branch tests pass on torch 2.14.0 with the clone on the path; the dry-run
merge into `cuda-migration` is clean. Verdict: **passes on substance, not merged yet**, three edits
first: (1) imports through the §16.2 seam and `importorskip` in both torch test files; (2) the
evidence re-run on torch 2.14.0, the plan's floor, since the record's isolated runtime was 2.9.1;
(3) `WS-A.patch` and `patch.json` do not merge — a record cites the commit range, it does not carry
a 2,584-line copy of the diff that goes stale at the first edit. Then the full-checkpoint gates run
locally under §16.1. Noted, not blocking: `residuals()` costs an observation forward plus the
hand-run loop, which the graph-once estimator (§6.3) addresses for production.

### 16.7 The multi-device gate compares gradients, not losses; the chunked loss runs inside the root unit

SWE-2 built the two-device arm on CPU (FSDP2 accepts a `gloo` mesh in torch 2.14.0, so the whole
gate is answerable on the laptop) and found that the gate as written would have passed a broken
configuration. With the blocks sharded and the root left replicated outside any unit, FSDP2 reduces
nothing for the embedding, final norm and tied head, so each rank keeps its own half-window gradient
forever. At step zero, identical weights, before any update:

| quantity | one device | two devices | relative |
|---|---:|---:|---:|
| loss | 4.1746088012 | 4.1746088012 | 0.00e+00 |
| grad norm, sharded blocks | 1.89720254 | 1.89720254 | 2.17e-10 |
| grad norm, root unit | 0.93408004 | 1.31364770 | **4.06e-01** |

Loss at step zero on identical weights and the same rows is identical by construction and asserts
nothing; afterwards it is one step behind the error (8.83e-04 over six steps, inside any tolerance
anyone would have written). **Ruling, applied to every statement of the gate:** at step zero,
per-parameter gradients agree between one device and W to float32 epsilon, as a max relative
deviation per parameter and not a norm, so a wrong slice cannot hide inside a right norm; after N
steps, per-parameter values agree likewise; the loss curve is reported and passes nothing. With the
root reduced correctly the deviations are 5.39e-08 (loss), 3.79e-07 and 1.39e-07 (gradient norms):
float32 epsilon, where reassociated summation lives. The rule behind it is in the method record: a
gate compares the quantity nearest the mechanism, never one downstream of it.

The cause was a design collision, and it is settled: `causal_lm_chunked_loss` reads `lm_head.weight`
directly to avoid a 262,208-wide logit tensor, and FSDP2 unshards a parameter only inside its unit's
forward, so touching the weight from outside raises the mixed Tensor/DTensor error. Neither hand
all-reduce (a second reduction path, which is the standing invitation to the silent error above) nor
the head as a per-chunk module call (unsharding the largest parameter once per chunk). The chunked
loss moves **inside the root unit's forward**: a thin wrapper module whose forward runs the inner
model and then the chunked cross-entropy against the raw weight; blocks sharded as units, the
wrapper sharded as the root, holding the tied embedding, the norm and the head together as a tied
pair must be. Inside that forward the weight is the unsharded tensor and the reduce-scatter hooks are
armed, which is how the fused-linear-cross-entropy kernels run under FSDP2 in the wild. Cost: the
root's parameters resident unsharded from forward through backward, once per step; the memory rung
is `reshard_after_forward=True` on the root, measured per §10.2. The hand-reduced arm stays in the
record as the negative control: the configuration the old gate passed.

**Executed, SWE-2, 8f9a178 on `cuda-ws-c`.** The wrapper is built and sharded as the root (under
tying the head *is* the embedding tensor, so the root holds one parameter and not a straddling
pair), and the amended gate ran three arms against single-process training on the same rows,
per parameter, `max|a − b|` over the tensor's own scale:

| arm | loss, step 0 | grad, step 0 | value, final |
|---|---:|---:|---:|
| blocks and root sharded | 0.00e+00 | 1.80e-07 | 8.56e-06 |
| blocks only, root hand-reduced | 0.00e+00 | 1.80e-07 | 8.58e-06 |
| control: root reduced by nothing | 0.00e+00 | **1.67e+00** | **1.12e+00** |

The control's loss is bit-identical to the correct runs while its gradient is 167% wrong and its
parameters end 112% wrong. Two correct configurations sit at float32 epsilon and the broken one
is off by more than one, so there is no tolerance at which the old gate separates them and none at
which the new one fails to. `reshard_after_forward` on the root is recorded as the memory rung and
not yet measured; that measurement is the device's. Two-rank `gloo` on CPU only; no CUDA, no NCCL,
no real checkpoint.

**Followed to its end (SWE-2, 909bd21): the wrapper is what `Trainer` is handed, and that caught a
third member of the family.** `Trainer._save` branches on `isinstance(model, PreTrainedModel)`; the
wrapper is not one, so it wrote a bare state dict with every tensor named `inner.*` and no
`config.json`, nothing raised, and the artefact was one `checkpoint_delta` could not name-match and
the registry could not load. `_save` now saves the inner model and the test asserts the checkpoint
is a **model**: config present, no `inner.` prefix, loads with `from_pretrained`. The rule
generalises: **upstream inspects the object it is handed; a wrapper changes both what runs and what
it is.** Anything handed a wrapper in place of the `PreTrainedModel` inherits this silently, and the
wrapper must also delegate `config` and the checkpointing methods or upstream reaches for them and
finds nothing. Every seat that wraps a model reads this paragraph before handing the wrapper to
anything of upstream's.

**Two claims, kept separate (SWE-2, 41808bc).** The sharded path is validated per parameter on CPU
in the two-device record; `stage_train_torch` runs, single-process; the two joined is **not done**.
Handed more processes, `Trainer` and `accelerate` would distribute under their own default rather
than the sharded path the order requires, invisibly, with the loss falling and a checkpoint
written and every memory figure describing a configuration that never ran. So the stage refuses a
multi-process run and names the gap. Wiring FSDP2 into the stage is WS-C's next task, validated to
two `gloo` processes on CPU; NCCL and more ranks are the device's. Under tying,
`lm_head.weight is embed_tokens.weight`: one parameter with two names, so no flat parameter
straddles two units and there is nothing to shard by halves; that sentence is now in the code.

**Joined (SWE-2, aa96cd8): the stage driving FSDP2 through `Trainer`, single process against two
`gloo` processes on CPU, the tiny wrapper checkpoint.**

| quantity | worst relative deviation | parameters |
|---|---:|---:|
| gradient, step zero, before any update | 1.24e-07 | 26 trainable |
| value, after the step | 0.0 | 80, all |

The gradient is the load-bearing number. The exact value agreement is real and weaker than it
looks: the difference the gradients imply is about 1.2e-11 at lr 1e-4 against a float32 ulp of
1.9e-09 near a typical weight, so the updates are bit-identical by construction at this step
count. The refusal is off; `describe_distribution` puts in every manifest what the distribution
has been measured at (two `gloo` processes on CPU) and what it has not (CUDA, NCCL, more than two
ranks, any real checkpoint). The sharded checkpoint shows the recipe: 54 bfloat16 tensors and 26
float32, frozen trunk at two bytes, trained slice at four.

The first run of the gate failed at 0.499 on gradients and 1.357 on values, and the gate was what
was wrong: `Trainer` shards data across ranks, so an accumulation count left alone doubles the rows
per optimiser step at world size two and the arms optimise different objectives; and even with the
global batch matched, a distributed sampler need not put the same rows in the same step, so any run
longer than one step compares two trajectories. One optimiser step over the whole dataset removes
both. **A disagreement is a defect only once the two arms are the same experiment, and getting
them there is most of the work**; every multi-device gate carries that sentence.

### 16.8 Cache strategies are arms of the gate, decided by fidelity; and a checkpoint never follows the box-state override

SWE-1 built the torch cache strategies against the real `DynamicCache` and measured two things a
stub could not have shown. Arming rollback (`activate_past_recording`) *after* a sliding window has
filled does not raise: the cache reports the right offset and holds one key where six should be,
and would attend over a five-token hole silently, so `enable_rollback` now refuses a cache that has
already advanced. And an armed sliding layer stores everything rather than `window - 1` entries,
so at the map's longest episode (2,749 positions against a 1,024 window, 29 of 34 layers sliding)
the KV cache is 2.15x its bounded size. That is the measured need the deferral was waiting on, and
it is a ratio of a small base: at this checkpoint's four KV heads and the class-default head
dimension of 256, about 136 KB per token, roughly 380 MB armed against 180 MB bounded. SWE-1 read
them off the loaded config (c36de39): 136.0 KiB per token, 177.9 MB bounded and 382.8 MB armed at
2,749 positions, 2.152x. One trap for any model: `head_dim` here is the class default because the
checkpoint's config leaves it null, and deriving it as hidden size over attention heads gives 320,
a quarter too large; read the attribute, never derive it. Memory therefore does not decide. **Ruling:** every
strategy stays refused under torch until the tolerance runner has its baseline at `none` against
the view; then each strategy is its own arm of the same gate and must reproduce the `none`
trajectories byte for byte within the backend. A strategy that changes one token is a defect, not
a speed setting, because the golden records were made under one rule and a rewindable sliding
cache is exactly where stored keys and the mask can part company.

Separately, SWE-2 reported that no stage could load a registered `models/...` checkpoint from a
worktree. The resolver was correct for every process without the override and wrong for every
process with it: `_resolve_checkpoint` read the primary through `box_state_root`, whose
`$AGENT_V2_BOX_STATE_DIR` redirect exists so an isolated run cannot take the machine's lock, and a
checkpoint that followed the redirect resolved into scratch. Two things shared one reader and only
one of them may be redirected. `primary_checkout_root()` is now the git-derived half on its own,
the resolver uses it, and the test builds a real linked worktree whose primary has a space in its
name, as the real one does. The registry comment that says "the project root" is frozen by the
records' `registry_sha256` pins and stays; `models.py` carries the truth.

Two rules from the same hour, both SWE-2's: anything that bypasses `forward` inherits none of what
upstream attaches to it and must supply it itself; and a suite reading is not a claim unless it
carries its skip count and window state on the same line.

### 16.9 One text-only loader for the torch path, with no model class named

Three seats were loading the checkpoint three ways: Codex's record companion named
`Gemma3ForCausalLM` and `Gemma3TextConfig` with a listed vision-prefix set, SWE-1's tolerance
baseline named `Gemma3ForCausalLM`, and SWE-2's train stage called `AutoModelForCausalLM` on a
plain path, which does not load the official multimodal snapshot text-only at all. §16.3 forbids
the first two in the package and the third is wrong for the checkpoint we have. `local_llm_lab.hf_text`
is the **only** loader, because `AutoModelForCausalLM` on any checkpoint we have builds the
multimodal wrapper: the official snapshot and the repository's own MLX conversion both declare
`model_type: gemma3` and a `text_config`, so "plain path" is not a case that exists among the
registry's entries (SWE-2, a9192a1). `checkpoint_metadata` reads the config and the safetensors
headers and loads no tensor; a wrapper is detected from the checkpoint's own `text_config` and `language_model.` prefix,
never from a model name; the text config goes through `AutoConfig.for_model`, the model through
`AutoModelForCausalLM` with the prefix mapped away; and every non-text prefix in the header must be
exactly the unexpected-key set, no more and no less. Codex's fail-closed checks are kept and
generalised: missing, mismatched or errored keys refuse; a text tensor the model did not take or
whose shape changed refuses; a tied weight that came back as two tensors refuses; a parameter on
the wrong device or in the wrong dtype refuses. The report is a reading of the loaded object. Seven
tests on tiny random models, plain and wrapped, with two foreign towers beside the text tower,
one of them mirroring the official snapshot as it is on disk (`model_type: gemma3`,
`architectures: [Gemma3ForConditionalGeneration]`, `text_config.model_type: gemma3_text`,
`language_model.model.*` beside `vision_tower.*`, `__metadata__: {format: pt}`), and one
proving the MLX conversion is refused by its own `format: mlx`, since nothing in its config
distinguishes it from the snapshot. Two rules from SWE-2's guard that passed the object it
existed to catch: **a synthetic fixture carries the real artefact's declared type and key layout,
or the loader path is untested by construction**; and **a guard that passes the object it
exists to catch is worse than none, because its silence is read as evidence.**

**Two more from the same seat, writing the joined gate (7f5c6e4).** First, a model loaded under a
key mapping saves with the mapping reversed by default (`save_pretrained(save_original_format=True)`),
so the loader and the save are each correct and together write `language_model.model.*` tensors
under a config that says `Gemma3ForCausalLM`: an artefact nothing reads. The stage now saves in
the model's own layout, and the loader neutralises the stored mapping so a plain save cannot
reverse it. Second, the assertion that let it through: "the checkpoint reloads" via
`Gemma3ForCausalLM.from_pretrained`, where missing keys are a warning, so every weight came back
freshly initialised and the assertion was satisfied by a model that had learned nothing. **An
assertion that an artefact loads is worth nothing unless the loader fails closed; `hf_text` is
the only reader in tests as well as in code.** Also verified against disk: both repository
conversions carry `format: mlx` and the snapshot `format: pt`, so the refusal blocks
`gemma3-4b-bf16` and the torch path's entry, `gemma3-4b-cuda-bf16`, loads as a wrapper with 439
non-text tensors beside 444 text ones.
Every torch load of a registered checkpoint goes through it: WS-B's baseline, WS-C's stage, WS-A's
gates. The CUDA memory rung, a `device_map` under `accelerate` instead of a CPU load and a move,
is deliberately not taken until measured.

**The counted run, owed since the resolver commit.** Main at 7e4c34e under the Chief's own window,
so the model-reaching files ran rather than skipped: **2,123 passed, 0 skipped, 0 failed in 122 s**,
and `tests/torch` 6 passed. Every earlier "full suite green" of the day lacked its count because
`addopts` already carries `-q` and a second `-q` suppresses the summary line; the exit codes were
real, the counts were not captured, and that is the Chief's, twice.

### 16.10 Lint: one auto-fix that is wrong, and one deliberate pass rather than four incidental ones

SWE-2 found that ruff's SIM118 auto-fix rewrites `for key in handle.keys()` to `for key in handle`
on the assumption of a mapping, and a safetensors `safe_open` handle is not one: the rewrite is
applied by `ruff check --fix`, produces no finding and no import error, and fails only at runtime
with "object is not iterable". Every reader of safetensors in this tree, in lens fitting, capture
and the registry, is a candidate. **Rule:** `--fix` is never run blind over a file that reads
safetensors; the `.keys()` call carries a `noqa: SIM118` with the reason beside it, so the next
person is told rather than tempted. Three of SWE-2's own findings were real rather than cosmetic and
are worth knowing as shapes: `Any` in annotations never imported, surviving only because
`from __future__ import annotations` never evaluates them; a `zip` without `strict=` over two lists
equal today, which is what stops a later edit truncating a batch in silence. The tree carries about
75 older findings, mostly line length in records scripts; they are swept in one deliberate WS-E pass
by one seat, not by incidental edits from four.

### 16.11 The first number that is not by construction

SWE-1, 4d553ef, `agentic-d2-calculate-0158`, the shortest episode of the fifteen, chosen so the box
could be released: the MLX 4-bit records against CPU bfloat16 through the merged view, teacher
forced one causal forward per turn, `cache_strategy: none`, `device.pin` first, eager attention,
83.5 s.

| | |
|---|---:|
| argmax agreement | 98 of 103 (0.9515) |
| flips | 5 |
| flips at recorded P ≥ 0.99 | 0 of 78 confident positions |
| flips among the 25 unconfident positions | 5 |
| chance of that under independence | 8.4e-4 |
| gate | PASS |
| peak memory, measured | 7.88 GiB against 7.56 projected from 7.23 GiB of text tensors |
| top-5 Jaccard, mean and worst | 0.7018, 0.4286 (reported, not gated) |

Every flip sits where a precision difference puts it and none where a mask, position, entry or norm
defect would. The projection missed by 4.2% in the safe direction, allocator and workspace, with its
basis stated. What it does not say: one episode of fifteen, 103 emissions of 5,245, and the
shortest, the same instance whose cheapness produced an earlier calibration error. Two things
follow. The report now prints every flip with its recorded probability and the episode's count of
confident positions, because "0 at P ≥ 0.99" cannot be told from a threshold nobody approached
without that count. And the remaining fourteen are scheduled by what they exercise, not by cost:
the eleven under twenty minutes next, about 75 minutes; then the four long ones, `update-0028`'s
2,607-position turn above all, as one announced block, because the sliding window is exercised only
beyond 1,024 positions and the mask-dispatch control on the tiny model bit only there. They are
the point of the gate, not its remainder. Full corpus about 324 minutes of CPU box time.

### 16.12 The laptop is for tests; the device is for runs

The Director, 2026-09-10 early: "All subsequent runs will use the rented GPU. Do not plan for long
runs that will tie up this box. Test as much as you can, but accept we may need to resolve bugs in
the actual environment." This amends §12.1's "CPU torch is the development environment": CPU is
where tests, fixtures and tiny models run, and every model-scale run from here is the device's.
Consequences, in order:

- SWE-1's eleven-episode block stops after the episode in flight and releases the box; what
  completed is the laptop calibration the device numbers are compared against. The remaining
  episodes and the four long ones, which exercise the sliding window, run in the device's first hour,
  where the whole corpus is seconds.
- Codex's next arm (the bf16-loop seam-exactness check, the three controls at 1,400 tokens, the
  measured peak) runs on the device, not on CPU. Gates 1–4 on the real checkpoint close there.
- WS-D's golden test, finite-difference against exact on the 4B, runs on the device; the adapter's
  fixture tests and the corpus freezing are the laptop's.
- Every seat's record becomes the first hour's checklist: each device item with the number it must
  produce and the local figure it is compared against, in execution order. `acceptance_gates.py`
  and the tolerance runner are the first hour's tools.
- Bugs found only on the device are expected and are not failures of the local work; they are
  resolved there, recorded there, and their tests come back here.

No announced blocks on this box from now; windows of minutes for tests are fine.
