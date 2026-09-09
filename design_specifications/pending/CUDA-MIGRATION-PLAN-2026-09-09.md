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
| `lens_fitting/jacobian.py` | 957 | ~400 (plan, gates, records) | ~30 | ~80-line adapter → `jlens.fitting` | 0 | **~450 deleted** |
| `lens_fitting/replay.py`, `runtime.py`, `prose.py`, `validation.py` | ~1,600 | most | ~60 | — | ~20 | 0 |
| `pipeline/jlens.py` | 1,987 | — | — | — | 0 | **defer** (probe-era, superseded; survey to confirm) |
| training: `train_expanded.py`, `depth_expansion.py` | ~600 | depth_expansion | ~40 | PEFT + accelerate | **~180** | `gated_delta_*` (614, DeltaNet) **deleted for Gemma** |
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

**Golden test.** A 40-row smoke train on one GPU and on two produces the same loss curve to
tolerance and an adapter `adapter_delta.py` reads; then arm 1's recipe (top-8 layers) reproduces the
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

**Budget.** ~80 adapter + ~45 regression edits + ~400 extensions; ~450 deleted.

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

- **Every workstream develops and validates on CPU torch in float32**, against the MLX golden
  records, on this machine. What passes on CPU is recorded — gate by gate, at a commit — in a
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
  fixed-topology property**; a sharded matmul reduces in a different order.
- Determinism on CUDA: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  TF32 off, a pinned SDPA backend.

### 13.5 The reuse ledger, the survey's, replacing §4

In-scope source 26,062 of 44,739 lines. **New 1,473 — 5.7% of in-scope — against 22,833 kept, 1,338
edited, 695 replaced by upstream, 18,677 deferred, 1,196 deleted. Net source delta −418.** Seventeen
lines kept, edited or replaced for every new line. Tests are a further ~1,450 new. The per-subsystem
table is in the survey record and its counts are the ones to hold seats to.

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
