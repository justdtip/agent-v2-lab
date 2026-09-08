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

**Owns.** New `src/local_llm_lab/arch_torch.py`; a small extraction of the backend-neutral parts of
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

## 8. Policy questions for the Director

The plan cannot decide these.

1. **One lock per box, or one per device?** `runlock` enforces one model load at a time because
   unified memory is one pool. On a multi-GPU box, two loads on two devices do not contend for
   memory but do for the CPU, the disk, and the timing of each other's runs (R61). My recommendation
   is one window per **device set**, with R61's announcement carrying the device list, and the
   suite's skip gate keyed on the devices a window names.
2. **Which quantisation does the CUDA path run?** The map was read at MLX 4-bit. CUDA offers bf16
   (matches the hosted lens; the bridge requires it) or bitsandbytes 4-bit (matches neither MLX's
   scheme nor bf16). Recommendation: **bf16 as the CUDA reference**, and the MLX-4-bit-versus-CUDA-bf16
   difference measured once on the golden episodes and recorded as the comparability bound. With
   memory no longer the constraint, 4-bit's reason for existing on the laptop does not transfer.
3. **Do the probes migrate?** Recommendation: no, except `patch.py`'s intervention core, until a
   probe is ordered on Gemma. Fifteen thousand lines nobody has run on this model are not reuse.
4. **Is the laptop MLX path kept?** Recommendation: yes, as the `[mlx]` extra, because it is the only
   place the golden records can be regenerated and the only cross-check on CUDA determinism. The
   cost is the shared base extraction in WS-A, ~40 lines.
5. **The branch.** Proposed `cuda-migration` from `48db38f`. The plan lands on the current branch so
   every seat reads it; code lands on the new one.

---

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
