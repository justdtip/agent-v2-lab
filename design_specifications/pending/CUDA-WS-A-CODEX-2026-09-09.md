# WS-A, for Codex through the Director: the torch seam, and the graph-once estimator

**From the Chief, 2026-09-09.** Read `CUDA-MIGRATION-PLAN-2026-09-09.md` first, §1, §3, §7, §10–12.
This is the hardest piece and the one everything else is measured against. Nothing CUDA runs until a
device is rented; **you develop and validate on CPU torch in float32 on the laptop**, against the MLX
golden records, and you record what passed gate by gate at a commit. Anything not run is marked
unexecuted. The Director's standing rule for this pivot: **reuse before invention; a new convention
where an implementation exists is a defect.**

## Use upstream directly, modified

The cloned `anthropics/jacobian-lens` at `/Users/daniel.tipton/reference/jacobian-lens` (Apache-2.0)
already contains two of the four things this stream builds. Build on them; do not reimplement beside
them.

| upstream | what it does | what you do with it |
|---|---|---|
| `jlens/hooks.py::ActivationRecorder` | forward hooks on `model.layers[i]`, stores each block's output keyed by index, `start_graph_at` roots autograd at a chosen layer | **`TorchCapture` extends it**: add a pre-hook for the entry residual and the per-layer kwargs, the sink protocol, and the intervention hook |
| `jlens/hf.py::_find_layout`, `HFLensModel` | structural discovery of `layers`/`embed`/`norm`/`lm_head`; `unembed`; `encode` | **the torch view's discovery delegates to `_find_layout`**; our `_TEXT_MODULE_PATHS` becomes the fallback, not the primary |
| `jlens/fitting.py::_check_layer_indices` | the layer convention: sources `0..target−1`, target `n_layers−1` | **the residual convention test asserts against it**, so ours can never drift from upstream's |

## What you build

**1. The shared base.** Extract from `arch.py` the backend-neutral parts — index validation,
`lora_targets` over a tuple of linear types, `_validate_scored_position`, `_validate_hidden_span`,
the docstrings that carry rulings — into `arch_base.py`. The MLX view inherits it unchanged; this
is ~40 lines moved, not written, and it is your first commit so the MLX suite stays green.

**2. `arch_torch.py::TorchArchitectureView`**, same public surface as `ArchitectureView`, no
signature change (plan §1). The specific design points:

- `_observe_forward`: a forward **pre-hook** on every `layers[i]` capturing `hidden_states`,
  `attention_mask`, `position_embeddings` as the model passes them. Observe, never reconstruct — the
  principle that found the entry-scale and boolean-mask defects on MLX (`arch.py:461-509`).
- `masks(h, cache)` returns `{i: {"attention_mask": m_i, "position_embeddings": pe_i}}`, an opaque
  bundle per block; `run_block(i, h, masks, cache_i)` unpacks it and calls
  `layers[i](h, attention_mask=..., position_embeddings=..., past_key_values=...)`. HF Gemma 3
  builds `causal_mask_mapping[layer_type]` and `position_embeddings[layer_type]` and dispatches on
  `config.layer_types[i]` — the same two-mask structure MLX had, so `attention_span(i)` reads
  `config.layer_types[i]` directly.
- The cache adapter: HF's single `DynamicCache` presented as the per-block list the repo expects,
  each entry exposing `offset` (from `get_seq_length`) and nothing else until a strategy other than
  `none` is ported. `make_cache` returns it; `cache_trimmable` is `crop`-backed.
- Every hand-run return is `.float()` (diagnostic dtype promotion, plan §3.3).
- `residuals`, `native_residuals`, `residual_source_agreement`, `cached_logits`, `tail`,
  `native_readout`, `unembed`, `final_norm`: as on MLX, over the torch model.

**3. `TorchCapture`**, extending `ActivationRecorder`: the sink protocol
`residual(layer, offset, h)` / `output(offset, ids, logits)` unchanged so `session.py` does not
know the backend changed; and **`intervene(layer, position, fn)`** where `fn: h → h'`, applied in the
forward hook, covering add (today's injection), replace, and interchange
(`h + D(z' − z)` on a frozen support). Design it for replacement from the start; the state
derivation's §7.1 and §11.3 need it and it should not be a second migration.

**4. After gates 1–4 pass: plan §6.3, the graph-once estimator.** Upstream's
`jacobian_for_prompt` replicates the prompt `dim_batch` times and retains the whole graph
(`fitting.py`: `input_ids.expand(dim_batch, -1)`, `retain_graph=True`); at 2,816 tokens that is
~3.7 GB per layer. Rewrite it as `jacobian_for_prompt_vjp`: one unreplicated forward,
`torch.func.vjp` with cotangents batched by `vmap`, the same one-hot-per-valid-position semantics,
the same `.mean` over sources, **a cotangent selector argument** so §6.2's position bands and §6.4's
spans are a mask and not a fork. Keep upstream's function beside it and **assert the two agree on a
short prompt to float32 tolerance** — that is the golden test for the rewrite. Then sharding:
prompts across devices, `JacobianLens.merge` (upstream, already written) as the reduction, world
size 1 being the single-device case.

## Confounds this stream resolves by construction

- The hand-run-versus-native disagreement that cost a day on MLX: on torch the native path *is*
  hooks, and `residual_source_agreement` becomes a regression test rather than a port gate.
- Head capture "architecturally unavailable" on Gemma: it was unavailable to MLX's emitter; a hook on
  `self_attn` is model-agnostic. Not ported now, but the design leaves the seam.
- The residual-convention ambiguity, settled yesterday by reading upstream's source: now asserted
  in a test against that source.
- The 21 GiB fit projection: not a batching choice — there was no batch — but a retained graph, and
  §6.3 removes it.

## Golden tests, in order, on CPU float32

1. Discovery reports 34 layers, 2,560 hidden, `layer_types` spans matching MLX's `attention_span`.
2. `residual_source_agreement` at 64 and 1,400 tokens: zero at both; the mask-dispatch negative
   control non-zero at 1,400 only. **This is the test that has found every seam defect so far.**
3. Layer-34 identity: native softmax top-1 is the emitted token, every decode step.
4. The readout gate with both negative controls, tolerance set from the measured fp32 distribution
   and recorded with its basis; `None` and `0.0` remain different values.
5. (§6.3) `jacobian_for_prompt_vjp` agrees with upstream on a 64-token prompt, layers 0–33, to
   float32 tolerance; then the memory at 2,816 tokens measured, not projected.

## Budget and what is unexecuted

~450 new lines for 1–3 against 1,120 kept; ~120 for 4. Everything marked unexecuted until the gate
above it passes on CPU, and the manifest of what passed at which commit ships with the branch so the
remote's first hour attributes any failure to the device change.
