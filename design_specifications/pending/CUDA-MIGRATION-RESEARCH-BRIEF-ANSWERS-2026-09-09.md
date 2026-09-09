# Answers to the CUDA migration research brief

To the Chief AI Research Scientist, from the Research Division. 2026-09-09.
Five sections, one verdict each. Sources and versions on every claim.

**Method note, and it changes the standing of most of this.** The brief says torch is not yet
installed. It is: **torch 2.14.0**, with **transformers 5.16.1**, in the project venv. So sections
1, 3 and much of 2 and 4 are **measured on this machine** rather than searched, on a Gemma 3 stack
built from `Gemma3TextConfig` with random weights, which exercises the real module code without
needing the checkpoint. Anything measured is marked so. Everything on CUDA behaviour remains
unmeasured, because there is still no GPU.

---

## 1. Batched Jacobian rows through `torch.func.vjp` + `vmap`

**It works, and hooks are not the hazard.** Measured: a real `Gemma3TextModel`, forward hooks on
two blocks exactly as `jlens/hooks.py` does it, the earliest source activation detached and marked
`requires_grad_(True)` as the graph root. Batching the backward over one-hot cotangents agrees
with the sequential result to `atol=1e-5`.

The structural reason hooks are safe: `vjp` runs the forward **once**, outside the batching, and
only the backward is batched. A `register_forward_hook` therefore fires exactly once. The
functorch hazards about hooks and in-place mutation apply to `vmap` over a *forward*, which is not
what this does.

**CORRECTION, 2026-09-09, after landing the tests.** An earlier version of this section reported
that batching is *slower* than the sequential loop under `sdpa`. That was a warm-up artefact and
it is withdrawn. The missing batching rule costs a **one-time first-call penalty**, not a per-call
one. Measured, per call in milliseconds, same fixture:

| `attn_implementation` | batched, call 1 | batched, steady | sequential | first-call ratio | steady ratio |
|---|---|---|---|---|---|
| `sdpa` | 22.9 | 9.2 | 13.6 | 0.62x | **1.50x** |
| `eager` | 10.0 | 9.9 | 15.1 | 1.56x | **1.48x** |

So **batching is worth about 1.5x under both implementations**, and the `sdpa` fallback costs
roughly 14 ms once. Against a fit that runs `ceil(d_model / dim_batch)` backward passes per prompt
over a thousand prompts, a one-time cost is irrelevant.

**What this does to the ruling.** The §1 argument for `attn_implementation="eager"` collapses: the
Jacobian path does not need it. The §2 argument stands on its own and is unaffected, so `eager`
may still be right, but for determinism reasons only. The two justifications I described as
converging do not converge; one of them was an artefact.

The error is worth naming because it is one I had already written down. A single-shot timing on
the first call is warm-up contaminated, and taking the best of several runs is the fix. Writing
the test with best-of-three is what surfaced it, which is an argument for landing measurements as
tests rather than keeping them as scripts.

**There is a blocker in the `vjp` route that the plan does not mention, and it is in our own
hooks.** `ActivationRecorder` calls `tensor.requires_grad_(True)` on an intermediate and returns
`None`. `torch.func.vjp` differentiates only with respect to the primals passed to it, so
`requires_grad_` on an intermediate does nothing there. Taking the `vjp` route means rewriting the
hook to *return* `output + z` with `z` a zero primal per source layer, zero-add rather than
substitution so the source-to-target path survives, since `jacobian_for_prompt` takes gradients
with respect to several source layers in one graph.

**So the cheaper correct change is `is_grads_batched`, and it needs no hook rewrite at all.**
Measured here, same model and shapes:

| `attn_implementation` | `is_grads_batched=True` | sequential loop | speedup | agrees |
|---|---|---|---|---|
| `sdpa` | 22.8 ms | 14.9 ms | 0.65x | yes |
| `eager` | 10.8 ms | 16.2 ms | **1.50x** | yes |

It uses the same vmap backend and inherits the same attention fallback, but keeps leaf and
`requires_grad_` semantics, so the existing `ActivationRecorder` works unchanged. It is documented
as prototype with "performance cliffs", which is precisely what the table shows.

**Why this matters more on CUDA than the CPU numbers suggest.** The missing batching rule is not a
CPU quirk. `aten::_scaled_dot_product_efficient_attention_backward` has no batching rule either;
[pytorch #117016](https://github.com/pytorch/pytorch/issues/117016) is open and its fix
[PR #176265](https://github.com/pytorch/pytorch/pull/176265) is unmerged as of September 2026. The
efficient backend is what a *masked* attention call dispatches to, and §2 below measures that
Gemma 3's sliding layers always carry an explicit mask. In real Gemma 3 that is five layers of
every six. So under `sdpa` on CUDA the majority of the model would take the fallback path.

**Two hard errors to avoid, not slowdowns.** `flash_attention_2` registers custom ops whose
autograd functions lack `setup_context`, which is a hard error under `torch.func`
([pytorch #170834](https://github.com/pytorch/pytorch/issues/170834), open).
`register_full_backward_hook` is likewise a hard error; `register_forward_hook`, which is what we
use, is fine. Avoid `output_hidden_states=True`: transformers v5 attaches its own forward hooks
that stash tensors in a dict.

**Memory, mechanism rather than assertion.** `vjp` saves the forward activations once and `vmap`
gives each backward intermediate a leading dimension; replicate-and-retain saves the activations
`dim_batch` times and keeps them for the whole loop. The saving is real and it is on the forward
tape only, paid for with N transient gradient buffers. This is why `jacrev` exposes `chunk_size`.

**Test to write:** compare batched against sequential timing on a small fixture and fail if the
ratio is below one, rather than asserting on the warning. I captured the batching-rule warning on
the `vmap` path but not on the `is_grads_batched` path, so a warning-based assertion would not
fire for the change I am recommending. The timing ratio is the reliable signal.

**Verdict: works with this change** — use
`torch.autograd.grad(..., is_grads_batched=True)` with the hooks as they are. It needs no hook
rewrite, and it is about 1.5x faster than the sequential loop under both `eager` and `sdpa` in
steady state. Do not adopt `eager` on this account; §2 decides that question. Restate §6.3's
memory comparison against what upstream actually does.

---

## 2. Deterministic greedy generation on CUDA for Gemma 3

**Two questions with opposite answers.** Same device, same process, batch 1, fixed shapes:
token-identical replay is achievable, because the forward contains no atomic adds
([Thinking Machines, *Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)).
**Reproducing MLX recordings on CUDA token-for-token is not achievable**, and the plan's
acceptance test assumes it.

Divergence is not gradual. It is one flipped argmax followed by complete separation
([Yuan et al., arXiv:2506.09501](https://arxiv.org/abs/2506.09501): bf16 greedy, >90% of samples
diverge; fp32 cuts it to ~2.2%). Over 2,000–2,700 tokens the probability of zero flips is
effectively zero. bf16 has 8 mantissa bits against a **262,208**-entry vocabulary (verified,
`configuration_gemma3.py:77`), so exact top-1 ties occur, and tie-breaking has historically
differed between CPU and CUDA ([pytorch #93630](https://github.com/pytorch/pytorch/issues/93630)).

**Measured here, and it bears on the whole section.** Under `sdpa` with no padding, Gemma 3 runs
**two different attention kernels in one forward**:

| layer | type | mask received | can take the `is_causal` fast path |
|---|---|---|---|
| 0, 2 | sliding | 4D tensor `(1,1,12,12)` | no |
| 1, 3 | full | `None` | yes |

`sdpa_attention.py:124` gates on `q_length > 1 and attention_mask is None and is_causal`, and
sliding layers always carry an explicit mask. `eager` removes this dispatch split entirely.

`torch.use_deterministic_algorithms(True)` does not help much here: its op list is overwhelmingly
backward-pass, and it does nothing about matmul reduction order or batch invariance
([2.14 reproducibility notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html)). Replay
must be batch size 1, unpadded, identically chunked; batch-size invariance does not hold.

**Design the test as two tests.** (a) CUDA-vs-CUDA exact replay, 15 of 15, bit-identical, within
one build and device. (b) MLX-vs-CUDA as a tolerance comparison: divergence index distribution
with a floor, top-1 agreement, top-5 Jaccard, per-step KL percentiles against the recorded
distributions. This is what llama.cpp and the transformers CI do; nobody gates on cross-backend
token identity.

**Verdict: does not work, do this instead** — split the acceptance test, and use `eager`, which is
also what §1 needs.

---

## 3. `transformers` 5.16 cache for Gemma 3

All measured on this machine, `DynamicCache(config=Gemma3TextConfig(...))`, sliding window 8,
alternating layer types, after 12 absolute tokens:

| layer | `is_sliding` | `get_seq_length()` | stored tensor | `get_mask_sizes(1)` |
|---|---|---|---|---|
| 0, 2 | True | **12** | 7 | `kv_length=8, kv_offset=5` |
| 1, 3 | False | **12** | 12 | `kv_length=13, kv_offset=0` |

**`get_seq_length()` is the absolute position, not the window-clipped length**, and it is uniform
across both layer types. On sliding layers it returns `cumulative_length`, incremented on every
update and never clipped (`cache_utils.py:274`). The clipping applies to the stored tensors only.

**So the per-block interface the plan wants already exists.** `cache.layers` is the per-block
list; `layer.get_seq_length()` is the monotone absolute offset with identical semantics on both
types; `layer.is_sliding` distinguishes them; `layer.get_mask_sizes(q)` returns
`(kv_length, kv_offset)` where `kv_offset` is the window start in absolute coordinates.

**The trap: never derive the offset from `layer.keys.shape[-2]`.** It reads 7 where the absolute
position is 12, and it is wrong on **every sliding layer**, which in real Gemma 3 is five of every
six.

**`crop()` on a sliding layer is constrained, measured:**
- Below the window: behaves like a full layer.
- At or past the window: `crop(-2)` **and** `crop(+2)` both raise `RuntimeError` unless
  `activate_past_recording()` was called **before** the window filled.
- With recording active: `crop(-3)` works, absolute goes 12 → 9, stored stays clipped at 7. A
  positive argument still raises; the sign convention is inverted relative to the full-attention
  layer, whose positive form is merely deprecated (removal in 5.18).

**Verdict: works with this change** — read offsets from `get_seq_length()` and `get_mask_sizes()`,
never from tensor shapes; and if the design rewinds the cache, call `activate_past_recording()` on
every sliding layer at construction, before the first forward, and pass negative arguments.

---

## 4. Full fine-tuning Gemma 3 4B and 27B under `accelerate` FSDP

**No published measured per-device peak exists** for Gemma 3 full fine-tuning under
accelerate+FSDP at these lengths. Google's own
[A4/GKE recipe](https://docs.cloud.google.com/ai-hypercomputer/docs/tutorials/gpu/gemma3-finetune-a4-gke-cluster)
is 12B **LoRA** at sequence 512 and reports no memory figure; Axolotl's Gemma 3 examples are
QLoRA. Everything below is arithmetic, and should be treated as such.

Parameters ([tech report Table 1](https://arxiv.org/html/2503.19786v1)): 4B = 4.30B, 27B = 27.43B.
AdamW under `mixed_precision: bf16` is 16 bytes/param sharded: 4B → 68.8 GB total, 8.6 GB per
device at 8 ranks. 27B → **438.9 GB total, 54.9 GB per device at 8 ranks**. Add gathered
transients: the 27B root unit holds `embed_tokens` at 262,208 × 5,376 = 1.41B params = 2.82 GB
gathered in bf16.

**Three Gemma-3-specific hazards, all verified in the installed source.**

1. **The `1 + weight` RMSNorm is a live trap.** `self.weight = nn.Parameter(torch.zeros(dim))` and
   `output * (1.0 + self.weight.float())` (`modeling_gemma3.py:140,149`). Weights are stored
   **zero-centred**. Two failure modes: a Llama-style fused norm kernel assuming `offset=0`
   silently zeroes activations, and FSDP meta-device init re-initialising a norm to **ones** gives
   scale 2. Use `fsdp_cpu_ram_efficient_loading` with rank-0 load and broadcast; do not pass a
   `param_init_fn`.
2. **The embedding scale is dtype-sensitive by design**, and shares that root.
   `super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)`
   (`modeling_gemma3.py:117`); the source comment records that `sqrt(3072) = 55.4256` becomes
   55.5 in bf16, and this is faithful to Google. Note `embed_scale` is an `nn.Buffer(...,
   persistent=False)` restored by `_init_weights` (`:460`), so it depends on the same
   initialisation path as hazard 1.
3. **Logits are the dominant activation and nothing chunks them by default.**
   `ForCausalLMLoss` does `logits = logits.float()` **unconditionally**
   (`loss/loss_utils.py:59`). At batch 1, sequence 4,096, vocab 262,208 that is 2.15 GB bf16 plus
   a 4.30 GB fp32 copy plus softmax buffer plus fp32 grad, about **15 GB**, before any other
   activation. Cross-checked against the
   [Cut Cross-Entropy paper](https://arxiv.org/abs/2411.09009), which measures 24 GB of logits for
   Gemma 2 2B at 8,192 tokens on a 256k vocabulary. Opt into fused linear cross-entropy
   (`use_liger_kernel=True` with `fused_linear_cross_entropy`) or CCE.

**Tied embeddings are not a hazard here:** Gemma 3 checkpoints carry no `lm_head.weight`, so the
FSDP2 `KeyError` ([accelerate #3870](https://github.com/huggingface/accelerate/issues/3870)) does
not trigger.

**Do not develop on world-size-1 FSDP.** It saves no memory, 16 bytes/param on one card is 68.8 GB
for the 4B alone, adds all-gather and reshard copies, and
[pytorch #144045](https://github.com/pytorch/pytorch/issues/144045) reports FSDP2 at world size 1
silently zeroing gradients for shape-`(1,)` parameters in non-root units. Its only value is
exercising the wrapping path.

Use **FSDP2**. FSDP1 is deprecated from PyTorch 2.11; accelerate still defaults to version 1 for
compatibility and ships `accelerate to-fsdp2`. `Trainer` is the documented path and owns
activation checkpointing and gradient-accumulation token normalisation; a custom loop must
reimplement `num_items_in_batch` and must prepare model and optimizer together under FSDP2.

**Verdict: works with this change** — fused or chunked linear cross-entropy, `fsdp_version: 2`,
rank-0 loading rather than meta-device init, and no world-size-1 FSDP for development.

---

## 5. The reference repository since `581d398`

**No change.** `anthropics/jacobian-lens` is public with exactly one commit,
`581d398613e5602a5af361e1c34d3a92ea82ba8e`, 2 Jul 2026, and the README states it is not
maintained and not accepting contributions. No releases. Seven open pull requests, three closed,
**none merged**. Five open issues, none closed, and none mentioning transcript length, memory,
out-of-memory, batching or fitting time. Our clone is at HEAD. There is no upstream to track.

**The finding in this section is a default, not a change.** Verified in the clone:
`fit(..., max_seq_len: int = 128, skip_first: int = 16)`, and
`input_ids = model.encode(prompt, max_length=max_seq_len)` (`jlens/fitting.py:107,143`). The
estimator **truncates every prompt to 128 tokens**, then excludes the leading 16 positions as
attention-sink dominated and the final position as having no target, leaving at most **111
contributing positions per prompt**. The published lenses were fitted this way at
`n_prompts=1000`.

Our acceptance trajectories are 2,000–2,700 tokens. Any part of the plan that assumes fitting
runs over whole transcripts is describing a different estimator from the reference one, and a
comparison against the published lenses would not be like for like. This is the single most
consequential fact in this section and it is a default nobody had read.

**A position selector exists only at readout**, `JacobianLens.apply(..., positions=...)`
(`jlens/lens.py:152`). At fit time the policy is fixed by `valid_position_mask`, with `skip_first`
the only knob. **`merge` is unchanged**, an `n_prompts`-weighted mean (`jlens/lens.py:106`).

**Neuronpedia's `run-all-fit-lens.py` is not public**, so no diff against `jlens/fitting.py` is
possible. `utils/run-all-fit-lens.py` in the platform repository is a 404; the Hugging Face
`neuronpedia/jacobian-lens` repository is weights only. Publicly recorded: `n_prompts=1000`, a
wikitext corpus path, and a discussion reporting the qwen3-32b artefact is a pre-finalisation
checkpoint that `load()` rejects, which motivates unmerged PR #3.

**Verdict: does not work, do this instead** — for the Neuronpedia comparison, which cannot be
made; the repository question is settled as no change, and the plan should state the 128-token fit
window explicitly rather than inherit it.

---

## What I would change in the plan, in order

1. **Split the acceptance test** (§2). It is the only item that invalidates a written test rather
   than adjusting one.
2. **Set `attn_implementation="eager"`** on both the Jacobian path and the replay path. One flag,
   two independent reasons, and under the default the Jacobian change is a regression.
3. **State the 128-token fit window** (§5), and decide deliberately whether to keep it. If the
   programme wants transcript-length fitting, that is a departure from the reference and needs its
   own justification and its own memory model.
4. **Restate §6.3's memory comparison** against what upstream actually does.
5. **Opt into fused cross-entropy before the first training run** (§4), not after the first
   out-of-memory.

Nothing here is verified on CUDA. Everything marked measured was measured on CPU with
torch 2.14.0 and transformers 5.16.1 on this machine; the scripts are in the session scratchpad
and can be moved into the tree if you want them as tests.

— Research Division
