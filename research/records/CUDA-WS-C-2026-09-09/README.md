# WS-C: what the port must not break, measured before any of it was written

**Engineer B, 2026-09-09.** Branch `cuda-ws-c` off `cuda-migration` at `9d68c27`. Everything below
ran on CPU torch on this box. Nothing here has run on CUDA, on a GPU, or against the real 4B
checkpoint, and every claim about those is marked unexecuted.

## The environment the order said did not exist

The order and plan §11.3 are both written on the premise that `import torch` fails here. It does
not, as of this morning:

| package | version | state |
|---|---|---|
| torch | 2.14.0 | installed, declared in `pyproject.toml`; MPS available, CUDA absent |
| transformers | 5.16.1 | installed; now declared |
| accelerate | 1.14.0 | **was absent**; installed here, supports `fsdp_version` and `use_orig_params` |

`accelerate` was the live blocker rather than torch: FSDP is the whole of this stream's first
deliverable and nothing named it as a dependency. The box was free at the time of the change
(no `outputs/.model-run.lock`, no `outputs/.box-window.json`, no model process).

## Gemma 3 is not Qwen 2, in two ways that a working port would hide

`depth_expansion.make_identity_block` is written against `mlx_lm.models.qwen2.TransformerBlock`.
Gemma 3's block differs in two respects, and **neither difference raises an error** — the wrong
model trains happily and reports a plausible loss.

**1. Four norms, not two.** Qwen 2 carries `input_layernorm` and `post_attention_layernorm`.
`Gemma3DecoderLayer` carries those plus `pre_feedforward_layernorm` and
`post_feedforward_layernorm`. A block copied field-by-field from the Qwen recipe inherits two
default-initialised norms.

**2. Attention type is indexed by position.** `Gemma3Attention.__init__` reads
`config.layer_types[layer_idx]`, and `Gemma3TextModel.forward` selects each layer's mask and rotary
embedding by `self.config.layer_types[i]`. On the 4B checkpoint global attention sits at layers
**5, 11, 17, 23, 29** and every other layer attends within a 1,024-token window. **Inserting a block
shifts every later layer's index**, so a splice into `model.layers` that does not repair
`config.layer_types` silently moves which layers see the whole context — the exact distinction
§6/WS-D's window-conditioned comparison is about.

## The mechanism survives both, and the identity is exact

`gemma3_identity_expansion.py` inserts one copied block, zeroes `self_attn.o_proj` and
`mlp.down_proj`, gives the copy the attention type of the block it copies, repairs `layer_types`,
`num_hidden_layers` and every `layer_idx`, and compares logits against the unexpanded model. Norm
weights are randomised first: Gemma's RMSNorm scales by `(1 + weight)` and initialises `weight` to
zero, so at default initialisation every norm is the identity and the test cannot tell a copied
norm from a missing one.

Across **all twelve insertion points** on a tiny randomly-initialised Gemma 3:

| quantity | value |
|---|---|
| worst `max abs delta logits` | **0.0** |
| exact identity at every insertion point | **yes** |

So the scientific property the mechanism exists to have — the expanded model is function-identical
to its source at initialisation, and stage 1 warms the zeroed outputs away from it — **transfers to
Gemma 3 unchanged**, provided the two differences above are handled. That is the load-bearing check
for deliverable 2 and it is now green rather than assumed.

**A consequence worth stating before a spec is written.** The copy inherits its source's attention
type, so inserting after a *global* layer produces two adjacent global layers, changing the
global-to-sliding ratio:

| insert after | source type | globals before | globals after |
|---:|---|---|---|
| 4 | sliding | `[5, 11]` | `[6, 12]` |
| **5** | **full** | `[5, 11]` | **`[5, 6, 12]`** |
| 6 | sliding | `[5, 11]` | `[5, 12]` |

`ExpansionSpec.evenly_spaced(34, 4)` yields `(7, 16, 24, 33)`, all of which are sliding layers on
Gemma 3 4B, so the default recipe does not hit this. It is recorded because the hazard is a property
of the spec and not of the code, and the next spec someone writes may hit it.

## Two more facts the port turns on

**The KV cache, not the training loop, is what a bad splice breaks first.** The identity check fails
with `use_cache=True` and passes with it off, because a sliding-window cache and the mask disagree
on key length once indices shift. Training never uses the cache, so WS-C is unaffected and **WS-B
is**: an expanded model that trains correctly can still generate incorrectly. Flagged across the
seam rather than fixed here.

**`StableAdamW` is almost entirely upstream.** Its `apply_single` computes
`p - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)` in float32 and casts back to the parameter dtype.
That is exactly `torch.optim.AdamW`'s decoupled update, which applies `p *= (1 - lr * wd)` before the
Adam step — the same quantity. What is genuinely ours is narrow: float32 moments and master weights
under bf16 parameters, and a **global** gradient-norm clip. Under FSDP the clip must be
`accelerator.clip_grad_norm_`; `torch.nn.utils.clip_grad_norm_` sees only the local shard and
computes the wrong norm, silently, on more than one device.

## Unexecuted

Everything about memory, sharding, multi-device loss agreement, the real checkpoint, and the arm-1
re-measurement. No number in this record was taken on a GPU.
