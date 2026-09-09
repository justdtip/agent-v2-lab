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

---

# The CPU smoke train passes, and four things it changed on the way

**Later the same day.** The order's golden gate — forty rows through the loop, the cadence, the
checkpoint and the manifest — runs on CPU against a tiny Gemma 3. Four optimizer steps, two
checkpoints at `save_steps=2`, a loss logged every step, no NaN, and an unstamped dataset directory
refused before the run starts. Every number below was taken on this laptop. **Nothing has run on a
GPU, nothing has been sharded, and no real checkpoint has been loaded.**

## float32 moments do not come from upstream, and my first draft said they did

`StableAdamW.init_single` allocates `m` and `v` as `mx.float32` **regardless of the parameter's
dtype**. `torch.optim.AdamW` allocates `exp_avg` and `exp_avg_sq` in the parameter's dtype. So
bfloat16 parameters get a bfloat16 second moment, nothing warns, the loss still falls, and
gradients below bfloat16's eight mantissa bits are rounded away. I wrote that this came for free
from upstream; a test disagreed.

The fix is not a custom optimizer. It is float32 master weights with bfloat16 compute, which is
what the Research Division's sixteen bytes per parameter describes: four for the parameter, four
for its gradient, eight for the two moments. `build_adamw` therefore **refuses** a bfloat16
trainable parameter rather than accommodating it, and only the trained slice is upcast — a frozen
trunk stays bfloat16 at two bytes.

The cost is recorded rather than discovered: a mixed-dtype stack is valid only under autocast, so
a runner that drops `bf16=True` gets a dtype error rather than a slow run. That is the better of
the two failures and it has its own test.

## The logits are the largest allocation, and they are avoidable

| what | bytes at the 2,688-token cap |
|---|---|
| logits, bfloat16 | 1.41 GB |
| logits upcast to float32 by cross-entropy | **2.82 GB** |
| both live | 4.23 GB |
| one 512-position chunk, float32 | 0.54 GB |

`chunked_ce` applies the head and the loss a slice at a time under checkpointing. A single chunk is
**bit-for-bit** the unchunked loss and gradient; many chunks agree to what reordering a float32 sum
costs, measured here at 2e-6 relative on one weight element in 1,552. The shift is checked against
`Gemma3ForCausalLM`'s own reported loss rather than a hand-written one.

## mlx-lm trains the model to emit token id 0, once per row

Its loss mask is `(steps >= offset) & (steps <= true_length)`. Target position `j` reads
`batch[j]`, real tokens occupy `0..length-1`, and padding to `1 + 32*ceil(L/32)` guarantees a pad
column at `j == length`. So **every untruncated row supervises exactly one pad token** — id 0,
directly after the final end-of-turn. An off-by-one meeting a padding rule, not a design choice.

Reproducible behind `supervise_one_pad`, off by default. Separately, mlx-lm passes **no attention
mask** and lets pads be attended over; on Gemma, whose sliding-window masks are built from that
mask, that is not safe, so the collator builds one.

## `iters` is micro-batches, and no code in the tree reads the key that says so

Arm 1's `iters: 1200` at accumulation 4 is **300 optimizer steps**. `steps_per_eval: 400` is 100
and `save_every: 400` is 100. The configs carry `iters_unit: batches` to say this and nothing reads
it. Handed to `transformers` unconverted, an arm trains four times as long and evaluates four times
as often, and finishes without complaint. Evaluation and save cadence must convert exactly, because
`ckpt 800` names a checkpoint; reporting cadence is rounded and the rounding is recorded.

## Still open, and not mine to settle

- **The arm-1 re-measurement has four variables moving at once**: Qwen3.5-4B against Gemma 3 4B,
  32 layers against 34 (so "top 8" is 8/32 against 8/34), 4-bit against bf16, LoRA against full
  fine-tuning. It cannot be a reproduction. The order's own escape clause is the operative one.
- **The source records disagree on which checkpoint won.** The arm-1 record selects 800 on
  validation loss; the quality record's full split scores 1,200 at 175/180 against 800's 159/180 at
  p = 6.1e-15 and says the one-line answer is wrong. **The selection criterion must be
  pre-registered before the re-measurement**, or it inherits the ambiguity.
- **`hf_checkpoint` does not exist.** No `.py` or `.yaml` in the tree contains the string; the
  field is `hf_id`. Adding it, and `backend:`, is WS-E's. `tests/test_pipeline.py` also pins the
  registry to exactly five stems, so a sixth breaks it in the same commit.
- **`delta.json` has no consumer.** `adapter_geometry.py` reads the safetensors directly and writes
  no file. Byte-compatibility is a self-imposed constraint; the thing to reproduce is that script's
  per-layer quadrature, which sums numerator and denominator separately — averaging per-module
  ratios inflates by roughly the square root of the module count, an error its own README records.
- **Deleting the gated-delta modules reaches five places outside the deletion list**, including two
  literal assertions in `test_repository_rules.py` and the MLX training path ruling 5 keeps.

---

## Which tree the tests actually exercised

The shared venv installs `local_llm_lab` from the main checkout, so from this worktree
`import local_llm_lab` resolves to `/Users/daniel.tipton/Desktop/An app/src/local_llm_lab` unless
`PYTHONPATH` says otherwise. A run like that reports on code nobody edited: a **new** module fails
to import, which is loud, but a **changed** one silently resolves to main's version and passes.

Every run recorded here set `PYTHONPATH` to this worktree's `src`, and the imports resolved to:

    /Users/daniel.tipton/worktrees/cuda-ws-c/src/local_llm_lab/__init__.py

`tests/test_import_tree.py` now asserts it, so the next run cannot be wrong about this quietly:
it passes with the variable set and fails with the path it actually reached and the export that
fixes it. A guard rather than a line in a record, because the failure it catches is green tests in
the wrong tree and nothing else in the suite would notice.

**103 tests pass** at this commit, including the whole of `tests/test_repository_rules.py`.

---

# Finding: every untruncated mlx-lm row supervises one padding token, and in Qwen that token is `!`

**Recorded as a finding at the Chief's direction, because it is a fact about every adapter this
programme has trained, not about the port.** It was found while writing the torch collator and it
is not a property of the port; the port merely had to decide whether to reproduce it.

## The mechanism: two short rules that never mention each other

`mlx_lm/tuner/trainer.py` builds a batch at lines 156-167 and masks the loss at lines 86-99.

    batch:  pad_to = 32
            width = min(1 + 32 * ceil(max(lengths) / 32), max_seq_length)
            batch_arr = np.zeros((rows, width), np.int32)      # the pad value is literally 0

    loss:   targets = batch[:, 1:]
            steps   = arange(1, targets.shape[1] + 1)
            mask    = (steps >= offset) & (steps <= true_length)

Target index `k` carries `steps[k] = k + 1` and therefore reads `batch[k + 1]`. The mask admits
`k + 1 <= true_length`, so **the last supervised target is `batch[true_length]`**. Real tokens
occupy `0 .. true_length - 1`. The `1 +` in the width guarantees `batch[true_length]` exists, and
it is padding.

Neither rule is wrong on its own. The padding rule guarantees a column; the mask rule supervises
it. Nothing in either file refers to the other.

## Verified, not inferred

`mlx_pad_supervision.py` transcribes both rules into numpy and fills real tokens with `1..length`
so that a zero in a supervised slot can only be padding. At a 4,096 cap:

| case | true length | padded width | supervised | of which padding |
|---|---:|---:|---:|---:|
| typical row | 100 | 129 | 91 | **1** |
| length an exact multiple of 32 | 32 | 33 | 23 | **1** |
| length one over a multiple | 33 | 65 | 24 | **1** |
| row truncated at the cap | 5,000 | 4,096 | 4,086 | **0** |

The boundary case matters: at `L = 32` the width is 33, so the `1 +` is doing the work alone and
the artefact still occurs. **A truncated row is the only one that escapes**, because the cap
removes the pad column.

## What the supervised token actually is

Token id 0 is not a special token in every vocabulary, and this programme's runs are the case where
it is not:

| tokenizer | id 0 |
|---|---|
| Qwen3.5-4B, which every existing training record used | **`!`** |
| Gemma 3 4B, the migration target | `<pad>` |

So every MLX-trained adapter this programme has produced was taught, once per untruncated row, to
emit **`!`** immediately after the row's final real token. The final real token of a rendered row is
its end-of-turn marker, which places the artefact **exactly at the stopping decision**.

Scale on arm 1: 1,200 rows at batch size 1, one pad target each, against roughly 1.32 million
supervised tokens — about **0.09%** of the supervised signal. Small in mass, but it is 100% of rows,
always the same target, and always the same structural position. It is not noise spread thinly; it
is a consistent instruction at one point.

**What it does not establish.** Nothing here measures an effect on behaviour. Whether 0.09% at the
stopping position changed any trained model's stopping is unmeasured, and this record does not
claim it did.

## The second half: padding is attended over

`default_loss` calls `model(inputs)` with no mask (line 90), so padded positions are attended to by
the real tokens and only their *logits* are discarded. That is survivable on a dense causal model
and is not safe on Gemma 3, whose sliding-window masks are built from the attention mask.

## The decision taken, on the Chief's ruling

For the arm-1 reproduction under full fine-tuning: **`supervise_one_pad` stays off and the collator
builds the attention mask**, with both carried as departures in the run manifest. The reproduction
already changes the model, the precision and the parameterisation, so matching a supervision bug
would add nothing but the bug. The flag is kept for a within-MLX-recipe check, where reproducing it
is the point.

---

# The two-device agreement gate runs on CPU, and a loss curve alone would have passed a broken one

**The order's golden test, on CPU, before the remote exists.** FSDP2 at world size one is not a
valid rung — it silently zeros gradients for some parameter shapes in non-root units
(pytorch #144045) — so this compares **plain single-process training against two-process FSDP2**
under `gloo`, which is the comparison the single-device path actually makes.

## FSDP2 does not refuse a CPU mesh

torch 2.14.0, `init_device_mesh("cpu", (2,))`, `fully_shard` on each block, `torchrun
--nproc_per_node=2`: **it works.** `DeviceMesh((2,), 'cpu', stride=(1,))`, forward, backward and
optimizer step all complete. So the whole gate is answerable on this laptop and none of the first
hour on rented hardware needs to be spent finding out.

## Sharding the root unit fails, and it fails on this stream's own loss design

`fully_shard(model)` on the root raises, at `Gemma3TextModel.forward`:

    RuntimeError: aten.embedding.default got mixed torch.Tensor and DTensor, need to convert
    all torch.Tensor to DTensor before calling distributed operators!
    (transformers/models/gemma3/modeling_gemma3.py:117)

The cause is ours, not upstream's. `causal_lm_chunked_loss` calls `model.model` directly and reaches
for `lm_head.weight`, both to avoid materialising a 262,208-wide logit tensor. Both bypass the
pre-forward hooks FSDP2 uses to unshard a unit, so the embedding is still a `DTensor` when
`F.embedding` meets a plain `input_ids`. **Chunked loss wants raw parameters; FSDP2 wants every
access to go through a module call.** The two designs trade directly against each other.

## Leaving the root replicated is the workaround, and it is silently wrong

**FSDP2 communicates gradients only for parameters inside a unit.** Shard the blocks and leave the
embedding, final norm and tied head outside, and nothing reduces their gradients: each rank keeps
only its own half of the accumulation window, forever.

Step 0, before any update, on identical weights:

| quantity | one device | two devices | relative |
|---|---:|---:|---:|
| loss | 4.1746088012 | 4.1746088012 | **0.00e+00** |
| gradient norm, sharded blocks | 1.89720254 | 1.89720254 | 2.17e-10 |
| gradient norm, **root unit** | 0.93408004 | 1.31364770 | **4.06e-01** |

**The loss was bit-identical while the gradient was forty per cent wrong.** The order's gate as
written — "one device and two devices produce the same loss curve to tolerance" — compares the
column that agreed. It would have passed this configuration, on the machine it was meant to
protect, and the damage would have appeared as a training run that quietly optimised something
else. The gate therefore records **gradients as well as losses**, and the loss-only form should be
treated as insufficient wherever else it appears.

## With the root reduced by hand, the arms agree to float32

An explicit all-reduce of the root unit's gradients, **mean and not sum**, to match the reduction
FSDP applies to the sharded parameters against the same `× world` loss scaling:

| quantity | max relative deviation over 6 steps |
|---|---:|
| loss | **5.39e-08** |
| gradient norm, sharded blocks | 3.79e-07 |
| gradient norm, root unit | 1.39e-07 |

Against 8.83e-04 on the loss before the fix: four orders of magnitude, and what remains sits at
float32 epsilon, which is where reassociated summation lives. Step 0 is exact and the token count
per step is identical in both arms, so the two are optimising the same objective on the same rows.

**The design that makes them comparable**, since it is the part that is easy to get wrong: the
global window is `accum` micro-batches either way; at world size `W` each rank takes `accum / W` of
them interleaved by rank; each micro-batch contributes a **summed** token loss divided by the
window's **global** token count, all-reduced; and each rank's loss is scaled by `W` because FSDP2
reduce-scatters with a mean. Drop that scaling and the two-device gradient is `1/W` of the
one-device gradient — a divergence that looks like a sharding bug and is arithmetic. Gradient
clipping is off throughout, so any deviation is attributable to sharding alone.

**The proper fix is not this one.** Reducing the root by hand works and is what the gate does;
putting the root inside an FSDP unit would need the loss to call `lm_head` as a module, unsharding
and resharding the model's largest parameter once per chunk. That is a real cost against a real
saving and it is the Chief's call, not one to take silently in a gate.

## Unexecuted

CPU and `gloo` only. Nothing here has run on CUDA or NCCL, on more than two processes, on a real
checkpoint, or at any size where the memory arithmetic bites. The model's shape is read from
`tiny_gemma3.json` so that nothing in the script names a layer count or a width.

---

## Proposed test for the padding finding, recorded so the claim and its test travel together

**Not run, and not mine to run.** This is an MLX-box measurement the Chief is putting to the
Director and the D-CRO as a candidate probe. It is recorded here because a mechanism without a test
beside it is the shape of claim this programme has agreed to stop making.

**Why it is worth minutes of the box.** The supervised slot is the position immediately after the
final end-of-turn token. Generation normally stops there, so nothing else constrains the model's
distribution at that position — which makes it both the place a trace would survive and the place
nobody would look. And `!` is an ordinary emittable token rather than an inert pad, so if a stop
rule ever fails and generation runs past end-of-turn, the first token sampled is drawn from exactly
the distribution this supervision shaped.

**The measurement.** `P(id 0 | context ending in end-of-turn)` on **held-out** rows, read at the
model's own output distribution — the final layer, which is exact — and optionally through the lens
profile for depth.

**Three arms and two controls, because the primary contrast alone cannot separate the accounts.**

| arm | what it tests |
|---|---|
| base against arm 1's checkpoint | the primary contrast: did the supervision leave a trace at all |
| arm 1 at 400, 800 and 1,200 | **dose-response**: exposure rises with steps, so a trace driven by this supervision should rise monotonically and a distributional drift need not |
| `P(!)` at *other* positions in the same rows | **position control**: separates "the model now likes `!` generally" from "the model learned `!` at this one slot", which is the only version that is this finding |

The checkpoints all exist; no training is needed. One model load, held-out rows only.

**Declared in advance, since the direction is predictable and this programme counts that as a
conflict.** I expect a rise at the slot, because the supervision is 100% of untruncated rows at a
fixed position with no competing signal. If it is absent, that is the more informative result and
the more useful one: it closes the mechanism as unmeasurable at 0.09% of the supervised tokens, and
that number then bounds every similar artefact anyone finds later.

**What closes it either way.** A rise confined to the slot, monotonic in exposure, makes the
mechanism a finding. Flat at the slot and flat across checkpoints closes it. A rise everywhere in
the distribution is a different result about training drift and is not this finding.

**One limit to carry.** The lens read is a bonus, not the measurement: the end-of-turn position of
an agentic row sits far outside the lens's fitted position range, so a depth profile there inherits
the extrapolation this record's own map section could not bound. The final-layer read does not, and
it is sufficient to answer the question.

---

# The amended gate, and the design that lets the loss keep its raw parameters

**Both on the Chief's rulings, and the second changed the design rather than working around it.**

## The loss belongs inside the root unit's forward

The earlier failure — `aten.embedding.default got mixed torch.Tensor and DTensor` — was read here as
FSDP2 refusing the chunked loss's raw-parameter design. It is not. **A parameter is an unsharded
tensor with its backward hooks armed only inside the forward of the unit that owns it**, and the
loss was touching it outside that window. So the window moves, not the loss.

`wrap_with_chunked_loss` puts the model and the loss in one thin `nn.Module` whose `forward` takes
`input_ids` and `labels`. `fully_shard` each block as before, then `fully_shard` **the wrapper** as
the root unit — which holds the tied embedding, the final norm and the head together, as a tied
pair must be. Inside that forward the raw weight is the unsharded tensor and the reduce-scatter
hooks are registered, so the loss keeps reading `lm_head.weight` directly and FSDP2 owns every
parameter. It works: the root shards, and nothing about the loss changed.

The cost is the root's parameters resident unsharded from forward through backward — **once per
step, never once per chunk**, which is what calling the head as a module per chunk would have cost.
`reshard_after_forward=True` on the root is the memory rung when a device is short, to be measured
on the device rather than assumed.

## Per parameter, not per norm, and not the loss

The gate now compares **per-parameter gradients at step zero, before any update**, and per-parameter
values after N steps. A norm was not enough for the same reason a loss was not: a wrong slice hides
inside a right norm. Deviation is per tensor, `max|a − b|` over the tensor's own scale.

Three arms against single-process training, on the same rows, tiny Gemma 3, `gloo`, two processes:

| arm | loss, step 0 | **grad, step 0** | grad, root unit | value, final |
|---|---:|---:|---:|---:|
| blocks **and root** sharded | 0.00e+00 | **1.80e-07** | 8.13e-08 | 8.56e-06 |
| blocks only, root reduced by hand | 0.00e+00 | 1.80e-07 | 8.51e-08 | 8.58e-06 |
| **negative control**: root reduced by nothing | **0.00e+00** | **1.67e+00** | **1.67e+00** | **1.12e+00** |

**The control's loss is bit-identical to the correct run at step zero while its gradient is 167%
wrong and its parameters end 112% wrong.** That is not a near miss: it is the whole quantity. The
loss at step zero is identical **by construction** in every arm — same weights, same rows, same
objective — so it asserts nothing at all, and afterwards it is one step behind the error.

**The rule this is the evidence for: a gate compares the quantity nearest the mechanism, never one
downstream of it.** Two correct configurations sit at float32 epsilon; the broken one is off by more
than one. There is no tolerance at which the old gate distinguishes them and no tolerance at which
the new one fails to.

Both correct arms agree, so the hand all-reduce is not wrong — it is a **second reduction path**,
and a second path is a standing invitation to exactly this error the moment anyone adds a parameter
outside a unit. It is kept only as the control that reproduces what the old gate passed.

## Unexecuted

CPU and `gloo`, two processes, a tiny model. No CUDA, no NCCL, no more than two ranks, no real
checkpoint, and no size at which the memory arithmetic bites. `reshard_after_forward` on the root is
named as a rung and has not been measured.

---

## What is validated, and what is only validated

Worth separating, because the two are easy to conflate and the difference costs money on rented
hardware.

| piece | state |
|---|---|
| the sharded path, per parameter against single-process | **validated on CPU**, two gloo processes, tiny model |
| the `train` stage's torch branch | **runs**, single process only |
| the two joined together | **not done** |

`stage_train_torch` does **not** wire FSDP2. Handed more than one process, `Trainer` and
`accelerate` would distribute the run under their own default strategy, which is not the sharded
path the order requires and which nothing the run reports would reveal: the loss falls, a checkpoint
is written, and the memory arithmetic every device decision rests on describes a configuration that
never ran. So the stage **refuses** a multi-process run and names the gap, rather than proceeding.
That refusal is the cheapest possible version of discovering it, and the first hour on rented
hardware is meant to be exactly that kind of hour.
