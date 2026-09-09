"""Full-parameter fine-tuning on `transformers`, with the pieces that are ours and no others.

What upstream owns and this module does not reimplement: activation checkpointing, gradient
accumulation and its token-weighted normalisation (`num_items_in_batch`), global gradient clipping
under whatever distribution strategy is active (`Trainer` calls `accelerator.clip_grad_norm_`,
which is the only form that computes a correct norm when parameters are sharded), the AdamW update
itself, checkpoint writing, and the evaluation cadence.

What is ours, and why:

- **Per-layer freezing.** The full-fine-tuning analogue of mlx-lm's `lora_layers: N`, which adapts
  the top N blocks and leaves the embedding, the final norm and the head alone. This is the knob
  the programme's most consequential training finding was measured with, so its meaning has to
  match rather than approximate.
- **The loss.** `chunked_ce` instead of the model's own, because a 262,208-token vocabulary makes
  the logits the largest allocation in the step. The model is therefore called without labels and
  the head applied separately.
- **float32 optimizer state under bfloat16 parameters.** `StableAdamW`'s substance once clipping is
  handed back to `Trainer`.

**Single device is plain training, not FSDP with one rank.** FSDP2 at world size one silently zeros
gradients for some parameter shapes in non-root units (pytorch #144045), so the development path
and the sharded path are different code paths on purpose, and any claim that they agree is a
measurement rather than an assumption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - torch stays out of module scope
    import torch

from local_llm_lab.training.chunked_ce import chunked_linear_cross_entropy, shift_for_causal_lm

#: Default slice of the sequence the head is applied to at once. 512 positions of a 262,208-token
#: vocabulary is 0.54 GB of float32 logits, against 2.82 GB for a whole 2,688-token row.
DEFAULT_LOSS_CHUNK = 512


def base_model_of(model: Any) -> Any:
    """The causal LM beneath whatever is wrapping it.

    Three layers can sit above it and each names its child differently: DDP and FSDP1 expose
    ``.module``; :func:`wrap_with_chunked_loss` exposes ``.inner``. Unwrapping repeatedly rather
    than once means the order they are applied in does not matter, which it otherwise would --
    `Trainer` wraps what it is handed, and what it is handed is already our wrapper.
    """
    inner = model
    for _ in range(4):  # bounded: a cycle here should raise below, not spin
        nxt = getattr(inner, "module", None) or getattr(inner, "inner", None)
        if nxt is None or nxt is inner:
            break
        inner = nxt
    if not hasattr(inner, "model") or not hasattr(inner, "lm_head"):
        raise TypeError(
            f"{type(inner).__name__} is not a causal LM with `.model` and `.lm_head`; this loss "
            "applies the head itself and cannot find it"
        )
    return inner


def freeze_all_but_top_layers(model: Any, top_layers: int | None) -> int:
    """Open the last ``top_layers`` decoder blocks and nothing else; return the trainable count.

    ``None`` trains every parameter. This mirrors `lora_layers`: mlx-lm adapts the last N blocks
    and never the embedding, the final norm or the head, so neither does this. A value larger than
    the stack is refused rather than clamped -- `lora_layers: 40` on a 34-layer model is a config
    that means something the model cannot do.
    """
    inner = base_model_of(model)
    layers = inner.model.layers
    if top_layers is None:
        for parameter in inner.parameters():
            parameter.requires_grad_(True)
        return sum(p.numel() for p in inner.parameters())
    if top_layers < 0:
        raise ValueError(f"top_layers must be >= 0 or None; got {top_layers}")
    if top_layers > len(layers):
        raise ValueError(
            f"top_layers={top_layers} exceeds the model's {len(layers)} layers; a recipe that "
            "names more layers than the model has is a recipe for a different model"
        )
    for parameter in inner.parameters():
        parameter.requires_grad_(False)
    opened = 0
    for layer in list(layers)[len(layers) - top_layers :] if top_layers else []:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
            opened += parameter.numel()
    return opened


def upcast_trainable_to_float32(model: Any) -> int:
    """Put the trainable parameters in float32, leaving the frozen trunk in its loaded dtype.

    Returns the number of parameters upcast. Frozen weights have no optimizer state and no
    gradient, so they stay bfloat16 and cost 2 bytes each; only what is trained pays 16.
    """
    import torch

    upcast = 0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype is not torch.float32:
            parameter.data = parameter.data.to(torch.float32)
            upcast += parameter.numel()
    return upcast


def build_adamw(model: Any, *, learning_rate: float, weight_decay: float = 0.01, eps: float = 1e-8):
    """AdamW over the trainable parameters, refusing any whose optimizer state would be bfloat16.

    `torch.optim.AdamW`'s decoupled update is `p -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)`,
    which is exactly what MLX's `StableAdamW.apply_single` computes, so the update itself needs no
    subclass. What MLX's class also did, and what upstream does **not** do, is force the moments to
    float32: `AdamW` allocates `exp_avg` and `exp_avg_sq` in the parameter's own dtype, so bfloat16
    parameters silently get a bfloat16 second moment. Nothing raises, the loss falls, and the small
    gradients this schedule is made of are rounded away -- bfloat16 carries eight mantissa bits.

    The fix is not a custom optimizer but float32 master weights with bfloat16 *compute*, which is
    what the Research Division's sixteen bytes per parameter describes: 4 for the parameter, 4 for
    its gradient, 8 for the two moments. `upcast_trainable_to_float32` puts them there and
    `Trainer(bf16=True)` casts for the forward. So this refuses rather than accommodates: a
    bfloat16 trainable parameter here is a configuration error with a silent numerical consequence,
    which is exactly the kind this programme has agreed to stop discovering afterwards.

    `foreach=False` keeps peak memory to one parameter's temporaries rather than the whole group's.
    """
    import torch

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters; freezing opened nothing")
    reduced = sorted({str(p.dtype) for p in trainable if p.dtype is not torch.float32})
    if reduced:
        raise TypeError(
            f"trainable parameters in {reduced}: AdamW would allocate its moments in that dtype "
            "and round away the gradients this schedule is made of. Call "
            "upcast_trainable_to_float32(model) first and let Trainer(bf16=True) cast for compute."
        )
    return torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=eps,
        weight_decay=weight_decay,
        foreach=False,
    )


def causal_lm_chunked_loss(
    model: Any,
    inputs: dict[str, Any],
    *,
    chunk_size: int = DEFAULT_LOSS_CHUNK,
    num_items_in_batch: torch.Tensor | int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Cross-entropy for a causal LM without materialising the whole logit tensor.

    **This loss owns its own precision context, and has to.** It calls the decoder stack directly
    and applies the head itself, both to avoid materialising a 262,208-wide logit tensor -- and both
    reach around ``forward``, which is where `accelerate` installs autocast and where FSDP2 installs
    its unshard hooks. A mixed-dtype stack (bfloat16 trunk, float32 trainable slice) therefore fails
    with ``expected m1 and m2 to have the same dtype`` under a `Trainer` configured for bfloat16,
    because the wrapper that would have cast never runs. Passing ``autocast_dtype`` makes the cast
    explicit here rather than depending on a wrapper this function is designed to bypass.

    Returns a **sum** divided by ``num_items_in_batch`` when `Trainer` supplies one, which is how
    upstream weights each micro-batch by its supervised token count across an accumulation cycle.
    Without one it falls back to this micro-batch's own token count, which is the same quantity
    when accumulation is 1 and is the wrong one otherwise -- so the caller passing it is not
    optional in a real run, and `Trainer` always does.
    """
    import contextlib

    import torch

    inner = base_model_of(model)
    labels = inputs["labels"]
    context = (
        torch.autocast(device_type=inputs["input_ids"].device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )
    with context:
        hidden = inner.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
        ).last_hidden_state

        flat_hidden, flat_labels = shift_for_causal_lm(hidden, labels)
        total, count = chunked_linear_cross_entropy(
            flat_hidden,
            inner.lm_head.weight,
            flat_labels,
            chunk_size=chunk_size,
            bias=getattr(inner.lm_head, "bias", None),
        )
    denominator = count if num_items_in_batch is None else num_items_in_batch
    return total / denominator.clamp(min=1) if hasattr(denominator, "clamp") else total / max(
        int(denominator), 1
    )


class ChunkedLossModule:
    """Namespace marker; the class itself is built by :func:`wrap_with_chunked_loss`."""


def wrap_with_chunked_loss(
    model: Any,
    *,
    chunk_size: int = DEFAULT_LOSS_CHUNK,
    autocast_dtype: torch.dtype | None = None,
) -> Any:
    """Put the chunked loss *inside* a module, so FSDP2 can own every parameter it touches.

    The loss reads `lm_head.weight` and calls the decoder stack directly, to avoid materialising a
    262,208-wide logit tensor. Under FSDP2 a parameter is an unsharded tensor with its backward
    hooks armed only **inside the forward of the unit that owns it**; touched anywhere else it is
    still a `DTensor`, and `aten.embedding` meeting a plain `input_ids` is FSDP2 saying so rather
    than refusing the design.

    So the window is what has to move, not the loss. This wrapper's ``forward`` is a unit boundary:
    ``fully_shard`` each block as usual and ``fully_shard`` this wrapper as the root, which holds
    the tied embedding, the final norm and the head **together**. Under `tie_word_embeddings` the
    head is not a second tensor that happens to be equal: ``lm_head.weight is embed_tokens.weight``,
    one parameter with two names. That is what makes the root unit clean -- there is no flat
    parameter straddling two units, and no way to shard half of a tied pair. Inside
    the forward the raw weight is unsharded and the reduce-scatter hooks are registered, so the
    loss keeps its raw-parameter design and every parameter stays FSDP2's.

    The cost is the root's parameters resident unsharded from forward through backward -- **once per
    step, never once per chunk**, which is what calling `lm_head` as a module per chunk would have
    cost. `reshard_after_forward=True` on the root is the memory rung when a device is short, and
    it is measured on the device rather than assumed (plan 10.2).
    """
    import torch

    base_model_of(model)  # fail here, with a clear message, rather than inside a sharded forward

    class _ChunkedLossModule(torch.nn.Module):
        loss_chunk_size = chunk_size
        loss_autocast_dtype = autocast_dtype

        def __init__(self) -> None:
            super().__init__()
            self.inner = model
            # `Trainer` reads `config` off the model it is handed and calls the checkpointing
            # methods on it. Delegating keeps the wrapper the only object anything is handed --
            # which is the point of the ruling: autocast, the accumulation window and the unshard
            # hooks all attach to the one `forward` that now runs.
            self.config = model.config

        def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
            self.inner.gradient_checkpointing_enable(**kwargs)

        def gradient_checkpointing_disable(self) -> None:
            self.inner.gradient_checkpointing_disable()

        def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            num_items_in_batch: torch.Tensor | int | None = None,
        ) -> torch.Tensor:
            return causal_lm_chunked_loss(
                self.inner,
                {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask},
                chunk_size=self.loss_chunk_size,
                num_items_in_batch=num_items_in_batch,
                autocast_dtype=self.loss_autocast_dtype,
            )

    return _ChunkedLossModule()


def make_loss_module_trainer_class() -> Any:
    """A `Trainer` for a model whose ``forward`` returns the loss.

    A factory rather than a module-level class so that importing this module does not import
    `transformers`, which the suite's import-closure rule cares about.

    Two overrides, and both exist because the model handed to `Trainer` is the wrapper rather than
    the `PreTrainedModel` inside it:

    ``compute_loss`` calls that wrapper's forward and returns what it returns. Everything else stays
    upstream's: activation checkpointing, the accumulation window and its ``num_items_in_batch``,
    ``accelerator.clip_grad_norm_`` under whatever strategy is active, the schedule and the cadence.

    ``_save`` saves the **inner** model. `Trainer._save` checks
    ``isinstance(model, PreTrainedModel)`` and, failing that, writes a bare state dict -- which
    would name every tensor ``inner.*`` and emit no ``config.json``, producing a checkpoint that
    `checkpoint_delta` cannot name-match against its base and the registry cannot load. Nothing
    would raise; the run would finish and the artefact would be wrong.
    """
    from transformers import Trainer

    class LossModuleTrainer(Trainer):
        def compute_loss(
            self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
        ):
            loss = model(
                input_ids=inputs["input_ids"],
                labels=inputs["labels"],
                attention_mask=inputs.get("attention_mask"),
                num_items_in_batch=num_items_in_batch,
            )
            return (loss, None) if return_outputs else loss

        def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
            inner = getattr(self.accelerator.unwrap_model(self.model), "inner", None)
            if inner is None:
                super()._save(output_dir, state_dict)
                return
            target = output_dir if output_dir is not None else self.args.output_dir
            inner.save_pretrained(target)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(target)

    return LossModuleTrainer
