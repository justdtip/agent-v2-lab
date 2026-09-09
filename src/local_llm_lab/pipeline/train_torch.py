"""The `train` stage's torch branch: full-parameter fine-tuning behind ``device.backend()``.

The MLX branch in :mod:`local_llm_lab.pipeline.cli` is untouched and remains the laptop's path. This
module is reached only when :func:`local_llm_lab.device.backend` reports ``torch``, and it imports
torch and transformers **inside functions** so that selecting the MLX backend never pays for them --
the rest of the tree's forty-odd MLX imports are likewise left alone (plan 12.1).

What it reuses rather than rewrites: ``tuner_data.load_rendered_splits`` for the rows and the
dataset-manifest gate, ``training.torch_config`` for the recipe in optimizer steps, ``torch_full``
for the freezing, the optimizer and the chunked loss, ``collator`` for padding and masking, and
``transformers.Trainer`` for activation checkpointing, the accumulation window, ``num_items_in_batch``
and ``accelerator.clip_grad_norm_``.

``device.pin`` is called **before the model loads**, because ``CUBLAS_WORKSPACE_CONFIG`` is read by
cuBLAS at first use and a pin after initialisation is not a pin. Its return -- a reading of what is
actually set, not a copy of what was asked for -- is written into the run manifest as the
determinism block, beside the list of every way this run departs from the MLX record of the same
config.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from local_llm_lab.project import PROJECT_ROOT
from local_llm_lab.training.torch_config import FullFinetuneConfig, load_full_finetune_config

#: Written beside the adapters, so a reader of the run has the determinism and the departures in one
#: place rather than reconstructing them from the log.
MANIFEST_NAME = "train_manifest.json"

#: Eager attention: the golden tests compare against records made without a fused kernel, and a
#: kernel choice is a numerical difference this stream has no reason to introduce silently.
ATTENTION = "eager"


def require_supported_distribution(world_size: int) -> None:
    """Refuse a multi-process run rather than silently training under the wrong strategy.

    FSDP2 **is** now wired: :func:`fsdp_arguments` builds the configuration and the stage passes it.
    What does not yet exist is the number. The standalone sharded path was checked per parameter
    against single-process training in ``two_device_agreement.py``; the *joined* path -- that
    configuration driven through `Trainer` by this stage -- has not been through the same gate.

    Those are two claims and this refusal keeps them two. It comes off when the joined path passes
    the per-parameter gate the standalone path passed, and not before, because the alternative is a
    stage that appears to work under a strategy nobody measured: the loss falls, the checkpoint is
    written, and the memory arithmetic every device decision rests on describes a configuration
    that never ran.
    """
    if world_size > 1:
        raise NotImplementedError(
            f"world size {world_size}: FSDP2 is configured by fsdp_arguments() but the joined path "
            "-- this stage driving that configuration through Trainer -- has not passed the "
            "per-parameter agreement gate that the standalone path passed in "
            "research/records/CUDA-WS-C-2026-09-09. Run single-process until that number exists."
        )


def rendered_rows(dataset: Any) -> list[dict[str, Any]]:
    """Materialise a ``RenderedRowsDataset`` into the pairs the collator takes.

    ``process`` returns mlx-lm's ``(tokens, offset)``, which is the same pair this path needs; going
    through it rather than reading the attributes keeps one definition of what a rendered row is.
    """
    rows = []
    for index in range(len(dataset)):
        tokens, offset = dataset.process(dataset[index])
        rows.append({"input_ids": list(tokens), "prompt_length": int(offset)})
    return rows


def decoder_layer_class_name(model: Any) -> str:
    """The block class FSDP2 wraps, read off the model rather than written here.

    Naming ``Gemma3DecoderLayer`` in source would make this file wrong for the next architecture
    and, worse, wrong silently: an auto-wrap policy that matches nothing shards nothing and trains
    without complaint. Plan 10.1's rule about dimensions applies to model names for the same
    reason.
    """
    from local_llm_lab.training.torch_full import base_model_of

    layers = base_model_of(model).model.layers
    if not len(layers):
        raise ValueError("model has no decoder layers to wrap")
    return type(layers[0]).__name__


def fsdp_arguments(model: Any, *, world_size: int) -> dict[str, Any]:
    """`TrainingArguments` fields for FSDP2, or nothing at world size one.

    **World size one is not a rung.** FSDP2 at one rank silently zeros gradients for some parameter
    shapes in non-root units (pytorch #144045), so the single-device path is plain training and the
    sharded path begins at two. Any claim that they agree is therefore a measurement between two
    different code paths, and the record says so rather than implying one path.

    `version: 2` is `transformers` 5.16's own default and is passed explicitly, because a default
    that changes underneath a recorded run is a difference nothing would report.
    """
    if world_size < 2:
        return {}
    return {
        "fsdp": "full_shard",
        "fsdp_config": {
            "version": 2,
            "transformer_layer_cls_to_wrap": [decoder_layer_class_name(model)],
            # The memory rung when a device is short; measured on the device, never assumed.
            "reshard_after_forward": True,
            # A checkpoint must be loadable by name, so it is gathered rather than sharded on save.
            "state_dict_type": "FULL_STATE_DICT",
        },
    }


def build_training_arguments(
    recipe: FullFinetuneConfig,
    *,
    output: Path,
    seed: int,
    selected: str,
    fsdp: dict[str, Any] | None = None,
) -> Any:
    """`Trainer`'s arguments, with the device forced to the one the shim chose.

    `Trainer` picks an accelerator itself and on this laptop picks MPS, so a run whose manifest says
    ``"device": "cpu"`` -- because `device.select()` said so -- would train somewhere else and record
    a provenance that is not true. `use_cpu` is therefore derived from the shim rather than left to
    upstream's detection, and the device the trainer ends up on is read back afterwards.
    """
    from transformers import TrainingArguments

    return TrainingArguments(
        **(fsdp or {}),
        use_cpu=selected == "cpu",
        output_dir=str(output / "checkpoints"),
        max_steps=recipe.max_steps,
        per_device_train_batch_size=recipe.per_device_micro_batch,
        gradient_accumulation_steps=recipe.gradient_accumulation_steps,
        learning_rate=recipe.learning_rate,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        logging_steps=recipe.logging_steps,
        save_steps=recipe.save_steps,
        save_strategy="steps",
        eval_strategy="steps" if recipe.val_batches else "no",
        eval_steps=recipe.eval_steps or None,
        report_to=[],
        seed=seed,
        bf16=True,
        gradient_checkpointing=True,
        # Checkpointing stays on for the whole run. mlx-lm installs it by rebinding the block
        # class, so its stage-two `grad_checkpoint=False` did not turn it off; porting that False
        # literally would run the long stage uncheckpointed.
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def stage_train_torch(
    config: dict[str, Any], iters: int | None = None, resume_from: Path | None = None
) -> dict[str, Any]:
    """Run the torch full-fine-tuning branch and return the manifest it wrote."""
    from local_llm_lab import device
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.training.collator import CausalCollator
    from local_llm_lab.training.torch_full import (
        build_adamw,
        freeze_all_but_top_layers,
        make_loss_module_trainer_class,
        upcast_trainable_to_float32,
        wrap_with_chunked_loss,
    )
    from local_llm_lab.tuner_data import load_rendered_splits

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    require_supported_distribution(world_size)
    seed = int(config["seed"])
    # Before anything imports or touches the device: the workspace variable is read at cuBLAS's
    # first use, so a pin after that point reports a determinism the run does not have.
    determinism = device.pin(seed=seed, attention=ATTENTION)

    recipe = load_full_finetune_config(config, root=PROJECT_ROOT)
    if iters is not None:
        recipe = FullFinetuneConfig(**{**vars(recipe), "max_steps": int(iters)})
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = load_model_spec(config["model"])
    source = str(PROJECT_ROOT / spec.hf_id) if not Path(spec.hf_id).is_absolute() else spec.hf_id
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation=ATTENTION
    )

    trainable = freeze_all_but_top_layers(model, recipe.trainable_top_layers)
    upcast = upcast_trainable_to_float32(model)
    build_adamw(model, learning_rate=recipe.learning_rate)  # refuses a dtype that would round

    train_set, valid_set = load_rendered_splits(
        recipe.data, tokenizer, max_seq_length=recipe.max_seq_length, splits=("train", "valid")
    )
    rows = rendered_rows(train_set)
    # Only loaded into the trainer when the recipe asks for validation; an eval cadence with no
    # eval set is a run that dies at the first eval step, hours in.
    validation = rendered_rows(valid_set) if recipe.val_batches else None

    manifest = {
        "backend": device.backend(),
        "device": device.select(),
        "determinism": determinism,
        "model": {"name": spec.name, "hf_id": spec.hf_id, "source": source},
        "recipe": {
            "source_iters": recipe.source_iters,
            "source_units": recipe.source_units,
            "max_steps": recipe.max_steps,
            "eval_steps": recipe.eval_steps,
            "save_steps": recipe.save_steps,
            "logging_steps": recipe.logging_steps,
            "gradient_accumulation_steps": recipe.gradient_accumulation_steps,
            "per_device_micro_batch": recipe.per_device_micro_batch,
            "learning_rate": recipe.learning_rate,
            "max_seq_length": recipe.max_seq_length,
            "trainable_top_layers": recipe.trainable_top_layers,
        },
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": trainable,
            "upcast_to_float32": upcast,
        },
        "rows": {"train": len(rows), "valid": len(valid_set)},
        "departures_from_the_mlx_record": list(recipe.departures),
        "resume_from": str(resume_from) if resume_from else None,
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # One object is handed to `Trainer`, and to `fully_shard` on a sharded run: the loss runs inside
    # this wrapper's forward, so autocast, the accumulation window and the unshard hooks all attach
    # to the one forward that runs. The explicit autocast dtype stays inside the loss regardless --
    # a nested autocast of the same dtype costs nothing and documents which precision the loss owns
    # rather than inherits.
    wrapped = wrap_with_chunked_loss(model, autocast_dtype=torch.bfloat16)
    selected = device.select()
    arguments = build_training_arguments(
        recipe,
        output=output,
        seed=seed,
        selected=selected,
        fsdp=fsdp_arguments(wrapped, world_size=world_size),
    )
    trainer = make_loss_module_trainer_class()(
        model=wrapped,
        args=arguments,
        train_dataset=rows,
        eval_dataset=validation,
        data_collator=CausalCollator(pad_token_id=tokenizer.pad_token_id or 0),
    )
    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)

    manifest["result"] = {
        "global_step": result.global_step,
        "training_loss": result.training_loss,
        # Read back rather than assumed: what the trainer actually put the model on.
        "device_trained_on": str(next(model.parameters()).device),
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
