"""Read an arm's training config for the torch full-fine-tuning path, in the units it is written in.

Every `configs/*.yaml` in this repository was written for mlx-lm, and **mlx-lm counts iterations in
micro-batches**. `iters: 1200` with `grad_accumulation_steps: 4` is 300 optimizer steps, not 1,200;
so are `steps_per_eval`, `save_every` and `steps_per_report`. `transformers` counts optimizer steps
everywhere. Handing these numbers to `Trainer` unconverted trains an arm four times as long as its
record says and evaluates it four times as often.

The configs carry `iters_unit: batches` to say so, and **nothing in the tree reads that key** -- it
is a comment with a colon in it. This module reads it, requires it to name a unit it knows, and
refuses a config that claims a unit whose conversion it does not implement. A field whose whole job
is to mark a unit must be compared, not merely present (the argument of #75, applied again).

The conversion is exact by construction: a config whose iteration counts are not whole multiples of
the accumulation is refused rather than rounded, because mlx-lm drops the final partial optimizer
update and a rounded conversion would quietly disagree with the record it is trying to reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Units `iters` and the cadence fields may be expressed in. `batches` is every config in the tree
#: today; `steps` exists so that a config written after this module can say the other thing rather
#: than inherit this one's assumption silently.
ITERS_UNITS: tuple[str, ...] = ("batches", "steps")

#: Config keys under `train:` that describe a LoRA adapter and have no full-fine-tuning meaning.
LORA_ONLY_KEYS: tuple[str, ...] = ("rank", "scale", "dropout")

#: Config keys under `train:` that configure the Qwen3.5 gated-delta recurrence. A dense model has
#: no such state, so they are inapplicable rather than merely unused.
RECURRENCE_KEYS: tuple[str, ...] = ("gated_delta_chunk", "gated_delta_mode")

#: Cadence fields mlx-lm expresses in the same unit as `iters`.
CADENCE_KEYS: tuple[str, ...] = ("iters", "steps_per_eval", "save_every", "steps_per_report")

#: Of those, the ones a rounded conversion would change the *experiment* rather than its log:
#: how long the arm trains, where it is evaluated, and which step each checkpoint is. `ckpt 800`
#: means a particular checkpoint, so a save cadence that lands elsewhere is a different run.
#: `steps_per_report` is left out on purpose -- it moves only how often a line is printed.
EXACT_CADENCE_KEYS: tuple[str, ...] = ("iters", "steps_per_eval", "save_every")


class TrainingConfigError(ValueError):
    """A config this path cannot run, as opposed to one it can run differently."""


@dataclass(frozen=True)
class FullFinetuneConfig:
    """An arm's recipe in optimizer steps, with what it cost to get there recorded."""

    model: str
    data: Path
    output: Path
    seed: int
    #: Optimizer steps, converted from the config's own unit.
    max_steps: int
    eval_steps: int
    save_steps: int
    logging_steps: int
    per_device_micro_batch: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_seq_length: int
    gradient_checkpointing: bool
    val_batches: int
    #: How many layers, counted from the output, are trainable. `None` trains the whole model.
    #: This is the full-fine-tuning analogue of mlx-lm's `lora_layers`, and it is what the top-8
    #: depth result was measured with.
    trainable_top_layers: int | None
    #: Verbatim source numbers, so a manifest can quote the config rather than the conversion.
    source_units: str
    source_iters: int
    #: Every way this run is not the MLX run of the same config. Written into the run manifest.
    departures: tuple[str, ...] = field(default_factory=tuple)


def _require_whole(name: str, value: int, accumulation: int) -> int:
    if value % accumulation:
        raise TrainingConfigError(
            f"train.{name}={value} is not a whole number of optimizer steps at "
            f"grad_accumulation_steps={accumulation}. mlx-lm drops the final partial update, so a "
            "rounded conversion would not reproduce the record this config describes; change the "
            "config rather than the arithmetic."
        )
    return value // accumulation


def load_full_finetune_config(payload: dict[str, Any], *, root: Path) -> FullFinetuneConfig:
    """Build a torch training config from a parsed arm yaml.

    ``root`` resolves the config's relative ``data`` and ``output`` paths, so a run from a worktree
    reads the same files as one from the project root.
    """
    train = dict(payload.get("train") or {})
    if not train:
        raise TrainingConfigError("config has no `train:` section")

    unit = str(train.get("iters_unit", "batches"))
    if unit not in ITERS_UNITS:
        raise TrainingConfigError(
            f"train.iters_unit={unit!r} is not one of {list(ITERS_UNITS)}; this reader converts "
            "the units it knows and refuses the ones it does not."
        )

    accumulation = int(train.get("grad_accumulation_steps", 1))
    if accumulation < 1:
        raise TrainingConfigError(f"train.grad_accumulation_steps={accumulation} must be >= 1")

    divisor = accumulation if unit == "batches" else 1
    converted: dict[str, int] = {}
    rounded: list[str] = []
    for key in CADENCE_KEYS:
        if key not in train:
            converted[key] = 0
            continue
        value = int(train[key])
        if divisor == 1:
            converted[key] = value
        elif key in EXACT_CADENCE_KEYS:
            converted[key] = _require_whole(key, value, divisor)
        else:
            converted[key] = max(1, round(value / divisor))
            if value % divisor:
                rounded.append(f"{key} {value}/{divisor} -> {converted[key]}")

    departures: list[str] = []
    if unit == "batches" and accumulation > 1:
        departures.append(
            f"cadence converted from micro-batches to optimizer steps by //{accumulation}: "
            + ", ".join(f"{key} {train[key]}->{converted[key]}" for key in CADENCE_KEYS if key in train)
        )
    if rounded:
        departures.append(
            "reporting cadence rounded to whole optimizer steps (" + ", ".join(rounded)
            + "); this moves how often a line is logged and nothing the run computes"
        )
    departures.append(
        "loss normalisation: transformers weights each optimizer step by its supervised token "
        "count (num_items_in_batch); mlx-lm averages per-batch means unweighted. Under "
        "length-sorted batching the two differ whenever batches hold unequal token counts."
    )
    departures.append(
        "batch order and padding are transformers' own: mlx-lm length-sorts, forms fixed batches, "
        "permutes them from numpy's global seeded state, and pads to 1 + 32*ceil(L/32)."
    )
    present_lora = [key for key in LORA_ONLY_KEYS if key in train]
    if present_lora:
        departures.append(
            "full fine-tuning ignores the adapter fields " + ", ".join(sorted(present_lora))
        )
    present_recurrence = [key for key in RECURRENCE_KEYS if key in train]
    if present_recurrence:
        departures.append(
            "gated-delta fields " + ", ".join(sorted(present_recurrence))
            + " describe a Qwen3.5 recurrence and do not apply to a dense model"
        )

    return FullFinetuneConfig(
        model=str(payload["model"]),
        data=(root / str(payload["data"])).resolve(),
        output=(root / str(payload["output"])).resolve(),
        seed=int(payload["seed"]),
        max_steps=converted["iters"],
        eval_steps=converted["steps_per_eval"],
        save_steps=converted["save_every"],
        logging_steps=max(1, converted["steps_per_report"]),
        per_device_micro_batch=int(train.get("batch_size", 1)),
        gradient_accumulation_steps=accumulation,
        learning_rate=float(train["learning_rate"]),
        max_seq_length=int(train["max_seq_length"]),
        gradient_checkpointing=bool(train.get("grad_checkpoint", True)),
        val_batches=int(train.get("val_batches", 0)),
        trainable_top_layers=(
            int(train["lora_layers"]) if train.get("lora_layers") is not None else None
        ),
        source_units=unit,
        source_iters=int(train["iters"]),
        departures=tuple(departures),
    )
