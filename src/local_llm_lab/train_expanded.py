from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset
from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
from mlx_lm.utils import get_total_parameters

from local_llm_lab.depth_expansion import (
    ExpansionSpec,
    load_expanded_model,
    trainable_parameter_count,
)
from local_llm_lab.evaluate_agent import DEFAULT_MODEL
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache


class StableAdamW(optim.AdamW):
    """AdamW with float32 moments and global norm clipping for float16 full tuning."""

    def __init__(self, *args: Any, max_grad_norm: float, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.max_grad_norm = max_grad_norm

    def init_single(self, parameter: mx.array, state: dict[str, mx.array]) -> None:
        state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
        state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)

    def update(self, model: Any, gradients: dict[str, Any]) -> None:
        gradients, _ = optim.clip_grad_norm(gradients, self.max_grad_norm)
        super().update(model, gradients)

    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict[str, mx.array],
    ) -> mx.array:
        gradient = gradient.astype(mx.float32)
        parameter32 = parameter.astype(mx.float32)
        learning_rate = self.learning_rate.astype(mx.float32)
        beta1, beta2 = self.betas
        moment = beta1 * state["m"] + (1 - beta1) * gradient
        variance = beta2 * state["v"] + (1 - beta2) * mx.square(gradient)
        state["m"] = moment
        state["v"] = variance
        if self.bias_correction:
            moment = moment / (1 - beta1**self.step)
            variance = variance / (1 - beta2**self.step)
        update = moment / (mx.sqrt(variance) + self.eps)
        update = update + self.weight_decay * parameter32
        return (parameter32 - learning_rate * update).astype(parameter.dtype)


class JsonlMetrics:
    """Persist trainer callbacks with an explicit curriculum stage."""

    def __init__(self, path: Path, stage: str):
        self.path = path
        self.stage = stage

    def _write(self, kind: str, values: dict[str, Any]) -> None:
        record = {"stage": self.stage, "kind": kind, **values}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
        self._write("train", train_info)

    def on_val_loss_report(self, val_info: dict[str, Any]) -> None:
        self._write("validation", val_info)


def _set_output_warmup(model: Any, added_indices: tuple[int, ...]) -> int:
    """Train only zero-initialized residual outputs while copied features stay fixed."""
    model.freeze()
    for index in added_indices:
        block = model.layers[index]
        block.self_attn.o_proj.unfreeze()
        block.mlp.down_proj.unfreeze()
    return trainable_parameter_count(model)


def _set_full_extension_training(
    model: Any,
    added_indices: tuple[int, ...],
    *,
    train_base_lora: bool,
) -> int:
    """Open added blocks and, optionally, the inherited policy's low-rank weights."""
    model.freeze()
    if train_base_lora:
        model.unfreeze(keys=["lora_a", "lora_b"])
    for index in added_indices:
        model.layers[index].unfreeze()
    return trainable_parameter_count(model)


def _make_optimizer(args: argparse.Namespace, learning_rate: float) -> StableAdamW:
    return StableAdamW(
        learning_rate=learning_rate,
        eps=args.adam_eps,
        weight_decay=0.01,
        bias_correction=True,
        max_grad_norm=args.max_grad_norm,
    )


def _training_args(
    args: argparse.Namespace,
    *,
    iters: int,
    adapter_file: Path,
    grad_checkpoint: bool,
) -> TrainingArgs:
    return TrainingArgs(
        batch_size=args.batch_size,
        iters=iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=min(args.steps_per_eval, iters),
        steps_per_save=min(args.save_every, iters),
        adapter_file=str(adapter_file),
        max_seq_length=args.max_seq_length,
        grad_checkpoint=grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_config(path: Path, args: argparse.Namespace, spec: ExpansionSpec) -> None:
    payload: dict[str, Any] = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    payload["expansion"] = {
        "insertion_after": list(spec.insertion_after),
        "initialization": spec.initialization,
        "version": spec.version,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train full-precision identity blocks over a frozen 4-bit Qwen+LoRA policy."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-adapter",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "agent-3b" / "best-adapter",
    )
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "expanded_agent_sft")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "complex-agent" / "run-a"
    )
    parser.add_argument("--added-layers", type=int, default=4)
    parser.add_argument("--warmup-iters", type=int, default=80)
    parser.add_argument("--warmup-learning-rate", type=float, default=5e-6)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--val-batches", type=int, default=36)
    parser.add_argument("--test-batches", type=int, default=60)
    parser.add_argument("--steps-per-report", type=int, default=5)
    parser.add_argument("--steps-per-eval", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--train-base-lora",
        action="store_true",
        help="Also retune the inherited LoRA weights during the full-extension stage.",
    )
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    args = parser.parse_args()
    positive_ints = (
        args.added_layers,
        args.warmup_iters,
        args.iters,
        args.batch_size,
        args.grad_accumulation_steps,
        args.max_seq_length,
        args.val_batches,
        args.test_batches,
        args.steps_per_report,
        args.steps_per_eval,
        args.save_every,
    )
    if (
        any(value < 1 for value in positive_ints)
        or args.warmup_learning_rate <= 0
        or args.learning_rate <= 0
        or args.adam_eps <= 0
        or args.max_grad_norm <= 0
    ):
        parser.error("training sizes and learning rate must be positive")
    if (
        args.warmup_iters % args.grad_accumulation_steps
        or args.iters % args.grad_accumulation_steps
    ):
        parser.error(
            "warmup-iters and iters must be divisible by grad-accumulation-steps; "
            "otherwise MLX-LM drops the final partial optimizer update"
        )

    configure_local_cache()
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = ExpansionSpec.evenly_spaced(36, args.added_layers)
    spec.save(output / "expansion.json")
    _write_config(output / "training_config.json", args, spec)

    print("Loading frozen 4-bit base and trained agent LoRA...", flush=True)
    model, tokenizer, added_indices = load_expanded_model(args.model, args.base_adapter, spec)
    total_parameters = get_total_parameters(model)
    print(f"Expanded layer indices: {added_indices}", flush=True)

    dataset_config = SimpleNamespace(mask_prompt=True)
    train_set, valid_set, test_set = load_local_dataset(
        args.data.resolve(), tokenizer, dataset_config
    )
    print(
        f"Rows: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}",
        flush=True,
    )
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    warmup_dir = output / "warmup"
    warmup_dir.mkdir(exist_ok=True)

    # Stage 1: copied q/k/v and MLP feature projections remain fixed. Only the zeroed
    # residual outputs learn to turn on, bounding the initial departure from exact identity.
    warmup_parameters = _set_output_warmup(model, added_indices)
    print(
        f"Stage 1/2, residual-output warmup: {warmup_parameters / 1e6:.3f}M trainable parameters",
        flush=True,
    )
    train(
        model=model,
        optimizer=_make_optimizer(args, args.warmup_learning_rate),
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=_training_args(
            args,
            iters=args.warmup_iters,
            adapter_file=warmup_dir / "warmup.safetensors",
            # MLX-LM installs checkpointing by patching the transformer block class.
            # Install it exactly once here; the wrapper remains active for stage 2.
            grad_checkpoint=not args.no_grad_checkpoint,
        ),
        training_callback=JsonlMetrics(metrics_path, "output_warmup"),
    )

    # Stage 2: all parameters in the new blocks become trainable at a lower LR. A fresh
    # optimizer avoids carrying output-only moment estimates into the newly opened weights.
    mx.clear_cache()
    np.random.seed(args.seed + 1)
    trainable_parameters = _set_full_extension_training(
        model,
        added_indices,
        train_base_lora=args.train_base_lora,
    )
    print(
        "Stage 2/2, full extension: "
        f"{trainable_parameters / 1e6:.3f}M trainable parameters "
        f"({100 * trainable_parameters / total_parameters:.3f}% of model)",
        flush=True,
    )
    train(
        model=model,
        optimizer=_make_optimizer(args, args.learning_rate),
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=_training_args(
            args,
            iters=args.iters,
            adapter_file=output / "extension.safetensors",
            grad_checkpoint=False,
        ),
        training_callback=JsonlMetrics(metrics_path, "full_extension"),
    )
    print("Testing token loss...", flush=True)
    test_loss = evaluate(
        model=model,
        dataset=CacheDataset(test_set),
        batch_size=args.batch_size,
        num_batches=args.test_batches,
        max_seq_length=args.max_seq_length,
    )
    summary = {
        "total_parameters": total_parameters,
        "warmup_trainable_parameters": warmup_parameters,
        "trainable_parameters": trainable_parameters,
        "added_layer_indices": list(added_indices),
        "test_loss": round(test_loss, 6),
        "test_perplexity": round(math.exp(test_loss), 6),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 6),
        "extension_sha256": _sha256(output / "extension.safetensors"),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
