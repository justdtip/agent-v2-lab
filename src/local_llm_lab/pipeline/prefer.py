"""Preference optimisation (DPO) of the SFT policy on mined step-level pairs.

The stage starts from the selected SFT LoRA adapter, uses that *adapted* model as the frozen
DPO reference (``precompute_ref_logprobs``), trains only the LoRA parameters with mlx-tune's
``DPOTrainer`` and writes an adapter directory that ``mlx_lm.load(model, adapter_path=...)``
accepts.

Notes on mlx-tune 0.6.0 behaviour this module works around (verified against the source):

* ``FastLanguageModel.from_pretrained`` ignores ``load_in_4bit`` and simply calls
  ``mlx_lm.load``; the already-quantized repo is never re-quantized.
* ``MLXModelWrapper.load_adapter`` sets ``_lora_applied`` (so the trainer does not wrap the
  layers a second time) but neither it nor ``mlx_lm.tuner.utils.load_adapters`` freezes the
  base weights, so the base model is frozen *before* the adapter is attached, mirroring
  ``mlx_lm.lora`` (freeze, then convert to LoRA layers).
* ``_save_adapters_and_config`` writes ``adapters.safetensors`` from the trainable parameters
  (exactly the LoRA tensors once the base is frozen) but synthesises ``adapter_config.json``
  from ``model.lora_config`` which is ``None`` here (wrong scale, missing keys).
  ``save_pretrained`` would additionally copy from ``_adapter_path``, i.e. the *input*
  adapter. Hence :func:`finalize_adapter` copies the trainer's weights and the input adapter's
  config verbatim: the LoRA layout is unchanged because training started from that adapter.
* ``gradient_accumulation_steps``/``warmup_steps``/``max_prompt_length`` are stored on the
  trainer but unused by ``_train_native``; the trainer also does not shuffle, so pairs are
  shuffled here with the run seed.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

from local_llm_lab.project import PROJECT_ROOT, configure_local_cache

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
DEFAULT_ADAPTER = PROJECT_ROOT / "outputs" / "agent-v2" / "best-adapter"
DEFAULT_PAIRS = PROJECT_ROOT / "data" / "preferences" / "pref1" / "pairs.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "agent-v2" / "prefer"
ADAPTER_FILES = ("adapters.safetensors", "adapter_config.json")
PAIR_KEYS = ("prompt", "chosen", "rejected")


# --------------------------------------------------------------------------- data


def _read_pairs(path: Path, max_chars: int | None) -> tuple[list[dict[str, str]], int]:
    """Validate ``pairs.jsonl``; return ``(kept rows, dropped-for-length count)``."""
    rows: list[dict[str, str]] = []
    dropped = 0
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON ({error.msg})") from error
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            for key in PAIR_KEYS:
                value = payload.get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{path}:{number}: '{key}' must be a non-empty string")
            row = {key: payload[key] for key in PAIR_KEYS}
            if row["chosen"] == row["rejected"]:
                raise ValueError(f"{path}:{number}: chosen and rejected are identical")
            if max_chars is not None and (
                len(row["prompt"]) + len(row["chosen"]) > max_chars
                or len(row["prompt"]) + len(row["rejected"]) > max_chars
            ):
                dropped += 1
                continue
            rows.append(row)
    return rows, dropped


def load_pairs(path: Path, *, max_chars: int | None = None) -> list[dict[str, str]]:
    """Read preference pairs written by the branch stage.

    Every row must carry non-empty string ``prompt``/``chosen``/``rejected`` with
    ``chosen != rejected``; ``metadata`` is stripped. Rows whose ``prompt + chosen`` or
    ``prompt + rejected`` exceed ``max_chars`` characters are dropped. Malformed rows raise
    ``ValueError`` naming the line.
    """
    return _read_pairs(path, max_chars)[0]


# --------------------------------------------------------------------------- config


def prefer_config(
    *,
    beta: float,
    learning_rate: float,
    max_steps: int,
    batch_size: int,
    max_seq_length: int,
    output_dir: Path,
    grad_accumulation: int,
    logging_steps: int = 10,
    save_steps: int = 100,
) -> Any:
    """Build mlx-tune's ``DPOConfig`` (pure; no side effects).

    The starting policy is the frozen reference: ``precompute_ref_logprobs=True`` makes the
    trainer cache reference log-probs from the model *as loaded* (base + SFT adapter) before
    any optimiser step. Parameter names mirror ``DPOConfig.__init__``.
    """
    from mlx_tune.rl_trainers import DPOConfig

    return DPOConfig(
        beta=beta,
        loss_type="sigmoid",
        precompute_ref_logprobs=True,
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        max_seq_length=max_seq_length,
        max_prompt_length=max_seq_length,
    )


# --------------------------------------------------------------------------- saving


def finalize_adapter(saved_weights: Path, source_adapter_dir: Path, target_dir: Path) -> Path:
    """Assemble a loadable mlx-lm adapter directory from the trainer's weights.

    ``target_dir`` ends up containing exactly ``adapters.safetensors`` (the trained weights)
    and ``adapter_config.json`` copied byte-for-byte from ``source_adapter_dir``: the LoRA
    layout (keys, rank, scale, layer count) is whatever the source adapter declared, because
    training started from those layers. Returns ``target_dir``.
    """
    source_config = source_adapter_dir / "adapter_config.json"
    if not saved_weights.is_file():
        raise FileNotFoundError(f"trained adapter weights not found: {saved_weights}")
    if not source_config.is_file():
        raise FileNotFoundError(f"source adapter config not found: {source_config}")
    config_bytes = source_config.read_bytes()
    target_dir.mkdir(parents=True, exist_ok=True)
    weights_target = target_dir / "adapters.safetensors"
    if saved_weights.resolve() != weights_target.resolve():
        shutil.copy2(saved_weights, weights_target)
    (target_dir / "adapter_config.json").write_bytes(config_bytes)
    for stray in target_dir.iterdir():
        if stray.name in ADAPTER_FILES:
            continue
        if stray.is_dir():
            shutil.rmtree(stray)
        else:
            stray.unlink()
    return target_dir


# --------------------------------------------------------------------------- training


def run_prefer(
    *,
    model_name: str,
    adapter: Path,
    pairs_path: Path,
    output: Path,
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    max_steps: int,
    batch_size: int = 1,
    max_seq_length: int = 3072,
    grad_accumulation: int = 4,
    max_chars: int | None = None,
    seed: int = 20260902,
) -> dict[str, Any]:
    """DPO-train the SFT adapter on mined pairs; write ``output/adapters`` + ``summary.json``."""
    adapter = adapter.resolve()
    pairs_path = pairs_path.resolve()
    for name in ADAPTER_FILES:
        if not (adapter / name).is_file():
            raise FileNotFoundError(
                f"{adapter} is not an mlx-lm adapter directory (missing {name})"
            )
    if not pairs_path.is_file():
        raise FileNotFoundError(f"pairs file not found: {pairs_path}")

    rows, dropped = _read_pairs(pairs_path, max_chars)
    if not rows:
        raise ValueError(f"no usable preference pairs in {pairs_path}")
    random.Random(seed).shuffle(rows)  # the trainer iterates in file order without shuffling
    print(
        f"Loaded {len(rows)} preference pairs from {pairs_path} ({dropped} dropped by --max-chars)"
    )

    configure_local_cache()
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_tune.model import FastLanguageModel
    from mlx_tune.rl_trainers import DPOTrainer

    mx.random.seed(seed)
    started = time.monotonic()
    print(f"Loading {model_name} (pre-quantized; mlx_lm.load, no re-quantization)")
    # `load_in_4bit` is accepted for API parity but unused by mlx-tune 0.6.0; the repo is
    # already 4-bit and `from_pretrained` only ever calls `mlx_lm.load`.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name, max_seq_length=max_seq_length, load_in_4bit=True
    )
    # Freeze the base BEFORE attaching the adapter so that the LoRA layers created by
    # `load_adapters` are the only unfrozen parameters (the mlx_lm.lora ordering).
    model.model.freeze()
    print(f"Attaching SFT adapter {adapter}")
    model.load_adapter(str(adapter))
    if not getattr(model, "_lora_applied", False):
        raise RuntimeError("load_adapter did not mark LoRA as applied; the trainer would re-wrap")
    trainable = [key for key, _ in tree_flatten(model.model.trainable_parameters())]
    if not trainable or any("lora" not in key for key in trainable):
        raise RuntimeError(
            "trainable parameters are not exactly the LoRA tensors: "
            + ", ".join(key for key in trainable if "lora" not in key)[:500]
        )
    print(f"Trainable tensors: {len(trainable)} (all LoRA); base model frozen")

    trainer_dir = output / "trainer"
    config = prefer_config(
        beta=beta,
        learning_rate=learning_rate,
        max_steps=max_steps,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        output_dir=trainer_dir,
        grad_accumulation=grad_accumulation,
    )
    if grad_accumulation != 1:
        print(
            f"Note: mlx-tune 0.6.0's native DPO loop ignores gradient_accumulation_steps={grad_accumulation}; "
            f"effective batch is {batch_size}"
        )
    trainer = DPOTrainer(
        model=model, ref_model=None, train_dataset=rows, tokenizer=tokenizer, args=config
    )
    if not trainer.use_native:
        raise RuntimeError("mlx-tune native DPO is unavailable; refusing the SFT fallback")
    print(
        f"Training: beta={beta} lr={learning_rate} steps={max_steps} batch={batch_size} "
        f"max_seq_length={max_seq_length} seed={seed}; reference = frozen SFT policy"
    )
    result = trainer.train()
    saved_weights = Path(result["adapter_path"]) / "adapters.safetensors"
    final_dir = finalize_adapter(saved_weights, adapter, output / "adapters")
    elapsed = round(time.monotonic() - started, 2)
    print(f"Adapter written to {final_dir} in {elapsed / 60:.1f} min")

    summary = {
        "model": model_name,
        "adapter": str(adapter),
        "pairs": str(pairs_path),
        "pairs_used": len(rows),
        "pairs_dropped": dropped,
        "max_chars": max_chars,
        "config": config.to_dict(),
        "trainable_tensors": len(trainable),
        "seed": seed,
        "elapsed_seconds": elapsed,
        "saved_adapter": str(final_dir),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    del trainer, model
    mx.clear_cache()
    return summary


# --------------------------------------------------------------------------- entry point


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DPO-train the SFT adapter on mined step-level preference pairs."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument(
        "--max-chars",
        type=int,
        help="Drop pairs whose prompt+completion exceeds this many characters.",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    for name in (
        "beta",
        "learning_rate",
        "max_steps",
        "batch_size",
        "max_seq_length",
        "grad_accumulation",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_chars is not None and args.max_chars <= 0:
        parser.error("--max-chars must be positive")
    run_prefer(
        model_name=args.model,
        adapter=args.adapter,
        pairs_path=args.pairs,
        output=args.output,
        beta=args.beta,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        grad_accumulation=args.grad_accumulation,
        max_chars=args.max_chars,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
