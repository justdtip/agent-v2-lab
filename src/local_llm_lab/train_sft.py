from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from local_llm_lab.project import PROJECT_ROOT, configure_local_cache


def _validate_dataset(data_dir: Path) -> None:
    missing = [
        name
        for name in ("train.jsonl", "valid.jsonl", "test.jsonl")
        if not (data_dir / name).is_file()
    ]
    if missing:
        raise SystemExit(f"Missing dataset files in {data_dir}: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview or run MLX-LM LoRA/QLoRA training.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "lora.yaml")
    parser.add_argument("--model", help="Override the model configured in the YAML file.")
    parser.add_argument("--iters", type=int, help="Override the configured training iterations.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Download/load the model and train. Without this flag, only validate and preview.",
    )
    args = parser.parse_args()

    config = args.config.resolve()
    if not config.is_file():
        raise SystemExit(f"Config does not exist: {config}")
    _validate_dataset(PROJECT_ROOT / "data" / "sft")
    cache = configure_local_cache()

    command = ["mlx_lm.lora", "--config", str(config)]
    if args.model:
        command.extend(["--model", args.model])
    if args.iters is not None:
        if args.iters < 1:
            parser.error("--iters must be positive")
        command.extend(["--iters", str(args.iters)])

    print(f"Model cache: {cache}")
    print("Command: " + shlex.join(command))
    if not args.run:
        print("Dataset and config look valid. Add --run to begin training.")
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
