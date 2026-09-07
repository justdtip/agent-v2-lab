from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from local_llm_lab import spawn
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache


def _in_project_root(command: list[str]) -> list[str]:
    """``command``, run from the project root, without ``subprocess``'s ``cwd``.

    ``mlx_lm``'s YAML resolves ``data:`` and ``adapter_path:`` against the working directory, so
    the directory is load-bearing and cannot simply be dropped. It moves into argv instead:
    CPython takes ``posix_spawn`` only when ``cwd`` is ``None``, and a fork in an interpreter
    that has initialised Metal aborts (R45, issue 84 item 3). ``spawn``'s docstring names this
    form -- put ``/bin/sh -c`` in argv, so the exec that actually happens is visible -- and
    ``exec`` means no extra process survives the call.
    """
    inner = shlex.join(command)
    return ["/bin/sh", "-c", f"cd {shlex.quote(str(PROJECT_ROOT))} && exec {inner}"]


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
    spawn.run(_in_project_root(command), check=True)
