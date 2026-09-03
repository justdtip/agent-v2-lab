from __future__ import annotations

import argparse
import subprocess

from local_llm_lab.project import PROJECT_ROOT, configure_local_cache

DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the MLX-LM chat interface.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Download/load the model and start chatting. Without this flag, print the command.",
    )
    args = parser.parse_args()

    cache = configure_local_cache()
    command = ["mlx_lm.chat", "--model", args.model]
    print(f"Model cache: {cache}")
    print("Command: " + " ".join(command))
    if not args.run:
        print("Preview only. Add --run to download/load the model.")
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
