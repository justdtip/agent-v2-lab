from __future__ import annotations

import argparse
import shlex

from local_llm_lab import spawn
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache

DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"


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
    spawn.run(_in_project_root(command), check=True)
