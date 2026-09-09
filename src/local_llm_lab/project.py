from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_local_cache() -> Path:
    """Keep downloaded models and datasets inside the ignored project cache, in the primary.

    The cache is a shared artefact like the lock, the window and `models/`: git-ignored and
    populated once, in the primary checkout. It used to hang off `PROJECT_ROOT`, the running
    checkout, so a stage in a worktree found an empty `.cache` and either refused or downloaded
    the weights again (SWE-1, 2026-09-09). It resolves through the git-derived primary, never
    through the box-state override, which redirects the lock and the window and nothing else.
    An explicit `HF_HOME` still wins, as before.
    """
    from local_llm_lab.runlock import primary_checkout_root

    cache = primary_checkout_root() / ".cache" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    return Path(os.environ["HF_HOME"])
