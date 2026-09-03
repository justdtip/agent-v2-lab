from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_local_cache() -> Path:
    """Keep downloaded models and datasets inside the ignored project cache."""
    cache = PROJECT_ROOT / ".cache" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    return Path(os.environ["HF_HOME"])
