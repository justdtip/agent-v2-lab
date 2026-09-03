from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from local_llm_lab.models import ModelSpec, ResolvedSpec
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION


def write_provenance(
    run_dir: Path,
    *,
    resolved: ResolvedSpec | None,
    spec: ModelSpec,
    extra: dict[str, Any],
) -> Path:
    payload = {
        "generator_version": GENERATOR_VERSION,
        "model": resolved.as_dict() if resolved is not None else asdict(spec),
        "extra": extra,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "provenance.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
