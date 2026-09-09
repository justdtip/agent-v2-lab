"""The pre-registration and its two refusals (order §1 stage 3, §7).

The main stage **refuses to start without the seal** and **refuses to re-derive once any main-run
row exists**. Both are functions that raise by name, not flags: a script that could be talked past
them by an argument would not be a seal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from local_llm_lab.pipeline.state_programme.record import digest, read_rows, write_json


class NotSealed(RuntimeError):
    """The main stage was asked to start without a pre-registration."""


class AlreadyRunning(RuntimeError):
    """A re-derivation was asked for after main-run rows exist."""


def seal(
    directory: Path,
    *,
    table: dict[str, Any],
    budget: dict[str, Any] | None,
    decoding: dict[str, Any],
    pilot_rows_path: Path,
) -> Path:
    refuse_rederive(directory)
    payload = {
        "schema_version": 1,
        "tolerances": table,
        "budget": budget,
        "decoding": decoding,
        "derived_from": {
            "pilot_rows_sha256": digest(pilot_rows_path),
            "pilot_row_count": len(read_rows(pilot_rows_path)),
        },
        "rules": {
            "drop": "a diagnostic whose bootstrap bound on |contrast| does not exceed zero is dropped with its reason; nothing is added after the pilot",
            "budget": "over budget loosens epsilon_sub and writes both numbers; never fewer diagnostics, never no control",
            "seal": "the main stage refuses to start without this file and refuses to re-derive once a main row exists",
        },
    }
    return write_json(directory / "preregistration.json", payload)


def seal_digest(directory: Path) -> str:
    return digest(directory / "preregistration.json")


def require_seal(directory: Path) -> dict[str, Any]:
    path = directory / "preregistration.json"
    if not path.exists():
        raise NotSealed(f"{path} does not exist: the main stage does not start without a pre-registration")
    payload = json.loads(path.read_text())
    for key in ("tolerances", "decoding", "derived_from"):
        if key not in payload:
            raise NotSealed(f"{path} is missing {key!r}; a partial seal is no seal")
    return payload


def refuse_rederive(directory: Path) -> None:
    rows = read_rows(directory / "main" / "rows.jsonl")
    if rows:
        raise AlreadyRunning(
            f"{len(rows)} main-run rows exist under {directory}; the tolerances are not re-derived "
            "once the main run has begun. Start a new record directory for a new derivation."
        )


def content_key(parts: dict[str, str]) -> str:
    """One digest over named parts, so a resume refusal can name every field that differs."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


__all__ = ["AlreadyRunning", "NotSealed", "content_key", "refuse_rederive", "require_seal", "seal", "seal_digest"]
