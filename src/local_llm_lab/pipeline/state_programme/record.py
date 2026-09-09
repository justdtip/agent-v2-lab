"""The record directory (order §6), every artefact written atomically, and a README from its files.

A tool that writes once at the end is not allowed in the device hour (first-hour runbook), so each
stage writes its own file when it finishes, and a crash leaves what ran readable. The README is
written **from those files and nothing else**: it opens them and formats them, so a number in the
README that is not in a file cannot exist.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from local_llm_lab.runlog import write_text_atomic

ARTEFACTS = (
    "manifest.json", "rate.json", "pilot/rows.jsonl", "preregistration.json",
    "main/rows.jsonl", "estimands.json", "README.md",
)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append one JSONL row; rows are the unit a resume reads, so each lands whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "absent"


def _fmt(value: Any) -> str:
    return f"{value:.5f}" if isinstance(value, float) else str(value)


def write_readme(directory: Path) -> Path:
    """The README, from the files. Each number carries its basis: the file it came from."""
    manifest = json.loads((directory / "manifest.json").read_text()) if (directory / "manifest.json").exists() else {}
    rate = json.loads((directory / "rate.json").read_text()) if (directory / "rate.json").exists() else None
    seal = json.loads((directory / "preregistration.json").read_text()) if (directory / "preregistration.json").exists() else None
    estimands = json.loads((directory / "estimands.json").read_text()) if (directory / "estimands.json").exists() else None
    pilot_rows = read_rows(directory / "pilot" / "rows.jsonl")
    main_rows = read_rows(directory / "main" / "rows.jsonl")

    lines = ["# State programme record", ""]
    lines.append(f"Decoding mode: `{manifest.get('decoding', {}).get('mode', 'unknown')}` — "
                 f"estimand: {manifest.get('decoding', {}).get('estimand', 'unknown')} *(manifest.json)*")
    lines.append(f"Fixture run: `{manifest.get('fixture', False)}` *(manifest.json)*")
    lines.append("")
    if rate is not None:
        lines.append(f"**Rate**: {_fmt(rate['episodes_per_hour'])} episodes/hour, measured over "
                     f"{rate['episodes']} episodes in mode `{rate['mode']}` *(rate.json)*")
    else:
        lines.append("**Rate**: not measured *(rate.json absent)*")
    lines.append(f"**Pilot rows**: {len(pilot_rows)} *(pilot/rows.jsonl)*")
    if seal is not None:
        t = seal["tolerances"]
        lines += ["", "## Pre-registration *(preregistration.json)*", "",
                  f"Retained diagnostics: {', '.join(t['retained'])} (M = {t['M']})",
                  "Dropped: " + ("; ".join(f"{d['name']} — {d['reason']}" for d in t["dropped"]) or "none"),
                  "", "| tolerance | value | derivation |", "|---|---:|---|"]
        for key in ("d_min", "epsilon_sub", "epsilon_perp", "epsilon_reuse", "epsilon_pred", "epsilon_dyn"):
            lines.append(f"| `{key}` | {_fmt(t[key])} | {t['derivation'].get(key, t['derivation'].get('epsilon_pred, epsilon_dyn', ''))} |")
        lines.append(f"| `n` | {t['n']} | {t['derivation']['n']} |")
        b = seal.get("budget")
        if b:
            lines.append(f"\nBudget: {'loosened' if b['loosened'] else 'within budget'} — {b['reason']}")
    else:
        lines += ["", "**Pre-registration**: absent — the main stage cannot start *(preregistration.json absent)*"]
    lines.append(f"\n**Main rows**: {len(main_rows)} *(main/rows.jsonl)*")
    if estimands is not None:
        lines += ["", "## Estimands *(estimands.json)*", "", "| estimand | distance | tolerance | passes |", "|---|---:|---:|---|"]
        for name, e in estimands["estimands"].items():
            lines.append(f"| {name} | {_fmt(e['distance'])} | {_fmt(e['tolerance'])} | {e['passes']} |")
        lines.append(f"\nLevel: {estimands['level']}")
    lines += ["", "The pilot is not a result and its numbers are not findings. Every figure above names the file it was read from."]
    target = directory / "README.md"
    write_text_atomic(target, "\n".join(lines) + "\n")
    return target


__all__ = ["ARTEFACTS", "append_row", "digest", "read_rows", "write_json", "write_readme"]
