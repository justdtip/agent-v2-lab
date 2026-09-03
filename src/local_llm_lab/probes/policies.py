"""The four policies every probe compares: the untouched base and the three adapter runs."""

from __future__ import annotations

from pathlib import Path

from local_llm_lab.project import PROJECT_ROOT

__all__ = ["POLICY_ADAPTERS", "POLICY_NAMES", "resolve_policy"]

POLICY_ADAPTERS: dict[str, str | None] = {
    "base": None,
    "A": "outputs/agent-v2/best-adapter",
    "B": "outputs/agent-v2b/best-adapter",
    "C": "outputs/agent-v2c/best-adapter",
}
POLICY_NAMES = tuple(POLICY_ADAPTERS)


def resolve_policy(name: str) -> Path | None:
    """Adapter directory for ``base``/``A``/``B``/``C``, or an explicit path passed through."""
    if name in POLICY_ADAPTERS:
        relative = POLICY_ADAPTERS[name]
        return None if relative is None else (PROJECT_ROOT / relative).resolve()
    path = Path(name)
    if not path.is_dir():
        raise ValueError(f"unknown policy {name!r}: expected one of {POLICY_NAMES} or a directory")
    return path.resolve()
