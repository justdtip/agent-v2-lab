"""Registry-backed policy paths and model-depth-aware probe layer selections."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from local_llm_lab.models import ModelSpec
from local_llm_lab.project import PROJECT_ROOT

__all__ = [
    "LayerSelection",
    "policy_names",
    "resolve_layers",
    "resolve_policy",
    "validate_layer_syntax",
]

_INTEGER_TOKEN = re.compile(r"[+-]?\d+")


@dataclass(frozen=True)
class LayerSelection:
    """A reproducible mapping from requested layer tokens to decoder indices."""

    source: str
    requested: tuple[str, ...]
    fractions: tuple[float, ...]
    indices: tuple[int, ...]
    num_layers: int

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "requested": list(self.requested),
            "fractions": list(self.fractions),
            "indices": list(self.indices),
            "num_layers": self.num_layers,
        }


def policy_names(spec: ModelSpec) -> tuple[str, ...]:
    """Return the policy names declared for one model, with the implicit base first."""
    return ("base", *(name for name in spec.policies if name != "base"))


def resolve_policy(name: str, spec: ModelSpec) -> Path | None:
    """Resolve a model's named policy or an explicitly supplied existing directory."""
    if name == "base":
        return None
    if name in spec.policies:
        return (PROJECT_ROOT / spec.policies[name]).resolve()
    path = Path(name)
    if path.is_dir():
        return path.resolve()
    raise ValueError(
        f"unknown policy {name!r}: expected one of {policy_names(spec)} or a directory"
    )


def _layer_tokens(raw: str) -> tuple[tuple[str, int | float], ...]:
    tokens = raw.split(",")
    if not tokens or any(not token.strip() for token in tokens):
        raise ValueError("--layers must be comma-separated indices or fractions")
    parsed: list[tuple[str, int | float]] = []
    for raw_token in tokens:
        token = raw_token.strip()
        if _INTEGER_TOKEN.fullmatch(token):
            value = int(token)
            if value <= 0:
                raise ValueError("--layers indices must be positive")
        else:
            try:
                value = float(token)
            except ValueError as error:
                raise ValueError("--layers must be comma-separated indices or fractions") from error
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("--layers fractions must be finite and within (0, 1]")
        parsed.append((token, value))
    return tuple(parsed)


def validate_layer_syntax(raw: str | None) -> None:
    """Validate lexical layer syntax without needing a model or its decoder depth."""
    if raw is not None:
        _layer_tokens(raw)


def resolve_layers(raw: str | None, spec: ModelSpec, num_layers: int) -> LayerSelection:
    """Resolve registry fractions or explicit mixed tokens against actual decoder depth."""
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer")

    source = "registry-default" if raw is None else "cli"
    if raw is None:
        tokens: tuple[tuple[str, int | float], ...] = tuple(
            (str(fraction), fraction) for fraction in spec.probe_layer_fractions
        )
    else:
        tokens = _layer_tokens(raw)

    requested: list[str] = []
    fractions: list[float] = []
    indices: list[int] = []
    seen: set[int] = set()
    for token, value in tokens:
        if isinstance(value, int):
            index = value
            fraction = index / num_layers
        else:
            fraction = value
            index = max(1, round(fraction * num_layers))
        if not 1 <= index <= num_layers:
            raise ValueError(f"--layers indices must be within [1, {num_layers}]")
        if index in seen:
            continue
        seen.add(index)
        requested.append(token)
        fractions.append(fraction)
        indices.append(index)

    return LayerSelection(
        source=source,
        requested=tuple(requested),
        fractions=tuple(fractions),
        indices=tuple(indices),
        num_layers=num_layers,
    )
