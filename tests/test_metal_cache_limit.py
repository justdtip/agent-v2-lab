"""The training stage caps the Metal allocator's cache before it loads the base (heartbeat
2026-09-06 14:05): the probe that fitted this trainer's step ran under a 2 GiB cap and the
pipeline, with none, died at the first optimizer step three times."""
from __future__ import annotations

import pytest

from local_llm_lab.pipeline import cli


def test_limit_metal_cache_sets_the_configured_bytes(monkeypatch) -> None:
    seen: list[int] = []
    import mlx.core as mx

    monkeypatch.setattr(mx, "set_cache_limit", lambda value: seen.append(value))
    assert cli._limit_metal_cache({"metal_cache_gib": 1.5}) == int(1.5 * 2**30)
    assert seen == [int(1.5 * 2**30)]


def test_limit_metal_cache_defaults_to_two_gib_and_refuses_nonpositive(monkeypatch) -> None:
    import mlx.core as mx

    monkeypatch.setattr(mx, "set_cache_limit", lambda value: None)
    assert cli._limit_metal_cache({}) == 2 * 2**30
    with pytest.raises(ValueError):
        cli._limit_metal_cache({"metal_cache_gib": 0})
