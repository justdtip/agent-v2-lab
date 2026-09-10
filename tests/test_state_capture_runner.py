"""The capture runner's own seams, which only the device would otherwise exercise.

`scripts/state_capture.py` is a script, so its `main` needs a model and a card and is not unit
tested. `emit_shard` is not: it is a small adapter between the writer's progress callback and the
run's event log, and it is the one seam in that file a test can reach. It had none, which is the
pattern the method record's thirty-fourth entry is about — the suite passed because nothing reached
the code, and the first caller would have been the capture pass on the card.

The adapter is imported by path rather than by package, because `scripts/` is not importable as one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "state_capture.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("state_capture_script", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubCuda:
    def __init__(self, peak_bytes: int, reserved: int | None = None,
                 free: int = 60 * 2**30, total: int = 96 * 2**30) -> None:
        self._peak = peak_bytes
        self._reserved = peak_bytes if reserved is None else reserved
        self._free, self._total = free, total

    def max_memory_allocated(self) -> int:
        return self._peak

    def max_memory_reserved(self) -> int:
        return self._reserved

    def mem_get_info(self) -> tuple[int, int]:
        return self._free, self._total


class _StubTorch:
    def __init__(self, peak_bytes: int, **kwargs) -> None:
        self.cuda = _StubCuda(peak_bytes, **kwargs)


def test_emit_shard_puts_the_device_peak_beside_the_writer_s_fields(runner) -> None:
    """The peak has to arrive with the first shard, because it informs a decision made before the end.

    Two capture passes share the card, and whether the second may start depends on the first's
    measured peak. A memory figure that arrives with the summary arrives after the decision it
    informs, so this adapter exists to put it on every shard event — and nothing called it.
    """
    seen: list[tuple[str, dict]] = []

    def emit(event: str, /, **fields) -> None:
        seen.append((event, fields))

    progress = runner.emit_shard(emit, _StubTorch(24 * 2**30))
    progress({"event": "shard", "shard": 0, "captured": 256, "written_this_pass": 256,
              "reused": 0, "requested": 7629})

    assert len(seen) == 1
    event, fields = seen[0]
    assert event == "shard"
    assert fields["peak_allocated_gib"] == 24.0
    assert fields["peak_reserved_gib"] == 24.0
    assert fields["device_used_gib"] == 36.0 and fields["device_free_gib"] == 60.0
    # The writer's own fields survive the adapter: it adds, it does not replace.
    assert fields["shard"] == 0 and fields["captured"] == 256 and fields["requested"] == 7629
    assert "event" not in fields, "the event name is the positional argument, never a keyword"


def test_emit_shard_defaults_the_event_name_rather_than_raising(runner) -> None:
    """A progress row without an `event` key must still be reported, not lose the shard's record."""
    seen: list[tuple[str, dict]] = []
    runner.emit_shard(lambda event, /, **fields: seen.append((event, fields)),
                      _StubTorch(2**30))({"shard": 3})
    assert seen[0][0] == "shard" and seen[0][1]["shard"] == 3


def test_the_reported_cost_is_reserved_and_not_only_allocated(runner) -> None:
    """The gap that decided an out-of-memory: reserved-unallocated is real memory the card cannot lend.

    The Chief's capture held 22.12 GiB allocated and 29.30 of process memory, with 6.52 reserved and
    unhanded-out; c3 showed 62.25 against 59.72. A headroom rule read on the allocator's high-water
    mark plans with a number 2.5 GiB smaller than the one that matters.
    """
    report = runner.memory(_StubTorch(20 * 2**30, reserved=27 * 2**30))
    assert report["peak_allocated_gib"] == 20.0
    assert report["peak_reserved_gib"] == 27.0, "reserved is what the card cannot lend elsewhere"
