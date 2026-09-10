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
    def __init__(self, peak_bytes: int) -> None:
        self._peak = peak_bytes

    def max_memory_allocated(self) -> int:
        return self._peak


class _StubTorch:
    def __init__(self, peak_bytes: int) -> None:
        self.cuda = _StubCuda(peak_bytes)


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
    assert fields["peak_gib"] == 24.0
    # The writer's own fields survive the adapter: it adds, it does not replace.
    assert fields["shard"] == 0 and fields["captured"] == 256 and fields["requested"] == 7629
    assert "event" not in fields, "the event name is the positional argument, never a keyword"


def test_emit_shard_defaults_the_event_name_rather_than_raising(runner) -> None:
    """A progress row without an `event` key must still be reported, not lose the shard's record."""
    seen: list[tuple[str, dict]] = []
    runner.emit_shard(lambda event, /, **fields: seen.append((event, fields)),
                      _StubTorch(2**30))({"shard": 3})
    assert seen == [("shard", {"peak_gib": 1.0, "shard": 3})]
