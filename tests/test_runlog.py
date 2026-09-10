"""Tests for :mod:`local_llm_lab.runlog`.

Fakes only (R10): every stream is an ``io.StringIO`` and the clock is a list-backed
counter, so elapsed values, ETAs and line text are exact rather than approximate. No model
is loaded and nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from pathlib import Path

import pytest

from local_llm_lab.runlog import (
    HealthThresholds,
    RunLog,
    Tee,
    TrainingAborted,
    TrainingHealth,
    gib_from_gb,
    git_commit,
    sha256_of,
)

BAR = "│"
LINE = re.compile(r"^(\d{2}:\d{2}:\d{2}) \+(\S+) (\S+) │ (.*)$")


class _Clock:
    """List-backed counter; holds the final value once the list is exhausted."""

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.calls = 0

    def __call__(self) -> float:
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


class _FakeStream:
    """A stream with only the attributes a caller opts into."""

    def __init__(self, **attributes: object) -> None:
        self.buffer: list[str] = []
        self.flushes = 0
        for key, value in attributes.items():
            setattr(self, key, value)

    def write(self, value: str) -> int:
        self.buffer.append(value)
        return len(value)

    def flush(self) -> None:
        self.flushes += 1


def _open(
    tmp_path: Path,
    ticks: list[float],
    **kwargs: object,
) -> tuple[RunLog, io.StringIO, io.StringIO]:
    out, err = io.StringIO(), io.StringIO()
    log = RunLog.open(
        tmp_path / "run",
        name="train",
        stdout=out,
        stderr=err,
        clock=_Clock(ticks),
        **kwargs,  # type: ignore[arg-type]
    )
    return log, out, err


def _events(log: RunLog) -> list[dict]:
    text = log.paths["events"].read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _bodies(lines: list[str]) -> list[str]:
    parsed = [LINE.match(line) for line in lines]
    assert all(parsed), lines
    return [match.group(4) for match in parsed if match]


# ---------------------------------------------------------------------------
# files and append mode
# ---------------------------------------------------------------------------


def test_open_creates_both_files_in_a_fresh_nested_directory(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "run"
    log = RunLog.open(target, name="probe", stdout=io.StringIO(), stderr=io.StringIO())
    try:
        assert log.paths == {"run_log": target / "run.log", "events": target / "events.jsonl"}
        assert log.paths["run_log"].is_file()
        assert log.paths["events"].is_file()
        assert log.name == "probe"
    finally:
        log.close()


def test_second_open_appends_and_keeps_both_start_events(tmp_path: Path) -> None:
    first, _, _ = _open(tmp_path, [0.0])
    first.info("one")
    first.close()
    second, _, _ = _open(tmp_path, [0.0])
    second.info("two")
    second.close()

    kinds = [event["kind"] for event in _events(second)]
    assert kinds == ["start", "info", "end", "start", "info", "end"]
    text = second.paths["run_log"].read_text(encoding="utf-8")
    assert text.count(f"{BAR} start") == 2
    assert f"{BAR} one" in text and f"{BAR} two" in text


# ---------------------------------------------------------------------------
# console line format
# ---------------------------------------------------------------------------


def test_start_line_format(tmp_path: Path) -> None:
    log, out, _ = _open(tmp_path, [0.0], command=["agent-pipeline", "train", "--iters", "400"])
    try:
        line = out.getvalue().splitlines()[0]
        expected = (
            f"start command=agent-pipeline train --iters 400 "
            f"log={log.paths['run_log']}"
        )
        match = LINE.match(line)
        assert match is not None, line
        assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", match.group(1))
        assert match.group(2) == "00:00"
        assert match.group(3) == "train"
        assert match.group(4) == expected
        assert line == f"{match.group(1)} +00:00 train {BAR} {expected}"
    finally:
        log.close()


def test_start_line_without_a_command_renders_a_dash(tmp_path: Path) -> None:
    log, out, _ = _open(tmp_path, [0.0])
    try:
        assert _bodies(out.getvalue().splitlines())[0].startswith("start command=- log=")
        assert _events(log)[0]["command"] is None
    finally:
        log.close()


def test_elapsed_prefix_rolls_over_past_an_hour(tmp_path: Path) -> None:
    log, out, _ = _open(tmp_path, [0.0, 5.0, 59.0, 60.0, 3599.0, 3600.0, 3661.0, 36000.0])
    try:
        for index in range(7):
            log.info(f"m{index}")
        prefixes = [LINE.match(line).group(2) for line in out.getvalue().splitlines()[1:]]  # type: ignore[union-attr]
        assert prefixes == [
            "00:05",
            "00:59",
            "01:00",
            "59:59",
            "1:00:00",
            "1:01:01",
            "10:00:00",
        ]
    finally:
        log.close()


def test_stdout_and_stderr_routing(tmp_path: Path) -> None:
    log, out, err = _open(tmp_path, [0.0])
    try:
        log.info("an info")
        log.warn("a warning")
        log.metric("a_metric", value=1)
        log.error("an error")
        log.progress(1, 2, "step")
        assert _bodies(out.getvalue().splitlines()) == [
            f"start command=- log={log.paths['run_log']}",
            "an info",
            "a_metric value=1",
            "[step 1/2 50%] eta 00:00",
        ]
        assert _bodies(err.getvalue().splitlines()) == ["a warning", "an error"]
    finally:
        log.close()


def test_run_log_holds_exactly_the_console_lines_in_order(tmp_path: Path) -> None:
    log, out, err = _open(tmp_path, [0.0])
    log.info("an info")
    log.warn("a warning")
    log.error("an error")
    log.close("ok", tokens=5)

    written = log.paths["run_log"].read_text(encoding="utf-8")
    assert written.endswith("\n") and "\n\n" not in written
    assert _bodies(written.splitlines()) == [
        f"start command=- log={log.paths['run_log']}",
        "an info",
        "a warning",
        "an error",
        "end status=ok elapsed=00:00 tokens=5",
    ]
    console = out.getvalue().splitlines() + err.getvalue().splitlines()
    assert set(console) == set(written.splitlines())


def test_end_line_format(tmp_path: Path) -> None:
    log, out, _ = _open(tmp_path, [0.0, 125.0])
    log.close("error", reason="killed", tokens=12)
    assert _bodies(out.getvalue().splitlines())[-1] == (
        "end status=error elapsed=02:05 reason=killed tokens=12"
    )


# ---------------------------------------------------------------------------
# field formatting
# ---------------------------------------------------------------------------


def test_field_formatting_rules(tmp_path: Path) -> None:
    log, out, _ = _open(tmp_path, [0.0])
    try:
        log.info(
            "fields",
            an_int=7,
            a_float=0.123456789,
            big_float=1234567.0,
            a_bool=True,
            off=False,
            missing=None,
            text="raw text",
            listed=[1, 2, 3],
            mapped={"a": 1, "b": "two"},
        )
        assert _bodies(out.getvalue().splitlines())[1] == (
            "fields an_int=7 a_float=0.1235 big_float=1.235e+06 a_bool=true off=false "
            'missing=- text=raw text listed=[1,2,3] mapped={"a":1,"b":"two"}'
        )
    finally:
        log.close()


def test_field_order_follows_kwargs_order(tmp_path: Path) -> None:
    log, out, _ = _open(tmp_path, [0.0])
    try:
        log.metric("m", zulu=1, alpha=2, mike=3)
        # the console keeps kwargs order; events.jsonl is written with sort_keys, which
        # sorts the nested fields object too, so the two orders differ by design
        assert _bodies(out.getvalue().splitlines())[1] == "m zulu=1 alpha=2 mike=3"
        assert list(_events(log)[1]["fields"]) == ["alpha", "mike", "zulu"]
        assert _events(log)[1]["fields"] == {"zulu": 1, "alpha": 2, "mike": 3}
    finally:
        log.close()


def test_non_json_safe_values_are_stringified_and_never_raise(tmp_path: Path) -> None:
    class _Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    log, out, _ = _open(tmp_path, [0.0])
    try:
        log.info("odd", where=tmp_path, tags={"a"}, thing=_Opaque())
        event = _events(log)[1]
        assert event["fields"]["where"] == str(tmp_path)
        assert event["fields"]["thing"] == "<opaque>"
        assert event["fields"]["tags"] == "{'a'}"
        assert "thing=<opaque>" in out.getvalue()
    finally:
        log.close()


# ---------------------------------------------------------------------------
# events.jsonl
# ---------------------------------------------------------------------------


def test_event_lines_are_sorted_json_with_the_documented_keys(tmp_path: Path) -> None:
    log, _, _ = _open(tmp_path, [0.0, 30.0])
    log.info("hello", a=1)
    log.close()

    raw = log.paths["events"].read_text(encoding="utf-8").splitlines()
    for line in raw:
        event = json.loads(line)
        assert list(event) == sorted(event), "events.jsonl must be written with sort_keys"
        assert {"ts", "elapsed", "run", "kind", "message", "fields"} <= set(event)
        assert event["run"] == "train"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T[\d:.]+Z", event["ts"])
    info = json.loads(raw[1])
    assert info == {
        "elapsed": 30.0,
        "fields": {"a": 1},
        "kind": "info",
        "message": "hello",
        "run": "train",
        "ts": info["ts"],
    }


def test_event_kinds_per_method(tmp_path: Path) -> None:
    log, _, _ = _open(tmp_path, [0.0])
    log.info("i")
    log.warn("w")
    log.error("e")
    log.metric("m", v=1)
    log.progress(1, 2, "label")
    log.close()
    events = _events(log)
    assert [event["kind"] for event in events] == [
        "start",
        "info",
        "warning",
        "error",
        "metric",
        "progress",
        "end",
    ]
    assert events[4]["message"] == "m"
    assert events[5]["message"] is None


def test_start_event_extras(tmp_path: Path) -> None:
    log, _, _ = _open(tmp_path, [0.0], command=["a", "b"])
    try:
        start = _events(log)[0]
        assert start["kind"] == "start"
        assert start["message"] == "start"
        assert start["elapsed"] == 0.0
        assert start["command"] == ["a", "b"]
        assert start["run"] == "train"
        assert isinstance(start["pid"], int) and start["pid"] > 0
        assert start["cwd"] == str(Path.cwd())
        assert re.fullmatch(r"\d+\.\d+\.\d+", start["python"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T[\d:.]+Z", start["started_at"])
        assert start["fields"] == {}
    finally:
        log.close()


def test_end_event_extras(tmp_path: Path) -> None:
    log, _, _ = _open(tmp_path, [0.0, 42.5])
    log.close("error", reason="killed")
    end = _events(log)[-1]
    assert end["kind"] == "end"
    assert end["message"] == "end"
    assert end["status"] == "error"
    assert end["elapsed"] == 42.5
    assert end["fields"] == {"reason": "killed"}


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------


def test_progress_fraction_and_eta_are_exact(tmp_path: Path) -> None:
    log, out, _ = _open(tmp_path, [0.0, 20.0])
    try:
        log.progress(3, 12, "layer", loss=0.5)
        event = _events(log)[1]
        assert event["step"] == 3
        assert event["total"] == 12
        assert event["fraction"] == 0.25
        assert event["eta_seconds"] == 60.0
        assert event["label"] == "layer"
        assert event["fields"] == {"loss": 0.5}
        assert _bodies(out.getvalue().splitlines())[1] == "[layer 3/12 25%] loss=0.5 eta 01:00"
    finally:
        log.close()


def test_progress_without_an_eta(tmp_path: Path) -> None:
    log, out, _ = _open(tmp_path, [0.0, 10.0, 10.0])
    try:
        log.progress(0, 12, "layer")
        log.progress(3, 0, "layer")
        first, second = _events(log)[1], _events(log)[2]
        assert first["eta_seconds"] is None and first["fraction"] == 0.0
        assert second["eta_seconds"] is None and second["fraction"] == 0.0
        bodies = _bodies(out.getvalue().splitlines())
        assert bodies[1] == "[layer 0/12 0%]"
        assert bodies[2] == "[layer 3/0 0%]"
    finally:
        log.close()


# ---------------------------------------------------------------------------
# identity (R26(e), issue #35)
# ---------------------------------------------------------------------------


def test_identity_reaches_the_start_event_and_line(tmp_path: Path) -> None:
    log, out, _ = _open(
        tmp_path,
        [0.0],
        command=["train"],
        identity={"model": "Qwen3.5-4B", "commit": "edb2e72", "config_sha": "abc123"},
    )
    try:
        body = _bodies(out.getvalue().splitlines())[0]
        assert body.endswith("model=Qwen3.5-4B commit=edb2e72 config_sha=abc123")
        assert _events(log)[0]["fields"] == {
            "model": "Qwen3.5-4B",
            "commit": "edb2e72",
            "config_sha": "abc123",
        }
    finally:
        log.close()


def test_identity_none_behaves_as_before(tmp_path: Path) -> None:
    log, out, _ = _open(tmp_path, [0.0], command=["train"], identity=None)
    try:
        assert _bodies(out.getvalue().splitlines())[0] == (
            f"start command=train log={log.paths['run_log']}"
        )
        assert _events(log)[0]["fields"] == {}
    finally:
        log.close()


# ---------------------------------------------------------------------------
# Tee
# ---------------------------------------------------------------------------


def test_tee_writes_to_both_and_flushes_both(tmp_path: Path) -> None:
    primary = _FakeStream()
    secondary = _FakeStream()
    tee = Tee(primary, secondary)
    assert tee.write("hello\n") == len("hello\n")
    assert primary.buffer == ["hello\n"] and secondary.buffer == ["hello\n"]
    assert primary.flushes == 1 and secondary.flushes == 1
    tee.flush()
    assert primary.flushes == 2 and secondary.flushes == 2


def test_runlog_tee_mirrors_into_run_log_on_disk(tmp_path: Path) -> None:
    log, _, _ = _open(tmp_path, [0.0])
    try:
        console = io.StringIO()
        stream = log.tee(console)
        stream.write("third-party line\n")
        assert console.getvalue() == "third-party line\n"
        # read from disk: proves the write was flushed, not just buffered
        assert log.paths["run_log"].read_text(encoding="utf-8").endswith("third-party line\n")
    finally:
        log.close()


def test_tee_proxies_isatty_fileno_and_encoding() -> None:
    assert Tee(io.StringIO(), io.StringIO()).isatty() is False
    assert Tee(_FakeStream(isatty=lambda: True), io.StringIO()).isatty() is True
    assert Tee(_FakeStream(), io.StringIO()).isatty() is False

    assert Tee(io.StringIO(), io.StringIO()).encoding is None
    assert Tee(_FakeStream(encoding="utf-8"), io.StringIO()).encoding == "utf-8"

    with pytest.raises(io.UnsupportedOperation):
        Tee(io.StringIO(), io.StringIO()).fileno()
    with pytest.raises(io.UnsupportedOperation):
        Tee(_FakeStream(), io.StringIO()).fileno()
    assert Tee(_FakeStream(fileno=lambda: 7), io.StringIO()).fileno() == 7


# ---------------------------------------------------------------------------
# close and the context manager
# ---------------------------------------------------------------------------


def test_close_is_idempotent(tmp_path: Path) -> None:
    log, out, _ = _open(tmp_path, [0.0])
    log.close("ok", n=1)
    log.close("error", n=2)
    log.info("after close")
    kinds = [event["kind"] for event in _events(log)]
    assert kinds == ["start", "end"]
    assert _events(log)[-1]["status"] == "ok"
    assert out.getvalue().count(f"{BAR} end") == 1
    assert "after close" not in out.getvalue()


def test_context_manager_closes_ok(tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    with RunLog.open(
        tmp_path / "run", name="train", stdout=out, stderr=err, clock=_Clock([0.0])
    ) as log:
        log.info("body")
        paths = log.paths
    events = [json.loads(line) for line in paths["events"].read_text().splitlines()]
    assert events[-1]["kind"] == "end" and events[-1]["status"] == "ok"


def test_context_manager_records_the_error_and_propagates(tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(ValueError, match="boom"), RunLog.open(
        tmp_path / "run", name="train", stdout=out, stderr=err, clock=_Clock([0.0])
    ) as log:
        paths = log.paths
        raise ValueError("boom")
    end = json.loads(paths["events"].read_text().splitlines()[-1])
    assert end["kind"] == "end"
    assert end["status"] == "error"
    assert end["fields"] == {"error": "ValueError: boom"}
    assert "end status=error" in out.getvalue()


def test_elapsed_property_uses_the_injected_clock(tmp_path: Path) -> None:
    log, _, _ = _open(tmp_path, [10.0, 25.0])
    try:
        assert log.elapsed == 15.0
    finally:
        log.close()


# ---------------------------------------------------------------------------
# HealthThresholds
# ---------------------------------------------------------------------------


def test_thresholds_defaults_and_as_dict() -> None:
    thresholds = HealthThresholds.from_config(None, memory_budget_gib=None)
    assert thresholds.as_dict() == {
        "loss_spike_factor": 2.0,
        "loss_window": 5,
        "val_rising_reports": 3,
        "throughput_drop_factor": 0.5,
        "memory_budget_gib": None,
    }
    json.dumps(thresholds.as_dict())


def test_thresholds_take_the_registry_budget_when_the_block_is_silent() -> None:
    assert HealthThresholds.from_config(None, memory_budget_gib=18.0).memory_budget_gib == 18.0
    assert (
        HealthThresholds.from_config({"loss_window": 9}, memory_budget_gib=18.0).memory_budget_gib
        == 18.0
    )


def test_thresholds_block_overrides_the_registry_budget() -> None:
    explicit = HealthThresholds.from_config({"memory_budget_gib": 9.5}, memory_budget_gib=18.0)
    assert explicit.memory_budget_gib == 9.5
    disabled = HealthThresholds.from_config({"memory_budget_gib": None}, memory_budget_gib=18.0)
    assert disabled.memory_budget_gib is None


def test_thresholds_valid_overrides() -> None:
    thresholds = HealthThresholds.from_config(
        {
            "loss_spike_factor": 3.5,
            "loss_window": 8,
            "val_rising_reports": 4,
            "throughput_drop_factor": 0.25,
        },
        memory_budget_gib=None,
    )
    assert thresholds.as_dict() == {
        "loss_spike_factor": 3.5,
        "loss_window": 8,
        "val_rising_reports": 4,
        "throughput_drop_factor": 0.25,
        "memory_budget_gib": None,
    }


def test_thresholds_unknown_key_names_the_key() -> None:
    with pytest.raises(ValueError, match="loss_windows"):
        HealthThresholds.from_config({"loss_windows": 5}, memory_budget_gib=None)


@pytest.mark.parametrize(
    "block",
    [
        {"loss_spike_factor": 0},
        {"loss_spike_factor": -1.0},
        {"throughput_drop_factor": 0.0},
        {"loss_window": 1},
        {"val_rising_reports": 1},
        {"loss_window": 2.5},
        {"loss_spike_factor": "big"},
        {"memory_budget_gib": 0},
    ],
)
def test_thresholds_reject_invalid_values(block: dict) -> None:
    with pytest.raises(ValueError):
        HealthThresholds.from_config(block, memory_budget_gib=None)


# ---------------------------------------------------------------------------
# TrainingHealth
# ---------------------------------------------------------------------------


def _health(**overrides: object) -> TrainingHealth:
    iters = overrides.pop("iters", 100)
    thresholds = HealthThresholds(**overrides)  # type: ignore[arg-type]
    return TrainingHealth(thresholds, iters=iters)  # type: ignore[arg-type]


def _train(
    health: TrainingHealth,
    iteration: int,
    *,
    loss: float = 1.0,
    tps: float = 100.0,
    memory: float = 4.0,
) -> list[str]:
    return health.on_train_report(
        iteration=iteration,
        train_loss=loss,
        learning_rate=1e-4,
        tokens_per_second=tps,
        trained_tokens=iteration * 1000,
        peak_memory_gb=memory,
    )


def test_flags_tuple_lists_every_rule() -> None:
    assert TrainingHealth.FLAGS == (
        "non_finite_loss",
        "loss_spike",
        "val_loss_rising",
        "throughput_drop",
        "memory_over_budget",
        "incomplete_run",
    )


def test_healthy_sequence_raises_nothing() -> None:
    health = _health()
    for iteration in range(1, 11):
        assert _train(health, iteration) == []
    for iteration in (2, 4, 6):
        assert health.on_val_report(iteration=iteration, val_loss=1.0 / iteration) == []
    assert health.flags == ()
    assert health.verdict == "healthy"


def test_loss_spike_boundary_is_strict() -> None:
    at_boundary = _health()
    for iteration in range(1, 6):
        _train(at_boundary, iteration)
    assert _train(at_boundary, 6, loss=2.0) == []
    assert at_boundary.verdict == "healthy"

    above = _health()
    for iteration in range(1, 6):
        _train(above, iteration)
    assert _train(above, 6, loss=2.5) == ["loss_spike"]
    flag = above.flags[0]
    assert flag["flag"] == "loss_spike"
    assert flag["severity"] == "warning"
    assert flag["iteration"] == 6
    assert flag["detail"] == {
        "train_loss": 2.5,
        "median": 1.0,
        "threshold": 2.0,
        "window": 5,
    }
    assert above.verdict == "warnings"


def test_loss_spike_is_not_evaluated_before_the_window_is_full() -> None:
    health = _health()
    for iteration in range(1, 5):  # four prior reports only
        _train(health, iteration)
    assert _train(health, 5, loss=100.0) == []
    assert health.flags == ()
    # the sixth report now has five priors and is compared against their median
    assert _train(health, 6, loss=100.0) == ["loss_spike"]


def test_throughput_drop_boundary_is_strict() -> None:
    at_boundary = _health()
    for iteration in range(1, 6):
        _train(at_boundary, iteration, tps=100.0)
    assert _train(at_boundary, 6, tps=50.0) == []

    below = _health()
    for iteration in range(1, 6):
        _train(below, iteration, tps=100.0)
    assert _train(below, 6, tps=49.0) == ["throughput_drop"]
    assert below.flags[0]["detail"] == {
        "tokens_per_second": 49.0,
        "median": 100.0,
        "threshold": 50.0,
        "window": 5,
    }


def test_memory_over_budget_fires_once_per_run() -> None:
    health = _health(memory_budget_gib=10.0)
    assert _train(health, 1, memory=9.5) == []
    assert _train(health, 2, memory=12.0) == ["memory_over_budget"]
    assert _train(health, 3, memory=13.0) == []
    assert [flag["flag"] for flag in health.flags] == ["memory_over_budget"]
    # Three keys, not two: both units are recorded so a reader checks the conversion rather
    # than trusting it (issue 93). 12.0 GB is 11.176 GiB, still over a 10 GiB budget.
    assert health.flags[0]["detail"] == {
        "peak_memory_gb": 12.0,
        "peak_memory_gib": pytest.approx(11.175870895385742),
        "budget_gib": 10.0,
    }
    assert health.flags[0]["iteration"] == 2


def test_memory_budget_none_disables_the_rule() -> None:
    health = _health(memory_budget_gib=None)
    assert _train(health, 1, memory=999.0) == []
    assert health.verdict == "healthy"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_train_loss_aborts(bad: float) -> None:
    health = _health()
    _train(health, 1)
    with pytest.raises(TrainingAborted) as excinfo:
        _train(health, 2, loss=bad)
    flag = excinfo.value.flag
    assert flag["flag"] == "non_finite_loss"
    assert flag["severity"] == "fatal"
    assert flag["iteration"] == 2
    assert flag["detail"] == {"train_loss": repr(bad)}
    assert health.verdict == "aborted"
    summary = health.summary(elapsed=12.0, status="aborted")
    assert summary["verdict"] == "aborted"
    assert summary["status"] == "aborted"
    json.dumps(summary)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_val_loss_aborts(bad: float) -> None:
    health = _health()
    health.on_val_report(iteration=10, val_loss=1.0)
    with pytest.raises(TrainingAborted) as excinfo:
        health.on_val_report(iteration=20, val_loss=bad)
    assert excinfo.value.flag["detail"] == {"val_loss": repr(bad)}
    assert excinfo.value.flag["severity"] == "fatal"
    assert health.verdict == "aborted"
    json.dumps(health.summary(elapsed=1.0, status="aborted"))


def test_val_loss_rising_flags_once_per_streak_and_again_after_a_decrease() -> None:
    health = _health()
    assert health.on_val_report(iteration=1, val_loss=1.0) == []
    assert health.on_val_report(iteration=2, val_loss=2.0) == []
    assert health.on_val_report(iteration=3, val_loss=3.0) == ["val_loss_rising"]
    assert health.on_val_report(iteration=4, val_loss=4.0) == []  # same streak
    assert health.on_val_report(iteration=5, val_loss=2.0) == []  # decrease breaks it
    assert health.on_val_report(iteration=6, val_loss=3.0) == []
    assert health.on_val_report(iteration=7, val_loss=4.0) == ["val_loss_rising"]
    flags = [flag for flag in health.flags if flag["flag"] == "val_loss_rising"]
    assert [flag["iteration"] for flag in flags] == [3, 7]
    assert flags[0]["detail"] == {"val_losses": [1.0, 2.0, 3.0], "reports": 3}
    assert health.verdict == "warnings"


def test_val_loss_equal_does_not_count_as_rising() -> None:
    health = _health()
    for iteration, loss in enumerate([1.0, 2.0, 2.0, 3.0], start=1):
        assert health.on_val_report(iteration=iteration, val_loss=loss) == []
    assert health.verdict == "healthy"


# ---------------------------------------------------------------------------
# on_finish / incomplete_run (R26(c), issue #35)
# ---------------------------------------------------------------------------


def test_on_finish_complete_run_stays_healthy() -> None:
    health = _health(iters=10)
    for iteration in range(1, 11):
        _train(health, iteration)
    assert health.on_finish(iterations_done=10, final_checkpoint=True) == []
    assert health.verdict == "healthy"
    summary = health.summary(elapsed=1.0, status="ok")
    assert summary["iterations_done"] == 10
    assert summary["final_checkpoint"] is True


def test_on_finish_short_run_is_incomplete_even_with_no_warnings() -> None:
    health = _health(iters=10)
    for iteration in range(1, 8):
        _train(health, iteration)
    assert health.on_finish(iterations_done=7, final_checkpoint=True) == ["incomplete_run"]
    assert health.verdict == "incomplete"
    flag = health.flags[-1]
    assert flag["flag"] == "incomplete_run"
    assert flag["severity"] == "fatal"
    assert flag["iteration"] == 7
    assert flag["detail"] == {
        "iterations_done": 7,
        "iters_planned": 10,
        "final_checkpoint": True,
    }
    summary = health.summary(elapsed=2.0, status="ok")
    assert summary["verdict"] == "incomplete"
    assert summary["iterations_done"] == 7
    assert summary["final_checkpoint"] is True
    json.dumps(summary)


def test_on_finish_missing_final_checkpoint_is_incomplete() -> None:
    health = _health(iters=10)
    _train(health, 10)
    assert health.on_finish(iterations_done=10, final_checkpoint=False) == ["incomplete_run"]
    assert health.verdict == "incomplete"
    assert health.flags[-1]["detail"]["final_checkpoint"] is False


def test_on_finish_without_a_planned_count_only_checks_the_checkpoint() -> None:
    health = _health(iters=None)
    _train(health, 5)
    assert health.on_finish(iterations_done=5, final_checkpoint=True) == []
    assert health.verdict == "healthy"
    assert health.summary(elapsed=1.0, status="ok")["iters_planned"] is None


def test_on_finish_does_not_raise_and_verdict_precedence_keeps_aborted() -> None:
    health = _health(iters=10)
    with pytest.raises(TrainingAborted):
        _train(health, 1, loss=float("nan"))
    assert health.on_finish(iterations_done=1, final_checkpoint=False) == ["incomplete_run"]
    assert health.verdict == "aborted"


def test_on_finish_is_idempotent_so_every_exit_path_may_call_it() -> None:
    """The training stage calls it on the normal path and again from its ``finally``."""
    health = _health(iters=10)
    _train(health, 3)

    assert health.on_finish(iterations_done=3, final_checkpoint=False) == ["incomplete_run"]
    assert health.on_finish(iterations_done=9, final_checkpoint=True) == []

    assert [flag["flag"] for flag in health.flags] == ["incomplete_run"]
    summary = health.summary(elapsed=1.0, status="error")
    assert summary["iterations_done"] == 3 and summary["final_checkpoint"] is False


def test_verdict_precedence_incomplete_outranks_warnings() -> None:
    health = _health(iters=10, memory_budget_gib=1.0)
    _train(health, 1, memory=99.0)
    assert health.verdict == "warnings"
    health.on_finish(iterations_done=1, final_checkpoint=True)
    assert health.verdict == "incomplete"


def test_summary_without_any_report_is_empty_but_well_formed() -> None:
    health = _health(iters=None)
    summary = health.summary(elapsed=0.0, status="ok")
    assert summary["iters_done"] == 0
    assert summary["iterations_done"] == 0
    assert summary["final_checkpoint"] is None
    assert summary["last_train"] is None
    assert summary["last_val"] is None
    assert summary["best_val"] is None
    assert summary["peak_memory_gb"] is None
    assert summary["median_tokens_per_second"] is None
    assert summary["verdict"] == "healthy"
    json.dumps(summary)


def test_summary_full_shape() -> None:
    health = _health(iters=40, memory_budget_gib=20.0)
    _train(health, 10, loss=2.0, tps=100.0, memory=4.0)
    _train(health, 20, loss=1.5, tps=200.0, memory=6.0)
    _train(health, 30, loss=1.0, tps=300.0, memory=5.0)
    health.on_val_report(iteration=10, val_loss=3.0)
    health.on_val_report(iteration=20, val_loss=1.0)
    health.on_val_report(iteration=30, val_loss=2.0)

    summary = health.summary(elapsed=123.5, status="ok")
    assert set(summary) == {
        "verdict",
        "status",
        "flags",
        "thresholds",
        "iters_planned",
        "iters_done",
        "iterations_done",
        "final_checkpoint",
        "train_reports",
        "val_reports",
        "last_train",
        "last_val",
        "best_val",
        "peak_memory_gb",
        "peak_memory_gib",
        "median_tokens_per_second",
        "elapsed",
    }
    assert summary["verdict"] == "healthy"
    assert summary["status"] == "ok"
    assert summary["flags"] == []
    assert summary["thresholds"]["memory_budget_gib"] == 20.0
    assert summary["iters_planned"] == 40
    assert summary["iters_done"] == 30
    assert summary["train_reports"] == 3
    assert summary["val_reports"] == 3
    assert summary["last_train"] == {
        "iteration": 30,
        "train_loss": 1.0,
        "learning_rate": 1e-4,
        "tokens_per_second": 300.0,
        "trained_tokens": 30000,
        "peak_memory_gb": 5.0,
        # Beside it, not instead of it, so every reader of the old field keeps working (issue 93).
        "peak_memory_gib": pytest.approx(4.656612873077393),
    }
    assert summary["last_val"] == {"iteration": 30, "val_loss": 2.0}
    assert summary["best_val"] == {"iteration": 20, "val_loss": 1.0}
    assert summary["peak_memory_gb"] == 6.0
    assert summary["peak_memory_gib"] == pytest.approx(5.587935447692871)
    assert summary["median_tokens_per_second"] == 200.0
    assert summary["elapsed"] == 123.5
    assert json.loads(json.dumps(summary)) == summary


def test_summary_flags_are_copies() -> None:
    health = _health(memory_budget_gib=1.0)
    _train(health, 1, memory=5.0)
    summary = health.summary(elapsed=1.0, status="ok")
    summary["flags"][0]["detail"]["peak_memory_gb"] = 0.0
    assert health.flags[0]["detail"]["peak_memory_gb"] == 5.0


def test_summary_is_strict_json_after_an_abort() -> None:
    health = _health()
    with pytest.raises(TrainingAborted):
        _train(health, 3, loss=float("nan"))
    summary = health.summary(elapsed=1.0, status="aborted")
    assert summary["last_train"]["train_loss"] == "nan"
    assert "NaN" not in json.dumps(summary)
    assert math.isfinite(summary["elapsed"])


# ---------------------------------------------------------------------------
# provenance helpers (R26, issue #35)
# ---------------------------------------------------------------------------


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    payload = b"the note is the only cross-turn memory\n" * 3
    target = tmp_path / "sample.bin"
    target.write_bytes(payload)
    assert sha256_of(target) == hashlib.sha256(payload).hexdigest()


def test_sha256_of_an_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.bin"
    target.write_bytes(b"")
    assert sha256_of(target) == hashlib.sha256(b"").hexdigest()


def test_sha256_of_spans_multiple_chunks(tmp_path: Path) -> None:
    payload = b"x" * ((1 << 20) + 17)
    target = tmp_path / "big.bin"
    target.write_bytes(payload)
    assert sha256_of(target) == hashlib.sha256(payload).hexdigest()


def test_git_commit_in_this_repository() -> None:
    repo = Path(__file__).resolve().parent.parent
    assert re.fullmatch(r"[0-9a-f]{40}", git_commit(repo))


def test_git_commit_outside_a_repository_is_unknown(tmp_path: Path) -> None:
    assert git_commit(tmp_path) == "unknown"


def test_git_commit_never_raises_on_a_missing_directory(tmp_path: Path) -> None:
    assert git_commit(tmp_path / "does-not-exist") == "unknown"


def test_events_stay_strict_json_when_a_logged_field_is_not_finite(tmp_path):
    """A NaN field must never become a bare ``NaN`` token in events.jsonl (Deputy review)."""
    from local_llm_lab.runlog import RunLog

    out, err = io.StringIO(), io.StringIO()
    with RunLog.open(tmp_path / "run", name="t", stdout=out, stderr=err, clock=lambda: 0.0) as log:
        log.info("nan field", loss=float("nan"), nested={"peak": float("inf")}, values=[1.0, float("-inf")])
    lines = log.paths["events"].read_text().splitlines()

    def refuse(constant: str) -> None:
        raise AssertionError(f"bare {constant} token in events.jsonl")

    parsed = [json.loads(line, parse_constant=refuse) for line in lines]
    event = next(item for item in parsed if item["kind"] == "info")
    assert event["fields"]["loss"] == "nan"
    assert event["fields"]["nested"]["peak"] == "inf"
    assert event["fields"]["values"] == [1.0, "-inf"]
    assert "NaN" not in "\n".join(lines) and "Infinity" not in "\n".join(lines)


def test_the_memory_flag_compares_gibibytes_at_the_boundary_the_hardware_has() -> None:
    """Issue 93: the trainer reports gigabytes and every budget here is gibibytes.

    The two cases are chosen at the boundary that matters on this machine, a 17.76 GiB
    recommended working set. Under the old comparison a 17.9 figure read as over budget when the
    run was using 16.7 GiB and had 1 GiB of headroom left, and 19.5 read as under it when the run
    was using 18.2 GiB and past the ceiling. One is a false alarm and the other is the miss, and
    the miss is the one that costs a run.
    """
    quiet = _health(memory_budget_gib=17.76)
    assert _train(quiet, 1, memory=17.9) == [], "17.9 GB is 16.7 GiB and fits"
    assert list(quiet.flags) == []

    loud = _health(memory_budget_gib=17.76)
    assert _train(loud, 1, memory=19.5) == [
        "memory_over_budget"
    ], "19.5 GB is 18.2 GiB and does not"
    detail = loud.flags[0]["detail"]
    assert detail["peak_memory_gb"] == 19.5
    assert detail["peak_memory_gib"] == pytest.approx(18.161, abs=1e-3)
    assert detail["budget_gib"] == 17.76


def test_gib_from_gb_names_the_trainers_unit_and_this_repositorys() -> None:
    """Arm A's own record, converted, so the number appears where a reader will meet it.

    `outputs/agent-v2e-qwen35-4b/health.json` records `peak_memory_gb: 10.4436`. That is
    **9.726 GiB**, against a device working set of 17.76 GiB — which is what the run was really
    using, and what the memory envelope in `preflight` should be checked against.
    """
    assert gib_from_gb(10.443645016) == pytest.approx(9.726, abs=5e-4)
    assert gib_from_gb(1024**3 / 1e9) == pytest.approx(1.0)
