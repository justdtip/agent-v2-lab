"""Run logging and training-health rules for pipeline stages.

Three artefacts per run: one console line per event, the same lines mirrored into
``run.log``, and a machine-readable ``events.jsonl``.  Stdlib only (the probe package is
numpy-only by design and this module is imported by the pipeline, the probes and the
gate) and no module-level state: the files, the clock and the streams all live on the
``RunLog`` instance, so a test can inject a counter clock and ``io.StringIO`` streams and
get byte-exact timestamps, elapsed values and ETAs.

Logging must never be the thing that kills a run, so field values that json cannot encode
are stringified rather than raised on, and emitting after ``close()`` is a no-op.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

__all__ = [
    "RunLog",
    "Tee",
    "HealthThresholds",
    "TrainingHealth",
    "TrainingAborted",
    "sha256_of",
    "git_commit",
]

RUN_LOG_NAME = "run.log"
EVENTS_NAME = "events.jsonl"

_HASH_CHUNK = 1 << 20  # 1 MiB: a multi-GB adapter must not land in memory to be hashed
_GIT_TIMEOUT = 10.0

_BAR = "│"
#: One space between the ``start`` message and its fields, like every other body.
_START_GUTTER = " "


def _utc_now() -> str:
    """UTC ISO 8601 with a trailing ``Z`` rather than ``+00:00``."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _duration(seconds: float) -> str:
    """``MM:SS``, rolling over to ``H:MM:SS`` past an hour. Non-finite or negative is zero."""
    value = float(seconds)
    total = int(value) if math.isfinite(value) and value > 0 else 0
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_value(value: Any) -> str:
    """Console rendering of one field value. ``bool`` is checked before ``int``."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _format_fields(values: Mapping[str, Any]) -> str:
    """``k=v k=v`` in the order given (kwargs order)."""
    return " ".join(f"{key}={_format_value(value)}" for key, value in values.items())


def _json_number(value: Any) -> Any:
    """NaN/inf are not valid JSON; carry them as ``"nan"``/``"inf"``/``"-inf"`` instead."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats so ``events.jsonl`` stays strict JSON.

    ``json.dumps`` would otherwise emit the bare tokens ``NaN``/``Infinity``, which strict
    readers reject; a machine-readable log that a strict reader cannot parse is not one.
    """
    if isinstance(value, float):
        return _json_number(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks so large weights are never held in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(cwd: Path | None = None) -> str:
    """``git rev-parse HEAD`` for provenance.

    Never raises: provenance is a nice-to-have and must not be able to fail a run, so any
    error (no git, not a repository, timeout) is reported as ``"unknown"``.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


class Tee:
    """Write to a primary stream and a file, flushing both on every write.

    Used to mirror a library's stdout (mlx-lm's trainer) into ``run.log`` without losing
    the live console output; flushing on every write is what keeps the two in step when
    the process is killed mid-run.
    """

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, value: str) -> int:
        self.primary.write(value)
        self.secondary.write(value)
        self.primary.flush()
        self.secondary.flush()
        return len(value)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        probe = getattr(self.primary, "isatty", None)
        if probe is None:
            return False
        try:
            return bool(probe())
        except (AttributeError, ValueError, io.UnsupportedOperation):
            return False

    def fileno(self) -> int:
        probe = getattr(self.primary, "fileno", None)
        if probe is None:
            raise io.UnsupportedOperation("fileno")
        try:
            return int(probe())
        except (AttributeError, ValueError) as exc:
            raise io.UnsupportedOperation("fileno") from exc

    @property
    def encoding(self) -> Any:
        return getattr(self.primary, "encoding", None)


class RunLog:
    """One run's console/file/event log.

    Construct with :meth:`open`; the direct constructor takes already-resolved handles.
    """

    def __init__(
        self,
        *,
        name: str,
        output: Path,
        run_log_path: Path,
        events_path: Path,
        run_log: TextIO,
        events: TextIO,
        started: float,
        clock: Callable[[], float],
        stdout: TextIO,
        stderr: TextIO,
    ) -> None:
        self._name = name
        self._output = output
        self._run_log_path = run_log_path
        self._events_path = events_path
        self._run_log = run_log
        self._events = events
        self._started = started
        self._clock = clock
        self._stdout = stdout
        self._stderr = stderr
        self._closed = False

    @classmethod
    def open(
        cls,
        output: Path,
        *,
        name: str,
        command: Sequence[str] | None = None,
        identity: Mapping[str, Any] | None = None,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> "RunLog":
        """Open ``output/run.log`` and ``output/events.jsonl`` in append mode and record the start.

        Append rather than truncate so a resumed or re-entered stage adds to the record
        instead of erasing the earlier attempt. ``stdout``/``stderr`` default to
        ``sys.stdout``/``sys.stderr`` resolved here, not at import time, so a caller that
        has already redirected them is honoured. ``identity`` (R26(e), issue #35) carries
        the run's provenance -- model id, config hash, git commit -- into the ``start``
        event's fields and onto the start line.
        """
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        run_log_path = directory / RUN_LOG_NAME
        events_path = directory / EVENTS_NAME
        run_log = run_log_path.open("a", encoding="utf-8")
        events = events_path.open("a", encoding="utf-8")
        started = clock()
        self = cls(
            name=name,
            output=directory,
            run_log_path=run_log_path,
            events_path=events_path,
            run_log=run_log,
            events=events,
            started=started,
            clock=clock,
            stdout=sys.stdout if stdout is None else stdout,
            stderr=sys.stderr if stderr is None else stderr,
        )
        argv = list(command) if command is not None else None
        joined = None if argv is None else " ".join(argv)
        marks = dict(identity) if identity else {}
        body = "start" + _START_GUTTER + _format_fields(
            {"command": joined, "log": str(run_log_path), **marks}
        )
        self._emit(
            kind="start",
            message="start",
            body=body,
            stream=self._stdout,
            elapsed=0.0,
            values=marks,
            extra={
                "command": argv,
                "pid": os.getpid(),
                "cwd": os.getcwd(),
                "python": "{}.{}.{}".format(*sys.version_info[:3]),
                "started_at": _utc_now(),
            },
        )
        return self

    # -- emission ---------------------------------------------------------------

    def _emit(
        self,
        *,
        kind: str,
        message: str | None,
        body: str,
        stream: TextIO,
        elapsed: float,
        values: Mapping[str, Any],
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if self._closed:
            return
        line = f"{time.strftime('%H:%M:%S')} +{_duration(elapsed)} {self._name} {_BAR} {body}"
        stream.write(line + "\n")
        stream.flush()
        self._run_log.write(line + "\n")
        self._run_log.flush()
        self._write_event(kind=kind, message=message, elapsed=elapsed, values=values, extra=extra)

    def _write_event(
        self,
        *,
        kind: str,
        message: str | None,
        elapsed: float,
        values: Mapping[str, Any],
        extra: Mapping[str, Any] | None,
    ) -> None:
        event: dict[str, Any] = {
            "ts": _utc_now(),
            "elapsed": float(elapsed),
            "run": self._name,
            "kind": kind,
            "message": message,
            "fields": dict(values),
        }
        if extra:
            event.update(extra)
        event = _json_safe(event)
        try:
            payload = json.dumps(event, sort_keys=True, default=str, allow_nan=False)
        except (TypeError, ValueError):
            event["fields"] = {key: str(value) for key, value in values.items()}
            payload = json.dumps(event, sort_keys=True, default=str, allow_nan=False)
        self._events.write(payload + "\n")
        self._events.flush()

    def _body(self, message: str, values: Mapping[str, Any]) -> str:
        rendered = _format_fields(values)
        return f"{message} {rendered}" if rendered else message

    # -- public logging ---------------------------------------------------------

    def info(self, message: str, **fields: Any) -> None:
        self._emit(
            kind="info",
            message=message,
            body=self._body(message, fields),
            stream=self._stdout,
            elapsed=self.elapsed,
            values=fields,
        )

    def warn(self, message: str, **fields: Any) -> None:
        self._emit(
            kind="warning",
            message=message,
            body=self._body(message, fields),
            stream=self._stderr,
            elapsed=self.elapsed,
            values=fields,
        )

    def error(self, message: str, **fields: Any) -> None:
        self._emit(
            kind="error",
            message=message,
            body=self._body(message, fields),
            stream=self._stderr,
            elapsed=self.elapsed,
            values=fields,
        )

    def metric(self, name: str, **fields: Any) -> None:
        self._emit(
            kind="metric",
            message=name,
            body=self._body(name, fields),
            stream=self._stdout,
            elapsed=self.elapsed,
            values=fields,
        )

    def progress(self, step: int, total: int, label: str, **fields: Any) -> None:
        """One progress line; the caller chooses the cadence, this never rate-limits."""
        elapsed = self.elapsed
        step_n = int(step)
        total_n = int(total)
        fraction = (step_n / total_n) if total_n > 0 else 0.0
        eta: float | None
        if step_n <= 0 or total_n <= 0:
            eta = None
        else:
            eta = elapsed / step_n * (total_n - step_n)
        body = f"[{label} {step_n}/{total_n} {fraction * 100:.0f}%]"
        rendered = _format_fields(fields)
        if rendered:
            body = f"{body} {rendered}"
        if eta is not None:
            body = f"{body} eta {_duration(eta)}"
        self._emit(
            kind="progress",
            message=None,
            body=body,
            stream=self._stdout,
            elapsed=elapsed,
            values=fields,
            extra={
                "step": step_n,
                "total": total_n,
                "fraction": fraction,
                "eta_seconds": eta,
                "label": label,
            },
        )

    def tee(self, stream: TextIO) -> TextIO:
        """A stream that writes to ``stream`` and mirrors into ``run.log``."""
        return Tee(stream, self._run_log)

    def close(self, status: str = "ok", **summary: Any) -> None:
        """Write the ``end`` event and close both files. A second call is a no-op."""
        if self._closed:
            return
        elapsed = self.elapsed
        body_fields: dict[str, Any] = {"status": status, "elapsed": _duration(elapsed)}
        body_fields.update(summary)
        self._emit(
            kind="end",
            message="end",
            body=self._body("end", body_fields),
            stream=self._stdout,
            elapsed=elapsed,
            values=summary,
            extra={"status": status},
        )
        self._closed = True
        self._run_log.close()
        self._events.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close(status="ok")
        else:
            self.close(status="error", error=f"{exc_type.__name__}: {exc}")
        return None

    # -- accessors --------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return self._clock() - self._started

    @property
    def paths(self) -> dict[str, Path]:
        return {"run_log": self._run_log_path, "events": self._events_path}

    @property
    def name(self) -> str:
        return self._name


def _reject_bool(key: str, value: Any) -> None:
    if isinstance(value, bool):
        raise ValueError(f"train.health.{key} must be a number, got {value!r}")


def _positive_float(key: str, value: Any) -> float:
    _reject_bool(key, value)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"train.health.{key} must be a number, got {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            f"train.health.{key} must be a finite number greater than 0, got {value!r}"
        )
    return number


def _window_int(key: str, value: Any) -> int:
    _reject_bool(key, value)
    if not isinstance(value, (int, float)):
        raise ValueError(f"train.health.{key} must be an integer, got {value!r}")
    number = int(value)
    if number != value:
        raise ValueError(f"train.health.{key} must be a whole number, got {value!r}")
    if number < 2:
        raise ValueError(f"train.health.{key} must be at least 2, got {value!r}")
    return number


@dataclass(frozen=True)
class HealthThresholds:
    """Documented engineering defaults for the training-health rules."""

    loss_spike_factor: float = 2.0      # train loss above this multiple of the trailing median is a spike
    loss_window: int = 5                # trailing train reports the median is taken over
    val_rising_reports: int = 3         # this many consecutive rising validation losses flags overfitting
    throughput_drop_factor: float = 0.5 # tokens/s below this fraction of the trailing median flags a throttle
    memory_budget_gib: float | None = None  # peak memory above this flags over-budget; None disables

    @classmethod
    def from_config(
        cls,
        block: Mapping[str, Any] | None,
        *,
        memory_budget_gib: float | None,
    ) -> "HealthThresholds":
        """Build from the optional ``train.health:`` mapping.

        The memory budget comes from the model registry; the config block overrides it
        only when it names ``memory_budget_gib`` explicitly, so a run that says nothing
        about memory still inherits the machine's budget.
        """
        names = tuple(spec.name for spec in fields(cls))
        raw = dict(block) if block else {}
        for key in raw:
            if key not in names:
                raise ValueError(
                    f"unknown train.health key {key!r}; allowed keys are {', '.join(names)}"
                )
        values: dict[str, Any] = {}
        for key in ("loss_spike_factor", "throughput_drop_factor"):
            if key in raw:
                values[key] = _positive_float(key, raw[key])
        for key in ("loss_window", "val_rising_reports"):
            if key in raw:
                values[key] = _window_int(key, raw[key])
        if "memory_budget_gib" in raw:
            budget = raw["memory_budget_gib"]
            values["memory_budget_gib"] = (
                None if budget is None else _positive_float("memory_budget_gib", budget)
            )
        elif memory_budget_gib is not None:
            values["memory_budget_gib"] = _positive_float("memory_budget_gib", memory_budget_gib)
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainingAborted(RuntimeError):
    """Raised on a fatal health flag; ``.flag`` is the flag record that fired."""

    def __init__(self, message: str, flag: dict[str, Any]) -> None:
        super().__init__(message)
        self.flag = flag


class TrainingHealth:
    """Turns the trainer's metric callbacks into flags, a verdict and a summary.

    Rules are cheap and local on purpose: they run inside the training callback, so they
    may not allocate, may not look at the model, and must never be the reason a healthy
    run stops.
    """

    FLAGS = (
        "non_finite_loss",
        "loss_spike",
        "val_loss_rising",
        "throughput_drop",
        "memory_over_budget",
        "incomplete_run",
    )

    def __init__(self, thresholds: HealthThresholds, *, iters: int | None) -> None:
        self._thresholds = thresholds
        self._iters = None if iters is None else int(iters)
        self._train_reports: list[dict[str, Any]] = []
        self._val_reports: list[dict[str, Any]] = []
        self._flags: list[dict[str, Any]] = []
        self._memory_flagged = False
        self._val_rising_active = False
        self._finished: dict[str, Any] | None = None

    # -- internals --------------------------------------------------------------

    def _record(
        self, name: str, *, severity: str, iteration: int, detail: Mapping[str, Any]
    ) -> dict[str, Any]:
        flag = {
            "flag": name,
            "severity": severity,
            "iteration": int(iteration),
            "detail": {key: _json_number(value) for key, value in detail.items()},
        }
        self._flags.append(flag)
        return flag

    # -- callbacks --------------------------------------------------------------

    def on_train_report(
        self,
        *,
        iteration: int,
        train_loss: float,
        learning_rate: float,
        tokens_per_second: float,
        trained_tokens: int,
        peak_memory_gb: float,
    ) -> list[str]:
        """Record one train report and return the names of flags it newly raised."""
        prior_losses = [report["train_loss"] for report in self._train_reports]
        prior_throughput = [report["tokens_per_second"] for report in self._train_reports]
        self._train_reports.append(
            {
                "iteration": int(iteration),
                "train_loss": float(train_loss),
                "learning_rate": float(learning_rate),
                "tokens_per_second": float(tokens_per_second),
                "trained_tokens": int(trained_tokens),
                "peak_memory_gb": float(peak_memory_gb),
            }
        )
        raised: list[str] = []
        window = self._thresholds.loss_window

        if not math.isfinite(train_loss):
            flag = self._record(
                "non_finite_loss",
                severity="fatal",
                iteration=iteration,
                detail={"train_loss": float(train_loss)},
            )
            raise TrainingAborted(
                f"training aborted at iteration {int(iteration)}: train loss is {train_loss}",
                flag,
            )

        if len(prior_losses) >= window:
            median = statistics.median(prior_losses[-window:])
            threshold = self._thresholds.loss_spike_factor * median
            if train_loss > threshold:
                self._record(
                    "loss_spike",
                    severity="warning",
                    iteration=iteration,
                    detail={
                        "train_loss": float(train_loss),
                        "median": float(median),
                        "threshold": float(threshold),
                        "window": window,
                    },
                )
                raised.append("loss_spike")

        if len(prior_throughput) >= window:
            median_tps = statistics.median(prior_throughput[-window:])
            floor = self._thresholds.throughput_drop_factor * median_tps
            if tokens_per_second < floor:
                self._record(
                    "throughput_drop",
                    severity="warning",
                    iteration=iteration,
                    detail={
                        "tokens_per_second": float(tokens_per_second),
                        "median": float(median_tps),
                        "threshold": float(floor),
                        "window": window,
                    },
                )
                raised.append("throughput_drop")

        budget = self._thresholds.memory_budget_gib
        if budget is not None and not self._memory_flagged and peak_memory_gb > budget:
            self._memory_flagged = True
            self._record(
                "memory_over_budget",
                severity="warning",
                iteration=iteration,
                detail={"peak_memory_gb": float(peak_memory_gb), "budget_gib": float(budget)},
            )
            raised.append("memory_over_budget")

        return raised

    def on_val_report(self, *, iteration: int, val_loss: float) -> list[str]:
        """Record one validation report and return the names of flags it newly raised."""
        previous = self._val_reports[-1]["val_loss"] if self._val_reports else None
        self._val_reports.append({"iteration": int(iteration), "val_loss": float(val_loss)})

        if not math.isfinite(val_loss):
            flag = self._record(
                "non_finite_loss",
                severity="fatal",
                iteration=iteration,
                detail={"val_loss": float(val_loss)},
            )
            raise TrainingAborted(
                f"training aborted at iteration {int(iteration)}: validation loss is {val_loss}",
                flag,
            )

        raised: list[str] = []
        if previous is not None and val_loss <= previous:
            self._val_rising_active = False
        needed = self._thresholds.val_rising_reports
        if len(self._val_reports) >= needed:
            window = [report["val_loss"] for report in self._val_reports[-needed:]]
            rising = all(later > earlier for earlier, later in zip(window, window[1:]))
            if rising and not self._val_rising_active:
                self._val_rising_active = True
                self._record(
                    "val_loss_rising",
                    severity="warning",
                    iteration=iteration,
                    detail={"val_losses": [float(value) for value in window], "reports": needed},
                )
                raised.append("val_loss_rising")
        return raised

    def on_finish(self, *, iterations_done: int, final_checkpoint: bool) -> list[str]:
        """Called once on whichever path the training stage leaves by (R26(c), issue #35).

        A run that ends without completing its iterations, or without leaving a final
        checkpoint, produced an adapter nothing downstream may select from; that is fatal for
        the run's verdict but does not raise, because the run has already ended. It holds
        just as much when an exception ended the run, so the caller may call this from the
        normal path and again from its ``finally``: the first call wins and every later one
        is a no-op returning no flags, so one exit records one ``incomplete_run``.
        """
        if self._finished is not None:
            return []
        done = int(iterations_done)
        complete_checkpoint = bool(final_checkpoint)
        self._finished = {"iterations_done": done, "final_checkpoint": complete_checkpoint}
        short = self._iters is not None and done < self._iters
        if not (short or not complete_checkpoint):
            return []
        self._record(
            "incomplete_run",
            severity="fatal",
            iteration=done,
            detail={
                "iterations_done": done,
                "iters_planned": self._iters,
                "final_checkpoint": complete_checkpoint,
            },
        )
        return ["incomplete_run"]

    # -- verdict ----------------------------------------------------------------

    @property
    def flags(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._flags)

    @property
    def verdict(self) -> str:
        """``aborted`` outranks ``incomplete``, which outranks ``warnings`` (R26(c), issue #35)."""
        names = {flag["flag"] for flag in self._flags}
        if "non_finite_loss" in names:
            return "aborted"
        if "incomplete_run" in names:
            return "incomplete"
        return "warnings" if self._flags else "healthy"

    def summary(self, *, elapsed: float, status: str) -> dict[str, Any]:
        """A JSON-safe record of the run's health, for ``events.jsonl`` and the gate."""
        last_train = self._train_reports[-1] if self._train_reports else None
        last_val = self._val_reports[-1] if self._val_reports else None
        finite_val = [
            report for report in self._val_reports if math.isfinite(report["val_loss"])
        ]
        best_val = min(finite_val, key=lambda report: report["val_loss"]) if finite_val else None
        memories = [
            report["peak_memory_gb"]
            for report in self._train_reports
            if math.isfinite(report["peak_memory_gb"])
        ]
        throughput = [
            report["tokens_per_second"]
            for report in self._train_reports
            if math.isfinite(report["tokens_per_second"])
        ]
        return {
            "verdict": self.verdict,
            "status": status,
            "flags": [
                {**flag, "detail": dict(flag["detail"])} for flag in self._flags
            ],
            "thresholds": self._thresholds.as_dict(),
            "iters_planned": self._iters,
            "iters_done": last_train["iteration"] if last_train else 0,
            "iterations_done": (
                self._finished["iterations_done"]
                if self._finished is not None
                else (last_train["iteration"] if last_train else 0)
            ),
            "final_checkpoint": (
                self._finished["final_checkpoint"] if self._finished is not None else None
            ),
            "train_reports": len(self._train_reports),
            "val_reports": len(self._val_reports),
            "last_train": _safe_record(last_train),
            "last_val": _safe_record(last_val),
            "best_val": _safe_record(best_val),
            "peak_memory_gb": max(memories) if memories else None,
            "median_tokens_per_second": statistics.median(throughput) if throughput else None,
            "elapsed": float(elapsed),
        }


def _safe_record(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {key: _json_number(value) for key, value in record.items()}
