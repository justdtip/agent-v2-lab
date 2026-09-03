from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TextIO

from local_llm_lab.agent_protocol import Action


def iter_task_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "task_id" in record:
                yield record


def _supports_color(stream: TextIO) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


def _collapse_thinking(thinking: str | None) -> str:
    lines = [line.strip() for line in (thinking or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return lines[0] if lines else ""
    return f"{lines[0]} … {lines[-1]}"


@dataclass
class Transcript:
    """Live, human-readable trace of one task, mirrored to Markdown and JSONL on disk."""

    stream: TextIO | None = field(default_factory=lambda: sys.stdout)
    directory: Path | None = None
    color: bool | None = None
    observation_lines: int = 4
    _lines: list[str] = field(default_factory=list)
    _record: dict[str, Any] = field(default_factory=dict)
    _active_runs: ClassVar[dict[Path, str]] = {}

    @classmethod
    def start_run(cls, directory: Path) -> str:
        """Start one transcript run, replacing only records from the previous run."""
        target = directory.resolve()
        target.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        (target / "transcripts.jsonl").write_text(
            json.dumps({"run_id": run_id}, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        cls._active_runs[target] = run_id
        return run_id

    @classmethod
    def _run_id(cls, directory: Path) -> str:
        target = directory.resolve()
        return cls._active_runs.get(target) or cls.start_run(target)

    def __post_init__(self) -> None:
        if self.color is None:
            self.color = self.stream is not None and _supports_color(self.stream)

    def _paint(self, text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.color else text

    def _emit(self, text: str, markdown: str | None = None) -> None:
        if self.stream is not None:
            print(text, file=self.stream, flush=True)
        self._lines.append(text if markdown is None else markdown)

    def start(self, task: Any, label: str) -> None:
        run_id = None
        if self.directory is not None:
            run_id = self._run_id(self.directory)
        self._record = {
            "task_id": task.task_id,
            "family": task.family,
            "variant": task.variant,
            "label": label,
            "prompt": task.prompt,
            "steps": [],
        }
        if run_id is not None:
            self._record["run_id"] = run_id
        self._lines = []
        self._emit(
            self._paint(f"\n=== {label} :: {task.task_id} [{task.family}/{task.variant}]", "1;36"),
            f"# {label} :: {task.task_id}\n\n*family* `{task.family}` · *variant* `{task.variant}`\n",
        )
        self._emit(self._paint("task: ", "2") + task.prompt, f"**Task.** {task.prompt}\n")

    def step(
        self,
        index: int,
        thought: str,
        action: Action | None,
        observation: str | None,
        *,
        thinking: str | None = None,
        think_tokens: int = 0,
        raw: str | None = None,
        parse_error: str | None = None,
    ) -> None:
        entry = {
            "index": index,
            "thought": thought,
            "action": None
            if action is None
            else {"name": action.name, "arguments": action.arguments},
            "observation": observation,
            "thinking": thinking,
            "think_tokens": think_tokens,
            "raw": raw,
            "parse_error": parse_error,
        }
        self._record["steps"].append(entry)
        head = self._paint(f"[{index:02d}]", "1")
        thinking_summary = _collapse_thinking(thinking)
        thinking_text = (
            f"\n     {self._paint('think', '34')}  {thinking_summary}" if thinking_summary else ""
        )
        thinking_markdown = (
            f"\n\n**Thinking.** _{thinking_summary}_" if thinking_summary else ""
        )
        if parse_error:
            self._emit(
                f"{head} {self._paint('PARSE ERROR: ' + parse_error, '31')}"
                f"{thinking_text}\n     raw: {raw!r}",
                f"\n**Step {index}.** PARSE ERROR: {parse_error}"
                f"{thinking_markdown}\n\n```\n{raw}\n```\n",
            )
            return
        note = thought if thought else "(no note)"
        args = json.dumps(action.arguments, ensure_ascii=False) if action else ""
        call = f"{action.name} {args}" if action else ""
        is_error = observation is not None and observation.startswith("ERROR")
        obs_lines = (observation or "").splitlines()
        shown = "\n".join(obs_lines[: self.observation_lines])
        if len(obs_lines) > self.observation_lines:
            shown += f"\n     … ({len(obs_lines) - self.observation_lines} more lines)"
        obs_text = self._paint(shown, "31" if is_error else "2")
        self._emit(
            f"{head} {self._paint('note', '33')}  {note}{thinking_text}\n"
            f"     {self._paint('call', '32')}  {call}\n"
            f"     {self._paint('obs ', '35')}  {obs_text.replace(chr(10), chr(10) + '           ')}",
            f"\n**Step {index}.** _{note}_{thinking_markdown}"
            f"\n\n`{call}`\n\n```\n{observation or ''}\n```\n",
        )

    def finish(self, verdict: dict[str, Any], elapsed: float) -> None:
        self._record["verdict"] = verdict
        self._record["elapsed_seconds"] = round(elapsed, 2)
        mark = "PASS" if verdict["success"] else "FAIL"
        color = "1;32" if verdict["success"] else "1;31"
        detail = "" if verdict["success"] else " (" + "; ".join(verdict["reasons"]) + ")"
        if verdict["success"] and verdict["errors"]:
            detail = f" (recovered from {verdict['errors']} tool error(s))"
        answer = f"answer={verdict['answer']!r} expected={verdict['expected_answer']!r}"
        self._emit(
            f"{self._paint(mark, color)}{detail}  {answer}  {elapsed:.1f}s",
            f"\n**{mark}**{detail} — {answer} — {elapsed:.1f}s\n",
        )
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            name = f"{self._record['label']}-{self._record['task_id']}".replace("/", "_")
            (self.directory / f"{name}.md").write_text(
                "\n".join(self._lines) + "\n", encoding="utf-8"
            )
            with (self.directory / "transcripts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self._record, ensure_ascii=False) + "\n")

    @property
    def record(self) -> dict[str, Any]:
        return self._record


def summary_table(summary: dict[str, Any]) -> str:
    rows = [
        ("tasks", summary["tasks"]),
        ("success", f"{summary['successes']} ({summary['success_rate']:.1%})"),
        ("clean success", f"{summary['clean_successes']} ({summary['clean_rate']:.1%})"),
        ("valid action rate", f"{summary['valid_action_rate']:.1%}"),
        ("schema validity", f"{summary.get('schema_validity_rate', 0.0):.1%}"),
        ("executable calls", f"{summary.get('executable_call_rate', 0.0):.1%}"),
        ("tool errors", summary["tool_errors"]),
        ("recovered errors", summary["recovered_errors"]),
        ("loop failures", summary.get("loop_failures", 0)),
        ("budget exhausted", summary.get("exhausted", 0)),
        ("tokens/success", _or_dash(summary.get("tokens_per_success"))),
        (
            "latency p50/p95",
            f"{summary.get('latency_p50_seconds', 0.0):.1f}s / {summary.get('latency_p95_seconds', 0.0):.1f}s",
        ),
        ("mean steps", summary["mean_steps"]),
    ]
    width = max(len(name) for name, _ in rows)
    lines = [f"  {name:<{width}}  {value}" for name, value in rows]
    lines.append("  by family:")
    for family, stats in summary["by_family"].items():
        lines.append(f"    {family:<20} {stats['successes']}/{stats['tasks']}")
    if summary.get("failure_reasons"):
        lines.append("  failure reasons:")
        for reason, count in summary["failure_reasons"].items():
            lines.append(f"    {count:3d}  {reason}")
    return "\n".join(lines)


def _or_dash(value: Any) -> str:
    """Render ``None`` (e.g. tokens per success with zero successes) as a dash."""
    return "-" if value is None else str(value)
