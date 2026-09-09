from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from local_llm_lab.agent_protocol import TOOL_SPECS, Action
from local_llm_lab.agent_tasks import _calculate

TOOL_NAMES = ("list_files", "read_file", "search_files", "calculate", "replace_text", "finish")
TRANSIENT_ERROR = "ERROR: temporary failure while executing the tool; retry the same call"

# JSON-schema types handled by ``validate_arguments`` and the Python types that satisfy them.
# ``bool`` is a subclass of ``int``, so integer/number checks reject it explicitly.
_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass(frozen=True)
class Fault:
    """A transient failure injected at a given 0-based call index."""

    call_index: int
    message: str = TRANSIENT_ERROR


@dataclass(frozen=True)
class Verdict:
    success: bool
    clean: bool
    reasons: tuple[str, ...]
    answer: str | None
    expected_answer: str
    errors: int
    recovered_errors: int
    calls: int = 0
    schema_failures: int = 0
    executable_calls: int = 0
    unexpected_files: tuple[str, ...] = ()
    raw_answer: str | None = None
    contains_expected: bool = False
    """Whether the answer *contains* the expected value, under the same normaliser.

    ``success`` requires exact normalised equality, which is stricter than what many task
    prompts actually ask for. Reporting the looser bound beside the strict one means every
    pass rate carries the gap between what was graded and what was requested, instead of the
    gap being rediscovered later as a result.
    """

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "clean": self.clean,
            "contains_expected": self.contains_expected,
            "reasons": list(self.reasons),
            "answer": self.answer,
            "expected_answer": self.expected_answer,
            "errors": self.errors,
            "recovered_errors": self.recovered_errors,
            "calls": self.calls,
            "schema_failures": self.schema_failures,
            "executable_calls": self.executable_calls,
            "unexpected_files": list(self.unexpected_files),
            "raw_answer": self.raw_answer,
        }


def normalize(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def normalize_answer(value: str) -> str:
    """Compare answer values without harmless Markdown fencing or terminal punctuation."""
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned[1:-1].strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()
    if len(cleaned) >= 2 and cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned[1:-1].strip()
    return normalize(cleaned)


def _json_type_name(value: Any) -> str:
    """Name of ``value``'s JSON type, for error messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_type(value: Any, schema_type: str) -> bool:
    accepted = _JSON_TYPES.get(schema_type)
    if accepted is None:  # unknown schema type: nothing to check
        return True
    if schema_type in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, accepted)


def validate_arguments(action: Action, tools: list[dict[str, Any]] = TOOL_SPECS) -> None:
    """Check ``action`` against the JSON-schema tool definitions before it is executed.

    Raises ``ValueError`` with a precise message when the tool is unknown, a required argument
    is missing, an argument is not declared in the schema, or an argument has the wrong JSON
    type. Validator-first execution turns each of these into an observation the policy must
    recover from, instead of an arbitrary execution error.
    """
    functions = {spec["function"]["name"]: spec["function"] for spec in tools}
    if action.name not in functions:
        raise ValueError(f"unknown tool: {action.name}; available tools are {', '.join(functions)}")
    parameters = functions[action.name].get("parameters", {})
    properties: dict[str, dict[str, Any]] = parameters.get("properties", {})
    required = [name for name in parameters.get("required", []) if name in properties]
    expected = ", ".join(properties) or "(no arguments)"
    arguments = action.arguments
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(
            f"missing required argument(s): {', '.join(missing)}; expected: {expected}"
        )
    unexpected = [name for name in arguments if name not in properties]
    if unexpected:
        raise ValueError(f"unexpected argument(s): {', '.join(unexpected)}; expected: {expected}")
    for name, value in arguments.items():
        schema_type = properties[name].get("type")
        if isinstance(schema_type, str) and not _matches_type(value, schema_type):
            raise ValueError(f"{name} must be {schema_type}, got {_json_type_name(value)}")


@dataclass
class Simulator:
    """Deterministic in-memory workspace shared by data generation, evaluation, and rollouts."""

    files: dict[str, str]
    expected_answer: str
    expected_files: dict[str, str] = field(default_factory=dict)
    required_tools: frozenset[str] = frozenset()
    faults: tuple[Fault, ...] = ()
    calls: list[Action] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    errors: int = 0
    schema_failures: int = 0
    executable_calls: int = 0
    finished_answer: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=lambda: TOOL_SPECS)
    initial_files: dict[str, str] = field(default_factory=dict)
    executed_tools: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.initial_files:
            self.initial_files = dict(self.files)

    @classmethod
    def for_task(cls, task: Any, faults: tuple[Fault, ...] | None = None) -> Simulator:
        return cls(
            files=dict(task.files),
            expected_answer=task.expected_answer,
            expected_files=dict(task.expected_files),
            required_tools=frozenset(task.required_tools),
            faults=tuple(task.faults) if faults is None else faults,
            initial_files=dict(task.files),
        )

    @property
    def finished(self) -> bool:
        return self.finished_answer is not None

    def execute(self, action: Action) -> str:
        """Run one call: injected faults first (they are the environment), then schema
        validation, then execution. Every failure counts as an error; only calls that ran
        without any failure count as executable."""
        index = len(self.calls)
        self.calls.append(action)
        fault = next((fault for fault in self.faults if fault.call_index == index), None)
        if fault is not None:
            self.errors += 1
            self.observations.append(fault.message)
            return fault.message
        try:
            validate_arguments(action, self.tools)
        except ValueError as error:
            self.errors += 1
            self.schema_failures += 1
            result = f"ERROR: invalid call: {error}"
            self.observations.append(result)
            return result
        try:
            result = self._execute(action)
            self.executable_calls += 1
            self.executed_tools.add(action.name)
        except Exception as error:  # noqa: BLE001 - a tool must never take down the harness
            # Arguments are model-generated, so any failure here is input-driven and must come
            # back as an observation the policy can recover from. A base model passing
            # "Status: queued" to the calculator raises SyntaxError, which is not a ValueError
            # and previously aborted the whole evaluation run.
            self.errors += 1
            result = f"ERROR: {error}"
        self.observations.append(result)
        return result

    def _execute(self, action: Action) -> str:
        args = action.arguments
        if action.name == "list_files":
            given = _string(args, "directory")
            directory = given.rstrip("/")
            paths = sorted(path for path in self.files if path.startswith(directory + "/"))
            if not paths:
                #: A workspace is a flat set of paths, so it holds no empty directories: every
                #: directory that exists contains at least one file, and "nothing matched this
                #: prefix" is therefore the same proposition as "this directory does not exist".
                #: Answering it with `FILES: (none)` asserted the first while meaning the second,
                #: and the model has no way to tell those apart. On 2026-09-08 that cost us an
                #: entire episode: `update-0028` opened with `list_files("/")`, was told the
                #: workspace was empty, correctly concluded there was no file to edit, and spent
                #: its remaining twenty-three turns trying to create one with no tool that can.
                #: Every note it wrote was sound reasoning from a falsehood we handed it. R56(f).
                #: Echo what the caller sent, not the stripped form: `list_files("/")` strips
                #: to the empty string, and an error naming nothing teaches nothing.
                raise ValueError(f"directory not found: {given}")
            return "FILES: " + ", ".join(paths)
        if action.name == "read_file":
            path = _string(args, "path")
            if path not in self.files:
                raise ValueError(f"file not found: {path}")
            return self.files[path]
        if action.name == "search_files":
            query = _string(args, "query").casefold()
            paths = sorted(
                path
                for path, content in self.files.items()
                if query in path.casefold() or query in content.casefold()
            )
            return "MATCHES: " + (", ".join(paths) if paths else "(none)")
        if action.name == "calculate":
            expression = _string(args, "expression")
            try:
                return "RESULT: " + _calculate(expression)
            except SyntaxError as error:
                raise ValueError(
                    "expression is not valid arithmetic; use only numbers and + - * / ( )"
                ) from error
        if action.name == "replace_text":
            path = _string(args, "path")
            old = _string(args, "old")
            new = _string(args, "new")
            if path not in self.files:
                raise ValueError(f"file not found: {path}")
            if old not in self.files[path]:
                raise ValueError("old text not found")
            self.files[path] = self.files[path].replace(old, new, 1)
            return f"UPDATED: {path}"
        if action.name == "finish":
            self.finished_answer = _string(args, "answer")
            return "FINISHED"
        raise ValueError(
            f"unknown tool: {action.name}; available tools are {', '.join(TOOL_NAMES)}"
        )

    def verdict(self) -> Verdict:
        reasons = []
        if self.finished_answer is None:
            reasons.append("no finish call")
        elif normalize_answer(self.finished_answer) != normalize_answer(self.expected_answer):
            reasons.append("wrong answer")
        for path, content in self.expected_files.items():
            if self.files.get(path) != content:
                reasons.append(f"file state wrong: {path.rsplit('/', 1)[-1]}")
        unexpected_files = tuple(
            path
            for path in sorted(set(self.initial_files) | set(self.files))
            if path not in self.expected_files and self.files.get(path) != self.initial_files.get(path)
        )
        reasons.extend(
            f"unexpected file change: {path.rsplit('/', 1)[-1]}" for path in unexpected_files
        )
        missing = sorted(self.required_tools - self.executed_tools)
        if missing:
            reasons.append("required tools unused: " + ", ".join(missing))
        success = not reasons
        # The looser grade uses the same normaliser as the strict one, so the two differ only
        # in exact-versus-containment and never in whitespace, case or fencing.
        given = None if self.finished_answer is None else normalize_answer(self.finished_answer)
        wanted = normalize_answer(self.expected_answer)
        contains_expected = bool(given is not None and wanted and wanted in given)
        return Verdict(
            success=success,
            contains_expected=contains_expected,
            clean=success and self.errors == 0,
            reasons=tuple(reasons),
            answer=None if self.finished_answer is None else normalize_answer(self.finished_answer),
            expected_answer=normalize_answer(self.expected_answer),
            errors=self.errors,
            recovered_errors=self.errors if success else 0,
            calls=len(self.calls),
            schema_failures=self.schema_failures,
            executable_calls=self.executable_calls,
            unexpected_files=unexpected_files,
            raw_answer=self.finished_answer,
        )


def _string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value
