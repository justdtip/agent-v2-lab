from __future__ import annotations

import ast
import operator
import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from local_llm_lab.agent_protocol import Action

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    category: str
    prompt: str
    files: dict[str, str]
    expert_actions: tuple[Action, ...]
    expected_answer: str
    required_tools: frozenset[str]
    expected_files: dict[str, str] = field(default_factory=dict)


def _number(value: float | int) -> str:
    decimal = Decimal(str(value)).normalize()
    return format(decimal, "f")


def _calculate(expression: str) -> str:
    def evaluate(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("unsupported expression")

    parsed = ast.parse(expression, mode="eval")
    return _number(evaluate(parsed))


class ToolSimulator:
    """Small deterministic environment used for both demonstrations and evaluation."""

    def __init__(self, task: AgentTask):
        self.task = task
        self.files = dict(task.files)
        self.calls: list[Action] = []
        self.errors = 0
        self.finished_answer: str | None = None
        self.plan: dict[str, Any] | None = None
        self.plan_updates: list[dict[str, Any]] = []

    def execute(self, action: Action) -> str:
        self.calls.append(action)
        try:
            result = self._execute(action)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            self.errors += 1
            return f"ERROR: {error}"
        return result

    def _execute(self, action: Action) -> str:
        args = action.arguments
        if action.name == "set_plan":
            goal = self._required_string(args, "goal")
            steps = self._required_string_list(args, "steps", minimum=2)
            self.plan = {"goal": goal, "steps": steps}
            return f"PLAN SET: {len(steps)} steps"
        if action.name == "update_plan":
            completed = self._required_string_list(args, "completed", minimum=1)
            next_action = self._required_string(args, "next")
            evidence = self._required_string(args, "evidence")
            update = {"completed": completed, "next": next_action, "evidence": evidence}
            self.plan_updates.append(update)
            return f"PLAN UPDATED: {len(completed)} completed; next={next_action}"
        if action.name == "list_files":
            given = self._required_string(args, "directory")
            directory = given.rstrip("/")
            paths = sorted(path for path in self.files if path.startswith(directory + "/"))
            if not paths:
                #: Same defect and same reasoning as the pipeline simulator: a flat path set has
                #: no empty directories, so a prefix that matches nothing names a directory that
                #: does not exist, and saying `(none)` tells the model the opposite of the truth.
                raise ValueError(f"directory not found: {given}")
            return "FILES: " + ", ".join(paths)
        if action.name == "read_file":
            path = self._required_string(args, "path")
            if path not in self.files:
                raise ValueError(f"file not found: {path}")
            return self.files[path]
        if action.name == "search_files":
            query = self._required_string(args, "query").casefold()
            paths = sorted(
                path
                for path, content in self.files.items()
                if query in path.casefold() or query in content.casefold()
            )
            return "MATCHES: " + (", ".join(paths) if paths else "(none)")
        if action.name == "calculate":
            expression = self._required_string(args, "expression")
            return "RESULT: " + _calculate(expression)
        if action.name == "replace_text":
            path = self._required_string(args, "path")
            old = self._required_string(args, "old")
            new = self._required_string(args, "new")
            if path not in self.files:
                raise ValueError(f"file not found: {path}")
            if old not in self.files[path]:
                raise ValueError("old text not found")
            self.files[path] = self.files[path].replace(old, new, 1)
            return f"UPDATED: {path}"
        if action.name == "finish":
            self.finished_answer = self._required_string(args, "answer")
            return "FINISHED"
        raise ValueError(f"unknown tool: {action.name}")

    @staticmethod
    def _required_string(arguments: dict[str, Any], key: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty string")
        return value

    @staticmethod
    def _required_string_list(arguments: dict[str, Any], key: str, minimum: int) -> list[str]:
        value = arguments.get(key)
        if (
            not isinstance(value, list)
            or len(value) < minimum
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise ValueError(f"{key} must contain at least {minimum} non-empty strings")
        return value

    @property
    def success(self) -> bool:
        called = {action.name for action in self.calls}
        answer_ok = _normalize(self.finished_answer or "") == _normalize(self.task.expected_answer)
        files_ok = all(
            self.files.get(path) == content for path, content in self.task.expected_files.items()
        )
        return (
            answer_ok
            and files_ok
            and self.task.required_tools.issubset(called)
            and self.errors == 0
        )


def _normalize(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def make_tasks(split: str, count: int, seed: int = 20260902) -> list[AgentTask]:
    """Create deterministic, split-isolated tool-use tasks."""
    categories = ("read", "search", "calculate", "synthesis", "update", "list")
    tasks = []
    for index in range(count):
        rng = random.Random(f"{seed}:{split}:{index}")
        category = categories[index % len(categories)]
        tasks.append(_make_task(split, index, category, rng))
    return tasks


def _make_task(split: str, index: int, category: str, rng: random.Random) -> AgentTask:
    task_id = f"{split}-{category}-{index:04d}"
    root = f"workspace/{split}/{index:04d}"
    if category == "read":
        owner = f"owner-{rng.randrange(1000, 9999)}"
        path = f"{root}/project.md"
        files = {path: f"Project: Atlas-{index}\nOwner: {owner}\nStatus: active"}
        actions = (
            Action("read_file", {"path": path}),
            Action("finish", {"answer": owner}),
        )
        prompt = f"Read {path} and report the Owner value exactly."
        return AgentTask(task_id, category, prompt, files, actions, owner, frozenset({"read_file"}))

    if category == "search":
        token = f"KEY-{split.upper()}-{index:04d}"
        status = rng.choice(("queued", "approved", "blocked", "complete"))
        target = f"{root}/archive/report-{rng.randrange(10, 99)}.txt"
        files = {
            f"{root}/notes.txt": "Routine notes with no matching identifier.",
            target: f"Identifier: {token}\nStatus: {status}\n",
            f"{root}/archive/other.txt": "Status: unknown\n",
        }
        actions = (
            Action("search_files", {"query": token}),
            Action("read_file", {"path": target}),
            Action("finish", {"answer": status}),
        )
        prompt = f"Find the file containing {token}, read it, and report its Status value exactly."
        return AgentTask(
            task_id,
            category,
            prompt,
            files,
            actions,
            status,
            frozenset({"search_files", "read_file"}),
        )

    if category == "calculate":
        left = rng.randrange(12, 90)
        right = rng.randrange(3, 12)
        expression = f"({left} + {right}) * {right}"
        answer = _calculate(expression)
        actions = (
            Action("calculate", {"expression": expression}),
            Action("finish", {"answer": answer}),
        )
        prompt = f"Use the calculator tool to compute {expression}. Return only the numeric result."
        return AgentTask(task_id, category, prompt, {}, actions, answer, frozenset({"calculate"}))

    if category == "synthesis":
        rate = rng.randrange(35, 95)
        hours = rng.randrange(6, 28)
        rate_path = f"{root}/billing/rate.txt"
        hours_path = f"{root}/billing/hours.txt"
        expression = f"{rate} * {hours}"
        answer = _calculate(expression)
        files = {rate_path: f"hourly_rate={rate}", hours_path: f"hours={hours}"}
        actions = (
            Action("read_file", {"path": rate_path}),
            Action("read_file", {"path": hours_path}),
            Action("calculate", {"expression": expression}),
            Action("finish", {"answer": answer}),
        )
        prompt = (
            f"Read {rate_path} and {hours_path}, then use the calculator to compute "
            "the total cost. "
            "Return only the numeric total."
        )
        return AgentTask(
            task_id,
            category,
            prompt,
            files,
            actions,
            answer,
            frozenset({"read_file", "calculate"}),
        )

    if category == "update":
        old_mode, new_mode = rng.sample(("safe", "fast", "audit", "strict"), 2)
        path = f"{root}/config.ini"
        before = f"service=worker-{index}\nmode={old_mode}\nretries=3\n"
        after = before.replace(f"mode={old_mode}", f"mode={new_mode}")
        answer = f"mode={new_mode}"
        actions = (
            Action("read_file", {"path": path}),
            Action(
                "replace_text",
                {"path": path, "old": f"mode={old_mode}", "new": f"mode={new_mode}"},
            ),
            Action("finish", {"answer": answer}),
        )
        prompt = (
            f"In {path}, change mode from {old_mode} to {new_mode}. Inspect the file first and "
            "report the new setting after the update succeeds."
        )
        return AgentTask(
            task_id,
            category,
            prompt,
            {path: before},
            actions,
            answer,
            frozenset({"read_file", "replace_text"}),
            {path: after},
        )

    label = f"milestone-{rng.randrange(100, 999)}"
    directory = f"{root}/notes"
    target = f"{directory}/summary-{index}.md"
    files = {
        target: f"{label}\nThis is the selected summary.",
        f"{directory}/scratch.txt": "ignore this scratch file",
        f"{root}/outside.md": "outside the requested directory",
    }
    actions = (
        Action("list_files", {"directory": directory}),
        Action("read_file", {"path": target}),
        Action("finish", {"answer": label}),
    )
    prompt = (
        f"List files in {directory}, read the Markdown summary there, and report "
        "its first line exactly."
    )
    return AgentTask(
        task_id,
        category,
        prompt,
        files,
        actions,
        label,
        frozenset({"list_files", "read_file"}),
    )
