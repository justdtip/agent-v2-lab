"""The existence family: one workspace, two arms, the state entering only through a tool result.

**Why a new factory and not a thirteenth entry in ``FAMILIES``.** ``make_tasks`` assigns a family
by ``index % len(FAMILIES)`` and derives every task id and R28 fingerprint from that position;
``JSPACE_SPLITS`` records that ``ledger_reconcile`` sits at index 7 of twelve so that naming a split
"changes no task the recorded sweep drew from". Appending a family would move every task after it
in every recorded corpus. So the maker registers in ``_MAKERS`` beside ``_list`` — which is where
the order puts it — and is reached through :func:`make_existence_pairs`, the way the P2 and J-space
splits are reached through their own factories rather than through the cycle.

**The pair.** Arm E has the target file; arm A has the identical workspace without it. Same task,
prompt, paths, distractors and history length. The prompt is the same string in both arms, so
nothing about the state is said to the model: it learns whether the file exists from the listing,
which is a tool observation, and from nothing else.

**The reliability instrument.** A false observation is a :class:`Fault` whose message is a
**well-formed but false** tool result, applied by ``Simulator.execute`` before validation exactly as
a transient fault is. Two false results exist here, one per arm, so the false result always
contradicts the truth: in arm E a listing that omits the file; in arm A a listing that includes it.
The rate and seed are the caller's and go in the manifest, because that rate is the likelihood
ratio the derivation's door model needs.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, replace

from local_llm_lab.agent_protocol import Action
from local_llm_lab.pipeline.env import Fault
from local_llm_lab.pipeline.protocol import Step
from local_llm_lab.pipeline.tasks import Task

FAMILY = "existence"
#: What the finish answer must say when there is no summary. Fixed here so D5 and the verdict read
#: the same string, and so no policy is scored on a paraphrase the diagnostics do not recognise.
NO_SUMMARY = "no summary"


@dataclass(frozen=True)
class ExistencePair:
    """One matched pair: the same episode twice, differing only in whether the target exists."""

    pair_id: str
    exists: Task
    absent: Task
    target: str
    directory: str

    def arms(self) -> Iterator[tuple[str, Task]]:
        yield "E", self.exists
        yield "A", self.absent


def _step(thought: str, name: str, **arguments) -> Step:
    return Step(thought, Action(name, arguments))


def _existence(split: str, index: int, level: int, rng: random.Random) -> Task:
    """The ``_MAKERS`` entry: arm E of the pair, so the cycle probe sees an ordinary task.

    ``applicable_variants`` probes every maker with ``("train", 0, level, rng)`` and reads the step
    names, so this must return a complete task with expert steps. It returns the exists arm; the
    absent arm is derived from it by :func:`absent_arm`, which is how the pair stays byte-matched.
    """
    return make_existence_pairs(split, 1, level, seed=rng.random(), start=index)[0].exists


def absent_arm(exists: Task) -> Task:
    """Arm A from arm E: remove the target, keep everything else identical, rescript the expert."""
    target = _target_of(exists)
    files = {path: content for path, content in exists.files.items() if path != target}
    directory = target.rsplit("/", 1)[0]
    steps = (
        exists.steps[0],
        _step(
            f"Listing shows no Markdown summary in {directory.rsplit('/', 1)[-1]}. Reporting that.",
            "finish",
            answer=NO_SUMMARY,
        ),
    )
    return replace(
        exists,
        task_id=exists.task_id.replace("-E", "-A", 1),
        files=files,
        steps=steps,
        expected_answer=NO_SUMMARY,
        required_tools=frozenset({"list_files"}),
    )


def _target_of(task: Task) -> str:
    targets = [path for path in task.files if path.endswith(".md") and "/notes/" in path]
    if len(targets) != 1:
        raise ValueError(f"an existence task has exactly one target; found {targets}")
    return targets[0]


def make_existence_pairs(
    split: str, count: int, level: int, *, seed, start: int = 0
) -> list[ExistencePair]:
    """``count`` matched pairs. Distractors scale with ``level``; nothing else does."""
    rng = random.Random(f"existence:{split}:{seed}:{start}")
    pairs = []
    for offset in range(count):
        index = start + offset
        root = f"workspace/{split}/{index:04d}"
        directory = f"{root}/notes"
        label = f"milestone-{rng.randrange(100, 999)}"
        target = f"{directory}/summary-{rng.randrange(10, 99)}.md"
        files = {
            target: f"{label}\nThis is the selected summary.",
            f"{directory}/scratch.txt": "ignore this scratch file",
            f"{root}/outside.md": "outside the requested directory",
        }
        for extra in range(level):
            files[f"{directory}/log-{extra}.txt"] = "log entry, not a summary"
        # One string, both arms. The state is never in the prompt.
        prompt = (
            f"List files in {directory}. If a Markdown summary is present there, read it and "
            f"report its first line exactly; if there is none, reply exactly '{NO_SUMMARY}'."
        )
        steps = (
            _step(
                f"Plan: list {directory.rsplit('/', 1)[-1]}, read the .md summary if present, "
                "report its first line or that there is none. Listing.",
                "list_files",
                directory=directory,
            ),
            _step(f"Listing shows one Markdown file: {target}. Reading it.", "read_file", path=target),
            _step(f"First line: {label}; pending: none.", "finish", answer=label),
        )
        exists = Task(
            f"{split}-{FAMILY}-{index:04d}-E",
            FAMILY,
            "clean",
            prompt,
            files,
            steps,
            label,
            frozenset({"list_files", "read_file"}),
            difficulty=level,
        )
        pairs.append(
            ExistencePair(f"{split}-{FAMILY}-{index:04d}", exists, absent_arm(exists), target, directory)
        )
    return pairs


# --------------------------------------------------------------------- the relation test (D10)


@dataclass(frozen=True)
class RelationPair:
    """Two files in one directory with **swapped** states across two episodes (order §4, D10).

    Episode 1: A present, B absent. Episode 2: A absent, B present. One prompt string in both.
    D10 asks whether the action on file B is the one B's state implies and not A's; under swapped
    states "follows B" and "follows A" are logical negations, which is what lets the relation enter
    the derivation table as an ordinary contrast (see :func:`relation_scores`).
    """

    pair_id: str
    episode_1: Task
    episode_2: Task
    file_a: str
    file_b: str
    directory: str

    def episodes(self) -> Iterator[tuple[str, Task]]:
        yield "T1", self.episode_1
        yield "T2", self.episode_2


def _relation_answer(present_label: str, present_name: str, absent_name: str) -> str:
    return f"{present_name}: {present_label}; {absent_name}: {NO_SUMMARY}"


def make_relation_pairs(
    split: str, count: int, level: int, *, seed, start: int = 0
) -> list[RelationPair]:
    """``count`` relation pairs. The two summaries are named in the prompt; their states are not."""
    rng = random.Random(f"relation:{split}:{seed}:{start}")
    pairs = []
    for offset in range(count):
        index = start + offset
        root = f"workspace/{split}/{index:04d}"
        directory = f"{root}/notes"
        label_a, label_b = f"alpha-{rng.randrange(100, 999)}", f"beta-{rng.randrange(100, 999)}"
        name_a, name_b = f"summary-a{rng.randrange(10, 99)}.md", f"summary-b{rng.randrange(10, 99)}.md"
        file_a, file_b = f"{directory}/{name_a}", f"{directory}/{name_b}"
        common = {
            f"{directory}/scratch.txt": "ignore this scratch file",
            f"{root}/outside.md": "outside the requested directory",
        }
        for extra in range(level):
            common[f"{directory}/log-{extra}.txt"] = "log entry, not a summary"
        prompt = (
            f"List files in {directory}. Two summaries may be there, {name_a} and {name_b}. For "
            f"each that is present, read it and report its first line exactly; for each that is "
            f"absent, report '{NO_SUMMARY}'. Answer as '<name>: <line or {NO_SUMMARY}>; <name>: ...'."
        )

        build = dict(split=split, index=index, level=level, prompt=prompt, common=common,
                     directory=directory)
        pairs.append(RelationPair(
            f"{split}-relation-{index:04d}",
            _relation_episode("T1", present=file_a, present_label=label_a, absent=file_b, **build),
            _relation_episode("T2", present=file_b, present_label=label_b, absent=file_a, **build),
            file_a, file_b, directory,
        ))
    return pairs


def _relation_episode(
    tag: str, *, present: str, present_label: str, absent: str, split: str, index: int,
    level: int, prompt: str, common: dict[str, str], directory: str,
) -> Task:
    """One episode of a relation pair. A module-level function rather than a closure in the loop:
    the repository lints B023, and a closure over loop variables is benign only while it is called
    inside its own iteration, which is a property the next edit can silently lose."""
    files = {**common, present: f"{present_label}\nThis is the selected summary."}
    present_name, absent_name = present.rsplit("/", 1)[-1], absent.rsplit("/", 1)[-1]
    answer = _relation_answer(present_label, present_name, absent_name)
    steps = (
        _step(
            f"Plan: list {directory.rsplit('/', 1)[-1]}, read whichever of the two summaries is "
            "present, report both. Listing.",
            "list_files",
            directory=directory,
        ),
        _step(
            f"Listing shows {present_name} present and {absent_name} absent. Reading {present_name}.",
            "read_file",
            path=present,
        ),
        _step(f"{present_name}: {present_label}; {absent_name}: {NO_SUMMARY}. Done.",
              "finish", answer=answer),
    )
    return Task(
        f"{split}-relation-{index:04d}-{tag}", "relation", "clean", prompt, files, steps, answer,
        frozenset({"list_files", "read_file"}), difficulty=level,
    )


# ------------------------------------------------------------------ the reliability instrument


def false_listing(task: Task, *, arm: str, target: str) -> str:
    """A well-formed ``list_files`` result that is false about ``target``, for the given arm.

    ``target`` is the pair's, passed through by the caller and never recovered from the task. The
    first draft parsed it out of the expert's note and listed ``summary-00.md`` in the absent arm,
    a file no diagnostic reads, so the falsification there was invisible to D4 and to the gap that
    defines ε_pred: half the instrument scored nothing. A review found it rather than a test, which
    is why the test now asserts ``target in false_listing(..., arm="A")`` and that a falsified
    absent-arm row's state observation reads as *present* to the diagnostics.
    """
    directory = target.rsplit("/", 1)[0]
    truthful = sorted(path for path in task.files if path.startswith(directory + "/"))
    if arm == "E":
        listed = [p for p in truthful if p != target]
    elif arm == "A":
        listed = sorted({*truthful, target})
    else:
        raise ValueError(f"arm must be 'E' or 'A'; got {arm!r}")
    return "FILES: " + ", ".join(listed)


def with_false_observation(task: Task, *, arm: str, target: str, at_call: int = 0) -> Task:
    """Attach the false listing as a :class:`Fault` at ``at_call`` (the listing call, by default).

    The fault is the environment, applied before validation; the message is a result the model
    cannot distinguish from a true one. Recorded on the task so the transcript's ``faults`` carries
    the index, as every other fault does.
    """
    return replace(
        task,
        variant="false_observation",
        faults=(Fault(call_index=at_call, message=false_listing(task, arm=arm, target=target)),),
    )


def apply_reliability(
    pairs: list[ExistencePair], *, rate: float, seed: int
) -> list[tuple[str, str, Task, bool]]:
    """Every arm of every pair, with a false observation at ``rate``, seeded. Returns
    ``(pair_id, arm, task, was_falsified)`` rows so the manifest can say which episodes were."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"fault rate must lie in [0, 1]; got {rate!r}")
    rng = random.Random(f"reliability:{seed}:{rate}")
    rows = []
    for pair in pairs:
        for arm, task in pair.arms():
            falsify = rng.random() < rate
            chosen = with_false_observation(task, arm=arm, target=pair.target) if falsify else task
            rows.append((pair.pair_id, arm, chosen, falsify))
    return rows


__all__ = [
    "FAMILY",
    "NO_SUMMARY",
    "ExistencePair",
    "RelationPair",
    "make_relation_pairs",
    "absent_arm",
    "apply_reliability",
    "false_listing",
    "make_existence_pairs",
    "with_false_observation",
]
