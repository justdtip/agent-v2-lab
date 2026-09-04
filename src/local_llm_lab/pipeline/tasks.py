from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field, replace
from typing import Any

from local_llm_lab.agent_protocol import Action
from local_llm_lab.agent_tasks import _calculate
from local_llm_lab.pipeline.env import Fault, Simulator
from local_llm_lab.pipeline.protocol import Step

FAMILIES = (
    "read",
    "search",
    "calculate",
    "synthesis",
    "update",
    "list",
    "pointer_chain",
    "ledger_reconcile",
    "cross_reference",
    "conditional_update",
    "batch_update",
    "aggregate_report",
)
LONG_HORIZON_FAMILIES = FAMILIES[6:]
VARIANTS = ("clean", "wrong_path", "transient", "unknown_tool", "stale_path", "failed_edit")
GENERATOR_VERSION = 4

# SPEC-004 §2 probe splits: name, explicit difficulty, perturb. Difficulty is explicit because
# ``_difficulty_level`` would otherwise alternate 0/1 on an unknown split name, and perturb is
# explicit because ``make_tasks`` would otherwise perturb every split that is not valid/test.
# Shape: a 3-tuple per split, the ``MIX_PLAN`` (name, perturb) pair widened by the difficulty
# the P2 design fixes; iterate it as ``for name, level, perturb in P2_SPLITS`` or, preferably,
# call :func:`make_p2_tasks`, which applies both fields for the caller.
P2_SPLITS: tuple[tuple[str, int, bool], ...] = (
    ("p2-d0", 0, True),
    ("p2-d1", 1, True),
    ("p2-d2", 2, False),
)
P2_SPLIT_NAMES: tuple[str, ...] = tuple(name for name, _level, _perturb in P2_SPLITS)

# EXP-001 §3.1 (issue #54): the J-space sweep split, in the same (name, difficulty, perturb)
# shape and for the same reason. ``_difficulty_level`` alternates 0/1 on a split name it does
# not know and ``make_tasks`` perturbs any split that is not valid/test, so a sweep that leaned
# on those defaults could not be replayed from its artifact.
#
# Purely declarative: ``make_tasks`` and ``_difficulty_level`` are untouched, and
# :func:`make_jspace_tasks` is what applies the two fields for the caller, exactly as
# ``make_p2_tasks`` does. Difficulty 1 is not a new choice either -- it is the value the
# alternation already produces for every ``ledger_reconcile`` task of this split, because that
# family sits at index 7 of the twelve ``FAMILIES`` and 7, 19, 31, ... are all odd -- so naming
# it changes no task the recorded sweep drew from.
JSPACE_SPLITS: tuple[tuple[str, int, bool], ...] = (("jsweep", 1, False),)
JSPACE_SPLIT_NAMES: tuple[str, ...] = tuple(name for name, _level, _perturb in JSPACE_SPLITS)
# EXP-001 §3.1: "720 tasks".
JSPACE_SPLIT_LIMIT = 720
# SPEC-004 §2: "120 tasks each".
P2_SPLIT_LIMIT = 120
# The placeholder the split name is rewritten to before a task is fingerprinted (R28).
FINGERPRINT_SPLIT_PLACEHOLDER = "<SPLIT>"


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    variant: str
    prompt: str
    files: dict[str, str]
    steps: tuple[Step, ...]
    expected_answer: str
    required_tools: frozenset[str]
    expected_files: dict[str, str] = field(default_factory=dict)
    faults: tuple[Fault, ...] = ()
    difficulty: int = -1

    @property
    def horizon(self) -> int:
        return len(self.steps)


def _difficulty_level(split: str, index: int, override: int | None) -> int:
    if override is not None:
        if override < 0:
            raise ValueError("difficulty must be non-negative")
        return override
    return {"train": 0, "valid": 1, "test": 2}.get(split, index % 2)


def difficulty(split: str, index: int, override: int | None = None) -> int:
    """train/valid/test are ordered by horizon; rollout splits alternate train/valid difficulty."""
    return _difficulty_level(split, index, override)


_APPLICABLE_VARIANTS_CACHE: dict[tuple[str, int], tuple[str, ...]] = {}


def applicable_variants(family: str, level: int) -> tuple[str, ...]:
    """Which of ``VARIANTS`` a ``family`` can structurally realise, always including ``clean``
    and ``transient``. Determined from a probe task's step shape rather than a hardcoded table,
    so it stays correct if a family's maker changes. Cached per family and level."""
    cache_key = (family, level)
    cached = _APPLICABLE_VARIANTS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    probe = _MAKERS[family]("train", 0, level, random.Random(f"probe:{level}"))
    names = [step.action.name for step in probe.steps]
    has_trailing_read = any(name == "read_file" for name in names[1:])
    starts_with_list = bool(names) and names[0] == "list_files"
    has_replace = any(name == "replace_text" for name in names)
    list_indices = [i for i, name in enumerate(names) if name == "list_files"]
    read_indices = [i for i, name in enumerate(names) if name == "read_file"]
    has_stale = any(k >= i + 3 for i in list_indices for k in read_indices)
    supported = {
        "clean": True,
        "wrong_path": has_trailing_read or starts_with_list,
        "transient": True,
        "unknown_tool": has_replace,
        "stale_path": has_stale,
        "failed_edit": has_replace,
    }
    result = tuple(variant for variant in VARIANTS if supported[variant])
    _APPLICABLE_VARIANTS_CACHE[cache_key] = result
    return result


def make_tasks(
    split: str,
    count: int,
    seed: int = 20260902,
    *,
    perturb: bool | None = None,
    difficulty: int | None = None,
) -> list[Task]:
    """Deterministic, split-isolated tasks. Training splits interleave recovery variants."""
    if count < 1:
        raise ValueError("count must be positive")
    if perturb is None:
        perturb = split not in {"valid", "test"}
    tasks = []
    for index in range(count):
        rng = random.Random(f"v2:{seed}:{split}:{index}")
        family = FAMILIES[index % len(FAMILIES)]
        level = _difficulty_level(split, index, difficulty)
        draft = _MAKERS[family](split, index, level, rng)
        if perturb:
            applicable = applicable_variants(family, level)
            variant = applicable[(index // len(FAMILIES)) % len(applicable)]
        else:
            variant = "clean"
        task = _apply_variant(draft, variant, rng)
        assert task.variant == variant, (
            f"{family}: requested variant {variant!r} but realised {task.variant!r} "
            f"(applicable_variants fell back unexpectedly)"
        )
        tasks.append(
            replace(
                task,
                task_id=f"{split}-{family}-{index:04d}-{task.variant}",
                difficulty=level,
            )
        )
    return tasks


def family_balanced_tasks(
    split: str,
    *,
    difficulty: int,
    per_family: dict[str, int],
    seed: int = 20260902,
) -> list[Task]:
    """Return clean tasks with the requested default and long-horizon family quotas.

    The selector wraps the unchanged deterministic generator so screening cannot alter task rows.
    """
    unknown = set(per_family) - {"default", "long"}
    if unknown:
        raise ValueError(f"unknown family quotas: {', '.join(sorted(unknown))}")
    quotas = {key: per_family.get(key, 0) for key in ("default", "long")}
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in quotas.values()):
        raise ValueError("family quotas must be non-negative integers")
    maximum = max(quotas.values())
    if maximum == 0:
        return []
    candidates = make_tasks(
        split,
        maximum * len(FAMILIES),
        seed,
        perturb=False,
        difficulty=difficulty,
    )
    seen = {family: 0 for family in FAMILIES}
    selected = []
    for task in candidates:
        category = "long" if task.family in LONG_HORIZON_FAMILIES else "default"
        if seen[task.family] < quotas[category]:
            selected.append(task)
            seen[task.family] += 1
    return selected


def p2_split(name: str) -> tuple[str, int, bool]:
    """Return the ``P2_SPLITS`` entry for ``name``, or raise if it is not a P2 split."""
    for entry in P2_SPLITS:
        if entry[0] == name:
            return entry
    known = ", ".join(P2_SPLIT_NAMES)
    raise ValueError(f"{name!r} is not a P2 probe split (known: {known})")


def make_p2_tasks(name: str, limit: int, seed: int) -> list[Task]:
    """Generate one SPEC-004 §2 probe split with its pre-registered difficulty and perturbation.

    ``limit`` is explicit rather than defaulted so a test can ask for a handful of rows; the
    designed size is ``P2_SPLIT_LIMIT``. ``seed`` is explicit for the same reason every other
    entry point takes one: the data seed belongs to the config or the CLI, never to source
    (briefing §1.8). Nothing else about the derivation changes -- the rows come from the same
    ``make_tasks`` generator, so the split name alone separates them from training rows.
    """
    _name, level, perturb = p2_split(name)
    return make_tasks(_name, limit, seed, perturb=perturb, difficulty=level)


def jspace_split(name: str) -> tuple[str, int, bool]:
    """Return the ``JSPACE_SPLITS`` entry for ``name``, or raise if it is not a sweep split."""
    for entry in JSPACE_SPLITS:
        if entry[0] == name:
            return entry
    known = ", ".join(JSPACE_SPLIT_NAMES)
    raise ValueError(f"{name!r} is not a J-space sweep split (known: {known})")


def make_jspace_tasks(name: str, count: int, seed: int) -> list[Task]:
    """Generate the EXP-001 §3.1 sweep split with its declared difficulty and perturbation.

    The counterpart of :func:`make_p2_tasks`: the helper, not the caller, carries the two
    fields the split table fixes, so an artifact that records the split name and the seed
    records everything needed to regenerate the probe points (R12/R23 replay).
    """
    _name, level, perturb = jspace_split(name)
    return make_tasks(_name, count, seed, perturb=perturb, difficulty=level)


def split_of_task_id(task_id: str) -> str:
    """The split name a ``make_tasks`` id was minted under (``{split}-{family}-{index}-{variant}``)."""
    try:
        split, family, _index, variant = task_id.rsplit("-", 3)
    except ValueError as error:
        raise ValueError(f"{task_id}: invalid task id") from error
    if not split or family not in FAMILIES or variant not in VARIANTS:
        raise ValueError(f"{task_id}: invalid task id")
    return split


def _normalise_split_tokens(text: str, split: str) -> str:
    """Replace the two places a split name is embedded in generated content (R28).

    Workspace roots ``workspace/<split>/`` and ``lab/<split>/`` (``tasks.py`` family builders)
    and the ``KEY-<SPLIT>-`` / ``REF-<SPLIT>-`` lookup tokens (`_search`, `_cross_reference`)
    are the only split-derived text in a task's prompt, files, or expected answer; everything
    else that differs between splits differs because the RNG is seeded by the split name, which
    is the content difference the fingerprint is meant to detect.
    """
    placeholder = FINGERPRINT_SPLIT_PLACEHOLDER
    text = re.sub(
        rf"(?<![0-9A-Za-z_])(workspace|lab)/{re.escape(split)}/",
        rf"\1/{placeholder}/",
        text,
    )
    return re.sub(
        rf"(?<![0-9A-Za-z_])(KEY|REF)-{re.escape(split.upper())}-",
        rf"\1-{placeholder}-",
        text,
    )


def task_fingerprint(task: Task) -> str:
    """Content identity of a task with its split erased (SPEC-004 §2, ruling R28).

    SHA-256 over canonical JSON of family, difficulty, variant, the prompt, the sorted files,
    and the expected answer, with the split root and ``KEY-``/``REF-`` tokens normalised to
    ``FINGERPRINT_SPLIT_PLACEHOLDER``. ``task_id`` is excluded (it names the split by
    construction) and so are the expert steps (they are a rendering of the same inputs). Two
    tasks share a fingerprint exactly when they pose the same problem over the same files, so
    equality across two splits is a genuine content overlap, not a naming coincidence.
    """
    split = split_of_task_id(task.task_id)
    payload = {
        "family": task.family,
        "difficulty": int(task.difficulty),
        "variant": task.variant,
        "prompt": _normalise_split_tokens(task.prompt, split),
        "files": sorted(
            [_normalise_split_tokens(path, split), _normalise_split_tokens(content, split)]
            for path, content in task.files.items()
        ),
        "expected_answer": _normalise_split_tokens(task.expected_answer, split),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def task_from_id(
    task_id: str,
    seed: int,
    difficulty: int | None = None,
) -> Task:
    return _task_from_id(task_id, seed, difficulty)


def replay_task_from_id(
    task_id: str,
    seed: int,
    generator_version: int,
    difficulty: int | None = None,
) -> Task:
    """Rebuild one historical note template while retaining HEAD task structure."""
    version = _validate_generator_version(generator_version)
    task = _task_from_id(task_id, seed, difficulty)
    if version == GENERATOR_VERSION:
        return task
    if task.variant == "clean":
        return _replay_clean_notes(task, version)
    clean_id = task_id.rsplit("-", 1)[0] + "-clean"
    clean = _replay_clean_notes(_task_from_id(clean_id, seed, difficulty), version)
    return _replay_recovery_notes(_with_historical_base_notes(task, clean), version)


def _validate_generator_version(generator_version: int) -> int:
    if type(generator_version) is not int or not 1 <= generator_version <= GENERATOR_VERSION:
        raise ValueError(f"invalid generator_version {generator_version!r}")
    return generator_version


def _task_from_id(
    task_id: str,
    seed: int,
    difficulty: int | None = None,
) -> Task:
    try:
        split, family, index_text, variant = task_id.rsplit("-", 3)
    except ValueError as error:
        raise ValueError(f"{task_id}: invalid task id") from error
    if not split or family not in FAMILIES or variant not in VARIANTS:
        raise ValueError(f"{task_id}: invalid task id")
    try:
        index = int(index_text)
    except ValueError as error:
        raise ValueError(f"{task_id}: invalid task index") from error
    if index < 0:
        raise ValueError(f"{task_id}: invalid task index")
    if family != FAMILIES[index % len(FAMILIES)]:
        raise ValueError(f"{task_id}: family does not match task index")
    try:
        level = _difficulty_level(split, index, difficulty)
    except ValueError as error:
        raise ValueError(f"{task_id}: {error}") from error
    rng = random.Random(f"v2:{seed}:{split}:{index}")
    draft = _MAKERS[family](split, index, level, rng)
    try:
        task = _apply_variant(draft, variant, rng)
    except (IndexError, RuntimeError) as error:
        raise ValueError(f"{task_id}: variant {variant!r} is not structurally possible") from error
    if task.variant != variant:
        raise ValueError(f"{task_id}: variant {variant!r} realised as {task.variant!r}")
    return replace(task, task_id=task_id, difficulty=level)


def _replay_clean_notes(task: Task, version: int) -> Task:
    """Render v1/v2 clean notes from the current task's unchanged files and actions.

    Version 1 and 2 share clean notes.  Version 3 is Run-D; version 4 only revises the
    state-preserving terminal-read wording.
    """
    if version == 4:
        return task
    if version == 3:
        return _replay_v3_clean_notes(task)
    steps = list(task.steps)
    answer = task.expected_answer
    finish = {
        "read": f"The file shows Owner: {answer}. Task complete.",
        "search": f"The matching file shows Status: {answer}. Task complete.",
        "calculate": f"Calculator result is {answer}. Task complete.",
        "synthesis": f"Total cost is {answer}. Task complete.",
        "update": f"Verified the file now shows {answer}. Task complete.",
        "list": f"First line of the summary is {answer}. Task complete.",
        "pointer_chain": f"The final node reports Result: {answer}. Task complete.",
    }.get(task.family)
    if finish is not None:
        steps[-1] = replace(steps[-1], thought=finish)
    else:
        {
            "ledger_reconcile": _replay_v2_ledger,
            "cross_reference": _replay_v2_cross_reference,
            "conditional_update": _replay_v2_conditional,
            "batch_update": _replay_v2_batch,
            "aggregate_report": _replay_v2_aggregate,
        }[task.family](task, steps)
    return replace(task, steps=tuple(steps))


def _replay_v3_clean_notes(task: Task) -> Task:
    """Undo only 83ca7e1's clean-row state-preserving corrections."""
    steps = list(task.steps)
    if task.family == "ledger_reconcile":
        for index, step in enumerate(steps):
            steps[index] = replace(
                step,
                thought=step.thought.replace(
                    "then sum the approved amounts and update the summary.", "pending: none."
                ),
            )
    elif task.family == "conditional_update":
        steps[0] = replace(steps[0], thought=steps[0].thought.removeprefix("loads so far: none. "))
        steps[1] = replace(steps[1], thought=steps[1].thought.removeprefix("loads so far: none. "))
        for index, step in enumerate(steps):
            steps[index] = replace(
                step,
                thought=step.thought.replace("then select the highest load.", "pending: none."),
            )
    elif task.family == "aggregate_report":
        for index, step in enumerate(steps):
            steps[index] = replace(
                step,
                thought=step.thought.replace(
                    "then calculate the two subtotals.", "pending: none."
                ),
            )
    return replace(task, steps=tuple(steps))


def _replay_recovery_notes(task: Task, version: int) -> Task:
    """Retain current recovery actions while selecting their historical note profile."""
    steps = list(task.steps)
    if task.variant == "wrong_path":
        _replay_wrong_path_notes(steps, version)
    elif task.variant == "transient":
        _replay_transient_notes(steps, task.faults)
    elif task.variant == "unknown_tool":
        _replay_unknown_tool_notes(steps)
    elif task.variant == "stale_path":
        _replay_stale_path_notes(steps, version)
    elif task.variant == "failed_edit":
        _replay_failed_edit_notes(steps, version)
    return replace(task, steps=tuple(steps))


def _with_historical_base_notes(task: Task, clean: Task) -> Task:
    """Put historical clean thoughts back onto the current recovery action sequence."""
    steps = list(task.steps)
    if task.variant == "transient":
        return _with_historical_transient_notes(task, clean)
    bad = next(index for index, step in enumerate(steps) if not step.supervise)
    skip = {bad}
    if task.variant in {"stale_path", "failed_edit"}:
        skip.add(bad + 1)
    base = iter(clean.steps)
    for index, step in enumerate(steps):
        if index in skip:
            continue
        historical = next(base)
        if step.action != historical.action:
            raise RuntimeError(f"{task.task_id}: recovery no longer matches clean action structure")
        steps[index] = replace(step, thought=historical.thought)
    try:
        next(base)
    except StopIteration:
        return replace(task, steps=tuple(steps))
    raise RuntimeError(f"{task.task_id}: recovery omitted a clean action")


def _with_historical_transient_notes(task: Task, clean: Task) -> Task:
    """A transient fault duplicates one supervised call instead of adding a bad call."""
    steps = list(task.steps)
    fault_index = task.faults[0].call_index
    for index, step in enumerate(steps):
        source = clean.steps[fault_index] if index == fault_index + 1 else clean.steps[
            index - (index > fault_index + 1)
        ]
        if step.action != source.action:
            raise RuntimeError(f"{task.task_id}: transient no longer matches clean action structure")
        steps[index] = replace(step, thought=source.thought)
    return replace(task, steps=tuple(steps))


def _replay_wrong_path_notes(steps: list[Step], version: int) -> None:
    bad = next(index for index, step in enumerate(steps) if not step.supervise)
    guess = steps[bad].action.arguments["path"]
    recovery = steps[bad + 1]
    if bad == 0:
        thought = (
            steps[bad + 1].thought.replace("Listing.", "Trying a likely file first.")
            if version == 1
            else f"Trying guessed path {guess} before listing the directory."
        )
        steps[bad] = replace(steps[bad], thought=thought)
        steps[bad + 1] = replace(
            recovery,
            thought=(
                "That guessed path does not exist. List the directory first, then use only the exact paths it reports. "
                + recovery.thought
            ),
        )
        return
    if version == 1:
        # v1 used the interrupted thought; v2 introduced the explicit guessed-path sentence.
        steps[bad] = replace(steps[bad], thought=recovery.thought.rsplit(" instead of guessing: ", 1)[-1])
    else:
        steps[bad] = replace(steps[bad], thought=f"Trying guessed path {guess}.")
    steps[bad + 1] = replace(
        recovery,
        thought=(
            f"That path does not exist; use the exact path from the earlier tool result instead of guessing: "
            f"{recovery.action.arguments['path']}. {recovery.thought}"
        ),
    )


def _replay_stale_path_notes(steps: list[Step], version: int) -> None:
    bad = next(index for index, step in enumerate(steps) if not step.supervise)
    guess = steps[bad].action.arguments["path"]
    if version == 1:
        recovered = steps[bad + 2].thought.removeprefix("The listing gives the exact name. ")
        steps[bad] = replace(steps[bad], thought=recovered)
    else:
        steps[bad] = replace(steps[bad], thought=f"Trying stale guessed path {guess}.")
    steps[bad + 1] = replace(
        steps[bad + 1],
        thought=(
            "That path does not exist and the earlier listing is no longer visible, so list the "
            "directory again for the exact names instead of guessing."
        ),
    )
    recovered = steps[bad + 2]
    steps[bad + 2] = replace(
        recovered, thought="The listing gives the exact name. " + recovered.thought
    )


def _replay_failed_edit_notes(steps: list[Step], _version: int) -> None:
    bad = next(index for index, step in enumerate(steps) if not step.supervise)
    reread = steps[bad + 1]
    steps[bad + 1] = replace(
        reread,
        thought=(
            "The replacement failed because that exact text is not in the file, so re-read it to "
            "see its current contents instead of retrying the same edit."
        ),
    )
    recovered = steps[bad + 2]
    steps[bad] = replace(steps[bad], thought=recovered.thought)
    steps[bad + 2] = replace(
        recovered,
        thought="The file's current text confirms the exact string to replace. " + recovered.thought,
    )


def _replay_transient_notes(steps: list[Step], faults: tuple[Fault, ...]) -> None:
    retry = faults[0].call_index + 1
    steps[retry] = replace(
        steps[retry],
        thought="The tool reported a transient failure; the call itself was correct, so retry it unchanged. "
        + steps[retry].thought,
    )


def _replay_unknown_tool_notes(steps: list[Step]) -> None:
    bad = next(index for index, step in enumerate(steps) if not step.supervise)
    recovered = steps[bad + 1]
    bogus = steps[bad].action.name
    steps[bad] = replace(steps[bad], thought=recovered.thought)
    steps[bad + 1] = replace(
        recovered,
        thought=(
            f"{bogus} is not an available tool; the workspace exposes replace_text. Apply the same exact replacement with it. "
            + recovered.thought
        ),
    )


def _replay_v2_ledger(task: Task, steps: list[Step]) -> None:
    invoice_paths = sorted(path for path in task.files if "/invoice-" in path)
    approved: list[int] = []
    held: list[int] = []
    invoice_steps = [index for index, step in enumerate(steps) if "/invoice-" in step.action.arguments.get("path", "")]
    for position, (step_index, path) in enumerate(zip(invoice_steps, invoice_paths, strict=True)):
        amount = int(re.search(r"amount=(\d+)", task.files[path]).group(1))
        status = re.search(r"status=(\w+)", task.files[path]).group(1)
        steps[step_index] = replace(
            steps[step_index],
            thought=(
                f"Invoices read: {position} of {len(invoice_paths)}. approved: {_join(approved)}; "
                f"held (skip): {_join(held)}. Reading {_short(path)}; {_pending(invoice_paths[position + 1:])}."
            ),
        )
        (approved if status == "approved" else held).append(amount)
    summary = next(path for path in task.files if path.endswith("/summary.txt"))
    total = str(sum(approved))
    action_indices = {step.action.name: [] for step in steps}
    for index, step in enumerate(steps):
        action_indices.setdefault(step.action.name, []).append(index)
    calculate = action_indices["calculate"][-1]
    report_reads = [index for index in action_indices["read_file"] if steps[index].action.arguments["path"] == summary]
    replace_index = action_indices["replace_text"][-1]
    steps[calculate] = replace(steps[calculate], thought=(f"All {len(invoice_paths)} invoices read. approved: {_join(approved)}; held skipped: {_join(held)}. Summing approved amounts."))
    steps[report_reads[0]] = replace(steps[report_reads[0]], thought=f"Approved total = {total}. Inspecting the summary before editing.")
    steps[replace_index] = replace(steps[replace_index], thought=f"Summary contains approved_total=PENDING. Replacing PENDING with {total}.")
    steps[report_reads[-1]] = replace(steps[report_reads[-1]], thought="Replacement confirmed by the tool. Re-reading the summary to verify.")
    steps[-1] = replace(steps[-1], thought=f"Verified the summary reads approved_total={total}. Task complete.")


def _replay_v2_cross_reference(task: Task, steps: list[Step]) -> None:
    searches = [index for index, step in enumerate(steps) if step.action.name == "search_files"]
    targets = [steps[index + 1].action.arguments["path"] for index in searches]
    for hop, (search_index, target) in enumerate(zip(searches, targets, strict=True)):
        key = steps[search_index].action.arguments["query"]
        search_note = (
            f"Plan: search each key, read the matching record, follow Next-Key until a Resolution appears. Searching {key}."
            if hop == 0
            else f"Hop {hop}: record gave Next-Key {key}, no Resolution yet. Searching for it; the file whose Lookup-Key equals it is the new one."
        )
        matches = _search_matches(task.files, key)
        steps[search_index] = replace(steps[search_index], thought=search_note)
        steps[search_index + 1] = replace(
            steps[search_index + 1], thought=_match_note(matches, target, targets[:hop])
        )
    steps[-1] = replace(steps[-1], thought=f"The record shows Resolution: {task.expected_answer}. Task complete.")


def _replay_v2_conditional(task: Task, steps: list[Step]) -> None:
    policy = next(path for path in task.files if path.endswith("/policy.txt"))
    threshold = int(re.search(r"threshold=(\d+)", task.files[policy]).group(1))
    services = sorted(path for path in task.files if "/service-" in path)
    loads = [int(re.search(r"load=(\d+)", task.files[path]).group(1)) for path in services]
    steps[0] = replace(steps[0], thought="Plan: read the policy threshold, list services, read every service load, throttle the highest load if above threshold, re-read to verify, report. Reading policy.")
    steps[1] = replace(steps[1], thought=f"threshold={threshold}. Listing the service files.")
    best_index: int | None = None
    best_load: int | None = None
    for position, path in enumerate(services):
        step_index = next(index for index, step in enumerate(steps) if step.action.arguments.get("path") == path)
        label = f"service-{best_index}={best_load}" if best_index is not None else "none"
        steps[step_index] = replace(steps[step_index], thought=(f"threshold={threshold}. highest so far: {label}. Reading service {position + 1} of {len(services)}: {_short(path)}; {_pending(services[position + 1:])}."))
        if best_load is None or loads[position] > best_load:
            best_index, best_load = position, loads[position]
    target = services[best_index]
    replace_index = next(index for index, step in enumerate(steps) if step.action.name == "replace_text")
    verify_index = next(index for index, step in enumerate(steps[replace_index + 1:], replace_index + 1) if step.action.name == "read_file")
    steps[replace_index] = replace(steps[replace_index], thought=(f"threshold={threshold}. highest so far: service-{best_index}={best_load} (final), above threshold, so throttle {_short(target)}."))
    steps[verify_index] = replace(steps[verify_index], thought=f"Replacement confirmed for service-{best_index}. Re-reading to verify.")
    steps[-1] = replace(steps[-1], thought=f"Verified service-{best_index} now has mode=throttled. Task complete.")


def _replay_v2_batch(task: Task, steps: list[Step]) -> None:
    manifest = next(path for path in task.files if path.endswith("/manifest.txt"))
    targets = []
    for line in task.files[manifest].splitlines():
        path, modes = line.split("|", 1)
        old, new = modes.removeprefix("mode=").split("->mode=", 1)
        targets.append((path, old, new))
    count = len(targets)
    worker_paths = {path for path, _, _ in targets}
    reads = [
        index
        for index, step in enumerate(steps)
        if step.action.name == "read_file" and step.action.arguments.get("path") in worker_paths
    ]
    inspect, verify = reads[:count], reads[count:]
    for number, (step_index, target) in enumerate(zip(inspect, targets, strict=True)):
        prefix = (
            f"Phase inspect begins; manifest read, {count} workers queued. "
            if number == 0
            else "Phase inspect. "
        )
        steps[step_index] = replace(
            steps[step_index],
            thought=(f"{prefix}Next: {_queue_line(target)}. Remaining after this: {_queue_tail(targets[number + 1:])}."),
        )
    replacements = [index for index, step in enumerate(steps) if step.action.name == "replace_text"]
    for number, (step_index, target) in enumerate(zip(replacements, targets, strict=True)):
        prefix = (
            f"Phase apply begins; all {count} workers inspected. "
            if number == 0
            else "Phase apply. "
        )
        steps[step_index] = replace(
            steps[step_index],
            thought=(f"{prefix}Next: {_queue_line(target)}. Remaining after this: {_queue_tail(targets[number + 1:])}."),
        )
    for number, (step_index, target) in enumerate(zip(verify, targets, strict=True)):
        prefix = (
            f"Phase apply complete, all {count} replacements confirmed; phase verify begins. "
            if number == 0
            else "Phase verify. "
        )
        steps[step_index] = replace(
            steps[step_index],
            thought=(f"{prefix}Next: {_verify_line(target)}. Remaining after this: {_verify_tail(targets[number + 1:])}."),
        )
    steps[-1] = replace(
        steps[-1], thought=f"Phase verify complete; all {count} workers verified. Task complete."
    )


def _replay_v2_aggregate(task: Task, steps: list[Step]) -> None:
    metrics = sorted(path for path in task.files if "/metric-" in path)
    values = [int(task.files[path].rsplit("value=", 1)[1]) for path in metrics]
    split = len(values) // 2
    first_half: list[int] = []
    second_half: list[int] = []
    metric_steps = [
        index
        for index, step in enumerate(steps)
        if step.action.name == "read_file" and step.action.arguments.get("path") in metrics
    ]
    for position, (step_index, path) in enumerate(zip(metric_steps, metrics, strict=True)):
        first_label = (
            f"{_join(first_half)} (full)" if len(first_half) == split else _join(first_half)
        )
        steps[step_index] = replace(
            steps[step_index],
            thought=(f"first half: {first_label}; second half: {_join(second_half)}. Reading metric {position + 1} of {len(metrics)}: {_short(path)}; {_pending(metrics[position + 1:])}."),
        )
        (first_half if position < split else second_half).append(values[position])
    first_expression = " + ".join(map(str, values[:split]))
    second_expression = " + ".join(map(str, values[split:]))
    first_total, second_total = sum(values[:split]), sum(values[split:])
    grand_total = first_total + second_total
    calculations = [index for index, step in enumerate(steps) if step.action.name == "calculate"]
    steps[calculations[0]] = replace(steps[calculations[0]], thought=(f"first half complete: {first_expression}; second half complete: {second_expression}. Computing the first subtotal."))
    steps[calculations[1]] = replace(steps[calculations[1]], thought=(f"First subtotal = {first_total}. Computing the second subtotal {second_expression}."))
    steps[calculations[2]] = replace(steps[calculations[2]], thought=f"Subtotals {first_total} and {second_total}. Adding them for the grand total.")
    report = next(path for path in task.files if path.endswith("/report.txt"))
    report_reads = [index for index, step in enumerate(steps) if step.action.name == "read_file" and step.action.arguments.get("path") == report]
    replacement = next(index for index, step in enumerate(steps) if step.action.name == "replace_text")
    steps[report_reads[0]] = replace(steps[report_reads[0]], thought=f"Grand total = {grand_total}. Inspecting report.txt before editing.")
    steps[replacement] = replace(steps[replacement], thought=f"Report contains grand_total=PENDING. Replacing PENDING with {grand_total}.")
    steps[report_reads[-1]] = replace(steps[report_reads[-1]], thought="Replacement confirmed by the tool. Re-reading the report to verify.")
    steps[-1] = replace(steps[-1], thought=f"Verified the report reads grand_total={grand_total}. Task complete.")


def render_expert_note(task: Task, step_index: int) -> str:
    """Return the canonical ground-truth note for one expert step.

    Keeping the renderer at the task seam lets later consumers reuse the generator's notes
    without copying family templates or reconstructing task-local state.
    """
    if step_index < 0 or step_index >= len(task.steps):
        raise IndexError(f"{task.task_id}: step index {step_index} is out of range")
    return task.steps[step_index].thought


# --------------------------------------------------------------------------- helpers


def _step(thought: str, name: str, **arguments: Any) -> Step:
    return Step(thought, Action(name, arguments))


def _noise(rng: random.Random, lines: int, label: str) -> str:
    return "".join(
        f"context-{label}-{line}=historical observation {rng.randrange(100000, 999999)} "
        f"for unrelated component {rng.choice(('amber', 'blue', 'green', 'silver'))}\n"
        for line in range(lines)
    )


def _join(values: list[Any]) -> str:
    return ", ".join(map(str, values)) if values else "none"


def _short(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _pending(paths: list[str]) -> str:
    """``pending: a, b`` naming the files still to visit (short names), or ``pending: none``.

    Notes are the policy's only memory once a listing is hidden, so every read note must
    carry the paths that remain instead of a bare count.
    """
    return "pending: " + (", ".join(map(_short, paths)) if paths else "none")


def _queue_line(item: tuple[str, str, str]) -> str:
    """``worker-2.ini mode=audit -> mode=fast``: a worker's short name with its manifest
    old->new pair attached, so the pair is always adjacent to the file it applies to."""
    path, old, new = item
    return f"{_short(path)} mode={old} -> mode={new}"


def _verify_line(item: tuple[str, str, str]) -> str:
    """``worker-2.ini, expect mode=fast``: during verification the edit is already applied, so
    the note states the expected current state. Naming an ``old -> new`` transition here reads
    as an instruction to apply it again, which is exactly the oscillation this family fell into
    (repeatedly flipping the last worker between two modes, each edit succeeding silently)."""
    path, _old, new = item
    return f"{_short(path)}, expect mode={new}"


def _verify_tail(items: list[tuple[str, str, str]]) -> str:
    return "; ".join(map(_verify_line, items)) if items else "none"


def _queue_tail(items: list[tuple[str, str, str]]) -> str:
    """``worker-2.ini mode=audit -> mode=fast; worker-3.ini ...`` for the items still in the
    queue after the current head, or ``none`` -- never a count or index to look up."""
    return "; ".join(map(_queue_line, items)) if items else "none"


def _search_matches(files: dict[str, str], query: str) -> list[str]:
    """Exactly what ``Simulator`` returns for ``search_files``: every path whose name or
    content contains ``query`` case-insensitively, sorted."""
    needle = query.casefold()
    return sorted(
        path
        for path, content in files.items()
        if needle in path.casefold() or needle in content.casefold()
    )


def _match_note(matches: list[str], target: str, already_read: list[str]) -> str:
    """Read note after a search: single match, or several with the rule for choosing."""
    if target not in matches:
        raise RuntimeError(f"search would not match the target {target}")
    if len(matches) == 1:
        return f"Search matched {_short(target)}. Reading it."
    others = [match for match in matches if match != target]
    if any(other not in already_read for other in others):
        raise RuntimeError(f"unexpected extra search matches for {target}: {others}")
    verb = "is" if len(others) == 1 else "are"
    return (
        f"Search matched {len(matches)} files; {_join([_short(o) for o in others])} {verb} "
        f"already read, so reading the new one: {_short(target)}."
    )


# --------------------------------------------------------------------------- simple families


def _read(split: str, index: int, level: int, rng: random.Random) -> Task:
    root = f"workspace/{split}/{index:04d}"
    owner = f"owner-{rng.randrange(1000, 9999)}"
    path = f"{root}/project.md"
    files = {
        path: f"Project: Atlas-{index}\nOwner: {owner}\n{_noise(rng, 2 + level, 'p')}Status: active"
    }
    prompt = rng.choice(
        [
            f"Read {path} and report the Owner value exactly.",
            f"Open {path}. What is the Owner field? Reply with only that value.",
            f"Report exactly the Owner value stored in {path}.",
        ]
    )
    steps = (
        _step(
            f"Plan: read {_short(path)} and report its Owner field exactly.", "read_file", path=path
        ),
        _step(f"Owner: {owner}; pending: none.", "finish", answer=owner),
    )
    return Task(
        f"{split}-read-{index:04d}",
        "read",
        "clean",
        prompt,
        files,
        steps,
        owner,
        frozenset({"read_file"}),
    )


def _search(split: str, index: int, level: int, rng: random.Random) -> Task:
    root = f"workspace/{split}/{index:04d}"
    token = f"KEY-{split.upper()}-{index:04d}-{rng.randrange(100, 999)}"
    status = rng.choice(("queued", "approved", "blocked", "complete"))
    target = f"{root}/archive/report-{rng.randrange(10, 99)}.txt"
    files = {
        f"{root}/notes.txt": "Routine notes with no matching identifier.",
        target: f"Identifier: {token}\n{_noise(rng, 2 + level, 'r')}Status: {status}\n",
        f"{root}/archive/other.txt": "Status: unknown\n",
    }
    prompt = rng.choice(
        [
            f"Find the file containing {token}, read it, and report its Status value exactly.",
            f"Locate the file that mentions {token} and tell me exactly its Status.",
            f"Search the workspace for {token}; report the Status field of the matching file.",
        ]
    )
    steps = (
        _step(
            f"Plan: search for {token}, read the matching file, report its Status. Searching.",
            "search_files",
            query=token,
        ),
        _step(
            f"Search matched {target}. Reading it for the Status field.", "read_file", path=target
        ),
        _step(f"Status: {status}; pending: none.", "finish", answer=status),
    )
    return Task(
        f"{split}-search-{index:04d}",
        "search",
        "clean",
        prompt,
        files,
        steps,
        status,
        frozenset({"search_files", "read_file"}),
    )


def _calc(split: str, index: int, level: int, rng: random.Random) -> Task:
    left = rng.randrange(12, 90)
    right = rng.randrange(3, 12)
    extra = rng.randrange(2, 9)
    expression = (
        f"({left} + {right}) * {right}" if level == 0 else f"({left} + {right}) * {right} - {extra}"
    )
    answer = _calculate(expression)
    prompt = rng.choice(
        [
            f"Use the calculator tool to compute {expression}. Return only the numeric result.",
            f"Compute {expression} with the calculator and reply with just the number.",
        ]
    )
    steps = (
        _step(
            f"Plan: evaluate {expression} with the calculator, then report the number.",
            "calculate",
            expression=expression,
        ),
        _step(f"Calculator result: {answer}; pending: none.", "finish", answer=answer),
    )
    return Task(
        f"{split}-calculate-{index:04d}",
        "calculate",
        "clean",
        prompt,
        {},
        steps,
        answer,
        frozenset({"calculate"}),
    )


def _synthesis(split: str, index: int, level: int, rng: random.Random) -> Task:
    root = f"workspace/{split}/{index:04d}/billing"
    rate = rng.randrange(35, 95)
    hours = rng.randrange(6, 28)
    rate_path, hours_path = f"{root}/rate.txt", f"{root}/hours.txt"
    expression = f"{rate} * {hours}"
    answer = _calculate(expression)
    files = {
        rate_path: f"{_noise(rng, level, 'a')}hourly_rate={rate}\n",
        hours_path: f"hours={hours}\n{_noise(rng, level, 'b')}",
    }
    prompt = rng.choice(
        [
            f"Read {rate_path} and {hours_path}, then use the calculator to compute the total cost. Return only the numeric total.",
            f"Multiply the hourly_rate in {rate_path} by the hours in {hours_path} using the calculator; reply with only the number.",
        ]
    )
    steps = (
        _step(
            "Plan: read the rate file, read the hours file, multiply with the calculator, report the number. Reading rate.",
            "read_file",
            path=rate_path,
        ),
        _step(f"hourly_rate={rate}. Reading hours next.", "read_file", path=hours_path),
        _step(
            f"hourly_rate={rate}, hours={hours}. Computing {expression}.",
            "calculate",
            expression=expression,
        ),
        _step(f"Total cost: {answer}; pending: none.", "finish", answer=answer),
    )
    return Task(
        f"{split}-synthesis-{index:04d}",
        "synthesis",
        "clean",
        prompt,
        files,
        steps,
        answer,
        frozenset({"read_file", "calculate"}),
    )


def _update(split: str, index: int, level: int, rng: random.Random) -> Task:
    old_mode, new_mode = rng.sample(("safe", "fast", "audit", "strict"), 2)
    path = f"workspace/{split}/{index:04d}/config.ini"
    before = f"service=worker-{index}\n{_noise(rng, 1 + level, 'c')}mode={old_mode}\nretries=3\n"
    after = before.replace(f"mode={old_mode}", f"mode={new_mode}")
    answer = f"mode={new_mode}"
    prompt = rng.choice(
        [
            f"In {path}, change mode from {old_mode} to {new_mode}. Inspect the file first, re-read it to verify, and report it as mode=<target>.",
            f"Update {path} after inspecting it: set its mode to {new_mode}, then verify the saved setting and report it as mode=<target>.",
        ]
    )
    steps = (
        _step(
            f"Plan: inspect {_short(path)}, replace mode={old_mode} with mode={new_mode}, re-read to verify, report. Reading.",
            "read_file",
            path=path,
        ),
        _step(
            f"File contains mode={old_mode} as expected. Applying the exact replacement.",
            "replace_text",
            path=path,
            old=f"mode={old_mode}",
            new=f"mode={new_mode}",
        ),
        _step(
            "Replacement confirmed by the tool. Re-reading to verify the new setting.",
            "read_file",
            path=path,
        ),
        _step(f"Verified {answer}; pending: none.", "finish", answer=answer),
    )
    return Task(
        f"{split}-update-{index:04d}",
        "update",
        "clean",
        prompt,
        {path: before},
        steps,
        answer,
        frozenset({"read_file", "replace_text"}),
        {path: after},
    )


def _list(split: str, index: int, level: int, rng: random.Random) -> Task:
    root = f"workspace/{split}/{index:04d}"
    label = f"milestone-{rng.randrange(100, 999)}"
    directory = f"{root}/notes"
    target = f"{directory}/summary-{rng.randrange(10, 99)}.md"
    files = {
        target: f"{label}\nThis is the selected summary.",
        f"{directory}/scratch.txt": "ignore this scratch file",
        f"{root}/outside.md": "outside the requested directory",
    }
    for extra in range(level):
        files[f"{directory}/log-{extra}.txt"] = "log entry, not a summary"
    prompt = rng.choice(
        [
            f"List files in {directory}, read the Markdown summary there, and report its first line exactly.",
            f"Look inside {directory}, open the .md summary file, and reply with exactly its first line.",
        ]
    )
    steps = (
        _step(
            f"Plan: list {_short(directory)}, read the .md summary, report its first line. Listing.",
            "list_files",
            directory=directory,
        ),
        _step(f"Listing shows one Markdown file: {target}. Reading it.", "read_file", path=target),
        _step(f"First line: {label}; pending: none.", "finish", answer=label),
    )
    return Task(
        f"{split}-list-{index:04d}",
        "list",
        "clean",
        prompt,
        files,
        steps,
        label,
        frozenset({"list_files", "read_file"}),
    )


# --------------------------------------------------------------------------- long-horizon families


def _pointer_chain(split: str, index: int, level: int, rng: random.Random) -> Task:
    root = f"lab/{split}/{index:04d}/chain"
    hops = rng.randrange(4 + level, 7 + level)
    paths = [f"{root}/node-{step}-{rng.randrange(100, 999)}.txt" for step in range(hops)]
    result = f"artifact-{rng.randrange(10000, 99999)}"
    files = {}
    for step, path in enumerate(paths):
        context = _noise(rng, 8 + 3 * level, f"node-{step}")
        if step + 1 < len(paths):
            files[path] = f"Node: {step}\n{context}Next: {paths[step + 1]}\n"
        else:
            files[path] = f"Node: final\n{context}Result: {result}\n"
    files[f"{root}/decoy.txt"] = "Result: ignore-this-decoy"
    prompt = rng.choice(
        [
            f"Start by reading {paths[0]}. Follow each exact Next path until a Result field is reached, then report that Result exactly. Do not use the decoy.",
            f"Begin at {paths[0]} and keep following the Next path in each node. When a node contains a Result field, report that Result exactly and ignore decoy files.",
        ]
    )
    steps = [
        _step(
            "Plan: follow Next pointers from the start node until a node shows a Result field; ignore the decoy. Reading the start node.",
            "read_file",
            path=paths[0],
        )
    ]
    for hop in range(1, hops):
        steps.append(
            _step(
                f"Visited {hop} node(s), no Result yet. Node {hop - 1} points to {paths[hop]}; following it.",
                "read_file",
                path=paths[hop],
            )
        )
    steps.append(
        _step(f"Result: {result}; pending: none.", "finish", answer=result)
    )
    return Task(
        f"{split}-pointer_chain-{index:04d}",
        "pointer_chain",
        "clean",
        prompt,
        files,
        tuple(steps),
        result,
        frozenset({"read_file"}),
    )


def _ledger_reconcile(split: str, index: int, level: int, rng: random.Random) -> Task:
    root = f"lab/{split}/{index:04d}/ledger"
    entries = 4 + level
    files, entry_paths, records = {}, [], []
    for item in range(entries):
        amount = rng.randrange(20, 180)
        status = "approved" if item % 3 != 1 else "held"
        path = f"{root}/invoice-{item}-{rng.randrange(100, 999)}.txt"
        files[path] = (
            f"invoice={item}\n{_noise(rng, 12 + 4 * level, f'invoice-{item}')}amount={amount}\nstatus={status}\n"
        )
        entry_paths.append(path)
        records.append((amount, status))
    summary_path = f"{root}/summary.txt"
    old_summary, approved = "approved_total=PENDING", [a for a, s in records if s == "approved"]
    total = str(sum(approved))
    new_summary = f"approved_total={total}"
    files[summary_path] = old_summary
    expression = " + ".join(map(str, approved))
    prompt = rng.choice(
        [
            f"Reconcile {root}. List the files, read every invoice, sum only amounts whose status is approved using the calculator, replace PENDING in {summary_path} with the numeric total, read the summary again to verify it, and report the complete approved_total setting.",
            f"In {root}: list the directory, open each invoice file, add up the amount of every invoice with status=approved (use the calculator), write that total over PENDING in {summary_path}, re-read the summary to confirm, and report the full approved_total line.",
        ]
    )
    steps = [
        _step(
            "Plan: list the ledger, read every invoice, sum approved amounts only, inspect the summary, replace PENDING, re-read to verify, report approved_total. Listing.",
            "list_files",
            directory=root,
        )
    ]
    approved_so_far: list[int] = []
    held_so_far: list[int] = []
    ordered = sorted(zip(entry_paths, records, strict=True))
    for position, (path, (amount, status)) in enumerate(ordered):
        remaining = [later for later, _ in ordered[position + 1 :]]
        steps.append(
            _step(
                f"Invoices read: {position} of {entries}. approved: {_join(approved_so_far)}; held (skip): {_join(held_so_far)}. Reading {_short(path)}; "
                + (
                    _pending(remaining) + "."
                    if remaining
                    else "then sum the approved amounts and update the summary."
                ),
                "read_file",
                path=path,
            )
        )
        (approved_so_far if status == "approved" else held_so_far).append(amount)
    steps.extend(
        [
            _step(
                f"Invoices read: {entries} of {entries}. approved: {_join(approved_so_far)}; held skipped: {_join(held_so_far)}. Summing approved amounts.",
                "calculate",
                expression=expression,
            ),
            _step(
                f"approved: {_join(approved_so_far)}; summary {_short(summary_path)}; total {total}. Inspecting before editing.",
                "read_file",
                path=summary_path,
            ),
            _step(
                f"Summary {_short(summary_path)} contains {old_summary}; approved: {_join(approved_so_far)}. Replacing PENDING with {total}.",
                "replace_text",
                path=summary_path,
                old=old_summary,
                new=new_summary,
            ),
            _step(
                f"Replacement confirmed for {_short(summary_path)}; approved: {_join(approved_so_far)}. Re-reading to verify.",
                "read_file",
                path=summary_path,
            ),
            _step(
                f"Verified {_short(summary_path)} reads {new_summary}; pending: none.",
                "finish",
                answer=new_summary,
            ),
        ]
    )
    return Task(
        f"{split}-ledger_reconcile-{index:04d}",
        "ledger_reconcile",
        "clean",
        prompt,
        files,
        tuple(steps),
        new_summary,
        frozenset({"list_files", "read_file", "calculate", "replace_text"}),
        {summary_path: new_summary},
    )


def _cross_reference(split: str, index: int, level: int, rng: random.Random) -> Task:
    root = f"lab/{split}/{index:04d}/records"
    links = 3 + level
    keys = [
        f"REF-{split.upper()}-{index:04d}-{rng.randrange(1000, 9999)}-{n}" for n in range(links)
    ]
    answer = f"resolution-{rng.randrange(10000, 99999)}"
    files, target_paths = {}, []
    for hop, key in enumerate(keys):
        path = f"{root}/record-{rng.randrange(100, 999)}-{hop}.txt"
        target_paths.append(path)
        tail = f"Next-Key: {keys[hop + 1]}\n" if hop + 1 < links else f"Resolution: {answer}\n"
        files[path] = f"Lookup-Key: {key}\n{_noise(rng, 12 + 4 * level, f'record-{hop}')}{tail}"
    for decoy in range(3 + level):
        files[f"{root}/decoy-{decoy}.txt"] = f"Lookup-Key: DECOY-{rng.randrange(99999)}"
    prompt = rng.choice(
        [
            f"Search for {keys[0]}. Read its matching record, then repeatedly search each Next-Key and read its record until you reach Resolution. Report the Resolution exactly.",
            f"Starting from key {keys[0]}, search the workspace for each key, open the matching record, and follow every Next-Key the same way until a record contains Resolution. Reply with exactly that Resolution.",
        ]
    )
    steps = []
    for hop, (key, path) in enumerate(zip(keys, target_paths, strict=True)):
        if hop == 0:
            thought = f"Hop 1: current key {key}. Searching for its record."
        else:
            thought = f"Hop {hop + 1}: current key {key}; no Resolution yet. Searching for its record."
        steps.append(_step(thought, "search_files", query=key))
        # The previous record also contains this key (as its Next-Key), so the search returns
        # two files; the note must say which one is already read and which one to open.
        matches = _search_matches(files, key)
        steps.append(
            _step(
                f"Hop {hop + 1}: current key {key}. {_match_note(matches, path, target_paths[:hop])}",
                "read_file",
                path=path,
            )
        )
    steps.append(
        _step(f"Resolution: {answer}; pending: none.", "finish", answer=answer)
    )
    return Task(
        f"{split}-cross_reference-{index:04d}",
        "cross_reference",
        "clean",
        prompt,
        files,
        tuple(steps),
        answer,
        frozenset({"search_files", "read_file"}),
    )


def _conditional_update(split: str, index: int, level: int, rng: random.Random) -> Task:
    root = f"lab/{split}/{index:04d}/services"
    count = 3 + level
    threshold = rng.randrange(55, 75)
    loads = rng.sample(range(30, 99), count)
    target_index = max(range(count), key=loads.__getitem__)
    if loads[target_index] <= threshold:
        loads[target_index] = threshold + rng.randrange(5, 20)
    policy_path = f"{root}/policy.txt"
    files = {policy_path: f"threshold={threshold}\naction=change highest load to throttled\n"}
    service_paths = []
    for service, load in enumerate(loads):
        path = f"{root}/service-{service}.ini"
        files[path] = (
            f"name=service-{service}\n{_noise(rng, 12 + 4 * level, f'service-{service}')}load={load}\nmode=active\n"
        )
        service_paths.append(path)
    target_path = service_paths[target_index]
    expected = files[target_path].replace("mode=active", "mode=throttled")
    answer = f"service-{target_index}:mode=throttled"
    prompt = rng.choice(
        [
            f"Read {policy_path}, list and inspect every service file in {root}, identify the service with the highest load, and if it exceeds the policy threshold change its mode from active to throttled. Read the changed file to verify it and report service-N:mode=throttled.",
            f"Using the threshold in {policy_path}, examine every service-*.ini in {root} and throttle the single service with the highest load if that load exceeds the threshold (mode=active becomes mode=throttled). Verify by re-reading it and report service-N:mode=throttled.",
        ]
    )
    steps = [
        _step(
            "loads so far: none. Plan: read the policy threshold, list services, read every service load, throttle the highest load if above threshold, re-read to verify, report. Reading policy.",
            "read_file",
            path=policy_path,
        ),
        _step(
            f"loads so far: none. threshold={threshold}. Listing the service files.",
            "list_files",
            directory=root,
        ),
    ]
    best_index: int | None = None
    best_load: int | None = None
    observed_loads: list[str] = []
    for position, path in enumerate(service_paths):
        best_label = f"service-{best_index}={best_load}" if best_index is not None else "none"
        remaining = service_paths[position + 1 :]
        steps.append(
            _step(
                f"threshold={threshold}. loads so far: {_join(observed_loads)}. highest so far: {best_label}. Reading service {position + 1} of {count}: {_short(path)}; "
                + (_pending(remaining) + "." if remaining else "then select the highest load."),
                "read_file",
                path=path,
            )
        )
        observed_loads.append(f"service-{position}={loads[position]}")
        if best_load is None or loads[position] > best_load:
            best_index, best_load = position, loads[position]
    assert (best_index, best_load) == (target_index, loads[target_index]), (
        "running best diverged from the precomputed target"
    )
    steps.extend(
        [
            _step(
                f"threshold={threshold}. loads so far: {_join(observed_loads)}. highest so far: service-{best_index}={best_load}; "
                f"above threshold, so throttle {_short(target_path)}.",
                "replace_text",
                path=target_path,
                old="mode=active",
                new="mode=throttled",
            ),
            _step(
                f"loads so far: {_join(observed_loads)}. Replacement confirmed for {_short(target_path)}. Re-reading to verify.",
                "read_file",
                path=target_path,
            ),
            _step(
                f"loads so far: {_join(observed_loads)}. Verified service-{target_index} now has mode=throttled; pending: none.",
                "finish",
                answer=answer,
            ),
        ]
    )
    return Task(
        f"{split}-conditional_update-{index:04d}",
        "conditional_update",
        "clean",
        prompt,
        files,
        tuple(steps),
        answer,
        frozenset({"read_file", "list_files", "replace_text"}),
        {target_path: expected},
    )


def _batch_update(split: str, index: int, level: int, rng: random.Random) -> Task:
    root = f"lab/{split}/{index:04d}/deployment"
    target_count = 2 + level
    manifest_path = f"{root}/manifest.txt"
    files, targets, expected_files, manifest_lines = {}, [], {}, []
    modes = ("safe", "audit", "strict", "fast", "observe")
    for target in range(target_count):
        old, new = rng.sample(modes, 2)
        path = f"{root}/worker-{target}.ini"
        before = f"worker={target}\n{_noise(rng, 12 + 4 * level, f'worker-{target}')}mode={old}\nversion={rng.randrange(1, 10)}\n"
        files[path] = before
        expected_files[path] = before.replace(f"mode={old}", f"mode={new}")
        targets.append((path, old, new))
        manifest_lines.append(f"{path}|mode={old}->mode={new}")
    files[manifest_path] = "\n".join(manifest_lines) + "\n"
    files[f"{root}/unmanaged.ini"] = "worker=unmanaged\nmode=leave-alone\n"
    answer = f"updated-and-verified={target_count}"
    prompt = rng.choice(
        [
            f"Read {manifest_path}. For every managed worker listed there, inspect its file, apply the exact mode replacement, then read every changed file again. Do not alter unmanaged.ini. Once all {target_count} updates are verified, report as updated-and-verified=<count>.",
            f"Apply the deployment described in {manifest_path}: inspect each listed worker file, perform its exact mode replacement, and re-read each worker afterwards to verify. Leave unmanaged.ini untouched and report all {target_count} verified updates as updated-and-verified=<count>.",
        ]
    )
    steps = [
        _step(
            "Plan: read the manifest, inspect every listed worker, apply each exact replacement, re-read every changed worker, report the count. Reading manifest.",
            "read_file",
            path=manifest_path,
        )
    ]
    for n, (path, _, _) in enumerate(targets):
        steps.append(
            _step(
                f"Inspected {n} of {target_count}. Next: {_queue_line(targets[n])}. "
                f"queue: {_queue_tail(targets[n:])}; {_pending([item[0] for item in targets[n:]])}.",
                "read_file",
                path=path,
            )
        )
    for n, (path, old, new) in enumerate(targets):
        steps.append(
            _step(
                f"Applied {n} of {target_count}. Next: {_queue_line(targets[n])}. "
                f"queue: {_queue_tail(targets[n:])}; {_pending([item[0] for item in targets[n:]])}.",
                "replace_text",
                path=path,
                old=f"mode={old}",
                new=f"mode={new}",
            )
        )
    for n, (path, _, _) in enumerate(targets):
        steps.append(
            _step(
                f"Applied {target_count} of {target_count}; verified {n} of {target_count}. Next: {_verify_line(targets[n])}. "
                f"queue: {_queue_tail(targets[n:])}; {_pending([item[0] for item in targets[n:]])}.",
                "read_file",
                path=path,
            )
        )
    steps.append(
        _step(
            f"Applied {target_count} of {target_count}; verified {target_count} of {target_count}; pending: none.",
            "finish",
            answer=answer,
        )
    )
    return Task(
        f"{split}-batch_update-{index:04d}",
        "batch_update",
        "clean",
        prompt,
        files,
        tuple(steps),
        answer,
        frozenset({"read_file", "replace_text"}),
        expected_files,
    )


def _aggregate_report(split: str, index: int, level: int, rng: random.Random) -> Task:
    root = f"lab/{split}/{index:04d}/metrics"
    metric_count = 4 + level
    values = [rng.randrange(10, 90) for _ in range(metric_count)]
    files, metric_paths = {}, []
    for metric, value in enumerate(values):
        path = f"{root}/metric-{metric}.txt"
        files[path] = (
            f"metric={metric}\n{_noise(rng, 12 + 4 * level, f'metric-{metric}')}value={value}\n"
        )
        metric_paths.append(path)
    split_at = metric_count // 2
    first_expression = " + ".join(map(str, values[:split_at]))
    second_expression = " + ".join(map(str, values[split_at:]))
    first_total, second_total = _calculate(first_expression), _calculate(second_expression)
    grand_expression = f"{first_total} + {second_total}"
    grand_total = _calculate(grand_expression)
    report_path = f"{root}/report.txt"
    old_report, new_report = "grand_total=PENDING", f"grand_total={_calculate(grand_expression)}"
    files[report_path] = old_report
    prompt = rng.choice(
        [
            f"List {root} and read all {metric_count} metric files. Use the calculator to subtotal the first half, subtotal the second half, and add the two subtotals. Inspect report.txt, replace PENDING with the grand total, read it again to verify, and report the complete setting.",
            f"In {root} there are {metric_count} metric-*.txt files and a report.txt. List the directory, read every metric value, compute the first-half subtotal, the second-half subtotal, and their sum with the calculator, then replace PENDING in report.txt with the grand total, re-read it, and report the full grand_total line.",
        ]
    )
    steps = [
        _step(
            "Plan: list the metrics, read every value, subtotal each half, add the subtotals, inspect the report, replace PENDING, re-read to verify, report grand_total. Listing.",
            "list_files",
            directory=root,
        )
    ]
    values_so_far: list[int] = []
    for position, path in enumerate(metric_paths):
        remaining = metric_paths[position + 1 :]
        steps.append(
            _step(
                f"values so far: {_join(values_so_far)}; split after {split_at} of {metric_count}. "
                f"Reading metric {position + 1} of {metric_count}: {_short(path)}; "
                + (_pending(remaining) + "." if remaining else "then calculate the two subtotals."),
                "read_file",
                path=path,
            )
        )
        values_so_far.append(values[position])
    steps.extend(
        [
            _step(
                f"values so far: {_join(values_so_far)}; split after {split_at} of {metric_count}. Computing the first subtotal {first_expression}.",
                "calculate",
                expression=first_expression,
            ),
            _step(
                f"values so far: {_join(values_so_far)}; split after {split_at} of {metric_count}. First subtotal = {first_total}; computing {second_expression}.",
                "calculate",
                expression=second_expression,
            ),
            _step(
                f"values so far: {_join(values_so_far)}; split after {split_at} of {metric_count}. Subtotals {first_total} and {second_total}; adding them.",
                "calculate",
                expression=grand_expression,
            ),
            _step(
                f"values so far: {_join(values_so_far)}; split after {split_at} of {metric_count}. Grand total = {grand_total}; inspecting {_short(report_path)}.",
                "read_file",
                path=report_path,
            ),
            _step(
                f"values so far: {_join(values_so_far)}; split after {split_at} of {metric_count}. {_short(report_path)} contains {old_report}; replacing PENDING with {grand_total}.",
                "replace_text",
                path=report_path,
                old=old_report,
                new=new_report,
            ),
            _step(
                f"values so far: {_join(values_so_far)}; split after {split_at} of {metric_count}. Replacement confirmed for {_short(report_path)}; re-reading to verify.",
                "read_file",
                path=report_path,
            ),
            _step(
                f"values so far: {_join(values_so_far)}; split after {split_at} of {metric_count}. Verified {_short(report_path)} reads {new_report}; pending: none.",
                "finish",
                answer=new_report,
            ),
        ]
    )
    return Task(
        f"{split}-aggregate_report-{index:04d}",
        "aggregate_report",
        "clean",
        prompt,
        files,
        tuple(steps),
        new_report,
        frozenset({"list_files", "read_file", "calculate", "replace_text"}),
        {report_path: new_report},
    )


_MAKERS = {
    "read": _read,
    "search": _search,
    "calculate": _calc,
    "synthesis": _synthesis,
    "update": _update,
    "list": _list,
    "pointer_chain": _pointer_chain,
    "ledger_reconcile": _ledger_reconcile,
    "cross_reference": _cross_reference,
    "conditional_update": _conditional_update,
    "batch_update": _batch_update,
    "aggregate_report": _aggregate_report,
}


# --------------------------------------------------------------------------- recovery variants


def _apply_variant(task: Task, variant: str, rng: random.Random) -> Task:
    if variant == "clean":
        return task
    if variant == "transient":
        return _transient(task, rng)
    if variant == "unknown_tool":
        if any(step.action.name == "replace_text" for step in task.steps):
            return _unknown_tool(task, rng)
        return _wrong_path(task, rng)
    if variant == "wrong_path":
        return _wrong_path(task, rng)
    if variant == "stale_path":
        return _stale_path(task, rng)
    if variant == "failed_edit":
        return _failed_edit(task, rng)
    raise ValueError(f"unsupported variant: {variant}")


def _injected_state_prefix(task: Task, fallback: str) -> str:
    """Keep a recovery guess inside the family state representation it interrupts."""
    if task.family == "aggregate_report":
        total = sum("value=" in content for content in task.files.values())
        return f"values so far: none; split after {total // 2} of {total}. {fallback}"
    if task.family == "conditional_update":
        return f"loads so far: none. {fallback}"
    return fallback


def _transient(task: Task, rng: random.Random) -> Task:
    candidates = [i for i, step in enumerate(task.steps) if step.action.name != "finish"]
    k = rng.choice(candidates)
    step = task.steps[k]
    retry = Step(
        "The tool reported a transient failure; the call itself was correct, so retry it unchanged. "
        + step.thought,
        step.action,
    )
    steps = (*task.steps[: k + 1], retry, *task.steps[k + 1 :])
    return replace(task, variant="transient", steps=steps, faults=(Fault(call_index=k),))


def _wrong_path(task: Task, rng: random.Random) -> Task:
    reads = [i for i, step in enumerate(task.steps) if step.action.name == "read_file" and i >= 1]
    if task.steps[0].action.name == "list_files" and (not reads or rng.random() < 0.5):
        directory = task.steps[0].action.arguments["directory"]
        candidates = [
            f"{directory}/{name}"
            for name in ("index.txt", "summary.md", "data.txt", "main.ini")
            if f"{directory}/{name}" not in task.files
        ]
        if not candidates:
            raise RuntimeError(f"{task.task_id}: could not construct a wrong_path variant")
        guess = rng.choice(candidates)
        wrong = Step(
            f"Trying guessed path {guess} before listing the directory. "
            f"{_injected_state_prefix(task, task.steps[0].thought)}",
            Action("read_file", {"path": guess}),
            supervise=False,
        )
        recovery = Step(
            "That guessed path does not exist. List the directory first, then use only the exact paths it reports. "
            + task.steps[0].thought,
            task.steps[0].action,
        )
        return replace(task, variant="wrong_path", steps=(wrong, recovery, *task.steps[1:]))
    if not reads:
        raise RuntimeError(f"{task.task_id}: could not construct a wrong_path variant")
    k = rng.choice(reads)
    step = task.steps[k]
    correct = step.action.arguments["path"]
    stem, _, ext = correct.rpartition(".")
    candidates = (
        [f"{stem}{suffix}.{ext}" for suffix in ("-old", "-copy", "1")]
        if stem
        else [correct + ".bak"]
    )
    candidates = [candidate for candidate in candidates if candidate not in task.files]
    if not candidates:
        raise RuntimeError(f"{task.task_id}: could not construct a wrong_path variant")
    guess = rng.choice(candidates)
    wrong = Step(
        f"Trying guessed path {guess}. {step.thought}",
        Action("read_file", {"path": guess}),
        supervise=False,
    )
    recovery = Step(
        f"That path does not exist; use the exact path from the earlier tool result instead of guessing: {correct}. "
        + step.thought,
        step.action,
    )
    return replace(
        task, variant="wrong_path", steps=(*task.steps[:k], wrong, recovery, *task.steps[k + 1 :])
    )


def _unknown_tool(task: Task, rng: random.Random) -> Task:
    edits = [i for i, step in enumerate(task.steps) if step.action.name == "replace_text"]
    k = rng.choice(edits)
    step = task.steps[k]
    bogus = rng.choice(("update_file", "write_file", "apply_replacement", "edit_file"))
    wrong = Step(step.thought, Action(bogus, dict(step.action.arguments)), supervise=False)
    recovery = Step(
        f"{bogus} is not an available tool; the workspace exposes replace_text. Apply the same exact replacement with it. "
        + step.thought,
        step.action,
    )
    return replace(
        task, variant="unknown_tool", steps=(*task.steps[:k], wrong, recovery, *task.steps[k + 1 :])
    )


_NUMERIC_SUFFIX = re.compile(r"(\d+)(\.[^./]+)$")


def _stale_guess(correct: str, rng: random.Random, files: dict[str, str]) -> str | None:
    """A path that keeps ``correct``'s directory and stem but changes its numeric suffix,
    guaranteed not to exist in ``files`` (e.g. ``invoice-2-537.txt`` -> ``invoice-2-878.txt``)."""
    match = _NUMERIC_SUFFIX.search(correct)
    if not match:
        return None
    prefix = correct[: match.start(1)]
    ext = match.group(2)
    width = len(match.group(1))
    low = 10 ** (width - 1) if width > 1 else 0
    high = 10**width - 1
    for _ in range(100):
        candidate = f"{prefix}{rng.randrange(low, high + 1)}{ext}"
        if candidate != correct and candidate not in files:
            return candidate
    return None


def _stale_path(task: Task, rng: random.Random) -> Task:
    """A ``read_file`` guesses a stale numeric variant of a path learned from an earlier
    ``list_files`` call that has since scrolled out of the context window; the only correct
    recovery is to list the directory again rather than guess another path."""
    list_indices = [i for i, step in enumerate(task.steps) if step.action.name == "list_files"]
    candidates: list[tuple[int, str, str]] = []
    for i in list_indices:
        for k, step in enumerate(task.steps):
            if step.action.name != "read_file" or k < i + 3:
                continue
            correct = step.action.arguments["path"]
            guess = _stale_guess(correct, rng, task.files)
            if guess is None:
                continue
            directory = correct.rsplit("/", 1)[0]
            candidates.append((k, guess, directory))
    if not candidates:
        raise RuntimeError(f"{task.task_id}: could not construct a stale_path variant")
    k, guess, directory = rng.choice(candidates)
    step = task.steps[k]
    wrong = Step(
        f"Trying stale guessed path {guess}. {step.thought}",
        Action("read_file", {"path": guess}),
        supervise=False,
    )
    listing = Step(
        "That path does not exist and the earlier listing is no longer visible, so list the "
        f"directory again for the exact names instead of guessing. {step.thought}",
        Action("list_files", {"directory": directory}),
    )
    recovered = Step("The listing gives the exact name. " + step.thought, step.action)
    return replace(
        task,
        variant="stale_path",
        steps=(*task.steps[:k], wrong, listing, recovered, *task.steps[k + 1 :]),
    )


_KV_LINE = re.compile(r"^[A-Za-z][\w.-]*=.+$", re.MULTILINE)


def _kv_candidates(content: str) -> list[str]:
    return [line.strip() for line in _KV_LINE.findall(content)]


def _perturb_value(text: str) -> str:
    """A ``key=value``-shaped string with its value scrambled, for tasks with no other file or
    edit to draw a wrong ``old`` guess from (never reuses ``new``, so old != new is preserved)."""
    key, sep, value = text.partition("=")
    if not sep:
        return text + "-guess"
    scrambled = value[::-1] or "x"
    return f"{key}={scrambled}" if scrambled != value else f"{key}={value}-guess"


def _wrong_old(
    task: Task,
    k: int,
    path: str,
    correct_old: str,
    new: str,
    current_content: str,
    rng: random.Random,
) -> str | None:
    """A value used elsewhere in the task (another edit's old/new text, or a key=value line
    from a different file) that is genuinely absent from ``current_content``. ``new`` is never
    offered as a candidate: reusing it would make the wrong call replace ``new`` with itself."""
    pool: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        if candidate and candidate not in (correct_old, new) and candidate not in seen:
            seen.add(candidate)
            pool.append(candidate)

    for i, other in enumerate(task.steps):
        if i == k or other.action.name != "replace_text":
            continue
        add(other.action.arguments["old"])
        add(other.action.arguments["new"])
    for other_path, content in task.files.items():
        if other_path == path:
            continue
        for candidate in _kv_candidates(content):
            add(candidate)
    valid = [candidate for candidate in pool if candidate not in current_content]
    if valid:
        return rng.choice(valid)
    perturbed = _perturb_value(correct_old)
    if perturbed not in (correct_old, new) and perturbed not in current_content:
        return perturbed
    return None


def _build_failed_edit(task: Task, k: int, rng: random.Random) -> Task | None:
    step = task.steps[k]
    path = step.action.arguments["path"]
    correct_old = step.action.arguments["old"]
    new = step.action.arguments["new"]
    simulator = Simulator.for_task(task, faults=())
    for prior in task.steps[:k]:
        simulator.execute(prior.action)
    current_content = simulator.files.get(path, "")
    wrong_old = _wrong_old(task, k, path, correct_old, new, current_content, rng)
    if wrong_old is None:
        return None
    wrong = Step(
        step.thought,
        Action("replace_text", {"path": path, "old": wrong_old, "new": new}),
        supervise=False,
    )
    reread = Step(
        "The replacement failed because that exact text is not in the file, so re-read it to "
        "see its current contents instead of retrying the same edit. "
        + step.thought,
        Action("read_file", {"path": path}),
    )
    recovered = Step(
        "The file's current text confirms the exact string to replace. " + step.thought, step.action
    )
    return replace(
        task,
        variant="failed_edit",
        steps=(*task.steps[:k], wrong, reread, recovered, *task.steps[k + 1 :]),
    )


def _failed_edit(task: Task, rng: random.Random) -> Task:
    """A ``replace_text`` guesses an ``old`` value that is not in the target file; the only
    correct recovery is to re-read the file rather than retry the same failing edit."""
    edits = [i for i, step in enumerate(task.steps) if step.action.name == "replace_text"]
    if not edits:
        return _transient(task, rng)
    order = list(edits)
    rng.shuffle(order)
    for k in order:
        built = _build_failed_edit(task, k, rng)
        if built is not None:
            return built
    raise RuntimeError(f"{task.task_id}: could not construct a wrong 'old' for failed_edit")
