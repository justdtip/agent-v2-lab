"""Coherence to completion: the unified run's capability standard (UNIFIED-RUN-2026-09-06,
section 2 revised at 10:05 and the rulings of 10:12).

One task is one evaluation point. A trajectory's *incoherence events* are the first step at
which each of four causes fires, kept apart as competing risks and never pooled into a length:

* ``integrity``: the note-integrity checker's first violation (``integrity.first_violation``,
  which carries its step).
* ``loop``: the first step at which ``detect_loop`` closes a repetition window, recomputed here
  over the stored steps because the runner records only the flag; the function is pure in the
  steps, so the recomputation is exact.
* ``invalid_action``: a parse error (the runner appends a step carrying ``parse_error``) or a
  call the validator refused for a reason other than an unknown tool (``ERROR: invalid call:``,
  a missing or unexpected argument or a wrong type).
* ``tool_error``: an ``ERROR`` observation from the simulator (a transient fault, a wrong or
  stale path, a failed edit) or an unknown tool, that is **not recovered within the window**:
  recovery is a later successful call, within the next ``window`` *executed* steps, **with the
  same tool, whatever its arguments** (the 10:12 ruling as corrected by the Research Division's
  review at 13:30: the ruling's "same primary argument" clause recognised only the retry of an
  identical call, which is recovery from a transient and nothing else; every other error
  recovers by correction, and a correction changes the argument because the argument was what
  was wrong. Against the base evaluation the primary-argument rule recovered 0 of 273 errors,
  the same pathology as the field it replaced; the same-tool rule recovers 46, and the
  generator's own ``wrong_path`` and ``stale_path`` experts recover by re-calling the failed
  tool with the corrected argument, which is the test below). An unknown tool has no later call
  with the same tool by construction, so it is recovered by any later successful call within
  the window (the ``unknown_tool`` clause; the data's ``unknown_tool`` variant teaches exactly
  that correction). ``primary_arguments`` is kept in the summary for the record, not for the
  rule. The field ``recovered_errors`` in the verdict is **not** used: it is the error count on
  success and zero otherwise (``env.py``), which on the base evaluation read 0 against 273
  errors. An error inside the last ``window`` executed steps of a trajectory is censored for
  this cause whatever the outcome, since recovery cannot be observed there; the censoring is
  symmetric so that a late error scores the same in a success and in a failure.

The headline is the fraction of trajectories **coherent to completion**: success with no event,
with Wilson intervals, overall and per family, variant and difficulty. Beside it: the
cause-specific hazard per step (events of a cause at step s over trajectories still at risk at s)
and the cumulative incidence by cause, which is what an adapter that halves loops while
worsening tool errors shows as. No median coherence length is a statistic; the Kaplan-Meier
median is reported as a descriptive with ``"not reached"`` when fewer than half fail, and the
informative-censoring caveat (completion is correlated with capability) travels with it.
Comparisons across families or difficulties are made only on scale-free quantities, the per-step
hazard or the first event's step as a fraction of the task's own horizon; raw step counts are
not comparable across cells and the summary says so.
"""
from __future__ import annotations

import math
from typing import Any

from local_llm_lab.agent_protocol import TOOL_SPECS

CAUSES = ("integrity", "loop", "invalid_action", "tool_error")
DEFAULT_WINDOW = 3
INVALID_CALL = "ERROR: invalid call:"
UNKNOWN_TOOL = "ERROR: invalid call: unknown tool"


def primary_arguments(tools: list[dict[str, Any]] = TOOL_SPECS) -> dict[str, str | None]:
    """The primary argument of each tool: its first required schema parameter, else its first
    property, else None for a tool that takes no arguments (``finish``)."""
    out: dict[str, str | None] = {}
    for spec in tools:
        function = spec["function"]
        parameters = function.get("parameters", {}) or {}
        properties = list((parameters.get("properties") or {}).keys())
        required = [name for name in parameters.get("required", []) if name in properties]
        out[function["name"]] = required[0] if required else (properties[0] if properties else None)
    return out


def _executed(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [step for step in steps if "action" in step]


def _observation_kind(step: dict[str, Any]) -> str | None:
    """None for a clean observation; otherwise ``unknown_tool``, ``invalid_action`` or
    ``tool_error``."""
    observation = str(step.get("observation", ""))
    if not observation.startswith("ERROR"):
        return None
    if observation.startswith(UNKNOWN_TOOL):
        return "unknown_tool"
    if observation.startswith(INVALID_CALL):
        return "invalid_action"
    return "tool_error"


def _recovered(failed: dict[str, Any], later: list[dict[str, Any]], *, kind: str, window: int) -> bool:
    """Is the failed call recovered by a later successful call within the next ``window`` executed
    steps? Same tool, any arguments; for an unknown tool, any successful call."""
    candidates = later[:window]
    if kind == "unknown_tool":
        return any(_observation_kind(step) is None for step in candidates)
    name = failed["action"]["name"]
    return any(_observation_kind(step) is None and step["action"]["name"] == name for step in candidates)


def _step_index(step: dict[str, Any], position: int, horizon: int) -> int:
    """The step's index on the one scale every cause is reported on.

    This is ``integrity._policy_index``'s rule, deliberately: the integrity cause is the
    checker's ``Violation.step``, computed that way (``integrity.py:231-235``), and
    ``first_cause`` is a minimum across the four causes, so a cause on a different scale would
    make that minimum meaningless rather than merely wrong. Reading ``step["index"]`` alone made
    the module require a key only the runner writes, which is what the seven ``test_patch``
    failures were: hand-built steps carry none.

    It matches the checker *exactly*, including admitting ``True`` as the integer 1, because a
    test pins the two functions against each other and a well-meant improvement here is a
    divergence there. That test caught exactly that.
    """
    saved = step.get("index")
    if isinstance(saved, int) and 0 <= saved < horizon:
        return saved
    return position


def loop_step(steps: list[dict[str, Any]], horizon: int | None = None) -> int | None:
    """The index of the step at which ``detect_loop`` first closes, or None.

    ``horizon`` is the scale every cause is reported on (the task's step count when the caller
    knows it); without it the trajectory's own length stands in. ``trajectory_events`` always
    passes its resolved horizon, so the loop cause sits on the same scale as the other three:
    a saved index valid at the task horizon but at or beyond the trajectory's length would
    otherwise be replaced by its position here and nowhere else, and ``first_cause``, being a
    minimum, would flip on that artefact (Deputy, 7 September, reproduced).
    """
    from local_llm_lab.pipeline.runner import detect_loop  # local import: runner imports env

    scale = max(len(steps), 1) if horizon is None else max(int(horizon), len(steps), 1)
    for count in range(1, len(steps) + 1):
        if detect_loop(steps[:count]):
            return _step_index(steps[count - 1], count - 1, scale)
    return None


def trajectory_events(
    trajectory: Any,
    *,
    window: int = DEFAULT_WINDOW,
    tools: list[dict[str, Any]] = TOOL_SPECS,
) -> dict[str, Any]:
    """Per-trajectory coherence record: the first step of each cause (None if it never fires),
    the first event overall, whether the trajectory was coherent to completion, and the
    tool-error censoring."""
    steps = list(getattr(trajectory, "steps", []) or [])
    events: dict[str, int | None] = {cause: None for cause in CAUSES}
    # The checker's horizon is the task's step count, not the number of steps actually taken
    # (``integrity.py:364``), so a trajectory that stopped early still admits its saved indices.
    horizon = int(getattr(trajectory, "horizon", -1) or -1)
    if horizon < len(steps):
        horizon = len(steps)
    index_of = {
        id(step): _step_index(step, position, horizon)
        for position, step in enumerate(steps)
    }
    last_index = max(index_of.values(), default=0)
    # integrity: the checker's first violation carries its step
    integrity = getattr(trajectory, "integrity", {}) or {}
    first = integrity.get("first_violation") if isinstance(integrity, dict) else None
    if isinstance(first, dict) and first.get("step") is not None:
        events["integrity"] = int(first["step"])
    # invalid action: a parse-error step, or a refused call other than an unknown tool
    for step in steps:
        if "parse_error" in step:
            events["invalid_action"] = index_of[id(step)]
            break
        if "action" in step and _observation_kind(step) == "invalid_action":
            events["invalid_action"] = index_of[id(step)]
            break
    # loop: recomputed over the stored steps, on the same scale as the other causes
    events["loop"] = loop_step(steps, horizon)
    # tool error: the first error not recovered within the next `window` executed steps; an
    # error with fewer than `window` executed steps after it is censored, whatever the outcome
    executed = _executed(steps)
    censored_tail = False
    for position, step in enumerate(executed):
        kind = _observation_kind(step)
        if kind not in ("tool_error", "unknown_tool"):
            continue
        later = executed[position + 1 :]
        if _recovered(step, later, kind=kind, window=window):
            continue
        if len(later) < window:
            censored_tail = True  # recovery cannot be observed inside the last `window` steps
            continue
        events["tool_error"] = index_of[id(step)]
        break
    fired = {cause: step for cause, step in events.items() if step is not None}
    first_cause = min(fired, key=lambda cause: (fired[cause], CAUSES.index(cause))) if fired else None
    success = bool(getattr(trajectory, "success", False))
    horizon = int(getattr(trajectory, "horizon", -1) or -1)
    first_step = fired[first_cause] if first_cause else None
    return {
        "events": events,
        "first_cause": first_cause,
        "first_step": first_step,
        "coherent_to_completion": success and first_cause is None,
        "success": success,
        "steps": last_index,
        "horizon": horizon,
        "first_step_over_horizon": (
            round(first_step / horizon, 4) if first_step is not None and horizon > 0 else None
        ),
        "tool_error_censored_tail": censored_tail,
        "window": window,
    }


def wilson(successes: int, total: int, z: float = 1.959964) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0, "point": 0.0}
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"low": round(max(0.0, centre - half), 4), "high": round(min(1.0, centre + half), 4), "point": round(p, 4)}


def _cell(records: list[dict[str, Any]], max_steps: int) -> dict[str, Any]:
    n = len(records)
    coherent = sum(1 for r in records if r["coherent_to_completion"])
    incidence = {cause: sum(1 for r in records if r["first_cause"] == cause) for cause in CAUSES}
    # cause-specific hazard per step: events of the cause at step s over trajectories at risk at s
    # (no event before s, and still running at s)
    hazard: dict[str, list[float | None]] = {cause: [] for cause in CAUSES}
    at_risk_by_step: list[int] = []
    for s in range(1, max_steps + 1):
        at_risk = [r for r in records if (r["first_step"] is None or r["first_step"] >= s) and r["steps"] >= s]
        at_risk_by_step.append(len(at_risk))
        for cause in CAUSES:
            events = sum(1 for r in at_risk if r["first_cause"] == cause and r["first_step"] == s)
            hazard[cause].append(round(events / len(at_risk), 4) if at_risk else None)
    # Kaplan-Meier survival of coherence, censoring at the last step for trajectories without an
    # event (completion or exhaustion); a descriptive only, with its caveat
    survival = 1.0
    km_median: float | str = "not reached"
    for s in range(1, max_steps + 1):
        at_risk = at_risk_by_step[s - 1]
        events = sum(1 for r in records if r["first_step"] == s)
        if at_risk:
            survival *= 1 - events / at_risk
        if km_median == "not reached" and survival <= 0.5:
            km_median = s
    fractions = [r["first_step_over_horizon"] for r in records if r["first_step_over_horizon"] is not None]
    fractions.sort()
    return {
        "tasks": n,
        "coherent_to_completion": coherent,
        "coherent_to_completion_rate": round(coherent / n, 4) if n else 0.0,
        "wilson_95": {"coherent_to_completion": wilson(coherent, n)},
        "successes": sum(1 for r in records if r["success"]),
        "cumulative_incidence_by_cause": {cause: round(count / n, 4) if n else 0.0 for cause, count in incidence.items()},
        "events_by_cause": incidence,
        "no_event": n - sum(incidence.values()),
        "hazard_by_cause_per_step": hazard,
        "at_risk_per_step": at_risk_by_step,
        "km_median_step_descriptive": km_median,
        "first_event_step_over_horizon_median": (fractions[len(fractions) // 2] if fractions else None),
        "tool_error_censored_tails": sum(1 for r in records if r["tool_error_censored_tail"]),
    }


def coherence_summary(
    trajectories: list[Any],
    *,
    max_steps: int = 24,
    window: int = DEFAULT_WINDOW,
    tools: list[dict[str, Any]] = TOOL_SPECS,
) -> dict[str, Any]:
    """The coherence section of an evaluation summary."""
    records = [trajectory_events(t, window=window, tools=tools) for t in trajectories]
    for record, trajectory in zip(records, trajectories):
        record["task_id"] = getattr(trajectory, "task_id", None)
        record["family"] = getattr(trajectory, "family", None)
        record["variant"] = getattr(trajectory, "variant", None)
        record["difficulty"] = getattr(trajectory, "difficulty", None)

    def grouped(key: str) -> dict[str, dict[str, Any]]:
        table: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            table.setdefault(str(record[key]), []).append(record)
        return {name: _cell(rows, max_steps) for name, rows in sorted(table.items())}

    return {
        "standard": "coherent to completion: success with no incoherence event; four causes as competing risks (UNIFIED-RUN-2026-09-06 section 2 revised, rulings of 10:12)",
        "window": window,
        "primary_arguments": primary_arguments(tools),
        "overall": _cell(records, max_steps),
        "by_family": grouped("family"),
        "by_variant": grouped("variant"),
        "by_difficulty": grouped("difficulty"),
        "per_trajectory": records,
        "comparable_across_cells": ["hazard_by_cause_per_step", "first_event_step_over_horizon_median", "cumulative_incidence_by_cause", "coherent_to_completion_rate"],
        "not_comparable_across_cells": ["km_median_step_descriptive", "steps"],
        "caveats": [
            "the Kaplan-Meier median is a descriptive: censoring at completion is informative because completion is correlated with capability",
            "verdict.recovered_errors is not a recovery rule (env.py: the error count on success, else zero) and is not used",
            "raw step counts are not comparable across families or difficulties, whose horizons increase by construction",
        ],
    }
