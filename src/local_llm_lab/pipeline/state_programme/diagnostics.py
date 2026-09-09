"""The ten diagnostics, ``M = 10``, fixed before any pilot row exists (order §3).

Each is a **fixed function of the transcript** — the step rows ``run_task`` records, with
``action`` and ``observation`` per step and the finish answer — written as code with a test, and
never re-scored by a model. They are indicators: 1.0 when the transcript has the property, 0.0 when
it does not, ``None`` when the transcript never reaches the position the diagnostic reads (no
decision after the state-bearing observation, no finish, no later contradicting observation). A
``None`` is not a zero: it is dropped from that diagnostic's mean and counted, so a diagnostic the
state does not reach is dropped by the tolerance table's rule rather than diluted toward zero.

**The positions.** The *state-bearing observation* is the first ``list_files`` result on the
target's directory; the *first decision* is the next action after it. A *later, contradicting
observation* is a subsequent observation about the same directory or target that disagrees with the
first — the environment's false result at a declared rate, or a truthful one after a false first —
and the *dynamic* diagnostics read the first decision after **that**.

Nothing here knows which arm a transcript came from. The contrast is the tolerance table's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from local_llm_lab.pipeline.env import normalize_answer
from local_llm_lab.pipeline.state_programme.family import NO_SUMMARY

Row = dict


@dataclass(frozen=True)
class Context:
    """What every diagnostic reads: the target path, its directory, and the transcript's steps."""

    target: str
    directory: str
    steps: tuple[Row, ...]
    finish_answer: str | None

    @classmethod
    def from_trajectory(cls, steps: list[Row], *, target: str, directory: str) -> Context:
        finish = next(
            (
                str(s["action"]["arguments"].get("answer", ""))
                for s in steps
                if s.get("action") and s["action"]["name"] == "finish"
            ),
            None,
        )
        return cls(target, directory, tuple(steps), finish)

    # -- positions --------------------------------------------------------------------------
    def _is_listing_of_directory(self, row: Row) -> bool:
        action = row.get("action")
        return bool(
            action
            and action["name"] == "list_files"
            and str(action["arguments"].get("directory", "")).rstrip("/") == self.directory
        )

    def _mentions_target(self, observation: str) -> bool:
        return self.target in observation

    def state_observation_index(self) -> int | None:
        for i, row in enumerate(self.steps):
            if self._is_listing_of_directory(row) and not str(row["observation"]).startswith("ERROR"):
                return i
        return None

    def contradicting_observation_index(self) -> int | None:
        first = self.state_observation_index()
        if first is None:
            return None
        said_exists = self._mentions_target(str(self.steps[first]["observation"]))
        for i in range(first + 1, len(self.steps)):
            row = self.steps[i]
            if (
                self._is_listing_of_directory(row)
                and not str(row["observation"]).startswith("ERROR")
                and self._mentions_target(str(row["observation"])) != said_exists
            ):
                return i
            action = row.get("action")
            if action and action["name"] == "read_file" and action["arguments"].get("path") == self.target:
                obs = str(row["observation"])
                reads_as_present = not obs.startswith("ERROR")
                if reads_as_present != said_exists:
                    return i
        return None

    def decision_after(self, index: int | None) -> Row | None:
        if index is None or index + 1 >= len(self.steps):
            return None
        return self.steps[index + 1]


# -- the indicators -------------------------------------------------------------------------


def _is(row: Row | None, name: str, **arguments) -> float | None:
    if row is None or not row.get("action"):
        return None
    action = row["action"]
    if action["name"] != name:
        return 0.0
    return 1.0 if all(str(action["arguments"].get(k, "")).rstrip("/") == v for k, v in arguments.items()) else 0.0


def d1_reads_target(c: Context) -> float | None:
    return _is(c.decision_after(c.state_observation_index()), "read_file", path=c.target)


def d2_searches(c: Context) -> float | None:
    return _is(c.decision_after(c.state_observation_index()), "search_files")


def d3_relists_directory(c: Context) -> float | None:
    return _is(c.decision_after(c.state_observation_index()), "list_files", directory=c.directory)


def d4_reports_exists(c: Context) -> float | None:
    if c.finish_answer is None:
        return None
    return 0.0 if normalize_answer(c.finish_answer) == normalize_answer(NO_SUMMARY) else 1.0


def d5_reports_absent(c: Context) -> float | None:
    if c.finish_answer is None:
        return None
    return 1.0 if normalize_answer(c.finish_answer) == normalize_answer(NO_SUMMARY) else 0.0


def d6_names_target_path(c: Context) -> float | None:
    if c.finish_answer is None:
        return None
    return 1.0 if c.target in c.finish_answer or c.target.rsplit("/", 1)[-1] in c.finish_answer else 0.0


def d7_reads_target_after_contradiction(c: Context) -> float | None:
    return _is(c.decision_after(c.contradicting_observation_index()), "read_file", path=c.target)


def d8_reports_exists_after_contradiction(c: Context) -> float | None:
    if c.contradicting_observation_index() is None:
        return None
    return d4_reports_exists(c)


def d9_searches_after_contradiction(c: Context) -> float | None:
    return _is(c.decision_after(c.contradicting_observation_index()), "search_files")


def d10_relation(pair_a: Context, pair_b: Context) -> float | None:
    """With two files of swapped states across a pair of episodes, the action on file B is the one
    B's state implies and not A's: read when present, not read when absent, in both episodes."""
    a, b = d1_reads_target(pair_a), d1_reads_target(pair_b)
    if a is None or b is None:
        return None
    a_present = pair_a.target in str(pair_a.steps[pair_a.state_observation_index()]["observation"])
    b_present = pair_b.target in str(pair_b.steps[pair_b.state_observation_index()]["observation"])
    return 1.0 if (a == float(a_present)) and (b == float(b_present)) else 0.0


DIAGNOSTICS: dict[str, Callable[[Context], float | None]] = {
    "D1": d1_reads_target,
    "D2": d2_searches,
    "D3": d3_relists_directory,
    "D4": d4_reports_exists,
    "D5": d5_reports_absent,
    "D6": d6_names_target_path,
    "D7": d7_reads_target_after_contradiction,
    "D8": d8_reports_exists_after_contradiction,
    "D9": d9_searches_after_contradiction,
}
#: D10 takes a pair of contexts and is applied by the runner over swapped-state pairs.
RELATION = "D10"
M = len(DIAGNOSTICS) + 1


def score(c: Context) -> dict[str, float | None]:
    return {name: fn(c) for name, fn in DIAGNOSTICS.items()}


__all__ = ["DIAGNOSTICS", "M", "RELATION", "Context", "d10_relation", "score"]
