"""Codex's six witnesses from `WSA-E2-READER-RECHECK-2026-09-11`, re-run against the fixed reader.

Each check is their reproduction, asserting the **fixed** behaviour instead of the defect. Committed
so the review can be replayed rather than taken on my word, and so a later change that reintroduces
any of the six fails here.

    python check_findings.py            # all six, exit 0 only if every one is fixed

No capture, model, residual or estimand is touched: every fixture is synthetic or metadata-only.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))

import read_e2  # noqa: E402
from local_llm_lab.pipeline.state_programme.read_gate import (  # noqa: E402
    NotSealed, SealBroken, require_addendum, require_seal,
)


def f1_addendum_required() -> list[str]:
    """F1: the reader must require an active, non-superseded addendum that lists its own bytes."""
    problems = []
    seal = require_seal(HERE)
    listed = {read_e2.READER: HERE / "read_e2.py",
              read_e2.RULE_MODULE: HERE.parents[2] / read_e2.RULE_MODULE}

    with tempfile.TemporaryDirectory() as tmp:
        bare = Path(tmp) / "research" / "records" / HERE.name
        bare.mkdir(parents=True)
        for name in ("seal.json", "folds.json", "read_e2.py"):
            shutil.copy(HERE / name, bare / name)
        try:  # no addendum at all
            require_addendum(bare, parent=seal, must_list=listed)
            problems.append("F1: an absent addendum was accepted")
        except NotSealed:
            pass

        shutil.copy(HERE / "addendum-1.json", bare / "addendum-1.json")
        (bare / "addendum-1-SUPERSEDED.md").write_text("void")
        try:  # present but superseded
            require_addendum(bare, parent=seal, must_list=listed)
            problems.append("F1: a superseded addendum was accepted")
        except NotSealed:
            pass

        (bare / "addendum-1-SUPERSEDED.md").unlink()
        try:  # active, but it does not list this reader's current bytes
            require_addendum(bare, parent=seal, must_list=listed)
            problems.append("F1: an addendum that does not list the reader's bytes was accepted")
        except (NotSealed, SealBroken):
            pass
    return problems


def f2_gate_gates() -> list[str]:
    """F2: a capability below the threshold must stop scoring, not print beside a result."""
    if read_e2.GATE != 0.5:
        return [f"F2: the gate threshold is {read_e2.GATE}, not §5's 0.5"]
    source = (HERE / "read_e2.py").read_text()
    gate_at = source.index("# PHASE 1")
    score_at = source.index("# PHASE 2")
    refuse_at = source.index("no stratum is scored on either model")
    if not gate_at < refuse_at < score_at:
        return ["F2: the refusal does not sit between the gate and the scoring phase"]
    return []


def f3_tolerances_attached() -> list[str]:
    """F3: every registered stratum carries its sealed tolerance, requirement, bound and reading."""
    seal = require_seal(HERE)
    problems = []
    for label, key in read_e2.STRATA:
        tolerance = read_e2.tolerance_for(seal, key)
        if key is None:
            if tolerance is not None:
                problems.append(f"F3: {label} is descriptive but carries a tolerance")
            continue
        if tolerance is None:
            problems.append(f"F3: {label} has no tolerance")
            continue
        for field in ("epsilon", "needs_n_at_least", "m", "alpha", "range_width"):
            if field not in tolerance:
                problems.append(f"F3: {label}'s tolerance lacks {field}")
    expected = {"ordinary_train": 0.13, "corrective": 0.15,
                "corrective_contiguous": 0.23, "corrective_across_a_gap": 0.20}
    for label, epsilon in expected.items():
        got = read_e2.tolerance_for(seal, dict(read_e2.STRATA)[label])["epsilon"]
        if abs(got - epsilon) > 1e-12:
            problems.append(f"F3: {label} resolves to ε = {got}, not {epsilon}")
    return problems


class _Rule:
    """A stub with a fixed rank ceiling and an exact affine map, for F4 and F5."""

    def __init__(self, max_rank: int, out=None):
        self.max_rank = max_rank
        self._out = out

    def apply(self, x, *, rank, times=1):
        if self._out is not None:
            return np.asarray(self._out, dtype=np.float64)
        return np.asarray(x, dtype=np.float64)


def f4_short_rank_coverage() -> list[str]:
    """F4: a rank one fold cannot reach must reduce the reported population, not be absorbed."""
    features = np.zeros((3, 2), dtype=np.float32)
    moves = [dict(task_id="a", source=0, target=1, m=1, fold=0, candidates=[0, 1, 2]),
             dict(task_id="b", source=0, target=1, m=1, fold=1, candidates=[0, 1, 2])]
    rules = {"a": _Rule(1), "b": _Rule(2)}
    rows, unreached = read_e2.score(features, moves, rules, 2)
    problems = []
    if len(rows) != 1 or unreached != {0}:
        problems.append(f"F4: score scored {len(rows)} rows and named {unreached} unreachable")
    cell = read_e2.evaluate(moves, features, rules, 2,
                            {"name": "x", "epsilon": 0.1, "needs_n_at_least": 1,
                             "m": 12, "alpha": 0.05, "range_width": 2.0}, 1, True)
    if cell["status"] != "reduced population":
        problems.append(f"F4: status is {cell['status']!r}, not 'reduced population'")
    if cell["requested_transitions"] != 2 or cell["scored_transitions"] != 1:
        problems.append("F4: requested and scored transitions are not both reported")
    if cell["tolerance"]["applied"]:
        problems.append("F4: the complete stratum's tolerance was applied to a reduced population")
    if 0 not in cell["folds_unavailable"]:
        problems.append("F4: the unavailable fold is not named")
    return problems


def f5_float32_at_the_boundary() -> list[str]:
    """F5: Codex's tie witness — float64 gives a hit, §4.2's float32 gives a tie and so a miss."""
    features = np.array([[0.0], [2.0]], dtype=np.float32)
    prediction = 1.0 + 2.0 ** -26  # distinguishable in float64, exactly 1.0 in float32
    moves = [dict(task_id="t", source=0, target=1, m=1, fold=0, candidates=[0, 1])]
    rows, _ = read_e2.score(features, moves, {"t": _Rule(8, out=[prediction])}, 8)
    if rows[0]["hit"] != 0.0:
        return ["F5: the retrieval ran in float64; the tie witness was scored a hit"]
    return []


def f6_labels() -> list[str]:
    """F6: the contrast's nonpositive mass is at-or-below chance; rates keep exact endpoints."""
    contrast = read_e2.tails(np.array([-0.5, 0.0, 0.5]), kind="contrast")
    rate = read_e2.tails(np.array([0.0, 0.5, 1.0]), kind="rate")
    problems = []
    if "share_at_zero" in contrast:
        problems.append("F6: a contrast still reports 'share_at_zero'")
    if abs(contrast.get("share_at_or_below_chance", -1) - 2 / 3) > 1e-12:
        problems.append("F6: the at-or-below-chance share is wrong")
    if abs(contrast.get("share_exactly_at_chance", -1) - 1 / 3) > 1e-12:
        problems.append("F6: the exact-at-chance share is wrong")
    if abs(rate.get("share_exactly_zero", -1) - 1 / 3) > 1e-12:
        problems.append("F6: the hit-rate zero mass is wrong")
    return problems


def main() -> int:
    checks = (("F1 addendum required", f1_addendum_required),
              ("F2 the gate gates", f2_gate_gates),
              ("F3 tolerances attached", f3_tolerances_attached),
              ("F4 short-rank coverage", f4_short_rank_coverage),
              ("F5 float32 at the boundary", f5_float32_at_the_boundary),
              ("F6 labels", f6_labels))
    failures = []
    for name, check in checks:
        try:
            found = check()
        except Exception as exc:  # a witness that cannot even run is a failed witness, not a crash
            found = [f"{name} could not run: {type(exc).__name__}: {exc}"]
        print(f"  {'FAIL' if found else 'ok  '}  {name}")
        for problem in found:
            print(f"          {problem}")
        failures += found
    print(f"\n{'FAILED' if failures else 'all six of Codex 3af4c56 fixed'}: "
          f"{len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
