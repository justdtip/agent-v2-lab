"""Stages 0–4 (order §1), with two seams so the fixture run and the device run share one code path.

**The policy seam.** ``run_task`` in the runner takes a model and a tokenizer and has no other way
in, so this module drives the environment itself: a :class:`Policy` maps the transcript so far to
one ``(thought, Action)``. The fixture uses :class:`ScriptedPolicy`; the device driver wraps the
model behind the same protocol. Step rows are written in the runner's own shape (``index``,
``thought``, ``action``, ``observation``, ``raw``, ``parse_error``), so a diagnostic is one function
over both.

**The wrapper seam.** The estimands patch a state direction into the reference arm's decoding
through Codex's decoder intervention (WS-A order, 2026-09-10). On the laptop that wrapper is a stub
taking a context and returning the rows it produces; the estimand code reads rows and never the
wrapper, so the device swaps the stub for the real one without touching a distance.

**Resume keys on the tree that ran** (first-hour runbook): the working tree's content including
untracked files, the checkpoint digest, the device reading and the seal digest. A row whose key
matches is not re-run; a row whose key differs refuses, naming every field that differs.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from local_llm_lab import device, spawn
from local_llm_lab.agent_protocol import Action
from local_llm_lab.pipeline.env import Simulator
from local_llm_lab.pipeline.state_programme import diagnostics as dx
from local_llm_lab.pipeline.state_programme import tolerances as tol
from local_llm_lab.pipeline.state_programme.family import (
    ExistencePair,
    apply_reliability,
    make_existence_pairs,
)
from local_llm_lab.pipeline.state_programme.record import (
    append_row,
    read_rows,
    write_json,
    write_readme,
)
from local_llm_lab.pipeline.state_programme.seal import (
    content_key,
    refuse_rederive,
    require_seal,
    seal,
    seal_digest,
)
from local_llm_lab.pipeline.tasks import Task

ESTIMANDS = {
    "greedy": "context-averaged outcome frequency over the task distribution (one outcome per context)",
    "sampled": "the model's distribution over outcomes in a context (the derivation's estimand)",
}


class Policy(Protocol):
    def __call__(self, task: Task, steps: list[dict[str, Any]]) -> tuple[str, Action]: ...


@dataclass
class ScriptedPolicy:
    """Follows the task's expert steps; ``deviations`` maps a step index to a replacement, so a
    test can make a scripted policy do the wrong thing on purpose."""

    deviations: dict[int, tuple[str, Action]] = field(default_factory=dict)

    def __call__(self, task: Task, steps: list[dict[str, Any]]) -> tuple[str, Action]:
        index = len(steps)
        if index in self.deviations:
            return self.deviations[index]
        if index >= len(task.steps):
            return ("Nothing further to do.", Action("finish", {"answer": task.expected_answer}))
        step = task.steps[index]
        return step.thought, step.action


def run_episode(task: Task, policy: Policy, *, max_steps: int = 24) -> dict[str, Any]:
    started = time.monotonic()
    simulator = Simulator.for_task(task)
    steps: list[dict[str, Any]] = []
    for index in range(1, max_steps + 1):
        thought, action = policy(task, steps)
        observation = simulator.execute(action)
        steps.append({
            "index": index, "thought": thought,
            "action": {"name": action.name, "arguments": dict(action.arguments)},
            "observation": observation, "raw": None, "parse_error": None,
        })
        if action.name == "finish":
            break
    return {
        "task_id": task.task_id, "family": task.family, "variant": task.variant,
        "steps": steps, "verdict": simulator.verdict().as_dict(),
        "faults": [f.call_index for f in task.faults], "elapsed_s": time.monotonic() - started,
    }


# ------------------------------------------------------------------------------- stage 0


def tree_content_digest(root: Path) -> str:
    """Tracked and untracked files alike, by content: the tree that ran, not the commit."""
    # `spawn.run`, never `subprocess`: a fork in an interpreter with Metal up aborts in
    # libplatform without raising (R45; `local_llm_lab.spawn` carries the crash report).
    listed = spawn.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True, check=True,
    ).stdout.decode().split("\0")
    h = hashlib.sha256()
    for rel in sorted(p for p in listed if p):
        path = root / rel
        if path.is_file():
            for piece in (rel.encode(), b"\0", path.read_bytes(), b"\0"):
                h.update(piece)
    return h.hexdigest()


def preflight(
    *, root: Path, registry_name: str, checkpoint_digest: str, lens_identity: dict[str, Any],
    dictionary_layers: dict[str, str], wrapper_version: str, decoding: str, temperature: float | None,
    seed: int, fault_rate: float, fault_seed: int, fixture: bool,
) -> dict[str, Any]:
    """Stage 0: written before any episode. A missing input stops here, by name."""
    missing = [name for name, value in (
        ("registry_name", registry_name), ("checkpoint_digest", checkpoint_digest),
        ("lens_identity", lens_identity), ("dictionary_layers", dictionary_layers),
        ("wrapper_version", wrapper_version),
    ) if not value]
    if missing:
        raise ValueError(f"preflight stops: missing {missing}")
    if decoding not in ESTIMANDS:
        raise ValueError(f"decoding must be one of {sorted(ESTIMANDS)}; got {decoding!r}")
    if decoding == "greedy" and temperature not in (None, 0.0):
        raise ValueError("greedy decoding carries no temperature; a temperature is the sampled mode's")
    return {
        "schema_version": 1, "fixture": fixture, "registry_name": registry_name,
        "checkpoint_digest": checkpoint_digest, "lens_identity": lens_identity,
        "dictionary_layers": dictionary_layers, "wrapper_version": wrapper_version,
        "decoding": {"mode": decoding, "temperature": temperature, "seed": seed,
                     "estimand": ESTIMANDS[decoding]},
        "faults": {"rate": fault_rate, "seed": fault_seed},
        "device": device.describe(), "tree_content_digest": tree_content_digest(root),
    }


# ------------------------------------------------------------------------------- stage 1


def _score_row(row: dict[str, Any], pair: ExistencePair) -> dict[str, float | None]:
    ctx = dx.Context.from_trajectory(row["steps"], target=pair.target, directory=pair.directory)
    return dx.score(ctx)


def pilot(
    directory: Path, *, pairs: list[ExistencePair], policy: Policy, decoding: str,
    fault_rate: float, fault_seed: int,
) -> Path:
    """Stage 1: matched pairs, plus the reliability arm. ``rate.json`` is written first."""
    rows_path = directory / "pilot" / "rows.jsonl"
    started = time.monotonic()
    episodes = 0
    for pair in pairs:
        for arm, task in pair.arms():
            row = run_episode(task, policy)
            row.update(pair_id=pair.pair_id, arm=arm, falsified=False, scores=_score_row(row, pair))
            append_row(rows_path, row)
            episodes += 1
    for pair_id, arm, task, falsified in apply_reliability(pairs, rate=fault_rate, seed=fault_seed):
        pair = next(p for p in pairs if p.pair_id == pair_id)
        row = run_episode(task, policy)
        row.update(pair_id=pair_id, arm=f"R{arm}", falsified=falsified, scores=_score_row(row, pair))
        append_row(rows_path, row)
        episodes += 1
    hours = (time.monotonic() - started) / 3600
    write_json(directory / "rate.json", {
        "episodes": episodes, "hours": hours, "episodes_per_hour": episodes / hours if hours else None,
        "mode": decoding, "basis": "measured-here",
    })
    return rows_path


# ------------------------------------------------------------------------------- stage 2


def split_half_log_loss_gap(rows: list[dict[str, Any]]) -> float:
    """The noise floor of the pilot's own fitted belief update: fit a Bernoulli on D4 in each half
    of the reliability arm, score log loss on the other half, and take the gap between halves."""
    scored = [r["scores"]["D4"] for r in rows if r["arm"].startswith("R") and r["scores"]["D4"] is not None]
    if len(scored) < 4:
        return 0.0
    half = len(scored) // 2
    a, b = scored[:half], scored[half:]

    def loss(fit: list[float], score: list[float]) -> float:
        p = min(max(sum(fit) / len(fit), 1e-6), 1 - 1e-6)
        return -sum(s * math.log(p) + (1 - s) * math.log(1 - p) for s in score) / len(score)

    return abs(loss(a, b) - loss(b, a))


def derive_tolerances(rows: list[dict[str, Any]], *, seed: int) -> tol.Table:
    by_arm = {"E": {}, "A": {}}
    for row in rows:
        if row["arm"] in by_arm:
            for name, value in row["scores"].items():
                by_arm[row["arm"]].setdefault(name, []).append(value)
    rows_c = tol.contrasts(by_arm["E"], by_arm["A"], seed=seed)
    return tol.derive(rows_c, split_half_log_loss_gap=split_half_log_loss_gap(rows))


# ------------------------------------------------------------------------------- stage 4


Wrapper = Callable[[str, Task, str], dict[str, Any]]
"""``(patch_kind, task, arm) -> row``: decode ``task`` with the named patch applied. ``patch_kind``
is ``"substitution"``, ``"control"`` or ``"reuse"``. The stub on the laptop returns the reference
arm's scripted row; the device passes Codex's intervention."""


def resume_key(manifest: dict[str, Any], directory: Path) -> tuple[str, dict[str, str]]:
    parts = {
        "tree_content_digest": manifest["tree_content_digest"],
        "checkpoint_digest": manifest["checkpoint_digest"],
        "device": json.dumps(manifest["device"], sort_keys=True),
        "seal_digest": seal_digest(directory),
    }
    return content_key(parts), parts


def main_run(
    directory: Path, *, manifest: dict[str, Any], pairs: Iterable[ExistencePair], policy: Policy,
    wrapper: Wrapper,
) -> Path:
    """Stage 4: ``n`` matched pairs, resumable; then the estimands on the same contexts."""
    require_seal(directory)
    key, parts = resume_key(manifest, directory)
    rows_path = directory / "main" / "rows.jsonl"
    existing = read_rows(rows_path)
    foreign = [r for r in existing if r["resume_key"] != key]
    if foreign:
        theirs = foreign[0]["resume_parts"]
        differing = sorted(k for k in parts if theirs.get(k) != parts[k])
        raise RuntimeError(
            f"{len(foreign)} rows under {rows_path} were run under a different key; fields that "
            f"differ: {differing}. A modified tree resumes only its own records."
        )
    done = {(r["pair_id"], r["arm"], r["condition"]) for r in existing}
    for pair in pairs:
        for arm, task in pair.arms():
            if (pair.pair_id, arm, "base") not in done:
                row = run_episode(task, policy)
                row.update(pair_id=pair.pair_id, arm=arm, condition="base",
                           scores=_score_row(row, pair), resume_key=key, resume_parts=parts)
                append_row(rows_path, row)
        for condition in ("substitution", "control", "reuse"):
            if (pair.pair_id, "A", condition) not in done:
                row = wrapper(condition, pair.absent, "A")
                row.update(pair_id=pair.pair_id, arm="A", condition=condition,
                           scores=_score_row(row, pair), resume_key=key, resume_parts=parts)
                append_row(rows_path, row)
    return rows_path


def _mean(values: list[float | None]) -> float | None:
    seen = [v for v in values if v is not None]
    return sum(seen) / len(seen) if seen else None


def estimands(directory: Path) -> Path:
    """Distances between arm means over the retained diagnostics, at the level the seal fixes."""
    sealed = require_seal(directory)
    t = sealed["tolerances"]
    rows = read_rows(directory / "main" / "rows.jsonl")
    retained = t["retained"]

    def means(arm: str, condition: str) -> dict[str, float | None]:
        sel = [r for r in rows if r["arm"] == arm and r["condition"] == condition]
        return {d: _mean([r["scores"][d] for r in sel]) for d in retained}

    reference, absent = means("E", "base"), means("A", "base")
    sub, ctrl, reuse = means("A", "substitution"), means("A", "control"), means("A", "reuse")

    def distance(x: dict, y: dict) -> float:
        pairs_ = [(x[d], y[d]) for d in retained if x[d] is not None and y[d] is not None]
        return max((abs(a - b) for a, b in pairs_), default=float("nan"))

    dyn = [r for r in rows if r["arm"] == "A" and r["condition"] == "base"]
    d_pred = _mean([r["scores"].get("D8") for r in dyn])
    out = {
        "substitution": {"distance": distance(sub, reference), "tolerance": t["epsilon_sub"]},
        "specificity": {"distance": distance(ctrl, absent), "tolerance": t["epsilon_perp"]},
        "reuse": {"distance": distance(reuse, reference), "tolerance": t["epsilon_reuse"]},
        "predictive": {"distance": 0.0 if d_pred is None else abs(d_pred - (reference.get("D8") or 0.0)),
                        "tolerance": t["epsilon_pred"]},
    }
    for e in out.values():
        e["passes"] = (e["distance"] == e["distance"]) and e["distance"] <= e["tolerance"]
    payload = {
        "schema_version": 1, "level": "J-space vectors (diagnostic means); never a coefficient table",
        "retained": retained, "arm_means": {"E_base": reference, "A_base": absent,
                                             "A_substitution": sub, "A_control": ctrl, "A_reuse": reuse},
        "estimands": out, "rows": len(rows),
    }
    return write_json(directory / "estimands.json", payload)


# --------------------------------------------------------------------------- the whole script


WrapperFactory = Callable[[list[ExistencePair]], Wrapper]


def run(
    directory: Path, *, root: Path, policy: Policy, wrapper: WrapperFactory, decoding: str,
    pilot_pairs: int, level: int, seed: int, fault_rate: float, fault_seed: int,
    hours_bought: float | None, manifest_inputs: dict[str, Any], main_pairs: int | None = None,
    temperature: float | None = None, fixture: bool = True,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = preflight(root=root, decoding=decoding, temperature=temperature, seed=seed,
                         fault_rate=fault_rate, fault_seed=fault_seed, fixture=fixture, **manifest_inputs)
    write_json(directory / "manifest.json", manifest)
    pairs = make_existence_pairs("pilot", pilot_pairs, level, seed=seed)
    rows_path = pilot(directory, pairs=pairs, policy=policy, decoding=decoding,
                      fault_rate=fault_rate, fault_seed=fault_seed)
    refuse_rederive(directory)
    table = derive_tolerances(read_rows(rows_path), seed=seed)
    rate = json.loads((directory / "rate.json").read_text())
    budget = None
    if hours_bought is not None and rate["episodes_per_hour"]:
        budget = tol.budget_rule(table, hours_bought=hours_bought, rate_per_hour=rate["episodes_per_hour"]).as_dict()
    seal(directory, table=table.as_dict(), budget=budget, decoding=manifest["decoding"], pilot_rows_path=rows_path)
    n = main_pairs if main_pairs is not None else (budget["n_final"] if budget else table.n)
    main_pairs_ = make_existence_pairs("main", n, level, seed=f"{seed}:main")
    main_run(directory, manifest=manifest, policy=policy, wrapper=wrapper(main_pairs_), pairs=main_pairs_)
    estimands(directory)
    write_readme(directory)
    return {"n": n, "retained": table.retained, "seal_digest": seal_digest(directory)}


def scripted_wrapper(policy: Policy, pairs: Iterable[ExistencePair]) -> Wrapper:
    """The laptop stub, with the wrapper's shape and none of its content.

    A 'substitution' or 'reuse' patch decodes the pair's *exists* arm — the reference the patch is
    meant to move the absent arm toward — and a 'control' decodes the absent arm unchanged. So on
    fixtures substitution passes and specificity passes by construction, which is what a stub can
    show: that the estimand code reads rows correctly. It shows nothing about a model, and every row
    it writes says ``wrapper: stub``.
    """
    by_pair = {p.pair_id: p for p in pairs}

    def apply(condition: str, task: Task, arm: str) -> dict[str, Any]:
        pair = by_pair[task.task_id[:-2]]
        target = pair.absent if condition == "control" else pair.exists
        return run_episode(target, policy) | {"wrapper": "stub", "patched": condition, "arm_decoded": arm}

    return apply


__all__ = ["ESTIMANDS", "Policy", "ScriptedPolicy", "Wrapper", "WrapperFactory", "derive_tolerances", "estimands",
           "main_run", "pilot", "preflight", "resume_key", "run", "run_episode",
           "scripted_wrapper", "split_half_log_loss_gap", "tree_content_digest"]
