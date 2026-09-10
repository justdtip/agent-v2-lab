"""The derivation table: every tolerance a line with its inputs, none typed by hand (order §2, §7).

The bound is Hoeffding's over ``M`` fixed diagnostics at confidence ``1 − α``:
``n ≥ ln(2M/α) / ε²``. The pilot's design tool is a seeded bootstrap lower bound on each arm
contrast; the Hoeffding interval is wider by construction and is the *main* run's guarantee.

Two rules are mechanisms here and not advice. **The drop rule**: a diagnostic whose bootstrap lower
bound does not exceed zero is dropped from ``M`` with its reason, because the state does not reach it
and a tolerance for it would be a tolerance for noise; nothing is added to ``M`` after the pilot.
**The budget rule**: if the main run's device time exceeds the hours bought, ``ε_sub`` is loosened
to the value the budget allows and **both** numbers and the reason are written before the run —
never fewer diagnostics, never no control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ALPHA = 0.05
RESAMPLES = 10_000


@dataclass(frozen=True)
class Contrast:
    name: str
    mean_e: float
    mean_a: float
    difference: float
    lower_bound: float  # one-sided (1 - alpha) lower bound on mean_e - mean_a
    upper_bound: float  # one-sided (1 - alpha) upper bound, for a contrast that lies below zero
    n_e: int
    n_a: int
    reached: bool  # both arms produced at least one non-None score


def bootstrap_bounds(
    e: np.ndarray, a: np.ndarray, *, resamples: int = RESAMPLES, seed: int, alpha: float = ALPHA
) -> tuple[float, float]:
    """One-sided ``(1 − α)`` lower and upper bounds on ``mean(E) − mean(A)``, each arm resampled
    with replacement, seeded. Both sides are returned because a contrast the state pushes *down*
    is as much a reach as one it pushes up, and the drop rule must read the side the contrast is on."""
    e = np.asarray(e, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    if e.size == 0 or a.size == 0:
        raise ValueError("a bootstrap needs at least one observation in each arm")
    rng = np.random.default_rng(seed)
    idx_e = rng.integers(0, e.size, size=(resamples, e.size))
    idx_a = rng.integers(0, a.size, size=(resamples, a.size))
    diffs = e[idx_e].mean(axis=1) - a[idx_a].mean(axis=1)
    return float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha))


def paired_bootstrap_lower_bound(
    e: np.ndarray, a: np.ndarray, *, resamples: int = RESAMPLES, seed: int, alpha: float = ALPHA
) -> float:
    return bootstrap_bounds(e, a, resamples=resamples, seed=seed, alpha=alpha)[0]


def contrasts(
    scores_e: dict[str, list[float | None]],
    scores_a: dict[str, list[float | None]],
    *,
    seed: int,
    resamples: int = RESAMPLES,
) -> list[Contrast]:
    out = []
    for name in scores_e:
        e = np.array([v for v in scores_e[name] if v is not None], dtype=np.float64)
        a = np.array([v for v in scores_a.get(name, []) if v is not None], dtype=np.float64)
        reached = e.size > 0 and a.size > 0
        nan = float("nan")
        if not reached:
            out.append(Contrast(name, nan, nan, nan, nan, nan, e.size, a.size, False))
            continue
        d = float(e.mean() - a.mean())
        lb, ub = bootstrap_bounds(e, a, resamples=resamples, seed=seed)
        out.append(Contrast(name, float(e.mean()), float(a.mean()), d, lb, ub, e.size, a.size, True))
    return out


@dataclass(frozen=True)
class Dropped:
    name: str
    reason: str


def drop_rule(rows: list[Contrast]) -> tuple[list[Contrast], list[Dropped]]:
    """Keep a diagnostic only if its lower bound exceeds zero; record why each other one is dropped.

    The sign is the diagnostic's own: a contrast that is reliably *negative* (the absent arm does
    the thing more) is as much a reach as a positive one, so the test is on ``|difference|`` with
    the lower bound taken on the side of the difference.
    """
    kept, dropped = [], []
    for row in rows:
        if not row.reached:
            dropped.append(Dropped(row.name, "not reached: one arm produced no scorable transcript"))
            continue
        if magnitude_bound(row) <= 0.0:
            dropped.append(
                Dropped(
                    row.name,
                    f"the state does not reach it: bound on |contrast| is "
                    f"{magnitude_bound(row):.4f} ≤ 0 (lower {row.lower_bound:.4f}, "
                    f"upper {row.upper_bound:.4f})",
                )
            )
            continue
        kept.append(row)
    return kept, dropped


def magnitude_bound(row: Contrast) -> float:
    """The (1 − α) bound on ``|mean_e − mean_a|`` from the side the contrast lies on: the lower
    bound for a positive contrast, the negated upper bound for a negative one. Positive iff the
    interval excludes zero on that side."""
    return row.lower_bound if row.difference >= 0 else -row.upper_bound


def required_n(m: int, epsilon: float, *, alpha: float = ALPHA) -> int:
    if m < 1 or epsilon <= 0 or not (0 < alpha < 1):
        raise ValueError("need M ≥ 1, ε > 0, 0 < α < 1")
    return math.ceil(math.log(2 * m / alpha) / epsilon**2)


#: The two closed-form checks the order fixes: at M = 10, ε = 0.05 → 2,397 and ε = 0.075 → 1,066.
CLOSED_FORM_CHECKS = ((10, 0.05, 2397), (10, 0.075, 1066))


def verify_closed_forms() -> None:
    for m, eps, expected in CLOSED_FORM_CHECKS:
        got = required_n(m, eps)
        if got != expected:
            raise AssertionError(f"required_n({m}, {eps}) = {got}, expected {expected}")


@dataclass(frozen=True)
class Table:
    retained: tuple[str, ...]
    dropped: tuple[Dropped, ...]
    m: int
    d_min: float
    epsilon_sub: float
    epsilon_perp: float
    epsilon_reuse: float
    epsilon_pred: float
    epsilon_dyn: float
    n: int
    alpha: float = ALPHA
    derivation: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "retained": list(self.retained),
            "dropped": [{"name": d.name, "reason": d.reason} for d in self.dropped],
            "M": self.m,
            "d_min": self.d_min,
            "epsilon_sub": self.epsilon_sub,
            "epsilon_perp": self.epsilon_perp,
            "epsilon_reuse": self.epsilon_reuse,
            "epsilon_pred": self.epsilon_pred,
            "epsilon_dyn": self.epsilon_dyn,
            "n": self.n,
            "alpha": self.alpha,
            "derivation": self.derivation,
        }


def derive(rows: list[Contrast], *, split_half_log_loss_gap: float, alpha: float = ALPHA) -> Table:
    """The table, each line from its inputs, refusing when nothing is retained."""
    kept, dropped = drop_rule(rows)
    if not kept:
        raise ValueError("every diagnostic was dropped: the state reaches nothing measurable")
    d_min = min(magnitude_bound(r) for r in kept)
    eps_sub = d_min / 4
    eps_perp = 2 * eps_sub / 3
    eps_reuse = eps_sub
    eps_pred = eps_dyn = float(split_half_log_loss_gap)
    m = len(kept)
    n = required_n(m, eps_sub, alpha=alpha)
    return Table(
        retained=tuple(r.name for r in kept),
        dropped=tuple(dropped),
        m=m,
        d_min=d_min,
        epsilon_sub=eps_sub,
        epsilon_perp=eps_perp,
        epsilon_reuse=eps_reuse,
        epsilon_pred=eps_pred,
        epsilon_dyn=eps_dyn,
        n=n,
        alpha=alpha,
        derivation={
            "d_min": "min over retained diagnostics of the (1 - alpha) bootstrap bound on |contrast|",
            "epsilon_sub": "d_min / 4",
            "epsilon_perp": "2 * epsilon_sub / 3",
            "epsilon_reuse": "epsilon_sub",
            "epsilon_pred, epsilon_dyn": "split-half log-loss gap of the pilot's fitted belief update",
            "n": "ceil(ln(2M/alpha) / epsilon_sub^2)",
            "closed_form_checks": [
                {"M": m_, "epsilon": e_, "n": n_} for m_, e_, n_ in CLOSED_FORM_CHECKS
            ],
        },
    )


@dataclass(frozen=True)
class Budget:
    """The budget rule's output: both ``ε_sub`` values and the reason, written before the run."""

    hours_bought: float
    rate_per_hour: float
    n_at_derived: int
    hours_at_derived: float
    epsilon_sub_derived: float
    epsilon_sub_final: float
    n_final: int
    hours_final: float
    loosened: bool
    reason: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def budget_rule(table: Table, *, hours_bought: float, rate_per_hour: float) -> Budget:
    """Device time for the main run is ``(2n + 2n) / rate``: each context decoded once per arm and
    once each for the patch and its control. If over budget, loosen ``ε_sub`` — never M, never the
    control — to the largest n the hours allow, and say so."""
    if rate_per_hour <= 0:
        raise ValueError("the rate must be measured and positive; the laptop projects nothing here")
    hours_at = 4 * table.n / rate_per_hour
    if hours_at <= hours_bought:
        return Budget(hours_bought, rate_per_hour, table.n, hours_at, table.epsilon_sub,
                      table.epsilon_sub, table.n, hours_at, False, "within budget at the derived tolerance")
    n_allowed = math.floor(hours_bought * rate_per_hour / 4)
    if n_allowed < 1:
        raise ValueError(f"{hours_bought} h at {rate_per_hour}/h affords no episode; the run cannot start")
    eps_final = math.sqrt(math.log(2 * table.m / table.alpha) / n_allowed)
    return Budget(
        hours_bought, rate_per_hour, table.n, hours_at, table.epsilon_sub, eps_final, n_allowed,
        4 * n_allowed / rate_per_hour, True,
        f"loosened epsilon_sub from {table.epsilon_sub:.5f} to {eps_final:.5f} so that 4n fits "
        f"{hours_bought} h at {rate_per_hour:.2f} episodes/h; M and the control are unchanged",
    )


__all__ = [
    "ALPHA", "CLOSED_FORM_CHECKS", "RESAMPLES", "Budget", "Contrast", "Dropped", "Table",
    "bootstrap_bounds", "budget_rule", "contrasts", "derive", "drop_rule", "magnitude_bound",
    "paired_bootstrap_lower_bound",
    "required_n", "verify_closed_forms",
]
