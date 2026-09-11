"""E1 as the pre-registration defines it: decodability at matched rank (§4.1, §5, §7).

This is a comparison of **encoding**, not of representational richness, and every figure it produces
carries its rank and its null because §5 refuses a figure without them.

The probe at rank `r` is fitted only on training rows: features standardised with the training mean
and scale, then `r` **supervised** components — the classical PLS1 construction, each component the
training direction most predictive of what the previous ones left — and a linear head on them. The
prediction is rounded to an integer and clipped to the range the *training* targets span, so the
probe can never name a value it was not shown.

The components are supervised on purpose. An unsupervised bottleneck would spend the rank on
whichever directions happen to carry variance, so `r` would measure "how much of this stream's
variance" rather than "how much capacity the probe is allowed", and the 12B's wider residual would
be penalised for its width instead of matched on capacity. §5 fixes the rank to equalise what the
probe may use; a supervised subspace is that quantity. Nothing consults an evaluation label: the
components come from training rows and training targets alone.

The decomposition is computed once to the top of the ladder and sliced for every rank, because PLS1
is greedy — the first `r` components of a 32-component fit *are* the `r`-component fit — so the
ladder is one decomposition rather than six unrelated ones. A null with different labels gets its
own decomposition, which is what "fitted the same way" requires.

Two nulls, fitted the same way at every rank on every model:

* the **permutation null**, whose training labels are permuted within family, so it measures what a
  probe of this capacity extracts from features carrying no label information;
* the **step-0 null**, whose features are the residual at the first decision of the same task, which
  carries what the task prompt alone tells — family, difficulty — and nothing of the progress.

The unit is the **episode**: each contributes one bounded observation, its own exact-match rate, and
the contrast against a null is paired on the same episodes, spanning [-1, 1]. That is the range the
tolerances in §7 are derived at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The ladder, identical on both models and on the controls (§5).
LADDER = (1, 2, 4, 8, 16, 32)
#: The registry's six depths, as fractions of the stack.
LAYER_FRACTIONS = (0.167, 0.333, 0.5, 0.667, 0.833, 1.0)
#: Fixed in advance so no layer and no rank is selected after the fact (§5).
HEADLINE_RANK = 8
HEADLINE_FRACTION = 0.5
#: Both read from the task, never from the model (§4.1).
TARGETS = ("step_index", "steps_remaining")


def layers_for(depth: int) -> dict[float, int]:
    """The six layer numbers for a stack of `depth`, one-based, as §5 names them."""
    return {fraction: max(1, min(depth, round(fraction * depth))) for fraction in LAYER_FRACTIONS}


@dataclass(frozen=True)
class Components:
    """The PLS1 decomposition of one (features, targets) training set, to the top of the ladder."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray  # (features, max_rank)
    loadings: np.ndarray  # (features, max_rank)
    coefficients: np.ndarray  # (max_rank,)
    target_mean: float
    low: int
    high: int

    @property
    def max_rank(self) -> int:
        return int(self.weights.shape[1])


@dataclass(frozen=True)
class Fit:
    """One fitted probe at one rank: a linear map and the range it may predict in."""

    mean: np.ndarray
    scale: np.ndarray
    beta: np.ndarray
    intercept: float
    rank: int
    low: int
    high: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        raw = ((features - self.mean) / self.scale) @ self.beta + self.intercept
        return np.clip(np.rint(raw), self.low, self.high).astype(np.int64)


def _standardise(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0  # a dead coordinate contributes nothing rather than infinity
    return mean, scale


def decompose(train: np.ndarray, targets: np.ndarray, max_rank: int) -> Components:
    """PLS1 on the training rows, to `max_rank` components, greedy and therefore nested."""
    mean, scale = _standardise(train)
    x = ((train - mean) / scale).astype(np.float64)
    y = targets.astype(np.float64)
    target_mean = float(y.mean())
    residual = y - target_mean

    rank = min(max_rank, *x.shape)
    weights = np.zeros((x.shape[1], rank))
    loadings = np.zeros((x.shape[1], rank))
    coefficients = np.zeros(rank)
    for component in range(rank):
        w = x.T @ residual
        norm = np.linalg.norm(w)
        if norm < 1e-12:  # nothing left that these features can explain
            weights, loadings = weights[:, :component], loadings[:, :component]
            coefficients = coefficients[:component]
            break
        w /= norm
        t = x @ w
        tt = float(t @ t)
        if tt < 1e-12:
            weights, loadings = weights[:, :component], loadings[:, :component]
            coefficients = coefficients[:component]
            break
        p = (x.T @ t) / tt
        q = float(residual @ t) / tt
        weights[:, component], loadings[:, component], coefficients[component] = w, p, q
        x = x - np.outer(t, p)
        residual = residual - q * t
    return Components(mean, scale, weights, loadings, coefficients,
                      target_mean, int(targets.min()), int(targets.max()))


def fit_at_rank(components: Components, rank: int) -> Fit:
    """The rank-`rank` probe, sliced from the decomposition rather than refitted."""
    if rank > components.max_rank:
        raise ValueError(f"rank {rank} exceeds the {components.max_rank} components computed")
    w = components.weights[:, :rank]
    p = components.loadings[:, :rank]
    q = components.coefficients[:rank]
    beta = w @ np.linalg.solve(p.T @ w, q)
    return Fit(components.mean, components.scale, beta, components.target_mean, rank,
               components.low, components.high)


def permute_within(labels: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The permutation null's labels: permuted within family, so family stays informative."""
    out = labels.copy()
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        out[rows] = labels[rng.permutation(rows)]
    return out


def episode_scores(predicted: np.ndarray, actual: np.ndarray, episodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One bounded observation per episode: its exact-match rate. Returns the episodes and the rates."""
    order = np.unique(episodes)
    hits = (predicted == actual).astype(np.float64)
    return order, np.array([hits[episodes == episode].mean() for episode in order])


def mean_absolute_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.abs(predicted - actual).mean())


def r_squared(predicted: np.ndarray, actual: np.ndarray) -> float:
    variance = float(((actual - actual.mean()) ** 2).sum())
    if variance == 0.0:
        return float("nan")
    return float(1.0 - ((actual - predicted) ** 2).sum() / variance)


@dataclass
class Reading:
    """One (target, layer, rank) cell: the probe, both nulls, and the paired contrasts."""

    target: str
    layer: int
    fraction: float
    rank: int
    n_episodes: int
    probe: float
    permutation_null: float
    step0_null: float
    over_permutation: float
    over_step0: float
    mae: float
    r2: float
    extras: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "target": self.target, "layer": self.layer, "fraction": self.fraction, "rank": self.rank,
            "n_episodes": self.n_episodes, "probe": self.probe,
            "permutation_null": self.permutation_null, "step0_null": self.step0_null,
            "over_permutation": self.over_permutation, "over_step0": self.over_step0,
            "mae": self.mae, "r2": self.r2, **self.extras,
        }


__all__ = [
    "Components", "Fit", "HEADLINE_FRACTION", "HEADLINE_RANK", "LADDER", "LAYER_FRACTIONS", "Reading", "TARGETS",
    "decompose", "episode_scores", "fit_at_rank", "layers_for", "mean_absolute_error",
    "permute_within", "r_squared",
]
