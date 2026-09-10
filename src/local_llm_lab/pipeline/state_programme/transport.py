"""E2's transport rule under Amendment 1: a rank-r supervised map, fitted on ordinary transitions.

PLS2, the classical NIPALS construction, which is E1's PLS1 when the target has one column. The
target here is the successor residual, so the map takes a residual to a residual and can be applied
`m` times for a transition of cost `m`.

No random number enters the fit. The leading direction of each component is found by power iteration
on the cross-product with a deterministic start, and the cross-product is **updated** by the rank-one
deflation terms rather than recomputed, which is what makes thirty-two components affordable at this
width.

The input standardisation is internal and the centring is added back, so the rule's output is in the
untransformed residual space and no normalisation of the rule enters the metric — the clause §4.2
asks a transport rule to answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POWER_ITERATIONS = 200
POWER_TOLERANCE = 1e-10
DEFLATION_FLOOR = 1e-12


@dataclass(frozen=True)
class Transport:
    """One fitted decomposition, to the top of the ladder, sliced for every rank."""

    mean: np.ndarray
    scale: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray  # (features, rank)
    loadings: np.ndarray  # (features, rank)
    target_loadings: np.ndarray  # (features, rank)
    iterations: tuple[int, ...]

    @property
    def max_rank(self) -> int:
        return int(self.weights.shape[1])

    def coefficients(self, rank: int) -> np.ndarray:
        if rank > self.max_rank:
            raise ValueError(f"rank {rank} exceeds the {self.max_rank} components computed")
        w, p, c = self.weights[:, :rank], self.loadings[:, :rank], self.target_loadings[:, :rank]
        return w @ np.linalg.solve(p.T @ w, c.T)

    def apply(self, x: np.ndarray, *, rank: int, times: int = 1) -> np.ndarray:
        """`times` applications of the one-step rule, which is how a cost-m transition is transported."""
        if times < 1:
            raise ValueError("a transition of cost below one is not a transition")
        beta = self.coefficients(rank)
        out = np.atleast_2d(x).astype(np.float64)
        for _ in range(times):
            out = self.target_mean + ((out - self.mean) / self.scale) @ beta
        return out.reshape(np.shape(x))


def _leading_direction(cross: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, int]:
    """The leading left singular vector of XᵀY, by power iteration on (XᵀY)(XᵀY)ᵀ. Deterministic."""
    v = start / np.linalg.norm(start)
    for iteration in range(1, POWER_ITERATIONS + 1):
        nxt = cross @ (cross.T @ v)
        norm = np.linalg.norm(nxt)
        if norm < DEFLATION_FLOOR:
            return v, iteration
        nxt /= norm
        if np.linalg.norm(nxt - v) < POWER_TOLERANCE:
            return nxt, iteration
        v = nxt
    return v, POWER_ITERATIONS


def fit(sources: np.ndarray, targets: np.ndarray, max_rank: int) -> Transport:
    """Fit on the fitting folds' ordinary transitions alone."""
    mean = sources.mean(axis=0)
    scale = sources.std(axis=0)
    scale[scale < 1e-8] = 1.0
    target_mean = targets.mean(axis=0)

    x = ((sources - mean) / scale).astype(np.float64)
    y = (targets - target_mean).astype(np.float64)
    cross = x.T @ y  # updated by rank-one terms below, never recomputed

    features = x.shape[1]
    weights = np.zeros((features, max_rank))
    loadings = np.zeros((features, max_rank))
    target_loadings = np.zeros((features, max_rank))
    iterations: list[int] = []

    ones = np.ones(y.shape[1])
    for component in range(max_rank):
        if np.linalg.norm(cross) < DEFLATION_FLOOR:
            break
        # The start is recomputed from the CURRENT cross-product each component. Restarting from
        # the previous component's direction stalls: after deflation that direction lies in the
        # part just removed, so the iteration can return a vector with no projection on the data.
        start = cross @ ones
        if np.linalg.norm(start) < DEFLATION_FLOOR:
            start = cross[:, 0].copy()
        w, used = _leading_direction(cross, start)
        t = x @ w
        tt = float(t @ t)
        if tt < DEFLATION_FLOOR:
            break
        p = (x.T @ t) / tt
        c = (y.T @ t) / tt
        weights[:, component], loadings[:, component], target_loadings[:, component] = w, p, c
        iterations.append(used)
        # Deflate X and Y, and carry the same deflation into the cross-product.
        xt = x.T @ t
        ty = y.T @ t
        cross = cross - np.outer(xt, c) - np.outer(p, ty) + tt * np.outer(p, c)
        x -= np.outer(t, p)
        y -= np.outer(t, c)
    kept = len(iterations)
    return Transport(mean, scale, target_mean, weights[:, :kept], loadings[:, :kept],
                     target_loadings[:, :kept], tuple(iterations))


def nearer_the_successor(transported: np.ndarray, source: np.ndarray, successor: np.ndarray) -> np.ndarray:
    """Amendment 1's gate, as a two-point comparison and not the retrieval score."""
    to_source = np.linalg.norm(transported - source, axis=-1)
    to_successor = np.linalg.norm(transported - successor, axis=-1)
    return to_successor < to_source


__all__ = ["Transport", "fit", "nearer_the_successor"]
