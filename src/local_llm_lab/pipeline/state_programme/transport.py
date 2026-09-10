"""E2's transport rule under Amendment 1: a rank-r supervised map, fitted on ordinary transitions.

PLS2, the classical NIPALS construction, which is E1's PLS1 when the target has one column. The
target here is the successor residual, so the map takes a residual to a residual and can be applied
`m` times for a transition of cost `m`.

No random number enters the fit, and no start vector: the leading direction of each component is the
leading left singular vector of the current cross-product, from a full singular value decomposition
(LAPACK, deterministic), so it is the leading one by construction rather than a fixed point of an
iteration that might have started in the wrong place (Codex, WSA-TRANSPORT-AMENDMENT-REVIEW F1: a
power iteration from `Xᵀ(Y·1)` could divide zero by zero, or converge to a non-leading direction and
certify it). The cross-product is **updated** by the rank-one deflation terms rather than recomputed.
A non-finite cross-product or component is a refusal, never a silent NaN.

The input standardisation is internal and the centring is added back, so the rule's output is in the
untransformed residual space and no normalisation of the rule enters the metric — the clause §4.2
asks a transport rule to answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFLATION_FLOOR = 1e-12


@dataclass(frozen=True)
class Transport:
    """One fitted decomposition, to the top of the ladder, sliced for every rank."""

    mean: np.ndarray
    scale: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray  # (features, rank)
    loadings: np.ndarray  # (features, rank)
    target_loadings: np.ndarray  # (target features, rank)
    leading_singular_values: tuple[float, ...]  # of the deflated cross-product, per component

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
        return out.reshape(np.shape(x)[:-1] + (out.shape[-1],))  # the target width, on x's leading shape


def _leading_direction(cross: np.ndarray) -> tuple[np.ndarray, float]:
    """The leading left singular vector of the cross-product and its singular value, by a full SVD.

    Deterministic (LAPACK on the same bytes), no start vector, and the leading one by construction.
    The sign is fixed so that the entry of largest magnitude is positive, which makes two fits of
    one input agree bit for bit. A non-finite cross-product is refused.
    """
    if not np.isfinite(cross).all():
        raise ValueError("the cross-product is not finite; the fit refuses rather than carry NaNs")
    u, sigma, _ = np.linalg.svd(cross, full_matrices=False)
    w = u[:, 0]
    pivot = int(np.argmax(np.abs(w)))
    if w[pivot] < 0:
        w = -w
    return w, float(sigma[0])


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
    target_loadings = np.zeros((y.shape[1], max_rank))  # the target width, which need not equal the source width
    leading: list[float] = []

    for component in range(max_rank):
        w, sigma = _leading_direction(cross)
        if sigma < DEFLATION_FLOOR:
            break  # nothing left to explain: the rank is honestly short
        t = x @ w
        tt = float(t @ t)
        if tt < DEFLATION_FLOOR:
            break
        p = (x.T @ t) / tt
        c = (y.T @ t) / tt
        if not (np.isfinite(p).all() and np.isfinite(c).all()):
            raise ValueError(f"component {component + 1} is not finite; the fit refuses")
        weights[:, component], loadings[:, component], target_loadings[:, component] = w, p, c
        leading.append(sigma)
        # Deflate X and Y, and carry the same deflation into the cross-product.
        xt = x.T @ t
        ty = y.T @ t
        cross = cross - np.outer(xt, c) - np.outer(p, ty) + tt * np.outer(p, c)
        x -= np.outer(t, p)
        y -= np.outer(t, c)
    kept = len(leading)
    return Transport(mean, scale, target_mean, weights[:, :kept], loadings[:, :kept],
                     target_loadings[:, :kept], tuple(leading))


def nearer_the_successor(transported: np.ndarray, source: np.ndarray, successor: np.ndarray) -> np.ndarray:
    """Amendment 1's gate, as a two-point comparison and not the retrieval score."""
    to_source = np.linalg.norm(transported - source, axis=-1)
    to_successor = np.linalg.norm(transported - successor, axis=-1)
    return to_successor < to_source


__all__ = ["Transport", "fit", "nearer_the_successor"]
