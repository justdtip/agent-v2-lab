"""DRAFT, unsealed: Krylov convergence diagnostics with a mandatory reference fallback.

The former residual/Ritz-gap test can accept an exact non-leading triplet. Neither diagnostic
controls the spectrum outside the explored subspace, so leading_triplet always reports
certified=False. leading_direction uses the sealed full-SVD operation on every input. There is
currently no certified fast path and no supported speedup claim.

certified_fit retains its draft API but mirrors the sealed fitter's stopping, component-finiteness
checks and floating-point deflation order. The reference module and reader integration are unchanged.
Sign agreement and a small direction error do not establish bitwise or strict-retrieval equivalence
for an approximate solver. A future bypass requires a reviewed global bound and equivalence contract.
"""

from __future__ import annotations

import numpy as np

# These are convergence diagnostics only; passing them cannot establish leadingness.
RESIDUAL_TOLERANCE = 1e-10
GAP_TOLERANCE = 1e-6
KRYLOV_SCHEDULE = (48, 96, 192, 384)


def leading_triplet(cross: np.ndarray, *, residual_tolerance: float = RESIDUAL_TOLERANCE,
                    gap_tolerance: float = GAP_TOLERANCE, dimension: int | None = None,
                    schedule: tuple[int, ...] = KRYLOV_SCHEDULE
                    ) -> tuple[np.ndarray, float, np.ndarray, dict]:
    """Grow until diagnostic convergence; return a triplet that remains uncertified."""
    if not np.isfinite(cross).all():
        raise ValueError("the cross-product is not finite; the fit refuses rather than carry NaNs")
    if dimension is not None:
        return _at_dimension(cross, residual_tolerance, gap_tolerance, dimension)
    result = None
    for size in schedule:
        result = _at_dimension(cross, residual_tolerance, gap_tolerance, size)
        if result[3].get("converged", False):
            return result
        if size >= cross.shape[0]:
            break
    return result


def _at_dimension(cross: np.ndarray, residual_tolerance: float, gap_tolerance: float,
                  dimension: int) -> tuple[np.ndarray, float, np.ndarray, dict]:
    """A candidate singular triplet of `cross`, with convergence diagnostics. No RNG.

    Lanczos on the symmetric operator `S = A Aᵀ`, applied as two matrix–vector products so `S` is
    never formed. The start vector is `A` summed along its columns, which depends only on `A`.
    """
    p = cross.shape[0]
    dimension = int(min(dimension, p))
    start = cross.sum(axis=1)
    norm = np.linalg.norm(start)
    if norm < 1e-300:
        start = cross[:, 0].copy()
        norm = np.linalg.norm(start)
        if norm < 1e-300:
            return np.zeros(p), 0.0, np.zeros(cross.shape[1]), {"certified": False, "reason": "zero"}

    basis = np.empty((dimension, p))
    basis[0] = start / norm
    alpha, beta = [], []
    for step in range(dimension):
        w = cross @ (cross.T @ basis[step])
        a = float(basis[step] @ w)
        alpha.append(a)
        w -= a * basis[step]
        if step:
            w -= beta[-1] * basis[step - 1]
        w -= basis[: step + 1].T @ (basis[: step + 1] @ w)  # full reorthogonalisation
        b = float(np.linalg.norm(w))
        if step + 1 == dimension or b < 1e-300:
            used = step + 1
            break
        beta.append(b)
        basis[step + 1] = w / b
    else:  # pragma: no cover - the loop always breaks at `dimension`
        used = dimension

    tridiagonal = np.diag(alpha[:used])
    for i, b in enumerate(beta[: used - 1]):
        tridiagonal[i, i + 1] = tridiagonal[i + 1, i] = b
    values, vectors = np.linalg.eigh(tridiagonal)
    order = np.argsort(values)[::-1]
    theta = float(values[order[0]])
    second = float(values[order[1]]) if used > 1 else 0.0
    u = basis[:used].T @ vectors[:, order[0]]
    u /= np.linalg.norm(u)

    sigma = float(np.sqrt(max(theta, 0.0)))
    if sigma < 1e-300:
        return u, 0.0, np.zeros(cross.shape[1]), {"certified": False, "reason": "null operator"}
    v = cross.T @ u / sigma
    residual = max(float(np.linalg.norm(cross @ v - sigma * u)),
                   float(np.linalg.norm(cross.T @ u - sigma * v))) / sigma
    gap = (theta - second) / theta if theta > 0 else 0.0
    certificate = {
        "relative_residual": residual, "ritz_gap": gap, "krylov_steps": used,
        "residual_tolerance": residual_tolerance, "gap_tolerance": gap_tolerance,
        "converged": bool(residual <= residual_tolerance and gap >= gap_tolerance),
        "certified": False,
        "reason": "global leadingness is not established by residual and projected Ritz gap",
    }
    # Match the reference's sign convention; this alone does not imply numerical identity.
    pivot = int(np.argmax(np.abs(u)))
    if u[pivot] < 0:
        u, v = -u, -v
    return u, sigma, v, certificate


def leading_direction(cross: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Return the reference full-SVD direction; diagnostics never authorize a bypass."""
    from local_llm_lab.pipeline.state_programme.transport import _leading_direction

    try:
        _u, _sigma, _v, certificate = leading_triplet(cross)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        certificate = {"certified": False, "reason": f"diagnostics unavailable: {exc}"}
    u, sigma = _leading_direction(cross)
    return u, sigma, {**certificate, "fell_back_to_full_svd": True}


def certified_fit(sources: np.ndarray, targets: np.ndarray, max_rank: int):
    """Mirror transport.fit, with reference fallback and diagnostics for each retained component.

    Keep the reference's arithmetic order, stop tests and refusals. Do not simplify the deflation:
    an algebraically equal expression can change floating-point results at strict retrieval ties.
    """
    from local_llm_lab.pipeline.state_programme.transport import DEFLATION_FLOOR, Transport

    mean = sources.mean(axis=0)
    scale = sources.std(axis=0)
    scale[scale < 1e-8] = 1.0
    target_mean = targets.mean(axis=0)
    x = ((sources - mean) / scale).astype(np.float64)
    y = (targets - target_mean).astype(np.float64)
    cross = x.T @ y

    features, width = x.shape[1], y.shape[1]
    weights = np.zeros((features, max_rank))
    loadings = np.zeros((features, max_rank))
    target_loadings = np.zeros((width, max_rank))
    singular, certificates = [], []
    for component in range(max_rank):
        w, sigma, certificate = leading_direction(cross)
        if sigma < DEFLATION_FLOOR:
            break
        t = x @ w
        tt = float(t @ t)
        if tt < DEFLATION_FLOOR:
            break
        p_load = (x.T @ t) / tt
        c_load = (y.T @ t) / tt
        if not (np.isfinite(p_load).all() and np.isfinite(c_load).all()):
            raise ValueError(f"component {component + 1} is not finite; the fit refuses")
        index = len(singular)
        weights[:, index], loadings[:, index], target_loadings[:, index] = w, p_load, c_load
        singular.append(float(sigma))
        certificates.append(certificate)
        xt = x.T @ t
        ty = y.T @ t
        cross = (cross - np.outer(xt, c_load) - np.outer(p_load, ty)
                 + tt * np.outer(p_load, c_load))
        x -= np.outer(t, p_load)
        y -= np.outer(t, c_load)
    kept = len(singular)
    fitted = Transport(mean, scale, target_mean, weights[:, :kept], loadings[:, :kept],
                       target_loadings[:, :kept], tuple(singular))
    return fitted, certificates


__all__ = ["GAP_TOLERANCE", "KRYLOV_SCHEDULE", "RESIDUAL_TOLERANCE",
           "certified_fit", "leading_direction", "leading_triplet"]
