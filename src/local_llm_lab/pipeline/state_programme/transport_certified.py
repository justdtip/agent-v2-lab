"""DRAFT, unsealed and unrun: a certified leading triplet in place of a full SVD.

Amendment 1 §2 obtains each component's direction from `numpy.linalg.svd` of the deflated
cross-product — a complete O(p³) decomposition to extract one singular vector from a matrix whose
rank is at most n < p. Codex F1 required that the direction be **certified**, not that it be obtained
by decomposing everything; this module proposes the cheaper route with a certificate attached.

**It does not touch `transport.py`.** That file is what an addendum sealed and what Codex read. If
this draft is accepted, the routine moves there under a new addendum; until then the sealed fitter is
the only one any reader uses.

**What the certificate does and does not establish, stated plainly because it is the crux.** For a
candidate triplet `(u, σ, v)` of `A` the routine reports the relative residual
`max(‖Av − σu‖, ‖Aᵀu − σv‖) / σ`, which bounds the distance to *a* singular triplet of `A`
(Bauer–Fike for the symmetric dilation). It also reports the Ritz gap `(θ₁ − θ₂) / θ₁`. A small
residual with a clear gap is strong evidence the triplet is the **leading** one; it is **not a proof
of leadingness**, and this draft does not claim one. The guarantees actually relied on are:

1. `θ₁` is a Rayleigh quotient, so `σ₁ ≥ sqrt(θ₁)` always — the routine never overstates;
2. anything failing either declared threshold falls back to the full SVD, so the sealed result is
   reproduced exactly rather than approximated;
3. agreement with the full SVD is tested, on the fixtures and at the production width.

`scipy` is deliberately not used: it is absent from the card's environment and is not a declared
dependency of this repository, so a routine built on it could not run where the reading runs.
"""

from __future__ import annotations

import numpy as np

#: A triplet is accepted only below this relative residual. Declared, not assumed: it is three
#: orders of magnitude above float64's epsilon at these magnitudes, so it admits ordinary rounding
#: and refuses a direction that has not converged.
RESIDUAL_TOLERANCE = 1e-10
#: ...and only with this much separation from the next Ritz value, relative to the largest. Below
#: it the two are not distinguishable by the residual alone and the full SVD decides.
GAP_TOLERANCE = 1e-6
#: Krylov dimensions tried in order until the certificate passes. Declared as a schedule rather than
#: a single guess because the right size depends on the operator's spectral gap, which varies by
#: model width and by fold: at the 3,840-wide production size, 24 steps leave the direction wrong
#: (cosine 0.32 to the true leading vector) and the certificate correctly refuses; 96 certify.
#: Growing until certified is self-tuning and always ends either certified or in the full SVD.
KRYLOV_SCHEDULE = (48, 96, 192, 384)


def leading_triplet(cross: np.ndarray, *, residual_tolerance: float = RESIDUAL_TOLERANCE,
                    gap_tolerance: float = GAP_TOLERANCE, dimension: int | None = None,
                    schedule: tuple[int, ...] = KRYLOV_SCHEDULE
                    ) -> tuple[np.ndarray, float, np.ndarray, dict]:
    """Grow the Krylov space until the certificate passes, or hand back an uncertified triplet."""
    if dimension is not None:
        return _at_dimension(cross, residual_tolerance, gap_tolerance, dimension)
    result = None
    for size in schedule:
        result = _at_dimension(cross, residual_tolerance, gap_tolerance, size)
        if result[3]["certified"]:
            return result
        if size >= cross.shape[0]:
            break
    return result


def _at_dimension(cross: np.ndarray, residual_tolerance: float, gap_tolerance: float,
                  dimension: int) -> tuple[np.ndarray, float, np.ndarray, dict]:
    """The leading singular triplet of `cross`, with a certificate. Deterministic: no RNG.

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
        "certified": bool(residual <= residual_tolerance and gap >= gap_tolerance),
    }
    if not certificate["certified"]:
        certificate["reason"] = ("residual above tolerance" if residual > residual_tolerance
                                 else "Ritz gap below tolerance")
    # Sign convention identical to the sealed fitter's, so the two agree elementwise and not only
    # up to sign.
    pivot = int(np.argmax(np.abs(u)))
    if u[pivot] < 0:
        u, v = -u, -v
    return u, sigma, v, certificate


def leading_direction(cross: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """The sealed fitter's contract, served by the certified routine with a full-SVD fallback."""
    u, sigma, _v, certificate = leading_triplet(cross)
    if certificate["certified"]:
        return u, sigma, certificate
    reference, values, _ = np.linalg.svd(cross, full_matrices=False)
    u = reference[:, 0]
    pivot = int(np.argmax(np.abs(u)))
    if u[pivot] < 0:
        u = -u
    return u, float(values[0]), {**certificate, "fell_back_to_full_svd": True}


def certified_fit(sources: np.ndarray, targets: np.ndarray, max_rank: int):
    """`transport.fit`'s procedure with the certified direction in place of the full SVD.

    Identical in every other respect — the same standardisation, centring, deflation, cross-product
    update and stopping rules — so the two differ in exactly one step and can be compared directly.
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
    for _component in range(max_rank):
        if np.linalg.norm(cross) < DEFLATION_FLOOR:
            break
        w, sigma, certificate = leading_direction(cross)
        t = x @ w
        tt = float(t @ t)
        if tt < DEFLATION_FLOOR:
            break
        p_load = (x.T @ t) / tt
        c_load = (y.T @ t) / tt
        index = len(singular)
        weights[:, index], loadings[:, index], target_loadings[:, index] = w, p_load, c_load
        singular.append(float(sigma))
        certificates.append(certificate)
        cross = cross - tt * np.outer(p_load, c_load)
        x -= np.outer(t, p_load)
        y -= np.outer(t, c_load)
    kept = len(singular)
    fitted = Transport(mean, scale, target_mean, weights[:, :kept], loadings[:, :kept],
                       target_loadings[:, :kept], tuple(singular))
    return fitted, certificates


__all__ = ["GAP_TOLERANCE", "KRYLOV_SCHEDULE", "RESIDUAL_TOLERANCE",
           "certified_fit", "leading_direction", "leading_triplet"]
