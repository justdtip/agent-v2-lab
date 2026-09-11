"""Krylov diagnostics and the mandatory reference fallback, using synthetic NumPy inputs.

Diagnostic agreement is measured separately from certification. Even converged candidates must
remain uncertified until a global leadingness check exists, and the public direction uses full SVD.
"""

from __future__ import annotations

import numpy as np
import pytest

from local_llm_lab.pipeline.state_programme.transport_certified import (
    GAP_TOLERANCE,
    KRYLOV_SCHEDULE,
    RESIDUAL_TOLERANCE,
    leading_direction,
    leading_triplet,
)


def reference(cross: np.ndarray) -> tuple[np.ndarray, float]:
    """The sealed fitter's answer, with its sign convention."""
    u, values, _ = np.linalg.svd(cross, full_matrices=False)
    u = u[:, 0]
    return (-u if u[int(np.argmax(np.abs(u)))] < 0 else u), float(values[0])


def cross_product(rng, n: int, p: int, noise: float = 0.05) -> np.ndarray:
    x = rng.normal(size=(n, p))
    return x.T @ (x + noise * rng.normal(size=(n, p)))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_diagnostics_agree_numerically_on_the_fixtures(seed):
    cross = cross_product(np.random.default_rng(seed), 120, 60)
    u, sigma, _v, certificate = leading_triplet(cross)
    want_u, want_sigma = reference(cross)
    assert certificate["converged"] and not certificate["certified"], certificate
    assert abs(u @ want_u) > 1 - 1e-10
    assert abs(sigma - want_sigma) / want_sigma < 1e-10


def test_diagnostics_agree_numerically_at_the_production_width():
    """3,840 wide, the 12B's residual. The width is the point of the change, so it is tested."""
    cross = cross_product(np.random.default_rng(0), 400, 3840)
    u, sigma, _v, certificate = leading_triplet(cross)
    want_u, want_sigma = reference(cross)
    assert certificate["converged"] and not certificate["certified"], certificate
    assert abs(u @ want_u) > 1 - 1e-10
    assert abs(sigma - want_sigma) / want_sigma < 1e-10


def test_diagnostics_detect_a_direction_that_has_not_converged():
    """Codex F1's actual requirement. Twenty-four steps at this width give a wrong vector."""
    cross = cross_product(np.random.default_rng(0), 400, 1024)
    u, _sigma, _v, certificate = leading_triplet(cross, dimension=8)
    want_u, _ = reference(cross)
    assert not certificate["certified"] and not certificate["converged"]
    assert abs(u @ want_u) < 1 - 1e-6, "the fixture no longer separates converged from not"


def test_an_uncertified_triplet_falls_back_to_the_full_svd_and_says_so():
    cross = cross_product(np.random.default_rng(0), 400, 1024)
    u, sigma, certificate = leading_direction(cross)
    want_u, want_sigma = reference(cross)
    assert certificate["fell_back_to_full_svd"]
    assert not certificate["certified"]
    assert np.array_equal(u, want_u) and sigma == want_sigma


def test_the_thresholds_are_declared_not_assumed():
    assert RESIDUAL_TOLERANCE == 1e-10 and GAP_TOLERANCE == 1e-6
    cross = cross_product(np.random.default_rng(1), 120, 60)
    loose = leading_triplet(cross, residual_tolerance=1e-30, dimension=4)[3]
    assert not loose["certified"] and loose["residual_tolerance"] == 1e-30
    assert loose["reason"]


def test_the_schedule_grows_until_diagnostic_convergence():
    cross = cross_product(np.random.default_rng(0), 400, 1024)
    certificate = leading_triplet(cross)[3]
    assert certificate["converged"] and not certificate["certified"]
    assert certificate["krylov_steps"] in KRYLOV_SCHEDULE
    small = leading_triplet(cross, dimension=KRYLOV_SCHEDULE[0])[3]
    assert certificate["krylov_steps"] >= small["krylov_steps"]


def test_the_fitted_coefficients_and_the_gate_quantity_agree():
    """§4's claim: the rule a reading uses, not only the directions behind it."""
    from local_llm_lab.pipeline.state_programme.transport import fit, nearer_the_successor
    from local_llm_lab.pipeline.state_programme.transport_certified import certified_fit

    rng = np.random.default_rng(20260911)
    n, p, rank = 600, 512, 8
    x = rng.normal(size=(n, p)).astype(np.float32)
    y = x + 0.05 * rng.normal(size=(n, p)).astype(np.float32)
    held = rng.normal(size=(200, p)).astype(np.float32)
    held_target = held + 0.05 * rng.normal(size=(200, p)).astype(np.float32)

    sealed = fit(x, y, rank)
    certified, certificates = certified_fit(x, y, rank)
    assert all(c["fell_back_to_full_svd"] and not c["certified"] for c in certificates)
    a, b = sealed.coefficients(rank), certified.coefficients(rank)
    assert np.array_equal(a, b)

    def gate(fitted):
        moved = np.asarray(fitted.apply(held, rank=rank), dtype=np.float32)
        return float(nearer_the_successor(moved, held, held_target).mean())

    assert gate(sealed) == gate(certified), "the quantity §5's table prints must not move"


def test_it_is_deterministic():
    cross = cross_product(np.random.default_rng(2), 120, 60)
    a, b = leading_triplet(cross), leading_triplet(cross)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert a[3]["krylov_steps"] == b[3]["krylov_steps"]


def test_the_sign_convention_matches_the_sealed_fitter():
    cross = cross_product(np.random.default_rng(3), 120, 60)
    u = leading_triplet(cross)[0]
    assert u[int(np.argmax(np.abs(u)))] > 0


def test_a_null_operator_is_refused_rather_than_certified():
    certificate = leading_triplet(np.zeros((8, 8)))[3]
    assert not certificate["certified"]
