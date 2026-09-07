"""Independent map agreement, content readability, and output diagnostics.

Director's 2026-09-07 ruling: only a failed Jacobian-response check requests refitting.
All thresholds are supplied by the caller before inspecting the measured responses.
"""

from __future__ import annotations

import numpy as np


def layer_verdict(map_check, concept_check, output_divergence):
    allowed = {"pass", "fail", "inconclusive"}
    if map_check not in allowed or concept_check not in allowed:
        raise ValueError("check outcome must be pass, fail or inconclusive")
    if output_divergence is not None and (
        not np.isfinite(output_divergence) or output_divergence < 0
    ):
        raise ValueError("output divergence must be finite and nonnegative, or missing")
    outcome = "inconclusive"
    if map_check == "pass":
        outcome = {"pass": "readable", "fail": "not readable", "inconclusive": "inconclusive"}[
            concept_check
        ]
    return {
        "outcome": outcome,
        "refit_required": map_check == "fail",
        "map_check": map_check,
        "concept_check": concept_check,
        "output_divergence": output_divergence,
    }


def response_agreement(measured, predicted, *, atol, rtol, stable):
    """Compare like-for-like derivative responses in the fitted final-residual space.

    The caller must use the fit's source/target averaging, not compare a single local
    derivative to a corpus-averaged map. Unstable finite differences are inconclusive.
    """
    if not np.isfinite([atol, rtol]).all() or atol < 0 or rtol < 0:
        raise ValueError("nonnegative finite thresholds are required")
    measured, predicted = np.asarray(measured), np.asarray(predicted)
    if measured.shape != predicted.shape or not measured.size:
        raise ValueError("derivative responses must have the same nonempty shape")
    if not stable or not np.isfinite(measured).all() or not np.isfinite(predicted).all():
        return {"outcome": "inconclusive", "max_error": None, "atol": atol, "rtol": rtol}
    error = np.abs(measured - predicted)
    passed = bool(np.all(error <= atol + rtol * np.abs(measured)))
    return {
        "outcome": "pass" if passed else "fail",
        "max_error": float(error.max()),
        "atol": atol,
        "rtol": rtol,
    }
