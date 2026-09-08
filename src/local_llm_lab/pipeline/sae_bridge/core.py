"""Model-free A1 arithmetic: J[output,input], D[residual,feature], W[token,residual].

The only composed matrix is JD. Vocabulary scores are computed one feature at a
time; this module never constructs the vocabulary-by-feature transfer matrix.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class FeatureTopK:
    feature_id: int
    token_ids: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class ScoreComparison:
    passed: bool
    absolute_l2: float
    reference_l2: float
    relative_l2: float
    max_absolute_error: float
    rtol: float
    atol: float


def _matrix(value, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or 0 in raw.shape or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a nonempty real numeric matrix")
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.asarray(raw, dtype=np.float32)
    # Keep validation scratch bounded even for a full unembedding matrix.
    for start in range(0, len(matrix), 4096):
        if not np.isfinite(matrix[start : start + 4096]).all():
            raise ValueError(f"{name} must contain finite float32-compatible values")
    return matrix


def _integer(value, name: str, lower: int, upper: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} is out of range")
    return value


def _lens_decoder(j, d):
    j, d = _matrix(j, "J"), _matrix(d, "D")
    if j.shape[0] != j.shape[1] or j.shape[1] != d.shape[0]:
        raise ValueError("J must be square [residual,residual] and D [residual,feature]")
    return j, d


def compose_decoder(j, d) -> np.ndarray:
    """Return float32 J @ D with residual-by-feature orientation."""
    j, d = _lens_decoder(j, d)
    with np.errstate(over="ignore", invalid="ignore"):
        result = j @ d
    return _matrix(result, "JD")


def _readout(w, jd):
    w, jd = _matrix(w, "W"), _matrix(jd, "JD")
    if w.shape[1] != jd.shape[0]:
        raise ValueError("W must be [token,residual] matching JD [residual,feature]")
    return w, jd


def _scores(w, jd, feature_id):
    with np.errstate(over="ignore", invalid="ignore"):
        scores = w @ jd[:, feature_id]
    if not np.isfinite(scores).all():
        raise ValueError("feature scores are not finite in float32")
    return scores


def feature_scores(w, jd, feature_id: int) -> np.ndarray:
    """Return every vocabulary score for one named feature, in float32."""
    w, jd = _readout(w, jd)
    feature_id = _integer(feature_id, "feature_id", 0, jd.shape[1] - 1)
    return _scores(w, jd, feature_id)


def iter_feature_topk(
    w,
    jd,
    k: int,
    feature_ids: Iterable[int] | None = None,
) -> Iterator[FeatureTopK]:
    """Yield largest signed scores; equal scores prefer the lower token ID.

    Feature selection preserves the supplied order and rejects duplicates. Input
    validation happens on iteration; no features-by-vocabulary array is allocated.
    """
    w, jd = _readout(w, jd)
    k = _integer(k, "k", 1, w.shape[0])
    if feature_ids is None:
        selected = range(jd.shape[1])
    else:
        selected = [_integer(i, "feature_id", 0, jd.shape[1] - 1) for i in feature_ids]
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("feature_ids must be nonempty and contain no duplicates")
    for feature_id in selected:
        scores = _scores(w, jd, feature_id)
        cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
        above = np.flatnonzero(scores > cutoff)
        tied = np.flatnonzero(scores == cutoff)[: k - len(above)]
        ids = np.concatenate((above, tied))
        ids = ids[np.lexsort((ids, -scores[ids]))]
        row = FeatureTopK(feature_id, ids, scores[ids])
        del scores  # Release this column before the next vocabulary product is computed.
        yield row


def reference_feature_scores(w, j, d, feature_id: int, *, row_chunk_size=4096) -> np.ndarray:
    """Independent float64 reference: (W_row_chunk @ J) @ D[:,feature].

    Inputs are first canonicalised to float32, as in the composed path. The
    reference then reassociates the product in float64 without reusing JD.
    """
    j, d = _lens_decoder(j, d)
    w = _matrix(w, "W")
    if w.shape[1] != j.shape[0]:
        raise ValueError("W residual dimension must match J")
    feature_id = _integer(feature_id, "feature_id", 0, d.shape[1] - 1)
    row_chunk_size = _integer(row_chunk_size, "row_chunk_size", 1)
    j64 = j.astype(np.float64)
    column64 = d[:, feature_id].astype(np.float64)
    result = np.empty(w.shape[0], dtype=np.float64)
    for start in range(0, w.shape[0], row_chunk_size):
        stop = min(start + row_chunk_size, w.shape[0])
        result[start:stop] = (w[start:stop].astype(np.float64) @ j64) @ column64
    if not np.isfinite(result).all():
        raise ValueError("reference feature scores are not finite")
    return result


def compare_score_vectors(actual, reference, *, rtol=1e-5, atol=1e-5) -> ScoreComparison:
    """Compare complete vectors elementwise and report absolute error with its norm.

    Relative L2 error is zero for equal zero vectors, infinity for a nonzero
    error against a zero reference. Acceptance uses abs(error) <= atol +
    rtol * abs(reference) for every token; a matching top-k alone cannot pass.
    """
    vectors = []
    for value in (actual, reference):
        raw = np.asarray(value)
        if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "fiu":
            raise ValueError("scores must be nonempty real numeric vectors")
        vector = raw.astype(np.float64)
        if not np.isfinite(vector).all():
            raise ValueError("scores must be finite")
        vectors.append(vector)
    actual64, reference64 = vectors
    if actual64.shape != reference64.shape:
        raise ValueError("score vector shapes must match")
    for tolerance in (rtol, atol):
        if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerances must be finite nonnegative scalars")
    error = actual64 - reference64
    absolute = float(np.linalg.norm(error))
    norm = float(np.linalg.norm(reference64))
    relative = absolute / norm if norm else (0.0 if absolute == 0 else float("inf"))
    return ScoreComparison(
        passed=bool(np.all(np.abs(error) <= atol + rtol * np.abs(reference64))),
        absolute_l2=absolute,
        reference_l2=norm,
        relative_l2=relative,
        max_absolute_error=float(np.max(np.abs(error))),
        rtol=float(rtol),
        atol=float(atol),
    )
