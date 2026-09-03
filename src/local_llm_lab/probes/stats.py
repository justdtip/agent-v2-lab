"""Numpy-only statistics shared by the probes.

Deliberately dependency-free (no scipy, no sklearn): the project pins mlx-lm and datasets and
nothing else, and every number a probe reports should be traceable to a few lines here.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "accuracy",
    "auc",
    "macro_f1",
    "mae",
    "mann_whitney_u",
    "r2",
    "ranks",
]


def ranks(values: np.ndarray) -> np.ndarray:
    """Ascending ranks starting at 1, with ties sharing their mean rank."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        result[order[start:stop]] = 0.5 * (start + stop + 1)
        start = stop
    return result


def mann_whitney_u(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Two-sided Mann-Whitney U with a tie-corrected normal approximation.

    Returns ``U`` for ``first`` (the number of (a, b) pairs with a > b, ties counting a half),
    the common-language effect size ``U / (n1 * n2)`` (0.5 = no separation), ``z`` and ``p``.
    The normal approximation is the usual one and is only trustworthy once both samples have
    roughly eight or more observations; ``n1``/``n2`` are reported so a reader can judge.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    n1, n2 = len(first), len(second)
    if n1 == 0 or n2 == 0:
        return {
            "u": float("nan"),
            "effect": float("nan"),
            "z": float("nan"),
            "p": float("nan"),
            "n1": float(n1),
            "n2": float(n2),
        }
    combined = np.concatenate([first, second])
    combined_ranks = ranks(combined)
    rank_sum = float(combined_ranks[:n1].sum())
    u_first = rank_sum - n1 * (n1 + 1) / 2
    mean = n1 * n2 / 2
    _, counts = np.unique(combined, return_counts=True)
    total = n1 + n2
    tie_term = float(np.sum(counts**3 - counts))
    variance = (n1 * n2 / 12) * ((total + 1) - tie_term / (total * (total - 1)))
    if variance <= 0:
        z = 0.0
    else:
        # Continuity correction, then the standard normal survival function via erfc.
        z = (u_first - mean - math.copysign(0.5, u_first - mean)) / math.sqrt(variance)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "u": float(u_first),
        "effect": float(u_first / (n1 * n2)),
        "z": float(z),
        "p": float(min(1.0, p)),
        "n1": float(n1),
        "n2": float(n2),
    }


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, as the rank statistic ``U / (n_pos * n_neg)``."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    positive = scores[labels == 1]
    negative = scores[labels != 1]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    return float(mann_whitney_u(positive, negative)["effect"])


def accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(actual) == 0:
        return float("nan")
    return float(np.mean(np.asarray(predicted) == np.asarray(actual)))


def macro_f1(predicted: np.ndarray, actual: np.ndarray, classes: np.ndarray) -> float:
    """Unweighted mean F1 over ``classes`` (a class absent from both counts as 0)."""
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    scores = []
    for label in classes:
        true_positive = float(np.sum((predicted == label) & (actual == label)))
        false_positive = float(np.sum((predicted == label) & (actual != label)))
        false_negative = float(np.sum((predicted != label) & (actual == label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores)) if scores else float("nan")


def r2(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Coefficient of determination against the held-out mean (can be negative)."""
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    total = float(np.sum((actual - actual.mean()) ** 2))
    if total <= 0:
        return float("nan")
    return float(1.0 - np.sum((actual - predicted) ** 2) / total)


def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64))
        )
    )
