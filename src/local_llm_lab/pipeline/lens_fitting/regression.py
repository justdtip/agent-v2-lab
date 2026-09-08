"""All-layer ridge calibration from float32 sufficient sums, never activation datasets.

Requirements §§3.2/12 fix the *total* penalty at alpha * mean_diag(XTX).
The row solve is X W = Y; artifacts carry J = W.T for the existing lens reader.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

ALPHA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


@dataclass
class SufficientStats:
    xtx: Any
    xty: Any
    yty: Any
    n: int = 0

    @classmethod
    def zeros(cls, dimension: int) -> SufficientStats:
        import mlx.core as mx

        return cls(
            mx.zeros((dimension, dimension), dtype=mx.float32),
            mx.zeros((dimension, dimension), dtype=mx.float32),
            mx.zeros((), dtype=mx.float32),
        )

    def add(self, x: Any, y: Any) -> None:
        """Enqueue one sequence; the streaming caller evaluates all layers together."""
        import mlx.core as mx

        if x.ndim != 2 or x.shape != y.shape or x.shape[1] != self.xtx.shape[0]:
            raise ValueError("residual pairs must have matching (positions, hidden_size) shape")
        x, y = x.astype(mx.float32), y.astype(mx.float32)
        self.xtx = self.xtx + x.T @ x
        self.xty = self.xty + x.T @ y
        self.yty = self.yty + mx.sum(y * y)
        self.n += x.shape[0]


#: The two producers of the residual contract, named so a fit records which one it used.
RESIDUAL_SOURCES = ("hand_run", "native")


def validated_score_positions(row: dict) -> list[int] | None:
    """Validate an optional selection without changing the context sent to the forward."""
    if "score_positions" not in row:
        return None
    positions = row["score_positions"]
    if (
        not isinstance(positions, list)
        or not positions
        or any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, Integral)
            or i < 0
            or i >= len(row["ids"])
            for i in positions
        )
        or any(left >= right for left, right in zip(positions, positions[1:], strict=False))
    ):
        raise ValueError("score_positions must be nonempty sorted unique integer indices in bounds")
    return [int(i) for i in positions]


def accumulate(
    view: Any,
    rows: Iterable[dict],
    *,
    progress: Callable | None = None,
    residual_source: str = "hand_run",
) -> tuple:
    """One uncached all-layer forward per sequence, separate fit and selection sums.

    ``residual_source`` chooses the producer and is **explicit with no inference**, because a fit
    whose residual source changed silently would be unattributable afterwards: the numbers are
    the same shape either way and nothing in the artifact would say which forward made them. The
    caller passes it and the caller records it.

    ``"hand_run"`` is ``view.residuals``, this repository's own decoder loop, which is what every
    Qwen fit used. ``"native"`` is ``view.native_residuals``, the model's own forward tapped
    through capture, which inherits the entry transform and the mask construction rather than
    reproducing them. On a model whose loop is correct the two agree exactly; on Gemma 3 they do
    not until the architecture-view port lands, and ``view.residual_source_agreement`` is how a
    caller checks which case it is in.
    """
    import mlx.core as mx

    if residual_source not in RESIDUAL_SOURCES:
        raise ValueError(
            f"residual_source must be one of {list(RESIDUAL_SOURCES)}; got {residual_source!r}"
        )
    produce = view.residuals if residual_source == "hand_run" else view.native_residuals

    d, depth = view.hidden_size, view.num_layers
    if d <= 0 or depth < 2:
        raise ValueError("regression requires positive hidden size and nonfinal layers")
    layers = tuple(range(1, depth + 1))
    sums = {
        split: {layer: SufficientStats.zeros(d) for layer in layers[:-1]}
        for split in ("fit", "held")
    }
    counts = {
        split: {"sequences": 0, "positions": 0, "scored_positions": 0, "input_positions": 0}
        for split in sums
    }
    # Carried in the counts so it reaches the artifact by the same route the sequence and
    # position totals do, rather than depending on a caller remembering to stamp it.
    counts["residual_source"] = residual_source
    for number, row in enumerate(rows, 1):
        split, ids = row["split"], row["ids"]
        if split not in sums or not ids:
            raise ValueError("every sequence needs nonempty ids and fit/held membership")
        positions = validated_score_positions(row)
        residuals = produce(ids, layers)
        if set(residuals) != set(layers) or any(
            h.shape != (1, len(ids), d) for h in residuals.values()
        ):
            raise ValueError(
                "forward must return every requested (1, positions, hidden_size) residual"
            )
        selection = slice(None) if positions is None else mx.array(positions)
        target = residuals[depth][0][selection].astype(mx.float32)
        for layer in layers[:-1]:
            sums[split][layer].add(residuals[layer][0][selection], target)
        # Materialize before dropping the forward so lazy graphs cannot retain the corpus.
        mx.eval(*[a for s in sums[split].values() for a in (s.xtx, s.xty, s.yty)])
        del residuals, target
        counts[split]["sequences"] += 1
        scored = len(ids) if positions is None else len(positions)
        counts[split]["positions"] += scored
        counts[split]["scored_positions"] += scored
        counts[split]["input_positions"] += len(ids)
        if progress:
            progress(
                {
                    "event": "sequence",
                    "sequence": number,
                    "counts": {
                        key: dict(value) if isinstance(value, dict) else value
                        for key, value in counts.items()
                    },
                }
            )
    if any(not counts[split]["positions"] for split in sums):
        raise ValueError("both fit and held selection splits require positions")
    return sums, counts


def squared_error(sums: SufficientStats, w: Any) -> float:
    """Unnormalized SSE from sums; preserve float32 cancellation in the reported value."""
    import mlx.core as mx

    return float((sums.yty - 2 * mx.sum(w * sums.xty) + mx.sum((sums.xtx @ w) * w)).item())


def solve_layer(fit: SufficientStats, held: SufficientStats) -> tuple[np.ndarray, dict]:
    """Evaluate every fixed alpha, choosing by held relative SSE (first wins ties)."""
    import mlx.core as mx

    if fit.n <= 0 or held.n <= 0:
        raise ValueError("both sufficient-statistics splits require positions")
    if not all(
        bool(mx.all(mx.isfinite(a)).item()) for s in (fit, held) for a in (s.xtx, s.xty, s.yty)
    ):
        raise ValueError("nonfinite sufficient statistics")
    dbar = float(mx.mean(mx.diag(fit.xtx)).item())
    energy = float(held.yty.item())
    if dbar <= 0 or energy <= 0:
        raise ValueError("ridge scaling and relative error require positive residual energy")
    eye = mx.eye(fit.xtx.shape[0], dtype=mx.float32)
    candidates, best_w, best = [], None, None
    for alpha in ALPHA_GRID:
        penalty = alpha * dbar
        # MLX's solve is CPU-only. Accumulators remain float32 device arrays.
        w = mx.linalg.solve(fit.xtx + penalty * eye, fit.xty, stream=mx.cpu)
        mx.eval(w)
        if not bool(mx.all(mx.isfinite(w)).item()):
            raise ValueError(f"nonfinite ridge solution for alpha={alpha}; fixed grid not reduced")
        error = squared_error(held, w)
        relative = error / energy
        if not math.isfinite(relative):
            raise ValueError("nonfinite held reconstruction error")
        row = {
            "alpha": alpha,
            "absolute_penalty": penalty,
            "lambda_per_position": penalty / fit.n,
            "held_out_squared_error": error,
            "held_out_relative_error": relative,
            "held_out_uncentered_r2": 1 - relative,
        }
        candidates.append(row)
        if best is None or relative < best["held_out_relative_error"]:
            best, best_w = row, w
    return np.array(best_w, dtype=np.float32), {
        **best,
        "dbar": dbar,
        "n": fit.n,
        "fit_positions": fit.n,
        "held_positions": held.n,
        "held_target_squared_norm": energy,
        "candidates": candidates,
        "error_arithmetic": "float32 sufficient sums; small negative SSE can reflect cancellation",
    }


@dataclass(frozen=True)
class RegressionResult:
    maps: dict[int, np.ndarray]
    per_layer: dict[str, dict]
    counts: dict[str, Any]
    elapsed_s: float
    peak_memory_gib: float


def fit_regression(
    view: Any,
    rows: Iterable[dict],
    *,
    progress: Callable | None = None,
    residual_source: str = "hand_run",
) -> RegressionResult:
    """Fit every nonfinal residual index and return hosted-orientation float32 maps.

    ``residual_source`` is passed through and reaches the artifact in ``counts``, so a fitted
    lens says which forward produced the residuals it was fitted on.
    """
    import mlx.core as mx

    started = time.monotonic()
    sums, counts = accumulate(view, rows, progress=progress, residual_source=residual_source)
    maps, per_layer = {}, {}
    for layer in range(1, view.num_layers):
        layer_started = time.monotonic()
        w, info = solve_layer(sums["fit"].pop(layer), sums["held"].pop(layer))
        maps[layer] = w.T.copy()
        per_layer[str(layer)] = info | {"elapsed_s": time.monotonic() - layer_started}
        if progress:
            progress({"event": "layer", "layer": layer, **per_layer[str(layer)]})
    return RegressionResult(
        maps, per_layer, counts, time.monotonic() - started, mx.get_peak_memory() / 2**30
    )
