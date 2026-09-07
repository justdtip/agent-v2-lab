"""CPU reducers with explicit token coordinates, ties, and missing future observations."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


class FutureRanks:
    """A residual at t is scored against emitted tokens t+1, t+4 and t+8.

    Ranks are competition ranks (1 + number of strictly greater probabilities).
    reset() must be called between turns: harness/tool tokens are not generated futures.
    Future events never observed before reset are censored, not assigned a failure rank.
    """

    def __init__(self, *, horizons=(1, 4, 8), capacity=16):
        if not horizons or any(type(h) is not int or h <= 0 for h in horizons):
            raise ValueError("horizons must be positive integers")
        if capacity <= max(horizons):
            raise ValueError("capacity must exceed the greatest horizon")
        self.horizons = tuple(horizons)
        self.capacity = capacity
        self.buffer = OrderedDict()

    def reset(self):
        self.buffer.clear()

    def capture(self, position, probabilities):
        if self.buffer and position <= next(reversed(self.buffer)):
            raise ValueError("capture positions must increase within a turn")
        stored = {}
        for layer, values in probabilities.items():
            a = np.asarray(values)
            if (
                a.ndim != 1
                or not a.size
                or not np.isfinite(a).all()
                or (a < 0).any()
                or not np.isclose(a.sum(), 1.0, rtol=1e-5, atol=1e-7)
            ):
                raise ValueError("expected a finite normalized probability vector")
            stored[layer] = a.copy()
        self.buffer[position] = stored
        while len(self.buffer) > self.capacity:
            self.buffer.popitem(last=False)

    def observe(self, position, token_id):
        records = []
        for horizon in self.horizons:
            source = position - horizon
            for layer, p in self.buffer.get(source, {}).items():
                if not 0 <= token_id < len(p):
                    raise ValueError("token id outside vocabulary")
                records.append(
                    {
                        "position": source,
                        "layer": layer,
                        "horizon": horizon,
                        "token_id": int(token_id),
                        "rank": int(np.count_nonzero(p > p[token_id])) + 1,
                        "probability": float(p[token_id]),
                    }
                )
        return records


def transport_overlap(written_top, source_top, attention, *, source_positions, target_position):
    """Attention-weighted overlap with earlier sources only; self mass is reported apart.

    This is an observational overlap statistic, not a claim that softmax is linear or that
    a particular direction was causally transported. The full row is sufficient to audit it.
    """
    weights = np.asarray(attention, dtype=np.float64)
    if (
        len(source_top) != len(weights)
        or len(source_positions) != len(weights)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1.0, atol=1e-5)
    ):
        raise ValueError("expected one normalized attention row aligned to source positions")
    words = set(map(int, written_top))
    if not words or len(words) != len(written_top):
        raise ValueError("written top-k must be nonempty and unique")
    prior = [i for i, p in enumerate(source_positions) if p < target_position]
    overlap = [len(words.intersection(source_top[i])) / len(words) for i in prior]
    mass = float(weights[prior].sum())
    score = float(sum(weights[i] * o for i, o in zip(prior, overlap, strict=True)))
    return {
        "score": score,
        "past_mass": mass,
        "conditional_overlap": score / mass if mass > 0 else None,
        "source_positions": [source_positions[i] for i in prior],
    }
