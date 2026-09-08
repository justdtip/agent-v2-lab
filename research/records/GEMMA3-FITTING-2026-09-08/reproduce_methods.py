"""Exact arithmetic illustrations of the estimand and state contracts; no model imports.

These are counterexamples to universal claims, not measurements on Qwen or Gemma.
Run with any Python 3: python reproduce_methods.py
"""

import json
from fractions import Fraction as F


def examples():
    # Value regression and averaged derivatives need not agree, even without ridge.
    x = tuple(map(F, [-2, -1, 1, 2]))
    value_slope = sum(t**4 for t in x) / sum(t**2 for t in x)
    average_derivative = sum(3 * t**2 for t in x) / len(x)
    assert value_slope == F(17, 5) and average_derivative == F(15, 2)

    # A causal cross-position route creates another difference, even when linear.
    pairs = [(F(a), F(b)) for a in (-1, 1) for b in (-1, 1)]
    slope = sum(a * a + b * (a + b) for a, b in pairs) / sum(a * a + b * b for a, b in pairs)
    future_jacobian = F(2 + 1, 2)  # source 1 reaches both targets; source 2 one.
    assert slope == 1 and future_jacobian == F(3, 2)

    # A recurrent computation is a defined function given its entry state/history.
    def recurrent(tokens, state=F(0)):
        out = []
        for token in tokens:
            state = state / 2 + token
            out.append(state)
        return out

    tokens = list(map(F, [1, 2, 3, 4]))
    baseline = recurrent(tokens)
    prefix_state = recurrent(tokens[:2])[-1]
    assert recurrent(tokens[2:], prefix_state) == baseline[2:]
    step = F(1, 100)
    plus, minus = tokens.copy(), tokens.copy()
    plus[2] += step
    minus[2] -= step
    whole = (recurrent(plus)[-1] - recurrent(minus)[-1]) / (2 * step)
    cached = (recurrent(plus[2:], prefix_state)[-1] - recurrent(minus[2:], prefix_state)[-1]) / (
        2 * step
    )
    assert whole == cached == F(1, 2)
    changed_prefix = tokens.copy()
    changed_prefix[0] += 1
    assert recurrent(changed_prefix)[-1] - baseline[-1] == F(1, 8)
    # Reusing the old prefix after changing its input computes the wrong function.
    stale = recurrent(changed_prefix[2:], prefix_state)[-1]
    assert stale == baseline[-1] and stale != recurrent(changed_prefix)[-1]
    return {
        "status": "exact_arithmetic_checks_passed",
        "scope": "analytical examples; no checkpoint or native tensor library",
        "nonlinear_value_vs_derivative": {
            "function": "y=x^3",
            "x": [-2, -1, 1, 2],
            "unregularized_value_regression": str(value_slope),
            "mean_derivative": str(average_derivative),
        },
        "causal_value_vs_future_derivative": {
            "function": "y1=x1; y2=x1+x2",
            "x": [[-1, -1], [-1, 1], [1, -1], [1, 1]],
            "same_position_value_regression": str(slope),
            "mean_source_sum_target_derivative": str(future_jacobian),
        },
        "recurrence": {
            "function": "s[t]=s[t-1]/2+x[t]; initial state 0",
            "whole_and_correct_cached_derivative": str(whole),
            "wrong_prefix_cache_misses_output_change": "1/8",
        },
    }


if __name__ == "__main__":
    print(json.dumps(examples(), indent=2))
