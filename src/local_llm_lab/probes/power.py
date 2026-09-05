"""Exact power and detectable-effect arithmetic for WO-STAT-001.

Two recorded results are being read as nulls, and the question is whether they are nulls or
whether the samples were too small to have shown anything:

- P6's ``previous_notes`` flip rate, 4 of 5 against controls at 0 of 5;
- EXP-001's 4B decisive row, paired at matched-only 10 against mismatched-only 4, and its
  decomposition at 19 of 42.

Nothing here computes *observed* power -- power evaluated at the effect that was observed. That
quantity is a monotone function of the p-value, so it restates the p-value and licenses no new
sentence. Every function here takes a **hypothesised** effect and returns the power to detect
*that* effect, so the reader can name which effects the recorded n excludes and which it does
not, and at which n each becomes detectable.

Everything is enumerated exactly over the discrete distribution with :func:`math.comb` and
:func:`math.fsum`; nothing is a normal approximation and nothing samples. The module imports
:mod:`math` and nothing else -- no model, no network, no numpy -- because it must be runnable
and reviewable as pure arithmetic. :func:`wilson_interval` and :func:`sign_test_p` therefore
restate ``pipeline.evaluate.wilson`` and ``probes.jspace_sweep.sign_test`` rather than importing
them, since importing either would drag the architecture view and the model registry into a
module whose whole claim is that it touches neither; ``tests/test_power.py`` asserts the two
pairs agree so the restatement cannot drift.

**Three different objects, and a reader will assume they are one.** Each grid in the work order
enumerates a different distribution and the word "power" means a different thing in each:

- **A1, two-sample Fisher** (:func:`fisher_power`). Two independent binomial arms, n cases each,
  treatment successes ~ Bin(n, pi_T) and control successes ~ Bin(n, pi_C). Fisher's exact test
  conditions on the total number of successes and reads the hypergeometric tail, so the *test*
  is conditional; the *power* is unconditional, summing the joint probability of every (a, b)
  table whose conditional p-value clears alpha. This is the P6 arrangement: patched flips
  against control flips, independent arms, no pairing.
- **A2, exact McNemar / paired binomial** (:func:`mcnemar_power`). Each of n probe points is
  concordant or discordant, discordant with probability d; given D discordant pairs, the
  matched-only count is Bin(D, q) and the test is the exact two-sided binomial of that count
  against a fair coin. The test conditions on D -- that is what makes it McNemar -- but D is
  itself random when you are choosing n, so the power here sums over D ~ Bin(n, d) as well.
  Power conditional on a fixed D is available as :func:`mcnemar_power_given_discordant`, and
  the two differ by enough to matter -- neither is a substitute for the other, and which of the
  two is larger depends on n rather than being fixed by an inequality (see that function).
- **A3, unconditional sign test** (:func:`sign_test_power`). Every one of the n cases yields a
  comparison, there is no discordance to condition on, wins ~ Bin(n, r), and the test is the
  exact two-sided binomial against a fair coin. n is the whole sample, not a subset of it,
  which is why A3 reaches a given power at a much smaller n than A2 does.

**A4, Holm.** :func:`holm_alpha` returns the level a row must clear on its own inside a family
of m simultaneous tests. Holm rejects the smallest p in a family exactly when it clears
alpha / m, so alpha / m is the exact threshold for the most significant row and a lower bound on
the threshold for any other row -- meaning the power computed at it is exact for the row that
matters and conservative elsewhere. The recorded EXP-001 lens rows are sign tests, so A4 is A3
re-run at the Holm level. The decisive row sits outside every Holm family and is reported
uncorrected, so no Holm figure here describes it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_POWER_TARGET",
    "binomial_pmf",
    "expected_successes",
    "fisher_power",
    "fisher_p",
    "holm_alpha",
    "mcnemar_power",
    "mcnemar_power_given_discordant",
    "sign_test_p",
    "sign_test_power",
    "smallest_detectable_matched_share",
    "smallest_detectable_sign_rate",
    "smallest_n_for_fisher_power",
    "smallest_n_for_mcnemar_power",
    "smallest_n_for_sign_power",
    "smallest_n_for_wilson_half_width",
    "wilson_half_width",
    "wilson_interval",
]

DEFAULT_ALPHA = 0.05
DEFAULT_POWER_TARGET = 0.80

# Every p-value here is an exact rational -- a sum of integer binomial weights over an integer
# denominator -- evaluated in binary floating point, so the single division at the end is the
# only rounding. A p-value that is mathematically equal to alpha can therefore land a fraction
# of an ulp above it and read as a non-rejection, which would silently move a grid cell. The
# comparison is made at a relative slack far below any real difference in p and far above that
# rounding, so equality decides as the mathematics says it should rather than as the last bit
# happens to fall.
_P_VALUE_SLACK = 1e-12


def _rejects(p_value: float, alpha: float) -> bool:
    """Whether an exact p-value clears alpha, tolerant of a last-bit rounding at equality."""
    return p_value <= alpha * (1.0 + _P_VALUE_SLACK)


def binomial_pmf(successes: int, trials: int, rate: float) -> float:
    """Exact Bin(successes; trials, rate) via :func:`math.comb`."""
    if not 0 <= successes <= trials:
        return 0.0
    # The two boundary rates are written out because 0.0 ** 0 is 1.0 in Python but the general
    # expression would still evaluate the other factor, and at rate 0 or 1 that other factor is
    # 0 ** 0 too; spelling them out keeps a degenerate arm (P6's controls are hypothesised at
    # exactly 0.0) from depending on that coincidence.
    if rate <= 0.0:
        return 1.0 if successes == 0 else 0.0
    if rate >= 1.0:
        return 1.0 if successes == trials else 0.0
    return math.comb(trials, successes) * rate**successes * (1.0 - rate) ** (trials - successes)


def _binomial_row(trials: int, rate: float) -> list[float]:
    """The whole pmf for one arm, computed once so a power sum does not rebuild it per cell."""
    return [binomial_pmf(k, trials, rate) for k in range(trials + 1)]


# --------------------------------------------------------------------------- the sign test (A3)


def sign_test_p(wins: int, total: int) -> float:
    """Exact two-sided binomial p against a fair coin.

    The rule is the minimum-likelihood one -- the total probability of every outcome no likelier
    than the observed one -- which is what ``probes.jspace_sweep.sign_test`` uses and therefore
    what produced the recorded EXP-001 p-values. Under a fair coin it coincides with the doubled
    tail by symmetry, but the two part company under any other null, so the rule is stated rather
    than assumed.
    """
    if total == 0:
        return 1.0
    weights = [math.comb(total, i) for i in range(total + 1)]
    observed = weights[wins]
    return min(1.0, math.fsum(w for w in weights if w <= observed) / 2**total)


@cache
def _sign_test_rejection_set(total: int, alpha: float) -> frozenset[int]:
    """The win counts the two-sided sign test rejects at alpha, enumerated once per n.

    Cached because :func:`mcnemar_power` needs this for every discordant count from 0 to n on
    every candidate n, and the set depends on neither the hypothesised rate nor which grid cell
    is asking.

    Built by one sorted prefix sum rather than by calling :func:`sign_test_p` per win count:
    that function rebuilds the whole coefficient row each time, which makes the set quadratic in
    n and the paired sample-size search cubic. ``tests/test_power.py`` pins this against
    :func:`sign_test_p` so the fast path cannot drift from the readable one.
    """
    weights = [math.comb(total, i) for i in range(total + 1)]
    denominator = float(2**total)
    rejecting: set[int] = set()
    running = 0.0
    index = 0
    order = sorted(range(total + 1), key=lambda k: weights[k])
    while index <= total:
        # Counts sharing a coefficient share a p-value (the distribution is symmetric, so k and
        # n - k always tie); admitting them as a group keeps the sort order from deciding which
        # side of a symmetric pair rejects.
        group = [order[index]]
        while index + 1 <= total and weights[order[index + 1]] == weights[group[0]]:
            index += 1
            group.append(order[index])
        running += math.fsum(weights[k] for k in group)
        if _rejects(running / denominator, alpha):
            rejecting.update(group)
        else:
            # The mass only grows as likelier outcomes join, so nothing further can reject.
            break
        index += 1
    return frozenset(rejecting)


def sign_test_power(total: int, rate: float, alpha: float = DEFAULT_ALPHA) -> float:
    """Power of the exact two-sided sign test at a hypothesised win rate.

    Unconditional: every one of ``total`` cases contributes a comparison, wins ~ Bin(total,
    rate), and the power is the probability that the win count falls in the rejection set.
    """
    rejection = _sign_test_rejection_set(total, alpha)
    row = _binomial_row(total, rate)
    return math.fsum(row[k] for k in rejection)


# ------------------------------------------------------------- the paired / McNemar test (A2)


@cache
def mcnemar_power_given_discordant(
    discordant: int, matched_share: float, alpha: float = DEFAULT_ALPHA
) -> float:
    """Power of the exact paired test *conditional on* a fixed discordant count.

    This is the power of the test as actually performed, since McNemar conditions on the number
    of discordant pairs. It is not the power of a plan to collect n probe points, because that
    plan does not get to choose the discordant count -- see :func:`mcnemar_power`.

    Cached because :func:`mcnemar_power` sums this over every discordant count up to n, and the
    sample-size search then repeats that sum for every candidate n: without the cache the search
    is cubic in the largest n considered, and measured, it does not finish.
    """
    rejection = _sign_test_rejection_set(discordant, alpha)
    row = _binomial_row(discordant, matched_share)
    return math.fsum(row[k] for k in rejection)


def mcnemar_power(
    total: int,
    discordance: float,
    matched_share: float,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Power of the exact paired test over ``total`` probe points, averaging over discordance.

    Two stages of randomness, enumerated exactly: the discordant count D ~ Bin(total,
    discordance), then the matched-only count within it ~ Bin(D, matched_share). Averaging over
    D rather than fixing it at its expectation is what makes this the number to plan an n
    against, and the difference is not a rounding: at the observed d and q it runs *above* the
    power conditional on the expected D at n = 42 (0.271 against 0.190, because the exact test
    on fourteen discordant pairs is unusually lumpy and the luckier draws of D more than repay
    the unluckier ones) and *below* it from n = 84 upward. Neither figure substitutes for the
    other, so both are exposed.
    """
    # The conditional power is reused across every D, and D runs to `total`, so the per-D
    # rejection sets and pmf rows are built inside the helper once each rather than per term.
    outer = _binomial_row(total, discordance)
    return math.fsum(
        outer[d] * mcnemar_power_given_discordant(d, matched_share, alpha)
        for d in range(total + 1)
        # A zero-discordance draw can never reject (there is nothing to test), and its
        # conditional power is 0.0, so it is skipped rather than summed as a no-op.
        if outer[d] > 0.0 and d > 0
    )


# ------------------------------------------------------------------ Fisher's exact test (A1)


def _hypergeometric_weights(
    treatment_n: int, control_n: int, total_successes: int
) -> tuple[int, list[int]]:
    """Integer weights of the hypergeometric law for one fixed success total.

    Returns the smallest attainable treatment count and the weight of each count from it upward.
    Working in integers keeps the conditional distribution exact until the single division that
    turns it into a p-value.
    """
    low = max(0, total_successes - control_n)
    high = min(treatment_n, total_successes)
    weights = [
        math.comb(treatment_n, a) * math.comb(control_n, total_successes - a)
        for a in range(low, high + 1)
    ]
    return low, weights


def fisher_p(
    treatment_successes: int,
    treatment_n: int,
    control_successes: int,
    control_n: int,
    *,
    tail: str = "greater",
) -> float:
    """Exact Fisher p for one 2x2 table, conditional on both margins.

    ``tail="greater"`` is the one-sided alternative "treatment rate exceeds control rate" that
    the work order's A1 specifies: the upper hypergeometric tail from the observed treatment
    count. ``tail="two-sided"`` is the minimum-likelihood rule, provided because figures quoted
    elsewhere for this same contrast are two-sided and the two differ by roughly a factor of two
    on these tables -- a report that does not say which tail it used is not reproducible.
    """
    total_successes = treatment_successes + control_successes
    low, weights = _hypergeometric_weights(treatment_n, control_n, total_successes)
    denominator = math.fsum(weights)
    if denominator <= 0.0:
        # Both margins degenerate (no successes anywhere, or none possible): nothing to test.
        return 1.0
    index = treatment_successes - low
    if tail == "greater":
        return min(1.0, math.fsum(weights[index:]) / denominator)
    if tail == "two-sided":
        observed = weights[index]
        return min(1.0, math.fsum(w for w in weights if w <= observed) / denominator)
    raise ValueError("tail must be 'greater' or 'two-sided'")


@cache
def _fisher_rejection(
    treatment_n: int, control_n: int, alpha: float, tail: str
) -> tuple[frozenset[int], ...]:
    """For each success total, the treatment counts Fisher rejects at alpha.

    Built once per (n, alpha, tail) and cached, because the sample-size search re-enters this
    for every candidate n and the tails do not depend on the hypothesised rates -- only the
    weighting of the resulting table does.

    The tails are accumulated rather than recomputed per cell. Calling :func:`fisher_p` for each
    of the O(n^2) cells would rebuild that cell's whole hypergeometric row, making the search
    quartic in n and, measured, unusable past a few dozen cases; a suffix sum over a row built
    once is linear in the row. The binomial coefficient rows are likewise built once per arm
    instead of once per cell.
    """
    treatment_choose = [math.comb(treatment_n, i) for i in range(treatment_n + 1)]
    control_choose = [math.comb(control_n, i) for i in range(control_n + 1)]
    rejection: list[frozenset[int]] = []
    for total_successes in range(treatment_n + control_n + 1):
        low = max(0, total_successes - control_n)
        high = min(treatment_n, total_successes)
        weights = [treatment_choose[a] * control_choose[total_successes - a] for a in range(low, high + 1)]
        denominator = math.fsum(weights)
        if denominator <= 0.0:
            rejection.append(frozenset())
            continue
        if tail == "greater":
            # One suffix sum over the row gives every upper tail at once, largest a first.
            rejecting = set()
            running = 0.0
            for offset in range(len(weights) - 1, -1, -1):
                running += weights[offset]
                if _rejects(running / denominator, alpha):
                    rejecting.add(low + offset)
                else:
                    # The upper tail only grows as a falls, so once it fails it fails for good.
                    break
            rejection.append(frozenset(rejecting))
            continue
        if tail != "two-sided":
            raise ValueError("tail must be 'greater' or 'two-sided'")
        # Minimum-likelihood two-sided: a cell rejects when the mass of every outcome no likelier
        # than it clears alpha. Sorting the row once turns that into a single prefix sum.
        order = sorted(range(len(weights)), key=lambda offset: weights[offset])
        rejecting = set()
        running = 0.0
        index = 0
        while index < len(order):
            # Equal weights are one tie group: they share a p-value, so they must be admitted
            # together or the ordering of equal cells would decide which of them rejects.
            group = [order[index]]
            while index + 1 < len(order) and weights[order[index + 1]] == weights[group[0]]:
                index += 1
                group.append(order[index])
            running += math.fsum(weights[offset] for offset in group)
            if _rejects(running / denominator, alpha):
                rejecting.update(low + offset for offset in group)
            else:
                break
            index += 1
        rejection.append(frozenset(rejecting))
    return tuple(rejection)


def fisher_power(
    per_arm_n: int,
    treatment_rate: float,
    control_rate: float,
    alpha: float = DEFAULT_ALPHA,
    *,
    tail: str = "greater",
) -> float:
    """Unconditional exact power of Fisher's test with ``per_arm_n`` cases in each arm.

    The two arms are independent binomials at the hypothesised rates; the test is conditional on
    their total, so the power is the joint probability of the tables whose conditional p-value
    clears alpha. Summed over all (n + 1)^2 tables, not approximated.
    """
    rejection = _fisher_rejection(per_arm_n, per_arm_n, alpha, tail)
    treatment_row = _binomial_row(per_arm_n, treatment_rate)
    control_row = _binomial_row(per_arm_n, control_rate)
    return math.fsum(
        treatment_row[a] * control_row[b]
        for a in range(per_arm_n + 1)
        for b in range(per_arm_n + 1)
        if a in rejection[a + b]
    )


# ----------------------------------------------------------------- Wilson interval width (A1)


def wilson_interval(
    successes: int, trials: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval, clamped to [0, 1].

    Restates ``pipeline.evaluate.wilson``, including its default z, so the half-widths reported
    beside a power figure are the same interval the probe artifacts already record.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    if trials == 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    radius = (
        z
        / denominator
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    )
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def expected_successes(trials: int, rate: float) -> int:
    """The success count a hypothesised rate implies at this n, rounded half up.

    Python's :func:`round` is banker's rounding, which would send a rate of 0.5 at an even n to
    the wrong side and make an interval width depend on the parity of n. Half-up is stated here
    so the reported width at a fractional expectation (0.8 of 26 cases is 20.8) is reproducible.
    """
    return math.floor(trials * rate + 0.5)


def wilson_half_width(trials: int, rate: float, z: float = 1.96) -> float:
    """Half-width of the Wilson interval at the success count ``rate`` implies for this n.

    The interval is a property of an observed count, not of a rate, so a width quoted against a
    planned n is the width that would be seen if the hypothesised rate came out exactly.
    """
    low, high = wilson_interval(expected_successes(trials, rate), trials, z)
    return (high - low) / 2.0


# ------------------------------------------------------------------------- Holm's level (A4)


def holm_alpha(alpha: float, family_size: int) -> float:
    """The level one row must clear on its own inside a Holm family of ``family_size`` tests.

    Holm rejects the smallest p in the family exactly when it clears alpha / m, and rejects no
    row at all unless the smallest one does, so alpha / m is the exact threshold for the most
    significant row and a floor for every other. Power computed at this level is therefore exact
    for the row that would carry the family and conservative for the rest -- which is the number
    a reader wants, since the question A4 answers is what the corrected family could have
    detected at all.
    """
    if family_size < 1:
        raise ValueError("a Holm family contains at least one test")
    return alpha / family_size


# ---------------------------------------------------------- sample size and detectable effect


def _stable_crossing(predicate, candidates: Sequence[int]) -> int | None:
    """Smallest n in ``candidates`` from which ``predicate`` holds for every larger candidate.

    Exact discrete tests are not monotone in n: adding a case can move the critical value a whole
    step and cost power, so the first n to touch a target is not always an n that keeps it. The
    stable crossing is reported instead, because an n that meets the target only on a sawtooth
    tooth is not an n to collect to. Walking the candidates backwards makes this one pass -- the
    answer is the start of the unbroken run of successes that reaches the top of the range, so
    the walk stops at the first failure rather than continuing and letting the smaller n (which
    all fail) erase the run that was found.
    """
    holding: int | None = None
    for n in reversed(candidates):
        if not predicate(n):
            break
        holding = n
    return holding


def smallest_n_for_sign_power(
    rate: float,
    target: float = DEFAULT_POWER_TARGET,
    alpha: float = DEFAULT_ALPHA,
    *,
    max_n: int = 400,
) -> int | None:
    """Smallest n at which the sign test holds ``target`` power against ``rate``."""
    return _stable_crossing(
        lambda n: sign_test_power(n, rate, alpha) >= target, range(1, max_n + 1)
    )


def smallest_n_for_mcnemar_power(
    discordance: float,
    matched_share: float,
    target: float = DEFAULT_POWER_TARGET,
    alpha: float = DEFAULT_ALPHA,
    *,
    max_n: int = 600,
    step: int = 1,
) -> int | None:
    """Smallest number of probe points at which the paired test holds ``target`` power."""
    return _stable_crossing(
        lambda n: mcnemar_power(n, discordance, matched_share, alpha) >= target,
        range(step, max_n + 1, step),
    )


def smallest_n_for_fisher_power(
    treatment_rate: float,
    control_rate: float,
    target: float = DEFAULT_POWER_TARGET,
    alpha: float = DEFAULT_ALPHA,
    *,
    max_n: int = 200,
    tail: str = "greater",
) -> int | None:
    """Smallest per-arm n at which Fisher's test holds ``target`` power against the pair."""
    return _stable_crossing(
        lambda n: fisher_power(n, treatment_rate, control_rate, alpha, tail=tail) >= target,
        range(1, max_n + 1),
    )


def smallest_n_for_wilson_half_width(
    rate: float, limit: float, z: float = 1.96, *, max_n: int = 400
) -> int | None:
    """Smallest n whose Wilson half-width at ``rate`` stays under ``limit``.

    Stable rather than first-crossing for the same reason as the power searches: the half-width
    at a fractional expected count wobbles as the rounding of that count moves.
    """
    return _stable_crossing(
        lambda n: wilson_half_width(n, rate, z) < limit, range(1, max_n + 1)
    )


def _bisect_rate(power_of, target: float, tolerance: float) -> float | None:
    """Smallest rate above a fair coin reaching ``target`` power, by bisection.

    Power is a polynomial in the rate and increases with it above 0.5, so bisection converges on
    the unique crossing. The bracket is opened at the top first: if even a rate of 1.0 misses the
    target then no rate reaches it at this n, and returning None says so rather than returning
    the bracket's endpoint as though it were an answer.
    """
    low, high = 0.5, 1.0
    if power_of(high) < target:
        return None
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if power_of(middle) >= target:
            high = middle
        else:
            low = middle
    return high


def smallest_detectable_sign_rate(
    total: int,
    target: float = DEFAULT_POWER_TARGET,
    alpha: float = DEFAULT_ALPHA,
    *,
    tolerance: float = 1e-6,
) -> float | None:
    """Smallest true win rate the sign test detects at ``target`` power with ``total`` cases."""
    return _bisect_rate(lambda r: sign_test_power(total, r, alpha), target, tolerance)


def smallest_detectable_matched_share(
    total: int,
    discordance: float,
    target: float = DEFAULT_POWER_TARGET,
    alpha: float = DEFAULT_ALPHA,
    *,
    tolerance: float = 1e-6,
) -> float | None:
    """Smallest matched-only share the paired test detects at ``target`` power over ``total``."""
    return _bisect_rate(
        lambda q: mcnemar_power(total, discordance, q, alpha), target, tolerance
    )
