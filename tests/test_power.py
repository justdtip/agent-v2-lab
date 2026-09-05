"""Every figure in ``under_review/POWER-ANALYSIS-2026-09-05.md`` reproduces here.

The report's grids are pinned cell by cell, because the acceptance criterion for WO-STAT-001
Part A is that the markdown carries no number the code does not produce.

The *observed* statistics the grids are built around (P6's 4 of 5, EXP-001's 10 against 4 and
19 of 42) enter as literal inputs rather than being read out of ``outputs/``: those run
artifacts are untracked, so a test that parsed them would pass only on the machine that holds
them. :func:`test_recorded_statistics_match_their_artifacts` closes that gap where it can --
it reads the artifacts when they are present and skips when they are not -- and the report
carries the path and field of every figure for the reader who does have them.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from local_llm_lab.probes.power import (
    DEFAULT_ALPHA,
    _fisher_rejection,
    _sign_test_rejection_set,
    binomial_pmf,
    expected_successes,
    fisher_p,
    fisher_power,
    holm_alpha,
    mcnemar_power,
    mcnemar_power_given_discordant,
    sign_test_p,
    sign_test_power,
    smallest_detectable_matched_share,
    smallest_detectable_sign_rate,
    smallest_n_for_fisher_power,
    smallest_n_for_mcnemar_power,
    smallest_n_for_sign_power,
    smallest_n_for_wilson_half_width,
    wilson_half_width,
    wilson_interval,
)

# The report's own grids, named once so a table and its test cannot drift apart.
A1_CASES = (5, 10, 15, 20, 30, 40)
A2_POINTS = (42, 84, 126, 168, 252)
A3_POINTS = (42, 84, 168)

# EXP-001's observed paired configuration: 14 discordant pairs out of 42 probe points, of which
# 10 favour the matched context. Written as ratios so the provenance stays legible.
OBSERVED_DISCORDANCE = 14 / 42
OBSERVED_MATCHED_SHARE = 10 / 14


def _close(value: float, expected: float, tolerance: float = 5e-5) -> bool:
    return abs(value - expected) < tolerance


# ------------------------------------------------------------------------ machinery sanity


def test_sign_test_at_a_fair_coin_returns_the_attained_size() -> None:
    """A test's power against the null it tests *is* its size, and must not exceed alpha.

    The discrete sign test cannot spend alpha exactly -- the binomial has no outcome at the
    boundary -- so each figure sits below 0.05 rather than at it. A value above alpha would mean
    the rejection set had been built wrong.
    """
    attained = {n: sign_test_power(n, 0.5) for n in A3_POINTS}
    for n, size in attained.items():
        assert size <= DEFAULT_ALPHA, f"n={n} spends more than alpha under the null"
    assert _close(attained[42], 0.043559)
    assert _close(attained[84], 0.037530)
    assert _close(attained[168], 0.036927)


def test_paired_test_at_a_fair_share_returns_the_attained_size() -> None:
    """Same check for A2's object, which is conservative again through the discordance draw."""
    for n in A2_POINTS:
        assert mcnemar_power(n, OBSERVED_DISCORDANCE, 0.5) <= DEFAULT_ALPHA
    assert _close(mcnemar_power(42, OBSERVED_DISCORDANCE, 0.5), 0.026523)


def test_fisher_power_rises_with_the_effect() -> None:
    """Monotone in the treatment rate at fixed n -- a bigger gap is never harder to detect."""
    powers = [fisher_power(20, rate / 100, 0.0) for rate in range(50, 101)]
    assert powers == sorted(powers)


def test_fisher_power_is_not_monotone_in_n() -> None:
    """The work order's sanity check asserts monotonicity in n; exact discrete tests break it.

    At three cases per arm the most extreme table (3 of 3 against 0 of 3) has p exactly 0.05 and
    rejects, so power is 0.8 ** 3. At four, that table's p falls to 0.0143 but the next one in
    (3 of 4 against 0 of 4) is 0.0714 and cannot reject, so only the perfect table rejects and
    power falls to 0.8 ** 4. This is pinned rather than tolerated: it is why every sample-size
    search in the module reports a *stable* crossing instead of a first crossing.
    """
    assert _close(fisher_p(3, 3, 0, 3), 0.05)
    assert _close(fisher_p(4, 4, 0, 4), 0.014286)
    assert _close(fisher_p(3, 4, 0, 4), 0.071429)
    assert fisher_power(4, 0.8, 0.0) < fisher_power(3, 0.8, 0.0)
    assert _close(fisher_power(3, 0.8, 0.0), 0.8**3)
    assert _close(fisher_power(4, 0.8, 0.0), 0.8**4)


def test_binomial_pmf_handles_the_degenerate_control_arm() -> None:
    """P6's controls are hypothesised at exactly zero, so the boundary rates must be exact."""
    assert binomial_pmf(0, 5, 0.0) == 1.0
    assert binomial_pmf(1, 5, 0.0) == 0.0
    assert binomial_pmf(5, 5, 1.0) == 1.0
    assert _close(sum(binomial_pmf(k, 7, 0.3) for k in range(8)), 1.0)


def test_expected_successes_rounds_half_up() -> None:
    """Banker's rounding would make a reported interval width depend on the parity of n."""
    assert expected_successes(26, 0.8) == 21
    assert expected_successes(5, 0.8) == 4
    assert expected_successes(2, 0.5) == 1


# ------------------------------------------------------- the fast paths match the plain ones


def test_sign_rejection_set_matches_the_plain_p_value() -> None:
    """The cached prefix-sum rejection set must equal one built from :func:`sign_test_p`."""
    for n in range(0, 60):
        plain = frozenset(k for k in range(n + 1) if sign_test_p(k, n) <= DEFAULT_ALPHA)
        assert _sign_test_rejection_set(n, DEFAULT_ALPHA) == plain, f"n={n}"


def test_fisher_rejection_table_matches_the_plain_p_value() -> None:
    """Same for both Fisher tails, whose tables are built by accumulation rather than per cell."""
    for n in range(1, 22):
        for tail in ("greater", "two-sided"):
            table = _fisher_rejection(n, n, DEFAULT_ALPHA, tail)
            for total in range(2 * n + 1):
                low, high = max(0, total - n), min(n, total)
                plain = frozenset(
                    a
                    for a in range(low, high + 1)
                    if fisher_p(a, n, total - a, n, tail=tail) <= DEFAULT_ALPHA
                )
                assert table[total] == plain, f"n={n} tail={tail} total={total}"


def test_restatements_match_the_repository_functions() -> None:
    """Wilson and the sign test are restated locally to keep this module import-free.

    If either drifts from the function that produced the recorded artifacts, the power figures
    stop describing the results they are attached to, so the agreement is asserted rather than
    assumed.
    """
    from local_llm_lab.pipeline.evaluate import wilson
    from local_llm_lab.probes.jspace_sweep import sign_test

    for successes, trials in ((4, 5), (0, 5), (19, 42), (30, 42), (21, 26)):
        assert wilson_interval(successes, trials) == wilson(successes, trials)
    for wins, total in ((4, 5), (10, 14), (19, 42), (24, 42), (10, 42)):
        assert sign_test_p(wins, total) == sign_test(wins, total)


# ------------------------------------------------------------------ the recorded statistics


def test_recorded_p6_statistics() -> None:
    """P6's `previous_notes` row as the report states it."""
    low, high = wilson_interval(4, 5)
    assert _close(low, 0.375528) and _close(high, 0.963777)
    assert _close(wilson_half_width(5, 0.8), 0.2941)
    assert wilson_interval(0, 5) == (0.0, pytest.approx(0.434491, abs=5e-5))
    assert _close(fisher_p(4, 5, 0, 5), 0.023810)
    assert _close(fisher_p(4, 5, 0, 5, tail="two-sided"), 0.047619)


def test_recorded_exp001_statistics() -> None:
    """EXP-001's decisive paired row, its decomposition, and the cross-model pair."""
    assert _close(sign_test_p(10, 14), 0.179565)
    assert _close(sign_test_p(19, 42), 0.643969)
    low, high = wilson_interval(19, 42)
    assert _close(low, 0.312232) and _close(high, 0.600511)
    assert _close(sign_test_p(10, 42), 0.000941)
    assert _close(sign_test_p(24, 42), 0.440799)


def test_p6_at_five_cases_separates_only_at_four_flips() -> None:
    """Which flip counts at n = 5 are distinguishable from controls at zero: only 4 and 5.

    The whole row is pinned because the report prints it whole: everything at or below 3 of 5 is
    indistinguishable from controls at zero, which is the bound on what the recorded result may
    claim.
    """
    rejection = _fisher_rejection(5, 5, DEFAULT_ALPHA, "greater")
    rejects = {k for k in range(6) if k in rejection[k]}
    assert rejects == {4, 5}
    row = (1.0, 0.5, 0.222222, 0.083333, 0.023810, 0.003968)
    for flips, want in enumerate(row):
        assert _close(fisher_p(flips, 5, 0, 5), want), f"{flips}/5"


# ------------------------------------------------------------------------------- A1 tables


@pytest.mark.parametrize(
    ("treatment", "control", "expected", "n_for_power"),
    [
        (0.8, 0.0, (0.7373, 0.9991, 1.0000, 1.0000, 1.0000, 1.0000), 6),
        (0.6, 0.0, (0.3370, 0.9452, 0.9981, 0.9997, 1.0000, 1.0000), 8),
        (0.5, 0.1, (0.1210, 0.4713, 0.6732, 0.8375, 0.9542, 0.9895), 19),
        (0.4, 0.1, (0.0548, 0.2911, 0.4627, 0.6190, 0.8009, 0.9151), 30),
    ],
)
def test_a1_fisher_power_grid(
    treatment: float, control: float, expected: tuple[float, ...], n_for_power: int
) -> None:
    for n, want in zip(A1_CASES, expected, strict=True):
        assert _close(fisher_power(n, treatment, control), want), f"n={n}"
    assert smallest_n_for_fisher_power(treatment, control, max_n=200) == n_for_power


@pytest.mark.parametrize(
    ("rate", "expected", "n_for_width"),
    [
        (0.8, (0.2941, 0.2266, 0.1907, 0.1677, 0.1390, 0.1213), 26),
        (0.6, (0.3258, 0.2596, 0.2221, 0.1973, 0.1654, 0.1453), 38),
        (0.5, (0.3258, 0.2634, 0.2254, 0.2007, 0.1685, 0.1480), 39),
        (0.4, (0.3258, 0.2596, 0.2221, 0.1973, 0.1654, 0.1453), 38),
    ],
)
def test_a1_wilson_half_width_grid(
    rate: float, expected: tuple[float, ...], n_for_width: int
) -> None:
    for n, want in zip(A1_CASES, expected, strict=True):
        assert _close(wilson_half_width(n, rate), want), f"n={n}"
    assert smallest_n_for_wilson_half_width(rate, 0.15) == n_for_width


def test_a1_half_width_at_the_recommended_and_interval_sizes() -> None:
    """The two n the report puts side by side: twenty for the contrast, twenty-six for 0.15."""
    assert _close(wilson_half_width(20, 0.8), 0.1677)
    assert _close(wilson_half_width(26, 0.8), 0.1468)
    assert wilson_half_width(26, 0.8) < 0.15 <= wilson_half_width(20, 0.8)


@pytest.mark.parametrize(
    ("n", "detectable"),
    [
        (5, 0.8314),
        (10, 0.4837),
        (15, 0.3373),
        (20, 0.3133),
        (26, 0.2450),
        (30, 0.2139),
        (40, 0.1623),
    ],
)
def test_a1_smallest_detectable_flip_rate_against_zero_controls(
    n: int, detectable: float
) -> None:
    """At five cases the threshold sits *above* the rate that was observed.

    This is the figure behind the report's answer for P6: a design of five cases per arm does
    not reliably detect a true flip rate of 0.8, so the recorded 4 of 5 cannot be read as having
    measured the rate -- only as having separated it from zero.
    """
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if fisher_power(n, middle, 0.0) >= 0.80:
            high = middle
        else:
            low = middle
    assert _close(high, detectable, tolerance=5e-4)
    if n == 5:
        assert high > 0.8, "five cases must not reach 80% power at the observed rate"


@pytest.mark.parametrize(
    ("n", "one_sided", "two_sided"),
    [
        (5, 2.3810e-02, 4.7619e-02),
        (10, 3.5722e-04, 7.1443e-04),
        (15, 5.2609e-06, 1.0522e-05),
        (20, 7.7093e-08, 1.5419e-07),
        (26, 3.4262e-10, 6.8524e-10),
        (30, 1.6472e-11, 3.2944e-11),
        (40, 3.5098e-15, 7.0196e-15),
    ],
)
def test_a1_observed_configuration_p_values(n: int, one_sided: float, two_sided: float) -> None:
    """The contrast at the observed 0.8 flip rate against zero controls, both tails.

    Both are reported because A1 specifies the one-sided alternative while figures quoted for
    this same contrast elsewhere are two-sided, and on a 2x2 with an empty control arm the two
    differ by exactly a factor of two.
    """
    flips = expected_successes(n, 0.8)
    assert fisher_p(flips, n, 0, n) == pytest.approx(one_sided, rel=1e-3)
    assert fisher_p(flips, n, 0, n, tail="two-sided") == pytest.approx(two_sided, rel=1e-3)


# ------------------------------------------------------------------------------- A2 tables


@pytest.mark.parametrize(
    ("share", "expected"),
    [
        (0.60, (0.0706, 0.1328, 0.2090, 0.2786, 0.4074)),
        (0.65, (0.1343, 0.2817, 0.4411, 0.5715, 0.7604)),
        (OBSERVED_MATCHED_SHARE, (0.2709, 0.5561, 0.7692, 0.8862, 0.9754)),
        (0.80, (0.5422, 0.8815, 0.9780, 0.9965, 0.9999)),
    ],
)
def test_a2_paired_power_grid(share: float, expected: tuple[float, ...]) -> None:
    for n, want in zip(A2_POINTS, expected, strict=True):
        assert _close(mcnemar_power(n, OBSERVED_DISCORDANCE, share), want), f"n={n}"


def test_a2_sample_size_and_detectable_share() -> None:
    assert (
        smallest_n_for_mcnemar_power(OBSERVED_DISCORDANCE, OBSERVED_MATCHED_SHARE, max_n=400)
        == 135
    )
    assert _close(smallest_detectable_matched_share(42, OBSERVED_DISCORDANCE), 0.8763)
    assert _close(smallest_detectable_matched_share(84, OBSERVED_DISCORDANCE), 0.7731)


def test_a2_conditional_and_unconditional_power_differ_and_cross() -> None:
    """The two A2 objects are not interchangeable, and which is larger depends on n.

    Conditioning on the expected discordant count understates power at n = 42 -- the exact test
    on fourteen pairs is lumpy enough that luckier draws of D repay the unluckier ones -- and
    overstates it from n = 84 up. Pinned so neither figure can quietly stand in for the other.
    """
    conditional = {42: 0.1904, 84: 0.5939, 126: 0.8053, 168: 0.9062, 252: 0.9777}
    for n, want in conditional.items():
        discordant = round(n * OBSERVED_DISCORDANCE)
        assert _close(
            mcnemar_power_given_discordant(discordant, OBSERVED_MATCHED_SHARE), want
        ), f"n={n}"
    # The crossing: unconditional above at the recorded n, below from twice that size on.
    assert mcnemar_power(42, OBSERVED_DISCORDANCE, OBSERVED_MATCHED_SHARE) > conditional[42]
    for n in (84, 126, 168, 252):
        assert mcnemar_power(n, OBSERVED_DISCORDANCE, OBSERVED_MATCHED_SHARE) < conditional[n]


# ------------------------------------------------------------------------------- A3 tables


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0.60, (0.2366, 0.4061, 0.6996)),
        (0.65, (0.4808, 0.7626, 0.9696)),
        (0.70, (0.7430, 0.9565, 0.9995)),
    ],
)
def test_a3_sign_test_power_grid(rate: float, expected: tuple[float, ...]) -> None:
    for n, want in zip(A3_POINTS, expected, strict=True):
        assert _close(sign_test_power(n, rate), want), f"n={n}"


@pytest.mark.parametrize(
    ("n", "detectable"), [(42, 0.7130), (84, 0.6565), (168, 0.6119)]
)
def test_a3_smallest_detectable_rate(n: int, detectable: float) -> None:
    assert _close(smallest_detectable_sign_rate(n), detectable)


def test_a3_refines_the_work_order_threshold() -> None:
    """WO-STAT-001 §1 A3 asserts n = 42 excludes a trace at r >= ~0.7. It is close, not exact.

    At exactly 0.70 the power is 0.743, short of the 0.80 convention; the rate that reaches 0.80
    is 0.713. The assertion survives as written -- its "~" covers 0.713 -- but the threshold is
    stated precisely here so the report does not round it into a claim the arithmetic misses.
    """
    assert sign_test_power(42, 0.70) < 0.80
    assert _close(sign_test_power(42, 0.70), 0.7430)
    assert _close(smallest_detectable_sign_rate(42), 0.7130)
    assert smallest_n_for_sign_power(0.70) == 54


# ------------------------------------------------------------------------------- A4 tables


@pytest.mark.parametrize(
    ("family", "expected", "detectable", "n_for_power"),
    [
        (5, (0.0860, 0.2411, 0.4957), 0.7574, 76),
        (6, (0.0860, 0.2411, 0.4957), 0.7574, 79),
        (8, (0.0449, 0.1499, 0.3632), 0.7793, 82),
        (9, (0.0449, 0.1499, 0.3632), 0.7793, 85),
    ],
)
def test_a4_holm_corrected_grid(
    family: int, expected: tuple[float, ...], detectable: float, n_for_power: int
) -> None:
    level = holm_alpha(DEFAULT_ALPHA, family)
    for rate, want in zip((0.60, 0.65, 0.70), expected, strict=True):
        assert _close(sign_test_power(42, rate, level), want), f"r={rate}"
    assert _close(smallest_detectable_sign_rate(42, alpha=level), detectable)
    assert smallest_n_for_sign_power(0.70, alpha=level) == n_for_power


def test_a4_family_size_five_and_six_are_indistinguishable_at_the_recorded_n() -> None:
    """The discrete sign test does not resolve the difference between adjacent family sizes.

    alpha/5 and alpha/6 fall between the same two attainable p-values at n = 42, as do alpha/8
    and alpha/9, so the recorded lens rows would have been corrected identically whichever of
    each pair the family really was. This is why the work order's imprecision about family size
    costs nothing in the power table, though it does move the n each family would need.
    """
    assert _sign_test_rejection_set(42, holm_alpha(DEFAULT_ALPHA, 5)) == _sign_test_rejection_set(
        42, holm_alpha(DEFAULT_ALPHA, 6)
    )
    assert _sign_test_rejection_set(42, holm_alpha(DEFAULT_ALPHA, 8)) == _sign_test_rejection_set(
        42, holm_alpha(DEFAULT_ALPHA, 9)
    )
    assert smallest_n_for_sign_power(0.70, alpha=holm_alpha(DEFAULT_ALPHA, 5)) != (
        smallest_n_for_sign_power(0.70, alpha=holm_alpha(DEFAULT_ALPHA, 6))
    )


def test_holm_alpha_rejects_an_empty_family() -> None:
    with pytest.raises(ValueError, match="at least one test"):
        holm_alpha(DEFAULT_ALPHA, 0)


def test_fisher_p_rejects_an_unknown_tail() -> None:
    with pytest.raises(ValueError, match="tail must be"):
        fisher_p(4, 5, 0, 5, tail="less")


# ------------------------------------------------------------------- provenance, when present


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PATCH_ARTIFACT = _PROJECT_ROOT / "outputs/probes/patch-C-r27-2026-09-05/patch.json"
_SWEEP_ARTIFACT = (
    _PROJECT_ROOT / "outputs/probes/jspace-qwen35-4b-base-2026-09-05-rerun/sweep.json"
)


@pytest.mark.skipif(
    not (_PATCH_ARTIFACT.exists() and _SWEEP_ARTIFACT.exists()),
    reason="run artifacts are untracked; the report carries their paths and fields instead",
)
def test_recorded_statistics_match_their_artifacts() -> None:
    """R38: the observed inputs are what the named artifacts actually hold.

    Skips where the untracked run outputs are absent, so the suite stays hermetic; where they
    are present this is the check that the report transcribed nothing.
    """
    patch = json.loads(_PATCH_ARTIFACT.read_text())
    notes = patch["cells"]["6:previous_notes"]
    assert notes["treatment"]["numerator"] == 4
    assert notes["treatment"]["denominator"] == 5
    assert notes["treatment"]["wilson_95"] == list(wilson_interval(4, 5))
    for control in notes["controls"].values():
        assert control["numerator"] == 0 and control["denominator"] == 5

    sweep = json.loads(_SWEEP_ARTIFACT.read_text())
    per_case = sweep["per_case"]
    assert len(per_case) == 42
    matched_only = mismatched_only = 0
    decomposition = 0
    for case in per_case:
        matched, mismatched = case["matched"]["model_output"], case["mismatched"]["model_output"]
        matched_win = matched["true"] > matched["false"]
        mismatched_win = mismatched["true"] > mismatched["false"]
        matched_only += matched_win and not mismatched_win
        mismatched_only += mismatched_win and not matched_win
        decomposition += matched["true"] > mismatched["true"]
    assert (matched_only, mismatched_only) == (10, 4)
    assert matched_only + mismatched_only == 14
    assert decomposition == 19
    assert _close(sign_test_p(matched_only, 14), 0.179565)
    assert _close(sign_test_p(decomposition, 42), 0.643969)
    assert sweep["results"]["model_output"]["matched_wins"] == 30
    assert sweep["results"]["model_output"]["mismatched_wins"] == 24

    # The two per-case figures the report quotes from this artifact in its own right.
    matched_true = [case["matched"]["model_output"]["true"] for case in per_case]
    mismatched_true = [case["mismatched"]["model_output"]["true"] for case in per_case]
    assert _close(max(matched_true), 0.227368)
    differences = sorted(a - b for a, b in zip(matched_true, mismatched_true, strict=True))
    assert _close(statistics.median(differences), -0.003843)
    assert _close(statistics.median([abs(d) for d in differences]), 0.017468)
    assert _close(statistics.median([d for d in differences if d > 0]), 0.023562)
    # The shift that would carry the win rate to the Wilson upper bound: ~0.0025, not 0.001.
    shifted = [d - differences[len(differences) - 25] for d in differences]
    assert _close(statistics.median(shifted), 0.002549)
    assert not _close(statistics.median(shifted), 0.001, tolerance=1e-3)


@pytest.mark.skipif(
    not _SWEEP_ARTIFACT.exists(),
    reason="run artifacts are untracked; the report carries their paths and fields instead",
)
def test_recorded_holm_family_sizes_vary_by_readout() -> None:
    """WO-STAT-001 §1 A4 says six on the 3B and eight on the 4B "per readout"; it varies.

    The final layer contributes only a ``self`` row -- ``all`` and ``future`` are excluded there
    because no decoder block remains in the tail -- so the ``self`` family is one larger than the
    other two on each model. Counted from the rows actually carrying a Holm-adjusted p.
    """
    expected = {
        _SWEEP_ARTIFACT: {"all": 8, "future": 8, "self": 9},
        _SWEEP_ARTIFACT.parent.parent
        / "jspace-qwen25-coder-3b-base-2026-09-05-rerun/sweep.json": {
            "all": 5,
            "future": 5,
            "self": 6,
        },
    }
    for artifact, want in expected.items():
        if not artifact.exists():
            pytest.skip(f"{artifact.name} comparator run is absent")
        results = json.loads(artifact.read_text())["results"]
        sizes: dict[str, int] = {}
        for key, row in results.items():
            if key.startswith("jlens_") and "matched_p_holm" in row:
                sizes[key.rsplit("_", 1)[1]] = sizes.get(key.rsplit("_", 1)[1], 0) + 1
        assert sizes == want, artifact.name
