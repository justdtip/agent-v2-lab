"""WP4's pre-check arithmetic and the sizing it licenses (issue 82, parts b and c).

No model. The share is a pure function of an alpha row and the vectors it weights, so every case
below is exact and the fixtures stand in for WP3's hooks at the interface the ratified design
declares: ``y_t = sum_s alpha[t, s] * v_s``.
"""

from __future__ import annotations

import math

import pytest

from local_llm_lab.probes.power import holm_alpha, mcnemar_power
from local_llm_lab.probes.read_share import (
    LICENSED_SHARE,
    NOT_WORTH_RUNNING,
    ReadShare,
    read_share,
    read_share_by_layer,
    size_two_by_two,
)


def _stand_in_rate(share: float) -> float:
    """A stand-in mapping from read share to matched-only share, for these tests only.

    Deliberately not a default of the module and deliberately not defensible as science. It
    exists so the sizing can be exercised, it is passed explicitly at every call site here, and
    the real mapping is the Head of Interpretability's to set. Chosen only so the licensed 0.05
    lands somewhere the paired test can reach at a plannable n: 0.05 maps to 0.70, which needs
    217 points at the Holm level for four arms.
    """
    return min(0.95, 0.5 + 4.0 * share)


# ------------------------------------------------------------------- the share itself


def test_the_share_is_a_norm_of_a_sum_not_a_sum_of_norms() -> None:
    """The distinction the pre-check turns on, in the smallest case that shows it.

    Two sources of equal weight pointing in opposite directions cancel to nothing. Summed as
    magnitudes the hidden span would look like half the read; as a norm of a sum, the read it
    contributes to is zero, and the share is undefined rather than 0.5.
    """
    row = [1.0, 1.0]
    values = [[1.0, 0.0], [-1.0, 0.0]]

    cancelling = read_share(row, values, span=[0])
    assert cancelling.total_norm == 0.0
    assert cancelling.defined is False
    assert math.isnan(cancelling.share)

    # Same weights, same span, sources no longer opposed: now the read exists and the span is
    # half of it.
    aligned = read_share(row, [[1.0, 0.0], [1.0, 0.0]], span=[0])
    assert aligned.share == pytest.approx(0.5)


def test_a_share_above_one_is_a_reading_and_is_not_clipped() -> None:
    """Cancellation makes shares above 1 real, and clipping them would hide the finding.

    The span carries 2.0 along an axis, the rest carries -1.5 along the same axis, so the read
    is 0.5 and the span accounts for four times it. Removing the span would move the read
    further than the read's own size, which is exactly what a reader needs to know and exactly
    what a clip to 1.0 would erase.
    """
    result = read_share([2.0, -1.5], [[1.0], [1.0]], span=[0])
    assert result.share == pytest.approx(4.0)


def test_the_span_is_checked_against_the_row_rather_than_silently_ignored() -> None:
    """A span index outside the row means the caller located the wrong thing, not an empty span.

    EXP-002 locates the masked spans from message boundaries in the rendered token sequence, and
    an off-by-one there would otherwise score a share of zero and read as "the hidden text
    contributes nothing" -- the exact false negative the experiment exists to avoid.
    """
    with pytest.raises(ValueError, match="outside the row"):
        read_share([1.0, 1.0], [[1.0], [1.0]], span=[0, 7])

    with pytest.raises(ValueError, match="index the same source positions"):
        read_share([1.0, 1.0, 1.0], [[1.0], [1.0]], span=[0])


def test_an_empty_span_contributes_nothing_and_a_full_span_contributes_everything() -> None:
    """The two ends, so the middle has something to be between."""
    row = [0.25, 0.75]
    values = [[1.0, 2.0], [3.0, -1.0]]

    assert read_share(row, values, span=[]).share == pytest.approx(0.0)
    assert read_share(row, values, span=[0, 1]).share == pytest.approx(1.0)


# ---------------------------------------------------------- aggregating to one figure a layer


def test_a_layer_reports_its_maximum_head_because_one_head_is_the_case_to_catch() -> None:
    """The rule asks whether the share is below a threshold at every layer.

    A mean over 32 heads would put a single head carrying the whole read below the threshold,
    which is the reading the pre-check exists to prevent. Undefined cells are counted, not
    dropped silently.
    """
    cells = {
        13: [
            ReadShare(span_norm=0.01, total_norm=1.0, defined=True),
            ReadShare(span_norm=0.90, total_norm=1.0, defined=True),
            ReadShare(span_norm=0.00, total_norm=0.0, defined=False),
        ],
        29: [ReadShare(span_norm=0.005, total_norm=1.0, defined=True)],
    }
    summary = read_share_by_layer(cells)

    assert summary[13]["max_share"] == pytest.approx(0.90)
    assert summary[13]["undefined"] == 1
    assert summary[13]["cells"] == 3
    assert summary[29]["max_share"] == pytest.approx(0.005)


def test_a_layer_whose_cells_are_all_undefined_reports_no_share_rather_than_zero() -> None:
    """ "No read to attribute" and "the span contributes none of the read" are different."""
    summary = read_share_by_layer({7: [ReadShare(0.0, 0.0, False)]})
    assert summary[7]["max_share"] is None
    assert summary[7]["undefined"] == 1


# ------------------------------------------------------------------------- the sizing


def test_a_negligible_share_everywhere_sizes_against_the_licensed_005_not_the_observed() -> None:
    """The plan's rule, first half: below 0.02 at every layer, size against 0.05 instead.

    Sizing against the observed share would ask the 2 by 2 to detect a thing the pre-check has
    already said is not there, which produces an n nobody will run rather than a decision.
    """
    verdict = size_two_by_two(
        {13: 0.004, 16: 0.011, 19: 0.002, 23: 0.008, 27: 0.001},
        max_head_share_by_layer={13: 0.004, 16: 0.011, 19: 0.002, 23: 0.008, 27: 0.001},
        matched_share_for=_stand_in_rate,
        matched_share_source="test stand-in",
        discordance=14 / 42,
    )
    assert verdict.inputs["every_layer_negligible"] is True
    assert verdict.licensed_share == LICENSED_SHARE
    assert verdict.observed_max_share == pytest.approx(0.011)
    assert verdict.outcome == "sized"


def test_a_share_above_the_threshold_at_one_layer_sizes_against_what_was_observed() -> None:
    """One layer above 0.02 is enough: the rule says *every* layer, and this is the other branch."""
    verdict = size_two_by_two(
        {13: 0.004, 16: 0.140, 19: 0.002},
        max_head_share_by_layer={13: 0.004, 16: 0.140, 19: 0.002},
        matched_share_for=_stand_in_rate,
        matched_share_source="test stand-in",
        discordance=14 / 42,
    )
    assert verdict.inputs["every_layer_negligible"] is False
    assert verdict.licensed_share == pytest.approx(0.140)
    # A larger effect needs fewer points, which is the direction that says the sizing is live.
    smaller = size_two_by_two(
        {13: 0.004},
        max_head_share_by_layer={13: 0.004},
        matched_share_for=_stand_in_rate,
        matched_share_source="test stand-in",
        discordance=14 / 42,
    )
    assert verdict.points < smaller.points


def test_not_worth_running_is_returned_as_an_outcome_rather_than_raised() -> None:
    """The rule's second half is a real result, and the plan says to record it plainly.

    A mapping that turns even the licensed share into a coin flip cannot be detected at any n,
    and the honest answer is the sentence, not an exception and not a very large number.
    """
    verdict = size_two_by_two(
        {13: 0.001},
        max_head_share_by_layer={13: 0.001},
        matched_share_for=lambda _share: 0.5,
        matched_share_source="test stand-in: a coin flip",
        discordance=14 / 42,
        max_points=200,
    )
    assert verdict.outcome == NOT_WORTH_RUNNING
    assert verdict.points is None
    assert verdict.power_at_points is None


def test_the_size_is_computed_at_the_holm_level_for_the_arm_family() -> None:
    """Four cells read against arm C is a family of four, and power is computed at alpha / 4.

    Sizing at the uncorrected alpha would name an n that cannot carry the family it belongs to.
    """
    verdict = size_two_by_two(
        {13: 0.06},
        max_head_share_by_layer={13: 0.06},
        matched_share_for=_stand_in_rate,
        matched_share_source="test stand-in",
        discordance=14 / 42,
        arms=4,
    )
    assert verdict.holm_alpha == pytest.approx(holm_alpha(0.05, 4))
    assert verdict.power_at_points == pytest.approx(
        mcnemar_power(verdict.points, 14 / 42, verdict.matched_share, verdict.holm_alpha)
    )
    assert verdict.power_at_points >= verdict.power_target

    uncorrected = size_two_by_two(
        {13: 0.06},
        max_head_share_by_layer={13: 0.06},
        matched_share_for=_stand_in_rate,
        matched_share_source="test stand-in",
        discordance=14 / 42,
        arms=1,
    )
    assert uncorrected.points < verdict.points, "the correction must cost points, not be free"


def test_the_reported_n_is_a_stable_crossing_and_not_the_first_touch() -> None:
    """Discrete tests are not monotone in n, so 'the n for 80 percent' has to mean the crossing.

    WO-STAT-001's correction records the shape: Fisher at 3 of 3 against 0 of 3 rejects with
    power 0.512, and the next attainable table at n = 4 has power 0.410. A sizing that returned
    the first n to touch the target would name a point the test drops below again.
    """
    verdict = size_two_by_two(
        {13: 0.06},
        max_head_share_by_layer={13: 0.06},
        matched_share_for=_stand_in_rate,
        matched_share_source="test stand-in",
        discordance=14 / 42,
    )
    assert all(
        mcnemar_power(n, verdict.discordance, verdict.matched_share, verdict.holm_alpha)
        >= verdict.power_target
        for n in range(verdict.points, verdict.points + 12)
    )


def test_every_input_is_carried_on_the_verdict_so_a_number_names_its_source() -> None:
    """R38: a figure names its source, and a sizing is a figure with several."""
    shares = {13: 0.004, 27: 0.001}
    verdict = size_two_by_two(
        shares,
        max_head_share_by_layer=shares,
        matched_share_for=_stand_in_rate,
        matched_share_source="test stand-in",
        discordance=14 / 42,
        arms=4,
    )
    assert verdict.inputs["layer_share_by_layer"] == shares
    assert verdict.inputs["negligible_threshold"] == 0.02
    assert verdict.inputs["licensed_threshold"] == 0.05
    assert "plan section 3, WP4" in verdict.inputs["rule"]
    assert verdict.discordance == pytest.approx(14 / 42)
    assert verdict.arms == 4


def test_a_pre_check_with_no_layers_is_refused_rather_than_sized() -> None:
    """An empty pre-check is a failed pre-check, and sizing off it would invent a licence."""
    with pytest.raises(ValueError, match="at least one layer"):
        size_two_by_two(
            {},
            max_head_share_by_layer={},
            matched_share_for=_stand_in_rate,
            matched_share_source="test stand-in",
            discordance=14 / 42,
        )


def test_the_layer_share_is_magnitude_weighted_and_the_max_head_is_not() -> None:
    """The two figures answer different questions, and a quiet head is where they part.

    One head reads loudly and takes almost none of its read from the span; another is a thousand
    times quieter and takes nearly all of its read from it. The maximum over heads is 0.9, which
    is the right answer to "is this layer negligible" and the wrong effect to size a 2 by 2
    against: nothing the experiment measures will produce it, because the head producing it
    contributes almost nothing to the layer.

    The magnitude-weighted figure is the norm share of the concatenated per-head reads, and it
    lands near the loud head's own share.
    """
    loud = ReadShare(span_norm=0.001, total_norm=1.0, defined=True)
    quiet = ReadShare(span_norm=0.0009, total_norm=0.001, defined=True)
    summary = read_share_by_layer({13: [loud, quiet]})

    assert summary[13]["max_share"] == pytest.approx(0.9)
    assert summary[13]["layer_share"] == pytest.approx(0.00134, abs=1e-5)
    assert summary[13]["layer_share"] < summary[13]["max_share"] / 100


def test_a_layer_with_no_defined_cells_reports_no_layer_share_either() -> None:
    """Both figures are absent together; neither may read as zero."""
    summary = read_share_by_layer({7: [ReadShare(0.0, 0.0, False)]})
    assert summary[7]["max_share"] is None
    assert summary[7]["layer_share"] is None


def test_a_pre_check_whose_every_layer_is_undefined_is_refused() -> None:
    """The zero rule one level up: "measured nothing" is not "measured nearly nothing".

    Sized as negligible, an all-undefined pre-check would fall through to the licensed 0.05 and
    hand back a plannable n, which reads as a licence the pre-check never gave.
    """
    with pytest.raises(ValueError, match="undefined"):
        size_two_by_two(
            {13: None, 16: None},
            max_head_share_by_layer={13: None, 16: None},
            matched_share_for=_stand_in_rate,
            matched_share_source="test stand-in",
            discordance=14 / 42,
        )


def test_the_quiet_head_case_sizes_against_the_licensed_share_and_raises_the_flag() -> None:
    """The pairing, and why the maximum cannot decide negligibility.

    ``layer_share <= max_head_share`` at every layer, always: ``span_h <= m * total_h`` termwise
    gives it. So negligibility on the maximum is strictly stricter, and it routes this case
    backwards — a loud head at 0.001 beside a quiet head at 0.9 would fail negligibility at 0.900,
    fall to the effect branch, be sized against a layer share of 0.00135, and come back "not worth
    running", when the plan's rule applied to the quantity the plan names sizes against 0.05 and
    returns something plannable.

    The maximum still earns its place: it is recorded, it never sizes, and
    ``single_head_dominates`` puts the case in front of the reader.
    """
    loud = ReadShare(span_norm=0.001, total_norm=1.0, defined=True)
    quiet = ReadShare(span_norm=0.0009, total_norm=0.001, defined=True)
    summary = read_share_by_layer({13: [loud, quiet]})
    layer = {13: summary[13]["layer_share"]}
    maximum = {13: summary[13]["max_share"]}

    verdict = size_two_by_two(
        layer,
        max_head_share_by_layer=maximum,
        matched_share_for=_stand_in_rate,
        matched_share_source="test stand-in",
        discordance=14 / 42,
    )
    assert verdict.licensed_share == LICENSED_SHARE
    assert verdict.outcome == "sized"
    assert verdict.points == 217
    assert verdict.single_head_dominates is True
    assert verdict.max_head_share_by_layer == maximum


def test_the_dominance_flag_is_false_when_the_two_statistics_agree() -> None:
    """The flag reports a disagreement, not merely a large maximum."""
    agreeing = size_two_by_two(
        {13: 0.004},
        max_head_share_by_layer={13: 0.006},
        matched_share_for=_stand_in_rate,
        matched_share_source="test stand-in",
        discordance=14 / 42,
    )
    assert agreeing.single_head_dominates is False

    both_large = size_two_by_two(
        {13: 0.4},
        max_head_share_by_layer={13: 0.9},
        matched_share_for=_stand_in_rate,
        matched_share_source="test stand-in",
        discordance=14 / 42,
    )
    assert both_large.single_head_dominates is False, "not negligible either way is not dominance"


def test_the_two_mappings_must_describe_the_same_layers() -> None:
    """A maximum for a layer with no layer share, or the reverse, is a pre-check half read."""
    with pytest.raises(ValueError, match="same layers"):
        size_two_by_two(
            {13: 0.004, 16: 0.004},
            max_head_share_by_layer={13: 0.004},
            matched_share_for=_stand_in_rate,
            matched_share_source="test stand-in",
            discordance=14 / 42,
        )


def test_the_mapping_is_named_on_the_verdict() -> None:
    """Same argument as ``matched_share_for`` having no default: a number names its source."""
    verdict = size_two_by_two(
        {13: 0.004},
        max_head_share_by_layer={13: 0.004},
        matched_share_for=_stand_in_rate,
        matched_share_source="test stand-in, 2026-09-08",
        discordance=14 / 42,
    )
    assert verdict.matched_share_source == "test stand-in, 2026-09-08"
    assert verdict.inputs["matched_share_source"] == "test stand-in, 2026-09-08"


def test_not_worth_running_fires_on_both_branches_and_they_are_distinguishable() -> None:
    """The sentinel is not the negligible branch's alone, and the verdict says which fired."""
    negligible = size_two_by_two(
        {13: 0.001},
        max_head_share_by_layer={13: 0.001},
        matched_share_for=lambda _share: 0.5,
        matched_share_source="test stand-in: a coin flip",
        discordance=14 / 42,
        max_points=200,
    )
    observed = size_two_by_two(
        {13: 0.9},
        max_head_share_by_layer={13: 0.9},
        matched_share_for=lambda _share: 0.5,
        matched_share_source="test stand-in: a coin flip",
        discordance=14 / 42,
        max_points=200,
    )
    assert negligible.outcome == observed.outcome == NOT_WORTH_RUNNING
    assert negligible.inputs["every_layer_negligible"] is True
    assert observed.inputs["every_layer_negligible"] is False
