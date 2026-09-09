"""The derivation table: no tolerance typed by hand, two rules as mechanisms, the closed forms."""

from __future__ import annotations

import numpy as np
import pytest

from local_llm_lab.pipeline.state_programme import tolerances as tol


def test_the_two_closed_form_checks_the_order_fixes() -> None:
    assert tol.required_n(10, 0.05) == 2397
    assert tol.required_n(10, 0.075) == 1066
    tol.verify_closed_forms()
    with pytest.raises(ValueError):
        tol.required_n(0, 0.05)


def test_the_bootstrap_bound_is_seeded_and_brackets_a_real_contrast() -> None:
    e = np.array([1.0] * 80 + [0.0] * 20)
    a = np.array([1.0] * 20 + [0.0] * 80)
    lb1, ub1 = tol.bootstrap_bounds(e, a, seed=1, resamples=2000)
    lb2, _ = tol.bootstrap_bounds(e, a, seed=1, resamples=2000)
    assert lb1 == lb2
    assert 0.4 < lb1 < 0.6 < ub1 < 0.8


def _contrast(name, e, a, seed=3):
    return tol.contrasts({name: e}, {name: a}, seed=seed, resamples=2000)[0]


def test_the_drop_rule_keeps_reached_contrasts_on_either_side_and_drops_noise() -> None:
    up = _contrast("up", [1.0] * 90 + [0.0] * 10, [1.0] * 10 + [0.0] * 90)
    down = _contrast("down", [1.0] * 10 + [0.0] * 90, [1.0] * 90 + [0.0] * 10)
    noise = _contrast("noise", [1.0, 0.0] * 50, [0.0, 1.0] * 50)
    unreached = _contrast("unreached", [None] * 20, [1.0] * 20)
    kept, dropped = tol.drop_rule([up, down, noise, unreached])

    assert [k.name for k in kept] == ["up", "down"]
    assert {d.name for d in dropped} == {"noise", "unreached"}
    reasons = {d.name: d.reason for d in dropped}
    assert "does not reach" in reasons["noise"] and "not reached" in reasons["unreached"]
    assert tol.magnitude_bound(down) > 0 and down.difference < 0


def test_the_table_derives_every_tolerance_from_d_min_and_writes_its_derivation() -> None:
    rows = [
        _contrast("D1", [1.0] * 95 + [0.0] * 5, [1.0] * 5 + [0.0] * 95),
        _contrast("D4", [1.0] * 70 + [0.0] * 30, [1.0] * 30 + [0.0] * 70),
        _contrast("D9", [1.0, 0.0] * 50, [0.0, 1.0] * 50),
    ]
    table = tol.derive(rows, split_half_log_loss_gap=0.031)

    assert table.retained == ("D1", "D4") and table.m == 2
    assert table.d_min == min(tol.magnitude_bound(rows[0]), tol.magnitude_bound(rows[1]))
    assert table.epsilon_sub == table.d_min / 4
    assert table.epsilon_perp == pytest.approx(2 * table.epsilon_sub / 3)
    assert table.epsilon_reuse == table.epsilon_sub
    assert table.epsilon_pred == table.epsilon_dyn == 0.031
    assert table.n == tol.required_n(2, table.epsilon_sub)
    d = table.as_dict()["derivation"]
    assert d["epsilon_sub"] == "d_min / 4" and d["n"].startswith("ceil(ln(2M/alpha)")
    assert {c["n"] for c in d["closed_form_checks"]} == {2397, 1066}


def test_an_empty_retained_set_refuses_rather_than_deriving_from_nothing() -> None:
    noise = _contrast("D2", [1.0, 0.0] * 50, [0.0, 1.0] * 50)
    with pytest.raises(ValueError, match="reaches nothing"):
        tol.derive([noise], split_half_log_loss_gap=0.01)


def test_the_budget_rule_loosens_epsilon_and_writes_both_numbers_never_fewer_diagnostics() -> None:
    rows = [_contrast("D1", [1.0] * 60 + [0.0] * 40, [1.0] * 40 + [0.0] * 60)]
    table = tol.derive(rows, split_half_log_loss_gap=0.02)
    within = tol.budget_rule(table, hours_bought=1e9, rate_per_hour=100.0)
    assert not within.loosened and within.n_final == table.n
    over = tol.budget_rule(table, hours_bought=10.0, rate_per_hour=100.0)

    assert over.loosened
    assert over.epsilon_sub_derived == table.epsilon_sub < over.epsilon_sub_final
    assert over.n_final == 250 and over.hours_final <= 10.0
    assert over.n_final == tol.required_n(table.m, over.epsilon_sub_final) or \
        tol.required_n(table.m, over.epsilon_sub_final) <= over.n_final + 1
    assert "M and the control are unchanged" in over.reason
    with pytest.raises(ValueError, match="laptop projects nothing"):
        tol.budget_rule(table, hours_bought=1.0, rate_per_hour=0.0)
