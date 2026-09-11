"""Regressions for review 3cb6247; synthetic NumPy inputs, no captures or models."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from local_llm_lab.pipeline.state_programme import transport, transport_certified


@pytest.fixture
def reader(monkeypatch):
    root = Path(transport.__file__).resolve().parents[4]
    loader = types.ModuleType("read_e1")

    def forbidden(*args, **kwargs):
        raise AssertionError("a synthetic test must never read a capture")

    loader.load_cells = loader.load_layers = forbidden
    monkeypatch.setitem(sys.modules, "read_e1", loader)
    path = root / "research/records/STATE-PLAN-PROGRESS-2026-09-09/read_e2.py"
    spec = importlib.util.spec_from_file_location("e2_review_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_moves():
    features = np.random.default_rng(70).normal(size=(18, 3)).astype(np.float32)
    moves = [dict(task_id=f"t{i}", source=2*i, target=2*i+1, m=1, fold=i % 2,
                  candidates=[2*i, 2*i+1]) for i in range(9)]
    folds = {m["task_id"]: m["fold"] for m in moves}
    strata = {m["task_id"]: ("train", "f", "v") for m in moves}
    return features, moves, folds, strata


def test_refit_preserves_repeated_episode_weights(reader):
    features, moves, folds, strata = fixture_moves()
    # Eight evaluated and fitting episodes. Seed 1 draws [3,4,6,7,0,1,6,7].
    # Their independently checked contrasts average to 0; unique tasks average to -1/6.
    result = reader.refitting_bootstrap(features, moves[:8], moves[:8], folds, strata,
                                       rank=1, resamples=1, seed=1)
    assert result["low"] == pytest.approx(0)
    assert result["high"] == pytest.approx(0)


def test_refit_includes_ordinary_only_training_episodes(reader, monkeypatch):
    features, moves, folds, strata = fixture_moves()
    real_fit = reader.transport_rule
    fitting_rows = []

    def observe(sources, targets, rank):
        fitting_rows.extend(sources)
        return real_fit(sources, targets, rank)

    monkeypatch.setattr(reader, "transport_rule", observe)
    # Make the ordinary-only episode a singleton stratum: it must be drawn exactly once.
    strata["t8"] = ("train", "f", "clean")
    result = reader.refitting_bootstrap(features, moves[:8], moves, folds, strata,
                                       rank=1, resamples=1, seed=1)
    assert any(np.array_equal(row, features[16]) for row in fitting_rows)
    assert result["complete"]


def test_training_and_evaluation_share_weights_and_fixed_fold_exclusion(reader, monkeypatch):
    features, moves, folds, strata = fixture_moves()
    real_fit = reader.transport_rule
    fits = []

    def observe(sources, targets, rank):
        fits.append([int(np.flatnonzero(np.all(features == row, axis=1))[0]) for row in sources])
        return real_fit(sources, targets, rank)

    monkeypatch.setattr(reader, "transport_rule", observe)
    reader.refitting_bootstrap(features, moves[:8], moves[:8], folds, strata,
                              rank=1, resamples=1, seed=1)
    # Other-fold rows only; repeated t6 and t7 rows must enter their respective fits twice.
    assert fits == [[2, 6, 14, 14], [0, 8, 12, 12]]


def test_disjoint_evaluation_and_training_populations_can_refit(reader):
    features, moves, folds, strata = fixture_moves()
    strata = {m["task_id"]: ("train", m["task_id"], "v") for m in moves}
    result = reader.refitting_bootstrap(features, moves[8:], moves[:8], folds, strata,
                                       rank=1, resamples=2, seed=1)
    assert result["resamples"] == 2
    assert result["complete"]


def test_partial_draw_invalidates_required_interval(reader):
    features, moves, folds, strata = fixture_moves()
    moves = moves[:3]
    strata = {m["task_id"]: ("train", m["task_id"], "v") for m in moves}
    result = reader.refitting_bootstrap(features, moves, moves, folds, strata,
                                       rank=1, resamples=2, seed=1)
    assert result["low"] is None and result["high"] is None
    assert result["complete"] is False
    assert result["folds_unavailable"] == [0]


@pytest.mark.parametrize("count", [0, -1])
def test_nonpositive_refit_count_is_refused(reader, count):
    features, moves, folds, strata = fixture_moves()
    with pytest.raises(reader.Refused, match="positive"):
        reader.refitting_bootstrap(features, moves, moves, folds, strata,
                                  rank=1, resamples=count, seed=1)


@pytest.fixture
def scored_cell(reader, monkeypatch):
    rows = [dict(task_id=f"e{i}", hit=float(i < 420), chance=.5, distance=1.)
            for i in range(600)]
    monkeypatch.setattr(reader, "score", lambda *args: (rows, set()))
    tolerance = dict(name="corrective", epsilon=.15, needs_n_at_least=549,
                     m=12, alpha=.05, range_width=2.)

    def evaluate(refit, registered=True):
        return reader.evaluate([None]*600, np.zeros((1, 1)), {}, 8, tolerance,
                               20260910, registered, refit)

    return evaluate


def valid_refit():
    return dict(low=.18, high=.22, resamples=10000, requested_resamples=10000,
                refitted=True, complete=True)


def test_hoeffding_competes_with_refit_not_fixed_fit_bootstrap(scored_cell):
    cell = scored_cell(valid_refit())
    halfwidth = math.sqrt(2 * math.log(480) / 600)
    assert cell["hoeffding_interval"]["low"] == pytest.approx(.2 - halfwidth)
    assert cell["hoeffding_interval"]["high"] == pytest.approx(.2 + halfwidth)
    assert cell["governing_interval"]["from"] == "Hoeffding"
    assert "straddles" in cell["reading"]


def test_wider_refitted_interval_governs(scored_cell):
    cell = scored_cell(dict(valid_refit(), low=-.5, high=.7))
    assert cell["governing_interval"]["from"] == "paired, refitted within each resample"
    assert cell["governing_interval"]["low"] == -.5


@pytest.mark.parametrize("refit", [None, dict(low=None, high=None, resamples=0),
    dict(valid_refit(), complete=False), dict(valid_refit(), resamples=9999),
    dict(valid_refit(), requested_resamples=1, resamples=1),
    dict(valid_refit(), low=float("nan")), dict(valid_refit(), high=.1),
    dict(valid_refit(), refitted=False)])
def test_missing_or_invalid_refit_cannot_produce_registered_reading(scored_cell, refit):
    cell = scored_cell(refit)
    assert cell["governing_interval"] is None
    assert cell["tolerance"]["applied"] is False
    assert "unavailable" in cell["reading"]


def test_exploratory_cell_does_not_apply_registered_tolerance(scored_cell):
    cell = scored_cell(None, registered=False)
    assert not cell["tolerance"]["applied"]
    assert "exploratory" in cell["reading"]


@pytest.fixture
def admission(reader, monkeypatch, tmp_path):
    record = tmp_path / "research/records/fixture"
    record.mkdir(parents=True)
    (record / "folds.json").write_text(json.dumps(dict(assignment_sha256="test", fold_of={})))
    (record / "capture-budget.json").write_text(json.dumps(dict(models={"a": {}, "b": {}})))
    seal = dict(verified=dict(files="fixture", baseline="fixture", seal_sha256="fixture"),
                folds=dict(assignment_sha256="test"))
    addendum = dict(verified=dict(addendum="fixture", sha256="fixture",
                    baseline_commit="fixture", superseded_ignored=[]))
    monkeypatch.setattr(reader, "require_seal", lambda *a, **kw: seal)
    monkeypatch.setattr(reader, "require_addendum", lambda *a, **kw: addendum)
    captures = tmp_path / "captures"
    for name in ("a", "b"):
        (captures / name).mkdir(parents=True)
        (captures / name / "manifest.jsonl").write_text("")
    calls = []

    def load(*args):
        calls.append("load")
        return [], np.zeros((2, 1)), {reader.HEADLINE_FRACTION: 17}, []

    monkeypatch.setattr(reader, "load_for", load)
    monkeypatch.setattr(reader, "fit_out_of_fold", lambda *a: {})
    monkeypatch.setattr(reader, "capability", lambda *a: dict(
        fraction=.75, requested=100, scored=100, folds_unavailable=[], complete=True))
    monkeypatch.setattr(reader, "read_model", lambda *a: dict(model=a[0], cells=[]))
    return ["--record", str(record), "--captures", str(captures),
            "--out", str(tmp_path / "out.json")], calls


@pytest.mark.parametrize("count", [0, -1, 1, 9999, 10001])
def test_main_refuses_unsealed_resample_count_before_loading(reader, admission, count):
    argv, calls = admission
    with pytest.raises(reader.Refused, match="10,?000"):
        reader.main([*argv, "--bootstrap-resamples", str(count)])
    assert calls == []


def test_main_accepts_declared_count(reader, admission):
    argv, calls = admission
    assert reader.main(argv) == 0
    assert calls == ["load", "load"]


def test_missing_capability_has_named_refusal(reader, admission, monkeypatch):
    argv, calls = admission
    monkeypatch.setattr(reader, "capability", lambda *a: dict(
        fraction=None, requested=0, scored=0, folds_unavailable=[0], complete=False))
    with pytest.raises(reader.Refused, match="gate fails"):
        reader.main(argv)


@pytest.mark.parametrize("cross", [np.array([[3., -3., 0.], [0., 0., 1.], [0., 0., 0.]]),
    np.array([[3., -3.], [1., 1.]]), np.pad([[3., -3.], [1., 1.]], ((0, 62), (0, 62))),
    np.array([[1. + 1e-8, -(1. + 1e-8)], [1., 1.]])])
def test_nonleading_or_near_degenerate_triplet_cannot_bypass_reference(cross):
    candidate = transport_certified.leading_triplet(cross)[3]
    assert not candidate["certified"]
    direction, sigma, certificate = transport_certified.leading_direction(cross)
    wanted, value = transport._leading_direction(cross)
    assert certificate["fell_back_to_full_svd"]
    assert np.array_equal(direction, wanted)
    assert sigma == value


@pytest.mark.parametrize("cross", [np.array([[3., -3.], [1., 1.]]),
                                    np.eye(2)*8e-13])
def test_full_fitter_reproduces_reference_direction_and_rank_floor(cross):
    x = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    y = x @ cross / 4
    expected = transport.fit(x, y, 2)
    result, certificates = transport_certified.certified_fit(x, y, 2)
    assert result.max_rank == expected.max_rank
    assert np.array_equal(result.weights, expected.weights)
    assert np.array_equal(result.loadings, expected.loadings)
    assert np.array_equal(result.target_loadings, expected.target_loadings)
    assert result.leading_singular_values == expected.leading_singular_values


def test_reference_path_preserves_floating_point_fitter_arithmetic():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 6))
    y = x + .03*rng.normal(size=x.shape)
    expected = transport.fit(x, y, 3)
    result, certificates = transport_certified.certified_fit(x, y, 3)
    assert np.array_equal(result.coefficients(3), expected.coefficients(3))
    assert np.array_equal(result.apply(x, rank=3), expected.apply(x, rank=3))
    assert all(c["fell_back_to_full_svd"] for c in certificates)


def test_nonfinite_cross_is_refused_before_decomposition():
    with pytest.raises(ValueError, match="finite"):
        transport_certified.leading_direction(np.array([[np.nan, 0.], [0., 1.]]))


def test_cost_estimate_refuses_incomplete_bootstrap(reader, admission, monkeypatch):
    argv, calls = admission
    features, moves, folds, strata = fixture_moves()
    subset = moves[:3]
    strata = {m["task_id"]: ("train", m["task_id"], "v") for m in subset}
    incomplete = reader.refitting_bootstrap(features, subset, subset, folds, strata,
                                           rank=1, resamples=100, seed=1)
    assert not incomplete["complete"] and incomplete["attempted_resamples"] == 1
    # Isolate timing-mode accounting; the completion record above comes from real numerical refits.
    monkeypatch.setattr(reader, "refitting_bootstrap", lambda *a, **kw: incomplete)
    original_seal = reader.require_seal
    monkeypatch.setattr(reader, "require_seal", lambda *a, **kw: dict(
        original_seal(*a, **kw), seed=20260910))
    with pytest.raises(reader.Refused, match="cost estimate unavailable"):
        reader.main([*argv, "--estimate-bootstrap", "100"])


def test_cost_estimate_accepts_completed_bootstrap_without_writing_reading(
        reader, admission, monkeypatch):
    argv, calls = admission
    features, moves, folds, strata = fixture_moves()
    strata = {m["task_id"]: ("train", m["task_id"], "v") for m in moves}
    complete = reader.refitting_bootstrap(features, moves, moves, folds, strata,
                                         rank=1, resamples=2, seed=1)
    assert complete["complete"]
    monkeypatch.setattr(reader, "refitting_bootstrap", lambda *a, **kw: complete)
    original_seal = reader.require_seal
    monkeypatch.setattr(reader, "require_seal", lambda *a, **kw: dict(
        original_seal(*a, **kw), seed=20260910))
    assert reader.main([*argv, "--estimate-bootstrap", "2"]) == 0
    assert not Path(argv[argv.index("--out") + 1]).exists()
