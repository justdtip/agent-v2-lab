"""Agentic pilot: real estimator equivalence and durable bounded execution."""

import json

import numpy as np
import pytest
from test_lens_upstream import SEQ_LEN, TinyLensModel, make_rows

from local_llm_lab.pipeline.lens_fitting import upstream as U


def test_graph_once_adapter_preserves_estimator_and_declares_actual_width():
    up = U.load_upstream()
    rows = make_rows(1)
    reference = U.fit_upstream_jacobian(
        TinyLensModel(),
        rows,
        source_layers=[1, 2],
        max_seq_len=SEQ_LEN,
        dim_batch=1,
        upstream=up,
    )
    actual = U.fit_upstream_jacobian(
        TinyLensModel(),
        rows,
        source_layers=[1, 2],
        max_seq_len=SEQ_LEN,
        dim_batch=3,
        upstream=up,
        estimator_schedule="graph-once",
    )
    for layer in reference.jacobians:
        np.testing.assert_allclose(
            actual.jacobians[layer], reference.jacobians[layer], rtol=3e-6, atol=3e-7
        )
    assert actual.precision["forward_batch"] == actual.precision["anchor_batch"] == 1
    assert actual.selector["dim_batch"] == 3
    assert U.declare_nu(actual, num_layers=4, corpus={})["estimator_schedule"] == "graph-once"


def test_unknown_schedule_refuses():
    with pytest.raises(ValueError, match="schedule"):
        U.fit_upstream_jacobian(
            TinyLensModel(), make_rows(1), max_seq_len=SEQ_LEN, estimator_schedule="invented"
        )


def test_resume_uses_each_completed_row_once_and_refuses_changed_contract(tmp_path):
    from local_llm_lab.pipeline.lens_fitting.agentic import RowStore

    contract = {"rows": [2, 5], "layers": [2], "hidden": 2, "seed": 0}
    store = RowStore(tmp_path, contract)
    store.put(2, {2: np.ones((2, 2), np.float32)}, {"seconds": 1})
    resumed = RowStore(tmp_path, contract)
    assert resumed.completed == [2]
    resumed.put(5, {2: np.full((2, 2), 3, np.float32)}, {"seconds": 1})
    np.testing.assert_array_equal(resumed.mean()[2], np.full((2, 2), 2, np.float32))
    with pytest.raises(ValueError, match="contract"):
        RowStore(tmp_path, contract | {"seed": 1})
    with pytest.raises(ValueError, match="already"):
        resumed.put(5, {2: np.full((2, 2), 9, np.float32)}, {})
    (tmp_path / "row-2.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        RowStore(tmp_path, contract)


def test_budget_selection_is_family_complete_and_episode_unique():
    from local_llm_lab.pipeline.lens_fitting.agentic import select_sample

    rows = [
        {"index": i, "task_id": f"train-{i}", "family": f"f{i % 3}", "ids": [1] * (40 + i)}
        for i in range(12)
    ]
    got = select_sample(rows, seconds_per_row=100, budget_seconds=800)
    assert len(got) == 6  # 20% headroom, complete rounds across families
    assert {r["family"] for r in got} == {"f0", "f1", "f2"}
    assert len({r["task_id"] for r in got}) == len(got)
    assert got == select_sample(list(reversed(rows)), seconds_per_row=100, budget_seconds=800)
    with pytest.raises(ValueError, match="family"):
        select_sample(rows, seconds_per_row=1000, budget_seconds=800)


def test_real_fit_resumes_and_matches_uninterrupted_mean(tmp_path):
    from local_llm_lab.pipeline.lens_fitting.agentic import fit_selected

    rows = make_rows(2)
    args = dict(
        binding={"checkpoint": "tiny"},
        layers=[2, 3],
        dim_batch=2,
        max_seq_len=SEQ_LEN,
        budget_seconds=1000,
    )
    done = fit_selected(TinyLensModel(), rows, tmp_path / "one", **args)
    again = fit_selected(TinyLensModel(), rows, tmp_path / "one", **args)
    reference = U.fit_upstream_jacobian(
        TinyLensModel(), rows, source_layers=[1, 2], dim_batch=1, max_seq_len=SEQ_LEN
    )
    assert done["status"] == again["status"] == "complete"
    with np.load(tmp_path / "one" / "exact-maps.npz") as stored:
        for layer in [2, 3]:
            np.testing.assert_allclose(
                stored[f"J{layer}"], reference.jacobians[layer - 1], atol=3e-7, rtol=3e-6
            )
    nu = json.loads((tmp_path / "one" / "nu.json").read_text())
    assert nu["pair_weighting"]["n_prompts"] == 2
    assert nu["precision"]["forward_batch"] == 1
    assert nu["endpoint"]["source_layers_repo"] == [2, 3]


def test_fit_refuses_truncation_before_creating_store(tmp_path):
    from local_llm_lab.pipeline.lens_fitting.agentic import fit_selected

    with pytest.raises(ValueError, match="truncat"):
        fit_selected(
            TinyLensModel(),
            make_rows(1),
            tmp_path / "bad",
            binding={},
            layers=[2],
            dim_batch=1,
            max_seq_len=10,
            budget_seconds=1,
        )
    assert not (tmp_path / "bad").exists()


def test_calibration_gate_cannot_pass_wrong_maps_or_missing_lengths():
    from local_llm_lab.pipeline.lens_fitting.agentic import check_exactness, freeze_plan

    good = {1: np.eye(2, dtype=np.float32)}
    assert check_exactness(good, good)["passed"]
    with pytest.raises(ValueError, match="exactness"):
        check_exactness(good, {1: -good[1]})
    rows = [{"index": i, "task_id": str(i), "family": "f", "ids": [0] * (20 + i)} for i in range(3)]
    with pytest.raises(ValueError, match="calibration"):
        freeze_plan(
            rows,
            [{"index": 0, "seconds": 1}],
            binding={},
            dim_batch=1,
            budget_seconds=100,
            peak_limit_gib=1,
        )
