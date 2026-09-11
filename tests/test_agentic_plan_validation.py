"""Calibration evidence must survive JSON and authorize only the measured recipe."""

import copy
import json

import numpy as np
import pytest

from local_llm_lab.pipeline.lens_fitting.agentic import (
    atomic_json,
    canonical,
    check_exactness,
    freeze_plan,
    sha,
    validate_plan,
)


def test_exactness_json_and_nonfinite_reference():
    good = {17: np.eye(2, dtype=np.float32), 23: np.eye(2, dtype=np.float32)}
    json.loads(canonical(check_exactness(good, good)))
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            check_exactness(
                {17: np.array([[value]], dtype=np.float32)},
                {17: np.zeros((1, 1), dtype=np.float32)},
            )


@pytest.fixture
def ready(tmp_path):
    rows = [dict(index=i, task_id=f"train-{i}", family="f", ids=[1] * (20 + i)) for i in range(3)]
    measurements = [
        dict(
            index=i,
            context_tokens=20 + i,
            seconds=1.0,
            peak_gib=1.0,
            dim_batch=1,
            forward_batch=1,
            layers=[18, 24],
        )
        for i in range(3)
    ]
    binding = {"corpus_sha256": "frozen"}
    plan = freeze_plan(
        rows, measurements, binding=binding, dim_batch=1, budget_seconds=100, peak_limit_gib=2
    )
    good = {17: np.eye(2, dtype=np.float32), 23: np.eye(2, dtype=np.float32)}
    checks = check_exactness(good, good)
    receipt = tmp_path / "calibration.json"
    atomic_json(
        receipt,
        dict(
            binding=binding,
            exactness=checks,
            measurements=measurements,
            recipe=dict(dim_batch=1, budget_seconds=100, peak_limit_gib=2, layers=[18, 24]),
        ),
    )
    plan.update(exactness=checks, calibration_sha256=sha(receipt))
    return rows, binding, plan, receipt


def test_verified_plan(ready):
    rows, binding, plan, receipt = ready
    validate_plan(plan, rows, binding=binding, peak_limit_gib=2, calibration_path=receipt)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dim_batch", 2),
        ("layers", [18]),
        ("budget_seconds", 101),
        ("rows", [0]),
        ("max_seq_len", 100),
        ("exactness", {"passed": False}),
        ("calibration_sha256", "bad"),
    ],
)
def test_changed_plan_refuses(ready, field, value):
    rows, binding, plan, receipt = ready
    altered = copy.deepcopy(plan)
    altered[field] = value
    with pytest.raises(ValueError):
        validate_plan(altered, rows, binding=binding, peak_limit_gib=2, calibration_path=receipt)


def test_changed_measurement_schedule_refuses(ready):
    rows, binding, plan, receipt = ready
    plan["calibration"][0]["dim_batch"] = 2
    with pytest.raises(ValueError, match="calibration|schedule"):
        validate_plan(plan, rows, binding=binding, peak_limit_gib=2, calibration_path=receipt)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seconds", float("nan")),
        ("seconds", float("inf")),
        ("peak_gib", -1),
        ("context_tokens", 30),
        ("dim_batch", 2),
        ("forward_batch", 2),
    ],
)
def test_invalid_calibration(ready, field, value):
    rows, binding, plan, receipt = ready
    measurements = copy.deepcopy(plan["calibration"])
    measurements[0][field] = value
    with pytest.raises(ValueError, match="calibration"):
        freeze_plan(
            rows, measurements, binding=binding, dim_batch=1, budget_seconds=100, peak_limit_gib=2
        )


def test_cli_bad_plan_refuses_before_lock_or_load(ready, tmp_path, monkeypatch):
    from local_llm_lab import hf_text, runlock
    from local_llm_lab.pipeline.lens_fitting import agentic, corpus

    rows, binding, plan, receipt = ready
    snapshot = tmp_path / "models--google--gemma-3-4b-it" / "snapshots" / "fixture"
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}")
    cp = tmp_path / "corpus.json"
    cp.write_text(
        json.dumps({"sources": {"tokenizer": {"sha256": sha(snapshot / "tokenizer.json")}}})
    )
    pp = tmp_path / "plan.json"
    plan["dim_batch"] = 2
    atomic_json(pp, plan)
    monkeypatch.setattr(corpus, "read_corpus", lambda _: [r | {"domain": "agentic"} for r in rows])
    monkeypatch.setattr(hf_text, "checkpoint_metadata", lambda _: {"sha256": {}})
    monkeypatch.setattr(agentic, "source_binding", lambda *a: binding)
    monkeypatch.setattr(runlock, "hold_model_run_lock", lambda **k: pytest.fail("lock acquired"))
    monkeypatch.setattr(hf_text, "load_text_causal_lm", lambda *a, **k: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="recipe"):
        agentic.main(
            [
                "fit",
                "--corpus",
                str(cp),
                "--snapshot",
                str(snapshot),
                "--output",
                str(tmp_path / "out"),
                "--plan",
                str(pp),
                "--peak-limit-gib",
                "2",
            ]
        )
