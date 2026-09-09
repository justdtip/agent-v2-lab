"""The script end to end on fixtures: the seal's refusals, resume, and every artefact §6 names."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from local_llm_lab.pipeline.state_programme import run as sp
from local_llm_lab.pipeline.state_programme.family import make_existence_pairs
from local_llm_lab.pipeline.state_programme.record import ARTEFACTS, read_rows
from local_llm_lab.pipeline.state_programme.seal import (
    AlreadyRunning,
    NotSealed,
    refuse_rederive,
    require_seal,
)

ROOT = Path(__file__).resolve().parents[1]


def _cli():
    spec = importlib.util.spec_from_file_location(
        "state_programme_cli", ROOT / "scripts/state_programme.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_run(out: Path, **extra) -> int:
    argv = ["--out", str(out), "--decoding", "greedy", "--fixture", "--pilot-pairs", "5",
            "--main-pairs", "3", "--fault-rate", "0.4", "--hours-bought", "100"]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return _cli().main(argv)


def test_the_whole_script_runs_on_fixtures_and_writes_every_artefact(tmp_path) -> None:
    out = tmp_path / "rec"
    assert _fixture_run(out) == 0
    for name in ARTEFACTS:
        assert (out / name).exists(), name
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["fixture"] is True and manifest["decoding"]["mode"] == "greedy"
    assert "context-averaged" in manifest["decoding"]["estimand"]
    assert manifest["faults"] == {"rate": 0.4, "seed": 7}
    assert manifest["device"]["python"] and len(manifest["tree_content_digest"]) == 64


def test_rate_is_measured_here_and_the_pilot_has_both_arms_and_a_reliability_arm(tmp_path):
    out = tmp_path / "rec"
    _fixture_run(out)
    rate = json.loads((out / "rate.json").read_text())
    rows = read_rows(out / "pilot" / "rows.jsonl")
    assert rate["basis"] == "measured-here" and rate["mode"] == "greedy"
    assert rate["episodes"] == len(rows) == 5 * 2 + 5 * 2
    assert {r["arm"] for r in rows} == {"E", "A", "RE", "RA"}
    assert any(r["falsified"] for r in rows if r["arm"].startswith("R"))
    # A falsified exists-arm episode was told the file is absent, and the scripted expert still
    # tried to read it: the diagnostics see a contradiction, so D7 is scorable there.
    contradicted = [r for r in rows if r["arm"] == "RE" and r["falsified"]]
    assert contradicted and all(r["scores"]["D7"] is not None for r in contradicted)


def test_the_seal_carries_the_derived_table_and_the_pilot_digest(tmp_path) -> None:
    out = tmp_path / "rec"
    _fixture_run(out)
    seal = require_seal(out)
    t = seal["tolerances"]
    assert set(t["retained"]) <= {f"D{i}" for i in range(1, 10)} and t["M"] == len(t["retained"])
    assert "D1" in t["retained"] and "D5" in t["retained"], "the expert's contrast is exact"
    assert t["epsilon_sub"] == t["d_min"] / 4 and t["n"] >= 1
    assert seal["derived_from"]["pilot_row_count"] == 20
    assert len(seal["derived_from"]["pilot_rows_sha256"]) == 64
    assert seal["budget"]["reason"]


def test_the_main_stage_refuses_without_a_seal_and_rederivation_refuses_with_main_rows(tmp_path):
    out = tmp_path / "rec"
    (out / "main").mkdir(parents=True)
    with pytest.raises(NotSealed, match="does not start"):
        sp.main_run(out, manifest={}, pairs=[], policy=sp.ScriptedPolicy(), wrapper=lambda *a: {})
    (out / "main" / "rows.jsonl").write_text('{"resume_key": "x"}\n')
    with pytest.raises(AlreadyRunning, match="not re-derived"):
        refuse_rederive(out)


def test_resume_skips_matching_rows_and_refuses_a_changed_tree_by_naming_the_field(tmp_path):
    out = tmp_path / "rec"
    _fixture_run(out)
    manifest = json.loads((out / "manifest.json").read_text())
    before = read_rows(out / "main" / "rows.jsonl")
    policy = sp.ScriptedPolicy()
    pairs = make_existence_pairs("main", 3, 1, seed="20260910:main")
    wrapper = sp.scripted_wrapper(policy, pairs)
    sp.main_run(out, manifest=manifest, pairs=pairs, policy=policy, wrapper=wrapper)
    assert read_rows(out / "main" / "rows.jsonl") == before, "nothing re-run"
    changed = dict(manifest, checkpoint_digest="other")
    with pytest.raises(RuntimeError, match=r"differ: \['checkpoint_digest'\]"):
        sp.main_run(out, manifest=changed, pairs=pairs, policy=policy, wrapper=wrapper)


def test_estimands_are_distances_with_tolerances_and_the_stub_says_it_is_a_stub(tmp_path) -> None:
    out = tmp_path / "rec"
    _fixture_run(out)
    e = json.loads((out / "estimands.json").read_text())
    assert set(e["estimands"]) == {"substitution", "specificity", "reuse", "predictive", "dynamic"}
    keys = {"distance", "tolerance", "rows", "measured", "untestable", "passes"}
    assert all(keys <= set(v) for v in e["estimands"].values())
    assert "coefficient table" in e["level"]
    rows = read_rows(out / "main" / "rows.jsonl")
    assert all(r.get("wrapper") == "stub" for r in rows if r["condition"] != "base")


def test_the_main_run_carries_the_reliability_arm_so_a_contradiction_can_exist(tmp_path) -> None:
    """Edit 2, part one: without this no main row carried a contradiction and the predictive
    estimand was reported from no rows as a pass that could not fail."""
    out = tmp_path / "rec"
    _fixture_run(out)
    rows = read_rows(out / "main" / "rows.jsonl")
    arms = {r["arm"] for r in rows if r["condition"] == "base"}
    assert arms == {"E", "A", "RE", "RA"}
    base = [r for r in rows if r["condition"] == "base"]
    assert any(r["falsified"] for r in base if r["arm"] in ("RE", "RA"))
    assert all(r["falsified"] is False for r in base if r["arm"] in ("E", "A"))


def test_a_zero_tolerance_is_untestable_and_a_missing_row_is_not_measured_never_a_pass(tmp_path):
    """Edit 2, parts two and three. On a scripted policy the pilot's update never varies, so
    epsilon_pred is exactly zero and the predictive estimand must say untestable, not pass."""
    out = tmp_path / "rec"
    _fixture_run(out)
    e = json.loads((out / "estimands.json").read_text())["estimands"]
    assert e["predictive"]["tolerance"] == 0.0
    assert e["predictive"]["untestable"] is True and e["predictive"]["passes"] is None
    assert "never varied" in e["predictive"]["reason"]
    # Nothing in the record is a pass at zero, and no unmeasured estimand is a pass.
    for name, est in e.items():
        if est["tolerance"] == 0.0 or not est["measured"]:
            assert est["passes"] is None, name
    from local_llm_lab.pipeline.state_programme.run import _estimand
    empty = _estimand(None, 0.25, rows=0)
    assert empty == {"distance": None, "tolerance": 0.25, "rows": 0, "measured": False,
                     "untestable": False, "passes": None}
    assert _estimand(0.1, 0.25, rows=3)["passes"] is True
    assert _estimand(0.3, 0.25, rows=3)["passes"] is False


def test_read_rows_refuses_a_partial_trailing_line_by_number(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n{"c": 3, "trunc')
    with pytest.raises(ValueError, match=r"rows.jsonl:3: partial or corrupt row"):
        read_rows(path)


def test_a_non_fixture_preflight_refuses_an_unpinned_device(tmp_path, monkeypatch) -> None:
    """The device driver's third obligation: device.pin() precedes preflight."""
    from local_llm_lab import device

    monkeypatch.setattr(device, "describe", lambda: {"determinism": "UNPINNED", "python": "x"})
    inputs = dict(registry_name="r", checkpoint_digest="d", lens_identity={"base": "b"},
                  dictionary_layers={"1": "h"}, wrapper_version="w", decoding="greedy",
                  temperature=None, seed=1, fault_rate=0.1, fault_seed=1)
    with pytest.raises(ValueError, match="UNPINNED"):
        sp.preflight(root=ROOT, fixture=False, **inputs)
    assert sp.preflight(root=ROOT, fixture=True, **inputs)["device"]["determinism"] == "UNPINNED"


def test_the_readme_is_written_from_the_files_and_names_each_ones_source(tmp_path) -> None:
    out = tmp_path / "rec"
    _fixture_run(out)
    text = (out / "README.md").read_text()
    for source in ARTEFACTS[:-1]:
        assert source in text
    assert "not a result" in text
    n = json.loads((out / "preregistration.json").read_text())["tolerances"]["n"]
    assert f"| `n` | {n} |" in text
    assert "untestable (degenerate tolerance)" in text


def test_sampled_decoding_is_refused_by_name_until_it_is_built(tmp_path, capsys) -> None:
    assert _cli().main(["--out", str(tmp_path / "s"), "--decoding", "sampled", "--fixture"]) == 3
    assert "temperature is unruled" in capsys.readouterr().err


def test_a_non_empty_record_directory_is_refused(tmp_path) -> None:
    out = tmp_path / "rec"
    _fixture_run(out)
    with pytest.raises(SystemExit):
        _fixture_run(out)
