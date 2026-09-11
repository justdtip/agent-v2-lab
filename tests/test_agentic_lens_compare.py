import copy
import hashlib
import json

import numpy as np
import pytest

from local_llm_lab.pipeline.lens_fitting.agentic_compare import (
    compare,
    directional_check,
    paired_metrics,
)


def row(i=0, episode="episode-a"):
    return {
        "row": i,
        "position": "P_act",
        "layer": 24,
        "episode": episode,
        "expert_token_id": 11,
        "tool_token_ids": [10, 11, 12, 13, 14, 15],
        "model": {
            "six": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "mass": 0.9,
            "top_token_id": 10,
            "top_margin": 3.0,
        },
        "prose": {"six": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "mass": 0.0001, "top_token_id": 10},
        "agentic": {"six": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "mass": 0.5, "top_token_id": 10},
    }


def test_coverage_is_not_agreement_and_expert_is_not_model():
    result = paired_metrics([row()])
    cell = result["rows"][0]
    assert cell["prose"]["six_agreement"] is None
    assert cell["prose"]["full_argmax_agreement"] is True
    assert cell["agentic"]["six_agreement"] is True
    assert cell["agentic"]["expert_argmax_agreement"] is False
    group = result["groups"]["L24/P_act"]
    assert group["prose"]["coverage"] == 0
    assert group["prose"]["agreement_of_resolved"] is None
    assert group["agentic"]["coverage"] == 1
    assert group["joint_resolved"] == 0
    assert result["perturbation"]["status"] == "not measured"


def test_missing_full_argmax_does_not_become_six_winner():
    value = row()
    del value["model"]["top_token_id"]
    result = paired_metrics([value])
    assert result["rows"][0]["agentic"]["full_argmax_agreement"] is None


def test_zero_tool_mass_stays_unresolved_without_fabricating_a_winner():
    value = row()
    value["agentic"].update(mass=0.0, six=[0.0] * 6)
    result = paired_metrics([value])
    assert result["rows"][0]["agentic"]["six_token_id"] is None
    assert result["groups"]["L24/P_act"]["agentic"]["coverage"] == 0


def test_episode_distributions_preserve_tail_and_equal_episode_weight():
    rows = [row(i) for i in range(4)] + [row(4, "episode-b")]
    rows[-1]["agentic"]["six"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    result = paired_metrics(rows)
    summary = result["groups"]["L24/P_act"]["agentic"]
    assert summary["agreement_of_resolved"] == 0.8
    assert summary["episode_agreement"]["median"] == 0.5
    assert summary["episode_agreement"]["minimum"] == 0
    assert summary["episode_agreement"]["maximum"] == 1
    assert summary["episode_agreement"]["share_at_zero"] == 0.5


@pytest.mark.parametrize("bad", ["duplicate", "mass", "six", "tokens", "empty"])
def test_invalid_population_refuses(bad):
    values = [row()]
    if bad == "duplicate":
        values.append(copy.deepcopy(values[0]))
    elif bad == "empty":
        values = []
    elif bad == "mass":
        values[0]["prose"]["mass"] = float("nan")
    elif bad == "six":
        values[0]["model"]["six"] = [1, 1, 0, 0, 0, 0]
    else:
        values[0]["tool_token_ids"] = [10] * 6
    with pytest.raises(ValueError):
        paired_metrics(values)


def test_directional_check_distinguishes_exact_from_bad_lens_and_same_state():
    import torch

    h = torch.tensor([2.0, 3.0], dtype=torch.float32)
    direction = torch.tensor([0.1, -0.2], dtype=torch.float32)
    matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    result = directional_check(lambda x: matrix @ x, h, direction, np.eye(2), baseline=matrix @ h)
    assert result["same_state_equal"]
    assert result["exact_vs_actual"]["relative_error"] < 1e-5
    assert result["lens_vs_actual"]["relative_error"] > 0.5
    with pytest.raises(ValueError, match="same-state"):
        directional_check(lambda x: matrix @ x, h, direction, np.eye(2), baseline=matrix @ h + 1)


@pytest.mark.parametrize("mutation", [None, "missing", "pairing", "identity", "nan_limit"])
def test_file_comparison_refuses_incomplete_unpaired_and_misidentified_rows(
    tmp_path, monkeypatch, mutation
):
    from local_llm_lab.probes import measure_pairings, sae_bridge

    identity = {"fixture": "independent-pair"}
    prov = {
        "lens_sha256": "lens",
        "nu_sha256": "nu",
        "corpus_sha256": "corpus",
        "positions_sha256": "positions",
    }
    cells = [
        {
            "cell": {"row": 0, "position": "P_act", "token_id": 11},
            "prompt_identity": {"task_id": "episode-a"},
            "identity": identity,
        }
    ]
    score_provenance = dict(prov)
    prov.update(
        files_sha256={"index": "index"},
        checkpoint_sha256={"model": "model"},
        lens_sidecar_sha256="sidecar",
        fit_widths=[{"forward_batch": 1}],
    )
    cells[0].update(
        input_ids=[1, 2, 3],
        prompt_sha256="prompt",
        residuals={24: np.ones(2, np.float32)},
        capture_forward={"output_attentions": False, "logits_to_keep": [1, 2]},
    )
    monkeypatch.setattr(
        measure_pairings,
        "_admit",
        lambda _: (cells, {"model": {"base": "tiny"}, "fit_widths": prov["fit_widths"]}, prov),
    )
    monkeypatch.setattr(sae_bridge, "_supplied_pairing_table", lambda table: table)
    table = {
        "tiny": {
            "measured_pairings": {"24": [{"pair": identity, "relative": 0.0, "basis": "fixture"}]}
        }
    }
    table["_provenance"] = dict(
        prov,
        source_commit="a" * 40,
        implementation_sha256={"producer.py": "b" * 64},
        measurements={
            "24": [
                {
                    "cell": cells[0]["cell"],
                    "capture_forward": cells[0]["capture_forward"],
                    "prompt_identity": cells[0]["prompt_identity"],
                    "prompt_sha256": "prompt",
                    "input_ids_sha256": hashlib.sha256(b"[1,2,3]").hexdigest(),
                    "capture_residual_sha256": hashlib.sha256(
                        cells[0]["residuals"][24].tobytes()
                    ).hexdigest(),
                    "chunk_terms": [{"forward_batch": 1, "relative": 0.0}],
                }
            ]
        },
    )
    if mutation == "pairing":
        table["tiny"]["measured_pairings"] = {}
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps(table))
    rows = [] if mutation == "missing" else [row()]
    if mutation == "identity":
        rows[0]["episode"] = "wrong-episode"
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "readout_convention": "model-unembed-final-norm-and-softcap",
                "provenance": {"prose": score_provenance, "agentic": score_provenance},
                "rows": rows,
            }
        )
    )
    args = {
        key: str(tmp_path / key)
        for key in (
            "model_snapshot",
            "checkpoint_manifest",
            "capture_dir",
            "positions",
            "corpus",
            "output",
        )
    }
    args.update(
        layers=[24],
        scores=str(scores),
        scores_sha256=hashlib.sha256(scores.read_bytes()).hexdigest(),
        maximum_pairing_relative=float("nan") if mutation == "nan_limit" else 0.01,
    )
    for name in ("prose", "agentic"):
        args[name] = {"lens_archive": "archive", "lens_sidecar": "sidecar", "pairings": str(pairs)}
    if mutation:
        with pytest.raises(ValueError):
            compare(args)
        assert not (tmp_path / "output").exists()
    else:
        assert compare(args)["groups"]["L24/P_act"]["agentic"]["coverage"] == 1
        assert (tmp_path / "output/comparison.json").is_file()
