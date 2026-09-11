import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from local_llm_lab.pipeline.lens_fitting.agentic_directions import (
    freeze_directions,
    load_directions,
    matched_controls,
    run_directions,
)


def fixture():
    from test_torch_capture import fixture as capture_fixture

    view, _, ids = capture_fixture()
    matrix = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])

    class Linear(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states @ matrix.T

    view.model.layers[1] = Linear()
    view.input_device = torch.device("cpu")
    view.model.requires_grad_(False)
    h = (view.model.embed(ids) * 2)[0, 1].detach().numpy()
    item = {
        "cell": {"row": 7, "position": "P_act", "token_index": 1},
        "input_ids": ids[0].tolist(),
        "residuals": {1: h},
        "capture_forward": {"output_attentions": False, "logits_to_keep": [1]},
        "prompt_identity": {"task_id": "fixture-episode"},
    }
    # Accept the same real capture flags as the production native forward.
    native = view.model.forward

    def forward(input_ids, output_attentions=False, logits_to_keep=None):
        return native(input_ids)

    view.model.forward = forward
    lenses = {
        "prose": SimpleNamespace(maps={1: matrix.T.numpy()}),
        "agentic": SimpleNamespace(maps={1: matrix.numpy()}),
    }
    return view, item, lenses


def test_native_capture_directional_forward_separates_wrong_orientation():
    view, item, lenses = fixture()
    delta = np.array([0.01, 0.02, 0.03], np.float32)
    directions = {
        (7, "P_act", 1, arm): delta.copy()
        for arm in ("dictionary_error", "norm_random", "angle_random")
    }
    rows = run_directions(view, [item], lenses, directions)
    assert len(rows) == 3
    for row in rows:
        assert row["same_state_equal"]
        assert row["agentic"]["lens_vs_exact"]["relative_error"] < 1e-6
        assert row["prose"]["lens_vs_exact"]["relative_error"] > 0.1
        assert row["exact_vs_actual"]["relative_error"] < 1e-4
    assert not any(layer._forward_hooks for layer in view.layers)


def test_source_identity_checked_before_derivative():
    view, item, lenses = fixture()
    item["residuals"][1] = item["residuals"][1] + 1
    directions = {(7, "P_act", 1, "dictionary_error"): np.ones(3, np.float32)}
    with pytest.raises(ValueError, match="source residual"):
        run_directions(view, [item], lenses, directions)


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "coverage",
        "hash",
        "base",
        "provenance",
        "duplicate",
        "norm",
        "angle",
        "dtype",
        "nonfinite",
    ],
)
def test_frozen_direction_loader_binds_population_and_vectors(tmp_path, damage):
    _, item, _ = fixture()
    d = np.array([0.01, 0.02, 0.03], np.float32)
    arms = ("dictionary_error", "norm_random", "angle_random")
    arrays = {arm: d.copy() for arm in arms}
    if damage == "norm":
        arrays["norm_random"] *= 2
    elif damage == "angle":
        arrays["angle_random"] *= -1
    elif damage == "dtype":
        arrays["norm_random"] = d.astype(np.float64)
    elif damage == "nonfinite":
        arrays["norm_random"][0] = np.nan
    archive = tmp_path / "directions.npz"
    np.savez(archive, **arrays)
    dictionary = tmp_path / "dictionary"
    dictionary.write_bytes(b"frozen dictionary fixture")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    provenance = {
        "corpus_sha256": "corpus",
        "positions_sha256": "positions",
        "checkpoint_sha256": {"config": "checkpoint"},
        "files_sha256": {"capture": "capture"},
    }
    manifest = {
        "schema_version": 1,
        "npz": str(archive),
        "npz_sha256": sha(archive),
        "provenance": provenance,
        "dictionary_files": {str(dictionary): sha(dictionary)},
        "arms": list(arms),
        "control_definition": "frozen fixture directions",
        "records": [
            {
                "row": 7,
                "position": "P_act",
                "layer": 1,
                "arm": arm,
                "array": arm,
                "base_residual_sha256": hashlib.sha256(item["residuals"][1].tobytes()).hexdigest(),
            }
            for arm in arms
        ],
    }
    if damage == "coverage":
        manifest["records"].pop()
    elif damage == "hash":
        manifest["npz_sha256"] = "0" * 64
    elif damage == "base":
        manifest["records"][0]["base_residual_sha256"] = "0" * 64
    elif damage == "provenance":
        manifest["provenance"] = {"wrong": "corpus"}
    elif damage == "duplicate":
        manifest["records"].append(manifest["records"][0])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    if damage:
        with pytest.raises(ValueError):
            load_directions(path, [item], [1], provenance)
    else:
        assert len(load_directions(path, [item], [1], provenance)[0]) == 3


def test_keyed_controls_preserve_norm_angle_and_order():
    base = np.array([1.0, 2.0, 3.0], np.float32)
    error = np.array([0.03, -0.01, 0.02], np.float32)
    key = (7, "P_act", 1)
    first = matched_controls(base, error, key)
    matched_controls(base, error, (8, "P_note", 3))
    repeated = matched_controls(base, error, key)
    for arm in first:
        np.testing.assert_array_equal(first[arm], repeated[arm])
        assert np.linalg.norm(first[arm]) == pytest.approx(np.linalg.norm(error), rel=1e-6)
    assert np.dot(first["angle_random"], base) == pytest.approx(np.dot(error, base), rel=1e-6)
    assert not np.array_equal(first["norm_random"], first["angle_random"])
    assert all(not np.any(v) for v in matched_controls(base, np.zeros(3), key).values())


def test_freeze_uses_existing_encoder_decoder_and_declares_single_draw(tmp_path):
    from local_llm_lab.probes.sae_bridge import JumpReLUDictionary

    _, item, _ = fixture()
    dictionary = JumpReLUDictionary(
        w_enc=np.zeros((3, 2), np.float32),
        b_enc=np.zeros(2, np.float32),
        threshold=np.ones(2, np.float32),
        w_dec=np.zeros((2, 3), np.float32),
        b_dec=np.array([0.1, 0.2, 0.3], np.float32),
        config={},
        sha256="fixture",
        source="fixture",
    )
    receipt = tmp_path / "dictionary.receipt"
    receipt.write_bytes(b"fixture")
    provenance = {
        "corpus_sha256": "corpus",
        "positions_sha256": "positions",
        "checkpoint_sha256": {},
        "files_sha256": {},
    }
    path = tmp_path / "frozen.json"
    manifest = freeze_directions(
        [item],
        [1],
        {1: dictionary},
        path,
        provenance,
        {str(receipt): hashlib.sha256(receipt.read_bytes()).hexdigest()},
    )
    assert manifest["seed"] == 20260911
    assert manifest["draws_per_control_per_cell"] == 1
    directions, _ = load_directions(path, [item], [1], provenance)
    np.testing.assert_array_equal(
        directions[(7, "P_act", 1, "dictionary_error")], dictionary.b_dec - item["residuals"][1]
    )
