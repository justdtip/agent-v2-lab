"""Instrument contracts: Director's 2026-09-07 population and refit rulings.

Selection evidence: research/records/WP12-BROADCAST-HEADS-2026-09-06/out/selection_orth.json.
These tests do not load MLX or any model checkpoint.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from local_llm_lab.pipeline.live_lens.instruments import (
    FrozenPopulation,
    LensIdentity,
    LensIdentityError,
    LensMaps,
    read_band,
)
from local_llm_lab.pipeline.live_lens.records import FutureRanks, transport_overlap
from local_llm_lab.pipeline.live_lens.validation import layer_verdict

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "research/records/WP12-BROADCAST-HEADS-2026-09-06/out/selection_orth.json"


def test_population_is_frozen_from_the_record_and_tampering_fails(tmp_path):
    population = FrozenPopulation.from_selection(SELECTION, (11, 15, 19, 23, 27))
    assert len(population.primary) == 23
    assert len(population.sensitivity) == 26
    assert set(population.sensitivity) - set(population.primary) == {(27, 15), (19, 4), (27, 13)}
    assert population.source_sha256 == hashlib.sha256(SELECTION.read_bytes()).hexdigest()
    path = tmp_path / "frozen.json"
    population.write(path)
    assert FrozenPopulation.read(path) == population
    with pytest.raises(FileExistsError):
        population.write(path)
    obj = json.loads(path.read_text())
    obj["primary"][0][1] = 99
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="hash"):
        FrozenPopulation.read(path)


def test_band_comes_from_registry_and_is_verified_against_real_layer_kinds():
    kinds = ["attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(32)]
    band = read_band(ROOT / "configs/models/qwen35-4b.yaml", kinds)
    assert band == ((12, 13), (16, 17), (19, 20), (23, 24), (27, 28))
    kinds[11] = "linear_attention"
    with pytest.raises(ValueError, match="opposite"):
        read_band(ROOT / "configs/models/qwen35-4b.yaml", kinds)


def _stamped_lens(directory, name, arrays, identity, *, in_archive=True, sidecar_sha=None):
    """A lens on disk with its identity in the archive or in a digest-bound sidecar.

    Both routes, because both exist for a reason: new lenses carry it inside the archive the
    digest covers, and the two hosted lenses converted before the field existed carry it beside
    the file so their published digests stay valid (issue 99).
    """
    path = directory / name
    payload = dict(arrays)
    if in_archive:
        payload["identity"] = np.frombuffer(
            json.dumps(identity.as_dict(), sort_keys=True).encode("utf-8"), dtype=np.uint8
        )
    np.savez(path, **payload)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if not in_archive:
        path.with_suffix(".json").write_text(
            json.dumps({"npz_sha256": sidecar_sha or sha, "model": identity.as_dict()})
        )
    return path, sha


QWEN = LensIdentity("qwen35-4b", "mlx-community/Qwen3.5-4B-MLX-4bit", 32)
GEMMA = LensIdentity("gemma3-4b", "google/gemma-3-4b-it", 34)


def test_lens_orientation_identity_and_hash_verification(tmp_path):
    path, sha = _stamped_lens(
        tmp_path, "lens.npz", {"J0": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)},
        LensIdentity("toy", "toy/toy", 2),
    )
    identity = LensIdentity("toy", "toy/toy", 2)
    lens = LensMaps.load(
        path, expected_sha256=sha, hidden_size=2, num_layers=2, identity=identity
    )
    np.testing.assert_array_equal(lens.apply(np.array([5.0, 6.0]), 1), [17.0, 39.0])
    np.testing.assert_array_equal(lens.apply(np.array([5.0, 6.0]), 2), [5.0, 6.0])
    assert lens.identity == identity
    with pytest.raises(ValueError, match="hash"):
        LensMaps.load(
            path, expected_sha256="0" * 64, hidden_size=2, num_layers=2, identity=identity
        )


def test_the_qwen_lens_is_refused_against_gemma_which_is_the_case_that_motivated_this(tmp_path):
    """Issue 99, in the shape that actually happens: same width, fewer layers, silence.

    Qwen3.5-4B and Gemma 3 4B are both 2560-dimensional. The Qwen lens covers layers 1 to 31 and
    Gemma has 34, so every check that existed passes: the digest is the file's own, the maps are
    square at the right width, and `1 <= 31 < 34` holds for every one of them. The lens loaded
    and produced ranks that looked exactly like a finding.

    Sizes here are toy and the shape is the real one: a lens whose layer count is *below* the
    target's, which is the direction the old bound could not catch. The reverse direction failed
    only by arithmetic, and a guard that catches the harmless direction is worse than no guard,
    because it reads as protection.
    """
    maps = {f"J{i}": np.eye(2, dtype=np.float16) for i in range(31)}
    path, sha = _stamped_lens(tmp_path, "qwen-lens.npz", maps, QWEN)

    with pytest.raises(LensIdentityError) as error:
        LensMaps.load(
            path, expected_sha256=sha, hidden_size=2, num_layers=34, identity=GEMMA
        )
    message = str(error.value)
    assert "qwen35-4b" in message and "gemma3-4b" in message, "both identities are named"
    assert "mlx-community/Qwen3.5-4B-MLX-4bit" in message and "google/gemma-3-4b-it" in message

    # And the same file against its own model is unchanged.
    lens = LensMaps.load(
        path, expected_sha256=sha, hidden_size=2, num_layers=32, identity=QWEN
    )
    assert set(lens.maps) == set(range(1, 32)) and lens.identity == QWEN


def test_a_lens_with_no_identity_is_refused_and_told_how_to_get_one(tmp_path):
    """Silent acceptance of an unidentified lens is the defect, so it is a refusal."""
    path = tmp_path / "bare.npz"
    np.savez(path, J0=np.eye(2, dtype=np.float16))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(LensIdentityError, match="stamp_lens_identity"):
        LensMaps.load(path, expected_sha256=sha, hidden_size=2, num_layers=2, identity=QWEN)

    path.with_suffix(".json").write_text(json.dumps({"npz_sha256": sha}))
    with pytest.raises(LensIdentityError, match="no 'model' block"):
        LensMaps.load(path, expected_sha256=sha, hidden_size=2, num_layers=2, identity=QWEN)


def test_a_sidecar_that_describes_another_file_is_not_evidence_about_this_one(tmp_path):
    """The sidecar route is only as good as its binding, so the binding is checked.

    A sidecar is editable independently of the archive it sits beside, which is exactly why it
    carries the archive's digest. Without this check the retroactive route would be a way to
    claim any identity for any file by editing one JSON key.
    """
    path, sha = _stamped_lens(
        tmp_path, "hosted.npz", {"J0": np.eye(2, dtype=np.float16)}, QWEN,
        in_archive=False, sidecar_sha="f" * 64,
    )
    with pytest.raises(LensIdentityError, match="different file"):
        LensMaps.load(path, expected_sha256=sha, hidden_size=2, num_layers=32, identity=QWEN)


def test_the_sidecar_route_loads_a_hosted_lens_whose_bytes_must_not_change(tmp_path):
    """The two lenses converted before the field existed, without invalidating their digests."""
    path, sha = _stamped_lens(
        tmp_path, "hosted.npz", {"J0": np.eye(2, dtype=np.float16)}, QWEN, in_archive=False
    )
    lens = LensMaps.load(
        path, expected_sha256=sha, hidden_size=2, num_layers=32, identity=QWEN
    )
    assert lens.identity == QWEN

    with pytest.raises(LensIdentityError, match="fitted on"):
        LensMaps.load(
            path, expected_sha256=sha, hidden_size=2, num_layers=34, identity=GEMMA
        )


def test_future_ranks_use_source_positions_and_censor_unseen_future():
    ring = FutureRanks(horizons=(1, 4, 8), capacity=16)
    # Residual at absolute position 9 predicts the token at position 10.
    ring.capture(9, {20: np.array([0.1, 0.2, 0.7])})
    assert ring.observe(10, 0) == [
        {"position": 9, "layer": 20, "horizon": 1, "token_id": 0, "rank": 3, "probability": 0.1}
    ]
    assert ring.observe(13, 1)[0]["rank"] == 2
    assert ring.observe(17, 2)[0]["rank"] == 1
    ring.reset()
    assert ring.observe(18, 0) == []
    with pytest.raises(ValueError):
        FutureRanks(horizons=(8,), capacity=8)


def test_ranks_have_explicit_competition_ties_and_reject_nonprobabilities():
    ring = FutureRanks()
    ring.capture(0, {20: np.array([0.5, 0.5])})
    assert ring.observe(1, 1)[0]["rank"] == 1
    with pytest.raises(ValueError):
        ring.capture(1, {20: np.array([np.nan, 1.0])})


def test_transport_excludes_self_and_retains_mass_instead_of_renormalizing():
    # Half the mass is self-attention. Prior-source overlap is 1 and 1/2.
    result = transport_overlap(
        [1, 2],
        [[1, 2], [2, 3], [1, 2]],
        [0.2, 0.3, 0.5],
        source_positions=[0, 1, 2],
        target_position=2,
    )
    assert result["score"] == pytest.approx(0.35)
    assert result["past_mass"] == pytest.approx(0.5)
    assert result["conditional_overlap"] == pytest.approx(0.7)
    assert result["source_positions"] == [0, 1]
    assert (
        transport_overlap([1], [[1]], [1.0], source_positions=[2], target_position=2)[
            "conditional_overlap"
        ]
        is None
    )


def test_refit_is_only_triggered_by_a_failed_map_check():
    assert layer_verdict("pass", "pass", 999.0) == {
        "outcome": "readable",
        "refit_required": False,
        "map_check": "pass",
        "concept_check": "pass",
        "output_divergence": 999.0,
    }
    assert layer_verdict("pass", "fail", 0.0)["outcome"] == "not readable"
    assert layer_verdict("pass", "fail", 0.0)["refit_required"] is False
    assert layer_verdict("fail", "pass", 0.0)["refit_required"] is True
    assert layer_verdict("inconclusive", "pass", 0.0)["outcome"] == "inconclusive"
    assert layer_verdict("inconclusive", "pass", 0.0)["refit_required"] is False


def test_capture_record_is_deterministic_and_detects_tampering(tmp_path):
    from local_llm_lab.pipeline.live_lens.session import RecordWriter, read_record

    paths = [tmp_path / name for name in ("one.jsonl", "two.jsonl")]
    for path in paths:
        with RecordWriter(path, {"model_sha256": "a" * 64}) as write:
            write({"kind": "emitted", "position": 9, "token_id": 1})
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert read_record(paths[0])[1]["position"] == 9
    paths[0].write_text(paths[0].read_text().replace('"position":9', '"position":8'))
    with pytest.raises(ValueError, match="hash"):
        read_record(paths[0])
