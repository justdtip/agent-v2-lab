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


QWEN = LensIdentity(base="mlx-community/Qwen3.5-4B-MLX-4bit", num_layers=32)
GEMMA = LensIdentity(base="google/gemma-3-4b-it", num_layers=34)
GEMMA_TRAINED = LensIdentity(
    base="google/gemma-3-4b-it",
    num_layers=34,
    training={"kind": "lora", "rows": 1200, "adapter": "outputs/x"},
)


def test_lens_orientation_identity_and_hash_verification(tmp_path):
    path, sha = _stamped_lens(
        tmp_path, "lens.npz", {"J0": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)},
        LensIdentity(base="toy/toy", num_layers=2),
    )
    identity = LensIdentity(base="toy/toy", num_layers=2)
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
    assert "mlx-community/Qwen3.5-4B-MLX-4bit" in message, "the lens names its own model"
    assert "google/gemma-3-4b-it" in message, "and the model it is being loaded against"

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


def test_lens_identity_refuses_positional_construction_because_its_fields_have_moved() -> None:
    """The three fields are not the three this class started with, so position means nothing.

    Until ``a960d80`` the signature was ``(name, hf_id, num_layers)``. The lineage rule replaced
    the first two with one ``base`` and appended ``training``. Two-argument positional callers
    survived that by coincidence and were never re-read;
    ``research/records/GEMMA3-REGRESSION-2026-09-08/compare_maps.py`` did not, and still passes a
    filesystem path where ``num_layers`` now is. This test is the mechanism that makes the next
    such change break at the call rather than at a caller nobody thought to open.
    """
    with pytest.raises(TypeError):
        LensIdentity("google/gemma-3-4b-it", 34)  # type: ignore[misc]

    # Specifically the shape the stale record's script uses: a path bound to num_layers and a
    # layer count bound to training. Under the old signature this constructed; it must not now.
    with pytest.raises(TypeError):
        LensIdentity("gemma3-4b-bf16", "/models/gemma-3-4b-it-bf16", 34)  # type: ignore[misc]

    # And the keyword form is unchanged, so the refusal is about position and nothing else.
    assert LensIdentity(base="google/gemma-3-4b-it", num_layers=34) == GEMMA


def test_two_precisions_of_one_model_share_a_lens_and_the_refusal_is_about_the_model() -> None:
    """The correction the Gemma pilot forced, and the line the check has to hold.

    The identity carried the registry entry's name and its `hf_id`, and both describe a **file**.
    `gemma3-4b` and `gemma3-4b-bf16` are two precisions of one model; the pivot's ruling is that
    the pilot runs 4-bit while the hosted lens was fitted on bf16, with the precision mismatch
    disclosed rather than avoided. The old identity refused that pairing — true about the files,
    false about the experiment — and it refused at the moment stage one tried to load its lens.

    What must still be refused is a different model of the same width, which is what the check
    exists for and what no other check catches.
    """
    from local_llm_lab.models import load_model_spec

    four_bit = load_model_spec("gemma3-4b")
    bf16 = load_model_spec("gemma3-4b-bf16")

    assert four_bit.hf_id != bf16.hf_id, "different files"
    assert four_bit.base == bf16.base == "google/gemma-3-4b-it", "one base"
    assert four_bit.training is bf16.training is None, "neither carries training"
    assert LensIdentity(
        base=four_bit.base, num_layers=34, training=four_bit.training
    ) == LensIdentity(base=bf16.base, num_layers=34, training=bf16.training)
    assert (
        LensIdentity(base=four_bit.base, num_layers=34, training=four_bit.training) != QWEN
    )

    qwen = load_model_spec("qwen35-4b")
    assert qwen.base == qwen.hf_id, "an unconverted checkpoint is its own base"


def test_training_separates_two_checkpoints_of_one_base(tmp_path) -> None:
    """R57: a base and that base after training share family, depth and width, and are not one.

    Nothing about the architecture separates them, and a Jacobian lens is a map of a model's
    residual geometry, which training moves. This repository measured how much on 2026-09-08:
    full-depth training shifted every one of thirty-two layers by about a fifth of its weight
    norm, and two arms' updates to the layers they shared were orthogonal. An identity that
    stopped at the base would call a base-fitted lens a match for a trained checkpoint.

    Reading a trained model through its base's lens is a real measurement — it shows what training
    moved — so this is a refusal to do it *by accident*, not a refusal to do it.
    """
    maps = {f"J{i}": np.eye(2, dtype=np.float16) for i in range(33)}
    path, sha = _stamped_lens(tmp_path, "base-lens.npz", maps, GEMMA)

    with pytest.raises(LensIdentityError) as error:
        LensMaps.load(
            path, expected_sha256=sha, hidden_size=2, num_layers=34, identity=GEMMA_TRAINED
        )
    assert "base" in str(error.value) and "trained" in str(error.value)

    lens = LensMaps.load(
        path, expected_sha256=sha, hidden_size=2, num_layers=34, identity=GEMMA
    )
    assert lens.identity == GEMMA


def test_a_lens_naming_an_artifact_resolves_to_its_base_rather_than_being_rewritten(
    tmp_path,
) -> None:
    """R60: the loader resolves, the artefact is not migrated, and its digest stays valid.

    A lens stamped before the lineage rule carries an artefact path where the base belongs — a
    property of a file. Rewriting the archive to fix that would invalidate a digest already
    published in committed records, to restate something the registry knows. So the loader asks
    the registry which base that artefact descends from.
    """
    from local_llm_lab.models import base_of_artifact, load_model_spec

    bf16 = load_model_spec("gemma3-4b-bf16")
    assert base_of_artifact(bf16.hf_id) == "google/gemma-3-4b-it"
    assert base_of_artifact("gemma3-4b") == "google/gemma-3-4b-it", "an entry name resolves too"
    assert base_of_artifact("someone/unknown") == "someone/unknown", (
        "an unrecognised name reaches the comparison and is refused there with both names shown, "
        "rather than being silently rewritten"
    )

    maps = {f"J{i}": np.eye(2, dtype=np.float16) for i in range(33)}
    path = tmp_path / "old-stamp.npz"
    np.savez(
        path,
        identity=np.frombuffer(
            json.dumps({"hf_id": bf16.hf_id, "num_layers": 34}).encode("utf-8"), dtype=np.uint8
        ),
        **maps,
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    lens = LensMaps.load(path, expected_sha256=sha, hidden_size=2, num_layers=34, identity=GEMMA)
    assert lens.identity == GEMMA, "the old spelling resolved to the base it names"
