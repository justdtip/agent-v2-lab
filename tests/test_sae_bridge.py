"""Stage A of the SAE-to-J-lens bridge, on fixtures small enough to check by hand.

Every dimension here is a fixture's. The real dictionary, lens and checkpoint are exercised only
by the test marked so, which skips when the artefacts are absent from this checkout.

Two fixtures, because they test different things. The general one has a random, untied encoder:
right for the loader, the readout and the control, and wrong for the decomposition, since a
random encoder is not the decoder's inverse and re-encoding a reconstruction does not recover its
code. The A2 fixture has a tied, orthonormal decoder with no biases, so encode-then-decode is exact
on the support and an activation in the decoder's orthogonal complement has code exactly zero.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from local_llm_lab.pipeline.live_lens.instruments import LensIdentity
from local_llm_lab.probes import sae_bridge as B

HIDDEN, WIDTH, VOCAB, K = 16, 24, 40, 5
A2_WIDTH = 8  # fewer features than dimensions, so an orthonormal decoder exists


# ---------------------------------------------------------------------------------- fixtures


FIXTURE_MODEL = "fixture/base-it"


def _write_dictionary(
    directory: Path,
    seed: int = 0,
    *,
    block: int = 3,
    width: int = WIDTH,
    tied: bool = False,
    orthonormal: bool = False,
    biases: bool = True,
    threshold: tuple[float, float] = (0.5, 2.0),
    model_name: str = FIXTURE_MODEL,
) -> tuple[Path, Path]:
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    if orthonormal:
        q, _ = np.linalg.qr(rng.normal(size=(HIDDEN, width)))
        w_dec = np.ascontiguousarray(q.T, dtype=np.float32)
    else:
        w_dec = rng.normal(size=(width, HIDDEN)).astype(np.float32)
        w_dec /= np.linalg.norm(w_dec, axis=1, keepdims=True)
    w_enc = w_dec.T.copy() if tied else rng.normal(size=(HIDDEN, width)).astype(np.float32)
    tensors = {
        "w_enc": np.ascontiguousarray(w_enc, dtype=np.float32),
        "b_enc": (rng.normal(size=(width,)) if biases else np.zeros(width)).astype(np.float32),
        "threshold": rng.uniform(*threshold, size=(width,)).astype(np.float32),
        "w_dec": w_dec,
        "b_dec": (rng.normal(size=(HIDDEN,)) if biases else np.zeros(HIDDEN)).astype(np.float32),
    }
    directory.mkdir(parents=True, exist_ok=True)
    params = directory / "params.safetensors"
    config = directory / "config.json"
    save_file(tensors, str(params))
    config.write_text(json.dumps({
        "hf_hook_point_in": f"model.layers.{block}.output",
        "model_name": model_name,
        "hf_hook_point_out": f"model.layers.{block}.output",
        "width": width, "architecture": "jump_relu", "l0": 4, "type": "sae",
    }))
    return params, config


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (nearest even) and return the sixteen stored bits."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return rounded.astype("<u2")


def _save_safetensors(path: Path, tensors: dict[str, tuple[np.ndarray, str]]) -> None:
    """Write the container by hand, so a test can store dtypes NumPy has no name for."""
    import json as _json
    import struct as _struct

    header: dict = {}
    blobs: list[bytes] = []
    offset = 0
    for name, (array, dtype) in tensors.items():
        if dtype == "BF16":
            data = _to_bf16_bits(array).tobytes()
        elif dtype == "F32":
            data = np.ascontiguousarray(array, dtype="<f4").tobytes()
        else:
            raise ValueError(dtype)
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    encoded = _json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as handle:
        handle.write(_struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)


def _write_checkpoint(
    directory: Path,
    seed: int = 1,
    *,
    prefix: str = "language_model.model.",
    dtype: str = "BF16",
) -> Path:
    """A checkpoint shaped and typed like the real one: the readout in bfloat16 by default.

    The real bf16 checkpoint stores its embedding in bfloat16, which NumPy has no dtype for. A
    fixture that wrote float32 exercised a loader path the real artefact never takes, and the
    first real run died on exactly that line. The fixture now carries the real artefact's dtype.
    """
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    _save_safetensors(directory / "model.safetensors", {
        f"{prefix}embed_tokens.weight": (
            rng.normal(size=(VOCAB, HIDDEN)).astype(np.float32), dtype
        ),
        f"{prefix}norm.weight": (rng.normal(scale=0.3, size=(HIDDEN,)).astype(np.float32), dtype),
        f"{prefix}layers.0.input_layernorm.weight": (np.ones(HIDDEN, np.float32), dtype),
    })
    return directory


class _Lens:
    """The attributes of ``LensMaps`` the bridge touches, with random distinct orthogonal maps."""

    def __init__(self, layers: int, seed: int = 2, base: str | None = FIXTURE_MODEL):
        rng = np.random.default_rng(seed)
        self.maps = {
            L: np.linalg.qr(rng.normal(size=(HIDDEN, HIDDEN)))[0].astype(np.float32)
            for L in range(1, layers)
        }
        self.hidden_size = HIDDEN
        self.num_layers = layers
        self.sha256 = "fixture"
        self.identity = None if base is None else LensIdentity(base=base, num_layers=layers)
        self.storage_dtype = ("float32",)


@pytest.fixture
def parts(tmp_path: Path):
    params, config = _write_dictionary(tmp_path / "sae")
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    return (
        B.load_dictionary(params, config, source="fixture/sae"),
        B.load_unembedding(ckpt),
        _Lens(layers=6),
    )


@pytest.fixture
def a2_parts(tmp_path: Path):
    params, config = _write_dictionary(
        tmp_path / "sae2", width=A2_WIDTH, tied=True, orthonormal=True, biases=False,
        threshold=(0.1, 0.1),
    )
    ckpt = _write_checkpoint(tmp_path / "ckpt2")
    return B.load_dictionary(params, config), B.load_unembedding(ckpt), _Lens(layers=6)


# ---------------------------------------------------------------------------- alignment


@pytest.mark.parametrize("block", [0, 3, 17, 33])
def test_a_hook_on_block_n_reads_layer_n_plus_one(block: int) -> None:
    """The off-by-one, as a test: block N's output is the capture convention's layer N + 1."""
    assert B.layer_for_hook(f"model.layers.{block}.output") == block + 1


@pytest.mark.parametrize("hook", ["model.layers.3.input", "blocks.3.hook_mlp_out", "layer_3", ""])
def test_a_hook_that_is_not_a_block_output_is_refused(hook: str) -> None:
    with pytest.raises(ValueError, match="does not name a decoder block"):
        B.layer_for_hook(hook)


def test_alignment_refuses_a_lens_without_a_map_at_the_hooked_layer(tmp_path: Path) -> None:
    params, config = _write_dictionary(tmp_path / "sae", block=9)  # reads layer 10
    d = B.load_dictionary(params, config)
    with pytest.raises(ValueError, match="no map at that layer"):
        B.hook_alignment(d, _Lens(layers=6))
    assert B.hook_alignment(d, _Lens(layers=12)) == 10


# --------------------------------------------------------------------------------- loading


def test_dictionary_shapes_must_describe_one_hidden_width_pair(tmp_path: Path) -> None:
    from safetensors.numpy import save_file

    params, config = _write_dictionary(tmp_path / "sae")
    header, _ = B.safetensors_header(params)
    t = {k: B.read_tensor(params, k) for k in header}
    t["b_dec"] = t["b_dec"][:-1]
    save_file(t, str(params))
    with pytest.raises(ValueError, match="b_dec is"):
        B.load_dictionary(params, config)


def test_unembedding_reads_by_suffix_through_any_prefix_and_reports_tying(tmp_path: Path) -> None:
    u = B.load_unembedding(_write_checkpoint(tmp_path / "a", prefix="language_model.model."))
    assert u.tied and u.weight.shape == (VOCAB, HIDDEN) and u.gain.shape == (HIDDEN,)
    assert u.tensor_names["head"].endswith("embed_tokens.weight")

    v = B.load_unembedding(_write_checkpoint(tmp_path / "b", prefix="model."))
    assert v.tensor_names["head"] == "model.embed_tokens.weight"


def test_unembedding_gain_is_one_plus_the_stored_norm(tmp_path: Path) -> None:
    """Gemma stores its norm scale as an offset from one; applied raw it is near noise."""
    ckpt = _write_checkpoint(tmp_path / "c")
    stored = B.read_tensor(ckpt / "model.safetensors", "language_model.model.norm.weight")
    np.testing.assert_array_equal(B.load_unembedding(ckpt).gain, (1.0 + stored).astype(np.float32))


def test_bfloat16_is_widened_exactly_and_float32_is_read_as_is(tmp_path: Path) -> None:
    """The reader's one nontrivial branch, against values whose bfloat16 bits are known."""
    values = np.array([1.0, -2.0, 0.5, 3.140625, 1e-3, 65504.0, 0.0], dtype=np.float32)
    _save_safetensors(tmp_path / "t.safetensors", {"b": (values, "BF16"), "f": (values, "F32")})
    widened = B.read_tensor(tmp_path / "t.safetensors", "b")
    # What bfloat16 rounding stored, widened by the reader's own shift, must match bit for
    # bit.
    expected = (_to_bf16_bits(values).astype(np.uint32) << np.uint32(16)).view(np.float32)
    np.testing.assert_array_equal(widened, expected)
    assert widened.dtype == np.float32
    # Values exactly representable in bfloat16 survive the round trip untouched.
    for exact in (1.0, -2.0, 0.5, 0.0):
        assert exact in widened.tolist()
    np.testing.assert_array_equal(B.read_tensor(tmp_path / "t.safetensors", "f"), values)
    with pytest.raises(KeyError, match="no tensor"):
        B.read_tensor(tmp_path / "t.safetensors", "missing")


def test_a_float32_checkpoint_still_loads(tmp_path: Path) -> None:
    u = B.load_unembedding(_write_checkpoint(tmp_path / "f32", dtype="F32"))
    assert u.weight.dtype == np.float32 and u.weight.shape == (VOCAB, HIDDEN)


def test_unembedding_refuses_two_candidate_norms(tmp_path: Path) -> None:
    d = tmp_path / "d"
    d.mkdir()
    rng = np.random.default_rng(0)
    _save_safetensors(d / "model.safetensors", {
        "a.embed_tokens.weight": (rng.normal(size=(VOCAB, HIDDEN)).astype(np.float32), "BF16"),
        "a.norm.weight": (np.zeros(HIDDEN, np.float32), "BF16"),
        "b.norm.weight": (np.zeros(HIDDEN, np.float32), "BF16"),
    })
    with pytest.raises(ValueError, match="exactly one final norm"):
        B.load_unembedding(d)


# --------------------------------------------------------------------------------- encoding


def test_jumprelu_gates_at_each_features_own_threshold(parts) -> None:
    d, _, _ = parts
    h = np.random.default_rng(3).normal(size=HIDDEN).astype(np.float32)
    pre = h @ d.w_enc + d.b_enc
    z = B.encode(d, h)
    assert np.all(z[pre <= d.threshold] == 0)
    np.testing.assert_allclose(z[pre > d.threshold], pre[pre > d.threshold])


def test_decode_is_the_affine_map_the_shapes_say_it_is(parts) -> None:
    d, _, _ = parts
    rng = np.random.default_rng(4)
    z = np.zeros(WIDTH, np.float32)
    z[rng.choice(WIDTH, 4, replace=False)] = rng.uniform(1, 3, 4)
    np.testing.assert_allclose(B.decode(d, z), z @ d.w_dec + d.b_dec, rtol=1e-6)


def test_encode_inverts_decode_on_the_tied_orthonormal_fixture(a2_parts) -> None:
    """The property the A2 fixture exists to have, asserted before anything relies on it."""
    d, _, _ = a2_parts
    rng = np.random.default_rng(7)
    z = np.zeros(A2_WIDTH, np.float32)
    z[rng.choice(A2_WIDTH, 3, replace=False)] = rng.uniform(1, 3, 3)
    np.testing.assert_allclose(B.encode(d, B.decode(d, z)), z, rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------------- A1, two ways


def test_a1_batched_and_direct_agree_to_float32_tolerance(parts) -> None:
    """The acceptance check: ``L d_i`` through the composed path and by a direct product."""
    d, u, lens = parts
    J = lens.maps[B.hook_alignment(d, lens)]
    idx, val = B.feature_scores(d, u, J, k=K, chunk=7)  # a chunk that does not divide the width
    one, _ = B.feature_scores(d, u, J, k=VOCAB, chunk=1)
    full, _ = B.feature_scores(d, u, J, k=VOCAB, chunk=WIDTH)
    for i in (0, 5, WIDTH - 1):
        direct = B.feature_score_column(d, u, J, i)
        di, dv = B.top_tokens(direct, K)
        assert idx[i].tolist() == di[0].tolist()
        np.testing.assert_allclose(val[i], dv[0], rtol=1e-6, atol=1e-6)
        # The whole ordering, not just its top, from chunk=1 and chunk=width, against direct.
        expected = np.argsort(-direct, kind="stable").tolist()
        assert one[i].tolist() == full[i].tolist() == expected


class _SpyWeight(np.ndarray):
    """An unembedding whose matmuls record the row count of what multiplied it.

    The only intermediate of shape (rows, vocab) the readout can form is ``u_chunk @ W.T``, and
    this sees every such product from the ``W`` side, so a full ``(width, vocab)`` block cannot be
    formed without being recorded.
    """

    __array_priority__ = 1000
    seen: list[int] = []

    def __array_finalize__(self, obj):
        pass

    def __rmatmul__(self, other):
        _SpyWeight.seen.append(int(np.asarray(other).shape[0]))
        return np.asarray(other) @ np.asarray(self)

    def __matmul__(self, other):
        _SpyWeight.seen.append(int(np.asarray(self).shape[0]))
        return np.asarray(self) @ np.asarray(other)


def test_a1_never_forms_the_full_score_transfer(parts) -> None:
    """The order's cost rule, enforced at the matmul and not inferred from the code."""
    d, u, lens = parts
    _SpyWeight.seen = []
    spied = dataclasses.replace(u, weight=u.weight.view(_SpyWeight))
    B.feature_scores(d, spied, lens.maps[4], k=K, chunk=8)
    assert _SpyWeight.seen, "the spy recorded nothing, so it did not see the multiplies"
    assert max(_SpyWeight.seen) <= 8
    assert WIDTH not in _SpyWeight.seen


# ---------------------------------------------------------------------- negative control


def test_negative_control_bites_across_layers_and_fails_on_a_deliberate_mismatch(parts) -> None:
    d, u, lens = parts
    L = B.hook_alignment(d, lens)
    other = 1 if L != 1 else 2
    bites = B.negative_control(d, u, lens, L, other, k=K)
    assert bites["distinguishes"] and bites["mean_overlap"] < 0.9
    # Shown to fail: the same layer twice must not be reported as distinguishable.
    same = B.negative_control(d, u, lens, L, L, k=K)
    assert not same["distinguishes"] and same["mean_overlap"] == 1.0


# --------------------------------------------------------------- convention discriminator


def test_convention_check_returns_the_amendments_four_rows() -> None:
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8, 9], [10, 11, 12]])
    assert not B.convention_check(a, a, b)["stop"]
    assert "raw" in B.convention_check(a, a, b)["conclusion"]
    assert "gain" in B.convention_check(a, b, a)["conclusion"]
    assert "uninformative" in B.convention_check(a, a, a)["conclusion"]
    stop = B.convention_check(a, b, b)
    assert stop["stop"] and "orientation" in stop["conclusion"]


# ------------------------------------------------------------------------------------- A2


def test_a2_is_an_exact_identity_on_the_emitted_token(a2_parts) -> None:
    """``L h = L b + sum z_i (L d_i) + L e`` holds at the token, to float32, however h is made."""
    d, u, lens = a2_parts
    J = lens.maps[B.hook_alignment(d, lens)]
    rng = np.random.default_rng(5)
    for h in (rng.normal(size=HIDDEN), 50 * rng.normal(size=HIDDEN), np.zeros(HIDDEN)):
        out = B.decompose_position(d, u, J, h.astype(np.float32), int(rng.integers(VOCAB)))
        total = out["bias_term"] + out["feature_sum"] + out["residual_term"]
        assert out["identity_gap"] < 1e-3 * max(1.0, abs(out["score"]))
        assert abs(out["score"] - total) < 1e-3 * max(1.0, abs(out["score"]))


def test_a2_ranks_when_the_dictionary_explains_the_activation(a2_parts) -> None:
    d, u, lens = a2_parts
    J = lens.maps[B.hook_alignment(d, lens)]
    rng = np.random.default_rng(8)
    z = np.zeros(A2_WIDTH, np.float32)
    active = rng.choice(A2_WIDTH, 3, replace=False)
    z[active] = rng.uniform(2, 4, 3)
    h = B.decode(d, z) + rng.normal(scale=1e-3, size=HIDDEN).astype(np.float32)
    out = B.decompose_position(d, u, J, h, int(rng.integers(VOCAB)), k=3, dominance=0.5)
    assert out["ranked"] and out["residual_share"] < 0.05
    assert out["active_features"] == 3
    assert {f["feature"] for f in out["top_features"]} <= set(active.tolist())
    assert all(f["z"] > 0 for f in out["top_features"])


def test_a2_refuses_to_rank_under_a_dominant_residual(a2_parts) -> None:
    """An activation outside the decoder's span: code exactly zero, residual is everything."""
    d, u, lens = a2_parts
    J = lens.maps[B.hook_alignment(d, lens)]
    rng = np.random.default_rng(6)
    x = rng.normal(size=HIDDEN)
    h = (x - d.w_dec.T @ (d.w_dec @ x)).astype(np.float32)  # orthogonal complement projection
    assert np.abs(d.w_dec @ h).max() < 1e-5
    out = B.decompose_position(d, u, J, h, 3, dominance=0.5)
    assert out["active_features"] == 0
    assert out["residual_share"] > 0.99
    assert not out["ranked"] and "residual dominates" in out["reason"]
    assert "top_features" not in out


def test_a2_refuses_a_width_that_is_not_the_dictionarys(parts) -> None:
    d, u, lens = parts
    with pytest.raises(ValueError, match="width"):
        B.decompose_position(d, u, lens.maps[4], np.zeros(HIDDEN + 1, np.float32), 0)


# ------------------------------------------------------------------------------- provenance


def test_provenance_carries_every_required_field_and_the_unlabelled_reason(parts) -> None:
    d, u, lens = parts
    p = B.bridge_provenance(d, lens, u, 4, dictionary_repo="r", dictionary_folder="f")
    assert p["labels"] == "unlabelled" and p["labels_reason"] == B.UNLABELLED_REASON
    assert p["dictionary"]["params_sha256"] == d.sha256
    assert p["dictionary"]["decoder_rows_unit_norm"]
    assert p["lens"]["sha256"] == "fixture" and p["lens"]["layer"] == 4
    assert p["checkpoint"]["tied_embeddings"] is True
    assert "per-vector scalar" in p["score_map"]
    json.dumps(p)  # serialisable


# ------------------------------------------------------------------ the real artefacts, gated


def _real_paths():
    from local_llm_lab import runlock

    primary = runlock.primary_checkout_root()
    lens_npz = primary / "models" / "jlens" / "gemma-3-4b-it_jacobian_lens.npz"
    ckpt = primary / "models" / "gemma-3-4b-it-bf16"
    cache = primary / ".cache" / "huggingface" / "hub" / "models--google--gemma-scope-2-4b-it"
    pattern = "snapshots/*/resid_post_all/layer_*_width_16k_l0_small/params.safetensors"
    snaps = sorted(cache.glob(pattern))
    if not (lens_npz.is_file() and (ckpt / "config.json").is_file() and snaps):
        pytest.skip("real lens, checkpoint and dictionary are not all present on this checkout")
    return lens_npz, ckpt, snaps[0].parent


def test_the_real_dictionary_hook_is_a_layer_the_real_lens_carries() -> None:
    """The order's requirement that alignment be a test: real hook string, real archive keys."""
    from local_llm_lab.pipeline.live_lens.instruments import LensIdentity, LensMaps

    lens_npz, ckpt, folder = _real_paths()
    side = json.loads(lens_npz.with_suffix(".json").read_text())
    conf = json.loads((ckpt / "config.json").read_text())
    t = conf.get("text_config", conf)
    lens = LensMaps.load(
        lens_npz,
        expected_sha256=side["npz_sha256"],
        hidden_size=int(t["hidden_size"]),
        num_layers=int(t["num_hidden_layers"]),
        identity=LensIdentity.from_dict(side["model"]),
    )
    d = B.load_dictionary(folder / "params.safetensors", folder / "config.json")
    block = int(d.hook_point.split(".")[2])
    layer = B.hook_alignment(d, lens)
    assert layer == block + 1
    # The archive keys J{L-1}: the map the loader hands back at `layer` is the archive's J{block}.
    with np.load(lens_npz, allow_pickle=False) as a:
        np.testing.assert_array_equal(lens.maps[layer], a[f"J{block}"].astype(np.float32))


# ------------------------------------------------------------------------------ intervention


def test_intervention_parts_meet_the_wrapper_contract(parts):
    torch = pytest.importorskip("torch")
    from local_llm_lab.sae_intervention import SAEIntervention

    d, _, _ = parts
    encoder, decoder, bias, declared = B.intervention_parts(d, dtype=torch.float32)
    # Column convention: [residual, feature], the stored matrix transposed; bias separate.
    assert tuple(decoder.shape) == (d.hidden_size, d.width)
    assert torch.equal(decoder, torch.as_tensor(np.ascontiguousarray(d.w_dec.T)))
    assert torch.equal(bias, torch.as_tensor(d.b_dec.copy()))
    assert declared["dictionary_precision"] == "float32"
    assert declared["params_sha256"] == d.sha256

    rng = np.random.default_rng(3)
    h_np = (rng.standard_normal(d.hidden_size) * 4).astype(np.float32)
    h = torch.as_tensor(h_np)
    # The encoder callable agrees with the numpy reference encode, and does not touch its input.
    z = encoder(h)
    assert z.dtype == h.dtype and z.device == h.device
    np.testing.assert_allclose(z.numpy(), B.encode(d, h_np), rtol=1e-5, atol=1e-5)
    assert torch.equal(h, torch.as_tensor(h_np))

    # A bfloat16 residual gets a bfloat16 code, formed in the declared float32.
    z16 = encoder(h.to(torch.bfloat16))
    assert z16.dtype == torch.bfloat16

    # The real wrapper accepts the parts and the edit lands on the chosen feature.
    feature = int(np.argmax(B.encode(d, h_np)))
    target = torch.tensor([float(z[feature]) + 2.0])
    edit = SAEIntervention(encoder, decoder, bias, features=(feature,), target_values=target)
    replaced = edit(h)
    expected = h + decoder[:, feature] * 2.0
    torch.testing.assert_close(replaced, expected)
    record = edit.diagnostic_record()
    assert record["features"] == [feature] and record["basis"] == "measured-here"


def test_intervention_parts_refuse_nothing_but_record_precision(parts):
    torch = pytest.importorskip("torch")
    d, _, _ = parts
    _, decoder, _, declared = B.intervention_parts(d, dtype=torch.bfloat16)
    assert decoder.dtype == torch.bfloat16
    assert declared["dictionary_precision"] == "bfloat16"


# ------------------------------------------------------------------------- model identity


def test_a_dictionary_of_another_checkpoint_of_the_same_width_is_refused(tmp_path: Path) -> None:
    """The pt-against-it case: same family, depth and width, a different model."""
    params, config = _write_dictionary(tmp_path / "pt", model_name="fixture/base-pt")
    d = B.load_dictionary(params, config)
    expected = "trained on 'fixture/base-pt' and the lens maps 'fixture/base-it'"
    with pytest.raises(ValueError, match=expected):
        B.hook_alignment(d, _Lens(layers=12))


def test_a_dictionary_that_names_no_model_is_refused(tmp_path: Path) -> None:
    params, config = _write_dictionary(tmp_path / "anon")
    raw = json.loads(config.read_text())
    del raw["model_name"]
    config.write_text(json.dumps(raw))
    d = B.load_dictionary(params, config)
    with pytest.raises(ValueError, match="names no model"):
        B.hook_alignment(d, _Lens(layers=12))


def test_a_lens_without_identity_cannot_be_aligned(parts) -> None:
    d, _, _ = parts
    with pytest.raises(ValueError, match="carries no identity"):
        B.hook_alignment(d, _Lens(layers=12, base=None))


def test_the_registry_entry_is_the_third_party_to_the_identity_check(parts) -> None:
    d, _, lens = parts
    assert B.hook_alignment(d, lens, base=FIXTURE_MODEL) == B.layer_for_hook(d.hook_point)
    with pytest.raises(ValueError, match="registry entry descends from 'fixture/base-pt'"):
        B.hook_alignment(d, lens, base="fixture/base-pt")


def test_the_real_dictionary_names_the_registry_base_the_lens_resolves_to() -> None:
    """A registry name, its hf id and its base all resolve to one string, the one a lens carries."""
    from local_llm_lab.models import base_of_artifact, load_model_spec, registered_models

    for name in registered_models():
        spec = load_model_spec(name)
        assert base_of_artifact(name) == base_of_artifact(spec.hf_id) == spec.base


# --------------------------------------------------------- intervention: decode and identity


def test_intervention_decoder_agrees_with_numpy_decode_and_preserves_the_residual(a2_parts):
    torch = pytest.importorskip("torch")
    from local_llm_lab.sae_intervention import SAEIntervention

    d, _, _ = a2_parts
    encoder, decoder, bias, _ = B.intervention_parts(d, dtype=torch.float32)
    rng = np.random.default_rng(11)
    z_np = np.abs(rng.standard_normal(d.width)).astype(np.float32)
    # decode agreement to float32: the wrapper's `decoder @ z + bias` is the numpy decode.
    np.testing.assert_allclose(
        (decoder @ torch.as_tensor(z_np) + bias).numpy(), B.decode(d, z_np), rtol=1e-5, atol=1e-5
    )
    # residual preservation through the adapter: the edit changes the code where asked and
    # leaves the dictionary's reconstruction error exactly where it was.
    h_np = (B.decode(d, z_np) + 0.05 * rng.standard_normal(d.hidden_size)).astype(np.float32)
    h = torch.as_tensor(h_np)
    z = encoder(h)
    feature = int(torch.argmax(z))
    target = torch.tensor([float(z[feature]) + 1.5])
    edit = SAEIntervention(encoder, decoder, bias, features=(feature,), target_values=target)
    replaced = edit(h)
    z_new = z.clone()
    z_new[feature] = target[0]
    epsilon_before = h - bias - decoder @ z
    epsilon_after = replaced - bias - decoder @ z_new
    torch.testing.assert_close(epsilon_after, epsilon_before, rtol=1e-5, atol=1e-5)
    assert not torch.equal(replaced, h)


# ------------------------------------------------------------------- reconstruction budget


def test_reconstruction_budget_reports_share_active_and_the_declined_fraction(a2_parts):
    d, _, _ = a2_parts
    rng = np.random.default_rng(5)
    # Sites the dictionary explains exactly: codes above threshold through the tied orthonormal
    # decoder. Sites it cannot explain at all: vectors in the decoder's orthogonal complement.
    z = np.abs(rng.standard_normal((6, d.width))).astype(np.float32) + 1.0
    explained = B.decode(d, z)
    Q, _ = np.linalg.qr(rng.standard_normal((d.hidden_size, d.hidden_size)))
    complement = Q[:, d.width:].T[:4].astype(np.float32)
    complement -= (complement @ d.w_dec.T) @ d.w_dec  # exact projection out of the span
    sites = np.vstack([explained, complement])
    out = B.reconstruction_budget(d, sites, dominance=0.5)
    assert out["dominance"] == 0.5 and out["sites"] == 10
    np.testing.assert_allclose(out["residual_share"][:6], 0.0, atol=1e-5)
    np.testing.assert_allclose(out["residual_share"][6:], 1.0, atol=1e-5)
    assert list(out["active_features"][:6]) == [d.width] * 6
    assert list(out["active_features"][6:]) == [0] * 4
    assert out["share_over_dominance"] == pytest.approx(0.4)
    assert out["rankable"].tolist() == [True] * 6 + [False] * 4
    assert out["residual_share_quantiles"]["0.5"] == pytest.approx(0.0, abs=1e-5)
    assert out["active_features_quantiles"]["1.0"] == d.width


def test_reconstruction_budget_refuses_the_wrong_shape_and_an_undeclared_threshold(a2_parts):
    d, _, _ = a2_parts
    with pytest.raises(ValueError, match="dominance must be in"):
        B.reconstruction_budget(d, np.ones((2, d.hidden_size), np.float32), dominance=0.0)
    with pytest.raises(ValueError, match="residuals must be"):
        B.reconstruction_budget(d, np.ones((2, d.hidden_size + 1), np.float32), dominance=0.5)
    with pytest.raises(ValueError, match="zero residual"):
        B.reconstruction_budget(d, np.zeros((1, d.hidden_size), np.float32), dominance=0.5)


# ------------------------------------------------------------------- lens fit precision

FLOAT32_NU = {"precision": {"dtype": "float32", "fit_dtype": "float32",
                            "forward_batch": 1, "anchor_batch": 1}}  # fmt: skip
BF16_NU = {"precision": {"dtype": "bfloat16", "fit_dtype": "bfloat16",
                         "forward_batch": 64, "anchor_batch": 1}}  # fmt: skip
SILENT_NU = {"precision": {"dtype": "bfloat16", "device": "cpu"}}


def test_a_lens_that_does_not_say_what_it_was_fitted_in_is_refused_by_a_declaring_registry(parts):
    """The width rows: an undeclared fit is not assumed to be the declared one."""
    d, _, lens = parts
    with pytest.raises(ValueError, match="carries no `fit_dtype`"):
        B.hook_alignment(d, lens, lens_fit_dtype="float32", nu=SILENT_NU)
    # A lens with no nu at all is the same absence and refuses the same way.
    with pytest.raises(ValueError, match="carries no `fit_dtype`"):
        B.hook_alignment(d, lens, lens_fit_dtype="float32", nu=None)


def test_the_registry_and_the_lens_must_agree_on_the_fit_precision(parts):
    d, _, lens = parts
    with pytest.raises(ValueError, match="fit_dtype 'bfloat16'.*lens_fit_dtype 'float32'"):
        B.hook_alignment(d, lens, lens_fit_dtype="float32", nu=BF16_NU)
    # And the other way round, so the check is not one-sided.
    with pytest.raises(ValueError, match="fit_dtype 'float32'.*lens_fit_dtype 'bfloat16'"):
        B.hook_alignment(d, lens, lens_fit_dtype="bfloat16", nu=FLOAT32_NU)
    layer = B.layer_for_hook(d.hook_point)
    assert B.hook_alignment(d, lens, lens_fit_dtype="float32", nu=FLOAT32_NU) == layer


def test_both_silent_is_recorded_as_undeclared_rather_than_refused(parts):
    """Every lens fitted before the ruling, upstream's hosted one included."""
    d, _, lens = parts
    layer = B.layer_for_hook(d.hook_point)
    assert B.hook_alignment(d, lens) == layer
    assert B.hook_alignment(d, lens, nu=SILENT_NU) == layer
    record = B.fit_precision_record(SILENT_NU, declared=None)
    assert record["status"] == "undeclared"
    assert record["fit_dtype"] is None and record["declared_by_registry"] is None
    assert "does not say what arithmetic" in record["note"]
    assert "path_term" not in record


def test_a_float32_lens_on_a_native_capture_carries_the_declared_path_term():
    record = B.fit_precision_record(FLOAT32_NU, declared="float32", capture_dtype="native")
    assert record["status"] == "declared"
    assert record["forward_batch"] == 1 and record["anchor_batch"] == 1
    assert record["path_term"]["relative"] == 0.0124
    assert record["path_term"]["at"] == "64 tokens"
    assert "declared" in record["path_term"]["basis"]
    # The term is the cost of the *path difference*: no difference, no term.
    same = B.fit_precision_record(FLOAT32_NU, declared="float32", capture_dtype="float32")
    assert "path_term" not in same


def test_the_fit_precision_lands_in_the_provenance_block(parts):
    d, u, lens = parts
    record = B.fit_precision_record(FLOAT32_NU, declared="float32")
    block = B.bridge_provenance(
        d, lens, u, 10, dictionary_repo="r", dictionary_folder="f", fit_precision=record
    )
    assert block["lens"]["fit_precision"]["fit_dtype"] == "float32"
    assert block["lens"]["fit_precision"]["path_term"]["relative"] == 0.0124
    # Absent by default, so an artefact that did not check cannot look as though it had.
    plain = B.bridge_provenance(d, lens, u, 10, dictionary_repo="r", dictionary_folder="f")
    assert plain["lens"]["fit_precision"] is None
