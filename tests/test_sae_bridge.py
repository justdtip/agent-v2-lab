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
    with pytest.raises(ValueError, match="no map at layer 10"):
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
        out = B.decompose_position(d, u, J, h.astype(np.float32), int(rng.integers(VOCAB)),
                                   error_budget=_error_budget())
        total = out["bias_term"] + out["feature_sum"] + out["residual_term"]
        # float64: exact to rounding of the terms, not to a loose 1e-3
        assert out["identity_gap"] < 1e-9 * max(1.0, out["identity_terms_abs_sum"])
        assert abs(out["score"] - total) < 1e-9 * max(1.0, out["identity_terms_abs_sum"])
        assert abs(out["score"] - out["score_float32"]) < 1e-3 * max(1.0, abs(out["score"]))
        assert out["identity_gap_float32"] >= 0.0


def test_a2_identity_is_asserted_in_float64_and_still_bites_on_a_wrong_orientation(a2_parts, monkeypatch) -> None:
    """The device refusal of 2026-09-10: at 16,384 features the float32 sum missed the identity by
    1e-3 on a score of 0.1 while float64 held it at 1e-12. The check now runs in float64, and a
    mismatched orientation between the score path and the per-feature terms is still refused."""
    d, u, lens = a2_parts
    J = lens.maps[B.hook_alignment(d, lens)]
    h = (50 * np.random.default_rng(6).normal(size=HIDDEN)).astype(np.float32)
    out = B.decompose_position(d, u, J, h, 2, error_budget=_error_budget())
    assert out["identity_gap"] < 1e-9 * max(1.0, out["identity_terms_abs_sum"])
    # Break the score path's orientation only: the per-feature terms keep the right one.
    monkeypatch.setattr(B, "_through_lens", lambda x, m: x if m is None else x @ m)
    with pytest.raises(ValueError, match="not exact"):
        B.decompose_position(d, u, J, h, 2, error_budget=_error_budget())


def test_a2_ranks_when_the_dictionary_explains_the_activation(a2_parts) -> None:
    d, u, lens = a2_parts
    J = lens.maps[B.hook_alignment(d, lens)]
    rng = np.random.default_rng(8)
    z = np.zeros(A2_WIDTH, np.float32)
    active = rng.choice(A2_WIDTH, 3, replace=False)
    z[active] = rng.uniform(2, 4, 3)
    h = B.decode(d, z) + rng.normal(scale=1e-3, size=HIDDEN).astype(np.float32)
    out = B.decompose_position(
        d, u, J, h, int(rng.integers(VOCAB)), k=3, error_budget=_error_budget()
    )
    assert out["ranked"] and out["lens_score_error_share"] < 0.05
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
    out = B.decompose_position(d, u, J, h, 3, error_budget=_error_budget())
    assert out["active_features"] == 0
    assert out["lens_score_error_share"] > 0.99
    assert not out["ranked"] and "lens score error" in out["reason"]
    assert "top_features" not in out


def test_a2_refuses_a_width_that_is_not_the_dictionarys(parts) -> None:
    d, u, lens = parts
    with pytest.raises(ValueError, match="width"):
        B.decompose_position(d, u, lens.maps[4], np.zeros(HIDDEN + 1, np.float32), 0,
                             error_budget=_error_budget())


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
    out = B.raw_reconstruction_budget(d, sites, error_budget=_error_budget())
    assert out["error_budget"]["raw_reconstruction_threshold"] == 0.5 and out["sites"] == 10
    np.testing.assert_allclose(out["raw_reconstruction_share"][:6], 0.0, atol=1e-5)
    np.testing.assert_allclose(out["raw_reconstruction_share"][6:], 1.0, atol=1e-5)
    assert list(out["active_features"][:6]) == [d.width] * 6
    assert list(out["active_features"][6:]) == [0] * 4
    assert out["raw_over_threshold_fraction"] == pytest.approx(0.4)
    assert out["raw_admissible"] == [True] * 6 + [False] * 4
    assert out["raw_reconstruction_share_quantiles"]["0.5"] == pytest.approx(0.0, abs=1e-5)
    assert out["active_features_quantiles"]["1.0"] == d.width


def test_reconstruction_budget_refuses_the_wrong_shape_and_an_undeclared_threshold(a2_parts):
    d, _, _ = a2_parts
    with pytest.raises((ValueError, TypeError), match="error_budget"):
        B.raw_reconstruction_budget(d, np.ones((2, d.hidden_size), np.float32))
    with pytest.raises(ValueError, match="residuals must be"):
        B.raw_reconstruction_budget(d, np.ones((2, d.hidden_size + 1), np.float32),
                                    error_budget=_error_budget())
    zero = B.raw_reconstruction_budget(d, np.zeros((1, d.hidden_size), np.float32),
                                       error_budget=_error_budget())
    assert zero["raw_reconstruction_share"] == [None] and zero["raw_admissible"] == [False]


# ------------------------------------------------------------------- lens fit precision

FLOAT32_NU = {"precision": {"dtype": "float32", "fit_dtype": "float32",
                            "forward_batch": 1, "anchor_batch": 1}}  # fmt: skip
BF16_NU = {"precision": {"dtype": "bfloat16", "fit_dtype": "bfloat16",
                         "forward_batch": 64, "anchor_batch": 1}}  # fmt: skip
SILENT_NU = {"precision": {"dtype": "bfloat16", "device": "cpu"}}
#: The model the displacement control measured; the fixture dictionary names its own.
GEMMA = "google/gemma-3-4b-it"


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


def test_a_cross_path_reading_is_refused_by_name_at_a_layer_with_a_measured_sensitivity():
    """WS-D's displacement control: at layer 1 the crossing is not an error bar but the
    answer."""
    with pytest.raises(ValueError, match="refusing a cross-path reading at layer 1") as raised:
        B.fit_precision_record(
            FLOAT32_NU, declared="float32", layer=1, base=GEMMA, capture_dtype="bfloat16"
        )
    message = str(raised.value)
    assert "fitted in float32 and the capture is bfloat16" in message
    # The sampled distribution, cited as that: the amplification ratios were withdrawn by the
    # map/anchor audit because their denominator was one position's norm and the intervention
    # replaced the whole sequence, so no ratio appears here or anywhere in the record.
    assert "median 1.035116 of their own size in float32 (native 0.470689)" in message
    assert "over 18 projections in one draw, worst 35.1951" in message
    assert "not a condition number" in message
    assert "no cross-path pairing is registered at layer 1" in message
    assert "paired comparison at that layer" in message

    # The last block is refused too, and its own figure travels with it: benign-looking is not
    # measured, and the ruling conditions on the pairing rather than on the size of a proxy.
    with pytest.raises(ValueError, match="refusing a cross-path reading at layer 33") as raised:
        B.fit_precision_record(
            FLOAT32_NU, declared="float32", layer=33, base=GEMMA, capture_dtype="bfloat16"
        )
    # And its own median does not excuse it: 0.125 in the middle, 91.9 at the worst of the
    # same eighteen projections, which is why a late layer is not shown benign by its median.
    message = str(raised.value)
    assert "median 0.125179" in message and "worst 91.8969" in message
    assert "a median does not settle a layer" in message


def test_a_layer_the_control_never_measured_says_so_rather_than_borrowing_a_neighbour():
    with pytest.raises(ValueError, match="refusing a cross-path reading at layer 18") as raised:
        B.fit_precision_record(
            FLOAT32_NU, declared="float32", layer=18, base=GEMMA, capture_dtype="bfloat16"
        )
    message = str(raised.value)
    assert "was not measured; it was measured at layers [1, 17, 33]" in message
    assert "not carried across from a neighbouring layer" in message
    assert B.anchor_sensitivity(18, base=GEMMA) is None
    sampled = B.anchor_sensitivity(17, base=GEMMA)["sampled_displacement"]
    assert sampled["float32"]["median"] == 0.696514 and sampled["native"]["median"] == 0.585164

    # A model nobody measured is a third absence, and it says which one it is.
    with pytest.raises(ValueError, match="no layer of that model was") as raised:
        B.fit_precision_record(
            FLOAT32_NU, declared="float32", layer=17, base="acme/unmeasured",
            capture_dtype="bfloat16",
        )  # fmt: skip
    assert B.measured_layers("acme/unmeasured") == []
    assert B.measured_layers(GEMMA) == [1, 17, 33]


def test_a_capture_at_another_width_is_a_crossing_even_at_one_precision():
    """The rule names width as well as precision: same arithmetic, a different function."""
    with pytest.raises(ValueError, match="forward width 1 and the capture is at width 64"):
        B.fit_precision_record(
            FLOAT32_NU, declared="float32", layer=17, base=GEMMA,
            capture_dtype="float32", capture_batch=64,
        )  # fmt: skip
    # Same width, same precision: nothing crossed.
    same = B.fit_precision_record(
        FLOAT32_NU, declared="float32", layer=17, base=GEMMA,
        capture_dtype="float32", capture_batch=1,
    )  # fmt: skip
    assert "path_term" not in same and "crossing" not in same


def test_a_cross_path_reading_without_a_layer_cannot_be_decided_and_says_so():
    with pytest.raises(ValueError, match="no layer was given"):
        B.fit_precision_record(FLOAT32_NU, declared="float32", capture_dtype="bfloat16")


def test_a_same_path_reading_and_a_reading_of_no_capture_carry_no_term(parts):
    d, _, lens = parts
    same = B.fit_precision_record(
        FLOAT32_NU, declared="float32", layer=1, base=GEMMA, capture_dtype="float32"
    )
    assert "path_term" not in same and "crossing" not in same
    # No capture at all: the dictionary-direction readout, which has no anchor to have moved.
    none = B.fit_precision_record(FLOAT32_NU, declared="float32", layer=1, base=GEMMA)
    assert "path_term" not in none and none["capture_dtype"] is None
    # And that is why A1's own call is untouched by the ruling.
    assert B.hook_alignment(d, lens, lens_fit_dtype="float32", nu=FLOAT32_NU) == B.layer_for_hook(
        d.hook_point
    )
    with pytest.raises(ValueError, match="refusing a cross-path reading"):
        B.hook_alignment(
            d, lens, lens_fit_dtype="float32", nu=FLOAT32_NU, capture_dtype="bfloat16"
        )


def test_the_per_layer_term_states_the_crossing_without_offering_a_number():
    term = B.path_term_for_layer(1, base=GEMMA)
    assert term["layer"] == 1 and term["measured"] is False and term["base"] == GEMMA
    assert "no paired comparison exists yet" in term["status"]
    assert term["anchor_sensitivity"]["sampled_displacement"]["float32"]["median"] == 1.035116
    assert "e771c2b" in term["anchor_sensitivity_basis"]
    assert "ca396fb" in term["anchor_sensitivity_basis"]
    assert term["anchor_sensitivity_label"] == (
        "sampled displacement, one draw, eighteen projections"
    )
    assert any(
        "does not follow from the median" in note for note in term["anchor_sensitivity_notes"]
    )
    # An equal-norm random displacement moves them as much or more, so it is not a special
    # direction, and that context travels with the figure rather than in someone's memory.
    assert term["anchor_sensitivity"]["equal_norm_random_displacement"]["native"]["median"] > 1.0
    assert B.path_term_for_layer(18, base=GEMMA)["anchor_sensitivity"] is None

    # No bare number at the top level, so nothing can be lifted out as an error bar. `measured`
    # is excluded by name rather than by type, because a bool *is* an int in Python and a type
    # test alone would either pass vacuously or fail on the flag it exists to keep.
    numeric = {
        key: value
        for key, value in term.items()
        if key not in ("measured", "layer") and isinstance(value, (int, float))
    }
    assert numeric == {}

    # Both historical figures still travel with their lengths, as W3 requires.
    context = {entry["at_tokens"]: entry for entry in term["historical_context"]}
    assert context[64]["relative"] == 0.0124 and context[1400]["relative"] == 0.694
    assert "6.9%" in context[1400]["note"]

    # A defensive copy: an artefact cannot edit the declaration for every later reading.
    term["measured"] = True
    term["anchor_sensitivity"]["sampled_displacement"] = {}
    assert B.LENS_PATH_TERM["measured"] is False
    fresh = B.anchor_sensitivity(1, base=GEMMA)["sampled_displacement"]
    assert fresh["float32"]["median"] == 1.035116


READING = {
    "positions": "all",
    "reduction": "none: one position at a time",
    "endpoint": "residual after block 17, repository layer 18",
    "context_tokens": 512,
}


def _register(monkeypatch, layer, identity, **overrides):
    """Register a pairing that identifies the pair it measured, optionally mutated."""
    pair = {**identity, **overrides}
    monkeypatch.setitem(
        B.anchor_table()[GEMMA]["measured_pairings"],
        str(layer),
        {"pair": pair, "relative": 0.004, "basis": "paired comparison, measured-here"},
    )


#: Two archives that a single fitting declaration could have produced. Equal nu, different bytes.
LENS_SHA, OTHER_LENS_SHA = "a" * 64, "b" * 64


def _reading_identity(capture_dtype="bfloat16", capture_batch=1, reading=READING,
                      lens_sha256=LENS_SHA):  # fmt: skip
    return B.reading_identity(
        FLOAT32_NU, lens_sha256=lens_sha256, capture_dtype=capture_dtype,
        capture_batch=capture_batch, reading=reading,
    )  # fmt: skip


def _record(**kwargs):
    call = {
        "declared": "float32", "layer": 1, "base": GEMMA, "lens_sha256": LENS_SHA,
        "capture_dtype": "bfloat16", "capture_batch": 1, "reading": READING,
    }  # fmt: skip
    call.update(kwargs)
    return B.fit_precision_record(FLOAT32_NU, **call)


def test_a_registered_pairing_lifts_the_refusal_only_for_the_pair_it_measured(monkeypatch):
    """A measurement is evidence about an experiment, so it licenses that experiment
    and no other."""
    identity = _reading_identity()
    _register(monkeypatch, 1, identity)

    # Positive control: the reading the measurement was taken on.
    record = _record()
    assert record["path_term"]["measured"] is True
    assert record["path_term"]["relative"] == 0.004

    # Every bound field, mutated one at a time: each is a different pair, so each is unmeasured.
    # The registered measurement stays put; only the reading moves.
    for field, changed in (
        ("capture_dtype", {"capture_dtype": "float16"}),
        ("capture_width", {"capture_batch": 64}),
        ("positions", {"reading": {**READING, "positions": "emitted token only"}}),
        ("reduction", {"reading": {**READING, "reduction": "mean over positions"}}),
        ("endpoint", {"reading": {**READING, "endpoint": "residual after block 16"}}),
        ("context_tokens", {"reading": {**READING, "context_tokens": 1400}}),
        ("lens_sha256", {"lens_sha256": OTHER_LENS_SHA}),
    ):
        with pytest.raises(ValueError, match="measured a different pair") as raised:
            _record(**changed)
        assert field in str(raised.value), field

    # The lens's own side is read from its nu and cannot be asserted by the caller: a different
    # lens is a different nu digest, and a lens fitted at another width is a different fit width.
    other_nu = {"precision": {**FLOAT32_NU["precision"], "forward_batch": 64}}
    with pytest.raises(ValueError, match="measured a different pair") as raised:
        B.fit_precision_record(
            other_nu, declared="float32", layer=1, base=GEMMA, lens_sha256=LENS_SHA,
            capture_dtype="bfloat16", capture_batch=1, reading=READING,
        )  # fmt: skip
    assert "fit_width" in str(raised.value) and "nu_sha256" in str(raised.value)


def test_one_declaration_with_two_archives_is_two_pairings(monkeypatch, parts):
    """A calibration keyed on the fitting metadata must not transfer to a different matrix.

    Equal declarations produce equal declaration digests by design, so this needs no collision:
    the nu says how a lens was fitted, and two archives fitted that way answer to it equally. The
    archive's own verified digest is what says which matrix was measured.
    """
    _register(monkeypatch, 1, _reading_identity(lens_sha256=LENS_SHA))

    # The archive the pairing was measured on: permitted.
    assert _record(lens_sha256=LENS_SHA)["path_term"]["measured"] is True

    # A different archive, byte for byte the same declaration: a different pairing, refused, and
    # the refusal names the archive rather than the declaration, which is identical on both sides.
    with pytest.raises(ValueError, match="measured a different pair") as raised:
        _record(lens_sha256=OTHER_LENS_SHA)
    message = str(raised.value)
    assert "lens_sha256" in message
    assert "nu_sha256" not in message

    # The reverse control, so the test cannot pass by ignoring the declaration: hold the archive
    # and change the declaration, and the guard refuses on that instead.
    other_nu = {"precision": {**FLOAT32_NU["precision"], "anchor_batch": 8}}
    with pytest.raises(ValueError, match="measured a different pair") as raised:
        B.fit_precision_record(
            other_nu, declared="float32", layer=1, base=GEMMA, lens_sha256=LENS_SHA,
            capture_dtype="bfloat16", capture_batch=1, reading=READING,
        )  # fmt: skip
    assert "nu_sha256" in str(raised.value)


def test_the_archive_digest_comes_off_the_loaded_lens_not_the_caller(parts, monkeypatch):
    """hook_alignment reads it from the lens object; nobody is asked to assert it."""
    d, _, lens = parts
    captured = {}

    def spy(nu, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(B, "fit_precision_record", spy)
    B.hook_alignment(d, lens, lens_fit_dtype="float32", nu=FLOAT32_NU)
    assert captured["lens_sha256"] == lens.sha256


def test_a_measurement_does_not_transfer_to_a_neighbouring_layer(monkeypatch):
    _register(monkeypatch, 1, _reading_identity())
    assert _record(layer=1)["path_term"]["measured"] is True
    with pytest.raises(ValueError, match="no cross-path pairing is registered at layer 17"):
        _record(layer=17)


def test_a_pairing_that_does_not_identify_its_pair_measures_nothing(monkeypatch):
    """The bare number this test used to insert, which is the defect the audit found."""
    monkeypatch.setitem(
        B.anchor_table()[GEMMA]["measured_pairings"], "1", {"relative": 0.004}
    )
    with pytest.raises(ValueError, match="does not identify its pair") as raised:
        _record()
    assert "must carry exactly" in str(raised.value)

    # And one that names the fields but omits a single field is equally not a measurement.
    identity = _reading_identity()
    partial = {key: value for key, value in identity.items() if key != "context_tokens"}
    monkeypatch.setitem(
        B.anchor_table()[GEMMA]["measured_pairings"], "1", {"pair": partial, "relative": 0.004}
    )
    with pytest.raises(ValueError, match="missing \\['context_tokens'\\]"):
        _record()


def test_a_reading_that_declares_no_provenance_matches_no_measurement(monkeypatch):
    _register(monkeypatch, 1, _reading_identity())
    with pytest.raises(ValueError, match="does not declare") as raised:
        _record(reading=None)
    message = str(raised.value)
    for field in B.READING_FIELDS:
        assert field in message
    assert "the caller supplies its own provenance" in message


def test_the_fit_precision_lands_in_the_provenance_block(parts, monkeypatch):
    d, u, lens = parts
    record = B.fit_precision_record(
        FLOAT32_NU, declared="float32", layer=10, base=GEMMA, lens_sha256=LENS_SHA
    )
    block = B.bridge_provenance(
        d, lens, u, 10, dictionary_repo="r", dictionary_folder="f", fit_precision=record
    )
    assert block["lens"]["fit_precision"]["fit_dtype"] == "float32"
    # No capture was read, so there is no crossing to record and none is invented.
    assert "path_term" not in block["lens"]["fit_precision"]

    # Where a pairing has been measured, the measurement is what the artefact carries.
    identity = B.reading_identity(
        FLOAT32_NU, lens_sha256=LENS_SHA, capture_dtype="bfloat16", capture_batch=1,
        reading=READING,
    )  # fmt: skip
    monkeypatch.setitem(
        B.anchor_table()[GEMMA]["measured_pairings"],
        "10",
        {"pair": identity, "relative": 0.004, "basis": "paired comparison, measured-here"},
    )
    measured = B.fit_precision_record(
        FLOAT32_NU, declared="float32", layer=10, base=GEMMA, lens_sha256=LENS_SHA,
        capture_dtype="bfloat16", capture_batch=1, reading=READING,
    )  # fmt: skip
    block = B.bridge_provenance(
        d, lens, u, 10, dictionary_repo="r", dictionary_folder="f", fit_precision=measured
    )
    assert block["lens"]["fit_precision"]["path_term"]["relative"] == 0.004
    # Absent by default, so an artefact that did not check cannot look as though it had.
    plain = B.bridge_provenance(d, lens, u, 10, dictionary_repo="r", dictionary_folder="f")
    assert plain["lens"]["fit_precision"] is None


@pytest.mark.parametrize(
    "path",
    [
        ("layers", "1", "amplification_ratio"),
        ("layers", "1", "sampled_displacement", "amplification_ratio"),
        ("layers", "1", "sampled_displacement", "float32", "amplification_ratio"),
        ("layers", "1", "native_anchor_read_in_the_float32_tail", "amplification_ratio"),
        ("amplification_ratio",),
    ],
)
def test_a_quantity_this_record_never_named_is_refused_at_every_nesting_location(path, tmp_path):
    """The withdrawn ratios cannot re-enter, at any depth.

    Not by banning numbers, which would be the wrong check twice over: medians, ranges and counts
    are numbers, and a legitimate statistic may coincidentally equal a withdrawn ratio. What is
    checked is the quantity at each location, so a key nobody declared is refused wherever it
    appears. The previous version of this test looked only at each layer's immediate keys and then
    scanned the top-level values, which are dictionaries, so it passed without descending at all.
    """
    table = json.loads(json.dumps(B.anchor_table()))
    node = table[GEMMA]
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = 539

    with pytest.raises(ValueError, match="undeclared"):
        B._validate_anchor_table(table)


def test_the_shipped_table_validates_and_says_at_which_precision_it_was_copied():
    entry = B.anchor_table()[GEMMA]
    assert "six decimal places" in entry["precision_convention"]
    # The one field the pairing audit found misrounded, at the declared precision.
    random_arm = entry["layers"]["33"]["equal_norm_random_displacement"]["native"]
    assert random_arm["median"] == 0.312953


# ------------------------------------------ device endpoint and independently declared errors


def _two_axis_parts():
    """h=(1,.1), reconstruction=(1,0), L=diag(.01,1), calculated by hand."""
    d = B.JumpReLUDictionary(
        w_enc=np.array([[1.0], [0.0]], np.float32),
        b_enc=np.zeros(1, np.float32),
        threshold=np.array([0.05], np.float32),
        w_dec=np.array([[1.0, 0.0]], np.float32),
        b_dec=np.zeros(2, np.float32),
        config={"model_name": FIXTURE_MODEL, "hf_hook_point_in": "model.layers.1.output"},
        sha256="fixture", source="fixture",
    )
    u = B.Unembedding(
        weight=np.diag([0.01, 1.0]).astype(np.float32), gain=np.ones(2, np.float32),
        tied=True, checkpoint="fixture", tensor_names={"head": "head", "norm": "norm"},
    )
    return d, u, np.array([1.0, 0.1], np.float32)


def _error_budget(**overrides):
    return B.ErrorBudget.from_dict({
        "raw_reconstruction_threshold": 0.5,
        "lens_score_error_threshold": 0.5,
        "denominator_floor": 1e-8,
        "near_zero_policy": "refuse",
        **overrides,
    })


@pytest.mark.parametrize("depth", [34, 48])
def test_final_dictionary_uses_only_a_declared_identity_endpoint(tmp_path, depth):
    from local_llm_lab.pipeline.live_lens.instruments import LensMaps, file_sha256

    d, u, h = _two_axis_parts()
    d = dataclasses.replace(
        d, config={**d.config, "hf_hook_point_in": f"model.layers.{depth-1}.output"}
    )
    archive = tmp_path / "lens.npz"
    np.savez(archive, **{f"J{i}": np.eye(2, dtype=np.float32) for i in range(depth - 1)})
    archive.with_suffix(".json").write_text(json.dumps({
        "npz_sha256": file_sha256(archive),
        "model": {"base": FIXTURE_MODEL, "num_layers": depth, "endpoint": "identity"},
    }))
    # Existing callers know the model but need not invent endpoint metadata to load the stamp.
    lens = LensMaps.load(
        archive, expected_sha256=file_sha256(archive), hidden_size=2,
        num_layers=depth, identity=LensIdentity(base=FIXTURE_MODEL, num_layers=depth),
    )
    assert B.hook_alignment(d, lens) == depth
    assert lens.identity.as_dict()["endpoint"] == "identity"
    J = B.lens_map_for_layer(lens, depth)
    assert J is None
    np.testing.assert_allclose(B.lens_scores(u, J, h)[0], [0.01, 0.1])
    control = B.negative_control(d, u, lens, depth, depth, k=1)
    assert control["mean_overlap"] == 1.0 and not control["distinguishes"]
    undeclared = dataclasses.replace(
        lens, identity=LensIdentity(base=FIXTURE_MODEL, num_layers=depth)
    )
    with pytest.raises(ValueError, match="identity endpoint"):
        B.hook_alignment(d, undeclared)
    missing = dataclasses.replace(
        d, config={**d.config, "hf_hook_point_in": "model.layers.1.output"}
    )
    broken = dataclasses.replace(lens, maps={1: np.eye(2)})
    with pytest.raises(ValueError, match="no map at.*layer 2"):
        B.hook_alignment(missing, broken)
    with pytest.raises(ValueError, match="cross-path reading"):
        B.hook_alignment(d, lens, nu=FLOAT32_NU, lens_fit_dtype="float32",
                         capture_dtype="bfloat16", capture_batch=1)
    sidecar = json.loads(archive.with_suffix(".json").read_text())
    del sidecar["model"]["endpoint"]
    archive.with_suffix(".json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="endpoint"):
        LensMaps.load(
            archive, expected_sha256=file_sha256(archive), hidden_size=2, num_layers=depth,
            identity=LensIdentity(base=FIXTURE_MODEL, num_layers=depth, endpoint="identity"),
        )


def test_unknown_endpoint_is_not_silently_loaded():
    with pytest.raises(ValueError, match="endpoint"):
        LensIdentity.from_dict({"base": FIXTURE_MODEL, "num_layers": 34, "endpoint": "unknown"})


def test_raw_admission_never_stands_in_for_score_ranking():
    d, u, h = _two_axis_parts()
    budget = _error_budget()
    raw = B.raw_reconstruction_budget(d, h[None, :], error_budget=budget)
    out = B.decompose_position(d, u, None, h, 1, error_budget=budget)
    assert raw["raw_reconstruction_share"] == pytest.approx([0.099503719])
    assert raw["raw_admissible"] == [True]
    assert out["raw_reconstruction_share"] == pytest.approx(0.099503719)
    assert out["lens_score_error_share"] == pytest.approx(0.99503719)
    assert out["raw_admissible"] is True and out["ranked"] is False
    assert "top_features" not in out
    assert out["error_budget"] == budget.as_dict()
    json.dumps(out, allow_nan=False)
    json.dumps(raw, allow_nan=False)


def test_a2_cannot_admit_without_an_explicit_error_budget():
    d, u, h = _two_axis_parts()
    with pytest.raises((TypeError, ValueError), match="error_budget"):
        B.decompose_position(d, u, None, h, 1)


@pytest.mark.parametrize("field,value", [
    ("raw_reconstruction_threshold", None), ("lens_score_error_threshold", None),
    ("denominator_floor", None), ("near_zero_policy", None),
    ("raw_reconstruction_threshold", float("nan")),
    ("lens_score_error_threshold", float("inf")),
    ("raw_reconstruction_threshold", -0.1), ("denominator_floor", 0.0),
    ("denominator_floor", float("nan")), ("near_zero_policy", "clamp"),
])
def test_error_budget_refuses_missing_or_invalid_declarations(field, value):
    with pytest.raises(ValueError, match=field):
        _error_budget(**{field: value})


@pytest.mark.parametrize("zero_in", ["raw", "score"])
def test_near_zero_denominators_refuse_instead_of_adding_epsilon(zero_in):
    d, u, h = _two_axis_parts()
    if zero_in == "raw":
        h = np.array([1e-10, 0.0], np.float32)
    else:
        u = dataclasses.replace(u, weight=np.zeros_like(u.weight))
    out = B.decompose_position(d, u, None, h, 1, error_budget=_error_budget())
    assert out["ranked"] is False and "top_features" not in out
    assert "denominator" in out["reason"]
    key = "raw_reconstruction_share" if zero_in == "raw" else "lens_score_error_share"
    assert out[key] is None
    json.dumps(out, allow_nan=False)


def test_external_pairing_table_is_checked_without_mutating_the_builtin_table():
    table = json.loads(B.ANCHOR_SENSITIVITY_PATH.read_text())
    table[GEMMA]["measured_pairings"]["1"] = {
        "pair": _reading_identity(), "relative": 0.004, "basis": "fixture measurement",
    }
    assert _record(pairing_table=table)["path_term"]["measured"] is True
    with pytest.raises(ValueError, match="cross-path reading"):
        _record()
    table[GEMMA]["measured_pairings"]["1"]["pair"]["lens_sha256"] = OTHER_LENS_SHA
    with pytest.raises(ValueError, match="lens_sha256"):
        _record(pairing_table=table)
    table[GEMMA]["layers"]["1"]["amplification_ratio"] = 539
    with pytest.raises(ValueError, match="undeclared"):
        _record(pairing_table=table)


@pytest.mark.parametrize(
    "measurement, diagnostic",
    [
        ({}, "missing"),
        ({"relative": 0.004}, "basis"),
        ({"basis": "paired measurement"}, "relative"),
        ({"relative": float("nan"), "basis": "paired measurement"}, "relative"),
        ({"relative": float("inf"), "basis": "paired measurement"}, "relative"),
        ({"relative": -0.004, "basis": "paired measurement"}, "relative"),
        ({"relative": True, "basis": "paired measurement"}, "relative"),
        ({"relative": "0.004", "basis": "paired measurement"}, "relative"),
        ({"relative": 0.004, "basis": " \t"}, "basis"),
        ({"relative": 0.004, "basis": None}, "basis"),
        ({"relative": 0.004, "basis": 1}, "basis"),
        ({"relative": 0.004, "basis": "paired measurement", "amplification_ratio": 539},
         "amplification_ratio"),
        ({"relative": 0.004, "basis": "paired measurement", "unapproved": {}}, "unapproved"),
    ],
)
def test_external_pairing_requires_a_finite_measured_term_and_declared_basis(
    measurement, diagnostic,
):
    table = json.loads(json.dumps(B.anchor_table()))
    table[GEMMA]["measured_pairings"]["1"] = {"pair": _reading_identity(), **measurement}
    with pytest.raises(ValueError, match=diagnostic):
        _record(pairing_table=table)


def test_merged_width_schedule_is_preserved_and_requires_its_own_pairing():
    nu = {"precision": {
        "fit_dtype": "float32", "forward_batch": [16, 16, 8], "anchor_batch": [16, 16, 8],
    }}
    with pytest.raises(ValueError, match="cross-path reading"):
        B.fit_precision_record(nu, declared="float32", layer=1, base=GEMMA,
                               capture_dtype="float32", capture_batch=1)
    pair = B.reading_identity(nu, lens_sha256=LENS_SHA, capture_dtype="float32",
                              capture_batch=1, reading=READING)
    assert pair["fit_width"] == [16, 16, 8]
    table = json.loads(json.dumps(B.anchor_table()))
    table[GEMMA]["measured_pairings"]["1"] = {
        "pair": pair, "relative": 0.004, "basis": "fixture measurement of merged schedule",
    }
    out = B.fit_precision_record(nu, declared="float32", layer=1, base=GEMMA,
                                 lens_sha256=LENS_SHA, capture_dtype="float32", capture_batch=1,
                                 reading=READING, pairing_table=table)
    assert out["path_term"]["measured"] and out["proposed_pair"]["fit_width"] == [16, 16, 8]


def test_score_ranking_does_not_consume_the_raw_admission_mask():
    d, u, _ = _two_axis_parts()
    u = dataclasses.replace(u, weight=np.diag([1.0, 0.01]).astype(np.float32))
    out = B.decompose_position(d, u, None, np.array([0.1, 1.0], np.float32), 0,
                               error_budget=_error_budget())
    assert out["raw_reconstruction_share"] == pytest.approx(0.99503719)
    assert out["lens_score_error_share"] == pytest.approx(0.099503719)
    assert out["raw_admissible"] is False and out["ranked"] is True
    assert out["top_features"][0]["feature"] == 0


@pytest.mark.parametrize("missing", ["raw_reconstruction_threshold", "lens_score_error_threshold",
                                     "denominator_floor", "near_zero_policy"])
def test_missing_budget_config_field_cannot_supply_an_admitting_default(missing):
    config = _error_budget().as_dict()
    del config[missing]
    with pytest.raises(ValueError, match=missing):
        B.ErrorBudget.from_dict(config)
