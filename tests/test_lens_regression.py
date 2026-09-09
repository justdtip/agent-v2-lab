"""Requirements §§2, 3.2, 4 and 12: native sufficient-statistics regression.

Collection and execution require a FREE MLX process slot. These tests initialize
native arrays/models and must never run beside a held model.
"""

import importlib

import mlx.core as mx
import numpy as np
import pytest


def api():
    return importlib.import_module("local_llm_lab.pipeline.lens_fitting.regression")


def test_fixed_grid_scaling_and_replication_invariance():
    """§12: replicating positions scales the penalty, never the fitted map."""
    reg = api()
    assert reg.ALPHA_GRID == (0.001, 0.01, 0.1, 1.0, 10.0)
    x = np.array([[1, 0], [0, 2], [2, 1]], dtype=np.float32)
    y = x @ np.array([[1, 3], [-2, 0.5]], dtype=np.float32)
    f, h = reg.SufficientStats.zeros(2), reg.SufficientStats.zeros(2)
    f.add(mx.array(x), mx.array(y))
    h.add(mx.array(x), mx.array(y + 0.1))
    first, info = reg.solve_layer(f, h)
    np.testing.assert_allclose(
        first,
        np.linalg.solve(x.T @ x + info["absolute_penalty"] * np.eye(2), x.T @ y),
        atol=1e-5,
    )
    assert info["alpha"] == 0.001
    assert info["dbar"] == pytest.approx(np.trace(x.T @ x) / 2)
    assert info["absolute_penalty"] == info["alpha"] * info["dbar"]
    assert info["lambda_per_position"] == info["absolute_penalty"] / info["n"]
    assert [row["alpha"] for row in info["candidates"]] == list(reg.ALPHA_GRID)
    f.add(mx.array(x), mx.array(y))
    h.add(mx.array(x), mx.array(y + 0.1))
    second, repeated = reg.solve_layer(f, h)
    np.testing.assert_allclose(first, second, atol=1e-5)
    assert repeated["absolute_penalty"] == 2 * info["absolute_penalty"]
    assert repeated["lambda_per_position"] == info["lambda_per_position"]


def test_held_error_matches_direct_computation():
    """§3.2: held sums reproduce direct squared error, with uncentered R²."""
    reg = api()
    rng = np.random.default_rng(7)
    x, y, w = [rng.normal(size=shape).astype(np.float32) for shape in [(8, 3), (8, 3), (3, 3)]]
    sums = reg.SufficientStats.zeros(3)
    for start in (0, 4):
        sums.add(mx.array(x[start : start + 4]), mx.array(y[start : start + 4]))
    error = reg.squared_error(sums, mx.array(w))
    assert error == pytest.approx(np.sum((y - x @ w) ** 2), rel=2e-6)
    assert sums.n == 8
    assert all(a.dtype == mx.float32 for a in (sums.xtx, sums.xty, sums.yty))


def test_streams_once_per_sequence_with_disjoint_sums():
    """§3.2: one all-layer forward, no held observations in fit sums."""
    reg = api()

    class View:
        hidden_size, num_layers = 2, 3

        def __init__(self):
            self.calls = []

        def residuals(self, ids, layers):
            self.calls.append((ids, tuple(layers)))
            x = mx.array([[[float(i), 1] for i in ids]], dtype=mx.float16)
            return {1: x, 2: x * 2, 3: x * 3}

    view = View()
    rows = [{"ids": [1, 2], "split": "fit"}, {"ids": [9], "split": "held"}]
    sums, counts = reg.accumulate(view, rows)
    assert view.calls == [([1, 2], (1, 2, 3)), ([9], (1, 2, 3))]
    assert counts == {
        # Which forward produced the residuals travels with the totals, so a fitted lens says
        # it rather than depending on a caller to stamp it (the Gemma pivot's native source).
        "residual_source": "hand_run",
        "fit": {"sequences": 1, "positions": 2},
        "held": {"sequences": 1, "positions": 1},
    }
    np.testing.assert_array_equal(sums["fit"][1].xtx, [[5, 3], [3, 2]])
    np.testing.assert_array_equal(sums["held"][1].xtx, [[81, 9], [9, 1]])
    assert sums["fit"][2].xtx.dtype == mx.float32


def test_native_nonsymmetric_linear_recovery_and_all_layer_archive(tmp_path):
    """§§2/3.2/4: native tiny Qwen residuals recover a nonsymmetric map as J=W.T."""
    from test_live_lens_native import tiny_model

    from local_llm_lab.arch import ArchitectureView
    from local_llm_lab.pipeline.lens_fitting.artifacts import write_lens
    from local_llm_lab.pipeline.live_lens.instruments import LensIdentity, LensMaps

    reg = api()
    native = ArchitectureView.from_model(tiny_model())
    d = native.hidden_size
    # The nonzero ridge has a known small shrinkage; enough native contexts span the space.
    known = np.eye(d, dtype=np.float32) + np.diag(np.full(d - 1, 0.25), 1)
    calls = []

    class LinearTarget:
        hidden_size, num_layers = d, native.num_layers

        def residuals(self, ids, layers):
            calls.append(tuple(layers))
            residuals = native.residuals(ids, layers)
            residuals[native.num_layers] = residuals[1].astype(mx.float32) @ mx.array(known)
            return residuals

    rng = np.random.default_rng(51)
    rows = [
        {
            "ids": rng.integers(0, native.vocab_size, size=24).tolist(),
            "split": "held" if (i + 1) % 5 == 0 else "fit",
        }
        for i in range(30)
    ]
    result = reg.fit_regression(LinearTarget(), rows)
    assert len(calls) == len(rows)
    recovered = result.maps[1].T
    assert np.linalg.norm(recovered - known) / np.linalg.norm(known) < 0.08
    assert np.linalg.norm(recovered.T - known) > np.linalg.norm(recovered - known) * 2
    out = tmp_path / "tiny-agentic-regression.npz"
    identity = LensIdentity(base="example/tiny", num_layers=native.num_layers)
    metadata = write_lens(
        out,
        result.maps,
        hidden_size=d,
        num_layers=native.num_layers,
        metadata={"kind": "regression", "domain": "agentic"},
        identity=identity,
    )
    loaded = LensMaps.load(
        out,
        expected_sha256=metadata["npz_sha256"],
        hidden_size=d,
        num_layers=native.num_layers,
        identity=identity,
    )
    assert set(loaded.maps) == set(range(1, native.num_layers))
    with np.load(out, allow_pickle=False) as archive:
        # `identity` beside the maps: a lens says which model it was fitted on, and it goes
        # inside the archive so the digest every reader pins covers it (issue 99).
        maps = {f"J{i}" for i in range(native.num_layers - 1)}
        assert set(archive.files) == maps | {"identity"}
        assert all(archive[k].dtype == np.float32 for k in maps)
    probe = rng.normal(size=(4, d)).astype(np.float32)
    np.testing.assert_allclose(loaded.apply(probe, 1), probe @ recovered, atol=2e-6)
    assert all("held_out_uncentered_r2" in row for row in result.per_layer.values())


@pytest.mark.parametrize("problem", ["empty_held", "invalid_split", "nonfinite", "wrong_shape"])
def test_refuses_invalid_stream(problem):
    """§3.2: fail closed instead of emitting partial or invalid all-layer fits."""

    class View:
        hidden_size, num_layers = 2, 2

        def residuals(self, ids, layers):
            a = mx.ones((1, len(ids), 2))
            if problem == "nonfinite":
                a = a * float("nan")
            if problem == "wrong_shape":
                a = a[:, :, :1]
            return {1: a, 2: a}

    rows = [{"ids": [1], "split": "fit"}]
    if problem != "empty_held":
        rows.append({"ids": [2], "split": "other" if problem == "invalid_split" else "held"})
    with pytest.raises(ValueError):
        api().fit_regression(View(), rows)


def test_held_targets_select_alpha_instead_of_fit_error():
    """§3.2: held sequences choose alpha; large shrinkage can beat the fit optimum."""
    reg = api()
    fit, held = reg.SufficientStats.zeros(2), reg.SufficientStats.zeros(2)
    x = mx.eye(2, dtype=mx.float32)
    fit.add(x, 10 * x)
    held.add(x, x)
    w, info = reg.solve_layer(fit, held)
    assert info["alpha"] == 10.0
    np.testing.assert_allclose(w, np.eye(2) * (10 / 11), atol=1e-6)
    for row in info["candidates"]:
        direct = np.eye(2) * (10 / (1 + row["alpha"]))
        expected = np.sum((np.eye(2) - direct) ** 2)
        assert row["held_out_squared_error"] == pytest.approx(expected, rel=2e-6, abs=2e-6)


def test_the_two_residual_sources_agree_exactly_on_a_model_whose_loop_is_correct():
    """The substitution is demonstrated, not argued.

    `view.residuals` re-runs the decoder through this repository's own loop; `native_residuals`
    taps the model's forward. Where the loop describes the family correctly the two are the same
    numbers, and a fit may take either. Where it does not -- Gemma 3, whose entry scale the loop
    omits and whose second mask it does not build -- they differ, and `residual_source_agreement`
    reports by how much at each layer, which is the architecture port's acceptance evidence.

    A hybrid at toy width, so both mask kinds and both block kinds are exercised on the family
    the loop was written for. Exact equality, not a tolerance: the same arithmetic in the same
    order should give the same bits, and a tolerance here would hide the very drift this exists
    to detect.
    """
    import mlx.core as mx
    from test_live_lens_native import tiny_model

    from local_llm_lab.arch import ArchitectureView

    model = tiny_model()
    view = ArchitectureView.from_model(model)
    ids = mx.array([[1, 2, 3, 4, 5]])
    layers = tuple(range(1, view.num_layers + 1))

    loop = view.residuals(ids, layers)
    native = view.native_residuals(ids, layers)

    assert set(native) == set(loop)
    for layer in layers:
        assert native[layer].shape == loop[layer].shape == (1, 5, view.hidden_size)
    assert view.residual_source_agreement(ids, layers) == dict.fromkeys(layers, 0.0)


def test_a_fit_records_which_forward_produced_its_residuals():
    """A fit whose residual source changed silently would be unattributable afterwards.

    The numbers are the same shape from either producer and nothing else in the artifact would
    say which forward made them, so the choice is explicit at the call site, refuses an unknown
    name, and travels to the record inside `counts`.
    """
    from test_live_lens_native import tiny_model

    from local_llm_lab.arch import ArchitectureView
    from local_llm_lab.pipeline.lens_fitting import regression as reg

    model = tiny_model()
    view = ArchitectureView.from_model(model)
    rows = [
        {"split": "fit", "ids": [1, 2, 3]},
        {"split": "held", "ids": [4, 5, 6]},
    ]

    for source in reg.RESIDUAL_SOURCES:
        _, counts = reg.accumulate(view, rows, residual_source=source)
        assert counts["residual_source"] == source
        assert counts["fit"]["positions"] == 3 and counts["held"]["positions"] == 3

    with pytest.raises(ValueError, match="residual_source must be one of"):
        reg.accumulate(view, rows, residual_source="whatever_the_model_needs")
