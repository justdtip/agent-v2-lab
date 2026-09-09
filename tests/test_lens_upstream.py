"""The WS-D step-1 adapter, on a tiny synthetic torch decoder.

No 4B checkpoint, no MLX, no model-run lock. The model here is four small blocks with real causal
attention, and the attention is the point: upstream's own ``tests/tiny.py`` uses position-wise
blocks (``h + Wh``) where ``d h_target[p'] / d h_source[p] = 0`` for every ``p' != p``, so the
sum-over-targets estimator, a strict per-position estimator and an unmasked-target estimator all
agree there to 1e-7. A gate that cannot fail is not a gate. With attention present, off-diagonal
mass exists and the position rule is decidable — which is what
``test_the_default_selector_is_upstreams_own_mask`` and the negative controls below rely on.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from local_llm_lab.pipeline.lens_fitting import upstream as adapter  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.artifacts import write_lens  # noqa: E402
from local_llm_lab.pipeline.live_lens.instruments import (  # noqa: E402
    LensIdentity,
    LensIdentityError,
    LensMaps,
)

HIDDEN = 6
NUM_LAYERS = 4  # decoder blocks; repo layers 1..4, maps at 1..3, layer 4 is the identity
VOCAB = 23
#: Long enough that upstream's default mask (skip 16, drop the last) leaves real positions.
SEQ_LEN = 40


# --------------------------------------------------------------------------- the tiny model


class CausalBlock(torch.nn.Module):
    """One head of single-head causal attention plus an MLP, residual on both.

    Attention is what makes ``d h_target[p'] / d h_source[p]`` nonzero for ``p' > p``. Without it
    the estimator's target reduction is unobservable and every candidate rule agrees.
    """

    def __init__(self, hidden: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)

        def small(rows: int, cols: int) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                torch.randn(rows, cols, generator=generator, dtype=torch.float32) * 0.3
            )

        self.q, self.k, self.v = small(hidden, hidden), small(hidden, hidden), small(hidden, hidden)
        self.up, self.down = small(hidden, 2 * hidden), small(2 * hidden, hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores = (h @ self.q) @ (h @ self.k).transpose(-1, -2) / h.shape[-1] ** 0.5
        mask = torch.triu(torch.ones(h.shape[1], h.shape[1], dtype=torch.bool), diagonal=1)
        attended = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1) @ (h @ self.v)
        h = h + attended
        return h + torch.tanh(h @ self.up) @ self.down


class TinyLensModel(torch.nn.Module):
    """Satisfies ``jlens.protocol.LensModel``: n_layers, d_model, layers, forward, unembed."""

    def __init__(self, hidden: int = HIDDEN, n_layers: int = NUM_LAYERS, vocab: int = VOCAB):
        super().__init__()
        self.n_layers, self.d_model = n_layers, hidden
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList(
            CausalBlock(hidden, seed=100 + i) for i in range(n_layers)
        )
        self.norm = torch.nn.LayerNorm(hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)
        self.tokenizer = None
        # Every weight from a local generator, never the global RNG: two instances must be the
        # same model, or "dim_batch is numerically inert" would be comparing two different fits.
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab, hidden, generator=generator))
            self.head.weight.copy_(torch.randn(vocab, hidden, generator=generator))
        # Load-bearing, not cosmetic: upstream roots the graph with ``requires_grad_(True)`` on a
        # block output, which only works while that output is a leaf.
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @property
    def input_device(self) -> torch.device:
        return self.embed.weight.device

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        for block in self.layers:
            h = block(h)
        return h

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(residual))


def make_rows(count: int = 3, seq_len: int = SEQ_LEN) -> list[dict]:
    generator = np.random.default_rng(11)
    return [
        {
            "index": index,
            "split": "fit",
            "domain": "prose",
            "source": "synthetic",
            "n_prompt": 0,
            "ids": generator.integers(0, VOCAB, size=seq_len).tolist(),
            "spans": ["chat"] * seq_len,
        }
        for index in range(count)
    ]


CORPUS = {"manifest_sha256": "0" * 64, "domain": "prose", "split": "fit", "synthetic": True}
IDENTITY = LensIdentity("tiny-synthetic-decoder", NUM_LAYERS)


@pytest.fixture(scope="module")
def upstream():
    try:
        return adapter.load_upstream()
    except adapter.UpstreamUnavailable as error:  # pragma: no cover - box without the clone
        pytest.skip(str(error))


@pytest.fixture
def fitted(upstream):
    return adapter.fit_upstream_jacobian(
        TinyLensModel(), make_rows(), dim_batch=3, max_seq_len=SEQ_LEN, upstream=upstream
    )


# --------------------------------------------------------------- the clone, called not vendored


def test_a_missing_clone_names_the_path_it_looked_for(tmp_path):
    with pytest.raises(adapter.UpstreamUnavailable) as raised:
        adapter.load_upstream(tmp_path / "nowhere")
    assert str(tmp_path / "nowhere") in str(raised.value)
    assert adapter.JLENS_PATH_ENV in str(raised.value)


def test_upstream_is_imported_from_outside_this_package(upstream):
    assert "local_llm_lab" not in str(upstream.path)
    assert (upstream.path / "jlens" / "fitting.py").is_file()
    assert upstream.provenance()["vendored"] is False


# ------------------------------------------------------------------- the cotangent selector seam


def test_the_default_selector_is_upstreams_own_mask(upstream):
    """Not "agrees with" — *is*. The default path calls upstream's function object."""
    default = adapter.default_position_selector(upstream=upstream)
    for seq_len in (18, 40, 128):
        theirs = upstream.fitting.valid_position_mask(seq_len, skip_first=16)
        assert torch.equal(default(seq_len), theirs)
    assert upstream.skip_first_default == 16


def test_a_supplied_selector_covering_the_default_reproduces_the_default_fit(upstream):
    """The order's golden test for the seam: the canonical ν recovers the plain fit."""
    rows = make_rows()
    plain = adapter.fit_upstream_jacobian(
        TinyLensModel(), rows, dim_batch=3, max_seq_len=SEQ_LEN, upstream=upstream
    )
    banded = adapter.fit_upstream_jacobian(
        TinyLensModel(),
        rows,
        dim_batch=3,
        max_seq_len=SEQ_LEN,
        upstream=upstream,
        position_selector=adapter.default_position_selector(upstream=upstream),
    )
    for layer in plain.jacobians:
        assert np.array_equal(plain.jacobians[layer], banded.jacobians[layer])
    # ... and the seam still records that it was taken, so ν does not read as the default path.
    assert plain.selector["upstream_default_path"] is True
    assert banded.selector["upstream_default_path"] is False


def test_the_seam_is_a_real_seam_and_a_different_band_changes_the_fit(upstream):
    """Negative control for the test above.

    If a narrower band produced the same J, the seam would be decoration and the equality test
    above would prove nothing.
    """
    rows = make_rows()
    plain = adapter.fit_upstream_jacobian(
        TinyLensModel(), rows, dim_batch=3, max_seq_len=SEQ_LEN, upstream=upstream
    )

    def band(seq_len: int) -> torch.Tensor:
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[20:30] = True
        return mask

    narrowed = adapter.fit_upstream_jacobian(
        TinyLensModel(),
        rows,
        dim_batch=3,
        max_seq_len=SEQ_LEN,
        upstream=upstream,
        position_selector=band,
    )
    assert not np.allclose(plain.jacobians[0], narrowed.jacobians[0])


def test_an_empty_selection_is_refused_and_not_swallowed_as_a_skip(upstream):
    with pytest.raises(adapter.CotangentSelectionError):
        adapter.fit_upstream_jacobian(
            TinyLensModel(),
            make_rows(),
            dim_batch=3,
            max_seq_len=SEQ_LEN,
            upstream=upstream,
            position_selector=lambda seq_len: torch.zeros(seq_len, dtype=torch.bool),
        )
    # Upstream's fit() absorbs ValueError as "skip this prompt"; this must not be absorbable.
    assert not issubclass(adapter.CotangentSelectionError, ValueError)


def test_the_module_attribute_is_restored_after_a_selector_fit(upstream):
    before = upstream.fitting.valid_position_mask
    with pytest.raises(adapter.CotangentSelectionError):
        adapter.fit_upstream_jacobian(
            TinyLensModel(),
            make_rows(),
            dim_batch=3,
            max_seq_len=SEQ_LEN,
            upstream=upstream,
            position_selector=lambda seq_len: torch.zeros(seq_len, dtype=torch.bool),
        )
    assert upstream.fitting.valid_position_mask is before


# ------------------------------------------------------------------------------- the fit itself


def test_short_rows_are_counted_as_skips_not_absorbed(upstream):
    rows = make_rows(count=2) + [
        {**make_rows(count=1, seq_len=10)[0], "index": 99},
    ]
    fit = adapter.fit_upstream_jacobian(
        TinyLensModel(), rows, dim_batch=3, max_seq_len=SEQ_LEN, upstream=upstream
    )
    assert fit.n_prompts == 2
    assert [skip["index"] for skip in fit.skipped] == [99]


def test_a_corpus_with_no_usable_row_refuses_rather_than_returning_a_lens(upstream):
    with pytest.raises(ValueError, match="no corpus row"):
        adapter.fit_upstream_jacobian(
            TinyLensModel(),
            make_rows(count=2, seq_len=10),
            dim_batch=3,
            max_seq_len=SEQ_LEN,
            upstream=upstream,
        )


def test_declared_precision_must_be_the_observed_precision(upstream):
    model = TinyLensModel()
    with pytest.raises(ValueError, match="must be a measurement"):
        adapter.fit_upstream_jacobian(
            model,
            make_rows(),
            dim_batch=3,
            max_seq_len=SEQ_LEN,
            upstream=upstream,
            dtype="bfloat16",
        )


def test_an_unfrozen_model_is_refused_with_the_reason(upstream):
    model = TinyLensModel()
    model.layers[0].q.requires_grad_(True)
    with pytest.raises(ValueError, match="requires grad"):
        adapter.fit_upstream_jacobian(
            model, make_rows(), dim_batch=3, max_seq_len=SEQ_LEN, upstream=upstream
        )


def test_dim_batch_is_numerically_inert(upstream):
    rows = make_rows(count=2)
    one = adapter.fit_upstream_jacobian(
        TinyLensModel(), rows, dim_batch=1, max_seq_len=SEQ_LEN, upstream=upstream
    )
    four = adapter.fit_upstream_jacobian(
        TinyLensModel(), rows, dim_batch=4, max_seq_len=SEQ_LEN, upstream=upstream
    )
    for layer in one.jacobians:
        assert np.allclose(one.jacobians[layer], four.jacobians[layer], atol=1e-6)


def test_neither_end_relies_on_an_upstream_length_default(upstream):
    """Upstream fits at 128 and reads out at 512, both by default, and neither signature warns.

    The order's rule is that the adapter passes lengths explicitly at both ends. Enforced, not
    remembered: ``max_seq_len`` is required, and the wrapper's ``encode`` refuses to fall back to
    ``HFLensModel``'s 512.
    """
    with pytest.raises(TypeError, match="max_seq_len"):
        adapter.fit_upstream_jacobian(TinyLensModel(), make_rows(), upstream=upstream)

    wrapped = adapter.CorpusLensModel(TinyLensModel())
    wrapped.register("row:0", make_rows(count=1)[0]["ids"])
    with pytest.raises(ValueError, match="explicit max_length"):
        wrapped.encode("row:0")
    assert wrapped.encode("row:0", max_length=20).shape == (1, 20)


def test_the_attention_implementation_is_recorded_as_none_when_it_cannot_be_read(fitted):
    """None, not "eager". Upstream never sets it, and a hopeful default is a fabricated figure."""
    assert fitted.precision["attn_implementation"] is None


def test_the_corpus_ids_are_what_upstream_fitted_on(upstream):
    """The wrapper never tokenizes text; if upstream stopped calling encode the fit would drift."""
    wrapped = adapter.CorpusLensModel(TinyLensModel())
    adapter.fit_upstream_jacobian(
        wrapped, make_rows(count=2), dim_batch=3, max_seq_len=SEQ_LEN, upstream=upstream
    )
    assert wrapped.encode_calls == 2
    with pytest.raises(KeyError):
        wrapped.encode("some prose that was never registered", max_length=SEQ_LEN)


# ---------------------------------------------------------------------------- the artefact


def test_the_adapter_writes_an_artefact_our_own_loader_accepts(tmp_path, fitted, upstream):
    path = tmp_path / "tiny_upstream_jacobian.npz"
    written = adapter.write_upstream_lens(
        path,
        fitted,
        identity=IDENTITY,
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        corpus=CORPUS,
        upstream=upstream,
    )
    loaded = LensMaps.load(
        path,
        expected_sha256=written["npz_sha256"],
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        identity=IDENTITY,
    )
    assert set(loaded.maps) == {1, 2, 3}
    assert loaded.identity == IDENTITY
    with np.load(path) as archive:
        assert set(archive.files) == {"J0", "J1", "J2", "identity"}
    # Repo layer L carries upstream block L-1, copied with no transpose.
    for index, matrix in fitted.jacobians.items():
        assert np.array_equal(loaded.maps[adapter.repo_layer_of_upstream(index)], matrix)
    assert adapter.upstream_index_of_repo_layer(3) == 2


def test_the_declared_nu_round_trips(tmp_path, fitted, upstream):
    path = tmp_path / "nu_round_trip.npz"
    written = adapter.write_upstream_lens(
        path,
        fitted,
        identity=IDENTITY,
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        corpus=CORPUS,
        upstream=upstream,
    )
    nu = adapter.read_declared_nu(path)
    assert nu == written["nu"] == json.loads(path.with_suffix(".json").read_text())["nu"]

    # All six §6.1 fields, each saying what it is a statement about.
    assert (
        nu["estimator"] == adapter.ESTIMATOR_EXACT_AUTOGRAD != adapter.ESTIMATOR_FINITE_DIFFERENCE
    )
    assert nu["endpoint"]["target_layer_upstream"] == NUM_LAYERS - 1
    assert nu["endpoint"]["target_layer_repo"] == NUM_LAYERS
    assert nu["endpoint"]["source_layers_repo"] == [1, 2, 3]
    assert nu["position_weighting"]["source_reduction"].startswith("mean")
    assert nu["position_weighting"]["target_reduction"].startswith("sum")
    assert nu["position_weighting"]["skip_first"] == 16
    assert nu["position_weighting"]["n_valid_positions"]["min"] == SEQ_LEN - 16 - 1
    assert nu["pair_weighting"]["n_prompts"] == 3
    assert nu["pair_weighting"]["per_prompt"].startswith("equal")
    assert nu["corpus"] == CORPUS
    assert nu["precision"]["device"] == "cpu" and nu["precision"]["dtype"] == "float32"
    assert nu["precision"]["backward_accumulation_dtype"] == "float32"
    # The fitted length is recorded, and so is the fact that reading beyond it is extrapolation.
    assert nu["position_weighting"]["max_seq_len"] == SEQ_LEN
    assert "extrapolation" in nu["position_weighting"]["readout_warning"]


def test_a_lens_without_a_declared_nu_is_refused(tmp_path, fitted, upstream):
    """NEGATIVE CONTROL for ν: an absent declaration must not read like a canonical one."""
    path = tmp_path / "nu_present.npz"
    written = adapter.write_upstream_lens(
        path,
        fitted,
        identity=IDENTITY,
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        corpus=CORPUS,
        upstream=upstream,
    )
    assert adapter.read_declared_nu(path)["estimator"] == adapter.ESTIMATOR_EXACT_AUTOGRAD

    stripped = tmp_path / "nu_absent.npz"
    shutil.copyfile(path, stripped)
    meta = json.loads(path.with_suffix(".json").read_text())
    meta.pop("nu")
    stripped.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(adapter.MissingDeclaredNu, match="no 'nu' block"):
        adapter.read_declared_nu(stripped)

    # ... and a ν sitting beside a different file is not evidence about this one.
    moved = tmp_path / "nu_misbound.npz"
    shutil.copyfile(path, moved)
    moved.with_suffix(".json").write_text(
        json.dumps(json.loads(path.with_suffix(".json").read_text()) | {"npz_sha256": "0" * 64})
    )
    with pytest.raises(adapter.MissingDeclaredNu, match="different file"):
        adapter.read_declared_nu(moved)
    assert written["npz_sha256"] != "0" * 64


# ------------------------------------------------------------------------- negative controls


def _write_raw(path, maps, upstream_unused=None):
    """Write an artefact straight through ``write_lens``, bypassing the adapter's conversion."""
    return write_lens(
        path,
        maps,
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        metadata={"deliberately": "wrong, for a negative control"},
        identity=IDENTITY,
    )


def test_a_transposed_artefact_is_refused(tmp_path, fitted, upstream):
    """The house rule: a gate that cannot fail is not a gate.

    Our loader takes a transposed lens without complaint — the maps are square, finite and
    correctly named — and every downstream read would then be a precise measurement of the wrong
    artefact. This test writes exactly that file, shows ``LensMaps.load`` accepts it, and shows the
    orientation check against upstream's own ``transport`` refuses it.
    """
    path = tmp_path / "transposed.npz"
    written = _write_raw(
        path, {adapter.repo_layer_of_upstream(i): m.T.copy() for i, m in fitted.jacobians.items()}
    )
    loaded = LensMaps.load(
        path,
        expected_sha256=written["npz_sha256"],
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        identity=IDENTITY,
    )
    assert set(loaded.maps) == {1, 2, 3}  # the loader alone cannot catch this

    with pytest.raises(ValueError, match="different orientation or a different layer convention"):
        adapter.assert_orientation_matches_upstream(loaded, fitted, upstream=upstream)


def test_a_layer_shifted_artefact_is_refused(tmp_path, fitted, upstream):
    """Geometry cannot catch this — adjacent hosted maps are 0.72-0.98 alike. Upstream can."""
    order = sorted(fitted.jacobians)
    rotated = {
        adapter.repo_layer_of_upstream(index): fitted.jacobians[order[(n + 1) % len(order)]]
        for n, index in enumerate(order)
    }
    path = tmp_path / "shifted.npz"
    written = _write_raw(path, rotated)
    loaded = LensMaps.load(
        path,
        expected_sha256=written["npz_sha256"],
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        identity=IDENTITY,
    )
    assert set(loaded.maps) == {1, 2, 3}

    with pytest.raises(ValueError, match="does not transport like upstream block"):
        adapter.assert_orientation_matches_upstream(loaded, fitted, upstream=upstream)


def test_the_orientation_check_passes_on_the_artefact_the_adapter_writes(
    tmp_path, fitted, upstream
):
    """The positive half of the two controls above, on the real write path."""
    written = adapter.write_upstream_lens(
        tmp_path / "correct.npz",
        fitted,
        identity=IDENTITY,
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        corpus=CORPUS,
        upstream=upstream,
    )
    assert set(written["orientation_check"]) == {"1", "2", "3"}
    assert all(value < 1e-4 for value in written["orientation_check"].values())
    # The maps are neither symmetric nor alike layer to layer, so the two controls above are
    # discriminating rather than lucky.
    assert not np.allclose(fitted.jacobians[0], fitted.jacobians[0].T)
    assert not np.allclose(fitted.jacobians[0], fitted.jacobians[1])


def test_a_lens_of_another_model_is_refused_by_identity(tmp_path, fitted, upstream):
    path = tmp_path / "identity_control.npz"
    written = adapter.write_upstream_lens(
        path,
        fitted,
        identity=IDENTITY,
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        corpus=CORPUS,
        upstream=upstream,
    )
    with pytest.raises(LensIdentityError):
        LensMaps.load(
            path,
            expected_sha256=written["npz_sha256"],
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            identity=LensIdentity("some-other-decoder", NUM_LAYERS),
        )


def test_a_selector_fit_may_not_be_labelled_a_hosted_recipe_fit(tmp_path, upstream):
    """A band lens is a different estimator by construction.

    It must not inherit the hosted-recipe label, because §6.2 bands are never to be compared with
    the hosted lens as if they were the same instrument.
    """

    def band(seq_len: int) -> torch.Tensor:
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[20:30] = True
        return mask

    departed = adapter.fit_upstream_jacobian(
        TinyLensModel(),
        make_rows(),
        dim_batch=3,
        max_seq_len=SEQ_LEN,
        upstream=upstream,
        position_selector=band,
    )
    with pytest.raises(ValueError, match="declared departure"):
        adapter.write_upstream_lens(
            tmp_path / "band.npz",
            departed,
            identity=IDENTITY,
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            corpus=CORPUS,
            upstream=upstream,
        )
    written = adapter.write_upstream_lens(
        tmp_path / "band_named.npz",
        departed,
        identity=IDENTITY,
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        corpus=CORPUS,
        kind="jacobian-position-band",
        upstream=upstream,
    )
    assert written["kind"] == "jacobian-position-band"
    assert written["nu"]["position_weighting"]["upstream_default_path"] is False


def test_a_partial_fit_cannot_be_written(tmp_path, fitted, upstream):
    partial = adapter.UpstreamJacobianFit(
        **{**fitted.__dict__, "jacobians": {0: fitted.jacobians[0]}}
    )
    with pytest.raises(ValueError, match="Merge shards"):
        adapter.write_upstream_lens(
            tmp_path / "partial.npz",
            partial,
            identity=IDENTITY,
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            corpus=CORPUS,
            upstream=upstream,
        )


# ------------------------------------------------------ what the estimator is a measurement of


def test_the_estimator_sums_over_targets_and_means_over_sources(upstream):
    """Pin the reduction on a model where the candidates actually differ.

    The rival reductions agree to 1e-7 on a decoder without attention, so this is checked here
    where off-diagonal mass exists. If this ever starts passing on a position-wise toy, the check
    has stopped being able to fail.
    """
    model = TinyLensModel()
    ids = torch.as_tensor(make_rows(count=1)[0]["ids"], dtype=torch.int64).reshape(1, -1)
    wrapped = adapter.CorpusLensModel(model)
    wrapped.register("probe", ids[0].tolist())
    jacobians, seq_len, n_valid = upstream.fitting.jacobian_for_prompt(
        wrapped, "probe", [0], target_layer=NUM_LAYERS - 1, dim_batch=2, max_seq_len=SEQ_LEN
    )
    assert (seq_len, n_valid) == (SEQ_LEN, SEQ_LEN - 16 - 1)

    mask = upstream.fitting.valid_position_mask(seq_len, skip_first=16)
    positions = mask.nonzero(as_tuple=True)[0]
    reference = torch.zeros(HIDDEN, HIDDEN, dtype=torch.float64)
    for out_dim in range(HIDDEN):
        source = model.embed(ids)
        source = model.layers[0](source).detach().requires_grad_(True)
        h = source
        for block in model.layers[1:]:
            h = block(h)
        cotangent = torch.zeros_like(h)
        cotangent[0, positions, out_dim] = 1.0
        (grad,) = torch.autograd.grad(h, source, grad_outputs=cotangent)
        reference[out_dim] = grad[0, positions, :].double().mean(dim=0)
    assert np.allclose(jacobians[0].numpy(), reference.numpy(), atol=1e-5)

    # Negative control: the same object without the target sum is a different matrix, so the
    # assertion above is discriminating rather than trivially true.
    per_position = torch.zeros(HIDDEN, HIDDEN, dtype=torch.float64)
    for out_dim in range(HIDDEN):
        source = model.embed(ids)
        source = model.layers[0](source).detach().requires_grad_(True)
        h = source
        for block in model.layers[1:]:
            h = block(h)
        rows = []
        for position in positions.tolist():
            cotangent = torch.zeros_like(h)
            cotangent[0, position, out_dim] = 1.0
            (grad,) = torch.autograd.grad(h, source, grad_outputs=cotangent, retain_graph=True)
            rows.append(grad[0, position, :].double())
        per_position[out_dim] = torch.stack(rows).mean(dim=0)
    assert not np.allclose(jacobians[0].numpy(), per_position.numpy(), atol=1e-5)


# --------------------------------------------------------- what the adversarial passes found


def test_precision_is_measured_across_every_block_and_a_mixed_model_is_refused(tmp_path):
    """The one-sample read measured the one block that does not set the accumulation dtype.

    `_observed_precision` returned the first parameter of the first block. The cotangent is
    `zeros_like(target_activation)`, and the target activation is the *last* block's output, so the
    backward accumulates in the target block's dtype. An adversarial pass built a model with block 0
    in float32 and blocks 1 to 3 in bfloat16, declared `float32`, and was accepted — while the
    backward really ran in bfloat16 and J moved 0.37% of scale.

    Reachable on the CUDA leg through `device_map="auto"`, bitsandbytes and `_keep_in_fp32_modules`,
    none of which are exotic. Refused rather than described, because a mixed model has no single
    true answer to put in a field that declares one.
    """
    import torch

    from local_llm_lab.pipeline.lens_fitting.upstream import (
        MixedPrecisionModel,
        _observed_precision,
    )

    uniform = TinyLensModel()
    observed = _observed_precision(uniform)
    assert observed["dtype"] == "float32"
    assert observed["dtypes_observed"] == ["float32"], "the set is recorded, not one sample"
    assert observed["blocks_measured"] == len(list(uniform.layers)), "every block was looked at"

    mixed = TinyLensModel()
    for block in list(mixed.layers)[1:]:
        block.to(torch.bfloat16)
    with pytest.raises(MixedPrecisionModel, match="not uniform"):
        _observed_precision(mixed)


def test_the_selector_fingerprint_separates_two_selectors_the_summary_could_not(tmp_path):
    """Two incomparable lenses must not share a position-weighting hash.

    An adversarial pass built the pair below: same count, same first, same last, both
    non-contiguous, so all four summary numbers matched and the whole `position_weighting` block
    and its sha256 collided — while J differed by 17.5% of scale. A fingerprint that cannot
    separate two incomparable lenses is the appearance of one.
    """
    from local_llm_lab.pipeline.lens_fitting.upstream import _runs, selector_descriptor

    left = [20, 21, 22, 23, 24, 25, 26, 27, 28, 38]
    right = [20, 21, 22, 23, 24, 25, 26, 29, 30, 38]
    assert len(left) == len(right) and left[0] == right[0] and left[-1] == right[-1]
    assert _runs(left) != _runs(right), "the exact selection distinguishes them"

    import torch

    def selector_for(chosen):
        def selector(seq_len: int):
            mask = torch.zeros(seq_len, dtype=torch.bool)
            mask[[c for c in chosen if c < seq_len]] = True
            return mask

        return selector

    a = selector_descriptor(selector_for(left), probe_lengths=[40])
    b = selector_descriptor(selector_for(right), probe_lengths=[40])
    assert a["sha256"] != b["sha256"], "the two selectors no longer collide"


def test_a_contiguous_band_stays_cheap_to_fingerprint():
    """The lossless encoding must not make the ordinary case expensive.

    A band is one run, so the common selector costs three integers rather than one per position.
    """
    from local_llm_lab.pipeline.lens_fitting.upstream import _runs

    assert _runs(list(range(16, 127))) == [[16, 127]]


def test_a_nested_selector_fit_is_refused_rather_than_sharing_one_module_global():
    """The patch is a module attribute with one call site, so nesting would cross two runs' rules.

    The inner fit would silently take the outer's selector and declare its own, which is the
    quietest possible way to produce a lens whose sidecar describes a different position rule from
    the one it was fitted under.
    """
    from local_llm_lab.pipeline.lens_fitting import upstream as adapter

    up = adapter.load_upstream()
    original = up.fitting.valid_position_mask
    try:
        def _stand_in(seq_len, *, skip_first=0):
            return original(seq_len, skip_first=skip_first)

        _stand_in._wsd_patched = True
        up.fitting.valid_position_mask = _stand_in
        with pytest.raises(adapter.CotangentSelectionError, match="already in progress"):
            adapter.fit_upstream_jacobian(
                TinyLensModel(),
                rows=make_rows(1),
                source_layers=(0, 1),
                max_seq_len=SEQ_LEN,
                position_selector=lambda seq_len: original(seq_len, skip_first=16),
            )
    finally:
        up.fitting.valid_position_mask = original


def test_only_upstreams_short_prompt_refusal_counts_as_a_skipped_row():
    """A ValueError that is not the short-prompt refusal is a defect and must not read as a skip.

    Upstream raises a bare `ValueError` from two places. Catching the class absorbed both, so a bad
    layer index would have been counted as a skipped row and the fit would have reported a smaller
    corpus rather than failing.
    """
    from local_llm_lab.pipeline.lens_fitting.upstream import _is_short_prompt

    assert _is_short_prompt(ValueError("prompt too short: seq_len=4, need > 17 tokens"))
    assert not _is_short_prompt(ValueError("source layer 99 out of range"))
    assert not _is_short_prompt(ValueError("skip_first must be >= 0, got -1"))


def test_the_upstream_seam_is_one_object_reached_by_both_import_paths():
    """WS-A and WS-D must not each hold their own way in.

    The seam lives at `local_llm_lab.upstream_ref` and `lens_fitting.upstream` re-exports it. If the
    two ever diverged, a box holding both a read-only clone and a pip-installed `jlens` would use
    whichever the interpreter imported first, and the provenance recorded in a sidecar would
    describe the other one.
    """
    from local_llm_lab import upstream_ref
    from local_llm_lab.pipeline.lens_fitting import upstream as adapter

    for name in ("load_upstream", "Upstream", "UpstreamUnavailable", "EXPECTED_JLENS_COMMIT"):
        assert getattr(adapter, name) is getattr(upstream_ref, name), f"{name} diverged"


def test_an_absent_clone_reads_as_a_skip_rather_than_a_collection_error():
    """`UpstreamUnavailable` is an `ImportError` so lazy importers can treat it as a skip.

    A module doing `from jlens import ...` at top level aborts collection of the whole file on a
    box without the clone, taking unrelated tests with it. Raising an `ImportError` subclass that
    carries our own message lets `pytest.importorskip` skip with a reason instead.
    """
    from local_llm_lab.upstream_ref import UpstreamUnavailable, load_upstream

    assert issubclass(UpstreamUnavailable, ImportError)
    assert issubclass(UpstreamUnavailable, RuntimeError), "the original contract is kept"
    with pytest.raises(UpstreamUnavailable, match="was not found"):
        load_upstream("/nonexistent/jacobian-lens")


def test_the_recorded_commit_falls_back_to_an_installed_pin_rather_than_none():
    """On a GPU box upstream is a pip install with no `.git`, and `None` would read as verified.

    The `[cuda]` extra installs it as `jlens @ git+...@581d398...`, which pip records in
    `direct_url.json`. Without the fallback the provenance field is `None` on precisely the machine
    the golden test runs on — a missing figure reading as a passing one.
    """
    from local_llm_lab.upstream_ref import _installed_commit

    # No `jlens` distribution is installed here, so the honest answer is None rather than a guess.
    assert _installed_commit() is None
