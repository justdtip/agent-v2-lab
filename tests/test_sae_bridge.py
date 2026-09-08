"""Model-free arithmetic contracts for the ordered A1 dictionary readout."""

import numpy as np
import pytest

from local_llm_lab.pipeline.sae_bridge import (
    compare_score_vectors,
    compose_decoder,
    feature_scores,
    iter_feature_topk,
    reference_feature_scores,
)


@pytest.fixture
def matrices():
    # J is deliberately nonsymmetric; columns of D are distinct features.
    j = np.array([[1, 2], [3, -1]], dtype=np.float32)
    d = np.array([[2, -1, 0], [1, 2, -3]], dtype=np.float32)
    w = np.array([[1, 0], [0, 1], [1, 1], [-10, 0], [1, 1]], dtype=np.float32)
    return w, j, d


def test_analytic_orientation_and_complete_scores(matrices):
    w, j, d = matrices
    jd = compose_decoder(j, d)
    np.testing.assert_array_equal(jd, [[4, 3, -6], [5, -5, 3]])
    assert jd.dtype == np.float32
    np.testing.assert_array_equal(feature_scores(w, jd, 0), [4, 5, 9, -40, 9])
    np.testing.assert_array_equal(feature_scores(w, jd, 1), [3, -5, -2, -30, -2])
    reference = reference_feature_scores(w, j, d, 0, row_chunk_size=2)
    assert reference.dtype == np.float64
    np.testing.assert_array_equal(reference, [4, 5, 9, -40, 9])


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_signed_topk_ties_and_every_result_against_exhaustive_reference(matrices, k):
    w, j, d = matrices
    rows = list(iter_feature_topk(w, compose_decoder(j, d), k))
    assert [r.feature_id for r in rows] == [0, 1, 2]
    for row in rows:
        # Scalar sums form an independent exhaustive reference, including both axes.
        expected = [
            sum(
                float(w[v, a]) * float(j[a, b]) * float(d[b, row.feature_id])
                for a in range(2)
                for b in range(2)
            )
            for v in range(5)
        ]
        ids = sorted(range(5), key=lambda v: (-expected[v], v))[:k]
        np.testing.assert_array_equal(row.token_ids, ids)
        np.testing.assert_array_equal(row.scores, [expected[v] for v in ids])


def test_selection_preserves_requested_order(matrices):
    w, j, d = matrices
    rows = list(iter_feature_topk(w, compose_decoder(j, d), 1, feature_ids=[2, 0]))
    assert [r.feature_id for r in rows] == [2, 0]


@pytest.mark.parametrize("ids", [[], [0, 0], [-1], [3], [1.0], [True]])
def test_invalid_selection_rejected(matrices, ids):
    w, j, d = matrices
    with pytest.raises(ValueError):
        list(iter_feature_topk(w, compose_decoder(j, d), 1, feature_ids=ids))


@pytest.mark.parametrize("k", [0, -1, 6, 1.5, True])
def test_invalid_k_rejected(matrices, k):
    w, j, d = matrices
    with pytest.raises(ValueError):
        list(iter_feature_topk(w, compose_decoder(j, d), k))


@pytest.mark.parametrize(
    "bad", [np.zeros((0, 0)), np.zeros(2), [[np.nan]], [[np.inf]], [[1e100]], [[1j]], [["1"]]]
)
def test_bad_matrix_rejected(bad):
    with pytest.raises(ValueError):
        compose_decoder(bad, np.ones((1, 1)))


def test_shapes_and_generated_nonfinite_rejected(matrices):
    w, j, d = matrices
    for bad_j, bad_d in [(np.ones((2, 3)), d), (j, d.T), (j, np.empty((2, 0)))]:
        with pytest.raises(ValueError):
            compose_decoder(bad_j, bad_d)
    with pytest.raises(ValueError):
        compose_decoder(np.full((2, 2), 3e38), np.full((2, 1), 3e38))
    with pytest.raises(ValueError):
        feature_scores(w.T, compose_decoder(j, d), 0)
    with pytest.raises(ValueError):
        feature_scores(w, compose_decoder(j, d), -1)
    with pytest.raises(ValueError):
        reference_feature_scores(w, j, d, 0, row_chunk_size=0)


def test_reference_chunk_boundaries_and_float32_agreement():
    rng = np.random.default_rng(12)
    w = rng.normal(size=(13, 7)).astype(np.float32)
    j = rng.normal(size=(7, 7)).astype(np.float32)
    d = rng.normal(size=(7, 4)).astype(np.float32)
    actual = feature_scores(w, compose_decoder(j, d), 2)
    for chunk in [1, 4, 13, 20]:
        reference = reference_feature_scores(w, j, d, 2, row_chunk_size=chunk)
        result = compare_score_vectors(actual, reference)
        assert result.passed
        assert result.reference_l2 > 0
        assert result.absolute_l2 >= 0
        assert result.relative_l2 < 1e-5
        assert result.max_absolute_error >= 0


def test_comparison_reports_norm_and_detects_non_topk_error():
    result = compare_score_vectors([10, 2, -3], [10, 2, -4], rtol=0, atol=0)
    assert not result.passed
    assert result.absolute_l2 == 1
    assert result.reference_l2 == pytest.approx(np.sqrt(120))
    assert result.relative_l2 == pytest.approx(1 / np.sqrt(120))
    assert result.max_absolute_error == 1
    zero = compare_score_vectors([0, 0], [0, 0])
    assert zero.passed and zero.relative_l2 == 0
    assert np.isinf(compare_score_vectors([1, 0], [0, 0]).relative_l2)


@pytest.mark.parametrize(
    "actual,reference", [([], []), ([1], [1, 2]), ([np.nan], [0]), ([[1]], [[1]])]
)
def test_comparison_rejects_inapplicable_vectors(actual, reference):
    with pytest.raises(ValueError):
        compare_score_vectors(actual, reference)


def test_all_equal_partition_boundary_prefers_smallest_ids():
    rows = list(iter_feature_topk(np.zeros((8, 2)), np.ones((2, 1)), 3))
    np.testing.assert_array_equal(rows[0].token_ids, [0, 1, 2])
    np.testing.assert_array_equal(rows[0].scores, [0, 0, 0])


@pytest.mark.parametrize("feature_id", [-1, 3, 0.5, True])
def test_named_feature_access_rejects_invalid_id(matrices, feature_id):
    w, j, d = matrices
    with pytest.raises(ValueError):
        feature_scores(w, compose_decoder(j, d), feature_id)
    with pytest.raises(ValueError):
        reference_feature_scores(w, j, d, feature_id)


@pytest.mark.parametrize(
    "bad",
    [np.empty((0, 2)), np.full((1, 2), np.nan), np.full((1, 2), np.inf), np.full((1, 2), 1e100)],
)
def test_invalid_readout_rejected_in_both_paths(matrices, bad):
    _, j, d = matrices
    with pytest.raises(ValueError):
        feature_scores(bad, compose_decoder(j, d), 0)
    with pytest.raises(ValueError):
        reference_feature_scores(bad, j, d, 0)


def test_score_overflow_rejected_before_ranking():
    with pytest.raises(ValueError, match="not finite"):
        list(iter_feature_topk(np.full((2, 2), 3e38), np.full((2, 1), 3e38), 1))


@pytest.mark.parametrize("tolerance", [-1, np.inf, np.nan])
def test_invalid_comparison_tolerances_rejected(tolerance):
    with pytest.raises(ValueError):
        compare_score_vectors([1], [1], rtol=tolerance)
    with pytest.raises(ValueError):
        compare_score_vectors([1], [1], atol=tolerance)


def test_streaming_validates_once_and_multiplies_one_column_per_next(monkeypatch, matrices):
    from local_llm_lab.pipeline.sae_bridge import core

    w, j, d = matrices
    jd = compose_decoder(j, d)
    validations = []
    multiplied_shapes = []
    validate = core._matrix

    class ObservedReadout(np.ndarray):
        def __matmul__(self, other):
            multiplied_shapes.append(other.shape)
            assert other.ndim == 1, "must not form the vocabulary-by-feature transfer"
            return super().__matmul__(other)

    def observe_matrix(value, name):
        validations.append(name)
        result = validate(value, name)
        return result.view(ObservedReadout) if name == "W" else result

    monkeypatch.setattr(core, "_matrix", observe_matrix)
    stream = iter_feature_topk(w, jd, 2)
    assert validations == [] and multiplied_shapes == []
    assert next(stream).feature_id == 0
    assert multiplied_shapes == [(2,)]
    assert [row.feature_id for row in stream] == [1, 2]
    assert multiplied_shapes == [(2,), (2,), (2,)]
    assert validations == ["W", "JD"]


@pytest.fixture
def shipped_readout():
    return np.array([[2, 4], [3, 1], [0, 3]]), np.array([[3, 3], [10, 2], [0, 0]])


def test_identity_baseline_has_no_j_product_and_reports_predeclared_coverage(
    matrices,
    shipped_readout,
    monkeypatch,
):
    from local_llm_lab.pipeline.sae_bridge import readout

    w, _, d = matrices
    tokens, logits = shipped_readout
    monkeypatch.setattr(readout, "compose_decoder", lambda *a: pytest.fail("J composition reached"))
    result = readout.identity_baseline(w, d, tokens, logits, feature_ids=[1, 0], k=2)
    assert result["passed"] and result["status"] == "passed"
    assert result["mean_overlap"] == result["minimum_overlap"] == 1
    assert result["coverage"] == dict(
        compared_features=2,
        dictionary_features=3,
        feature_ids=[1, 0],
        all_features=False,
        vocabulary=5,
        shipped_topk_width=2,
    )
    assert [r["feature_id"] for r in result["features"]] == [1, 0]
    assert result["features"][0]["direct_tokens"] == [3, 1]
    assert result["features"][0]["direct_minus_shipped_scores"] == [0, 0]
    assert result["convention"] == "raw tied W; no final normalization"


def test_identity_membership_gate_retains_mismatch_evidence(matrices, shipped_readout):
    from local_llm_lab.pipeline.sae_bridge.readout import identity_baseline

    w, _, d = matrices
    tokens, logits = shipped_readout
    tokens[0] = [2, 0]
    logits[0] = [3, 2]
    result = identity_baseline(w, d, tokens, logits, feature_ids=[0, 1], k=2)
    assert not result["passed"] and result["status"] == "ruling_required"
    assert result["minimum_overlap"] == 0.5 and result["mean_overlap"] == 0.75
    assert result["features"][0]["direct_tokens"] == [2, 4]
    assert result["features"][0]["shipped_tokens"] == [2, 0]
    assert result["features"][0]["overlap_count"] == 1
    assert result["features"][1]["overlap_count"] == 2
    assert "unresolved" in result["interpretation"]


def test_identity_order_and_numeric_differences_are_descriptive(matrices, shipped_readout):
    from local_llm_lab.pipeline.sae_bridge.readout import identity_baseline

    w, _, d = matrices
    tokens, logits = shipped_readout
    tokens[0] = [4, 2]
    logits = logits.astype(float)
    logits[0] = [3.25, 2.5]
    result = identity_baseline(w, d, tokens, logits, feature_ids=[0], k=2)
    assert result["passed"]
    row = result["features"][0]
    assert row["set_overlap"] == 1 and not row["ordered_equal"]
    assert row["direct_scores_at_shipped_tokens"] == [3, 3]
    assert row["direct_minus_shipped_scores"] == [-0.25, 0.5]
    assert row["max_absolute_score_difference"] == 0.5


@pytest.mark.parametrize(
    "bad",
    [
        "shape",
        "float_tokens",
        "duplicate",
        "negative",
        "bounds",
        "nonfinite",
        "logit_shape",
        "feature_duplicate",
        "empty",
    ],
)
def test_identity_baseline_refuses_invalid_inputs(matrices, shipped_readout, bad):
    from local_llm_lab.pipeline.sae_bridge.readout import identity_baseline

    w, _, d = matrices
    tokens, logits = shipped_readout
    feature_ids = [0, 1]
    if bad == "shape":
        tokens = tokens[:2]
    elif bad == "float_tokens":
        tokens = tokens.astype(float)
    elif bad == "duplicate":
        tokens[0] = [2, 2]
    elif bad == "negative":
        tokens[0, 0] = -1
    elif bad == "bounds":
        tokens[0, 0] = 5
    elif bad == "nonfinite":
        logits = logits.astype(float)
        logits[0, 0] = np.nan
    elif bad == "logit_shape":
        logits = logits[:, :1]
    elif bad == "feature_duplicate":
        feature_ids = [0, 0]
    else:
        feature_ids = []
    with pytest.raises(ValueError):
        identity_baseline(w, d, tokens, logits, feature_ids=feature_ids, k=2)


def test_readout_adds_direct_and_shipped_comparisons_and_exact_unlabelled_reason(
    tmp_path,
    matrices,
    shipped_readout,
):
    import json

    from local_llm_lab.pipeline.sae_bridge.readout import write_readout

    w, j, d = matrices
    tokens, logits = shipped_readout
    path = tmp_path / "readout.jsonl"
    write_readout(
        path,
        w,
        compose_decoder(j, d),
        d=d,
        shipped_tokens=tokens,
        shipped_logits=logits,
        k=2,
        metadata={"probe_layer": 7},
        token_piece=str,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(row for row in rows if row["type"] == "feature" and row["feature_id"] == 1)
    assert row["feature_id"] == 1
    assert [t["id"] for t in row["tokens"]] == [0, 2]
    assert [t["id"] for t in row["direct_tokens"]] == [3, 1]
    assert row["direct_vs_j"]["set_overlap"] == 0
    assert row["shipped_vs_direct"]["set_overlap"] == 1
    assert row["label"] == "unlabelled"
    assert row["label_reason"] == (
        "no Neuronpedia source maps to `resid_post_all`; the residual labels index "
        "`resid_post/layer_17_width_16k_l0_medium`, a different training run at a sparsity "
        "this suite never published."
    )
    assert rows[0]["metadata"]["probe_layer"] == 7
    assert rows[-1]["type"] == "complete" and rows[-1]["features"] == 3
    assert rows[-1]["status"] == "passed"
    assert rows[-1]["direct_baseline"]["compared_features"] == 3
    assert [row["type"] for row in rows[1:4]] == ["direct_baseline_feature"] * 3


def test_full_direct_scan_failure_records_all_features_without_any_j_readout(
    tmp_path,
    matrices,
    shipped_readout,
    monkeypatch,
):
    import json

    from local_llm_lab.pipeline.sae_bridge import readout

    w, j, d = matrices
    jd = compose_decoder(j, d)
    tokens, logits = shipped_readout
    tokens[2] = [0, 1]  # The earlier benchmark features can pass while a later feature fails.
    iterate = readout.iter_feature_topk

    def forbid_j(w_arg, decoder, *args, **kwargs):
        if decoder is jd:
            pytest.fail("J readout began despite full direct-baseline failure")
        return iterate(w_arg, decoder, *args, **kwargs)

    monkeypatch.setattr(readout, "iter_feature_topk", forbid_j)
    path = tmp_path / "failed.jsonl"
    result = readout.write_readout(
        path,
        w,
        jd,
        d=d,
        shipped_tokens=tokens,
        shipped_logits=logits,
        k=2,
        metadata={},
        token_piece=str,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert result["status"] == "ruling_required"
    assert result["features"] == 0
    assert result["direct_baseline"]["compared_features"] == 3
    assert result["direct_baseline"]["minimum_overlap"] == 0.5
    assert len([r for r in rows if r["type"] == "direct_baseline_feature"]) == 3
    assert not any(r["type"] == "feature" for r in rows)
    assert rows[-1]["type"] == "complete" and rows[-1]["status"] == "ruling_required"


@pytest.mark.parametrize("missing", ["d", "shipped_tokens", "shipped_logits"])
def test_paired_readout_requires_all_direct_inputs_before_output(
    tmp_path,
    matrices,
    shipped_readout,
    missing,
):
    from local_llm_lab.pipeline.sae_bridge.readout import write_readout

    w, j, d = matrices
    tokens, logits = shipped_readout
    args = dict(d=d, shipped_tokens=tokens, shipped_logits=logits)
    del args[missing]
    path = tmp_path / "invalid.jsonl"
    with pytest.raises(ValueError, match="together"):
        write_readout(path, w, compose_decoder(j, d), k=2, metadata={}, token_piece=str, **args)
    assert not path.exists()
