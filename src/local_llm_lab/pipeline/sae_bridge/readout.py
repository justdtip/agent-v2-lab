"""Independent A1 acceptance and a bounded, hash-chained feature readout."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .core import (
    FeatureTopK,
    _integer,
    _readout,
    compare_score_vectors,
    compose_decoder,
    feature_scores,
    iter_feature_topk,
    reference_feature_scores,
)

INTERPRETATION = (
    "Raw linear W J d scores, not token probabilities or causal derivatives. "
    "No final normalization, bias or softmax. Small averaged-Jacobian readings can "
    "hide cancellation of contextual sensitivity. Repeated feature activity with "
    "a retained transcript does not establish persistent memory."
)


UNLABELLED_REASON = (
    "no Neuronpedia source maps to `resid_post_all`; the residual labels index "
    "`resid_post/layer_17_width_16k_l0_medium`, a different training run at a sparsity "
    "this suite never published."
)
DIRECT_CONVENTION = "raw tied W; no final normalization"
DIRECT_INTERPRETATION = (
    "Our direct raw W d_i score comparison. The shipped top_logits generation convention "
    "is unresolved; disagreement requires a ruling or investigation and alone does not "
    "establish a decoder orientation, checkpoint, or tokenizer error. Score differences "
    "are descriptive, with no fitted or post-hoc tolerance."
)


def _direct_inputs(w, d, shipped_tokens, shipped_logits, k):
    w, d = _readout(w, d)
    tokens, logits = np.asarray(shipped_tokens), np.asarray(shipped_logits)
    if (
        tokens.ndim != 2
        or tokens.shape[0] != d.shape[1]
        or not tokens.shape[1]
        or tokens.dtype.kind not in "iu"
        or logits.shape != tokens.shape
        or logits.dtype.kind not in "fiu"
    ):
        raise ValueError("shipped tokens/logits must be matching nonempty feature-by-topk arrays")
    if np.any(tokens < 0) or np.any(tokens >= w.shape[0]):
        raise ValueError("shipped token ID is outside the vocabulary")
    if np.any(np.diff(np.sort(tokens, axis=1), axis=1) == 0):
        raise ValueError("shipped top tokens contain duplicates within a feature")
    with np.errstate(over="ignore", invalid="ignore"):
        logits = logits.astype(np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("shipped logits must be finite")
    k = _integer(k, "k", 1, min(tokens.shape[1], w.shape[0]))
    return w, d, tokens.astype(np.int64), logits, k


def _overlap(left, right):
    count = len(set(left) & set(right))
    return {
        "overlap_count": count,
        "set_overlap": count / len(left),
        "ordered_equal": list(left) == list(right),
    }


def _direct_evidence(w, d, row, shipped_tokens, shipped_logits, k):
    tokens = shipped_tokens[row.feature_id, :k]
    logits = shipped_logits[row.feature_id, :k]
    with np.errstate(over="ignore", invalid="ignore"):
        calculated = w[tokens] @ d[:, row.feature_id]
    if not np.isfinite(calculated).all():
        raise ValueError("direct scores at shipped tokens are not finite")
    differences = calculated.astype(np.float64) - logits
    if not np.isfinite(differences).all():
        raise ValueError("direct-versus-shipped numeric differences are not finite")
    return {
        "feature_id": row.feature_id,
        "label": "unlabelled",
        "label_reason": UNLABELLED_REASON,
        "k": k,
        "direct_tokens": row.token_ids.tolist(),
        "direct_scores": row.scores.tolist(),
        "shipped_tokens": tokens.tolist(),
        "shipped_scores": logits.tolist(),
        "direct_scores_at_shipped_tokens": calculated.tolist(),
        "direct_minus_shipped_scores": differences.tolist(),
        "max_absolute_score_difference": float(np.max(np.abs(differences))),
        **_overlap(row.token_ids.tolist(), tokens.tolist()),
    }


def identity_baseline(w, d, shipped_tokens, shipped_logits, *, feature_ids, k):
    """Compare raw W d_i before any J product, retaining every declared feature's evidence.

    The fail-closed gate requires identical top-k membership for every selected
    feature. Ordering and aligned score differences are reported without a numeric
    score tolerance; a mismatch returns ruling_required rather than discarding evidence.
    """
    if feature_ids is None:
        raise ValueError("identity baseline requires predeclared feature IDs")
    w, d, shipped_tokens, shipped_logits, k = _direct_inputs(
        w, d, shipped_tokens, shipped_logits, k
    )
    records = [
        _direct_evidence(w, d, row, shipped_tokens, shipped_logits, k)
        for row in iter_feature_topk(w, d, k, feature_ids=feature_ids)
    ]
    passed = all(row["overlap_count"] == k for row in records)
    return {
        "status": "passed" if passed else "ruling_required",
        "passed": passed,
        "k": k,
        "gate": "all predeclared feature direct top-k memberships equal shipped top-k memberships",
        "convention": DIRECT_CONVENTION,
        "interpretation": DIRECT_INTERPRETATION,
        "mean_overlap": sum(row["set_overlap"] for row in records) / len(records),
        "minimum_overlap": min(row["set_overlap"] for row in records),
        "coverage": {
            "compared_features": len(records),
            "dictionary_features": d.shape[1],
            "feature_ids": [row["feature_id"] for row in records],
            "all_features": len(records) == d.shape[1],
            "vocabulary": w.shape[0],
            "shipped_topk_width": shipped_tokens.shape[1],
        },
        "features": records,
    }


def acceptance(w, j, wrong_j, d, *, feature_id, k, rtol, atol, relative_l2_limit):
    """The mismatched layer must fail the very gate passed by the matched layer."""
    if not math.isfinite(relative_l2_limit) or relative_l2_limit <= 0:
        raise ValueError("relative L2 bound must be positive and finite")
    jd = compose_decoder(j, d)
    reference = reference_feature_scores(w, j, d, feature_id)
    actual = feature_scores(w, jd, feature_id)
    wrong_jd = compose_decoder(wrong_j, d)
    wrong = feature_scores(w, wrong_jd, feature_id)

    def compare(scores):
        result = asdict(compare_score_vectors(scores, reference, rtol=rtol, atol=atol))
        result["elementwise_passed"] = result["passed"]
        result["relative_l2_limit"] = relative_l2_limit
        result["passed"] &= result["relative_l2"] <= relative_l2_limit
        # JSON has no infinity. A nonzero error against a zero vector is a failure,
        # not a missing denominator quietly turned into a passing relative error.
        result["zero_reference_norm"] = result["reference_l2"] == 0
        if not math.isfinite(result["relative_l2"]):
            result["relative_l2"] = None
        return result

    positive, negative = compare(actual), compare(wrong)
    good_top = next(iter_feature_topk(w, jd, k, feature_ids=[feature_id]))
    bad_top = next(iter_feature_topk(w, wrong_jd, k, feature_ids=[feature_id]))
    distance = len(set(good_top.token_ids.tolist()) ^ set(bad_top.token_ids.tolist()))
    return {
        "feature_id": feature_id,
        "k": k,
        "positive": positive,
        "wrong_layer": negative,
        "matched_topk": good_top.token_ids.tolist(),
        "wrong_layer_topk": bad_top.token_ids.tolist(),
        "topk_symmetric_difference": distance,
        "passed": positive["passed"] and not negative["passed"] and distance > 0,
        "reference": "float64 (W_row_chunk @ J) @ d_i from float32 operands",
        "interpretation": INTERPRETATION,
    }, jd


def write_readout(
    path: Path,
    w,
    jd,
    *,
    k: int,
    metadata: dict,
    token_piece,
    progress=None,
    d=None,
    shipped_tokens=None,
    shipped_logits=None,
):
    """Exclusive JSONL; absence of a complete footer means an incomplete artifact.

    All direct columns are checked before any J score column. A mismatch completes
    the direct evidence with ruling_required and zero J readout features. Returns
    the footer summary; a complete artifact is not necessarily a passed baseline.
    Labels carry the amendment's exact unlabelled reason.
    """
    direct = any(value is not None for value in (d, shipped_tokens, shipped_logits))
    if direct:
        if any(value is None for value in (d, shipped_tokens, shipped_logits)):
            raise ValueError(
                "direct readout requires D, shipped tokens, and shipped logits together"
            )
        w, d, shipped_tokens, shipped_logits, k = _direct_inputs(
            w, d, shipped_tokens, shipped_logits, k
        )
        if np.shape(jd) != d.shape:
            raise ValueError("direct D and JD feature/residual axes must match")
    previous = "0" * 64
    with Path(path).open("x", encoding="utf-8") as stream:

        def emit(row):
            nonlocal previous
            row = dict(row, previous_sha256=previous)
            raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            previous = hashlib.sha256(raw.encode()).hexdigest()
            stream.write(json.dumps(dict(row, sha256=previous), allow_nan=False) + "\n")
            stream.flush()

        emit(
            {
                "type": "header",
                "schema_version": 2 if direct else 1,
                "readout_heading": "our raw W J d_i top-token readout",
                "direct_readout_heading": "our raw W d_i top-token readout" if direct else None,
                "direct_convention": DIRECT_CONVENTION if direct else None,
                "direct_interpretation": DIRECT_INTERPRETATION if direct else None,
                "metadata": metadata,
                "interpretation": INTERPRETATION,
                "k": k,
                "ordering": "largest signed score; lower token ID breaks ties",
                "features": jd.shape[1],
                "vocabulary": w.shape[0],
            }
        )
        direct_summary = {}
        if direct:
            # Only feature-by-k summaries are retained; full vocabulary columns are
            # released by the iterator. Finish this gate before any WJD column.
            direct_ids = np.empty((d.shape[1], k), dtype=np.int64)
            direct_scores = np.empty((d.shape[1], k), dtype=np.float32)
            overlap_sum, overlap_min = 0.0, 1.0
            for direct_row in iter_feature_topk(w, d, k):
                evidence = _direct_evidence(w, d, direct_row, shipped_tokens, shipped_logits, k)
                emit({"type": "direct_baseline_feature", **evidence})
                direct_ids[direct_row.feature_id] = direct_row.token_ids
                direct_scores[direct_row.feature_id] = direct_row.scores
                overlap_sum += evidence["set_overlap"]
                overlap_min = min(overlap_min, evidence["set_overlap"])
                completed = direct_row.feature_id + 1
                if progress and (completed % 128 == 0 or completed == d.shape[1]):
                    progress(
                        {"event": "direct_feature", "completed": completed, "total": d.shape[1]}
                    )
            direct_summary = {
                "passed": overlap_min == 1.0,
                "status": "passed" if overlap_min == 1.0 else "ruling_required",
                "compared_features": d.shape[1],
                "dictionary_features": d.shape[1],
                "mean_overlap": overlap_sum / d.shape[1],
                "minimum_overlap": overlap_min,
            }
            if not direct_summary["passed"]:
                summary = {
                    "type": "complete",
                    "features": 0,
                    "status": "ruling_required",
                    "direct_baseline": direct_summary,
                }
                emit(summary)
                return summary
        count, j_overlap_sum = 0, 0.0
        for row in iter_feature_topk(w, jd, k):
            direct_row = (
                None
                if not direct
                else FeatureTopK(
                    row.feature_id, direct_ids[row.feature_id], direct_scores[row.feature_id]
                )
            )
            comparison = {}
            if direct:
                evidence = _direct_evidence(w, d, direct_row, shipped_tokens, shipped_logits, k)
                j_overlap = _overlap(direct_row.token_ids.tolist(), row.token_ids.tolist())
                j_overlap_sum += j_overlap["set_overlap"]
                comparison = {
                    "direct_tokens": [
                        {"id": int(token), "piece": token_piece(int(token)), "score": float(score)}
                        for token, score in zip(
                            direct_row.token_ids, direct_row.scores, strict=True
                        )
                    ],
                    "direct_vs_j": j_overlap,
                    "shipped_vs_direct": evidence,
                }
            emit(
                {
                    "type": "feature",
                    "feature_id": row.feature_id,
                    "label": "unlabelled",
                    "label_status": "unlabelled",
                    "label_reason": UNLABELLED_REASON,
                    **comparison,
                    "tokens": [
                        {"id": int(token), "piece": token_piece(int(token)), "score": float(score)}
                        for token, score in zip(row.token_ids, row.scores, strict=True)
                    ],
                }
            )
            count += 1
            if progress and (count % 128 == 0 or count == jd.shape[1]):
                progress({"event": "feature", "completed": count, "total": jd.shape[1]})
        summary = {"type": "complete", "features": count, "status": "passed"}
        if direct:
            summary.update(
                direct_baseline=direct_summary,
                direct_vs_j_mean_overlap=j_overlap_sum / count,
            )
        emit(summary)
        return summary
