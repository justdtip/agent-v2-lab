"""Independent A1 acceptance and a bounded, hash-chained feature readout."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from .core import (
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


def write_readout(path: Path, w, jd, *, k: int, metadata: dict, token_piece, progress=None):
    """Exclusive JSONL; absence of a complete footer means an incomplete artifact.

    No published label mapping is currently verified for the ordered all-layer
    dictionary. Labels are explicitly absent rather than borrowed from a variant.
    """
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
                "schema_version": 1,
                "metadata": metadata,
                "interpretation": INTERPRETATION,
                "k": k,
                "ordering": "largest signed score; lower token ID breaks ties",
                "features": jd.shape[1],
                "vocabulary": w.shape[0],
            }
        )
        count = 0
        for row in iter_feature_topk(w, jd, k):
            emit(
                {
                    "type": "feature",
                    "feature_id": row.feature_id,
                    "label": None,
                    "label_status": "unavailable_no_verified_dictionary_mapping",
                    "tokens": [
                        {"id": int(token), "piece": token_piece(int(token)), "score": float(score)}
                        for token, score in zip(row.token_ids, row.scores, strict=True)
                    ],
                }
            )
            count += 1
            if progress and (count % 128 == 0 or count == jd.shape[1]):
                progress({"event": "feature", "completed": count, "total": jd.shape[1]})
        emit({"type": "complete", "features": count})
