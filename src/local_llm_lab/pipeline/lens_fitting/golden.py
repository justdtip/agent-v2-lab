"""The golden comparison: two lenses of one model, per layer, with the controls that gate it.

**There is no pass threshold, and that is the ruling rather than an omission** (plan §16.13). Three
different things are being asked here and only two of them are gates:

1. *Exactness where it holds by construction.* The same estimator re-run on the same rows must
   reproduce itself **exactly**, at zero. That is a gate, and its tolerance is not a judgement call.
2. *The finding.* The finite-difference estimator against the exact one, per layer, is the number
   this whole stream exists to produce. It has no threshold because nobody knows what it is; a
   threshold chosen before the measurement would be a prediction wearing a gate's clothes.
3. *The controls.* A transposed artefact, a layer-shifted artefact and a fit on the wrong corpus
   must each disagree by **more** than the finding does. That is what makes the finding a
   measurement rather than a number: if a deliberately wrong lens were no further away than the
   candidate, the comparison could not tell them apart and the residual would mean nothing.

Two readings are fixed here before any number exists, because fixing them afterwards is how a floor
becomes a result.

**The storage floor.** The hosted lenses are stored ``float16``. A relative difference at or below
``2**-11`` (4.88e-4) per element is what two *identical* maps produce after a round trip through that
storage, so a residual there is **indistinguishable at storage precision** and is not agreement. The
report says which of the two it is for every layer, and never reports "agreement" for a number under
the floor.

**The estimator must be the only difference.** Two fits are comparable only if everything else in
their ν matches: endpoint, positions, pair weighting, corpus and precision. If any of those differ,
the residual is a sum of causes and attributing it to the estimator is a mistake nothing downstream
can catch. :func:`assert_estimator_is_the_only_difference` refuses that comparison rather than
annotating it.
"""

from __future__ import annotations

import numpy as np

from local_llm_lab.pipeline.live_lens.instruments import LensMaps

#: Relative floors by storage dtype: the mantissa step, which is what a round trip costs a map that
#: did not change at all. ``float16`` has 10 explicit mantissa bits, so 2**-11 is the half-step.
STORAGE_FLOORS = {"float16": 2.0**-11, "bfloat16": 2.0**-8, "float32": 2.0**-24, "float64": 2.0**-53}

#: What must agree for a residual to be attributable to the estimator alone: the ν blocks, and
#: within two of them the keys that describe **what was fitted** rather than **how**.
#:
#: The narrowing is not a relaxation, it is the fix for a gate that could never pass. Comparing
#: ``position_weighting`` and ``precision`` whole compares each estimator's own knobs — the exact
#: fit records ``dim_batch`` and a backward accumulation dtype there, the finite-difference fit
#: records ``direction_batch`` and has no backward pass at all — so two correct fits of one corpus
#: differ in those blocks **by construction**. A check that cannot pass is as useless as one that
#: cannot fail, and this repository has spent two days on the second kind; found by running the
#: real second operand through it, which is the only way it could have been found.
#:
#: What stays compared in those two blocks is everything that decides the fitted quantity: which
#: positions were selected (the selector fingerprint and the length rule), how they were reduced,
#: how many there were, and the precision the model actually ran at.
COMPARABILITY_FIELDS = ("endpoint", "position_weighting", "pair_weighting", "corpus", "precision")

#: Keys compared within a block; a block absent here is compared whole.
COMPARABLE_KEYS = {
    "position_weighting": (
        "sha256",
        "probe",
        "rule",
        "skip_first",
        "max_seq_len",
        "upstream_default_path",
        "source_and_target_tied",
        "source_reduction",
        "target_reduction",
        "n_valid_positions",
        "seq_len",
    ),
    "pair_weighting": ("source_target_pairs", "per_prompt", "n_prompts", "n_skipped"),
    "precision": (
        "capture_dtype",
        "device",
        "dtype",
        "dtypes_observed",
        "devices_observed",
        "blocks_measured",
        "requires_grad",
        "attn_implementation",
    ),
}


class NotComparable(ValueError):
    """Two fits differ in more than their estimator, so their residual has more than one cause."""


class ControlFailed(AssertionError):
    """A deliberately wrong lens was no further from the reference than the candidate."""


def storage_floor(dtypes: object) -> float:
    """The relative floor implied by the coarsest storage dtype present.

    Coarsest, not finest: a comparison is only as resolved as its blunter operand, and taking the
    finer one would report a floor the data cannot support.
    """
    names = [dtypes] if isinstance(dtypes, str) else list(dtypes)
    if not names:
        raise ValueError("no storage dtype given; the floor is not assumable")
    unknown = [name for name in names if name not in STORAGE_FLOORS]
    if unknown:
        raise ValueError(f"no declared storage floor for {unknown}; add it rather than guess")
    return max(STORAGE_FLOORS[name] for name in names)


def layer_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Cosine, relative difference and max absolute difference for one layer's two maps.

    ``relative_difference`` is ``||candidate - reference|| / ||reference||`` in Frobenius norm,
    which is the quantity the floor above is expressed in and the one the hosted comparison in
    ``GEMMA3-REGRESSION-2026-09-08`` reports.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} against {b.shape}")
    reference_norm = float(np.linalg.norm(a))
    if reference_norm == 0.0:
        raise ValueError("reference map has zero norm; a relative difference is undefined")
    difference = float(np.linalg.norm(b - a))
    denominator = reference_norm * float(np.linalg.norm(b))
    return {
        "cosine": float(np.sum(a * b) / denominator) if denominator else float("nan"),
        "relative_difference": difference / reference_norm,
        "max_abs": float(np.max(np.abs(b - a))),
        "reference_norm": reference_norm,
    }


def compare_lenses(reference: LensMaps | dict, candidate: LensMaps | dict) -> list[dict]:
    """Per-layer rows for two lenses of the same model, refusing a layer set that disagrees."""
    left = reference.maps if isinstance(reference, LensMaps) else reference
    right = candidate.maps if isinstance(candidate, LensMaps) else candidate
    if set(left) != set(right):
        raise ValueError(
            f"layer sets differ: {sorted(set(left) ^ set(right))} present in one lens only. "
            "A per-layer comparison over a partial intersection reports agreement it did not "
            "measure."
        )
    return [
        {"layer": layer, **layer_metrics(left[layer], right[layer])} for layer in sorted(left)
    ]


def assert_estimator_is_the_only_difference(reference_nu: dict, candidate_nu: dict) -> None:
    """Refuse a comparison whose residual would have more than one cause."""
    def _compared(nu: dict, field: str) -> object:
        block = nu.get(field)
        keys = COMPARABLE_KEYS.get(field)
        if keys is None or not isinstance(block, dict):
            return block
        return {key: block[key] for key in keys if key in block}

    differing = [
        field
        for field in COMPARABILITY_FIELDS
        if _compared(reference_nu, field) != _compared(candidate_nu, field)
    ]
    if differing:
        raise NotComparable(
            f"these two fits differ in {differing} as well as in their estimator, so the residual "
            "between them is a sum of causes. The golden comparison isolates the estimator; a "
            "corpus or endpoint difference is a different measurement and must be declared as one."
        )
    if reference_nu.get("estimator") == candidate_nu.get("estimator"):
        raise NotComparable(
            f"both fits declare estimator {reference_nu.get('estimator')!r}. Comparing an "
            "estimator with itself belongs in the exactness gate, where the expected residual is "
            "zero, not in the finding, where no threshold exists."
        )


def assert_exact_reproduction(reference: LensMaps | dict, candidate: LensMaps | dict) -> list[dict]:
    """Gate 1: the same estimator on the same rows reproduces itself at **zero**.

    Exactly zero, not "within tolerance". Any nonzero residual here is nondeterminism in the
    estimator or in the box, and reporting it under a tolerance would hide the one thing this gate
    is for. The rows are returned so a caller can put them in the record either way.
    """
    rows = compare_lenses(reference, candidate)
    nonzero = [row for row in rows if row["relative_difference"] != 0.0]
    if nonzero:
        worst = max(nonzero, key=lambda row: row["relative_difference"])
        raise ControlFailed(
            f"the exact estimator did not reproduce itself: layer {worst['layer']} differs by "
            f"{worst['relative_difference']:.3e} relative over {len(nonzero)} layers. This gate is "
            "zero by construction, so a nonzero value is nondeterminism, not a tolerance question."
        )
    return rows


def transposed_control(maps: dict) -> dict:
    """A control that is wrong only in orientation. Square maps make this invisible to shape checks."""
    return {layer: np.asarray(matrix).T.copy() for layer, matrix in maps.items()}


def layer_shifted_control(maps: dict) -> dict:
    """A control that is wrong only in which layer each map belongs to.

    The commonest off-by-one in this repository's history: upstream indexes blocks from zero and
    this repository names layer L the output of block L-1, so a shift of one is the mistake a
    conversion makes, and every shape, dtype and identity check passes it.
    """
    layers = sorted(maps)
    if len(layers) < 2:
        raise ValueError("a layer shift needs at least two layers to be a different lens")
    return {layer: np.asarray(maps[layers[(i + 1) % len(layers)]]).copy()
            for i, layer in enumerate(layers)}


def golden_report(
    *,
    reference: LensMaps | dict,
    candidate: LensMaps | dict,
    reference_nu: dict,
    candidate_nu: dict,
    storage_dtypes: object,
    finite_difference_epsilon: object,
    reproduction: LensMaps | dict | None = None,
    controls: dict[str, LensMaps | dict] | None = None,
    fixture_residual: float | None = None,
) -> dict:
    """The whole comparison as one dict: the gates, the finding, and what the finding rests on.

    Args:
        reference: the exact-autograd lens, which is the reference by construction.
        candidate: the finite-difference lens. The finding is this against the reference.
        reference_nu, candidate_nu: refused unless the estimator is their only difference.
        storage_dtypes: what the operands are *stored* at, measured not assumed. The floor comes
            from the coarsest.
        finite_difference_epsilon: declared beside the residual because the residual is a function
            of it; a residual without its epsilon cannot be reproduced or compared.
        reproduction: an independent re-run of the reference estimator. Gate 1 when present, and
            its absence is recorded rather than passed over.
        controls: named deliberately wrong lenses, each of which must disagree by more than the
            finding does.
        fixture_residual: the residual this comparison produced on synthetic operands, carried as
            the expected order of magnitude rather than as a threshold.
    """
    assert_estimator_is_the_only_difference(reference_nu, candidate_nu)
    floor = storage_floor(storage_dtypes)
    rows = compare_lenses(reference, candidate)
    for row in rows:
        row["at_or_below_storage_floor"] = row["relative_difference"] <= floor
    finding = max(row["relative_difference"] for row in rows)

    control_rows = {}
    failures = []
    for name, wrong in (controls or {}).items():
        worst = max(row["relative_difference"] for row in compare_lenses(reference, wrong))
        control_rows[name] = {
            "worst_relative_difference": worst,
            "margin_over_finding": (worst / finding) if finding else float("inf"),
            "separates": worst > finding,
        }
        if not control_rows[name]["separates"]:
            failures.append(name)
    if failures:
        raise ControlFailed(
            f"controls {failures} are no further from the reference than the candidate is "
            f"({finding:.3e}). A comparison that cannot separate a deliberately wrong lens from "
            "the candidate has not measured the candidate."
        )

    return {
        "schema_version": 1,
        "gates": {
            "exact_reproduces_itself": {
                "expected": 0.0,
                "measured": (
                    None
                    if reproduction is None
                    else max(
                        row["relative_difference"]
                        for row in assert_exact_reproduction(reference, reproduction)
                    )
                ),
                # A missing figure must never read as a passing one.
                "state": "not run" if reproduction is None else "passed",
            },
            "controls_separate": control_rows,
        },
        "finding": {
            "per_layer": rows,
            "worst_relative_difference": finding,
            "threshold": None,
            "threshold_note": (
                "no pass threshold exists for the finding (plan §16.13). The estimator difference "
                "is the measurement; a threshold set before it would be a prediction."
            ),
            "at_or_below_storage_floor": finding <= floor,
            "reading_if_at_floor": (
                "indistinguishable at storage precision, which is not agreement"
            ),
        },
        "declared": {
            "storage_dtypes": [storage_dtypes] if isinstance(storage_dtypes, str)
            else list(storage_dtypes),
            "storage_floor_relative": floor,
            "finite_difference_epsilon": finite_difference_epsilon,
            "fixture_residual": fixture_residual,
            "reference_estimator": reference_nu.get("estimator"),
            "candidate_estimator": candidate_nu.get("estimator"),
        },
    }


__all__ = [
    "COMPARABILITY_FIELDS",
    "COMPARABLE_KEYS",
    "STORAGE_FLOORS",
    "ControlFailed",
    "NotComparable",
    "assert_estimator_is_the_only_difference",
    "assert_exact_reproduction",
    "compare_lenses",
    "golden_report",
    "layer_metrics",
    "layer_shifted_control",
    "storage_floor",
    "transposed_control",
]
