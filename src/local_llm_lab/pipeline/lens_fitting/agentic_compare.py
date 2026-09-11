"""Paired readout accounting and explicit local-derivative diagnostics.

The scalar readout rows must come from the model's full unembed, including final
normalization and softcap. Gain-only bridge scores are not probabilities. This
module never infers a full-vocabulary winner from the six tool probabilities.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _distribution(values):
    x = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(x):
        return {"n": 0}
    return {
        "n": len(x),
        "minimum": float(x.min()),
        "p05": float(np.quantile(x, 0.05)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "maximum": float(x.max()),
        "share_at_zero": float(np.mean(x == 0)),
        "share_at_one": float(np.mean(x == 1)),
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _readout(value, tokens):
    six = np.asarray(value["six"], dtype=np.float64)
    mass = value["mass"]
    if isinstance(mass, bool) or not np.isfinite(mass) or not 0 <= mass <= 1:
        raise ValueError("raw tool mass must be a finite probability")
    if (
        six.shape != (6,)
        or not np.isfinite(six).all()
        or np.any(six < 0)
        or not (np.isclose(six.sum(), 1, rtol=1e-5, atol=1e-7) or (mass == 0 and six.sum() == 0))
    ):
        raise ValueError(
            "six must be six finite nonnegative conditional probabilities summing to one"
        )
    top = value.get("top_token_id")
    if top is not None and (type(top) is not int or top < 0):
        raise ValueError("top_token_id must be an observed nonnegative token id")
    margin = value.get("top_margin")
    if margin is not None and (not np.isfinite(margin) or margin < 0):
        raise ValueError("top_margin must be finite and nonnegative")
    return {
        "six_token_id": tokens[int(six.argmax())] if mass > 0 else None,
        "mass": float(mass),
        "top_token_id": top,
        "top_margin": margin,
    }


def paired_metrics(rows, *, mass_floor=0.001):
    """Summarize unique cell/layer rows without conflating accuracy and coverage.

    Each row names row/position/layer/episode, expert_token_id, six tool_token_ids,
    and model/prose/agentic readouts (six conditional probabilities, raw mass,
    optionally the independently observed full-vocabulary top_token_id/top_margin).
    Both cell-weighted rates and equal-episode distributions are descriptive.
    """
    if not rows or not np.isfinite(mass_floor) or not 0 < mass_floor <= 1:
        raise ValueError("a nonempty paired population and positive mass floor are required")
    seen, grouped, records = set(), defaultdict(list), []
    declared_tokens = None
    for row in rows:
        key = (row["row"], row["position"], row["layer"])
        if key in seen:
            raise ValueError("duplicate comparison cell/layer")
        seen.add(key)
        if (
            type(row["row"]) is not int
            or row["row"] < 0
            or type(row["layer"]) is not int
            or row["layer"] < 1
            or row["position"] not in {"P_note", "P_act"}
            or not isinstance(row["episode"], str)
            or not row["episode"]
        ):
            raise ValueError("invalid comparison cell identity")
        tokens = row["tool_token_ids"]
        if (
            len(tokens) != 6
            or len(set(tokens)) != 6
            or any(type(t) is not int or t < 0 for t in tokens)
        ):
            raise ValueError("six unique tool token ids required")
        if declared_tokens is not None and tokens != declared_tokens:
            raise ValueError("tool token ordering changes across paired rows")
        declared_tokens = tokens
        if type(row["expert_token_id"]) is not int or row["expert_token_id"] < 0:
            raise ValueError("expert_token_id must be a nonnegative token id")
        model = _readout(row["model"], tokens)
        record = {
            field: row[field]
            for field in ("row", "position", "layer", "episode", "expert_token_id")
        }
        record["model"] = model
        for name in ("prose", "agentic"):
            value = _readout(row[name], tokens)
            resolved = value["mass"] >= mass_floor and model["mass"] >= mass_floor
            value.update(
                resolved=resolved,
                six_agreement=(
                    value["six_token_id"] == model["six_token_id"] if resolved else None
                ),
                full_argmax_agreement=(
                    value["top_token_id"] == model["top_token_id"]
                    if value["top_token_id"] is not None and model["top_token_id"] is not None
                    else None
                ),
                expert_argmax_agreement=(
                    value["top_token_id"] == row["expert_token_id"]
                    if value["top_token_id"] is not None
                    else None
                ),
            )
            record[name] = value
        records.append(record)
        grouped[f"L{row['layer']}/{row['position']}"].append(record)
    groups = {}
    for key, cells in grouped.items():
        joint = [c for c in cells if c["prose"]["resolved"] and c["agentic"]["resolved"]]
        group = {
            "n": len(cells),
            "joint_resolved": len(joint),
            "paired_accuracy_delta_on_joint_support": _mean(
                [
                    int(c["agentic"]["six_agreement"]) - int(c["prose"]["six_agreement"])
                    for c in joint
                ]
            ),
            "model_margin": _distribution(c["model"]["top_margin"] for c in cells),
        }
        for name in ("prose", "agentic"):
            episodes = defaultdict(list)
            for cell in cells:
                episodes[cell["episode"]].append(cell[name])
            episode_rows = [
                {
                    "episode": ep,
                    "n": len(values),
                    "coverage": _mean(v["resolved"] for v in values),
                    "agreement_of_resolved": _mean(v["six_agreement"] for v in values),
                }
                for ep, values in sorted(episodes.items())
            ]
            group[name] = {
                "coverage": _mean(c[name]["resolved"] for c in cells),
                "agreement_of_resolved": _mean(c[name]["six_agreement"] for c in cells),
                "full_argmax_agreement": _mean(c[name]["full_argmax_agreement"] for c in cells),
                "full_argmax_measured_n": sum(
                    c[name]["full_argmax_agreement"] is not None for c in cells
                ),
                "raw_mass": _distribution(c[name]["mass"] for c in cells),
                "episode_agreement": _distribution(
                    e["agreement_of_resolved"] for e in episode_rows
                ),
                "episode_coverage": _distribution(e["coverage"] for e in episode_rows),
                "episodes": episode_rows,
            }
        groups[key] = group
    return {
        "schema_version": 1,
        "mass_floor": mass_floor,
        "rows": records,
        "groups": groups,
        "statistical_status": "descriptive; no confidence bound or new domain admission",
        "perturbation": {"status": "not measured"},
    }


def _vector_comparison(prediction, reference):
    p, r = np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    pn, rn = np.linalg.norm(p), np.linalg.norm(r)
    return {
        "relative_error": float(np.linalg.norm(p - r) / rn) if rn else None,
        "cosine": float(np.dot(p.ravel(), r.ravel()) / (pn * rn)) if pn and rn else None,
        "reference_norm": float(rn),
        "prediction_norm": float(pn),
    }


def directional_check(tail, residual, direction, lens_map, *, baseline):
    """Compare one frozen displacement using the caller's exact model tail.

    tail maps this layer's residual vector to the target pre-final-norm vector.
    The caller owns capture/hook identity, direction selection and model precision.
    No random direction, finite-difference step or scientific arm is invented here.
    """
    import torch

    if (
        residual.dtype != torch.float32
        or direction.dtype != residual.dtype
        or residual.ndim != 1
        or direction.shape != residual.shape
        or direction.device != residual.device
    ):
        raise ValueError("residual and frozen direction must be matching float32 vectors")
    if not torch.isfinite(residual).all() or not torch.isfinite(direction).all():
        raise ValueError("nonfinite residual or direction")
    matrix = torch.as_tensor(lens_map, dtype=residual.dtype, device=residual.device)
    if matrix.shape != (residual.numel(), residual.numel()) or not torch.isfinite(matrix).all():
        raise ValueError("lens map must be a finite square map at the same endpoint")
    same = tail(residual)
    if not torch.equal(same, baseline):
        raise ValueError("same-state control failed before directional measurement")
    _, exact = torch.autograd.functional.jvp(tail, residual, direction, strict=False)
    actual = tail(residual + direction) - same
    prediction = matrix @ direction
    values = [x.detach().cpu().numpy() for x in (exact, actual, prediction)]
    if any(not np.isfinite(x).all() for x in values):
        raise ValueError("nonfinite directional response")
    return {
        "endpoint": "pre-final-norm",
        "precision": "float32",
        "same_state_equal": True,
        "exact_vs_actual": _vector_comparison(values[0], values[1]),
        "lens_vs_actual": _vector_comparison(values[2], values[1]),
        "lens_vs_exact": _vector_comparison(values[2], values[0]),
        "exact": values[0].tolist(),
        "actual": values[1].tolist(),
        "lens_prediction": values[2].tolist(),
    }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_cell_pairings(table, cells, metadata, provenance, layers, limit):
    """Admit exact cell receipts; identical position/length pairs may belong to different prompts."""
    from local_llm_lab.probes import sae_bridge as B

    if isinstance(limit, bool) or not np.isfinite(limit) or limit < 0:
        raise ValueError("finite nonnegative pairing limit required")
    checked = B._supplied_pairing_table(table)
    recorded = table.get("_provenance")
    if not isinstance(recorded, dict):
        raise ValueError("pairing cell receipts are absent")
    for key in (
        "lens_sha256",
        "nu_sha256",
        "corpus_sha256",
        "positions_sha256",
        "files_sha256",
        "checkpoint_sha256",
        "lens_sidecar_sha256",
        "fit_widths",
    ):
        if key not in provenance or recorded.get(key) != provenance[key]:
            raise ValueError(f"pairing provenance differs: {key}")
    code = recorded.get("implementation_sha256")
    if (
        not isinstance(recorded.get("source_commit"), str)
        or not recorded["source_commit"]
        or not isinstance(code, dict)
        or not code
        or any(
            not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in code.values()
        )
    ):
        raise ValueError("pairing implementation provenance is incomplete")
    available = checked.get(metadata["model"]["base"], {}).get("measured_pairings", {})
    widths = [row["forward_batch"] for row in metadata["fit_widths"]]
    for layer in layers:
        entries = available.get(str(layer), [])
        entries = entries if isinstance(entries, list) else [entries]
        measured = recorded.get("measurements", {}).get(str(layer), [])
        if len(entries) != len(measured) or not entries:
            raise ValueError("pairing registration/measurement coverage differs")
        keyed = {}
        for entry, measure in zip(entries, measured, strict=True):
            cell = measure.get("cell", {})
            key = (cell.get("row"), cell.get("position"))
            if key in keyed:
                raise ValueError("duplicate pairing cell receipt")
            keyed[key] = (entry, measure)
        for item in cells:
            key = (item["cell"]["row"], item["cell"]["position"])
            if key not in keyed:
                raise ValueError("exact pairing cell was not measured")
            entry, measure = keyed[key]
            expected = {
                "cell": item["cell"],
                "capture_forward": item["capture_forward"],
                "prompt_identity": item["prompt_identity"],
                "prompt_sha256": item["prompt_sha256"],
                "input_ids_sha256": hashlib.sha256(
                    json.dumps(item["input_ids"], separators=(",", ":")).encode()
                ).hexdigest(),
                "capture_residual_sha256": hashlib.sha256(
                    item["residuals"][layer].tobytes()
                ).hexdigest(),
            }
            if any(measure.get(k) != v for k, v in expected.items()):
                raise ValueError("pairing receipt differs from the captured cell")
            if entry["pair"] != item["identity"]:
                raise ValueError("pairing rule identity differs")
            terms = measure.get("chunk_terms", [])
            if (
                not terms
                or [t.get("forward_batch") for t in terms] != widths
                or any(
                    isinstance(t.get("relative"), bool)
                    or not isinstance(t.get("relative"), (float, int))
                    or not np.isfinite(t["relative"])
                    or t["relative"] < 0
                    for t in terms
                )
                or max(t["relative"] for t in terms) != entry["relative"]
                or entry["relative"] > limit
            ):
                raise ValueError("pairing measured terms or displacement exceed the bound")


def compare(args):
    """Validate capture/lens/pairing receipts, then summarize frozen readout rows.

    args contains measure_pairings' common file paths plus prose/agentic mappings
    with lens_archive, lens_sidecar, pairings, and a scores JSON with pinned sha256.
    The scores producer must bind each lens nu/archive and capture/corpus digests
    in provenance. Model forward/scoring production remains a separate operation.
    """
    from local_llm_lab.probes import measure_pairings as P

    output = Path(args["output"])
    if output.exists():
        raise FileExistsError(output)
    limit = args["maximum_pairing_relative"]
    if isinstance(limit, bool) or not np.isfinite(limit) or limit < 0:
        raise ValueError("maximum_pairing_relative must be explicitly finite and nonnegative")
    score_path = Path(args["scores"])
    if _sha(score_path) != args["scores_sha256"]:
        raise ValueError("scores hash mismatch")
    scores = json.loads(score_path.read_text())
    if scores.get("readout_convention") != "model-unembed-final-norm-and-softcap":
        raise ValueError("scores require the model unembed convention, not gain-only scores")
    expected_keys = None
    expected_identity = {}
    receipts = {}
    for name in ("prose", "agentic"):
        paths = {
            key: Path(args[key])
            for key in (
                "model_snapshot",
                "checkpoint_manifest",
                "capture_dir",
                "positions",
                "corpus",
            )
        }
        paths.update({key: Path(args[name][key]) for key in ("lens_archive", "lens_sidecar")})
        paths["capture_reference_archive"] = Path(args["prose"]["lens_archive"])
        paths["capture_reference_sidecar"] = Path(args["prose"]["lens_sidecar"])
        paths["layers"] = list(args["layers"])
        cells, metadata, provenance = P._admit(paths)
        keys = {
            (c["cell"]["row"], c["cell"]["position"], layer)
            for c in cells
            for layer in paths["layers"]
        }
        if len(keys) != len(cells) * len(paths["layers"]):
            raise ValueError("duplicate requested cells")
        if expected_keys is not None and keys != expected_keys:
            raise ValueError("prose and agentic populations differ")
        expected_keys = keys
        for cell in cells:
            for layer in paths["layers"]:
                key = (cell["cell"]["row"], cell["cell"]["position"], layer)
                expected_identity[key] = (
                    cell["prompt_identity"]["task_id"],
                    cell["cell"]["token_id"],
                )
        if scores["provenance"][name] != {
            key: provenance[key]
            for key in ("lens_sha256", "nu_sha256", "corpus_sha256", "positions_sha256")
        }:
            raise ValueError("scores provenance does not bind the admitted lens and population")
        pairings_path = Path(args[name]["pairings"])
        validate_cell_pairings(
            json.loads(pairings_path.read_text()),
            cells,
            metadata,
            provenance,
            paths["layers"],
            limit,
        )
        receipts[name] = {"admission": provenance, "pairings_sha256": _sha(pairings_path)}
    actual_keys = {(r["row"], r["position"], r["layer"]) for r in scores["rows"]}
    if actual_keys != expected_keys:
        raise ValueError("scored population differs from the complete requested population")
    for row in scores["rows"]:
        key = (row["row"], row["position"], row["layer"])
        if (row["episode"], row["expert_token_id"]) != expected_identity[key]:
            raise ValueError("scored episode or expert token differs from captured row")
    result = paired_metrics(scores["rows"])
    result["provenance"] = {
        "receipts": receipts,
        "scores_sha256": _sha(score_path),
        "config": copy.deepcopy(args),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(serialized)
    return result
