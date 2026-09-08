"""Fail-closed R47 launch qualification; importing this module does not import MLX."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path


def require_owned_window():
    from local_llm_lab.pipeline.lens_fitting.runtime import primary_worktree
    from local_llm_lab.runlock import (
        BOX_STATE_DIR_ENV,
        WINDOW_HOLDER_ENV,
        WINDOW_RELATIVE_PATH,
        read_window,
    )

    root = primary_worktree()
    window = read_window(root / WINDOW_RELATIVE_PATH)
    if (
        window is None
        or not os.environ.get(WINDOW_HOLDER_ENV)
        or window.nonce != os.environ[WINDOW_HOLDER_ENV]
        or window.holder_state != "running"
        or window.expected_end_epoch is None
        or window.expected_end_epoch <= time.time()
    ):
        raise ValueError("memory qualification requires a current owned primary window")
    os.environ[BOX_STATE_DIR_ENV] = str(root)
    return root


def positive(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("memory values must be finite and positive")
    return value


def check_projection(projected_bytes, working_set_bytes):
    cap = 0.6 * positive(working_set_bytes)
    if positive(projected_bytes) > cap:
        raise ValueError("R47 projection exceeds 0.6 of device working set; no override")
    return cap


def device_working_set(*, require_window=None, device_info=None):
    (require_window or require_owned_window)()
    if device_info is None:
        import mlx.core as mx

        device_info = mx.device_info
    return positive(device_info()["max_recommended_working_set_size"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ladder(maximum: int) -> list[int]:
    if maximum < 2:
        raise ValueError("corpus too short for two calibration sizes")
    if maximum <= 512:
        return sorted(
            {n for n in (max(1, maximum // 4), max(2, maximum // 2), maximum) if n <= maximum}
        )
    sizes = {maximum}
    size = 256
    while size <= maximum:
        sizes.add(size)
        size *= 2
    return sorted(sizes)


def project_peak(rows: list[dict], tokens: int, *, fixed_bytes: float) -> float:
    """Fit variable memory over a fixed resident/sums floor, quadratic or steeper."""
    if len(rows) < 2 or not math.isfinite(fixed_bytes) or fixed_bytes < 0:
        raise ValueError("two measured sizes and a finite nonnegative floor are required")
    left, right = rows[-2:]
    t0, t1 = left["tokens"], right["tokens"]
    p0, p1 = left["peak_bytes"], right["peak_bytes"]
    if not all(math.isfinite(v) and v > 0 for v in (t0, t1, p0, p1, tokens)):
        raise ValueError("finite positive sizes and peaks are required")
    if not t0 < t1 < tokens:
        raise ValueError("projection must extend strictly increasing measured sizes")
    # A theoretical floor may exceed a measured peak (e.g. allocator aliasing).
    # Reduce it, rather than clipping the variable term to zero and projecting flat.
    floor = fixed_bytes if fixed_bytes < min(p0, p1) else 0.9 * min(p0, p1)
    v0, v1 = p0 - floor, p1 - floor
    exponent = max(2.0, math.log(v1 / v0) / math.log(t1 / t0))
    return max(p1, floor + v1 * (tokens / t1) ** exponent)


def runtime_fingerprint():
    # Distribution lookup and source-byte reads do not import either runtime.
    from importlib.metadata import distribution

    result = {}
    for name in ("mlx", "mlx-lm"):
        dist = distribution(name)
        files = {}
        for item in dist.files or ():
            path = str(item)
            if (path.endswith(".py") and path.startswith(("mlx/", "mlx_lm/"))) or path.endswith(
                ".dist-info/RECORD"
            ):
                files[path] = sha(dist.locate_file(item))
        if not files:
            raise ValueError("runtime distribution has no source inventory")
        result[name] = {"version": dist.version, "files": files}
    return result


def binding(prepared, residual_source, statistics_storage):
    if statistics_storage not in ("memory", "split_spill"):
        raise ValueError("invalid statistics storage")
    root = Path(__file__).resolve().parents[4]
    paths = [
        Path(__file__),
        Path(__file__).with_name("regression.py"),
        Path(__file__).with_name("runtime.py"),
        Path(__file__).with_name("artifacts.py"),
        root / "src/local_llm_lab/arch.py",
        root / "scripts/lens_regression_preflight.py",
        root / "scripts/lens_fit.py",
    ]
    config = json.loads((Path(prepared.snapshot["snapshot_path"]) / "config.json").read_bytes())
    config = config.get("text_config", config)
    geometry = {"hidden_size": config["hidden_size"], "num_layers": config["num_hidden_layers"]}
    if any(type(v) is not int or v <= 0 for v in geometry.values()) or geometry["num_layers"] < 2:
        raise ValueError("invalid snapshot geometry")
    rows = [
        {
            "input": len(r["ids"]),
            "scored": len(r.get("score_positions", r["ids"])),
            "split": r["split"],
        }
        for r in prepared.rows
    ]
    return {
        "schema_version": 1,
        "snapshot": prepared.snapshot,
        "corpus_manifest_sha256": prepared.corpus_manifest_sha256,
        "sequences_sha256": prepared.manifest["sequences"]["sha256"],
        "residual_source": residual_source,
        "statistics_storage": statistics_storage,
        "implementation": {str(p.relative_to(root)): sha(p) for p in paths},
        "workload": rows,
        "geometry": geometry,
        "runtime": runtime_fingerprint(),
        "allocator_cache_bytes": 0,
    }


def validate_evidence(path, expected, working_set_bytes):
    raw = Path(path).read_bytes()
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if (
        not events
        or events[0].get("event") != "begin"
        or events[0].get("qualification_binding") != expected
    ):
        raise ValueError("preflight binding differs from this fit")
    if events[-1] != {"event": "end", "status": "measured", "fit_started": False}:
        raise ValueError("preflight did not complete successfully")
    if any(e.get("event") in ("stopped", "end") for e in events[1:-1]):
        raise ValueError("preflight contains a refusal")
    begin = events[0]
    if begin.get("mode") != "qualification":
        raise ValueError("diagnostic preflight cannot qualify a full fit")
    if type(begin.get("repetitions_per_split")) is not int or begin["repetitions_per_split"] != 4:
        raise ValueError("qualification requires four repetitions per split")
    check_projection(begin["initial_bound_bytes"], working_set_bytes)
    measured = [e for e in events if e.get("event") == "measured"]
    canonical = ladder(max(r["input"] for r in expected["workload"]))
    if begin["lengths"] != canonical or [e["tokens"] for e in measured] != canonical:
        raise ValueError("preflight ladder is incomplete")
    for row in measured:
        counts = row.get("counts", {})
        if any(
            type(counts.get(split, {}).get("sequences")) is not int
            or counts[split]["sequences"] != 4
            for split in ("fit", "held")
        ):
            raise ValueError("calibration did not measure four sequences per split")
    solves = [e for e in events if e.get("event") == "measured_solve"]
    loads = [e for e in events if e.get("event") == "loaded"]
    if len(solves) != 1 or len(loads) != 1:
        raise ValueError("preflight loading/solve evidence is incomplete")
    envelopes = [e for e in events if e.get("event") == "full_fit_envelope"]
    if len(envelopes) != 1:
        raise ValueError("preflight lacks full-fit host and serialization reserve")
    envelope = envelopes[0]
    dense = positive(envelope["dense_matrix_bytes"])
    depth = envelope["nonfinal_layers"]
    if type(depth) is not int or depth <= 0:
        raise ValueError("invalid full-fit layer count")
    if (
        dense != 4 * expected["geometry"]["hidden_size"] ** 2
        or depth != expected["geometry"]["num_layers"] - 1
    ):
        raise ValueError("full-fit envelope geometry differs from snapshot")
    stats = (2 if expected["statistics_storage"] == "split_spill" else 4) * dense * depth
    if loads[0]["statistics_bytes"] != stats:
        raise ValueError("statistics floor differs from storage geometry")
    fixed = positive(loads[0]["resident_bytes"]) + stats
    for index, row in enumerate(measured):
        prediction = (
            begin["initial_bound_bytes"]
            if index < 2
            else project_peak(measured[:index], row["tokens"], fixed_bytes=fixed)
        )
        if row["projected_peak_bytes"] != prediction:
            raise ValueError("recorded ladder projection differs from measured recurrence")
        if index < 2 and row["peak_bytes"] > begin["initial_bound_bytes"]:
            raise ValueError("initial bound was falsified")
    reserve = 2 * depth * dense
    projected = positive(envelope["active_floor_bytes"]) + 16 * dense + reserve
    if (
        envelope["host_maps_and_serialization_bytes"] != reserve
        or envelope["solver_workspace_bytes"] != 16 * dense
        or envelope["projected_peak_bytes"] != projected
        or solves[0]["projected_peak_bytes"] != projected
    ):
        raise ValueError("full-fit analytical reserves do not match")
    check_projection(projected, working_set_bytes)
    check_projection(loads[0]["load_peak_bytes"], working_set_bytes)
    for event in measured + solves:
        check_projection(event["peak_bytes"], working_set_bytes)
        check_projection(event["projected_peak_bytes"], working_set_bytes)
    maximum = measured[-1]
    if any(
        r["input"] > maximum["tokens"] or r["scored"] > maximum["scored_positions_per_sequence"]
        for r in expected["workload"]
    ):
        raise ValueError("calibration does not dominate the full fit workload")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "binding": expected,
        "working_set_bytes": working_set_bytes,
        "cap_bytes": 0.6 * working_set_bytes,
        "full_fit_envelope": envelope,
        "qualified_peak_bytes": max(
            begin["initial_bound_bytes"],
            projected,
            *(e["projected_peak_bytes"] for e in measured),
            *(e["peak_bytes"] for e in measured + solves),
            loads[0]["load_peak_bytes"],
        ),
    }


def guard_runtime_event(event, qualification, *, require_window=None):
    """Refuse the next phase after a measured breach; cannot prevent its first spike."""
    if event.get("phase") in ("before_forward", "solve_start", "artifact_start"):
        (require_window or require_owned_window)()
    peak = positive(event["peak_memory_bytes"])
    cap = qualification["cap_bytes"]
    if peak > min(cap, qualification["qualified_peak_bytes"]):
        raise ValueError(
            "measured memory breach; stop before further work (not first-spike prevention): "
            f"peak_bytes={peak}, qualified_peak_bytes={qualification['qualified_peak_bytes']}, "
            f"cap_bytes={cap}"
        )
    if event.get("phase") in ("solve_start", "artifact_start"):
        envelope = qualification["full_fit_envelope"]
        projected = (
            positive(event["active_memory_bytes"]) + envelope["host_maps_and_serialization_bytes"]
        )
        if event["phase"] == "solve_start":
            projected += envelope["solver_workspace_bytes"]
        check_projection(projected, qualification["working_set_bytes"])
