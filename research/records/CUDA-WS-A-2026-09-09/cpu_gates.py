"""Local, explicitly executed CPU calibration for WS-A; never a CUDA launcher.

This record companion checks same-dtype seam exactness at 64 and 1,400 positions, records
the promoted cross-precision floor, and requires controls to exceed that floor. Gates 3–4
remain explicitly unexecuted; passing these measurements is not complete CUDA acceptance.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import math
import os
import resource
import sys
import time
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GIB = 2**30
LENGTHS = (64, 1400)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path):
    source = Path(path).resolve(strict=True)
    value = json.loads(source.read_text())
    ids = value.get("token_ids") if isinstance(value, dict) else value
    if not isinstance(ids, list) or len(ids) < max(LENGTHS):
        raise ValueError("token JSON must contain at least 1,400 fixed token IDs")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("token IDs must be nonnegative integers, never booleans")
    return ids, {
        "path": str(source),
        "sha256": sha256(source),
        "selection": "first 64 and first 1400 supplied IDs; no padding or resampling",
    }


def assert_own_window(runlock):
    window = runlock.read_window()
    nonce = os.environ.get(runlock.WINDOW_HOLDER_ENV)
    if (
        window is None
        or not nonce
        or nonce != window.nonce
        or window.holder_state != "running"
        or runlock.blocking_window() is not None
    ):
        raise RuntimeError("requires this process's nonce under a running announced box window")
    return {
        "seat": window.seat,
        "holder_pid": window.pid,
        "holder_state": window.holder_state,
        "path": str(window.path),
        "purpose": window.purpose,
    }


def verified_source_commit(declared):
    from local_llm_lab import runlog

    actual = runlog.git_commit(ROOT)
    if actual == "unknown" or declared != actual:
        raise ValueError(f"declared source commit {declared!r} differs from actual {actual!r}")
    return actual


def verify_structure(measured, text_config):
    """Check discovery against HF's interpretation of the local header and its defaults."""
    expected = {
        "num_layers": text_config.num_hidden_layers,
        "hidden_size": text_config.hidden_size,
        "vocab_size": text_config.vocab_size,
        "attention_spans": [
            {"sliding_attention": "sliding", "full_attention": "global"}.get(kind)
            for kind in getattr(text_config, "layer_types", [None] * text_config.num_hidden_layers)
        ],
    }
    for field, value in expected.items():
        if measured.get(field) != value:
            raise ValueError(f"structural discovery disagrees with header config: {field}")
    return {**measured, "expected_from_header_config": expected}


def memory_reading():
    import psutil

    high = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "rss_bytes": psutil.Process().memory_info().rss,
        "process_peak_bytes": int(high if sys.platform == "darwin" else high * 1024),
        "peak_basis": "fresh-process lifetime high-water mark; CPU counter cannot reset",
    }


def rotary_rounding(view, ids):
    """Evaluate the model's own rotary module with fp32 output, against observed bf16 tables."""
    import torch

    entry, bundles = view._observe_forward(view._ids(ids))
    rotary = getattr(view.text_module, "rotary_emb", None)
    if rotary is None:
        raise ValueError("the discovered text module has no native rotary module")
    rows = {}
    for index, kind in enumerate(view.config.layer_types):
        if kind in rows:
            continue
        kwargs = bundles[index]
        positions = kwargs.get("position_ids")
        if positions is None:
            raise ValueError("native block bundle did not provide position_ids")
        expected = rotary(entry.float(), positions, kind)
        observed = kwargs["position_embeddings"]
        rows[kind] = {
            "max_abs": max(
                float((a.float() - b.float()).abs().max())
                for a, b in zip(expected, observed, strict=True)
            ),
            "observed_dtypes": [str(value.dtype) for value in observed],
        }
    assert not torch.is_grad_enabled()
    return {
        "by_attention_kind": rows,
        "max_abs": max(row["max_abs"] for row in rows.values()),
        "basis": "same model rotary module and positions; fp32 output versus native tables",
    }


def measure_length(view, ids, checkpoint, measurement):
    """Persist shared-probe phase reports before checking memory and starting the next phase."""
    from research.acceptance.torch_seam import residual_precision_probe

    measurement.update(
        tokens=len(ids),
        precision_probe={"status": "unexecuted"},
        phases={},
        rotary_rounding={"status": "unexecuted"},
    )

    def progress(phase, phase_report):
        # Copy scalar metadata only; the shared helper owns every tensor and releases each arm.
        measurement["phases"][phase] = dict(phase_report)
        checkpoint(phase)
        if (
            phase == "native_dtype_loop"
            and phase_report.get("status") == "measured"
            and not native_exact(phase_report, num_layers=view.num_layers)
        ):
            measurement["acceptance"] = {
                "status": "fail",
                "reason": "native dtype seam is not exact",
            }
            checkpoint("native_dtype_exactness_failure")
            raise RuntimeError("native dtype seam failed; later diagnostic arms are unexecuted")

    try:
        measurement["precision_probe"] = {"status": "running"}
        measurement["precision_probe"] = residual_precision_probe(view, ids, checkpoint=progress)
        checkpoint("precision_probe_complete")
        measurement["rotary_rounding"] = {"status": "running"}
        checkpoint("before_rotary_rounding")
        measurement["rotary_rounding"] = {"status": "measured", **rotary_rounding(view, ids)}
        checkpoint("rotary_rounding")
    except Exception:
        for name in ("precision_probe", "rotary_rounding"):
            if measurement[name]["status"] == "running":
                measurement[name]["status"] = "failed"
        raise
    return measurement


def valid_precision_arm(arm, *, num_layers):
    """Require every site and recompute scalar summaries before using a gate value."""
    if not isinstance(arm, dict) or arm.get("status") != "measured":
        return False
    rows = arm.get("by_layer", {})
    expected_layers = set(range(num_layers + 1))
    try:
        if len(rows) != len(expected_layers) or {int(key) for key in rows} != expected_layers:
            return False
        values = [arm["max_abs"], arm["max_norm_relative"]]
        values.extend(
            value[field]
            for value in rows.values()
            for field in ("max_abs", "native_max_abs", "max_norm_relative")
        )
        if not all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values
        ):
            return False
        for row in rows.values():
            scale, error = row["native_max_abs"], row["max_abs"]
            if scale == 0 and error != 0:
                return False
            relative = error / scale if scale else 0.0
            if row["max_norm_relative"] != relative:
                return False
        return all(
            arm[field] == max(row[field] for row in rows.values())
            for field in ("max_abs", "max_norm_relative")
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def native_exact(arm, *, num_layers):
    return (
        valid_precision_arm(arm, num_layers=num_layers)
        and arm.get("exact") is True
        and arm["max_abs"] == 0.0
    )


def assess_precision_length(measurement, *, num_layers):
    """Apply the dated ruling; no bound is applied to the cross-precision floor."""
    checks = {}
    tokens = measurement.get("tokens")
    probe = measurement.get("precision_probe", {})

    def valid_arm(arm):
        return valid_precision_arm(arm, num_layers=num_layers)

    native = probe.get("native_dtype_loop", {})
    floor = probe.get("promoted_fp32_loop", {})
    checks["complete_probe"] = probe.get("status") == "measured" and probe.get("tokens") == tokens
    checks["native_dtype_exact"] = native_exact(native, num_layers=num_layers)
    checks["precision_floor_recorded"] = valid_arm(floor)
    rotary = measurement.get("rotary_rounding", {})
    rotary_error = rotary.get("max_abs")
    checks["rotary_rounding_recorded"] = (
        rotary.get("status") == "measured"
        and type(rotary_error) in (int, float)
        and math.isfinite(rotary_error)
        and rotary_error >= 0
    )
    controls = probe.get("controls", {})
    required_controls = ("mask_dispatch", "hook_site_off_by_one", "entry_transform_omission")
    for name in required_controls:
        control = controls.get(name, {})
        checks[f"{name}_measured"] = valid_arm(control)
        if tokens == max(LENGTHS):
            checks[f"{name}_outside_floor"] = (
                valid_arm(control)
                and valid_arm(floor)
                and control["max_norm_relative"] > floor["max_norm_relative"]
            )
    if tokens == min(LENGTHS):
        mask = controls.get("mask_dispatch", {})
        checks["short_mask_control_exact"] = valid_arm(mask) and mask["max_abs"] == 0.0
    checks["required_length"] = tokens in LENGTHS
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "rule": "native exact zero; long controls strictly above the global precision floor",
    }


def assess_gate2(measurements, *, num_layers):
    """Both required lengths must be present exactly once; absent evidence can never pass."""
    lengths = [item.get("tokens") for item in measurements]
    if sorted(lengths, key=str) != sorted(LENGTHS, key=str):
        return {
            "status": "incomplete",
            "reason": "requires exactly one result at each declared length",
        }
    results = {
        str(item["tokens"]): assess_precision_length(item, num_layers=num_layers)
        for item in measurements
    }
    return {
        "status": "pass" if all(item["status"] == "pass" for item in results.values()) else "fail",
        "by_length": results,
        "native_dtype_bound": 0.0,
        "cross_precision_bound": None,
        "threshold_changed": False,
    }


def write_report(path, report):
    """Replace a complete record atomically, preserving prior valid JSON on failure."""
    from local_llm_lab.runlog import write_text_atomic

    write_text_atomic(Path(path), json.dumps(report, indent=2, allow_nan=False) + "\n")


def finish_report(path, report, started, *, preserving_exception):
    """Take one last best-effort high-water reading, including after allocation failures."""
    report["elapsed_seconds"] = time.monotonic() - started
    try:
        report["memory"].append({"phase": "final", **memory_reading()})
    except Exception as error:
        report["final_memory_error"] = {"type": type(error).__name__, "message": str(error)}
    try:
        write_report(path, report)
    except Exception as error:
        if not preserving_exception:
            raise
        # A recording problem must not replace the load/forward failure being propagated.
        with suppress(Exception):
            print(
                f"Final record write failed while preserving the original error: {error}",
                file=sys.stderr,
                flush=True,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--token-ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cap-gib", required=True, type=float)
    parser.add_argument("--projected-peak-gib", required=True, type=float)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("no execution without --execute; this command loads a checkpoint")
    if (
        not 0 < args.projected_peak_gib <= args.cap_gib
        or not math.isfinite(args.cap_gib)
        or args.threads < 1
    ):
        parser.error(
            "a finite positive projection within the explicit cap and positive threads are required"
        )
    if args.output.exists():
        parser.error("output already exists; use a new record path")
    report = {
        "declared_source_commit": args.source_commit,
        "source_commit": None,
        "script_sha256": sha256(__file__),
        "scope": "CPU calibration, not complete checkpoint acceptance",
        "device": "cpu",
        "weights_dtype": "bfloat16",
        "diagnostic_dtype": "float32",
        "seam_comparison_dtype": "bfloat16",
        "attention": "eager",
        "cap_gib": args.cap_gib,
        "projected_peak_gib": args.projected_peak_gib,
        "gates": {str(i): {"status": "unexecuted"} for i in range(1, 5)},
        "memory": [],
        "measurements": [],
        "cuda": "unexecuted",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        write_report(args.output, report)

    def checkpoint(phase):
        value = {"phase": phase, **memory_reading()}
        report["memory"].append(value)
        save()
        print(json.dumps(value), flush=True)
        if value["process_peak_bytes"] > args.cap_gib * GIB:
            raise RuntimeError("measured CPU process peak exceeds the declared R47 cap; stopping")

    save()
    started = time.monotonic()
    try:
        if any(key == "mlx" or key.startswith("mlx.") for key in sys.modules):
            raise RuntimeError("MLX is already imported; calibration requires a fresh CPU process")
        report["source_commit"] = verified_source_commit(args.source_commit)
        from local_llm_lab import runlock

        report["window"] = assert_own_window(runlock)
        runlock.hold_model_run_lock(session="WS-A CPU bf16 calibration")
        original_import = builtins.__import__

        def no_mlx(name, *positional, **keyword):
            if name.split(".")[0] in ("mlx", "mlx_lm"):
                raise ImportError("MLX is forbidden in the CPU calibration process")
            return original_import(name, *positional, **keyword)

        builtins.__import__ = no_mlx
        os.environ.update(LLL_BACKEND="torch", LLL_DEVICE="cpu", HF_HUB_OFFLINE="1")
        import torch
        from packaging.version import Version
        from transformers import AutoConfig

        from local_llm_lab import device
        from local_llm_lab.arch_torch import TorchArchitectureView
        from local_llm_lab.hf_text import checkpoint_metadata, load_text_causal_lm
        from research.acceptance.torch_seam import structural_report

        if Version(torch.__version__.split("+")[0]) < Version("2.14.0"):
            raise RuntimeError("WS-A CPU evidence requires torch >= 2.14.0")
        torch.set_num_threads(args.threads)
        torch.set_default_device("cpu")
        report["runtime"] = device.pin(seed=0, attention="eager")
        report["resolved_arch_source"] = str(sys.modules[TorchArchitectureView.__module__].__file__)
        metadata = checkpoint_metadata(args.checkpoint)
        if metadata["storage_dtypes"] != ["BF16"]:
            raise ValueError("this CPU run requires text weights stored uniformly in bf16")
        if metadata["text_bytes"] >= args.cap_gib * GIB:
            raise RuntimeError("text weights alone exceed the explicit memory cap")
        ids, report["tokens"] = read_ids(args.token_ids)
        report["checkpoint"] = {
            key: metadata[key] for key in ("path", "sha256", "text_bytes", "storage_dtypes")
        }
        checkpoint("before_load")
        with torch.no_grad():
            model, report["loading"] = load_text_causal_lm(
                metadata["path"], dtype="bfloat16", attn_implementation="eager", device="cpu"
            )
            if report["loading"]["sha256"] != metadata["sha256"]:
                raise ValueError("checkpoint hashes changed between preflight and the shared load")
            checkpoint("after_load")
            view = TorchArchitectureView.from_model(model)
            if max(ids[: max(LENGTHS)]) >= view.vocab_size:
                raise ValueError("supplied token IDs exceed the loaded model vocabulary")
            report["gates"]["1"] = {"status": "running"}
            header_config = dict(metadata["text_config"])
            model_type = header_config.pop("model_type")
            checked_structure = verify_structure(
                structural_report(view), AutoConfig.for_model(model_type, **header_config)
            )
            report["gates"]["1"] = {"status": "pass", **checked_structure}
            checkpoint("gate_1_structure")
            report["gates"]["2"] = {"status": "calibration_running"}
            for length in LENGTHS:
                measured = {"tokens": length}
                report["measurements"].append(measured)
                save()
                measure_length(
                    view,
                    ids[:length],
                    lambda phase, length=length: checkpoint(f"{length}/{phase}"),
                    measured,
                )
                measured["acceptance"] = assess_precision_length(
                    measured, num_layers=view.num_layers
                )
                save()
                if measured["acceptance"]["status"] != "pass":
                    report["gates"]["2"] = {
                        "status": "fail",
                        "failing_length": length,
                        "result": measured["acceptance"],
                    }
                    raise RuntimeError("residual precision gate failed; later gates are unexecuted")
            report["gates"]["2"] = assess_gate2(report["measurements"], num_layers=view.num_layers)
            if report["gates"]["2"]["status"] != "pass":
                raise RuntimeError("incomplete residual precision gate evidence")
            report["gates"]["3"]["reason"] = "not executed by this bounded calibration companion"
            report["gates"]["4"]["reason"] = "not executed by this bounded calibration companion"
            report["status"] = "gates 1–2 passed; gates 3–4 unexecuted"
            checkpoint("complete")
    except Exception as error:
        report["status"] = "stopped"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        if report["gates"]["1"]["status"] == "running":
            report["gates"]["1"]["status"] = "fail"
        if report["gates"]["2"]["status"] == "calibration_running":
            failed = [
                item
                for item in report["measurements"]
                if item.get("acceptance", {}).get("status") == "fail"
            ]
            report["gates"]["2"] = (
                {
                    "status": "fail",
                    "failing_length": failed[-1]["tokens"],
                    "result": failed[-1]["acceptance"],
                }
                if failed
                else {"status": "incomplete"}
            )
        raise
    finally:
        finish_report(
            args.output, report, started, preserving_exception=sys.exc_info()[0] is not None
        )
    return 1  # Full acceptance remains unexecuted even when calibration completed successfully.


if __name__ == "__main__":
    raise SystemExit(main())
