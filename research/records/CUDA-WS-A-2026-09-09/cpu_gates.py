"""WS-A device gates: exact native seam, precision floor, controls and float32-load control.

The historical filename is retained for the first-hour runbook. Real checkpoints require CUDA;
CPU is limited to explicitly named, size-bounded serialized fixtures. Gates 3–4 (decode/readout)
remain separate and unexecuted by this script. No CUDA result is inferred from a fixture pass.
"""

from __future__ import annotations

import argparse
import builtins
import gc
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


def memory_reading(selected=None):
    import psutil

    high = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    value = {
        "rss_bytes": psutil.Process().memory_info().rss,
        "process_peak_bytes": int(high if sys.platform == "darwin" else high * 1024),
        "peak_method": "fresh-process lifetime high-water mark; CPU counter cannot reset",
    }
    if selected is None:
        return value
    from local_llm_lab import device

    value.update(
        allocated_bytes=device.working_set(selected),
        device_peak_bytes=device.peak(selected),
        device_peak_method="CUDA allocated tensor peak, not total occupancy"
        if selected.startswith("cuda")
        else "CPU process lifetime high-water mark, not a resettable allocator peak",
    )
    return value


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
        and all(
            row.get("actual_dtype") == row.get("native_dtype") == "torch.bfloat16"
            for row in arm["by_layer"].values()
        )
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
    checks["precision_floor_recorded"] = valid_arm(floor) and all(
        row.get("actual_dtype") == "torch.float32" and row.get("native_dtype") == "torch.bfloat16"
        for row in floor["by_layer"].values()
    )
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
        checks[f"{name}_measured"] = valid_arm(control) and all(
            row.get("actual_dtype") == row.get("native_dtype") == "torch.bfloat16"
            for row in control["by_layer"].values()
        )
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

    payload = (
        annotate_numbers(report) if report.get("number_schema") == "value-basis-v1" else report
    )
    write_text_atomic(Path(path), json.dumps(payload, indent=2, allow_nan=False) + "\n")


def finish_report(path, report, started, *, preserving_exception):
    """Take one last best-effort high-water reading, including after allocation failures."""
    report["elapsed_seconds"] = time.monotonic() - started
    try:
        reading = (
            memory_reading(report["device"]) if report.get("device_ready") else memory_reading()
        )
        report["memory"].append({"phase": "final", **reading})
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


# Numeric scalars remain plain inside the evaluator. On disk every number has an explicit basis.
NUMBER_BASES = frozenset({"measured-here", "laptop-basis", "expected"})
CPU_FIXTURE_MAX_FP32_BYTES = 1024 * 1024
FLOAT32_BOUND = 1e-3


def annotate_numbers(value, basis="measured-here"):
    if basis not in NUMBER_BASES:
        raise ValueError("unknown numeric basis")
    if type(value) in (int, float):
        return {"value": value, "basis": basis}
    if isinstance(value, (tuple, list)):
        return [annotate_numbers(item, basis) for item in value]
    if isinstance(value, dict):
        if set(value) == {"value", "basis"} and value.get("basis") in NUMBER_BASES:
            if type(value["value"]) not in (int, float):
                raise ValueError("numeric provenance wrapper contains a non-number")
            return dict(value)
        if value.get("basis") in NUMBER_BASES:
            basis = value["basis"]
        result = {}
        for key, item in value.items():
            name = str(key)
            child_basis = basis
            if (
                name
                in {
                    "expected",
                    "expected_from_header_config",
                    "plan",
                    "limits",
                    "thresholds",
                    "bound",
                }
                or "projected" in name
                or name.endswith("_bound")
                or name.endswith("_cap_bytes")
                or name == "cap_gib"
            ):
                child_basis = "expected"
            if name == "basis" and item not in NUMBER_BASES:
                name = "method"
            result[name] = annotate_numbers(item, child_basis)
        return result
    return value


def plain_numbers(value):
    if isinstance(value, dict):
        if set(value) == {"value", "basis"} and value["basis"] in NUMBER_BASES:
            if type(value["value"]) not in (int, float):
                raise ValueError("numeric provenance wrapper contains a non-number")
            return value["value"]
        return {key: plain_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain_numbers(item) for item in value]
    return value


def validate_execution_device(selected, *, fixture_cpu, metadata):
    """Refuse laptop model work before allocation; fixture names alone confer no exemption."""
    promoted_bytes = sum(math.prod(item["shape"]) * 4 for item in metadata["text"].values())
    if selected == "cpu":
        if not fixture_cpu or promoted_bytes > CPU_FIXTURE_MAX_FP32_BYTES:
            raise RuntimeError("CPU is fixtures only: explicit --fixture-cpu and <=1 MiB fp32 text")
    elif not selected.startswith("cuda") or fixture_cpu:
        raise RuntimeError("full checkpoints require CUDA; --fixture-cpu requires CPU")
    return promoted_bytes


def valid_float32_length(arm, *, num_layers):
    return (
        arm.get("tokens") in LENGTHS
        and valid_precision_arm(arm, num_layers=num_layers)
        and all(
            row.get("actual_dtype") == row.get("native_dtype") == "torch.float32"
            and row["max_norm_relative"] <= FLOAT32_BOUND
            for row in arm.get("by_layer", {}).values()
        )
    )


def assess_float32_control(measurements, *, num_layers):
    lengths = [item.get("tokens") for item in measurements]
    if sorted(lengths, key=str) != sorted(LENGTHS, key=str):
        return {"status": "incomplete", "reason": "requires exactly 64 and 1400 tokens"}
    checks = {}
    for arm in measurements:
        checks[str(arm["tokens"])] = valid_precision_arm(arm, num_layers=num_layers) and all(
            row.get("actual_dtype") == row.get("native_dtype") == "torch.float32"
            and row["max_norm_relative"] <= FLOAT32_BOUND
            for row in arm.get("by_layer", {}).values()
        )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "by_length": checks,
        "relative_bound": FLOAT32_BOUND,
        "rule": "all weights float32; promoted loop versus native float32, each residual site",
    }


def measure_float32(view, ids, checkpoint, measurement):
    from research.acceptance.torch_seam import residual_precision_probe

    measurement.update(tokens=len(ids), status="running", phases={})

    def progress(phase, result):
        measurement["phases"][phase] = dict(result)
        checkpoint(phase)

    try:
        probe = residual_precision_probe(view, ids, checkpoint=progress, include_controls=False)
        measurement.update(probe["promoted_fp32_loop"])
        checkpoint("float32_comparison_complete")
    except BaseException:
        if measurement["status"] == "running":
            measurement["status"] = "failed"
        raise
    return measurement


def valid_resource_evidence(readings, *, assessed_phase, device_cap, host_cap):
    if not isinstance(readings, list) or not readings:
        return False
    if not isinstance(readings[-1], dict) or readings[-1].get("phase") != assessed_phase:
        return False
    for row in readings:
        if not isinstance(row, dict):
            return False
        for field, cap in (("device_peak_bytes", device_cap), ("process_peak_bytes", host_cap)):
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= cap:
                return False
    return True


def source_fingerprint():
    """Identify executable inputs; generated reports do not invalidate their own resume keys."""
    paths = [Path(__file__), ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for directory in (ROOT / "src", ROOT / "research" / "acceptance"):
        paths.extend(directory.rglob("*.py"))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(paths))}
    from local_llm_lab.upstream_ref import load_upstream

    upstream = load_upstream()
    for path in sorted((upstream.path / "jlens").rglob("*.py")):
        hashes["upstream/" + str(path.relative_to(upstream.path))] = sha256(path)
    hashes["upstream-provenance"] = hashlib.sha256(
        json.dumps(upstream.provenance(), sort_keys=True).encode()
    ).hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def resume_store(directory, *, source_commit, fingerprint, metadata, runtime, inputs):
    from dataclasses import asdict

    from research.acceptance.gate_records import GateRecords, Identity, input_digest

    identity = Identity(
        tree_hash=fingerprint,
        checkpoint_sha256=hashlib.sha256(
            json.dumps(metadata["sha256"], sort_keys=True).encode()
        ).hexdigest(),
        device=json.dumps(runtime, sort_keys=True),
        # GateRecords treats a source commit as descriptive; this order requires it in the key.
        input_digest=input_digest(source_commit=source_commit, **inputs),
        source_commit=source_commit,
    )

    class AtomicRecords(GateRecords):
        def write(self, gate, result):
            payload = {
                "number_schema": "value-basis-v1",
                "gate": gate,
                "identity": asdict(self.identity),
                "source_commit": source_commit,
                "checkpoint_sha256": metadata["sha256"],
                "device_description": runtime,
                **result,
            }
            path = self._path(gate)
            write_report(path, payload)
            return path

        def completed(self, gate):
            try:
                value = super().completed(gate)
                if value is None:
                    return None
                value = plain_numbers(value)
                if value.get("status") != "pass":
                    self.refusals[gate] = (
                        "prior unit did not pass; rerunning it from native capture"
                    )
                    return None
                return value
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                self.refusals[gate] = f"invalid prior record: {error}"
                return None

    return AtomicRecords(directory, identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--token-ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cap-gib", required=True, type=float)
    parser.add_argument(
        "--projected-peak-gib",
        required=True,
        type=float,
        help="Worst device peak over both serial loads, including float32",
    )
    parser.add_argument(
        "--host-cap-gib", type=float, help="Host RSS cap; defaults to the shared R47 host budget"
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="Shared selector; real checkpoints require cuda or cuda:N")
    parser.add_argument("--fixture-cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("no execution without --execute; this command loads a checkpoint")
    if (
        not 0 < args.projected_peak_gib <= args.cap_gib
        or not math.isfinite(args.cap_gib)
        or args.threads < 1
        or (
            args.host_cap_gib is not None
            and (not math.isfinite(args.host_cap_gib) or args.host_cap_gib <= 0)
        )
    ):
        parser.error("finite positive caps/projection within cap and positive threads are required")
    if args.output.exists() and not args.resume:
        parser.error("output already exists; use --resume or a new record path")
    report = {
        "number_schema": "value-basis-v1",
        "declared_source_commit": args.source_commit,
        "source_commit": None,
        "script_sha256": sha256(__file__),
        "checkpoint": None,
        "device_description": None,
        "scope": "tiny CPU fixture" if args.fixture_cpu else "CUDA seam and precision gates",
        "device": None,
        "device_ready": False,
        "attention": "eager",
        "weights_dtypes": ["bfloat16", "float32"],
        "plan": {
            "lengths": LENGTHS,
            "seed": args.seed,
            "threads": args.threads,
            "cap_gib": args.cap_gib,
            "projected_peak_gib": args.projected_peak_gib,
        },
        "gates": {str(i): {"status": "unexecuted"} for i in range(1, 5)},
        "float32_control": {"status": "unexecuted"},
        "memory": [],
        "measurements": [],
        "float32_measurements": [],
        "resume": {},
        "cuda": "unexecuted",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    original_import = builtins.__import__
    model = view = None
    selected = None
    host_cap = None

    report_writable = not (args.resume and args.output.exists())

    def save():
        if report_writable:
            write_report(args.output, report)

    def checkpoint(phase):
        value = {
            "phase": phase,
            "elapsed_seconds": time.monotonic() - started,
            **memory_reading(selected),
        }
        report["memory"].append(value)
        save()
        print(json.dumps(annotate_numbers(value)), flush=True)
        if value["device_peak_bytes"] > args.cap_gib * GIB:
            raise RuntimeError("measured device peak exceeds the declared cap; stopping")
        if value["process_peak_bytes"] > host_cap:
            raise RuntimeError("measured host process peak exceeds the host cap; stopping")

    def no_mlx(name, *positional, **keyword):
        if name.split(".")[0] in ("mlx", "mlx_lm"):
            raise ImportError("MLX is forbidden in the WS-A torch gate process")
        return original_import(name, *positional, **keyword)

    save()
    try:
        if any(key.split(".")[0] in ("mlx", "mlx_lm") for key in sys.modules):
            raise RuntimeError("MLX is already imported; use a fresh torch process")
        report["source_commit"] = verified_source_commit(args.source_commit)
        builtins.__import__ = no_mlx
        os.environ.update(LLL_BACKEND="torch", HF_HUB_OFFLINE="1")
        from local_llm_lab import device

        # select checks availability, so set the deterministic environment before even that probe.
        os.environ[device.CUBLAS_ENV] = device.CUBLAS_DETERMINISTIC
        import torch
        from packaging.version import Version
        from transformers import AutoConfig

        from local_llm_lab.arch_torch import TorchArchitectureView
        from local_llm_lab.hf_text import checkpoint_metadata, load_text_causal_lm
        from research.acceptance.torch_seam import structural_report

        if Version(torch.__version__.split("+")[0]) < Version("2.14.0"):
            raise RuntimeError("WS-A evidence requires torch >= 2.14.0")
        if torch.cuda.is_initialized():
            raise RuntimeError("device gates require pinning before first CUDA initialization")
        selected = device.select(args.device)
        os.environ[device.DEVICE_ENV] = selected
        torch.set_num_threads(args.threads)
        device.pin(seed=args.seed, attention="eager")
        report["device_description"] = device.describe()
        report["device"] = selected
        report["device_ready"] = True
        report["cuda"] = (
            "selected; no checkpoint run yet" if selected.startswith("cuda") else "unexecuted"
        )
        report["device_info"] = device.device_info(selected)
        report["resolved_arch_source"] = str(sys.modules[TorchArchitectureView.__module__].__file__)
        metadata = checkpoint_metadata(args.checkpoint)
        report["checkpoint"] = {
            key: metadata[key] for key in ("path", "sha256", "text_bytes", "storage_dtypes")
        }
        promoted_bytes = validate_execution_device(
            selected, fixture_cpu=args.fixture_cpu, metadata=metadata
        )
        if metadata["storage_dtypes"] != ["BF16"]:
            raise ValueError("both precision arms must start from uniformly stored bf16 weights")
        if promoted_bytes > args.projected_peak_gib * GIB:
            raise RuntimeError("float32 text weights alone exceed the declared peak projection")
        device_cap = device.budget(device=selected)
        host_cap = (
            int(args.host_cap_gib * GIB) if args.host_cap_gib else device.budget(device="cpu")
        )
        if host_cap > device.budget(device="cpu"):
            raise RuntimeError("declared host cap exceeds the shared R47 host budget")
        if args.cap_gib * GIB > device_cap:
            raise RuntimeError("declared device cap exceeds the shared R47 budget")
        # CPU-load then move is the shared loader's current path. This is tensor storage only,
        # not a full host-peak prediction; native host high-water is measured at each phase.
        host_tensor_basis = promoted_bytes + metadata["text_bytes"]
        report["limits"] = {
            "device_cap_bytes": int(args.cap_gib * GIB),
            "host_cap_bytes": host_cap,
            "projected_peak_gib": args.projected_peak_gib,
            "float32_text_bytes": promoted_bytes,
            "host_tensor_storage_basis_bytes": host_tensor_basis,
            "host_basis_limit": "CPU load plus destination dtype, excludes runtime and activations",
            "native_dtype_bound": 0.0,
            "float32_bound": FLOAT32_BOUND,
        }
        if host_tensor_basis > host_cap:
            raise RuntimeError("shared loader host tensor storage basis exceeds the host cap")
        ids, report["tokens"] = read_ids(args.token_ids)
        header_config = dict(metadata["text_config"])
        config = AutoConfig.for_model(header_config.pop("model_type"), **header_config)
        if max(ids[: max(LENGTHS)]) >= config.vocab_size:
            raise ValueError("supplied token IDs exceed the checkpoint vocabulary")
        report["code_digest"], report["code_sha256"] = source_fingerprint()
        from local_llm_lab.upstream_ref import load_upstream

        report["upstream"] = load_upstream().provenance()
        report["code_digest_method"] = "SHA256 of executable source files; reports excluded"
        inputs = {
            "token_ids_sha256": report["tokens"]["sha256"],
            "lengths": LENGTHS,
            "dtypes": report["weights_dtypes"],
            "attention": "eager",
            "seed": args.seed,
            "limits": report["limits"],
            "fixture_cpu": args.fixture_cpu,
            "number_schema": report["number_schema"],
        }
        store = resume_store(
            args.output.parent / (args.output.stem + "-gates"),
            source_commit=report["source_commit"],
            fingerprint=report["code_digest"],
            metadata=metadata,
            runtime=report["device_description"],
            inputs=inputs,
        )
        import psutil

        report["box_state"] = {
            "cpu_percent_1s": psutil.cpu_percent(interval=1),
            "load_average": list(os.getloadavg()),
            "logical_cpus": psutil.cpu_count(),
            "idle": "not established; pre-run reading, not continuous measurement",
        }
        if args.fixture_cpu:
            report["window"] = {"scope": "size-bounded CPU fixtures require no model lock"}
        else:
            from local_llm_lab import runlock

            report["window"] = assert_own_window(runlock)
            runlock.hold_model_run_lock(session="WS-A CUDA seam and float32 control")
            device.set_cache_limit(int(args.cap_gib * GIB), selected)
        if not report_writable:
            from local_llm_lab.runlog import write_text_atomic

            old_hash = sha256(args.output)
            archive = args.output.with_name(args.output.stem + ".attempt-" + old_hash + ".json")
            write_text_atomic(archive, args.output.read_text())
            report["previous_attempt"] = {"path": str(archive), "sha256": old_hash}
            report_writable = True
        report["peak_reset"] = device.reset_peak(selected)
        checkpoint("before_load")
        structure = None
        for precision_index, dtype in enumerate(report["weights_dtypes"]):
            destination = report["measurements" if dtype == "bfloat16" else "float32_measurements"]
            pending = []
            for index, length in enumerate(LENGTHS):
                unit = 1 + precision_index * len(LENGTHS) + index
                cached = store.completed(unit) if args.resume else None
                if cached is not None:
                    try:
                        candidate = cached["measurement"]
                        if not valid_resource_evidence(
                            cached.get("memory"),
                            assessed_phase=f"{dtype}/{length}/assessed",
                            device_cap=args.cap_gib * GIB,
                            host_cap=host_cap,
                        ):
                            raise ValueError(
                                "missing, incomplete or over-cap original memory evidence"
                            )
                        if candidate.get("tokens") != length:
                            raise ValueError("wrong saved length")
                        if any(
                            "checkpoint_error" in arm
                            for arm in candidate.get("phases", {}).values()
                        ):
                            raise ValueError("prior unit has a checkpoint/resource failure")
                        accepted = (
                            assess_precision_length(candidate, num_layers=config.num_hidden_layers)[
                                "status"
                            ]
                            == "pass"
                            if dtype == "bfloat16"
                            else valid_float32_length(
                                candidate, num_layers=config.num_hidden_layers
                            )
                        )
                        if not accepted:
                            raise ValueError("saved measurements fail current acceptance checks")
                        structure = verify_structure(cached["structure"], config)
                        destination.append(candidate)
                        report.setdefault("resumed_resources", {})[str(unit)] = cached.get(
                            "memory", []
                        )
                        report["resume"][str(unit)] = "reused after identity and acceptance recheck"
                        continue
                    except (KeyError, TypeError, ValueError, AttributeError) as error:
                        store.refusals[unit] = f"saved unit is incomplete or invalid: {error}"
                report["resume"][str(unit)] = store.refusals.get(unit, "no completed matching unit")
                pending.append((unit, length))
            save()
            if not pending:
                continue
            checkpoint(f"before_{dtype}_load")
            with torch.no_grad():
                model, loading = load_text_causal_lm(
                    metadata["path"], dtype=dtype, attn_implementation="eager", device=selected
                )
                report.setdefault("loading", {})[dtype] = loading
                if loading["sha256"] != metadata["sha256"]:
                    raise ValueError("checkpoint hashes changed between preflight and shared load")
                expected_device = torch.device(selected)
                if expected_device.type == "cuda" and expected_device.index is None:
                    expected_device = torch.device("cuda", torch.cuda.current_device())
                if any(
                    parameter.device != expected_device or str(parameter.dtype) != f"torch.{dtype}"
                    for parameter in model.parameters()
                ):
                    raise RuntimeError(
                        "shared load parameter device/dtype differs from declared arm"
                    )
                checkpoint(f"after_{dtype}_load")
                view = TorchArchitectureView.from_model(model)
                structure = verify_structure(structural_report(view), config)
                report["gates"]["1"] = {"status": "pass", **structure}
                checkpoint(f"{dtype}/gate_1_structure")
                for unit, length in pending:
                    measured = {"tokens": length}
                    destination.append(measured)
                    save()

                    def callback(phase, length=length, dtype=dtype):
                        checkpoint(f"{dtype}/{length}/{phase}")

                    if dtype == "bfloat16":
                        measure_length(view, ids[:length], callback, measured)
                        measured["acceptance"] = assess_precision_length(
                            measured, num_layers=view.num_layers
                        )
                    else:
                        measure_float32(view, ids[:length], callback, measured)
                        measured["acceptance"] = {
                            "status": "pass"
                            if valid_float32_length(measured, num_layers=view.num_layers)
                            else "fail",
                            "relative_bound": FLOAT32_BOUND,
                        }
                    checkpoint(f"{dtype}/{length}/assessed")
                    if measured["acceptance"]["status"] != "pass":
                        raise RuntimeError(
                            f"{dtype} residual gate failed at {length}; later units unexecuted"
                        )
                    store.write(
                        unit,
                        {
                            "status": "pass",
                            "precision": dtype,
                            "measurement": measured,
                            "structure": structure,
                            "loading": loading,
                            "memory": report["memory"],
                            "memory_method": "original attempt readings through the completed unit",
                        },
                    )
                view = model = None
                gc.collect()
                device.clear_cache(selected)
                checkpoint(f"after_{dtype}_release")
        report["gates"]["1"] = {"status": "pass", **structure}
        report["gates"]["2"] = assess_gate2(
            report["measurements"], num_layers=config.num_hidden_layers
        )
        report["float32_control"] = assess_float32_control(
            report["float32_measurements"], num_layers=config.num_hidden_layers
        )
        if (
            report["gates"]["2"]["status"] != "pass"
            or report["float32_control"]["status"] != "pass"
        ):
            raise RuntimeError("incomplete precision gate evidence")
        for gate in ("3", "4"):
            report["gates"][gate]["reason"] = (
                "decode identity/readout distribution belong to the next device gate"
            )
        report["status"] = "pass"
        report["acceptance_scope"] = (
            "structure, residual seam and float32 control only; gates 3–4 unexecuted"
        )
        report["memory_scope"] = (
            "memory is this process only; resumed_resources preserves earlier model-load readings"
        )
        if selected.startswith("cuda"):
            report["cuda"] = (
                "measured-here" if report.get("loading") else "resumed; no new checkpoint forwards"
            )
        checkpoint("complete")
    except Exception as error:
        report["status"] = "stopped"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["gates"]["2"] = (
            assess_gate2(report["measurements"], num_layers=config.num_hidden_layers)
            if "config" in locals()
            else {"status": "unexecuted"}
        )
        report["float32_control"] = (
            assess_float32_control(
                report["float32_measurements"], num_layers=config.num_hidden_layers
            )
            if "config" in locals()
            else {"status": "unexecuted"}
        )
        raise
    finally:
        view = model = None
        builtins.__import__ = original_import
        if report_writable:
            finish_report(
                args.output, report, started, preserving_exception=sys.exc_info()[0] is not None
            )
    return 0  # Success of this bounded step only; not the entire device acceptance suite.


if __name__ == "__main__":
    raise SystemExit(main())
