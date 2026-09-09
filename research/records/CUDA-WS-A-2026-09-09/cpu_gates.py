"""Local, explicitly executed CPU calibration for WS-A; never a CUDA launcher.

This record companion measures the complete residual-comparison allocation at 64 and 1,400
positions. It does not promote a measurement to full acceptance: the unresolved precision
comparison and the two additional control conventions are recorded as unexecuted gates.
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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
GIB = 2**30
LENGTHS = (64, 1400)
KEY_MAPPING = {r"^language_model\.": ""}
VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_metadata(path):
    """Read local safetensors headers; tensor loading remains HF's responsibility."""
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("checkpoint must be an existing local directory")
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "gemma3" or "text_config" not in config:
        raise ValueError("this record companion requires the official Gemma 3 wrapper config")
    tensors, hashes = {}, {"config.json": sha256(root / "config.json")}
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError("checkpoint has no local safetensors shards")
    for shard in shards:
        with shard.open("rb") as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise ValueError("truncated safetensors header")
            header = json.loads(stream.read(int.from_bytes(size_bytes, "little")))
        hashes[shard.name] = sha256(shard)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"duplicate checkpoint tensor: {name}")
            if not name.startswith(("language_model.", *VISION_PREFIXES)):
                raise ValueError(f"unrecognised checkpoint tensor: {name}")
            if entry["dtype"] != "BF16":
                raise ValueError(f"checkpoint is not uniformly stored bf16: {name}")
            tensors[name] = entry
    text = {
        key.removeprefix("language_model."): value
        for key, value in tensors.items()
        if key.startswith("language_model.")
    }
    if not text:
        raise ValueError("checkpoint has no text tensors")
    return {
        "path": str(root),
        "config": config,
        "tensors": tensors,
        "text": text,
        "sha256": hashes,
        "text_bf16_bytes": sum(math.prod(value["shape"]) * 2 for value in text.values()),
    }


def load_text_model(metadata):
    """Use HF's native loading/conversion, then fail closed on every text loading gap."""
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    config = Gemma3TextConfig(**metadata["config"]["text_config"])
    model, info = Gemma3ForCausalLM.from_pretrained(
        metadata["path"],
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
        key_mapping=KEY_MAPPING,
    )
    for field in ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors"):
        if info.get(field):
            raise ValueError(f"text checkpoint load failed: {field}={info[field]}")
    expected_vision = {key for key in metadata["tensors"] if key.startswith(VISION_PREFIXES)}
    if set(info.get("unexpected_keys", ())) != expected_vision:
        raise ValueError("unexpected keys differ from the header's exact vision/projector set")
    state = model.state_dict()
    for key, entry in metadata["text"].items():
        if key not in state or tuple(state[key].shape) != tuple(entry["shape"]):
            raise ValueError(f"text tensor missing or shape changed: {key}")
    tied = model.all_tied_weights_keys
    uncovered = set(state) - set(metadata["text"])
    if any(key not in tied or tied[key] not in metadata["text"] for key in uncovered):
        raise ValueError(f"model contains unaccounted text tensors: {sorted(uncovered)}")
    for target, source in tied.items():
        if state[target].data_ptr() != state[source].data_ptr():
            raise ValueError(f"native weight tying was not restored: {target}")
    if any(
        tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16
        for tensor in model.parameters()
    ):
        raise ValueError("loaded parameters are not exclusively CPU bf16")
    model.eval().requires_grad_(False)
    serial_info = {
        key: sorted(value) if isinstance(value, set) else value for key, value in info.items()
    }
    return model, {
        "key_mapping": KEY_MAPPING,
        "loading_info": serial_info,
        "checkpoint_text_tensors": len(metadata["text"]),
        "model_state_tensors": len(state),
        "tied_keys": tied,
    }


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
            {"sliding_attention": "sliding", "full_attention": "global"}[kind]
            for kind in text_config.layer_types
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


def compare_retained(view, ids):
    """The same two view paths, with all requested tensors simultaneously retained."""
    import torch

    layers = range(view.num_layers + 1)
    loop, native = view.residuals(ids, layers), view.native_residuals(ids, layers)
    rows = {}
    for layer in loop:
        actual, reference = loop[layer].float(), native[layer].float()
        difference = float((actual - reference).abs().max())
        scale = float(reference.abs().max())
        rows[layer] = {
            "max_abs": difference,
            "native_max_abs": scale,
            "max_norm_relative": difference / scale if scale else None,
        }
    if any(not math.isfinite(row["max_abs"]) for row in rows.values()):
        raise ValueError("non-finite residual comparison")
    retained = sum(t.numel() * t.element_size() for t in (*loop.values(), *native.values()))
    assert not torch.is_grad_enabled()
    return {
        "by_layer": rows,
        "retained_residual_bytes": retained,
        "relative_definition": "max(abs(loop-native)) / max(abs(native)); descriptive",
        "memory_while_retained": memory_reading(),
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
    """Persist each completed phase before checking its peak or beginning the next phase."""
    measurement.update(
        tokens=len(ids),
        normal={"status": "unexecuted"},
        rotary_rounding={"status": "unexecuted"},
        mask_dispatch_control={"status": "unexecuted"},
        controls_unexecuted=["hook-site off-by-one", "entry-transform omission"],
        interpretation="native bf16 versus promoted fp32 includes block rounding, not only RoPE",
    )

    def measure_phase(name, operation):
        measurement[name] = {"status": "running"}
        try:
            checkpoint(f"before_{name}")
            measurement[name] = {"status": "measured", **operation()}
            checkpoint(name)
        except Exception:
            if measurement[name]["status"] == "running":
                measurement[name]["status"] = "failed"
            raise

    measure_phase("normal", lambda: compare_retained(view, ids))
    measure_phase("rotary_rounding", lambda: rotary_rounding(view, ids))
    original = view._observe_forward

    def global_mask_only(tokens, cache=None):
        entry, bundles = original(tokens, cache)
        global_index = next(i for i in bundles if view.attention_span(i) == "global")
        mask = bundles[global_index]["attention_mask"]
        return entry, {i: {**values, "attention_mask": mask} for i, values in bundles.items()}

    with patch.object(view, "_observe_forward", global_mask_only):
        measure_phase("mask_dispatch_control", lambda: compare_retained(view, ids))
    return measurement


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
        from transformers import Gemma3TextConfig

        from local_llm_lab import device
        from local_llm_lab.arch_torch import TorchArchitectureView
        from research.acceptance.torch_seam import structural_report

        if Version(torch.__version__.split("+")[0]) < Version("2.14.0"):
            raise RuntimeError("WS-A CPU evidence requires torch >= 2.14.0")
        torch.set_num_threads(args.threads)
        torch.set_default_device("cpu")
        report["runtime"] = device.pin(seed=0, attention="eager")
        report["resolved_arch_source"] = str(sys.modules[TorchArchitectureView.__module__].__file__)
        metadata = checkpoint_metadata(args.checkpoint)
        if metadata["text_bf16_bytes"] >= args.cap_gib * GIB:
            raise RuntimeError("text weights alone exceed the explicit memory cap")
        ids, report["tokens"] = read_ids(args.token_ids)
        report["checkpoint"] = {key: metadata[key] for key in ("path", "sha256", "text_bf16_bytes")}
        checkpoint("before_load")
        with torch.no_grad():
            model, report["loading"] = load_text_model(metadata)
            checkpoint("after_load")
            view = TorchArchitectureView.from_model(model)
            if max(ids[: max(LENGTHS)]) >= view.vocab_size:
                raise ValueError("supplied token IDs exceed the loaded model vocabulary")
            report["gates"]["1"] = {"status": "running"}
            checked_structure = verify_structure(
                structural_report(view), Gemma3TextConfig(**metadata["config"]["text_config"])
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
            report["gates"]["2"] = {
                "status": "measured_not_accepted",
                "reason": "precision comparison and extra negative controls require resolution",
                "registered_bf16_relative_bound": 1e-3,
                "threshold_changed": False,
            }
            report["gates"]["3"]["reason"] = "ordered after gate 2 acceptance"
            report["gates"]["4"]["reason"] = "ordered after gate 2 acceptance"
            report["status"] = "calibrated; acceptance incomplete"
            checkpoint("complete")
    except Exception as error:
        report["status"] = "stopped"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        if report["gates"]["1"]["status"] == "running":
            report["gates"]["1"]["status"] = "fail"
        if report["gates"]["2"]["status"] == "calibration_running":
            report["gates"]["2"]["status"] = "incomplete"
        raise
    finally:
        finish_report(
            args.output, report, started, preserving_exception=sys.exc_info()[0] is not None
        )
    return 1  # Full acceptance remains unexecuted even when calibration completed successfully.


if __name__ == "__main__":
    raise SystemExit(main())
