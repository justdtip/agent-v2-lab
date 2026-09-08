"""Model-free A1 bridge. Inspect by default; --execute requires an owned R61 window.

Run benchmark before readout. Inputs, tolerances, feature/control and peak basis
belong in the committed registration, not command-line overrides after seeing data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require_window():
    from local_llm_lab import runlock
    from local_llm_lab.pipeline.lens_fitting.runtime import primary_worktree

    path = primary_worktree() / "." / "outputs/.box-window.json"
    if runlock.read_window(path) is None or runlock.blocking_window(path) is not None:
        raise ValueError("heavy CPU readout requires this process's announced R61 window")


def load_inputs(registration):
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.live_lens.instruments import LensIdentity, LensMaps
    from local_llm_lab.pipeline.sae_bridge.assets import (
        checked_hash,
        decoder_from_files,
        read_bf16_matrix,
        validate_reference_config,
    )

    reg = registration
    spec = load_model_spec(reg["reference_model"])
    identity = LensIdentity.from_dict(reg["model_identity"])
    if (
        identity.base != spec.base
        or identity.training != spec.training
        or spec.training is not None
    ):
        raise ValueError("A1 requires the declared untrained canonical reference checkpoint")
    root = Path(spec.hf_id)
    checked_hash(root / "config.json", reg["reference_config_sha256"])
    config = json.loads((root / "config.json").read_bytes())
    depth, dimension, vocab = validate_reference_config(config)
    if depth != identity.num_layers or dimension != reg["hidden_size"]:
        raise ValueError("reference depth/width differs from registration")
    index_path = root / reg["readout"]["index"]
    checked_hash(index_path, reg["readout"]["index_sha256"])
    weight_map = json.loads(index_path.read_bytes())["weight_map"]
    key = reg["readout"]["key"]
    if config.get("model_type") != "gemma3" or key != "language_model.model.embed_tokens.weight":
        raise ValueError("unverified tied-unembedding convention")
    # mlx_lm's Gemma sanitize ties embeddings exactly when lm_head.weight is absent.
    if any(name.endswith("lm_head.weight") for name in weight_map):
        raise ValueError("checkpoint has an untied lm_head; raw embedding is not its readout")
    if weight_map.get(key) != reg["readout"]["shard"]:
        raise ValueError("readout shard differs from the checkpoint index")
    lens = reg["lens"]
    if lens["fitting_precision"] != "bf16":
        raise ValueError("A1 reference lens must be fitted against BF16 weights")
    maps = LensMaps.load(
        Path(lens["path"]),
        expected_sha256=lens["sha256"],
        hidden_size=dimension,
        num_layers=depth,
        identity=identity,
    )
    layer, wrong = reg["probe_layer"], reg["wrong_probe_layer"]
    if layer == wrong or layer not in maps.maps or wrong not in maps.maps:
        raise ValueError("acceptance needs two distinct nonfinal maps")
    j, wrong_j = maps.maps[layer], maps.maps[wrong]
    del maps  # Do not retain the other 31 maps while loading the readout.
    dictionary = reg["dictionary"]
    if dictionary["suite"] != "resid_post_all":
        raise ValueError("this order requires the every-layer residual dictionary suite")
    d = decoder_from_files(
        Path(dictionary["config"]),
        Path(dictionary["params"]),
        expected_config_sha256=dictionary["config_sha256"],
        expected_params_sha256=dictionary["params_sha256"],
        base=identity.base,
        probe_layer=layer,
        hidden_size=dimension,
        width=dictionary["width"],
        l0=dictionary["l0"],
        coordinate_convention=dictionary["coordinate_convention"],
    )
    w = read_bf16_matrix(
        root / reg["readout"]["shard"], key, expected_sha256=reg["readout"]["shard_sha256"]
    )
    if w.shape != (vocab, dimension):
        raise ValueError("readout shape differs from declared model geometry")
    checked_hash(root / "tokenizer.json", reg["tokenizer_sha256"])
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    return w, j, wrong_j, d, tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("benchmark", "readout"), default="benchmark")
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    from local_llm_lab.pipeline.live_lens.instruments import file_sha256
    from local_llm_lab.pipeline.sae_bridge.core import iter_feature_topk
    from local_llm_lab.pipeline.sae_bridge.readout import acceptance, write_readout

    registration_bytes = args.registration.read_bytes()
    reg = json.loads(registration_bytes)
    registration_sha = hashlib.sha256(registration_bytes).hexdigest()
    peak = reg.get("projected_peak_gib", 0)
    if not math.isfinite(peak) or peak <= 0 or not reg.get("peak_basis"):
        parser.error("registration must declare a projected peak and its basis")
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "inspection_only",
                    "registration_sha256": registration_sha,
                    "registration": reg,
                },
                indent=2,
            )
        )
        return 0
    require_window()  # Before any weight/matrix read, including the CPU-only acceptance.
    if args.output is None or args.output.exists():
        parser.error("execution needs a fresh --output directory")
    benchmark = None
    if args.mode == "readout":
        if args.benchmark is None:
            parser.error("readout requires an accepted --benchmark")
        benchmark = json.loads(args.benchmark.read_bytes())
        if (
            benchmark.get("registration_sha256") != registration_sha
            or benchmark.get("status") != "accepted"
            or benchmark.get("mode") != "benchmark"
        ):
            parser.error("benchmark is not accepted for this exact registration")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    w, j, wrong_j, d, tokenizer = load_inputs(reg)
    loaded_s = time.monotonic() - started
    evidence, jd = acceptance(
        w, j, wrong_j, d, feature_id=reg["feature_id"], k=reg["k"], **reg["tolerances"]
    )
    evidence.update(registration_sha256=registration_sha, instrument=reg)
    write_json(args.output / "acceptance.json", evidence)
    if not evidence["passed"]:
        print("A1 acceptance unmet; bounds and negative control are unchanged", flush=True)
        return 2
    before_scores = time.monotonic()
    if args.mode == "benchmark":
        count = min(16, d.shape[1])
        for _row in iter_feature_topk(w, jd, reg["k"], feature_ids=list(range(count))):
            pass
        seconds = time.monotonic() - before_scores
        projected_seconds = seconds / count * d.shape[1]
    else:

        def progress(row):
            print(json.dumps(row), flush=True)

        write_readout(
            args.output / "features.jsonl",
            w,
            jd,
            k=reg["k"],
            metadata={
                **reg,
                "registration_sha256": registration_sha,
                "benchmark_sha256": file_sha256(args.benchmark),
            },
            token_piece=tokenizer.id_to_token,
            progress=progress,
        )
        count = d.shape[1]
        seconds = time.monotonic() - before_scores
        projected_seconds = benchmark["projected_readout_seconds"]
    # This tool targets the local macOS CPU; ru_maxrss there is bytes.
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    result = {
        "status": "accepted" if peak_gib <= reg["projected_peak_gib"] else "peak_bound_exceeded",
        "mode": args.mode,
        "registration_sha256": registration_sha,
        "instrument": reg,
        "loaded_seconds": loaded_s,
        "elapsed_seconds": time.monotonic() - started,
        "measured_features": count,
        "feature_seconds": seconds,
        "projected_readout_seconds": projected_seconds,
        "peak_rss_gib": peak_gib,
        "projected_peak_gib": reg["projected_peak_gib"],
        "model_forward_count": 0,
    }
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
