"""Audit the completed record with model imports forbidden; no model execution."""

from __future__ import annotations

import hashlib
import importlib.abc
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT / "src"))


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx", "mlx_lm"}:
            raise RuntimeError("Model imports forbidden: " + fullname)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    sys.meta_path.insert(0, NoModelImports())
    from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus

    registered = json.loads((ROOT / "source-hashes.json").read_text())
    for relative, expected in registered.items():
        assert digest(PROJECT / relative) == expected, relative
    artifact = PROJECT / "models/jlens/gemma3-4b-bf16-prose-regression-native.npz"
    meta_path = artifact.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    comparison = json.loads((ROOT / "comparison.json").read_text())
    end = json.loads((ROOT / "run-end.json").read_text())
    assert end["status"] == "completed" and end["artifact_exists"]
    sha = digest(artifact)
    assert sha == meta["npz_sha256"] == comparison["regression_sha256"]
    assert digest(meta_path) == comparison["regression_sidecar_sha256"]
    assert meta["checkpoint"]["snapshot_sha256"] == (
        "107888c0f7d717d4357699e32077144ffce6d7665c5bd0a75dfd0d85e17f52ed"
    )
    assert meta["residual_source"] == meta["counts"]["residual_source"] == "native"
    assert meta["cache_strategy"] == "none"
    assert meta["layers"] == list(range(1, 34))
    assert meta["num_layers"] == 34 and meta["hidden_size"] == 2560
    assert meta["peak_memory_gib"] < 14 and end["elapsed_s"] < 65 * 60
    manifest = PROJECT / "data/lens-fitting/gemma3-4b-bf16-prose-128/manifest.json"
    assert digest(manifest) == meta["corpus_manifest_sha256"]
    rows = read_corpus(manifest)
    counts = {}
    for split in ("fit", "held"):
        selected = [row for row in rows if row["split"] == split]
        counts[split] = {
            "sequences": len(selected),
            "positions": sum(len(row["ids"]) for row in selected),
        }
        assert counts[split] == meta["counts"][split]
    assert counts == {
        "fit": {"sequences": 1608, "positions": 205824},
        "held": {"sequences": 402, "positions": 51456},
    }
    assert all(len(row["ids"]) == 128 for row in rows)
    for layer in range(1, 34):
        info = meta["per_layer"][str(layer)]
        assert [c["alpha"] for c in info["candidates"]] == meta["lambda_grid"]
        best = min(info["candidates"], key=lambda c: c["held_out_relative_error"])
        for key, value in best.items():
            assert info[key] == value and math.isfinite(value)
        assert info["fit_positions"] == counts["fit"]["positions"]
        assert info["held_positions"] == counts["held"]["positions"]
    assert [row["layer"] for row in comparison["rows"]] == list(range(1, 34))
    assert all(math.isfinite(value) for row in comparison["rows"] for value in row.values())
    result = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "scope": "integrity, provenance, counts and registered selection; not scientific validity",
        "model_imports": "forbidden",
        "registered_files_unchanged": len(registered),
        "artifact_sha256": sha,
        "artifact_bytes": artifact.stat().st_size,
        "sidecar_sha256": digest(meta_path),
        "counts": counts,
        "nonfinal_maps": 33,
        "comparison_loader_checks": "hash, identity, all layers, dimensions and finiteness",
        "grid_endpoint_selected_layers": sum(
            row["alpha"] == min(meta["lambda_grid"]) for row in meta["per_layer"].values()
        ),
        "peak_memory_gib": meta["peak_memory_gib"],
        "load_fit_save_seconds": end["elapsed_s"],
    }
    with (ROOT / "verification.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    with (ROOT / "lens-metadata.json").open("xb") as stream:
        stream.write(meta_path.read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
