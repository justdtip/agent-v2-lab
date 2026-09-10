"""Merge compatible, disjoint float32 device lens chunks without loading a model.

The original Chief's script weighted by prompt count but omitted the promised nu.json.
This entry point retains OUT CHUNK_DIR ... and writes complete declarations and digest
bindings through the shared admission module. OUT must not already exist; the as-run
chunks and any previous merged artifact remain untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_llm_lab.probes.device_lens_admission import merge_device_lens_chunks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("chunks", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        result = merge_device_lens_chunks(args.output, args.chunks)
    except (ValueError, OSError) as error:
        parser.exit(2, f"merge refused: {error}\n")
    print(
        json.dumps(
            {
                "event": "merged",
                "n_prompts": result["n_prompts"],
                "layers": len(result["layers"]),
                "sha256": result["exact_maps_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
