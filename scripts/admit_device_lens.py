"""Admit a device fit directory beside its immutable originals; file operations only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_llm_lab.probes.device_lens_admission import admit_device_lens  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_dir", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = admit_device_lens(
            args.fit_dir, checkpoint=args.checkpoint, model_base=args.model_base, output=args.output
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"admission refused: {error}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
