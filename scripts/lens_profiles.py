"""Pure §3.5 profiles from validated native evidence; see profiles.build_profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from local_llm_lab.pipeline.lens_fitting.profiles import build_profiles  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON with named sets and real instrument/identity evidence paths",
    )
    parser.add_argument("--out", type=Path, required=True, help="new exclusive directory")
    args = parser.parse_args()
    print(json.dumps(build_profiles(args.config, args.out), allow_nan=False))


if __name__ == "__main__":
    main()
