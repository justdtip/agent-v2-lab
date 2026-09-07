"""Replay recorded episodes through a new lens; exact native forward identity required."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--lens-sha256", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--layers", choices=("all", "source"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identity-atlas", type=Path)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args(argv)
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.lens_fitting.replay import prepare_replay, run_replay
    from local_llm_lab.pipeline.lens_fitting.runtime import (
        configure_allocator_cache,
        load_runtime,
        resource_snapshot,
    )

    try:
        prepared = prepare_replay(
            args.records,
            load_model_spec(args.model),
            args.out,
            args.lens,
            lens_sha256=args.lens_sha256,
            domain=args.domain,
            kind=args.kind,
            layers=args.layers,
            identity_atlas=args.identity_atlas,
            revision=args.revision,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    started = time.monotonic()
    loaded = load_runtime(prepared, capture=True)
    loaded.model.eval()
    allocator_cache = configure_allocator_cache()

    def progress(event):
        print(
            json.dumps(event | {"elapsed_s": time.monotonic() - started} | resource_snapshot()),
            flush=True,
        )

    try:
        result = run_replay(prepared, loaded, progress=progress, allocator_cache=allocator_cache)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"event": "stop_required", "reason": str(error)}), flush=True)
        return 2
    print(json.dumps({"event": "done", **result}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
