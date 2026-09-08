"""The instrument proving itself at the one layer where the answer is known in advance.

The final layer's lens is the identity: at layer 34 the readout applies no transport at all, so
the foreknowledge read there must return the token the model actually emitted, at rank 1, always.
It is a boundary condition rather than a result, and that is exactly its value — readout
orientation, lens application and position bookkeeping cannot all be wrong and still produce
1.000 across five thousand reads. R52's sense of an instrument that can fail and did not.

No model, no box: this reads stage one's committed records and counts. Run it against any run's
record directory and it answers the same question about that run.

    python identity_layer_check.py <record-dir> [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics


def horizon_one_ranks(directory: str) -> dict[int, list[int]]:
    """Every horizon-1 rank in every agentic episode of a record directory, by layer.

    Horizon 1 is the next token, which is the only horizon at which the identity layer has a
    predetermined answer: further horizons ask about tokens the residual at this position has not
    committed to, and there the identity carries no guarantee.
    """
    by_layer: dict[int, list[int]] = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, "agentic-*.jsonl"))):
        with open(path) as handle:
            for line in handle:
                try:
                    event = json.loads(line)["event"]
                except (ValueError, KeyError):
                    continue
                if event.get("kind") == "rank" and event.get("horizon") == 1:
                    by_layer[event["layer"]].append(event["rank"])
    return dict(by_layer)


def summarise(by_layer: dict[int, list[int]]) -> dict:
    layers = {}
    for layer in sorted(by_layer):
        ranks = by_layer[layer]
        layers[str(layer)] = {
            "reads": len(ranks),
            "median_rank": statistics.median(ranks),
            "rank_1_fraction": sum(1 for rank in ranks if rank == 1) / len(ranks),
        }
    return layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--identity-layer", type=int, default=34)
    parser.add_argument("--json")
    args = parser.parse_args()

    by_layer = horizon_one_ranks(args.directory)
    if not by_layer:
        raise SystemExit(f"no agentic records with rank rows under {args.directory}")
    layers = summarise(by_layer)
    identity = layers.get(str(args.identity_layer))
    if identity is None:
        raise SystemExit(
            f"layer {args.identity_layer} was not read in this run; the check needs the final "
            f"layer, and this run covered {sorted(int(k) for k in layers)}"
        )

    payload = {
        "record_directory": os.path.abspath(args.directory),
        "episodes": len(glob.glob(os.path.join(args.directory, "agentic-*.jsonl"))),
        "identity_layer": args.identity_layer,
        "identity_layer_rank_1_fraction": identity["rank_1_fraction"],
        "passed": identity["rank_1_fraction"] == 1.0,
        "layers": layers,
    }
    print(json.dumps(payload, indent=1))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=1)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
