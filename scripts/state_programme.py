"""The state programme's run script (STATE-PROGRAMME-RUN-ORDER-2026-09-10).

    .venv/bin/python scripts/state_programme.py --out <dir> --decoding greedy [--fixture]

``--decoding sampled`` is accepted by the parser and refused at run time until SWE-1's sampler is on
the branch this script runs from. The temperature is ruled (2026-09-10): sampled decoding for the
state programme only, temperature 1.0 with no truncation, seeded and recorded in every manifest.
Greedy runs whole. A record made in one mode is not compared with one made in the other; the
manifest names the estimand per mode.

``--fixture`` drives the whole script with a scripted policy, the real environment, a fixture lens
identity and dictionary, and a stub wrapper, on a handful of episodes, writing every artefact §6
names. It is the laptop's only way to run this, and its record says ``fixture: true``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SAMPLED_UNBUILT = (
    "sampled decoding is ruled (2026-09-10: temperature 1.0, no truncation, seeded, for the state "
    "programme only) and its sampler is SWE-1's under the WS-B order's amendment; it is not on the "
    "branch this script runs from yet, so --decoding sampled is refused by name until it is. Run "
    "--decoding greedy; the record names its estimand as the context-averaged outcome frequency."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decoding", choices=("greedy", "sampled"), required=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--pilot-pairs", type=int, default=100)
    parser.add_argument("--main-pairs", type=int, default=None, help="override n (fixtures only)")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--fault-rate", type=float, default=0.2)
    parser.add_argument("--fault-seed", type=int, default=7)
    parser.add_argument("--hours-bought", type=float, default=None)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)

    if args.decoding == "sampled":
        print(SAMPLED_UNBUILT, file=sys.stderr)
        return 3
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty; a record directory is written once")
    if not args.fixture:
        parser.error("only --fixture runs on this box; the device driver is the runbook's")

    from local_llm_lab.pipeline.state_programme.run import ScriptedPolicy, run, scripted_wrapper

    policy = ScriptedPolicy()
    result = run(
        args.out, root=Path(__file__).resolve().parents[1], policy=policy,
        wrapper=lambda pairs: scripted_wrapper(policy, pairs), decoding=args.decoding,
        temperature=args.temperature, pilot_pairs=args.pilot_pairs, main_pairs=args.main_pairs,
        level=args.level, seed=args.seed, fault_rate=args.fault_rate, fault_seed=args.fault_seed,
        hours_bought=args.hours_bought, fixture=True,
        manifest_inputs={
            "registry_name": "fixture", "checkpoint_digest": "fixture:" + "0" * 16,
            "lens_identity": {"base": "fixture", "num_layers": 4},
            "dictionary_layers": {"1": "model.layers.0.output"}, "wrapper_version": "stub",
        },
    )
    print({"event": "done", **result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
