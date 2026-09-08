"""Does the loop survive an environment that stops lying? The Chief's causal story, tested.

The register's account of `update-0028` is a chain: `list_files("/")` answered an unsatisfiable
directory with `FILES: (none)`, the model concluded the workspace was empty, and every later turn
followed soundly from that false premise. `020aa89` makes `list_files` raise instead.

So the account makes a prediction that can fail. **If it is right, the belief never forms and the
episode behaves differently. If the model loops anyway, the account is wrong** and the register
needs revising before anything is built on it. That is a real risk to the claim, which is why it is
worth the box.

One episode, no lens and no capture: this asks about the trajectory, and the readings the fixed
point rests on are already on disk. Those records are frozen evidence and this script does not
touch them — it writes its own directory and refuses to overwrite.

    python environment_falsification.py --out <dir> --i-am-a-record
"""
from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and "--i-am-a-record" not in _sys.argv:
    _sys.exit(
        "refusing to run: this file is the record of the environment fix's falsification test on "
        "2026-09-08, not a launcher. It loads a 4B checkpoint and takes the model-run lock. "
        "Re-run it deliberately with --i-am-a-record, inside an announced box window."
    )

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

_sys.path.insert(0, "/Users/daniel.tipton/Desktop/An app/src")

SEED = 20260902
TASK = "test-update-0028-clean"
DIFFICULTY = 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gemma3-4b")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--i-am-a-record", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "falsification.json"
    if destination.exists():
        raise SystemExit(f"{destination} exists; this script never overwrites a result")

    import mlx.core as mx

    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.evaluate import load_policy, make_sampler
    from local_llm_lab.pipeline.runner import run_task
    from local_llm_lab.pipeline.tasks import make_tasks

    spec = load_model_spec(args.model)
    # `none`, matching the runs this is compared against, so the trajectory is produced the same
    # way. Nothing here reads a residual, but a different cache strategy is a different run.
    spec = replace(spec, cache_strategy="none")
    task = {t.task_id: t for t in make_tasks("test", 180, SEED, difficulty=DIFFICULTY)}[TASK]

    mx.set_cache_limit(2 * 2**30)
    model, tokenizer, view, resolved = load_policy(spec, None)
    model.eval()
    started = time.monotonic()
    trajectory = run_task(
        model,
        tokenizer,
        task,
        sampler=make_sampler(0.0),
        spec=spec,
        view=view,
        resolved=resolved,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        keep_last=2,
    )
    steps = [
        {
            "action": step.get("action"),
            "observation": str(step.get("observation", ""))[:400],
        }
        for step in trajectory.steps
    ]
    signatures = [json.dumps(step["action"], sort_keys=True) for step in steps]
    longest = 1 if signatures else 0
    run = 1
    for earlier, later in zip(signatures, signatures[1:]):
        run = run + 1 if earlier == later else 1
        longest = max(longest, run)

    payload = {
        "task_id": TASK,
        "model": spec.hf_id,
        "base": spec.base,
        "observation_role": spec.chat.observation_role,
        "cache_strategy": resolved.cache_strategy,
        "max_steps": args.max_steps,
        "seed": SEED,
        "seconds": round(time.monotonic() - started, 1),
        "turns": trajectory.turns,
        "generated_tokens": trajectory.generated_tokens,
        "success": bool(trajectory.verdict.get("success")),
        "loop_detected": trajectory.loop_detected,
        "exhausted": trajectory.exhausted,
        "distinct_calls": len(set(signatures)),
        "longest_identical_run": longest,
        "steps": steps,
    }
    with destination.open("x") as handle:
        json.dump(payload, handle, indent=1)
    print(json.dumps({k: v for k, v in payload.items() if k != "steps"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
