"""Does KV-cache reuse change any output? It must not.

The selected cache strategy reuses model state across the turns of a task. That is a pure speed
optimisation, so every trajectory must be bit-identical to one produced without it. A stale or
misaligned cache would not raise; it would silently attend to the wrong keys and produce plausible
but different actions, which is exactly the kind of bug that survives a green test suite.

So this compares cached and uncached runs of the same tasks under greedy decoding and reports any
divergence, plus the measured speedup.

Run with: uv run python research/cache_equivalence.py
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import replace

TASKS = 6
MAX_STEPS = 24
MAX_TOKENS = 200


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen25-coder-3b", help="registered model name or HF id")
    parser.add_argument(
        "--strategy",
        choices=("auto", "trim", "snapshot", "none"),
        help="override the model registry's cache strategy",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    import mlx.core as mx

    from local_llm_lab.arch import ArchitectureView
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.evaluate import load_policy, make_sampler
    from local_llm_lab.pipeline.runner import run_task
    from local_llm_lab.pipeline.tasks import make_tasks
    from local_llm_lab.project import PROJECT_ROOT

    args = _parse_args(argv)
    spec = load_model_spec(args.model)
    if args.strategy is not None:
        spec = replace(spec, cache_strategy=args.strategy)
    model, tokenizer = load_policy(
        spec.hf_id,
        PROJECT_ROOT / "outputs" / "agent-v2" / "best-adapter",
    )
    view = ArchitectureView.from_model(model)
    resolved = spec.resolve(model, tokenizer)
    # Cover the long, listing-heavy families where prompts grow most, plus a couple of short ones.
    families = ("ledger_reconcile", "batch_update", "aggregate_report", "read", "search", "update")
    tasks = []
    for family in families:
        tasks.append(next(t for t in make_tasks("test", 60) if t.family == family))
    tasks = tasks[:TASKS]

    mismatches = 0
    cached_time = uncached_time = 0.0
    for task in tasks:
        runs = {}
        for use_cache in (False, True):
            mx.random.seed(20260902)
            started = time.perf_counter()
            runs[use_cache] = run_task(
                model,
                tokenizer,
                task,
                sampler=make_sampler(0.0),
                label="cache" if use_cache else "nocache",
                max_steps=MAX_STEPS,
                max_tokens=MAX_TOKENS,
                use_cache=use_cache,
                spec=spec,
                view=view,
                resolved=resolved,
            )
            elapsed = time.perf_counter() - started
            if use_cache:
                cached_time += elapsed
            else:
                uncached_time += elapsed

        plain, cached = runs[False], runs[True]
        same_actions = [s.get("action") for s in plain.steps] == [
            s.get("action") for s in cached.steps
        ]
        same_text = [s.get("raw") for s in plain.steps] == [s.get("raw") for s in cached.steps]
        same_verdict = plain.verdict == cached.verdict
        ok = same_actions and same_text and same_verdict
        mismatches += not ok
        mark = "IDENTICAL" if ok else "DIVERGED"
        print(
            f"{mark:10s} {task.task_id:42s} steps {len(plain.steps):2d}/{len(cached.steps):2d}  "
            f"success {plain.success}/{cached.success}"
        )
        if not ok:
            for i, (a, b) in enumerate(zip(plain.steps, cached.steps, strict=False)):
                if a.get("raw") != b.get("raw"):
                    print(f"   first divergence at step {i}:")
                    print(f"     uncached: {a.get('raw', '')[:160]!r}")
                    print(f"     cached:   {b.get('raw', '')[:160]!r}")
                    break

    print(
        f"\n{len(tasks) - mismatches}/{len(tasks)} trajectories identical; "
        f"uncached {uncached_time:.1f}s vs cached {cached_time:.1f}s "
        f"({100 * (1 - cached_time / uncached_time):.0f}% faster)"
    )
    if mismatches:
        raise SystemExit("KV cache changed model outputs; do not enable it")


if __name__ == "__main__":
    main()
