"""Does KV-cache reuse change any output? It must not.

`TurnCache` reuses the KV cache across the turns of a task by trimming it back to the longest
token prefix shared with the next prompt. That is a pure speed optimisation, so every trajectory
must be bit-identical to one produced without it. A stale or misaligned cache would not raise; it
would silently attend to the wrong keys and produce plausible but different actions, which is
exactly the kind of bug that survives a green test suite.

So this compares cached and uncached runs of the same tasks under greedy decoding and reports any
divergence, plus the measured speedup.

Run with: uv run python research/cache_equivalence.py
"""

from __future__ import annotations

import time

TASKS = 6
MAX_STEPS = 24
MAX_TOKENS = 200


def main() -> None:
    import mlx.core as mx

    from local_llm_lab.pipeline.evaluate import load_policy, make_sampler
    from local_llm_lab.pipeline.runner import run_task
    from local_llm_lab.pipeline.tasks import make_tasks
    from local_llm_lab.project import PROJECT_ROOT

    model, tokenizer = load_policy(
        "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        PROJECT_ROOT / "outputs" / "agent-v2" / "best-adapter",
    )
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
