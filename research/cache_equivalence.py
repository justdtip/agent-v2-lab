"""Does KV-cache reuse change any output? It must not.

The selected cache strategy reuses model state across the turns of a task. That is a pure speed
optimisation, so every trajectory must be bit-identical to one produced without it. A stale or
misaligned cache would not raise; it would silently attend to the wrong keys and produce plausible
but different actions, which is exactly the kind of bug that survives a green test suite.

The history strategy requires two fixed-token checks: reuse versus same-schedule recomputation,
and the candidate versus the installed native generator's ordinary schedule. Both decision
logits and complete logical cache state must be bit-identical, with actual reuse exercised.

The legacy trim/snapshot modes compare greedy trajectories. Their elapsed times are descriptive;
different trajectories are different workloads and cannot establish a cache speedup.

Run with: uv run python research/cache_equivalence.py --model qwen35-4b --strategy snapshot
Fixed history: --model qwen35-4b --strategy history --fixed-history corpus.json --report result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

TASKS = 6
MAX_STEPS = 24
MAX_TOKENS = 200


def _array_bits(value):
    import mlx.core as mx
    import numpy as np

    return np.array(mx.contiguous(value).view(mx.uint8)).tobytes()


def compare_logits(candidate, reference):
    """Bit equality is the gate; errors, argmax and margins are descriptive only."""
    import mlx.core as mx
    import numpy as np

    same_shape = candidate.shape == reference.shape
    finite = bool(mx.all(mx.isfinite(candidate))) and bool(mx.all(mx.isfinite(reference)))
    a = np.array(candidate.astype(mx.float32))
    b = np.array(reference.astype(mx.float32))

    def margin(row):
        top = np.sort(row.reshape(-1, row.shape[-1])[-1])[-2:]
        return float(top[-1] - top[-2]) if len(top) == 2 and finite else None

    return {
        "bit_identical": bool(
            same_shape
            and finite
            and candidate.dtype == reference.dtype
            and _array_bits(candidate) == _array_bits(reference)
        ),
        "argmax_equal": bool(same_shape and finite and np.array_equal(a.argmax(-1), b.argmax(-1))),
        "max_abs_error": float(np.max(np.abs(a - b))) if same_shape and finite else None,
        "winning_margin": [margin(a), margin(b)],
        "finite": finite,
        "candidate_sha256": hashlib.sha256(_array_bits(candidate)).hexdigest(),
        "reference_sha256": hashlib.sha256(_array_bits(reference)).hexdigest(),
        "candidate_dtype": str(candidate.dtype),
        "reference_dtype": str(reference.dtype),
    }


def _cache_fingerprint(cache):
    """Logical state plus metadata, excluding unused KV allocation capacity."""
    import mlx.core as mx

    finite = True

    def encode(value):
        nonlocal finite
        if isinstance(value, mx.array):
            finite &= bool(mx.all(mx.isfinite(value)))
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(_array_bits(value)).hexdigest(),
            }
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in value.items()}
        if value is None or isinstance(value, (int, float, str, bool)):
            return value
        raise TypeError(f"unsupported native cache metadata: {type(value).__name__}")

    layers = []
    for entry in cache:
        layers.append(
            {
                "class": type(entry).__module__ + "." + type(entry).__name__,
                "state": encode(entry.state),
                "meta_state": encode(entry.meta_state),
                "offset": getattr(entry, "offset", None),
                "lengths": encode(getattr(entry, "lengths", None)),
                "left_padding": encode(getattr(entry, "left_padding", None)),
            }
        )
    return {"finite": finite, "layers": layers}


class _ForwardTrace:
    """Observe native calls, using the same ledger as production cache/capture."""

    def __init__(self, model, ledger, *, records_inputs=True):
        self.model, self.ledger = model, ledger
        self.records_inputs = records_inputs
        self.rows = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, ids, *, cache):
        import mlx.core as mx

        offset = self.ledger.offset
        logits = self.model(ids, cache=cache)
        inputs = ids[0].tolist()
        if self.records_inputs:
            self.ledger.record(offset, inputs)
        elif self.ledger.offset != offset + len(inputs):
            raise ValueError("production cache failed to account for a forward")
        # Only the last query is required at each boundary. Do not materialize the
        # whole prefill x vocabulary matrix merely to diagnose reuse.
        last = logits[:, -1, :]
        mx.eval(last)
        self.rows.append({"offset": offset, "input_ids": inputs, "logits": last})
        return logits


def _generate_fixed(trace, prompt, continuation, cache, *, step_size=None):
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    index = 0

    def force(_logprobs):
        nonlocal index
        # The library predicts one lookahead beyond the last yielded token.
        token = continuation[index] if index < len(continuation) else 0
        index += 1
        return mx.array([token])

    options = {} if step_size is None else {"prefill_step_size": step_size}
    emitted = []
    for token, _ in generate_step(
        mx.array(prompt),
        trace,
        prompt_cache=cache,
        sampler=force,
        max_tokens=len(continuation),
        **options,
    ):
        emitted.append(token)
        trace.ledger.emitted(token)
    if emitted != continuation:
        raise ValueError("fixed continuation was not consumed exactly")


def _compare_traces(candidate, reference, candidate_state, reference_state, prompt_length):
    by_position = {row["offset"] + len(row["input_ids"]): row for row in reference}
    first_difference = None
    first_greedy = None
    max_error = 0.0
    for row in candidate:
        end = row["offset"] + len(row["input_ids"])
        other = by_position.get(end)
        if other is None:
            first_difference = first_difference or {
                "position": end - 1,
                "reason": "missing forward",
            }
            continue
        comparison = compare_logits(row["logits"], other["logits"])
        max_error = max(max_error, comparison["max_abs_error"] or 0.0)
        if not comparison["bit_identical"] and first_difference is None:
            first_difference = {"position": end - 1, **comparison}
        if not comparison["argmax_equal"] and end >= prompt_length and first_greedy is None:
            first_greedy = end - prompt_length
    state_equal = bool(
        candidate_state["finite"]
        and reference_state["finite"]
        and candidate_state == reference_state
    )
    first_layer = next(
        (
            index
            for index, (a, b) in enumerate(
                zip(candidate_state["layers"], reference_state["layers"], strict=True)
            )
            if a != b
        ),
        None,
    )
    return {
        "status": "passed" if candidate and first_difference is None and state_equal else "failed",
        "positions_compared": len(candidate),
        "max_abs_error": max_error,
        "first_difference": first_difference,
        "first_differing_greedy_position": first_greedy,
        "state_bit_identical": state_equal,
        "first_differing_state_layer": first_layer,
    }


def check_fixed_history(model, view, cases, *, prefill_step_size=None):
    """Two independent gates on fixed IDs, with actual cross-turn reuse required.

    A: rebuild the candidate's recorded native partition from scratch.
    B: the installed generator's existing default schedule, always from the full prompt.
    No tolerance, greedy-only escape, checkpoint load, lock, or registry mutation occurs here.
    Tiny random models can exercise this function but cannot certify checkpoint equivalence.
    """
    import mlx.core as mx
    from mlx_lm.generate import generation_stream

    from local_llm_lab.forward import ForwardLedger
    from local_llm_lab.models import NATIVE_PREFILL_STEP_SIZE
    from local_llm_lab.pipeline.runner import HistoryCache

    if prefill_step_size is None:
        prefill_step_size = NATIVE_PREFILL_STEP_SIZE

    if not cases:
        raise ValueError("fixed history needs at least one turn")
    for case in cases:
        prompt = list(case["prompt_ids"])
        continuation = list(case["continuation_ids"])
        ForwardLedger(prompt + continuation)
        if not prompt or any(token >= view.vocab_size for token in prompt + continuation):
            raise ValueError("fixed token IDs outside model vocabulary")
    history = HistoryCache(model, view, prefill_step_size=prefill_step_size)
    turns = []
    with mx.stream(generation_stream):
        for index, case in enumerate(cases):
            prompt, continuation = list(case["prompt_ids"]), list(case["continuation_ids"])
            suffix = history.prepare(prompt)
            restored = history.ledger.offset
            candidate = _ForwardTrace(
                history.generation_model, history.ledger, records_inputs=False
            )
            _generate_fixed(
                candidate, suffix, continuation, history.cache, step_size=prefill_step_size
            )
            history.commit(prompt, continuation)
            candidate_state = _cache_fingerprint(history.cache)

            # Gate A replays exact recorded chunks through the native model, including
            # the saved prefix, independently of the candidate's restore implementation.
            fresh_cache = view.make_cache()
            same = _ForwardTrace(model, ForwardLedger(prompt))
            for row in history.ledger.passes:
                same(mx.array([row.input_ids]), cache=fresh_cache)
            a = _compare_traces(
                candidate.rows,
                same.rows,
                candidate_state,
                _cache_fingerprint(fresh_cache),
                len(prompt),
            )
            del same, fresh_cache

            # Gate B never adopts the candidate cadence. Let the installed generator
            # choose its default prefill partition, as the current runner does.
            fresh_cache = view.make_cache()
            legacy = _ForwardTrace(model, ForwardLedger(prompt))
            _generate_fixed(legacy, prompt, continuation, fresh_cache)
            candidate_decisions = [
                r for r in candidate.rows if r["offset"] + len(r["input_ids"]) >= len(prompt)
            ]
            legacy_decisions = [
                r for r in legacy.rows if r["offset"] + len(r["input_ids"]) >= len(prompt)
            ]
            b = _compare_traces(
                candidate_decisions,
                legacy_decisions,
                candidate_state,
                _cache_fingerprint(fresh_cache),
                len(prompt),
            )
            partitions = {
                "candidate": [
                    {"offset": row.offset, "input_ids": list(row.input_ids)}
                    for row in history.ledger.passes
                ],
                "legacy": [
                    {"offset": row.offset, "input_ids": list(row.input_ids)}
                    for row in legacy.ledger.passes
                ],
            }
            turns.append(
                {
                    "turn": index,
                    "reused_tokens": restored,
                    "gate_a": a,
                    "gate_b": b,
                    "prompt_tokens": len(prompt),
                    "continuation_tokens": len(continuation),
                    "forwarded_tokens": history.ledger.offset,
                    "partitions": partitions,
                    "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                    "continuation_sha256": hashlib.sha256(
                        json.dumps(continuation).encode()
                    ).hexdigest(),
                }
            )
            del legacy, fresh_cache, candidate, candidate_state
    a_status = (
        "failed"
        if any(t["gate_a"]["status"] == "failed" for t in turns)
        else ("passed" if history.reused_tokens else "inconclusive")
    )
    b_status = "passed" if all(t["gate_b"]["status"] == "passed" for t in turns) else "failed"
    return {
        "schema_version": 1,
        "gate_a": {"status": a_status},
        "gate_b": {"status": b_status},
        "accepted": a_status == b_status == "passed",
        "checkpoint_certified": False,
        "reused_tokens": history.reused_tokens,
        "encoded_tokens": history.encoded_tokens,
        "snapshot_bytes": history.snapshot_bytes,
        "turns": turns,
        "candidate_prefill_step_size": prefill_step_size,
        "reference": "installed mlx_lm.generate.generate_step default schedule; native model",
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen25-coder-3b", help="registered model name or HF id")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=("trim", "snapshot", "history"),
        help="explicit non-disabled cache strategy to attest",
    )
    parser.add_argument(
        "--fixed-history", type=Path, help="JSON episodes with fixed prompt/continuation IDs"
    )
    parser.add_argument("--report", type=Path, help="new JSON result file; never updates registry")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        help="candidate only (default: the registry's native cadence); reference stays native",
    )
    args = parser.parse_args(argv)
    if args.strategy == "history" and (args.fixed_history is None or args.report is None):
        parser.error("history requires --fixed-history and --report for both acceptance checks")
    if args.fixed_history is not None and args.strategy != "history":
        parser.error("fixed-history checks require the history strategy")
    if args.prefill_step_size is not None and args.prefill_step_size < 1:
        parser.error("prefill step size must be positive")
    return args


def _read_corpus(path):
    from local_llm_lab.forward import ForwardLedger

    source = path.read_bytes()
    corpus = json.loads(source)
    if corpus.get("schema_version") != 1 or not corpus.get("episodes"):
        raise ValueError("fixed history requires schema_version 1 and nonempty episodes")
    for episode in corpus["episodes"]:
        if not isinstance(episode.get("id"), str) or not episode.get("turns"):
            raise ValueError("each episode requires an ID and nonempty turns")
        for turn in episode["turns"]:
            prompt, continuation = turn["prompt_ids"], turn["continuation_ids"]
            if not isinstance(prompt, list) or not isinstance(continuation, list):
                raise ValueError("fixed token IDs must be lists")
            ForwardLedger(prompt)
            ForwardLedger(prompt + continuation)
    return corpus, hashlib.sha256(source).hexdigest()


def _fixed_main(args, model, view, resolved, corpus, corpus_hash):
    from importlib.metadata import version

    import mlx.core as mx
    import mlx_lm

    from local_llm_lab.provenance import _sha256, source_tree_hashes

    # Installed model implementations can be locally patched without changing package
    # versions. Freeze their source and the compiled MLX runtime before the comparisons.
    library = Path(mlx_lm.__file__).parent
    runtime_sources = {
        "mlx_lm/" + str(path.relative_to(library)): _sha256(path)
        for path in sorted(library.rglob("*.py"))
    }
    runtime = Path(mx.__file__).parent
    for path in sorted([*runtime.glob("*.so"), *runtime.glob("lib/*.dylib")]):
        runtime_sources["mlx/" + str(path.relative_to(runtime))] = _sha256(path)
    source_hashes = source_tree_hashes(Path(__file__).resolve().parents[1])
    adapter = (
        None
        if args.adapter is None
        else {
            "path": str(args.adapter.resolve()),
            "files": {
                str(path.relative_to(args.adapter)): _sha256(path)
                for path in sorted(args.adapter.rglob("*"))
                if path.is_file()
            },
        }
    )
    episodes = [
        {
            "id": episode["id"],
            **check_fixed_history(
                model, view, episode["turns"], prefill_step_size=args.prefill_step_size
            ),
        }
        for episode in corpus["episodes"]
    ]
    # A control episode may offer no matching prefix. It must still have zero numerical
    # mismatches; require positive reuse across the corpus to avoid a vacuous certificate.
    a_status = (
        "failed"
        if any(e["gate_a"]["status"] == "failed" for e in episodes)
        else ("passed" if any(e["reused_tokens"] > 0 for e in episodes) else "inconclusive")
    )
    b_status = "passed" if all(e["gate_b"]["status"] == "passed" for e in episodes) else "failed"
    report = {
        "schema_version": 1,
        "scope": "loaded checkpoint, fixed corpus only",
        "accepted": a_status == b_status == "passed",
        "gate_a": {"status": a_status},
        "gate_b": {"status": b_status},
        "registry_updated": False,
        "model": resolved.as_dict(),
        "corpus_path": str(args.fixed_history.resolve()),
        "corpus_sha256": corpus_hash,
        "episodes": episodes,
        "packages": {name: version(name) for name in ("mlx", "mlx-lm", "numpy")},
        "source_hashes": source_hashes,
        "runtime_sources": runtime_sources,
        "harness_sha256": _sha256(Path(__file__)),
        "adapter": adapter,
    }
    with args.report.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")
    print(
        json.dumps(
            {"accepted": report["accepted"], "report": str(args.report), "registry_updated": False}
        )
    )
    if not report["accepted"]:
        raise SystemExit("Both fixed-history acceptance checks must pass; keep reuse disabled")


def main(argv: Sequence[str] | None = None) -> None:
    import mlx.core as mx

    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.evaluate import load_policy, make_sampler
    from local_llm_lab.pipeline.runner import run_task
    from local_llm_lab.pipeline.tasks import make_tasks

    args = _parse_args(argv)
    corpus, corpus_hash = (
        (None, None) if args.fixed_history is None else _read_corpus(args.fixed_history)
    )
    if args.report is not None and args.report.exists():
        raise SystemExit("report already exists; choose a new path")
    spec = load_model_spec(args.model)
    spec = replace(spec, cache_strategy=args.strategy)
    model, tokenizer, view, resolved = load_policy(spec, args.adapter)
    if resolved.cache_strategy == "none":
        raise SystemExit("cache equivalence requires a non-disabled resolved strategy")
    print(
        f"ATTESTATION selected_model={args.model} resolved_model={spec.name} "
        f"hf_id={spec.hf_id} selected_strategy={args.strategy} "
        f"resolved_strategy={resolved.cache_strategy} reason={resolved.cache_strategy_reason}"
    )
    if corpus is not None:
        _fixed_main(args, model, view, resolved, corpus, corpus_hash)
        return
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
