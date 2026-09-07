"""The runner's own cached and uncached code paths on the same prompt: first-token log-probabilities.

``scripts/cache_split_diagnostic.py`` models the schedules with the architecture view and finds only
rounding-level differences and exact restoration. This script runs the library's generation path as
the runner does: (a) ``stream_generate`` on the prompt string with no cache, (b) the same on the
prompt's token array, (c) the snapshot path exactly as ``generate_turn_with_count`` builds it —
``SnapshotCache.prepare`` on the encoded prompt, then ``stream_generate`` on the suffix with
``prompt_cache`` — and (d) path (a) a second time in the same process. It records the first
token's full log-probability vector under each and compares them: max |Δ|, argmax, margin.

    .venv/bin/python scripts/cache_runner_paths.py --out <dir> [--limit 8]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from local_llm_lab.models import load_model_spec  # noqa: E402
from local_llm_lab.pipeline.protocol import SYSTEM_PROMPT, build_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-eval", type=Path, default=REPO / "outputs/agent-v2/evals/base-test.json")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=12, help="generated tokens to compare per path")
    ap.add_argument("--model", default="qwen35-4b")
    args = ap.parse_args(); t0 = time.time(); args.out.mkdir(parents=True, exist_ok=True)
    import mlx.core as mx
    from mlx_lm import stream_generate
    from local_llm_lab.pipeline.evaluate import load_policy, make_sampler
    from local_llm_lab.pipeline.runner import SnapshotCache

    spec = load_model_spec(args.model)
    model, tok, view, resolved = load_policy(spec, None); model.eval()
    sampler = make_sampler(0.0)
    base = json.load(open(args.base_eval))["trajectories"][: args.limit]

    def run(prompt_input, prompt_cache=None):
        rows = []
        kwargs = {} if prompt_cache is None else {"prompt_cache": prompt_cache}
        for r in stream_generate(model, tok, prompt=prompt_input, max_tokens=args.tokens, sampler=sampler, **kwargs):
            lp = np.array(r.logprobs.astype(mx.float32)) if r.logprobs is not None else None
            rows.append({"token": int(r.token), "logprobs": lp})
        return rows

    def compare(a, b):
        first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x["token"] != y["token"]), None)
        la, lb = a[0]["logprobs"], b[0]["logprobs"]
        out = {"first_differing_generated_position": first_diff, "tokens_compared": min(len(a), len(b))}
        if la is not None and lb is not None:
            def margin(l):
                t = np.partition(l, -2)[-2:]; return float(t[1] - t[0])
            out.update(first_token={"max_abs_logprob_diff": round(float(np.abs(la - lb).max()), 6), "argmax": [int(la.argmax()), int(lb.argmax())],
                                    "margin": [round(margin(la), 5), round(margin(lb), 5)]})
        return out

    results = {"parameters": {"model": spec.hf_id, "tokens": args.tokens, "cases": len(base)}, "cases": []}
    for t in base:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t["prompt"]}]
        prompt = build_prompt(tok, messages, spec=spec, keep_last=2, generation=True)
        prefix_tokens = len(tok.encode(build_prompt(tok, messages[:2], spec=spec, keep_last=2, generation=False)))
        ids = list(tok.encode(prompt))
        a = run(prompt)
        b = run(mx.array(ids))
        snap = SnapshotCache(model, view, prefix_tokens)
        suffix = snap.prepare(ids)
        c = run(mx.array(suffix), snap.cache)
        d = run(prompt)
        case = {"task_id": t["task_id"], "prompt_tokens": len(ids), "prefix_tokens": prefix_tokens,
                "a_string_vs_b_array": compare(a, b), "a_string_vs_c_snapshot": compare(a, c), "b_array_vs_c_snapshot": compare(b, c),
                "a_vs_a_repeat": compare(a, d),
                "tokens": {"a": [x["token"] for x in a], "c": [x["token"] for x in c]},
                "texts": {"a": tok.decode([x["token"] for x in a]), "c": tok.decode([x["token"] for x in c])}}
        results["cases"].append(case)
        print(json.dumps({k: v for k, v in case.items() if k != "tokens"}), flush=True)
    results["parameters"]["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(results, open(args.out / "cache_runner_paths.json", "w"), indent=1)
    print(json.dumps({"event": "done", "elapsed_s": results["parameters"]["elapsed_s"]}))


if __name__ == "__main__":
    main()
