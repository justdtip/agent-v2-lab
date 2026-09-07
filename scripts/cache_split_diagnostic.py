"""Where does snapshot-cache non-equivalence come from? Isolate the execution schedule.

Astra's reading (2026-09-07, source only): the snapshot strategy processes the system/task prefix
in one model call and hands the remaining prompt to the library's prefill loop, so the forward
passes have different shapes from the ordinary path; the divergence seen at step 0 in
``research/cache_equivalence.py`` precedes any restoration. This script runs the isolation table
on fixed token ids, with the continuation teacher-forced so an early different choice does not
change later inputs:

  1. ordinary prefill vs the split schedule, with no state copied or restored   (schedule effect)
  2. the split schedule twice, the second time restoring copied state         (snapshot correctness)
  3. restore again after the live cache has advanced through the suffix       (aliasing / mutation)
  4. a control split at a different boundary (half the prompt)                (any split, or this one)

For every comparison: max |Δlogit| at the last prompt position, whether the greedy token agrees,
the winning margin (top-1 minus top-2) under each schedule, the first continuation position whose
greedy token differs, and the per-layer residual difference at the last prompt position.

    .venv/bin/python scripts/cache_split_diagnostic.py --out <dir> [--tasks id ...] [--limit N]
"""
from __future__ import annotations

import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from local_llm_lab.models import load_model_spec  # noqa: E402
from local_llm_lab.pipeline.protocol import SYSTEM_PROMPT, build_prompt  # noqa: E402

PREFILL_STEP = 2048  # mlx_lm generate_step default prefill_step_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, default=REPO / "research/records/ARM-A-DIVERGENCE-2026-09-07/pairs_dry_run.json")
    ap.add_argument("--transcripts", type=Path, default=REPO / "outputs/agent-v2e-qwen35-4b/transcripts/best-adapter-test/transcripts.jsonl")
    ap.add_argument("--base-eval", type=Path, default=REPO / "outputs/agent-v2/evals/base-test.json")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--model", default="qwen35-4b")
    args = ap.parse_args(); t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    from local_llm_lab.pipeline.evaluate import load_policy

    spec = load_model_spec(args.model)
    model, tok, view, resolved = load_policy(spec, None)
    model.eval()

    # prompts: the base evaluation's step-0 prompts (system + task) for the first tasks, with the base's recorded
    # first turn as the teacher-forced continuation; these are the runner's own prompts at the point where
    # research/cache_equivalence.py saw step-0 divergence.
    base = json.load(open(args.base_eval))["trajectories"]
    cases = []
    for t in base:
        if args.tasks and t["task_id"] not in args.tasks:
            continue
        steps = [s for s in t["steps"] if "action" in s]
        if not steps:
            continue
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t["prompt"]}]
        prompt = build_prompt(tok, messages, spec=spec, keep_last=2, generation=True)
        prefix = build_prompt(tok, messages[:2], spec=spec, keep_last=2, generation=False)
        cases.append({"task_id": t["task_id"], "prompt": prompt, "prefix_tokens": len(tok.encode(prefix)), "continuation": steps[0]["raw"]})
        if len(cases) >= args.limit:
            break

    def forward_schedule(ids, boundaries):
        """Run ``ids`` through the model with a fresh cache, one call per segment; return the logits at the last
        position of the last segment and the per-layer residual there. ``boundaries`` are segment ends."""
        cache = view.make_cache()
        residual_last = {}
        start = 0
        logits = None
        for end in boundaries:
            seg = mx.array(ids[start:end])[None]
            h = view.embed(seg)
            masks = view.masks(h, cache)
            for i in range(view.num_layers):
                h = view.run_block(i, h, masks, cache[i])
                if end == boundaries[-1] and (i + 1) in READ:
                    residual_last[i + 1] = np.array(h[0, -1])
            logits = np.array(view.unembed(view.final_norm(h))[0, -1])
            mx.eval(*[c.state for c in cache if hasattr(c, "state")])
            start = end
        return logits, residual_last, cache

    def native_boundaries(n):
        # generate_step: prompt[:-1] in prefill steps, then the last token alone
        b = list(range(PREFILL_STEP, n - 1, PREFILL_STEP)) + [n - 1, n]
        return sorted(set(x for x in b if 0 < x <= n))

    def split_boundaries(n, cut):
        # snapshot: the prefix in one call, then the remainder as generate_step would run it
        rest = n - cut
        b = [cut] + [cut + x for x in range(PREFILL_STEP, rest - 1, PREFILL_STEP)] + [n - 1, n]
        return sorted(set(x for x in b if 0 < x <= n))

    def margin(lg):
        top2 = np.partition(lg, -2)[-2:]
        return float(top2[1] - top2[0])

    def compare_logits(a, b):
        return {"max_abs": round(float(np.abs(a - b).max()), 5), "argmax_same": bool(a.argmax() == b.argmax()),
                "top1": [int(a.argmax()), int(b.argmax())], "margin": [round(margin(a), 4), round(margin(b), 4)]}

    def continuation_agreement(ids_prompt, cont_ids, boundaries_prompt):
        """Teacher-force the continuation token by token after each schedule; first position where the greedy tokens differ."""
        out = {}
        for name, bounds in boundaries_prompt.items():
            lg, _, cache = forward_schedule(ids_prompt, bounds)
            preds = [int(lg.argmax())]
            for tid in cont_ids:
                h = view.embed(mx.array([tid])[None]); masks = view.masks(h, cache)
                for i in range(view.num_layers):
                    h = view.run_block(i, h, masks, cache[i])
                preds.append(int(np.array(view.unembed(view.final_norm(h))[0, -1]).argmax()))
            out[name] = preds
        return out

    READ = (12, 16, 20, 24, 28, 32)
    results = {"parameters": {"model": spec.hf_id, "prefill_step": PREFILL_STEP, "read_layers": READ, "cases": len(cases)}, "cases": []}
    for c in cases:
        ids = list(tok.encode(c["prompt"])); n = len(ids); cut = c["prefix_tokens"]
        cont = list(tok.encode(c["continuation"], add_special_tokens=False))[:40]
        bN, bS, bH = native_boundaries(n), split_boundaries(n, cut), split_boundaries(n, n // 2)
        lgN, resN, _ = forward_schedule(ids, bN)
        lgS, resS, cacheS = forward_schedule(ids, bS)
        lgH, resH, _ = forward_schedule(ids, bH)
        # 2. split schedule with restore: copy the prefix state, advance, restore, re-run the suffix
        cache2 = view.make_cache()
        h = view.embed(mx.array(ids[:cut])[None]); masks = view.masks(h, cache2)
        for i in range(view.num_layers):
            h = view.run_block(i, h, masks, cache2[i])
        mx.eval(*[e.state for e in cache2])
        saved = [copy.deepcopy(e.state) for e in cache2]
        def run_rest(cache):
            start = cut; lg = None
            for end in bS[1:]:
                h = view.embed(mx.array(ids[start:end])[None]); masks = view.masks(h, cache)
                for i in range(view.num_layers):
                    h = view.run_block(i, h, masks, cache[i])
                lg = np.array(view.unembed(view.final_norm(h))[0, -1]); start = end
            return lg
        lg_first = run_rest(cache2)                       # live cache advanced through the suffix
        for e, st in zip(cache2, saved, strict=True):
            e.state = copy.deepcopy(st)
        lg_restored = run_rest(cache2)                    # 3. restore after advance, run again
        for e, st in zip(cache2, saved, strict=True):
            e.state = copy.deepcopy(st)
        lg_restored2 = run_rest(cache2)
        per_layer = {str(L): {"max_abs": round(float(np.abs(resN[L] - resS[L]).max()), 5), "rel_to_norm": round(float(np.abs(resN[L] - resS[L]).max() / (np.linalg.norm(resN[L]) + 1e-9)), 7)} for L in READ if L in resN and L in resS}
        agree = continuation_agreement(ids, cont, {"native": bN, "split": bS})
        first_diff = next((i for i, (x, y) in enumerate(zip(agree["native"], agree["split"])) if x != y), None)
        results["cases"].append({
            "task_id": c["task_id"], "prompt_tokens": n, "prefix_tokens": cut,
            "boundaries": {"native": bN, "split": bS, "half": bH},
            "1_native_vs_split": compare_logits(lgN, lgS),
            "4_native_vs_half_split": compare_logits(lgN, lgH),
            "2_split_fresh_vs_split_restored": compare_logits(lg_first, lg_restored),
            "3_restore_after_advance_vs_restore_again": compare_logits(lg_restored, lg_restored2),
            "2b_split_schedule_reproducible": compare_logits(lgS, lg_first),
            "per_layer_residual_native_vs_split_at_last_position": per_layer,
            "continuation_first_differing_greedy_position": first_diff,
            "continuation_positions_compared": len(agree["native"]),
        })
        print(json.dumps({"task": c["task_id"], "n": n, "cut": cut, "native_vs_split": results["cases"][-1]["1_native_vs_split"], "half": results["cases"][-1]["4_native_vs_half_split"]["max_abs"], "restore": results["cases"][-1]["2_split_fresh_vs_split_restored"]["max_abs"], "first_diff": first_diff}), flush=True)
    results["parameters"]["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(results, open(args.out / "cache_split_diagnostic.json", "w"), indent=1)
    print(json.dumps({"event": "done", "elapsed_s": results["parameters"]["elapsed_s"]}))


if __name__ == "__main__":
    main()
