"""Live-lens pilot: lens activity on the running (untrained) model over real episodes.

Items 1 and 2 of the Director's plan at pilot scale: agentic episodes from the test split at
three difficulties, and chat prompts of varying length, captured through the collaborator's
native capture path (``run_task(capture=...)``; cache strategy ``none``). Each episode writes
one hash-chained JSONL record: per-position lens top-k at the band's attention members and the
final layer, foreknowledge ranks of every emitted token at horizons 1/4/8 per layer, the native
forward hashes (exact replay), and the rendered prompt with its windowed messages per turn.
Head capture is off in the pilot (item 3 is a separate, targeted run).

    .venv/bin/python scripts/live_lens_pilot.py --out <dir> [--plan-only] [--max-steps 12]
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from local_llm_lab.models import load_model_spec  # noqa: E402
from local_llm_lab.pipeline.protocol import build_prompt  # noqa: E402
from local_llm_lab.pipeline.tasks import make_tasks  # noqa: E402

# The lens, its digest and the registry were module constants pinned to Qwen. That made a
# `--model` change move nothing, and it was also the only thing standing between this script and
# loading the Qwen lens onto another model of the same width (issue 99). Both are gone: the
# registry is derived from `--model` so it can never name a different model than the run does,
# and the lens must be named, because there is no defensible default now that two are on disk.
SEED = 20260902  # the evaluation's seed (configs/agent_v2e_qwen35_4b.yaml); a task-set property

# (task_id, difficulty): the test split's own ids; difficulty changes the content and the length.
AGENTIC = [
    ("test-read-0000-clean", 0), ("test-list-0005-clean", 0), ("test-pointer_chain-0006-clean", 0),
    ("test-search-0001-clean", 1), ("test-synthesis-0003-clean", 1), ("test-batch_update-0010-clean", 1),
    ("test-update-0004-clean", 2), ("test-pointer_chain-0006-clean", 2), ("test-cross_reference-0008-clean", 2),
    ("test-aggregate_report-0011-clean", 2),
]


def long_passage(words: int = 1100) -> str:
    text = (REPO / "research/agentic_paradigm.md").read_text().split()
    return " ".join(text[:words])


CHAT = [
    ("chat-short-factual", "Why does ethanol boil at a lower temperature than water at one atmosphere? Answer in three sentences."),
    ("chat-reasoning", "A tank holds 480 litres. One pump drains 30 litres a minute and a second pump fills 12 litres a minute. Both run at once from full. How many minutes until the tank is empty? Show the steps briefly."),
    ("chat-long-summary", "Summarise the following document in five sentences.\n\n" + long_passage()),
]
FOLLOW_UP = "Now give the single most important caveat to your answer, in one sentence."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="default: the band's attention members from the registry, plus the final layer")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--chat-tokens", type=int, default=160)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="episode labels to run")
    ap.add_argument("--model", default="qwen35-4b")
    ap.add_argument("--lens", type=Path, required=True, help="the .npz lens for --model")
    ap.add_argument("--lens-sha256", required=True, help="the lens digest this run pins")
    ap.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="default: configs/models/<model>.yaml, so it cannot name another model",
    )
    args = ap.parse_args(); t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = load_model_spec(args.model)
    registry = args.registry or (REPO / "configs/models" / f"{spec.name}.yaml")
    if not registry.is_file():
        ap.error(f"no registry file at {registry}")

    # the episode plan (no model): tasks resolved from the factory, prompts rendered for token counts
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_download(spec.hf_id, allow_patterns=["tokenizer*", "*.json"], local_files_only=True))
    plan = []
    by_difficulty = {d: {t.task_id: t for t in make_tasks("test", 180, SEED, difficulty=d)} for d in (0, 1, 2)}
    for task_id, d in AGENTIC:
        task = by_difficulty[d][task_id]
        label = f"agentic-d{d}-{task_id.removeprefix('test-').removesuffix('-clean')}"
        messages = [{"role": "system", "content": __import__('local_llm_lab.pipeline.protocol', fromlist=['SYSTEM_PROMPT']).SYSTEM_PROMPT}, {"role": "user", "content": task.prompt}]
        n = len(tok(build_prompt(tok, messages, spec=spec, keep_last=2, generation=True), add_special_tokens=False)["input_ids"])
        plan.append({"label": label, "kind": "agentic", "task_id": task_id, "difficulty": d, "family": task.family, "horizon": task.horizon, "opening_prompt_tokens": n})
    for label, prompt in CHAT:
        n = len(tok(build_prompt(tok, [{"role": "user", "content": prompt}], spec=spec, keep_last=2, generation=True), add_special_tokens=False)["input_ids"])
        plan.append({"label": label, "kind": "chat", "turns": 2, "opening_prompt_tokens": n})
    if args.only:
        plan = [p for p in plan if p["label"] in set(args.only)]
    for p in plan:
        print(json.dumps(p), flush=True)
    json.dump({"plan": plan, "layers": args.layers, "max_steps": args.max_steps, "max_tokens": args.max_tokens, "seed": SEED}, open(args.out / "plan.json", "w"), indent=1)
    if args.plan_only:
        print(json.dumps({"event": "plan_only", "episodes": len(plan)}))
        return

    # ---- the model (one load; the lock is taken in load_policy)
    import mlx.core as mx
    from local_llm_lab.pipeline.evaluate import load_policy, make_sampler
    from local_llm_lab.pipeline.runner import generate_turn_with_count, run_task
    from local_llm_lab.pipeline.live_lens.instruments import (
        LensIdentity,
        LensMaps,
        file_sha256,
        read_band,
    )
    from local_llm_lab.pipeline.live_lens.session import CaptureSession, LensReadout, RecordWriter, read_record

    mx.set_cache_limit(2 * 2**30)
    # Capture requires no cross-turn reuse; the registry default is `history` since 2026-09-07,
    # so this entry point resolves `none` explicitly rather than inheriting the default.
    from dataclasses import replace

    spec = replace(spec, cache_strategy="none")
    model, mtok, view, resolved = load_policy(spec, None)
    model.eval()
    assert resolved.cache_strategy == "none", resolved.cache_strategy
    # A band is a ruling, and a model may have none. Where one is declared it is read and
    # validated against the installed block kinds; where it is not, an explicit --layers is the
    # only way to say what to read, and the manifest records that no band was consulted rather
    # than leaving a reader to infer it from an empty list.
    try:
        band = read_band(registry, [view.layer_kind(i) for i in range(view.num_layers)])
    except KeyError:
        band = ()
    if not band and not args.layers:
        ap.error(
            f"{registry.name} declares no probes.live_lens_pairs, so there is no band to take "
            "layers from; pass --layers explicitly"
        )
    attention_members = tuple(p for pair in band for p in pair if view.layer_kind(p - 1) == "attention")
    layers = tuple(args.layers) if args.layers else attention_members + (view.num_layers,)
    # Issue 99: the spec says which model this is and the lens has to agree. The pinned
    # constants above made a mismatch impossible by accident; this makes it impossible.
    lens = LensMaps.load(args.lens, expected_sha256=args.lens_sha256,
                         hidden_size=view.hidden_size, num_layers=view.num_layers,
                         identity=LensIdentity(spec.name, spec.hf_id, view.num_layers))
    reader = LensReadout(view, lens)
    sampler = make_sampler(0.0)
    manifest = {"model": spec.hf_id, "lens_sha256": lens.sha256, "band": band, "layers": layers, "cache_strategy": resolved.cache_strategy,
                "registry_sha256": file_sha256(registry), "registry": str(registry),
                "lens_path": str(args.lens), "band_declared": bool(band), "top_k": args.top_k, "max_steps": args.max_steps, "max_tokens": args.max_tokens,
                "chat_tokens": args.chat_tokens, "seed": SEED, "episodes": []}
    print(json.dumps({"event": "loaded", "layers": layers, "band": band}), flush=True)

    for p in plan:
        path = args.out / f"{p['label']}.jsonl"
        if path.exists():
            print(json.dumps({"event": "skip_existing", "label": p["label"]})); continue
        t1 = time.monotonic(); mx.reset_peak_memory()
        entry = dict(p)
        with RecordWriter(path, {"model": spec.hf_id, "lens_sha256": lens.sha256, "episode": p, "layers": layers}) as write:
            session = CaptureSession(view, reader, write, layers=layers, attention_blocks=(), top_k=args.top_k)
            if p["kind"] == "agentic":
                task = by_difficulty[p["difficulty"]][p["task_id"]]
                traj = run_task(model, mtok, task, sampler=sampler, spec=spec, view=view, resolved=resolved,
                                max_steps=args.max_steps, max_tokens=args.max_tokens, keep_last=2, capture=session)
                entry.update(turns=traj.turns, generated_tokens=traj.generated_tokens, success=bool(traj.verdict.get("success")),
                             loop_detected=traj.loop_detected, exhausted=traj.exhausted, steps=[{"action": s.get("action"), "observation": str(s.get("observation", ""))[:200]} for s in traj.steps])
            else:
                prompt_text = dict(CHAT)[p["label"]]
                messages = [{"role": "user", "content": prompt_text}]
                replies = []
                for turn in range(2):
                    prompt = build_prompt(mtok, messages, spec=spec, keep_last=2, generation=True)
                    session.set_context(kind="chat", label=p["label"], turn=turn, messages=messages)
                    raw, n_tok, _ = generate_turn_with_count(model, mtok, prompt, sampler, args.chat_tokens, None, spec=spec, capture=session)
                    replies.append({"turn": turn, "prompt_tokens": len(mtok.encode(prompt)), "generated_tokens": n_tok, "text": raw})
                    messages = messages + [{"role": "assistant", "content": raw}, {"role": "user", "content": FOLLOW_UP}]
                entry.update(turns=2, replies=replies, generated_tokens=sum(r["generated_tokens"] for r in replies))
        rows = read_record(path)
        entry.update(seconds=round(time.monotonic() - t1, 1), peak_memory_gib=round(mx.get_peak_memory() / 2**30, 3), record=path.name,
                     record_sha256=file_sha256(path), record_rows=len(rows), rank_rows=sum(r["kind"] == "rank" for r in rows), reading_rows=sum(r["kind"] == "reading" for r in rows))
        manifest["episodes"].append(entry)
        json.dump(manifest, open(args.out / "manifest.json", "w"), indent=1)
        print(json.dumps({"event": "episode", **{k: v for k, v in entry.items() if k not in ("steps", "replies")}}), flush=True)
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(manifest, open(args.out / "manifest.json", "w"), indent=1)
    print(json.dumps({"event": "done", "episodes": len(manifest["episodes"]), "elapsed_s": manifest["elapsed_s"]}))


if __name__ == "__main__":
    main()
