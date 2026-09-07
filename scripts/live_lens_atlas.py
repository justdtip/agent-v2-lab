"""Summarise live-lens pilot records: foreknowledge ranks and lens–next-token agreement by span.

Reads the hash-chained records written by ``scripts/live_lens_pilot.py`` (verifying the chain), labels
every prompt position with its span (system, task, observation, note, call, chat prose) from the
windowed messages the session stored per turn, and writes ``atlas.json`` with the pre-registered
summaries (research/records/LIVE-LENS-PILOT-2026-09-07/README.md). No model.

    .venv/bin/python scripts/live_lens_atlas.py --records <dir> --out <atlas.json>
"""
from __future__ import annotations

import argparse, json, statistics, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from local_llm_lab.models import load_model_spec  # noqa: E402
from local_llm_lab.pipeline.live_lens.session import read_record  # noqa: E402

SPANS = ("system", "task", "observation", "note", "call", "template", "chat")


def spans_for_prompt(tok, prompt: str, messages: list[dict], kind: str) -> list[str]:
    """One span label per prompt token, from the rendered messages' text located inside the prompt."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    labels = ["template"] * len(enc["input_ids"])
    ranges = []
    cursor = 0
    for i, m in enumerate(messages):
        content = m["content"]
        if not content:
            continue
        k = prompt.find(content, cursor)
        if k < 0:
            continue
        if m["role"] == "system":
            ranges.append((k, k + len(content), "system"))
        elif m["role"] == "user":
            ranges.append((k, k + len(content), "chat" if kind == "chat" else "task"))
        elif m["role"] == "tool":
            ranges.append((k, k + len(content), "observation"))
        elif m["role"] == "assistant":
            fence = content.find("```json")
            if fence >= 0 and kind != "chat":
                ranges.append((k, k + fence, "note")); ranges.append((k + fence, k + len(content), "call"))
            else:
                ranges.append((k, k + len(content), "chat" if kind == "chat" else "note"))
        cursor = k + len(content)
    for t, (s, e) in enumerate(enc["offset_mapping"]):
        for a, b, lab in ranges:
            if s >= a and e <= b:
                labels[t] = lab; break
    return labels


def emitted_span(text_so_far: str, kind: str) -> str:
    if kind == "chat":
        return "chat"
    return "call" if "```json" in text_so_far else "note"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="qwen35-4b")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    spec = load_model_spec(args.model)
    tok = AutoTokenizer.from_pretrained(snapshot_download(spec.hf_id, allow_patterns=["tokenizer*", "*.json"], local_files_only=True))
    manifest = json.load(open(args.records / "manifest.json"))
    by_label = {e["label"]: e for e in manifest["episodes"]}

    # accumulators: (layer, horizon, kind, difficulty, span) -> ranks ; (layer, span, kind) -> [agree, total]
    ranks = defaultdict(list)
    agree = defaultdict(lambda: [0, 0])
    episodes = []
    for path in sorted(args.records.glob("*.jsonl")):
        rows = read_record(path)
        label = path.stem; ep = by_label.get(label, {}); kind = ep.get("kind", "agentic"); diff = ep.get("difficulty", "chat")
        turn_ctx = {}; prompt_ids = {}; labels = {}
        emitted_text = defaultdict(str)
        n_turns = 0; n_rank = 0; n_reading = 0
        for r in rows:
            if r["kind"] == "begin_turn":
                t = r["turn"]; n_turns += 1
                ctx = r["context"]; msgs = ctx.get("messages", [])
                turn_ctx[t] = ctx; prompt_ids[t] = r["prompt_ids"]
                labels[t] = spans_for_prompt(tok, r["prompt"], msgs, kind)
            elif r["kind"] == "reading":
                t = r["turn"]; pos = r["position"]; n_reading += 1
                ids = prompt_ids[t]
                if pos + 1 < len(ids):  # a prompt position with a known next token
                    span = labels[t][pos] if pos < len(labels[t]) else "template"
                    nxt = ids[pos + 1]
                    for layer, top in r["top"].items():
                        key = (int(layer), span, kind)
                        agree[key][1] += 1; agree[key][0] += int(top[0] == nxt)
            elif r["kind"] == "emitted":
                t = r["turn"]
                emitted_text[t] += tok.decode([r["token_id"]])
            elif r["kind"] == "rank":
                t = r["turn"]; n_rank += 1
                span = emitted_span(emitted_text[t], kind)
                ranks[(r["layer"], r["horizon"], kind, str(diff), span)].append(r["rank"])
        episodes.append({"label": label, "kind": kind, "difficulty": diff, "turns": n_turns, "rank_rows": n_rank, "reading_rows": n_reading,
                         "seconds": ep.get("seconds"), "success": ep.get("success"), "generated_tokens": ep.get("generated_tokens")})

    def summarise(values):
        return {"n": len(values), "median_rank": statistics.median(values) if values else None,
                "share_le_10": round(sum(v <= 10 for v in values) / len(values), 4) if values else None,
                "share_rank_1": round(sum(v == 1 for v in values) / len(values), 4) if values else None}
    fore = defaultdict(dict)
    # by layer × horizon × kind × span (difficulties pooled within kind: stated), and by difficulty at layer 20
    pooled = defaultdict(list)
    for (layer, h, kind, diff, span), v in ranks.items():
        pooled[(layer, h, kind, span)] += v
        pooled[(layer, h, kind, "all")] += v
        pooled[(layer, h, "all", "all")] += v
    for (layer, h, kind, span), v in pooled.items():
        fore[f"L{layer}"][f"h{h}|{kind}|{span}"] = summarise(v)
    by_diff = {}
    for (layer, h, kind, diff, span), v in ranks.items():
        if layer == 20 and kind == "agentic":
            by_diff.setdefault(f"h{h}|d{diff}|{span}", []).extend(v)
    agreement = {f"L{layer}": {} for layer in sorted({k[0] for k in agree})}
    for (layer, span, kind), (a, n) in agree.items():
        agreement[f"L{layer}"][f"{span}|{kind}"] = {"n": n, "top1_agreement": round(a / n, 4) if n else None}
    out = {"episodes": episodes, "foreknowledge": fore, "foreknowledge_layer20_by_difficulty": {k: summarise(v) for k, v in by_diff.items()},
           "prompt_agreement": agreement, "manifest": {k: v for k, v in manifest.items() if k != "episodes"}}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({"episodes": len(episodes), "rank_rows": sum(e["rank_rows"] for e in episodes), "reading_rows": sum(e["reading_rows"] for e in episodes)}))
    for L in ("L20", "L12", "L28", "L32"):
        for key in ("h1|agentic|call", "h1|agentic|note", "h1|chat|chat", "h4|agentic|all", "h8|agentic|all"):
            s = fore.get(L, {}).get(key)
            if s: print(L, key, s)


if __name__ == "__main__":
    main()
