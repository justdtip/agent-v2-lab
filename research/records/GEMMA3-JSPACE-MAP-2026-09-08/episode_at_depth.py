"""One episode read at every layer: its forks, its certainty, and what the other branches were.

Written for `update-0028` under the fixed simulator, where the model alternates two rejected
directory guesses for twenty-four turns, but general to any agentic record with span labels.

No model. The tokenizer is loaded for decoding only. Every rank-side number goes through the
validated joiner, so the span facet cannot exist without the join's assertions passing; every
reading-side number groups readings by turn, because their positions restart every turn and a flat
key silently answers about the last one.

    python episode_at_depth.py <record.jsonl> [--control <falsification.json>] [--json out]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys

sys.path.insert(0, "/Users/daniel.tipton/Desktop/An app/src")

LAYERS = tuple(range(1, 35))
HORIZONS = (1, 4, 8)


def load(path):
    events = [json.loads(line)["event"] for line in open(path) if line.startswith("{")]
    if events[-1].get("kind") != "end_record" or events[-1].get("status") != "complete":
        raise SystemExit(f"{path} is not a complete record; its footer says {events[-1]}")
    readings = collections.defaultdict(dict)
    pending = []
    for event in events:
        kind = event.get("kind")
        if kind == "reading":
            pending.append(event)
        elif kind == "forward":
            for row in pending:
                readings[event["turn"]][row["position"]] = {int(k): v for k, v in row["top"].items()}
            pending = []
    return events, readings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    parser.add_argument("--control", help="falsification.json from the no-capture run of the same task")
    parser.add_argument("--json")
    args = parser.parse_args()

    from local_llm_lab.pipeline.live_lens.spans import join_spans
    from local_llm_lab.project import configure_local_cache

    configure_local_cache()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/Users/daniel.tipton/Desktop/An app/models/gemma-3-4b-it-4bit")

    events, readings = load(args.record)
    join = join_spans(events)  # asserts before any number exists
    rows = [r for r in join.rows if r.horizon == 1]
    by_turn_pos = {(r.turn, r.position): r for r in rows}
    out = {"record": args.record, "turns": len({r.turn for r in rows}), "emitted": len({(r.turn, r.position) for r in rows})}

    # ---- 1. the forks: first argument token of each turn, read at every layer
    forks = {}
    for r in rows:
        if r.span == "call_argument" and (r.turn not in forks or r.position < forks[r.turn].position):
            forks[r.turn] = r
    fork_rows = collections.defaultdict(dict)   # turn -> layer -> (rank, prob)
    for r in join.rows:
        f = forks.get(r.turn)
        if f and r.position == f.position and r.horizon == 1:
            fork_rows[r.turn][r.layer] = (r.rank, r.probability)
    fork_table = []
    for t in sorted(forks):
        f = forks[t]
        ranks = {L: fork_rows[t].get(L, (None, None))[0] for L in LAYERS}
        first_rank1 = next((L for L in LAYERS if ranks.get(L) == 1), None)
        fork_table.append({
            "turn": t, "token": tok.decode([f.token_id]), "token_id": f.token_id,
            "first_layer_at_rank_1": first_rank1,
            "rank_at_23": ranks.get(23), "rank_at_24": ranks.get(24),
            "p_final": fork_rows[t].get(34, (None, None))[1],
            "worst_layer": max(LAYERS, key=lambda L: ranks.get(L) or 0),
            "worst_rank": max(v for v in ranks.values() if v is not None),
        })
    out["forks"] = fork_table

    # ---- 2. the other branch: at each fork, is the token the *other* turns chose in any top-10
    fork_ids = collections.Counter(f["token_id"] for f in fork_table)
    alternatives = [tid for tid, _ in fork_ids.most_common()]
    presence = []
    for f in fork_table:
        t, p = f["turn"], forks[f["turn"]].position
        src = p - 1
        tops = readings[t].get(src, {})
        others = [a for a in alternatives if a != f["token_id"]]
        layers_with_other = {tok.decode([a]): [L for L in LAYERS if a in tops.get(L, [])] for a in others}
        presence.append({"turn": t, "emitted": f["token"], "other_branch_layers": layers_with_other})
    out["other_branch"] = presence

    # ---- 3. the escape route: at the tool-name decision, is `search` ever a candidate at any depth
    search_ids = set(tok.encode("search", add_special_tokens=False)) | set(tok.encode(" search", add_special_tokens=False))
    list_ids = set(tok.encode("list", add_special_tokens=False)) | set(tok.encode(" list", add_special_tokens=False))
    # One row per emitted token, not one per layer: `rows` carries every layer's rank row for a
    # position, and the first version counted each decision 34 times.
    seen = set()
    tool_rows = []
    for r in rows:
        if r.span == "call_skeleton" and (r.token_id in list_ids or r.token_id in search_ids):
            if (r.turn, r.position) not in seen:
                seen.add((r.turn, r.position)); tool_rows.append(r)
    escape = []
    for r in tool_rows:
        tops = readings[r.turn].get(r.position - 1, {})
        escape.append({"turn": r.turn, "emitted": tok.decode([r.token_id]),
                       "search_in_top10_at_layers": [L for L in LAYERS if any(s in tops.get(L, []) for s in search_ids)]})
    out["escape_route"] = escape

    # ---- 4. span-split profile for this episode
    prof = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        prof[r.span][r.layer].append(r.rank)
    out["profile_by_span"] = {
        s: {str(L): {"median": statistics.median(v), "rank1": sum(1 for x in v if x == 1) / len(v)}
            for L, v in sorted(prof[s].items())}
        for s in prof
    }
    out["span_counts"] = {s: len(prof[s][1]) for s in prof}

    # ---- 5. the trajectory against the no-capture control, if given
    if args.control:
        control = json.load(open(args.control))
        begin = [e for e in events if e.get("kind") == "begin_turn"]
        out["control"] = {"control_turns": control["turns"], "record_turns": len(begin),
                          "control_loop_detected": control["loop_detected"],
                          "control_first_actions": [s["action"] for s in control["steps"][:4]]}

    print(json.dumps({k: v for k, v in out.items() if k != "profile_by_span"}, indent=1)[:6000])
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
