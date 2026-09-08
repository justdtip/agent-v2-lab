"""What the layers carry at the decision that emits the wrong path (issue: the stable error).

`update-0028` emits the same wrong path twenty-three times running, with the correct literal
present in the prompt at every turn. That makes the error a fixed point rather than noise, and a
fixed point at a known position in a corpus we already hold is something the map can read without
going near the box.

The fork is one token. The correct path `workspace/test/0028/config.ini` needs `' "'` after
`"path":` and then `workspace`; the model emits the single token `' "/'`, and everything after it
is forced. So the question is what the residual carries at the position *before* that token.

Two readings, and the second is the control for the first:

* at the fork, where does the emitted wrong token become rank 1, and does the correct alternative
  appear in any layer's top-k at all;
* across every other emitted token in the same episode, where does rank 1 normally arrive — because
  "rank 1 by layer 24" means nothing until you know what layer 24 does for ordinary tokens.

No model and no box: the record carries the per-layer top-k and the emitted token's rank. The
tokenizer is loaded for decoding only.

    python fixed_point.py <record.jsonl> [--checkpoint <dir>]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics

LAYERS = (11, 12, 17, 18, 23, 24, 30, 34)

#: The fork token and the two the correct path would have needed, in this checkpoint's vocabulary.
#: Written as ids with their text beside them because the ids are what the record holds, and
#: re-deriving them from a string would silently pick a different split in another tokenizer.
FORK = (9560, ' "/')
CORRECT_QUOTE = (623, ' "')
CORRECT_WORD = (44484, "workspace")


def read(path: str):
    """Emitted tokens, per-layer top-k readings, and emitted-token ranks, each keyed correctly.

    **The two coordinate systems are different and conflating them is silent.** A `rank` row's
    position counts the turn's own stream, so the row at ``p - 1`` scores the token emitted at
    ``p`` — and it carries that token's id, which is what makes the alignment checkable rather
    than assumed. A `reading` row's position is the *forward's* offset, and with no cache across
    turns every turn re-encodes from zero, so the same position occurs once per turn. Keyed
    flat, later turns overwrite earlier ones and every lookup silently answers about the last
    turn. This cost a published claim on 2026-09-08 before the layer-34 identity caught it.

    Readings are written inside ``output()`` before that chunk's own `forward` row, so file order
    assigns each run of them to a turn. The check that this is right is the identity: at the final
    layer the readout is the model's own distribution, so its top-1 must be the emitted token, and
    it is for 100% of them once grouped by turn against 4% when keyed flat.
    """
    emitted: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    reading: dict[int, dict[int, dict]] = collections.defaultdict(dict)
    rank: dict[int, dict[int, int]] = collections.defaultdict(dict)
    ranked_token: dict[int, int] = {}
    pending: list[dict] = []
    with open(path) as handle:
        for line in handle:
            try:
                event = json.loads(line)["event"]
            except (ValueError, KeyError):
                continue
            kind = event.get("kind")
            if kind == "emitted":
                emitted[event["turn"]].append((event["position"], event["token_id"]))
            elif kind == "reading":
                pending.append(event)
            elif kind == "forward":
                for row in pending:
                    reading[event["turn"]][row["position"]] = {
                        int(k): v for k, v in row["top"].items()
                    }
                pending = []
            elif kind == "rank" and event.get("horizon") == 1:
                rank[event["position"]][int(event["layer"])] = event["rank"]
                ranked_token[event["position"]] = event["token_id"]

    for turn in emitted:
        for position, token in emitted[turn]:
            if ranked_token.get(position - 1) not in (None, token):
                raise ValueError(
                    f"rank row at {position - 1} scores {ranked_token[position - 1]} but "
                    f"{token} was emitted at {position}: the coordinate systems do not line up"
                )
    return emitted, reading, rank


def earliest_rank_one(ranks: dict[int, int]) -> int | None:
    """The shallowest read layer at which the emitted token is already the argmax.

    `None` means no read layer had it first, which for the final layer cannot happen — the
    identity's readout is the model's own distribution — so `None` here is a missing row.
    """
    for layer in LAYERS:
        if ranks.get(layer) == 1:
            return layer
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    parser.add_argument("--json")
    args = parser.parse_args()

    emitted, reading, rank = read(args.record)
    forks = []
    for turn in sorted(emitted):
        for position, token in sorted(emitted[turn]):
            if token == FORK[0]:
                forks.append((turn, position - 1))
                break
    if not forks:
        raise SystemExit(f"no turn in {args.record} emits {FORK[1]!r} at a path slot")

    at_24 = [rank[d].get(24) for _, d in forks]
    quote_layers = [
        [layer for layer in sorted(reading[turn].get(d, {})) if CORRECT_QUOTE[0] in reading[turn][d][layer]]
        for turn, d in forks
    ]
    word_anywhere = sum(
        any(CORRECT_WORD[0] in ids for ids in reading[turn].get(d, {}).values())
        for turn, d in forks
    )

    others = [
        position - 1
        for turn in emitted
        for position, _ in emitted[turn]
        if (position - 1) in rank and (position - 1) not in {d for _, d in forks}
    ]
    spread = collections.Counter(earliest_rank_one(rank[d]) for d in others)

    payload = {
        "record": args.record,
        "forks": len(forks),
        "fork_rank_1_at_layer_24": sum(1 for r in at_24 if r == 1),
        "fork_rank_at_layer_23": [rank[d].get(23) for _, d in forks],
        "correct_quote_in_top_k": sum(1 for layers in quote_layers if layers),
        "correct_quote_layers": sorted({layer for layers in quote_layers for layer in layers}),
        "correct_word_in_top_k": word_anywhere,
        "other_tokens": len(others),
        "other_earliest_rank_1": {str(k): v for k, v in sorted(spread.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "other_median_rank_at_24": statistics.median(
            [rank[d][24] for d in others if 24 in rank[d]]
        ),
    }
    print(json.dumps(payload, indent=1))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
