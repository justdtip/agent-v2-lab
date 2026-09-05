"""Retrieval-heavy contexts with known source spans and read positions (Director, 2026-09-06).

Four context types. Facts sit in the first part, queries at the end, with at least MIN_GAP tokens
between the last fact and the first query, so every source is a far source by construction. Each
item records the fact's token span, the read position (the last query token before the answer) and
the expected answer, so the model's next token can be scored and the read weights at the read
position can be tested against the known span.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

MIN_GAP = 300

NOUNS = ["harbour", "ledger", "orchard", "council", "lantern", "meadow", "furnace", "archive", "compass", "bridge", "quarry", "garden", "market", "tower", "canal", "engine", "library", "valley", "mill", "chapel"]
ADJS = ["quiet", "narrow", "weathered", "bright", "distant", "careful", "restless", "heavy", "slender", "patient", "crowded", "silent", "early", "faded", "steady"]
VERBS = ["reported", "noted", "described", "measured", "recorded", "observed", "considered", "mentioned", "reviewed", "listed"]
SURNAMES = ["Okonkwo", "Lindqvist", "Ferreira", "Nakamura", "Abernathy", "Castellano", "Whitcombe", "Delacroix", "Oyelaran", "Marchetti", "Vasquez", "Thornbury"]
FIRST = ["Marisol", "Teodor", "Ingrid", "Kwame", "Beatrix", "Rafael", "Sunniva", "Emeka", "Cordelia", "Hamish", "Leocadia", "Anselm"]
TOWNS = ["Brennford", "Saltmarsh", "Kellowick", "Ardenmoor", "Thistlecombe", "Ravensby", "Oxcroft", "Wendlebury", "Marrowgate", "Pellstow"]


def filler(rng, sentences):
    out = []
    for _ in range(sentences):
        a, b, c = rng.choice(ADJS), rng.choice(NOUNS), rng.choice(NOUNS)
        out.append(f"The {a} {b} near the {c} was {rng.choice(VERBS)} in the {rng.choice(ADJS)} report of that season, and the {rng.choice(NOUNS)} committee {rng.choice(VERBS)} it again the following spring.")
    return " ".join(out)


def kv_needles(rng, n_pairs=16, n_queries=6):
    keys = [f"{rng.choice(['Delta', 'Sigma', 'Kappa', 'Omega', 'Theta', 'Lambda'])}-{rng.randint(2, 98)}" for _ in range(n_pairs)]
    keys = list(dict.fromkeys(keys))[:n_pairs]
    vals = [str(rng.randint(10000, 99999)) for _ in keys]
    parts, facts = [], []
    for k, v in zip(keys, vals):
        parts.append(filler(rng, rng.randint(2, 4)))
        parts.append(f"The access code for vault {k} is {v}.")
        facts.append((k, v))
    parts.append(filler(rng, 14))
    text = " ".join(parts)
    queries = []
    for k, v in rng.sample(facts, n_queries):
        queries.append({"query": f"\nQ: What is the access code for vault {k}?\nA: The access code is", "answer": f" {v}", "fact": f"The access code for vault {k} is {v}."})
    return text, queries


def reference_back(rng, n_people=8, n_queries=6):
    people = []
    parts = []
    for _ in range(n_people):
        name = f"{rng.choice(FIRST)} {rng.choice(SURNAMES)}"
        year = rng.randint(1951, 1999); town = rng.choice(TOWNS); pet = rng.choice(["heron", "tortoise", "lynx", "ferret", "otter", "falcon"])
        parts.append(filler(rng, rng.randint(2, 4)))
        parts.append(f"{name} was born in {year} in the town of {town} and later kept a {pet} as a companion.")
        people.append((name, year, town, pet))
    parts.append(filler(rng, 14))
    text = " ".join(parts)
    queries = []
    for name, year, town, pet in rng.sample(people, n_queries):
        kind = rng.choice(["year", "town", "pet"])
        if kind == "year":
            queries.append({"query": f"\nQ: In what year was {name} born?\nA: {name.split()[0]} was born in", "answer": f" {year}", "fact": f"{name} was born in {year}"})
        elif kind == "town":
            queries.append({"query": f"\nQ: In which town was {name} born?\nA: {name.split()[0]} was born in the town of", "answer": f" {town}", "fact": f"in the town of {town}"})
        else:
            queries.append({"query": f"\nQ: What animal did {name} keep as a companion?\nA: {name.split()[0]} kept a", "answer": f" {pet}", "fact": f"kept a {pet}"})
    return text, queries


def code_context(rng, n_consts=12, n_queries=6):
    names = rng.sample(["alpha_threshold", "retry_limit", "batch_window", "decay_rate", "max_depth", "seed_offset", "cache_span", "warmup_steps", "tolerance", "page_size", "burst_count", "cooldown_ms", "quorum_size", "fanout"], n_consts)
    vals = [str(rng.choice([rng.randint(2, 999), round(rng.uniform(0.01, 9.99), 3)])) for _ in names]
    lines = ["# configuration constants for the ingest service", "import math", ""]
    for n, v in zip(names, vals):
        lines.append(f"{n} = {v}")
        lines.append(f"# {rng.choice(ADJS)} setting used by the {rng.choice(NOUNS)} stage")
    lines.append("")
    lines.append("def run_stage(items):")
    for _ in range(40):
        lines.append(f"    total_{rng.choice(NOUNS)} = sum(len(str(x)) for x in items) + {rng.randint(1, 50)}  # {rng.choice(ADJS)} {rng.choice(NOUNS)}")
    lines.append("    return items")
    lines.append("")
    lines.append("# " + filler(rng, 12))
    text = "\n".join(lines)
    queries = []
    for n, v in rng.sample(list(zip(names, vals)), n_queries):
        queries.append({"query": f"\nassert {n} ==", "answer": f" {v}", "fact": f"{n} = {v}"})
    return text, queries


def icl_mapping(rng, n_symbols=8, n_examples=32, n_queries=6):
    symbols = rng.sample(["blorp", "quint", "zarn", "felmo", "trask", "wibble", "korrin", "plav", "drusk", "yemli", "santor", "gribe"], n_symbols)
    mapping = {s: rng.randint(0, 9) for s in symbols}
    lines = ["Each symbol maps to a digit. Learn the mapping from the examples."]
    seq = [rng.choice(symbols) for _ in range(n_examples)]
    for s in symbols:
        if s not in seq: seq[rng.randrange(n_examples)] = s
    for s in seq:
        lines.append(f"input: {s} -> output: {mapping[s]}")
    lines.append(filler(rng, 30))
    text = "\n".join(lines)
    queries = []
    for s in rng.sample(symbols, n_queries):
        queries.append({"query": f"\ninput: {s} -> output:", "answer": f" {mapping[s]}", "fact": f"input: {s} -> output: {mapping[s]}"})
    return text, queries


BUILDERS = {"kv_needles": kv_needles, "reference_back": reference_back, "code": code_context, "icl_mapping": icl_mapping}


CONTROLS = [
    {"query": "\nQ: What is seven plus five?\nA: Seven plus five is", "answer": " twelve"},
    {"query": "\nQ: Which season comes after spring?\nA: The season after spring is", "answer": " summer"},
    {"query": "\nQ: What is the opposite of cold?\nA: The opposite of cold is", "answer": " hot"},
]


def build(seed=20260906, per_type=2):
    """Each context carries its retrieval queries and three matched controls that need no lookup;
    a control's placebo span is the first retrieval query's fact, so the same span statistic runs
    on a question that has no reason to read it."""
    rng = random.Random(seed)
    contexts = []
    for kind, fn in BUILDERS.items():
        for i in range(per_type):
            text, queries = fn(rng)
            for q in queries: q["control"] = False
            placebo = queries[0]["fact"]
            controls = [{**c, "fact": placebo, "control": True} for c in CONTROLS]
            contexts.append({"kind": kind, "index": i, "text": text, "queries": queries + controls})
    return contexts


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus.json")
    contexts = build()
    out.write_text(json.dumps(contexts, indent=1))
    for c in contexts:
        print(c["kind"], c["index"], "chars", len(c["text"]), "queries", len(c["queries"]))
    print("wrote", out)
