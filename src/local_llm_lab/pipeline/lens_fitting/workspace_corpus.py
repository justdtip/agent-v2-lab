"""Capture-identical rendered workspace inputs; no model or chat re-rendering.

The source index is an identity receipt, not a source of token values. Every
indexed decision is reconstructed, including excluded episodes, before sampling.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_KIND = "rendered_workspace_v1"
SELECTION = "one-per-episode: sha256(task_id,step); ordered round-robin family/context-band"
BOS_POLICY = "No special tokens added; any literal rendered BOS is retained."


def _bytes(obj):
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _record(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=_sha(path.read_bytes()))


def _verify(record):
    data = Path(record["path"]).read_bytes()
    if _sha(data) != record["sha256"]:
        raise ValueError(f"source hash mismatch: {record['path']}")
    return data


def _rows(data):
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def _equal(a, b, label):
    if a != b:
        raise ValueError(f"{label} mismatch")


def _band(n):
    # Bands are one and two Gemma sliding windows, not model hidden dimensions.
    window = 1024
    return "below_window" if n <= window else "above_window" if n <= 2 * window else "long"


def load_workspace_candidates(
    corpus_path, index_path, positions_path, tokenizer_path, *, max_tokens=None
):
    """Return one frozen full-context candidate per eligible episode and provenance."""
    from tokenizers import Tokenizer

    from .corpus import TRAINING_SPLITS, _training_id

    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
        raise ValueError("max_tokens must be a positive integer or None")
    sources = {
        name: _record(path)
        for name, path in zip(
            ("corpus", "index", "positions", "tokenizer"),
            (corpus_path, index_path, positions_path, tokenizer_path),
            strict=True,
        )
    }
    seal = json.loads(_verify(sources["positions"]))
    _equal(seal.get("schema_version"), 1, "positions schema")
    _equal(seal.get("corpus_sha256"), sources["corpus"]["sha256"], "corpus digest")
    _equal(seal.get("tokenizer_sha256"), sources["tokenizer"]["sha256"], "tokenizer digest")
    _equal(
        seal.get("capture_files_sha256", {}).get("index.jsonl"),
        sources["index"]["sha256"],
        "index digest",
    )
    _equal(seal.get("reduction"), "per-position, no reduction", "positions reduction")
    _equal(seal.get("endpoint"), "pre-final-norm", "positions endpoint")
    raw, seen = [], set()
    for row in _rows(_verify(sources["corpus"])):
        meta = row.get("metadata", {})
        if "task_id" not in meta:
            continue
        key = (meta["task_id"], meta["step"])
        if type(key[0]) is not str or type(key[1]) is not int or key[1] < 0:
            raise ValueError("invalid task/step identity")
        if key not in seen:
            seen.add(key)
            raw.append(row)
    index = _rows(_verify(sources["index"]))
    _equal([r.get("i") for r in index], list(range(len(raw))), "complete capture index")
    tok = Tokenizer.from_file(sources["tokenizer"]["path"])
    reconstructed = []
    for i, (source, ix) in enumerate(zip(raw, index, strict=True)):
        meta = source["metadata"]
        for field in ("task_id", "step", "family", "variant", "recovery"):
            _equal(ix.get(field), meta.get(field), f"capture index {field}")
        prompt, completion = source["prompt"], source["completion"]
        p = tok.encode(prompt, add_special_tokens=False)
        c = tok.encode(completion, add_special_tokens=False)
        match = re.search(r'"name":\s*"([a-z_]+)"', completion)
        if match is None:
            raise ValueError(f"row {i} has no tool-name token")
        target = next((j for j, (a, b) in enumerate(c.offsets) if a <= match.start(1) < b), None)
        if target is None or not p.ids or not c.ids:
            raise ValueError("cannot reconstruct token offsets")
        actual = dict(
            P_note=len(p.ids) - 1,
            P_act=len(p.ids) + target - 1,
            n_prompt_tokens=len(p.ids),
            n_note_tokens=target,
        )
        for field, value in actual.items():
            _equal(ix.get(field), value, f"capture index {field}")
        ids = p.ids + c.ids[: target + 1]
        # Keep both original strings and offset origins, avoiding any concatenation re-encode.
        offsets = [list(x) for x in p.offsets] + [
            [a + len(prompt), b + len(prompt)] for a, b in c.offsets[: target + 1]
        ]
        fence = completion.find("```")
        call_start = fence if fence >= 0 else completion.find("{")
        tags = ["prompt_unclassified"] * len(p.ids) + [
            "note" if a < call_start else "call_skeleton" for a, _ in c.offsets[: target + 1]
        ]
        reconstructed.append(
            dict(
                index=i,
                ids=ids,
                offsets=offsets,
                spans=tags,
                split="fit",
                domain="agentic",
                task_id=meta["task_id"],
                step=meta["step"],
                family=meta["family"],
                variant=meta["variant"],
                metadata=meta,
                prompt=prompt,
                completion=completion,
                **actual,
                context_tokens=len(ids),
                context_band=_band(len(ids)),
                token_origins=["prompt"] * len(p.ids) + ["completion"] * (target + 1),
                target_token_ids={"P_note": c.ids[0], "P_act": c.ids[target]},
                input_ids_sha256=_sha(_bytes(ids)),
                original_row=i,
                bos_policy=BOS_POLICY,
                starts_with_rendered_bos=prompt.startswith("<bos>"),
            )
        )
    cells = seal.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("positions cells must be nonempty")
    excluded, used = set(), set()
    for cell in cells:
        i, pos = cell.get("row"), cell.get("position")
        if (
            type(i) is not int
            or not 0 <= i < len(raw)
            or pos not in ("P_note", "P_act")
            or (i, pos) in used
        ):
            raise ValueError("duplicate or invalid comparison cell")
        used.add((i, pos))
        r = reconstructed[i]
        _equal(cell.get("token_index"), r[pos], "comparison position")
        _equal(cell.get("token_id"), r["target_token_ids"][pos], "comparison target token")
        _equal(cell.get("context_tokens"), len(r["ids"]), "comparison context")
        excluded.add(r["task_id"])
    eligible, nontraining = [], set()
    for r in reconstructed:
        try:
            _training_id(r["task_id"])
        except ValueError:
            if r["task_id"].split("-")[0] in TRAINING_SPLITS:
                raise
            nontraining.add(r["task_id"])
            continue
        if r["task_id"] in excluded:
            continue
        if r["task_id"].split("-")[1] != r["family"] or r["task_id"].split("-")[3] != r["variant"]:
            raise ValueError("task family/variant disagrees with metadata")
        if max_tokens is not None and len(r["ids"]) > max_tokens:
            raise ValueError("eligible context exceeds max_tokens; truncation is forbidden")
        eligible.append(r)
    if not eligible:
        raise ValueError("no eligible training episodes remain")
    per_episode = defaultdict(list)
    for r in eligible:
        per_episode[r["task_id"]].append(r)
    candidates = [
        min(rs, key=lambda r: _sha(_bytes([r["task_id"], r["step"]])))
        for rs in per_episode.values()
    ]
    groups = defaultdict(list)
    for r in sorted(candidates, key=lambda r: _sha(_bytes([r["task_id"], r["step"]]))):
        groups[r["family"], r["context_band"]].append(r)
    ordered = []
    while groups:
        for key in sorted(list(groups)):
            ordered.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
    provenance = dict(
        source_kind=SOURCE_KIND,
        sources=sources,
        selection_rule=SELECTION,
        excluded_episode_ids=sorted(excluded),
        nontraining_episode_ids=sorted(nontraining),
        counts=dict(
            capture_rows=len(raw),
            comparison_cells=len(cells),
            eligible_rows=len(eligible),
            fit=len(ordered),
        ),
        bos_policy=BOS_POLICY,
        max_tokens=max_tokens,
        span_policy="Original prompt unclassified; completion note/call skeleton through tool name; no argument tokens retained.",
        position_histogram=dict(
            sorted(Counter(str(p) for r in ordered for p in range(len(r["ids"]))).items())
        ),
        context_histogram=dict(sorted(Counter(str(len(r["ids"])) for r in ordered).items())),
    )
    return ordered, provenance


def build_workspace_corpus(
    corpus_path, index_path, positions_path, tokenizer_path, out_dir, *, max_tokens=None
):
    rows, manifest = load_workspace_candidates(
        corpus_path, index_path, positions_path, tokenizer_path, max_tokens=max_tokens
    )
    dest = Path(out_dir)
    path = dest if dest.suffix == ".json" else dest / "manifest.json"
    rows_path = path.with_suffix(".jsonl") if dest.suffix == ".json" else dest / "rows.jsonl"
    if path.exists() or rows_path.exists():
        raise FileExistsError("immutable corpus already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x") as f:
        for row in rows:
            f.write(_bytes(row).decode() + "\n")
    manifest.update(schema_version=1, domain="agentic", rows=_record(rows_path), dropped_rows=[])
    manifest["manifest_sha256"] = _sha(_bytes(manifest))
    with path.open("x") as f:
        f.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_workspace_corpus(path):
    manifest = json.loads(Path(path).read_bytes())
    digest = manifest.pop("manifest_sha256", None)
    if digest != _sha(_bytes(manifest)):
        raise ValueError("manifest hash mismatch")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_kind") != SOURCE_KIND
        or manifest.get("domain") != "agentic"
    ):
        raise ValueError("unsupported workspace corpus manifest")
    for record in manifest["sources"].values():
        _verify(record)
    paths = [
        manifest["sources"][name]["path"] for name in ("corpus", "index", "positions", "tokenizer")
    ]
    expected, provenance = load_workspace_candidates(*paths, max_tokens=manifest["max_tokens"])
    for key, value in provenance.items():
        _equal(manifest.get(key), value, f"reconstructed manifest {key}")
    actual = _rows(_verify(manifest["rows"]))
    _equal(actual, expected, "reconstructed rows")
    return actual
