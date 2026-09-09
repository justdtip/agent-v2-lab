"""Exact fixed-history replay with independent legacy-atlas identity evidence.

Preflight is pure. Only replay_record imports the native array backend. One source
record is held at a time; production capture never retains raw logits.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import runpy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from local_llm_lab.forward import ForwardLedger, encode_prompt
from local_llm_lab.pipeline.lens_fitting.runtime import resolve_snapshot
from local_llm_lab.pipeline.live_lens.instruments import (
    LensIdentity,
    LensMaps,
    file_sha256,
)
from local_llm_lab.pipeline.live_lens.session import (
    CaptureSession,
    LensReadout,
    RecordWriter,
    read_record,
)
from local_llm_lab.project import PROJECT_ROOT
from local_llm_lab.spawn import run


def read_source(path: Path, *, expected: dict | None = None) -> tuple[list[dict], dict]:
    """Adapt gzip text to the existing, unchanged chain/footer validator."""
    path = Path(path)
    raw = path.read_bytes()
    decoded = gzip.decompress(raw) if path.suffix == ".gz" else raw
    identity = {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "decompressed_sha256": hashlib.sha256(decoded).hexdigest(),
    }
    if expected is not None and identity != expected:
        raise ValueError(f"source changed since preflight: {path}")
    events = read_record(SimpleNamespace(read_text=lambda: decoded.decode("utf-8")))
    validate_events(events)
    return events, identity


def validate_events(events: list[dict]) -> None:
    """Validate complete turn boundaries and successful-forward accounting, offline."""
    ledger = None
    turn = -1
    reading_positions = []
    rank_queue = []
    attention_rows = []
    top_ids = []
    vocab = None
    for row in events[1:-1]:
        kind = row["kind"]
        if rank_queue and kind != "rank":
            raise ValueError("missing or reordered rank events")
        if kind == "begin_turn":
            if ledger is not None or row["turn"] != turn + 1:
                raise ValueError("invalid begin turn order")
            turn += 1
            ledger = ForwardLedger(row["prompt_ids"])
            reading_positions = []
            rank_queue = []
            attention_rows = []
            top_ids = []
            if not isinstance(row["prompt"], str) or not isinstance(row["context"], dict):
                raise ValueError("invalid prompt/context")
            layers = row["layers"]
            if (
                not layers
                or layers != sorted(set(layers))
                or any(type(x) is not int or x < 1 for x in layers)
            ):
                raise ValueError("invalid capture layers")
            blocks = row["attention_blocks"]
            top_k = row["top_k"]
            if blocks != sorted(set(blocks)) or any(type(x) is not int or x < 0 for x in blocks):
                raise ValueError("invalid attention blocks")
            if (
                type(top_k) is not int
                or type(row["audit_modulus"]) is not int
                or top_k < 1
                or row["audit_modulus"] < 1
            ):
                raise ValueError("invalid capture policy")
            if row["rank_horizons"] != [1, 4, 8] or row["distribution_capacity"] != 16:
                raise ValueError("unsupported rank capture policy")
            continue
        if ledger is None or row.get("turn") != turn:
            raise ValueError("event outside its turn")
        if kind == "reading":
            if row["position"] != ledger.offset + len(reading_positions):
                raise ValueError("non-contiguous reading positions")
            if set(row["top"]) != {str(x) for x in layers}:
                raise ValueError("reading layers disagree with capture policy")
            for top in row["top"].values():
                if (
                    len(top) != top_k
                    or len(set(top)) != top_k
                    or any(type(x) is not int or x < 0 for x in top)
                ):
                    raise ValueError("invalid reading top-k")
                top_ids.extend(top)
            reading_positions.append(row["position"])
        elif kind == "forward":
            ids = row["input_ids"]
            ledger.record(row["offset"], ids)
            if len(reading_positions) != len(ids):
                raise ValueError("reading/forward partition mismatch")
            reading_positions = []
            attention_index = 0
            for block in blocks:
                for position in range(row["offset"], ledger.offset):
                    if attention_index >= len(attention_rows):
                        raise ValueError("missing attention source")
                    source = attention_rows[attention_index]
                    if (source["kind"], source.get("layer"), source["position"]) != (
                        "source",
                        block,
                        position,
                    ):
                        raise ValueError("invalid attention source order")
                    attention_index += 1
                head = 0
                while (
                    attention_index < len(attention_rows)
                    and attention_rows[attention_index]["kind"] == "head"
                ):
                    observed = attention_rows[attention_index]
                    if (observed["block"], observed["head"], observed["position"]) != (
                        block,
                        head,
                        ledger.offset - 1,
                    ):
                        raise ValueError("invalid attention head order")
                    head += 1
                    attention_index += 1
                if not head:
                    raise ValueError("missing attention heads")
            if attention_index != len(attention_rows):
                raise ValueError("unexpected attention events")
            attention_rows = []
            shape = row["logits_shape"]
            if (
                len(shape) != 3
                or shape[:2] != [1, len(ids)]
                or type(shape[2]) is not int
                or shape[2] < 1
            ):
                raise ValueError("invalid native logits shape")
            if vocab is not None and shape[2] != vocab:
                raise ValueError("native vocabulary changed within record")
            vocab = shape[2]
            if top_k > vocab or any(x >= vocab for x in ids + top_ids):
                raise ValueError("token outside native vocabulary")
            top_ids = []
            if len(row["argmax"]) != 1 or len(row["argmax"][0]) != len(ids):
                raise ValueError("invalid native argmax shape")
            if any(type(x) is not int or not 0 <= x < shape[2] for x in row["argmax"][0]):
                raise ValueError("invalid native argmax token")
            if not re.fullmatch("[0-9a-f]{64}", row["logits_sha256"]):
                raise ValueError("invalid native logit hash")
        elif kind == "emitted":
            if reading_positions or attention_rows or ledger.offset < len(ledger.prompt_ids):
                raise ValueError("emission before complete forward")
            if row["position"] != len(ledger.prompt_ids) + len(ledger.generated):
                raise ValueError("non-contiguous emission position")
            if type(row["token_id"]) is not int or not 0 <= row["token_id"] < vocab:
                raise ValueError("emitted token outside native vocabulary")
            ledger.emitted(row["token_id"])
            rank_queue = [
                (row["position"] - horizon, layer, horizon, row["token_id"])
                for horizon in (1, 4, 8)
                if max(0, ledger.offset - 16) <= row["position"] - horizon < ledger.offset
                for layer in layers
            ]
        elif kind == "rank":
            coordinate = tuple(row[k] for k in ("position", "layer", "horizon", "token_id"))
            if not rank_queue or coordinate != rank_queue.pop(0):
                raise ValueError("invalid rank order or coordinates")
            if type(row["rank"]) is not int or row["rank"] < 1 or not 0 <= row["probability"] <= 1:
                raise ValueError("invalid rank statistic")
        elif kind == "end_turn":
            if (
                row["status"] != "complete"
                or reading_positions
                or attention_rows
                or ledger.offset < len(ledger.prompt_ids)
                or row["forwarded_count"] != ledger.offset
                or row["emitted_count"] != len(ledger.generated)
            ):
                raise ValueError("incomplete or inconsistent end turn")
            ledger = None
        elif kind in {"source", "head"}:
            if reading_positions or not blocks:
                raise ValueError("unexpected attention event order")
            attention_rows.append(row)
        else:
            raise ValueError(f"unsupported event kind: {kind}")
    if ledger is not None or turn < 0:
        raise ValueError("record requires complete turns")


def first_difference(expected, actual, path="$") -> str | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return f"{path}: keys differ"
        for key in sorted(expected):
            difference = first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path}: length {len(expected)} != {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
    elif expected != actual or type(expected) is not type(actual):
        return f"{path}: {expected!r} != {actual!r}"
    return None


@dataclass(frozen=True)
class PreparedReplay:
    spec: object
    snapshot: dict
    source: Path
    manifest: dict
    manifest_sha256: str
    records: tuple[dict, ...]
    output: Path
    lens_path: Path
    lens_sha256: str
    layers: tuple[int, ...]
    domain: str
    kind: str
    identity_atlas: Path | None
    identity_atlas_sha256: str | None
    hidden_size: int
    num_layers: int


def prepare_replay(
    source: Path,
    spec,
    output: Path,
    lens_path: Path,
    *,
    lens_sha256: str,
    domain: str,
    kind: str,
    layers="all",
    identity_atlas: Path | None = None,
    revision="main",
) -> PreparedReplay:
    """Validate source set, lens and immutable model identity before native loading."""
    source, output, lens_path = (
        Path(source).absolute(),
        Path(output).absolute(),
        Path(lens_path).absolute(),
    )
    if output.exists():
        raise FileExistsError(output)
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in (domain, kind)):
        raise ValueError("invalid domain/kind")
    if any(x not in output.name for x in (spec.name.replace("/", "--"), domain, kind)):
        raise ValueError("output name must include model, domain and kind")
    manifest_path = source / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if "status" in manifest and manifest["status"] != "complete":
        raise ValueError("source capture manifest is not complete")
    if manifest["model"] != spec.hf_id:
        raise ValueError("source model differs from requested model")
    if identity_atlas is not None:
        identity_atlas = Path(identity_atlas).absolute()
        atlas_sha = file_sha256(identity_atlas)
        json.loads(identity_atlas.read_bytes())
        if lens_sha256 != manifest["lens_sha256"]:
            raise ValueError("identity requires source lens")
        if layers != "source":
            raise ValueError("identity requires source layers")
    else:
        atlas_sha = None
    identities, labels = [], set()
    for episode in manifest["episodes"]:
        label = episode["label"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label) or label in labels:
            raise ValueError("invalid or duplicate episode label")
        labels.add(label)
        if episode["record"] not in (label + ".jsonl", label + ".jsonl.gz"):
            raise ValueError("record filename must preserve episode label")
        path = source / episode["record"]
        if not path.exists() and path.suffix == ".jsonl":
            path = path.with_suffix(".jsonl.gz")
        events, identity = read_source(path)
        if identity["decompressed_sha256"] != episode["record_sha256"]:
            raise ValueError("source record hash differs from manifest")
        provenance = events[0]["provenance"]
        if provenance["model"] != spec.hf_id:
            raise ValueError("record model differs")
        policies = [
            {key: r[key] for key in ("attention_blocks", "top_k", "audit_modulus", "audit_seed")}
            for r in events
            if r["kind"] == "begin_turn"
        ]
        if any(p != policies[0] for p in policies):
            raise ValueError("capture policy changed within record")
        if any(r["layers"] != manifest["layers"] for r in events if r["kind"] == "begin_turn"):
            raise ValueError("source layers differ from manifest")
        native_vocabs = {r["logits_shape"][2] for r in events if r["kind"] == "forward"}
        identities.append(
            {
                "label": label,
                "identity": identity,
                "capture_policy": policies[0],
                "vocab_size": next(iter(native_vocabs)),
            }
        )
        del events
    if not identities:
        raise ValueError("source set is empty")
    known = {Path(r["identity"]["path"]).name for r in identities}
    actual = {p.name for p in source.iterdir() if p.name.endswith((".jsonl", ".jsonl.gz"))}
    if known != actual:
        raise ValueError("source set differs from manifest")
    snapshot = resolve_snapshot(spec, revision=revision)
    config = json.loads((Path(snapshot["snapshot_path"]) / "config.json").read_bytes())
    config = config.get("text_config", config)
    hidden, count = config["hidden_size"], config["num_hidden_layers"]
    if any(type(layer) is not int or not 1 <= layer <= count for layer in manifest["layers"]):
        raise ValueError("source layers outside snapshot architecture")
    for record in identities:
        if config.get("vocab_size", record["vocab_size"]) != record["vocab_size"]:
            raise ValueError("source vocabulary differs from snapshot")
        if any(block >= count for block in record["capture_policy"]["attention_blocks"]):
            raise ValueError("attention block outside snapshot architecture")
    selected = (
        tuple(range(1, count + 1))
        if layers == "all"
        else tuple(manifest["layers"])
        if layers == "source"
        else tuple(layers)
    )
    if (
        not selected
        or selected != tuple(sorted(set(selected)))
        or any(type(x) is not int or not 1 <= x <= count for x in selected)
    ):
        raise ValueError("invalid replay layers")
    if identity_atlas is None and count not in selected:
        raise ValueError("new readings require final identity layer")
    lens = LensMaps.load(
        lens_path,
        expected_sha256=lens_sha256,
        hidden_size=hidden,
        num_layers=count,
        # Issue 99: the snapshot's dimensions cannot separate two models of the same width.
        identity=LensIdentity(base=spec.base, num_layers=count, training=spec.training),
    )
    readout_layers = set(selected)
    for record in identities:
        for block in record["capture_policy"]["attention_blocks"]:
            readout_layers.update((block, block + 1))
    if not readout_layers - {count} <= lens.maps.keys():
        raise ValueError("lens lacks requested reading or attention readout layers")
    del lens
    if hashlib.sha256(raw).hexdigest() != file_sha256(manifest_path):
        raise ValueError("source manifest changed during preflight")
    return PreparedReplay(
        spec,
        snapshot,
        source,
        manifest,
        hashlib.sha256(raw).hexdigest(),
        tuple(identities),
        output,
        lens_path,
        lens_sha256,
        selected,
        domain,
        kind,
        identity_atlas,
        atlas_sha,
        hidden,
        count,
    )


def replay_record(
    view,
    tokenizer,
    lens,
    events,
    emit,
    *,
    layers,
    array_api=None,
    session_factory=CaptureSession,
    progress=None,
):
    """Drive only CaptureSession's public API, with a fresh model cache each turn."""
    validate_events(events)
    if array_api is None:
        import mlx.core as array_api
    controls = [
        r
        for r in events
        if r["kind"] in {"begin_turn", "forward", "emitted", "end_turn", "source", "head"}
    ]
    cursor = 0
    tokens = 0

    def checked_emit(row):
        nonlocal cursor, tokens
        if row["kind"] == "end_turn" and row["status"] == "aborted":
            # CaptureSession emits this while unwinding a failed forward. Preserve
            # that first exception and let the session finish restoring its state.
            # Do not advance the cursor: an unexplained abort cannot pass completion.
            emit(row)
            return
        if row["kind"] in {"begin_turn", "forward", "emitted", "end_turn", "source", "head"}:
            expected = controls[cursor]
            keys = {
                "begin_turn": ("kind", "turn", "prompt", "prompt_ids", "context"),
                "source": ("kind", "turn", "layer", "position"),
                "head": ("kind", "turn", "block", "head", "position"),
                "forward": (
                    "kind",
                    "turn",
                    "offset",
                    "input_ids",
                    "logits_sha256",
                    "logits_shape",
                    "argmax",
                ),
                "emitted": ("kind", "turn", "position", "token_id"),
                "end_turn": ("kind", "turn", "emitted_count", "forwarded_count", "status"),
            }[row["kind"]]
            for key in keys:
                if row[key] != expected.get(key):
                    raise ValueError(
                        f"native replay {row['kind']}.{key} differs at control {cursor}"
                    )
            cursor += 1
            if row["kind"] == "forward":
                tokens += len(row["input_ids"])
        emit(row)

    session = None
    index = 1
    while index < len(events) - 1:
        begin = events[index]
        if begin["kind"] != "begin_turn":
            raise ValueError("expected begin turn")
        if encode_prompt(tokenizer, begin["prompt"]) != begin["prompt_ids"]:
            raise ValueError("tokenizer prompt IDs changed")
        policy = {
            key: begin[key] for key in ("attention_blocks", "top_k", "audit_modulus", "audit_seed")
        }
        # One session per record preserves original turn IDs; policies must be stable.
        if session is None:
            session = session_factory(
                view,
                LensReadout(view, lens),
                checked_emit,
                layers=layers,
                retain_logits=False,
                **policy,
            )
            initial_policy = policy
        elif policy != initial_policy:
            raise ValueError("capture policy changed within record")
        session.set_context(**begin["context"])
        cache = view.make_cache()
        with session.generation(
            view.model, tokenizer, begin["prompt"], turn_cache=None
        ) as captured:
            index += 1
            while events[index]["kind"] != "end_turn":
                row = events[index]
                if row["kind"] == "forward":
                    logits = captured(array_api.array([row["input_ids"]]), cache=cache)
                    del logits
                elif row["kind"] == "emitted":
                    session.emitted(row["token_id"])
                index += 1
        del cache
        index += 1
        if progress:
            progress({"event": "turn", "turn": begin["turn"], "tokens_done": tokens})
    if cursor != len(controls):
        raise ValueError("native replay omitted controls")
    del session
    return {"forwards": sum(r["kind"] == "forward" for r in controls), "tokens": tokens}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def legacy_summary(records: Path, *, model: str, snapshot: Path) -> dict:
    """Produce atlas.json through the actual legacy summarizer for every replay."""
    import huggingface_hub

    script = PROJECT_ROOT / "scripts/live_lens_atlas.py"
    script_sha = file_sha256(script)
    output = records / "atlas.json"
    if output.exists():
        raise FileExistsError(output)
    record_hashes = {path.name: file_sha256(path) for path in sorted(records.glob("*.jsonl"))}
    manifest_hash = file_sha256(records / "manifest.json")
    original_download, original_argv = huggingface_hub.snapshot_download, sys.argv

    def pinned(hf_id, **kwargs):
        if not kwargs.get("local_files_only"):
            raise ValueError("atlas must remain offline")
        return str(snapshot)

    try:
        huggingface_hub.snapshot_download = pinned
        sys.argv = [str(script), "--records", str(records), "--out", str(output), "--model", model]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        huggingface_hub.snapshot_download, sys.argv = original_download, original_argv
    if file_sha256(script) != script_sha:
        raise ValueError("summary script changed during summarization")
    if record_hashes != {
        path.name: file_sha256(path) for path in records.glob("*.jsonl")
    } or manifest_hash != file_sha256(records / "manifest.json"):
        raise ValueError("replay inputs changed during summarization")
    evidence = {
        "output_atlas_sha256": file_sha256(output),
        "summarizer_sha256": script_sha,
        "output_records": record_hashes,
        "legacy_manifest_sha256": manifest_hash,
        "replay_manifest_sha256": (
            file_sha256(records / "replay-manifest.json")
            if (records / "replay-manifest.json").exists()
            else None
        ),
    }
    write_json(records / "atlas-provenance.json", evidence)
    return evidence


def legacy_identity(
    records: Path,
    source_atlas: Path,
    *,
    model: str,
    snapshot: Path,
    expected_atlas_sha256: str,
) -> dict:
    """Produce the actual legacy summary and compare exact decoded reference objects."""
    if file_sha256(source_atlas) != expected_atlas_sha256:
        raise ValueError("source atlas changed")
    evidence = legacy_summary(records, model=model, snapshot=snapshot)
    if file_sha256(source_atlas) != expected_atlas_sha256:
        raise ValueError("identity reference changed during summarization")
    difference = first_difference(
        json.loads(source_atlas.read_bytes()), json.loads((records / "atlas.json").read_bytes())
    )
    evidence |= {
        "source_atlas_sha256": expected_atlas_sha256,
        "exact_json_equal": difference is None,
        "first_difference": difference,
    }
    write_json(records / "identity.json", evidence)
    if difference:
        raise ValueError(f"legacy identity mismatch: {difference}")
    return evidence


def run_replay(prepared: PreparedReplay, loaded, *, progress=None, allocator_cache=None) -> dict:
    """Publish new records and separate provenance; failed runs never certify completion."""
    started = time.time()
    if (loaded.view.hidden_size, loaded.view.num_layers) != (
        prepared.hidden_size,
        prepared.num_layers,
    ):
        raise ValueError("architecture differs from preflight snapshot")
    if loaded.snapshot != prepared.snapshot or loaded.spec.hf_id != prepared.spec.hf_id:
        raise ValueError("loaded runtime identity differs")
    if loaded.resolved.cache_strategy != "none":
        raise ValueError("replay requires no reuse")
    if file_sha256(prepared.source / "manifest.json") != prepared.manifest_sha256:
        raise ValueError("source manifest changed")
    lens = LensMaps.load(
        prepared.lens_path,
        expected_sha256=prepared.lens_sha256,
        hidden_size=loaded.view.hidden_size,
        num_layers=loaded.view.num_layers,
        identity=LensIdentity(
            base=loaded.spec.base,
            num_layers=loaded.view.num_layers,
            training=loaded.spec.training,
        ),
    )
    prepared.output.mkdir(parents=True, exist_ok=False)
    # Retain byte-identical historical metadata, isolated from new runtime provenance.
    with (prepared.output / "manifest.json").open("xb") as stream:
        stream.write((prepared.source / "manifest.json").read_bytes())
    commit = run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    provenance = {
        "schema_version": 1,
        "snapshot": loaded.snapshot,
        "lens_sha256": lens.sha256,
        "layers": list(prepared.layers),
        "domain": prepared.domain,
        "kind": prepared.kind,
        "sources": list(prepared.records),
        "capture_policy": "per-turn source begin_turn policy",
        "model_run_lock": str(loaded.lock_path),
        "source_manifest_sha256": prepared.manifest_sha256,
        "source_commit": commit,
        "started_at_unix": started,
        "cache_strategy": "none",
        "allocator_cache": allocator_cache,
        "retain_logits": False,
    }
    write_json(prepared.output / "replay-manifest.json", provenance)
    outputs = []
    total_tokens = 0
    for record in prepared.records:
        events, identity = read_source(
            Path(record["identity"]["path"]), expected=record["identity"]
        )
        path = prepared.output / (record["label"] + ".jsonl")

        def episode_progress(event, label=record["label"], before=total_tokens):
            if progress:
                progress(event | {"label": label, "tokens_done": before + event["tokens_done"]})

        with RecordWriter(path, provenance | {"episode": record["label"]}) as writer:
            counts = replay_record(
                loaded.view,
                loaded.tokenizer,
                lens,
                events,
                writer,
                layers=prepared.layers,
                progress=episode_progress,
            )
        read_source(path)
        outputs.append({"label": record["label"], "sha256": file_sha256(path), **counts})
        total_tokens += counts["tokens"]
        del events
    del lens
    if file_sha256(prepared.source / "manifest.json") != prepared.manifest_sha256:
        raise ValueError("source manifest changed during replay")
    if prepared.identity_atlas is not None:
        legacy_identity(
            prepared.output,
            prepared.identity_atlas,
            model=prepared.spec.name,
            snapshot=Path(prepared.snapshot["snapshot_path"]),
            expected_atlas_sha256=prepared.identity_atlas_sha256,
        )
    else:
        legacy_summary(
            prepared.output,
            model=prepared.spec.name,
            snapshot=Path(prepared.snapshot["snapshot_path"]),
        )
    result = {"status": "complete", "outputs": outputs, "elapsed_s": time.time() - started}
    write_json(prepared.output / "replay-complete.json", result)
    return result
