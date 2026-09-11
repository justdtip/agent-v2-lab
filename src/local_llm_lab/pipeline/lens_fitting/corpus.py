"""Immutable token corpora for requirements §3.1 and §10; no model execution.

Indices are zero-based source-step ordinals. Every fifth *one-based* ordinal is
held out before filtering. All hashes use the exact file bytes, except the
manifest's self-digest, which uses canonical JSON without its digest field.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from local_llm_lab.agent_protocol import ActionParseError
from local_llm_lab.models import ModelSpec
from local_llm_lab.pipeline.protocol import (
    DEFAULT_KEEP_LAST,
    SYSTEM_PROMPT,
    assistant_message,
    build_prompt,
    generation_suffix,
    parse_turn,
    strip_thinking,
    tool_message,
    window_messages,
)
from local_llm_lab.pipeline.tasks import FAMILIES, VARIANTS

SPANS = ("system", "task", "observation", "note", "call", "template", "chat")
PROSE_CHUNK_TOKENS = 1024
TRAINING_SPLITS = ("train", "train1", "train2")
SPLIT_RULE = {"every": 5, "index_base": 0, "held_remainder": 4, "before_length_filter": True}
_LEGACY_PILOT_RULE = {
    "allowed_splits": list(TRAINING_SPLITS),
    "excluded_splits": ["test", "valid", "validation", "pilot"],
    "task_id_validation": "training split + registered family + numeric index + variant",
}

# Legacy schema-1 artifacts omitted the scope; their exclusion always applied to task IDs.
PILOT_RULE = {**_LEGACY_PILOT_RULE, "applies_to": "agentic_task_ids"}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_record(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": _sha(path.read_bytes())}


def _verify_file(record: dict) -> bytes:
    data = Path(record["path"]).read_bytes()
    if _sha(data) != record["sha256"]:
        raise ValueError(f"SHA256 hash mismatch: {record['path']}")
    return data


def _sources(paths: list[Path]) -> list[dict]:
    records = [_file_record(path) for path in paths]
    if not records:
        raise ValueError("at least one source is required")
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("duplicate source paths")
    return records


def _training_id(task_id: str) -> str:
    if not isinstance(task_id, str):
        raise ValueError("training task ID must be a string")
    fields = task_id.split("-")
    if (
        len(fields) != 4
        or fields[0] not in TRAINING_SPLITS
        or fields[1] not in FAMILIES
        or not fields[2].isdigit()
        or fields[3] not in VARIANTS
    ):
        raise ValueError(f"non-training or pilot task ID: {task_id}")
    return fields[0]


def _model_ids(declaration: Any) -> list[str]:
    if isinstance(declaration, str):
        return [declaration]
    if not isinstance(declaration, dict):
        raise ValueError("invalid source model declaration")
    ids = []
    if "hf_id" in declaration:
        ids.append(declaration["hf_id"])
    if "spec" in declaration:
        ids.extend(_model_ids(declaration["spec"]))
    if not ids:
        raise ValueError("source model declaration has no hf_id")
    return ids


def _validate_evaluation(payload: dict, model_hf_id: str, seen: set[str]) -> None:
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError("invalid evaluation summary")
    declared = _model_ids(summary["model"]) if "model" in summary else []
    if "split" in summary and summary["split"] not in TRAINING_SPLITS:
        raise ValueError("source split must be training; pilot/test/validation are excluded")
    if "keep_last" in summary and summary["keep_last"] != DEFAULT_KEEP_LAST:
        raise ValueError("source keep_last disagrees with runner window policy")
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("evaluation requires trajectories")
    for trajectory in trajectories:
        task_id = trajectory.get("task_id")
        split = _training_id(task_id)
        if "split" in summary and summary["split"] != split:
            raise ValueError("trajectory and source split disagree")
        if task_id in seen:
            raise ValueError(f"duplicate trajectory identity: {task_id}")
        seen.add(task_id)
        ids = declared + (_model_ids(trajectory["model"]) if "model" in trajectory else [])
        if not ids or any(hf_id != model_hf_id for hf_id in ids):
            raise ValueError("missing or mixed source model hf_id")
        if not isinstance(trajectory.get("prompt"), str):
            raise ValueError("trajectory requires a prompt string")
        if not isinstance(trajectory.get("steps"), list):
            raise ValueError("trajectory requires steps")


def _tokenizer_identity(tokenizer: Any, spec: ModelSpec, files: list[Path]) -> dict:
    records = _sources(files)
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError("tokenizer chat template identity is required")
    return {
        "model_hf_id": spec.hf_id,
        "files": records,
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "chat_template_sha256": _sha(_json_bytes(template)),
        "template_kwargs": spec.chat.template_kwargs,
        "generation_suffix": generation_suffix(spec),
        "bos_policy": "add specials unless text already starts with tokenizer.bos_token",
        "bos_token": getattr(tokenizer, "bos_token", None),
    }


def _encode(tokenizer: Any, text: str) -> tuple[list[int], list[list[int]]]:
    bos = getattr(tokenizer, "bos_token", None)
    encoded = tokenizer(
        text, add_special_tokens=not (bos and text.startswith(bos)), return_offsets_mapping=True
    )
    ids = list(encoded["input_ids"])
    offsets = [list(pair) for pair in encoded.get("offset_mapping", [])]
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("token IDs must be nonempty nonnegative integers")
    _validate_offsets(offsets, text_length=len(text), token_count=len(ids))
    return ids, offsets


def _validate_offsets(offsets: Any, *, text_length: int, token_count: int) -> None:
    """Enforce the same offset format during tokenization and immutable-corpus reads."""
    if not isinstance(offsets, list) or len(offsets) != token_count:
        raise ValueError("token alignment requires one offset per token")
    previous = (0, 0)
    for pair in offsets:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(n) is not int for n in pair)
            or not 0 <= pair[0] <= pair[1] <= text_length
        ):
            raise ValueError("invalid token alignment offsets")
        if pair != [0, 0]:
            if pair[0] == pair[1] or pair[0] < previous[0] or pair[1] < previous[1]:
                raise ValueError("nonmonotonic token alignment offsets")
            previous = tuple(pair)
    if text_length and previous == (0, 0):
        raise ValueError("token alignment has no nonempty offsets")


def _content_ranges(prompt: str, messages: list[dict], raw: str) -> list[tuple[int, int, str]]:
    ranges = []
    cursor = 0
    for message in messages:
        content = message["content"]
        if not content:
            continue
        start = prompt.find(content, cursor)
        if start < 0:
            raise ValueError("cannot establish message/template span alignment")
        end = start + len(content)
        role = message["role"]
        if role == "assistant":
            ranges.extend(_assistant_ranges(content, start))
        else:
            label = {"system": "system", "user": "task", "tool": "observation"}[role]
            ranges.append((start, end, label))
        cursor = end
    ranges.extend(_assistant_ranges(raw, len(prompt)))
    return ranges


def _assistant_ranges(content: str, start: int) -> list[tuple[int, int, str]]:
    # The atlas uses the fenced call as the note/call boundary. Raw completions
    # retain their own spacing/thinking; only prior turns are canonicalised.
    match = re.search(r"```(?:[A-Za-z]*)[ \t]*\r?\n?", content)
    fence = match.start() if match else len(content)
    return [(start, start + fence, "note"), (start + fence, start + len(content), "call")]


def _labels(offsets: list[list[int]], ranges: list[tuple[int, int, str]]) -> list[str]:
    return [
        next((label for a, b, label in ranges if a <= s < e <= b), "template") for s, e in offsets
    ]


def _prompt_count(offsets: list[list[int]], boundary: int) -> int:
    # A token crossing the raw boundary belongs to the completion, labelled template.
    # Leading zero-width BOS tokens count as prompt; no independently encoded prefix.
    for index, (start, end) in enumerate(offsets):
        if end > boundary or (start >= boundary and end > start):
            return index
    return len(offsets)


def _split(index: int) -> str:
    return "held" if index % SPLIT_RULE["every"] == SPLIT_RULE["held_remainder"] else "fit"


def _counts(rows: list[dict]) -> dict:
    counts = {
        split: {"sequences": 0, "tokens": 0, "spans": dict.fromkeys(SPANS, 0)}
        for split in ("fit", "held")
    }
    for row in rows:
        group = counts[row["split"]]
        group["sequences"] += 1
        group["tokens"] += len(row["ids"])
        for span, count in Counter(row["spans"]).items():
            group["spans"][span] += count
    return counts


def _write_corpus(manifest_path: Path, rows: list[dict], metadata: dict) -> dict:
    manifest_path = Path(manifest_path).absolute()
    sequences_path = manifest_path.with_suffix(".jsonl")
    if manifest_path == sequences_path:
        raise ValueError("manifest must have an extension other than .jsonl")
    if manifest_path.exists() or sequences_path.exists():
        raise FileExistsError("immutable corpus output already exists")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(_json_bytes(row) + b"\n" for row in rows)
    metadata.update(
        schema_version=1,
        split_rule=dict(SPLIT_RULE),
        pilot_exclusion=dict(PILOT_RULE),
        sequences={"path": str(sequences_path), "sha256": _sha(data)},
        counts=_counts(rows),
    )
    # Recheck source/asset bytes immediately before publishing their bound outputs.
    for record in [*metadata["sources"], *metadata["tokenizer"]["files"]]:
        _verify_file(record)
    if metadata.get("download_descriptor"):
        _verify_file(metadata["download_descriptor"])
    metadata["manifest_sha256"] = _sha(_json_bytes(metadata))
    # Exclusive opens protect against racing builders. A crash can leave an orphan
    # JSONL, which deliberately prevents silent reuse or overwrite.
    with sequences_path.open("xb") as stream:
        stream.write(data)
    with manifest_path.open("xb") as stream:
        stream.write(_json_bytes(metadata) + b"\n")
    return metadata


def build_agentic_corpus(
    sources: list[Path],
    tokenizer: Any,
    spec: ModelSpec,
    manifest_path: Path,
    *,
    max_tokens: int,
    tokenizer_files: list[Path],
) -> dict:
    """Render every training trajectory step, then publish its immutable corpus."""
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    records = _sources(sources)
    identity = _tokenizer_identity(tokenizer, spec, tokenizer_files)
    seen: set[str] = set()
    payloads = [json.loads(_verify_file(record)) for record in records]
    for payload in payloads:
        _validate_evaluation(payload, spec.hf_id, seen)
    rows, dropped = [], []
    index = 0
    for source, payload in zip(records, payloads, strict=True):
        source["task_ids"] = [t["task_id"] for t in payload["trajectories"]]
        source["source_steps"] = 0
        for trajectory in payload["trajectories"]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": trajectory["prompt"]},
            ]
            for step_index, step in enumerate(trajectory["steps"]):
                if step.get("index", step_index) != step_index:
                    raise ValueError("recorded step index does not match trajectory order")
                raw = step.get("raw")
                if not isinstance(raw, str):
                    raise ValueError("every source step must contain its raw completion")
                prompt = build_prompt(
                    tokenizer, messages, spec=spec, keep_last=DEFAULT_KEEP_LAST, generation=True
                )
                text = prompt + raw
                ids, offsets = _encode(tokenizer, text)
                ranges = _content_ranges(prompt, window_messages(messages, DEFAULT_KEEP_LAST), raw)
                base = {
                    "index": index,
                    "source": source["path"],
                    "task_id": trajectory["task_id"],
                    "step_index": step_index,
                    "split": _split(index),
                    "domain": "agentic",
                }
                if len(ids) > max_tokens:
                    dropped.append({**base, "tokens": len(ids), "reason": "over_max_tokens"})
                else:
                    rows.append(
                        {
                            **base,
                            "ids": ids,
                            "spans": _labels(offsets, ranges),
                            "n_prompt": _prompt_count(offsets, len(prompt)),
                            "prompt": prompt,
                            "text": text,
                            "offsets": offsets,
                        }
                    )
                index += 1
                source["source_steps"] += 1
                _, action_text = strip_thinking(raw)
                try:
                    turn = parse_turn(action_text)
                except ActionParseError:
                    break
                if step.get("parse_error"):
                    raise ValueError("recorded parse failure contradicts raw completion")
                action = {"name": turn.action.name, "arguments": turn.action.arguments}
                if step.get("action") != action or step.get("thought") != turn.thought:
                    raise ValueError("recorded action/note contradicts raw completion")
                if turn.action.name == "finish":
                    break
                observation = step.get("observation")
                if not isinstance(observation, str):
                    raise ValueError("valid nonterminal step requires an observation")
                messages += [
                    assistant_message(turn.thought, turn.action),
                    tool_message(turn.action.name, observation),
                ]
    return _write_corpus(
        manifest_path,
        rows,
        {
            "domain": "agentic",
            "model_hf_id": spec.hf_id,
            "sources": records,
            "tokenizer": identity,
            "source_sequence_count": index,
            "max_tokens": max_tokens,
            "keep_last": DEFAULT_KEEP_LAST,
            "system_prompt_sha256": _sha(SYSTEM_PROMPT.encode()),
            "dropped_rows": dropped,
            "discarded_trailing_tokens": 0,
        },
    )


def build_prose_corpus(
    sources: list[Path],
    tokenizer: Any,
    spec: ModelSpec,
    manifest_path: Path,
    *,
    tokenizer_files: list[Path],
    download_descriptor: Path | None = None,
    chunk_tokens: int = PROSE_CHUNK_TOKENS,
) -> dict:
    """Chunk each text source once into nonoverlapping fixed-size token windows."""
    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError("prose chunk_tokens must be a positive integer")
    records = _sources(sources)
    identity = _tokenizer_identity(tokenizer, spec, tokenizer_files)
    rows, trailing = [], 0
    for source in records:
        text = _verify_file(source).decode("utf-8")
        ids, offsets = _encode(tokenizer, text)
        tail = len(ids) % chunk_tokens
        source.update(tokens=len(ids), discarded_trailing_tokens=tail)
        trailing += tail
        for step, start in enumerate(range(0, len(ids) - chunk_tokens + 1, chunk_tokens)):
            end = start + chunk_tokens
            chunk_offsets = offsets[start:end]
            rows.append(
                {
                    "index": len(rows),
                    "source": source["path"],
                    "step_index": step,
                    "split": _split(len(rows)),
                    "domain": "prose",
                    "n_prompt": 0,
                    "ids": ids[start:end],
                    "spans": _labels(chunk_offsets, [(0, len(text), "chat")]),
                    "token_start": start,
                }
            )
    descriptor = _file_record(download_descriptor) if download_descriptor else None
    if descriptor:
        download = json.loads(_verify_file(descriptor))
        _validate_download(download)
        if download["text"] not in [{k: r[k] for k in ("path", "sha256")} for r in records]:
            raise ValueError("download descriptor does not identify a prose source")
    return _write_corpus(
        manifest_path,
        rows,
        {
            "domain": "prose",
            "model_hf_id": spec.hf_id,
            "sources": records,
            "tokenizer": identity,
            "source_sequence_count": len(rows),
            "chunk_tokens": chunk_tokens,
            "dropped_rows": [],
            "discarded_trailing_tokens": trailing,
            "download_descriptor": descriptor,
        },
    )


def read_corpus(manifest_path: Path) -> list[dict]:
    """Validate all bound bytes and row invariants before exposing any sequence."""
    manifest = json.loads(Path(manifest_path).read_bytes())
    if manifest.get("source_kind") == "rendered_workspace_v1":
        from .workspace_corpus import read_workspace_corpus

        return read_workspace_corpus(manifest_path)
    digest = manifest.pop("manifest_sha256", None)
    if digest != _sha(_json_bytes(manifest)):
        raise ValueError("manifest hash mismatch")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("domain") not in ("agentic", "prose")
        or manifest.get("split_rule") != SPLIT_RULE
        or manifest.get("pilot_exclusion") not in (PILOT_RULE, _LEGACY_PILOT_RULE)
    ):
        raise ValueError("unsupported corpus schema or split/exclusion rules")
    sources = manifest["sources"]
    paths = {source["path"] for source in sources}
    if not paths or len(paths) != len(sources):
        raise ValueError("duplicate or empty sources")
    seen: set[str] = set()
    expected_identities = []
    trailing = 0
    chunk_tokens = manifest.get("chunk_tokens")
    if manifest["domain"] == "prose" and (type(chunk_tokens) is not int or chunk_tokens <= 0):
        raise ValueError("prose chunk_tokens must be a positive integer")
    for source in sources:
        data = _verify_file(source)
        if manifest["domain"] == "agentic":
            payload = json.loads(data)
            _validate_evaluation(payload, manifest["model_hf_id"], seen)
            if source["task_ids"] != [t["task_id"] for t in payload["trajectories"]]:
                raise ValueError("source trajectory identity mismatch")
            before = len(expected_identities)
            for trajectory in payload["trajectories"]:
                for step_index, step in enumerate(trajectory["steps"]):
                    expected_identities.append((source["path"], trajectory["task_id"], step_index))
                    _, action_text = strip_thinking(step["raw"])
                    try:
                        turn = parse_turn(action_text)
                    except ActionParseError:
                        break
                    if turn.action.name == "finish":
                        break
            if source["source_steps"] != len(expected_identities) - before:
                raise ValueError("source step count mismatch")
        else:
            tokens, tail = source.get("tokens"), source.get("discarded_trailing_tokens")
            if (
                type(tokens) is not int
                or tokens < 0
                or type(tail) is not int
                or not 0 <= tail < chunk_tokens
                or tokens % chunk_tokens != tail
            ):
                raise ValueError("invalid prose source token/tail counts")
            trailing += tail
            expected_identities.extend(
                (source["path"], step, step * chunk_tokens)
                for step in range(tokens // chunk_tokens)
            )
    if manifest["domain"] == "prose" and (
        type(manifest.get("discarded_trailing_tokens")) is not int
        or manifest["discarded_trailing_tokens"] != trailing
    ):
        raise ValueError("prose discarded tail count mismatch")
    for asset in manifest["tokenizer"]["files"]:
        _verify_file(asset)
    if manifest.get("download_descriptor"):
        _validate_download(json.loads(_verify_file(manifest["download_descriptor"])))
    rows = [json.loads(line) for line in _verify_file(manifest["sequences"]).splitlines()]
    previous = -1
    for row in rows:
        index = row["index"]
        if (
            type(index) is not int
            or index <= previous
            or row["split"] != _split(index)
            or row["domain"] != manifest["domain"]
            or row["source"] not in paths
            or type(row["step_index"]) is not int
            or row["step_index"] < 0
        ):
            raise ValueError("invalid sequence identity/order/split")
        previous = index
        ids, spans = row["ids"], row["spans"]
        if (
            not isinstance(ids, list)
            or not ids
            or len(ids) != len(spans)
            or any(type(token) is not int or token < 0 for token in ids)
            or any(span not in SPANS for span in spans)
            or type(row["n_prompt"]) is not int
            or not 0 <= row["n_prompt"] <= len(ids)
        ):
            raise ValueError("invalid token/span row invariants")
        if row["domain"] == "agentic":
            _training_id(row["task_id"])
            source = next(s for s in sources if s["path"] == row["source"])
            if row["task_id"] not in source["task_ids"] or len(ids) > manifest["max_tokens"]:
                raise ValueError("invalid agentic source/length")
            _validate_offsets(row["offsets"], text_length=len(row["text"]), token_count=len(ids))
            if not row["text"].startswith(row["prompt"]) or row["n_prompt"] != _prompt_count(
                row["offsets"], len(row["prompt"])
            ):
                raise ValueError("invalid prompt alignment")
        elif (
            len(ids) != chunk_tokens
            or row["n_prompt"] != 0
            or set(spans) - {"chat", "template"}
            or type(row["token_start"]) is not int
            or row["token_start"] != row["step_index"] * chunk_tokens
        ):
            raise ValueError("invalid prose chunk")
    dropped = manifest["dropped_rows"]
    for row in dropped:
        if (
            row["source"] not in paths
            or row["split"] != _split(row["index"])
            or manifest["domain"] != "agentic"
            or row["tokens"] <= manifest["max_tokens"]
        ):
            raise ValueError("invalid dropped row")
        _training_id(row["task_id"])
    if (
        type(manifest["source_sequence_count"]) is not int
        or len(expected_identities) != manifest["source_sequence_count"]
    ):
        raise ValueError("source sequence count mismatch")
    for row in [*rows, *dropped]:
        identity = (
            (row["source"], row["task_id"], row["step_index"])
            if manifest["domain"] == "agentic"
            else (row["source"], row["step_index"], row["token_start"])
        )
        index = row["index"]
        if not 0 <= index < len(expected_identities) or identity != expected_identities[index]:
            raise ValueError("source step identity mismatch")
    indices = [row["index"] for row in [*rows, *dropped]]
    if sorted(indices) != list(range(manifest["source_sequence_count"])):
        raise ValueError("source-step coverage mismatch")
    if _counts(rows) != manifest["counts"]:
        raise ValueError("corpus count mismatch")
    return rows


DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "validation"
_VALIDATION_SHARD = re.compile(re.escape(DATASET_CONFIG) + r"/validation-\d+-of-\d+\.parquet")
TOKENIZER_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "config.json",
    "chat_template.jinja",
    "chat_templates/*.jinja",
)


def _validate_download(record: dict) -> None:
    content = {key: value for key, value in record.items() if key != "descriptor_sha256"}
    if record.get("descriptor_sha256") != _sha(_json_bytes(content)):
        raise ValueError("download descriptor hash mismatch")
    if (
        record.get("schema_version") != 1
        or record.get("dataset") != DATASET_ID
        or record.get("config") != DATASET_CONFIG
        or record.get("split") != DATASET_SPLIT
        or not re.fullmatch(r"[0-9a-f]{40,64}", record.get("revision", ""))
    ):
        raise ValueError("unauthorised dataset configuration/split or unresolved revision")
    files = record.get("files", [])
    if (
        not files
        or len({f["repo_path"] for f in files}) != len(files)
        or any(not _VALIDATION_SHARD.fullmatch(f["repo_path"]) for f in files)
    ):
        raise ValueError("download descriptor includes unauthorised split/configuration files")
    _verify_file(record["text"])


def download_prose(data_dir: Path, *, revision: str = "main") -> tuple[Path, Path]:
    """Download only authorised WikiText validation, resolving and recording its commit.

    An existing descriptor is reused offline only when its fixed identity, requested
    revision and exact text hash verify. Partial or corrupt outputs require a fresh
    caller-selected directory; this function never silently repairs/overwrites them.
    """
    data_dir = Path(data_dir).absolute()
    text_path = data_dir / "wikitext-103-raw-v1.validation.txt"
    descriptor_path = data_dir / "wikitext-103-raw-v1.validation.download.json"
    if descriptor_path.exists():
        record = json.loads(descriptor_path.read_bytes())
        _validate_download(record)
        if record["requested_revision"] != revision and record["revision"] != revision:
            raise ValueError("existing download revision differs from requested revision")
        if Path(record["text"]["path"]) != text_path:
            raise ValueError("existing download text is outside the selected data directory")
        return text_path, descriptor_path
    if text_path.exists():
        raise FileExistsError("download text exists without a verified descriptor")
    from huggingface_hub import HfApi, hf_hub_download
    from pyarrow import parquet

    hub = HfApi()
    resolved = hub.dataset_info(DATASET_ID, revision=revision).sha
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40,64}", resolved):
        raise ValueError("dataset revision must resolve to an immutable commit")
    # load_dataset(split=...) can prepare *all* splits before selecting one.
    # Restrict the actual file-download boundary, then read only local Parquet.
    filenames = sorted(
        name
        for name in hub.list_repo_files(DATASET_ID, repo_type="dataset", revision=resolved)
        if _VALIDATION_SHARD.fullmatch(name)
    )
    if not filenames:
        raise ValueError("pinned dataset revision contains no authorised validation shards")
    pieces, downloaded_files = [], []
    for name in filenames:
        path = Path(hf_hub_download(DATASET_ID, name, repo_type="dataset", revision=resolved))
        downloaded_files.append({"repo_path": name, "sha256": _sha(path.read_bytes())})
        texts = parquet.read_table(path, columns=["text"]).column("text").to_pylist()
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("downloaded validation row has no text")
            pieces.append(text + "\n")
    data = "".join(pieces).encode("utf-8")
    record = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "requested_revision": revision,
        "revision": resolved,
        "rows": len(pieces),
        "files": downloaded_files,
        "text": {"path": str(text_path), "sha256": _sha(data)},
    }
    record["descriptor_sha256"] = _sha(_json_bytes(record))
    data_dir.mkdir(parents=True, exist_ok=True)
    with text_path.open("xb") as stream:
        stream.write(data)
    with descriptor_path.open("xb") as stream:
        stream.write(_json_bytes(record) + b"\n")
    return text_path, descriptor_path


def load_corpus_tokenizer(
    spec: ModelSpec,
    *,
    local_files_only: bool = False,
) -> tuple[Any, list[Path]]:
    """Load only fast tokenizer/config assets, with remote code and weights excluded."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    directory = Path(spec.hf_id).expanduser()
    if directory.is_dir():
        directory = directory.resolve()
    else:
        directory = Path(
            snapshot_download(
                spec.hf_id,
                allow_patterns=list(TOKENIZER_PATTERNS),
                local_files_only=local_files_only,
            )
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    files = sorted(
        {
            path
            for pattern in TOKENIZER_PATTERNS
            for path in directory.glob(pattern)
            if path.is_file()
        }
    )
    if not files:
        raise ValueError("tokenizer snapshot contains no tokenizer/config assets")
    return tokenizer, files
