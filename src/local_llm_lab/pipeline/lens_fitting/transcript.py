"""Exact rollout input ledger and frozen native-replay corpus; no model imports.

Replay uses fresh bounded windows, not generation's cache partition or position origin.
No generation activations are retained and residual equivalence is not claimed.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from local_llm_lab.agent_protocol import ActionParseError
from local_llm_lab.forward import ForwardLedger, encode_prompt
from local_llm_lab.pipeline.lens_fitting.corpus import (
    _assistant_ranges,
    _labels,
    _training_id,
    _validate_offsets,
)
from local_llm_lab.pipeline.protocol import _as_declared_roles, parse_turn, system_prompt

FORMAT = "transcript-native-replay-v1"
SPANS = ("system", "task", "observation", "note", "call", "template")
REPLAY_CAVEAT = (
    "Native replay of captured token IDs in fresh 2048-token windows; generation forward "
    "partitions, longer context, and absolute position origins are not preserved. "
    "No partition-identical live residual or generation-activation equivalence is claimed."
)


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked_bytes(record):
    data = Path(record["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("transcript bound file hash mismatch")
    return data


class TranscriptWriter:
    """Exclusive hash-chain writer. A complete footer is mandatory for corpus use."""

    def __init__(self, path, provenance):
        self.stream = Path(path).open("x", encoding="utf-8")  # noqa: SIM115 - closed by this context manager
        self.previous, self.count = "0" * 64, 0
        self({"kind": "manifest", "format": FORMAT, "provenance": provenance})

    def __call__(self, event):
        row = {"event": event, "previous": self.previous, "sequence": self.count}
        row["sha256"] = digest(row)
        self.stream.write(encoded(row).decode() + "\n")
        self.stream.flush()
        self.previous, self.count = row["sha256"], self.count + 1

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            self({"kind": "end_record", "status": "complete" if error_type is None else "aborted"})
        finally:
            self.stream.close()


def read_transcript(path):
    previous, events = "0" * 64, []
    for number, line in enumerate(Path(path).read_bytes().splitlines()):
        row = json.loads(line)
        sha = row.pop("sha256")
        if row.get("previous") != previous or row.get("sequence") != number or digest(row) != sha:
            raise ValueError("transcript hash chain mismatch")
        events.append(row["event"])
        previous = sha
    if (
        len(events) < 3
        or events[0].get("kind") != "manifest"
        or events[0].get("format") != FORMAT
        or events[-1] != {"kind": "end_record", "status": "complete"}
    ):
        raise ValueError("incomplete transcript record")
    return events


class _LedgerModel:
    def __init__(self, owner, model):
        self.owner, self.model = owner, model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, ids, **kwargs):
        if ids.ndim != 2 or ids.shape[0] != 1 or kwargs.get("input_embeddings") is not None:
            raise ValueError("transcript capture requires one token-ID sequence")
        tokens = ids[0].tolist()
        ledger = self.owner.ledger
        offset = ledger.offset
        ledger.validate(offset, tokens)
        result = self.model(ids, **kwargs)
        # Force the returned lazy forward before certifying a successful input pass.
        self.owner.materialize(result)
        ledger.record(offset, tokens)
        self.owner.write("forward", offset=offset, ids=tokens)
        return result


def _materialize(result):
    import mlx.core as mx

    mx.eval(result)


class TranscriptCapture:
    """Runner capture protocol, observing native inputs only; no residual readout."""

    def __init__(self, emit, *, materialize=None):
        self.emit = emit
        self.materialize = _materialize if materialize is None else materialize
        self.context, self.turn, self.active = {}, -1, False

    def set_context(self, **metadata):
        if self.active:
            raise ValueError("cannot replace context during capture")
        self.context = json.loads(encoded(metadata))

    def write(self, kind, **fields):
        self.emit({"kind": kind, "turn": self.turn, **fields})

    @contextmanager
    def generation(self, model, tokenizer, prompt, *, turn_cache):
        if self.active or turn_cache is not None:
            raise ValueError("transcript capture requires one active turn and no cache reuse")
        self.ledger = ForwardLedger(encode_prompt(tokenizer, prompt))
        self.tokenizer = tokenizer
        self.turn += 1
        self.active = True
        self.write(
            "begin_turn", prompt=prompt, prompt_ids=self.ledger.prompt_ids, context=self.context
        )
        success = False
        try:
            yield _LedgerModel(self, model)
            success = True
        finally:
            self.write(
                "end_turn",
                forwarded_count=self.ledger.offset,
                emitted_count=len(self.ledger.generated),
                emitted_text=self.tokenizer.decode(self.ledger.generated),
                status="complete" if success else "aborted",
            )
            self.active = False

    def emitted(self, token):
        if not self.active:
            raise ValueError("emission outside capture")
        self.ledger.emitted(token)
        self.write("emitted", token=token)


def _turns(events):
    current, previous_turn = None, -1
    task_steps = {}
    for event in events[1:-1]:
        kind = event.get("kind")
        if kind == "begin_turn":
            if current is not None or event.get("turn") != previous_turn + 1:
                raise ValueError("overlapping or non-contiguous transcript turns")
            context = event["context"]
            task_id, step = context["task_id"], context["step"]
            _training_id(task_id)
            if type(step) is not int or step != task_steps.get(task_id, -1) + 1:
                raise ValueError("non-contiguous source task steps")
            task_steps[task_id] = step
            current = (event, ForwardLedger(event["prompt_ids"]))
            previous_turn = event["turn"]
        elif current is None or event.get("turn") != previous_turn:
            raise ValueError("event outside its transcript turn")
        elif kind == "forward":
            current[1].record(event["offset"], event["ids"])
        elif kind == "emitted":
            current[1].emitted(event["token"])
        elif kind == "end_turn":
            begin, ledger = current
            if (
                event.get("status") != "complete"
                or event["forwarded_count"] != ledger.offset
                or event["emitted_count"] != len(ledger.generated)
                or ledger.offset < len(ledger.prompt_ids)
            ):
                raise ValueError("incomplete transcript forward ledger")
            if not isinstance(event.get("emitted_text"), str):
                raise ValueError("source turn requires decoded actual emissions")
            begin = {**begin, "emitted_text": event["emitted_text"]}
            yield begin, ledger
            current = None
        else:
            raise ValueError("unknown transcript event")
    if current is not None:
        raise ValueError("unclosed transcript turn")


def _prompt_alignment(tokenizer, spec, begin):
    text, ids = begin["prompt"], begin["prompt_ids"]
    bos = getattr(tokenizer, "bos_token", None)
    alignment = tokenizer(
        text, add_special_tokens=not (bos and text.startswith(bos)), return_offsets_mapping=True
    )
    if list(alignment["input_ids"]) != ids:
        raise ValueError("prompt offset tokenizer disagrees with captured IDs")
    offsets = [list(pair) for pair in alignment["offset_mapping"]]
    _validate_offsets(offsets, text_length=len(text), token_count=len(ids))
    labels, owned, audit = _prompt_semantics(spec, begin, offsets)
    return labels, owned, audit, offsets


def _prompt_semantics(spec, begin, offsets):
    text, ids = begin["prompt"], begin["prompt_ids"]
    _validate_offsets(offsets, text_length=len(text), token_count=len(ids))
    semantic = begin["context"]["messages"]
    if (
        len(semantic) < 2
        or semantic[0] != {"role": "system", "content": system_prompt(spec=spec)}
        or semantic[1].get("role") != "user"
    ):
        raise ValueError("capture lacks the model-specific system observation convention")
    rendered = _as_declared_roles(semantic, spec)
    ranges, owned_ranges, audit, cursor = [], [], [], 0
    first = begin["context"]["step"] == 0
    if not semantic or (not first and semantic[-1].get("role") != "tool"):
        raise ValueError("source context lacks the newly introduced observation")
    for number, (message, displayed) in enumerate(zip(semantic, rendered, strict=True)):
        content = displayed["content"]
        # Gemma's template applies trim. Prefer the exact content; only trim edges,
        # retaining the explicit character offset audit rather than normalizing IDs.
        start = text.find(content, cursor) if content else -1
        trim_left = trim_right = 0
        if start < 0:
            stripped = content.strip()
            trim_left, trim_right = (
                len(content) - len(content.lstrip()),
                len(content) - len(content.rstrip()),
            )
            content = stripped
            start = text.find(content, cursor) if content else cursor
        if start < 0:
            raise ValueError("cannot align rendered message to captured prompt")
        end, role = start + len(content), message["role"]
        if role == "assistant":
            ranges.extend(_assistant_ranges(content, start))
        else:
            label = {"system": "system", "user": "task", "tool": "observation"}[role]
            ranges.append((start, end, label))
        owned = first or (number == len(semantic) - 1 and role == "tool")
        if owned:
            owned_ranges.append((start, end))
        audit.append(
            {
                "message": number,
                "role": role,
                "start": start,
                "end": end,
                "trim_left": trim_left,
                "trim_right": trim_right,
                "owned": owned,
            }
        )
        cursor = end
    labels = _labels(offsets, ranges)
    # Initial template positions belong to the initial prompt, once. In later prompts
    # only the new observation's content and declared wrapper are owned.
    owned = (
        list(range(len(ids)))
        if first
        else [
            i for i, (s, e) in enumerate(offsets) if any(s < b and a < e for a, b in owned_ranges)
        ]
    )
    return labels, owned, audit


def _generated_labels(tokenizer, tokens):
    # Generation is bounded by evaluation max_tokens (200). Prefix decoding is confined
    # to the continuation, never performed across the growing prompt history.
    text, offsets = "", []
    for end in range(1, len(tokens) + 1):
        decoded = tokenizer.decode(tokens[:end])
        if not decoded.startswith(text):
            raise ValueError(
                "continuation decode is not prefix-stable; explicit alignment required"
            )
        offsets.append([len(text), len(decoded)])
        text = decoded
    return _labels(offsets, _assistant_ranges(text, 0)), text, offsets


def _counts(rows):
    counts = {
        split: {
            "sequences": 0,
            "tokens": 0,
            "positions": 0,
            "spans": dict.fromkeys(SPANS, 0),
            "generated": 0,
            "positions_beyond_1024": 0,
            "full_2048_rows": 0,
            "short_rows": 0,
        }
        for split in ("fit", "held")
    }
    for row in rows:
        c = counts[row["split"]]
        c["sequences"] += 1
        c["tokens"] += len(row["ids"])
        c["full_2048_rows" if len(row["ids"]) == 2048 else "short_rows"] += 1
        for i in row["score_positions"]:
            c["positions"] += 1
            c["spans"][row["spans"][i]] += 1
            c["generated"] += int(row["generated_mask"][i])
            c["positions_beyond_1024"] += int(i >= 1024)
    return counts


def _parsed_action(text):
    try:
        action = parse_turn(text).action
    except ActionParseError as error:
        return {"canonical_action": None, "action_parse_error": str(error)}
    return {
        "canonical_action": {"name": action.name, "arguments": action.arguments},
        "action_parse_error": None,
    }


def transcript_acceptance(rows, turns):
    """Amended Task 1 concentration gate, recomputable without a model forward.

    Invalid actions break runs; they are counted explicitly, never interpreted as
    identical null actions. No episode or repeating turn is silently removed.
    """
    members, runs = set(), []
    by_task = {}
    for index, turn in enumerate(turns):
        by_task.setdefault(turn["task_id"], []).append(index)
    for indices in by_task.values():
        current = []

        def flush(group):
            if len(group) >= 2:
                members.update(group)
                runs.append(list(group))

        for index in indices:
            action = turns[index]["canonical_action"]
            if (
                current
                and action is not None
                and encoded(action) == encoded(turns[current[-1]]["canonical_action"])
                and turns[index]["step"] == turns[current[-1]]["step"] + 1
            ):
                current.append(index)
            else:
                flush(current)
                current = [index] if action is not None else []
        flush(current)
    previous = {(turn["task_id"], turn["step"]): i for i, turn in enumerate(turns)}
    concentration = {}
    for split in ("fit", "held", "combined"):
        episodes, total, repeated, invalid_positions = {}, 0, 0, 0
        included_turns = set()
        for row in rows:
            if split != "combined" and row["split"] != split:
                continue
            t = row["turn_index"]
            turn = turns[t]
            included_turns.add(t)
            count = len(row["score_positions"])
            total += count
            episodes[turn["task_id"]] = episodes.get(turn["task_id"], 0) + count
            predecessor = previous.get((turn["task_id"], turn["step"] - 1))
            for position in row["score_positions"]:
                generated = row["generated_mask"][position]
                if (
                    generated
                    and row["spans"][position] in ("note", "call")
                    and t in members
                    or not generated
                    and turn["step"] > 0
                    and predecessor in members
                ):
                    repeated += 1
                if generated and turn["canonical_action"] is None:
                    invalid_positions += 1
        largest = max(episodes, key=lambda task: (episodes[task], task)) if episodes else None
        largest_n = episodes.get(largest, 0)
        concentration[split] = {
            "positions": total,
            "episode_positions": episodes,
            "largest_episode": largest,
            "largest_episode_positions": largest_n,
            "largest_episode_share": largest_n / total if total else None,
            "repeated_run_positions": repeated,
            "repeated_run_share": repeated / total if total else None,
            "invalid_action_turns": sum(
                turns[i]["canonical_action"] is None for i in included_turns
            ),
            "invalid_action_generated_positions": invalid_positions,
        }
    fit = concentration["fit"]
    reasons = []
    if not fit["positions"] or not concentration["held"]["positions"]:
        reasons.append("both fit and held positions are required")
    if fit["positions"] and 3 * fit["largest_episode_positions"] > fit["positions"]:
        reasons.append("single largest episode exceeds one third of fitted positions")
    if fit["positions"] and 3 * fit["repeated_run_positions"] > fit["positions"]:
        reasons.append("identical repeated-call runs exceed one third of fitted positions")
    counts = _counts(rows)
    if not counts["fit"]["positions_beyond_1024"]:
        reasons.append("no fitted position actually exceeds the 1024-token sliding window")
    if not any(
        10 * counts["fit"]["spans"][span] > fit["positions"]
        for span in ("observation", "note", "call")
    ):
        reasons.append(
            "no agentic note/call/observation span exceeds ten percent of fitted positions"
        )
    return {
        "status": "ruling_required" if reasons else "passed",
        "reasons": reasons,
        "concentration": concentration,
        "repeated_runs": runs,
        "repeated_run_rule": (
            "consecutive canonical(name,arguments) equal calls; length>=2 includes first member; "
            "member-turn generated note/call tokens and associated next-prompt newly owned "
            "observation tokens including wrappers; initial system/task scaffold excluded; "
            "invalid actions break runs and remain in position denominators"
        ),
    }


def build_transcript_corpus(
    sources, tokenizer, spec, manifest_path, *, tokenizer_identity, model_identity, max_tokens=2048
):
    """Freeze exact consumed IDs and score first appearances once; no native execution."""
    if max_tokens != 2048:
        raise ValueError("transcript fitting context must be 2048 tokens")
    _validate_identity(model_identity, tokenizer_identity)
    if model_identity != {
        "base": spec.base,
        "training": spec.training,
        "num_layers": model_identity["num_layers"],
    }:
        raise ValueError("capture model identity differs from corpus model")
    records, rows, turns = [file_record(p) for p in sources], [], []
    if not records or len({r["path"] for r in records}) != len(records):
        raise ValueError("empty or duplicate transcript sources")
    seen_tasks = set()
    for record in records:
        events = read_transcript(record["path"])
        provenance = events[0]["provenance"]
        if (
            provenance.get("model_identity") != model_identity
            or provenance.get("tokenizer") != tokenizer_identity
        ):
            raise ValueError("capture identity differs from frozen corpus")
        source_tasks = set()
        for begin, ledger in _turns(events):
            context = begin["context"]
            if context["task_id"] in seen_tasks:
                raise ValueError("duplicate trajectory across source records")
            source_tasks.add(context["task_id"])
            spans, owned, audit, prompt_offsets = _prompt_alignment(tokenizer, spec, begin)
            n_prompt = len(ledger.prompt_ids)
            consumed = min(len(ledger.generated), ledger.offset - n_prompt)
            labels, generated_text, generated_offsets = _generated_labels(
                tokenizer, ledger.generated
            )
            if generated_text != begin["emitted_text"]:
                raise ValueError("source emission decode differs from corpus tokenizer")
            ids = ledger.tokens[: n_prompt + consumed]
            spans.extend(labels[:consumed])
            owned.extend(range(n_prompt, n_prompt + consumed))
            turn_index = len(turns)
            split = "held" if turn_index % 5 == 4 else "fit"
            turns.append(
                {
                    "source": record["path"],
                    "turn": begin["turn"],
                    "task_id": context["task_id"],
                    "step": context["step"],
                    "prompt_tokens": n_prompt,
                    "consumed_emitted": consumed,
                    "emitted_unconsumed": len(ledger.generated) - consumed,
                    "consumed_unemitted": ledger.offset - n_prompt - consumed,
                    "eligible_positions": owned,
                    "ids_sha256": digest(ids),
                    "prompt_boundary_audit": audit,
                    "prompt_offsets": prompt_offsets,
                    "generated_text": generated_text,
                    "generated_offsets": generated_offsets,
                    **_parsed_action(generated_text),
                }
            )
            # At most 1024 scores per row; maximal available preceding context for
            # the scored block, bounded by 2048. Gaps create additional windows.
            remaining = list(owned)
            while remaining:
                first = remaining[0]
                block = [i for i in remaining[:1024] if i < first + 1024]
                end = block[-1] + 1
                start = max(0, end - 2048)
                rows.append(
                    {
                        "index": len(rows),
                        "source": record["path"],
                        "task_id": context["task_id"],
                        "turn_index": turn_index,
                        "step_index": context["step"],
                        "window_start": start,
                        "split": split,
                        "domain": "agentic",
                        "ids": ids[start:end],
                        "spans": spans[start:end],
                        "n_prompt": min(max(0, n_prompt - start), end - start),
                        "score_positions": [i - start for i in block],
                        "generated_mask": [i >= n_prompt for i in range(start, end)],
                    }
                )
                remaining = remaining[len(block) :]
        seen_tasks.update(source_tasks)
    manifest_path = Path(manifest_path).absolute()
    sequences_path = manifest_path.with_suffix(".jsonl")
    if sequences_path == manifest_path or sequences_path.exists() or manifest_path.exists():
        raise FileExistsError("immutable transcript corpus output already exists")
    data = b"".join(encoded(row) + b"\n" for row in rows)
    manifest = {
        "schema_version": 2,
        "format": FORMAT,
        "domain": "agentic",
        "max_tokens": 2048,
        "model_identity": model_identity,
        "model_hf_id": spec.hf_id,
        "tokenizer": tokenizer_identity,
        "sources": records,
        "turns": turns,
        "sequences": {"path": str(sequences_path), "sha256": hashlib.sha256(data).hexdigest()},
        "counts": _counts(rows),
        "replay_caveat": REPLAY_CAVEAT,
        "split_rule": "every fifth source turn held; before windowing; trajectories may overlap",
        "score_rule": "initial prompt once; new observation wrapper/content once; consumed emitted tokens once",
        "window_rule": "at most1024 eligible positions per block; up to2048 tokens ending at block end",
        "span_rule": "observation wrapper included; generated is orthogonal; tokens intersecting a new observation are owned and boundary-crossing tokens labelled template",
        "observation_rendering": {
            "role": spec.chat.observation_role,
            "template": spec.chat.observation_template,
            "convention": spec.chat.observation_convention,
        },
    }
    manifest["acceptance"] = transcript_acceptance(rows, turns)
    manifest["manifest_sha256"] = digest(manifest)
    for record in records + tokenizer_identity["files"]:
        checked_bytes(record)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with sequences_path.open("xb") as stream:
        stream.write(data)
    with manifest_path.open("xb") as stream:
        stream.write(encoded(manifest) + b"\n")
    return manifest


def _validate_identity(identity, tokenizer):
    if (
        not isinstance(identity, dict)
        or set(identity) != {"base", "training", "num_layers"}
        or not isinstance(identity["base"], str)
        or not identity["base"]
        or identity["training"] is not None
        or type(identity["num_layers"]) is not int
        or identity["num_layers"] < 2
    ):
        raise ValueError("transcript model requires explicit untrained canonical base and depth")
    if (
        not tokenizer.get("files")
        or not tokenizer.get("assets")
        or not tokenizer.get("chat_template_sha256")
    ):
        raise ValueError("transcript tokenizer requires exact file, asset and template hashes")


def read_transcript_corpus(manifest_path):
    """Fail closed on bound bytes, ledger reconstruction and score ownership accounting."""
    manifest = json.loads(Path(manifest_path).read_bytes())
    sha = manifest.pop("manifest_sha256")
    if digest(manifest) != sha:
        raise ValueError("transcript corpus manifest hash mismatch")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("format") != FORMAT
        or manifest.get("domain") != "agentic"
        or manifest.get("max_tokens") != 2048
        or manifest.get("replay_caveat") != REPLAY_CAVEAT
    ):
        raise ValueError("unsupported transcript corpus schema")
    _validate_identity(manifest["model_identity"], manifest["tokenizer"])
    captures = {}
    for source in manifest["sources"]:
        checked_bytes(source)
        if source["path"] in captures:
            raise ValueError("duplicate transcript source")
        events = read_transcript(source["path"])
        if (
            events[0]["provenance"].get("model_identity") != manifest["model_identity"]
            or events[0]["provenance"].get("tokenizer") != manifest["tokenizer"]
        ):
            raise ValueError("source identity mismatch")
        captures[source["path"]] = list(_turns(events))
    for asset in manifest["tokenizer"]["files"]:
        checked_bytes(asset)
    flat = [
        (source["path"], begin, ledger)
        for source in manifest["sources"]
        for begin, ledger in captures[source["path"]]
    ]
    if len(flat) != len(manifest["turns"]):
        raise ValueError("source turn coverage mismatch")
    from local_llm_lab.models import load_model_spec

    spec = load_model_spec(manifest["model_hf_id"])
    if (
        {
            "role": spec.chat.observation_role,
            "template": spec.chat.observation_template,
            "convention": spec.chat.observation_convention,
        }
        != manifest["observation_rendering"]
        or spec.base != manifest["model_identity"]["base"]
        or spec.training != manifest["model_identity"]["training"]
    ):
        raise ValueError("source rendering or lineage differs from current registry")
    ids_by_turn, spans_by_turn = [], []
    for turn, (source, begin, ledger) in zip(manifest["turns"], flat, strict=True):
        n = len(ledger.prompt_ids)
        consumed = min(len(ledger.generated), ledger.offset - n)
        ids = ledger.tokens[: n + consumed]
        if (
            turn["source"] != source
            or turn["turn"] != begin["turn"]
            or turn["task_id"] != begin["context"]["task_id"]
            or turn["step"] != begin["context"]["step"]
            or turn["prompt_tokens"] != n
            or turn["consumed_emitted"] != consumed
            or turn["emitted_unconsumed"] != len(ledger.generated) - consumed
            or turn["consumed_unemitted"] != ledger.offset - n - consumed
            or turn["ids_sha256"] != digest(ids)
        ):
            raise ValueError("source turn ledger accounting mismatch")
        spans, owned, audit = _prompt_semantics(spec, begin, turn["prompt_offsets"])
        if turn["generated_text"] != begin["emitted_text"] or any(
            turn.get(key) != value for key, value in _parsed_action(begin["emitted_text"]).items()
        ):
            raise ValueError("source emitted action differs from corpus action audit")
        generated_offsets = turn["generated_offsets"]
        # Empty decode offsets are allowed for special tokens; all offsets must be
        # contiguous prefixes and bounded by the recorded decoded continuation.
        previous = 0
        if len(generated_offsets) != len(ledger.generated):
            raise ValueError("generated boundary count mismatch")
        for pair in generated_offsets:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(v) is not int for v in pair)
                or pair[0] != previous
                or not previous <= pair[1] <= len(turn["generated_text"])
            ):
                raise ValueError("invalid generated boundary audit")
            previous = pair[1]
        if previous != len(turn["generated_text"]):
            raise ValueError("incomplete generated boundary audit")
        spans.extend(
            _labels(generated_offsets, _assistant_ranges(turn["generated_text"], 0))[:consumed]
        )
        owned.extend(range(n, n + consumed))
        if owned != turn["eligible_positions"] or audit != turn["prompt_boundary_audit"]:
            raise ValueError("semantic score ownership audit mismatch")
        ids_by_turn.append(ids)
        spans_by_turn.append(spans)
    rows = [json.loads(line) for line in checked_bytes(manifest["sequences"]).splitlines()]
    expected_windows = []
    for turn_index, turn in enumerate(manifest["turns"]):
        remaining = list(turn["eligible_positions"])
        while remaining:
            first = remaining[0]
            block = [i for i in remaining[:1024] if i < first + 1024]
            end = block[-1] + 1
            start = max(0, end - 2048)
            expected_windows.append((turn_index, start, end, [i - start for i in block]))
            remaining = remaining[len(block) :]
    if len(rows) != len(expected_windows):
        raise ValueError("deterministic transcript window count mismatch")
    scored = [[] for _ in flat]
    for index, row in enumerate(rows):
        t, start = row["turn_index"], row["window_start"]
        if type(t) is not int or not 0 <= t < len(flat) or type(start) is not int or start < 0:
            raise ValueError("invalid transcript window identity")
        ids, positions = row["ids"], row["score_positions"]
        if (t, start, start + len(ids), positions) != expected_windows[index]:
            raise ValueError("deterministic transcript window rule mismatch")
        turn = manifest["turns"][t]
        if (
            row["index"] != index
            or row["domain"] != "agentic"
            or row["split"] != ("held" if t % 5 == 4 else "fit")
            or row["source"] != turn["source"]
            or row["task_id"] != turn["task_id"]
            or row["step_index"] != turn["step"]
            or not 0 < len(ids) <= 2048
            or ids != ids_by_turn[t][start : start + len(ids)]
            or row["spans"] != spans_by_turn[t][start : start + len(ids)]
            or not positions
            or any(type(p) is not int or not 0 <= p < len(ids) for p in positions)
            or positions != sorted(set(positions))
            or row["generated_mask"]
            != [i >= turn["prompt_tokens"] for i in range(start, start + len(ids))]
            or row["n_prompt"] != min(max(0, turn["prompt_tokens"] - start), len(ids))
        ):
            raise ValueError("invalid transcript window or score positions")
        if "score_mask" in row and row["score_mask"] != [i in positions for i in range(len(ids))]:
            raise ValueError("conflicting score mask")
        scored[t].extend(start + p for p in positions)
    for positions, turn in zip(scored, manifest["turns"], strict=True):
        if positions != sorted(set(positions)) or positions != turn["eligible_positions"]:
            raise ValueError("duplicated or missing transcript score ownership")
    if transcript_acceptance(rows, manifest["turns"]) != manifest["acceptance"]:
        raise ValueError("transcript acceptance accounting mismatch")
    if _counts(rows) != manifest["counts"]:
        raise ValueError("transcript corpus counts mismatch")
    return rows
