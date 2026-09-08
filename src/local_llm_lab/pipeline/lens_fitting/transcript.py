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
    _labels,
    _training_id,
    _validate_offsets,
)
from local_llm_lab.pipeline.live_lens.session import SpanLabeller
from local_llm_lab.pipeline.protocol import _as_declared_roles, parse_turn, system_prompt

FORMAT = "transcript-native-replay-v2"
GENERATED_SPANS = ("note", "call_skeleton", "call_argument", "chat_prose")
INPUT_SPANS = ("system", "task", "observation", "assistant_history", "template")
SPANS = INPUT_SPANS + GENERATED_SPANS
BOS_POLICY = (
    "Preserve captured IDs exactly; never prepend BOS to replay windows, including "
    "mid-transcript windows, so replay retains model-seen tokenization rather than "
    "introducing a new document-initial signal."
)
POSITION_POLICY = (
    "Original positions are zero-based captured token offsets within each generation turn, "
    "not asserted native RoPE positions. Replay positions are zero-based fresh-window indices. "
    "All-input histograms include repeated context; scored histograms count each owned position once."
)
REPLAY_CAVEAT = (
    "Native replay of captured token IDs in fresh windows of the registered context length; generation forward "
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
        self.labeller = SpanLabeller("agentic")
        self.decoded = ""
        self.turn += 1
        self.active = True
        self.write(
            "begin_turn",
            prompt=prompt,
            prompt_ids=self.ledger.prompt_ids,
            context=self.context,
            bos_token_id=getattr(tokenizer, "bos_token_id", None),
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
        decoded = self.tokenizer.decode(self.ledger.generated)
        if not decoded.startswith(self.decoded):
            raise ValueError(
                "continuation decode is not prefix-stable; explicit alignment required"
            )
        piece = decoded[len(self.decoded) :]
        self.write("emitted", token=token, text=piece, span=self.labeller.feed(piece))
        self.decoded = decoded


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
            bos = event.get("bos_token_id")
            if "bos_token_id" not in event or (
                bos is not None and (type(bos) is not int or bos < 0)
            ):
                raise ValueError("capture requires an explicit tokenizer BOS ID or null")
            labeller, pieces, emitted_spans = SpanLabeller("agentic"), [], []
            current = (event, ForwardLedger(event["prompt_ids"]))
            previous_turn = event["turn"]
        elif current is None or event.get("turn") != previous_turn:
            raise ValueError("event outside its transcript turn")
        elif kind == "forward":
            current[1].record(event["offset"], event["ids"])
        elif kind == "emitted":
            current[1].emitted(event["token"])
            piece = event.get("text")
            if not isinstance(piece, str) or event.get("span") != labeller.feed(piece):
                raise ValueError("captured generated span differs from shared SpanLabeller")
            pieces.append(piece)
            emitted_spans.append(event["span"])
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
            if event["emitted_text"] != "".join(pieces):
                raise ValueError("emitted token pieces differ from full source decode")
            cursor, emitted_offsets = 0, []
            for piece in pieces:
                emitted_offsets.append([cursor, cursor + len(piece)])
                cursor += len(piece)
            begin = {
                **begin,
                "emitted_text": event["emitted_text"],
                "emitted_spans": emitted_spans,
                "emitted_offsets": emitted_offsets,
            }
            yield begin, ledger
            current = None
        else:
            raise ValueError("unknown transcript event")
    if current is not None:
        raise ValueError("unclosed transcript turn")


def _prompt_alignment(tokenizer, spec, begin):
    if begin["bos_token_id"] != getattr(tokenizer, "bos_token_id", None):
        raise ValueError("capture BOS identity differs from the frozen tokenizer")
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
            ranges.append((start, end, "assistant_history"))
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


def _generated_labels(tokenizer, tokens, *, kind="agentic"):
    # Generation is bounded by evaluation max_tokens (200). Prefix decoding is confined
    # to the continuation, never performed across the growing prompt history.
    text, offsets, labels = "", [], []
    labeller = SpanLabeller(kind)
    for end in range(1, len(tokens) + 1):
        decoded = tokenizer.decode(tokens[:end])
        if not decoded.startswith(text):
            raise ValueError(
                "continuation decode is not prefix-stable; explicit alignment required"
            )
        offsets.append([len(text), len(decoded)])
        labels.append(labeller.feed(decoded[len(text) :]))
        text = decoded
    return labels, text, offsets


def _window_rule(max_tokens):
    return {
        "context_tokens": max_tokens,
        "max_scored_block_tokens": max_tokens // 2,
        "policy": "end at scored block end; retain maximal available preceding context",
    }


def _bos_evidence(ids, token_id):
    return {
        "token_id": token_id,
        "starts_with_captured_bos": token_id is not None and ids[0] == token_id,
        "bos_positions": [
            i for i, token in enumerate(ids) if token_id is not None and token == token_id
        ],
        "inserted_bos_tokens": 0,
    }


def _counts(rows, *, max_tokens):
    counts = {
        split: {
            "sequences": 0,
            "tokens": 0,
            "positions": 0,
            "spans": dict.fromkeys(SPANS, 0),
            "generated": 0,
            "generated_spans": dict.fromkeys(GENERATED_SPANS, 0),
            "input_spans": dict.fromkeys(INPUT_SPANS, 0),
            "position_distributions": {
                scope: {
                    kind: {} for kind in ("original_captured_offsets", "actual_replay_positions")
                }
                for scope in ("scored", "all_input")
            },
            "bos": {
                "rows_starting_with_captured_bos": 0,
                "rows_starting_without_bos": 0,
                "input_bos_positions": 0,
                "scored_bos_positions": 0,
                "inserted_bos_tokens": 0,
            },
            "positions_beyond_1024": 0,
            "full_context_rows": 0,
            "short_rows": 0,
        }
        for split in ("fit", "held")
    }
    for row in rows:
        c = counts[row["split"]]
        c["sequences"] += 1
        c["tokens"] += len(row["ids"])
        c["full_context_rows" if len(row["ids"]) == max_tokens else "short_rows"] += 1
        bos = row["bos"]
        c["bos"][
            "rows_starting_with_captured_bos"
            if bos["starts_with_captured_bos"]
            else "rows_starting_without_bos"
        ] += 1
        c["bos"]["input_bos_positions"] += len(bos["bos_positions"])
        for scope, positions in (
            ("scored", row["score_positions"]),
            ("all_input", range(len(row["ids"]))),
        ):
            for i in positions:
                for kind, value in (
                    ("original_captured_offsets", row["window_start"] + i),
                    ("actual_replay_positions", i),
                ):
                    histogram = c["position_distributions"][scope][kind]
                    histogram[str(value)] = histogram.get(str(value), 0) + 1
        for i in row["score_positions"]:
            if i in bos["bos_positions"]:
                c["bos"]["scored_bos_positions"] += 1
            facet = "generated_spans" if row["generated_mask"][i] else "input_spans"
            c[facet][row["spans"][i]] += 1
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


def transcript_acceptance(rows, turns, *, max_tokens):
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
    counts = _counts(rows, max_tokens=max_tokens)
    if not counts["fit"]["positions_beyond_1024"]:
        reasons.append("no fitted position actually exceeds the 1024-token sliding window")
    if not any(
        10 * counts["fit"]["spans"][span] > fit["positions"]
        for span in ("observation", "note", "call_skeleton", "call_argument")
    ):
        reasons.append(
            "no agentic note/call skeleton/call argument/observation span exceeds ten percent of fitted positions"
        )
    return {
        "status": "ruling_required" if reasons else "passed",
        "reasons": reasons,
        "concentration": concentration,
        "non_prose_span_check": {
            "denominator": "all fitted scored positions",
            "positions": fit["positions"],
            "eligible_counts": {
                name: counts["fit"]["spans"][name]
                for name in ("observation", "note", "call_skeleton", "call_argument")
            },
            "note_scope": "Linguistic prose, retained as the originally eligible agentic progress-note facet.",
            "generated_only_denominator": counts["fit"]["generated"],
            "generated_only_counts": counts["fit"]["generated_spans"],
        },
        "repeated_runs": runs,
        "repeated_run_rule": (
            "consecutive canonical(name,arguments) equal calls; length>=2 includes first member; "
            "all scored member-turn generated tokens including formatting/special/template spans; associated next-prompt newly owned "
            "observation tokens including wrappers; initial system/task scaffold excluded; "
            "invalid actions break runs and remain in position denominators"
        ),
    }


def build_transcript_corpus(
    sources, tokenizer, spec, manifest_path, *, tokenizer_identity, model_identity, max_tokens
):
    """Freeze exact consumed IDs and score first appearances once; no native execution."""
    if type(max_tokens) is not int or max_tokens <= 1024 or max_tokens % 2:
        raise ValueError("registered transcript context must be even and exceed the sliding window")
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
            provenance.get("fitting_context_tokens") != max_tokens
            or provenance.get("model_identity") != model_identity
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
            if generated_text != begin["emitted_text"] or labels != begin["emitted_spans"]:
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
                    "bos_token_id": begin["bos_token_id"],
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
            # Half a registered context of scores per row, with maximal available
            # preceding context up to the registered cap. Gaps create extra windows.
            remaining = list(owned)
            while remaining:
                first = remaining[0]
                block = [i for i in remaining[: max_tokens // 2] if i < first + max_tokens // 2]
                end = block[-1] + 1
                start = max(0, end - max_tokens)
                rows.append(
                    {
                        "index": len(rows),
                        "source": record["path"],
                        "task_id": context["task_id"],
                        "turn_index": turn_index,
                        "step_index": context["step"],
                        "window_start": start,
                        "original_position_range": [start, end],
                        "replay_position_range": [0, end - start],
                        "bos": _bos_evidence(ids[start:end], begin["bos_token_id"]),
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
        "max_tokens": max_tokens,
        "model_identity": model_identity,
        "model_hf_id": spec.hf_id,
        "tokenizer": tokenizer_identity,
        "sources": records,
        "turns": turns,
        "sequences": {"path": str(sequences_path), "sha256": hashlib.sha256(data).hexdigest()},
        "counts": _counts(rows, max_tokens=max_tokens),
        "replay_caveat": REPLAY_CAVEAT,
        "bos_policy": BOS_POLICY,
        "position_policy": POSITION_POLICY,
        "generated_span_classifier": "local_llm_lab.pipeline.live_lens.session.SpanLabeller",
        "split_rule": "every fifth source turn held; before windowing; trajectories may overlap",
        "score_rule": "initial prompt once; new observation wrapper/content once; consumed emitted tokens once",
        "window_rule": _window_rule(max_tokens),
        "span_rule": "observation wrapper included; generated is orthogonal; tokens intersecting a new observation are owned and boundary-crossing tokens labelled template",
        "observation_rendering": {
            "role": spec.chat.observation_role,
            "template": spec.chat.observation_template,
            "convention": spec.chat.observation_convention,
        },
    }
    manifest["acceptance"] = transcript_acceptance(rows, turns, max_tokens=max_tokens)
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
    max_tokens = manifest.get("max_tokens")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("format") != FORMAT
        or manifest.get("domain") != "agentic"
        or type(max_tokens) is not int
        or max_tokens <= 1024
        or max_tokens % 2
        or manifest.get("window_rule") != _window_rule(max_tokens)
        or manifest.get("replay_caveat") != REPLAY_CAVEAT
        or manifest.get("bos_policy") != BOS_POLICY
        or manifest.get("position_policy") != POSITION_POLICY
        or manifest.get("generated_span_classifier")
        != "local_llm_lab.pipeline.live_lens.session.SpanLabeller"
    ):
        raise ValueError("unsupported transcript corpus schema")
    _validate_identity(manifest["model_identity"], manifest["tokenizer"])
    captures = {}
    seen_tasks = set()
    for source in manifest["sources"]:
        checked_bytes(source)
        if source["path"] in captures:
            raise ValueError("duplicate transcript source")
        events = read_transcript(source["path"])
        if (
            events[0]["provenance"].get("fitting_context_tokens") != max_tokens
            or events[0]["provenance"].get("model_identity") != manifest["model_identity"]
            or events[0]["provenance"].get("tokenizer") != manifest["tokenizer"]
        ):
            raise ValueError("source identity mismatch")
        source_turns = list(_turns(events))
        task_ids = {begin["context"]["task_id"] for begin, _ in source_turns}
        if task_ids & seen_tasks:
            raise ValueError("duplicate trajectory across source records")
        seen_tasks.update(task_ids)
        captures[source["path"]] = source_turns
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
            or turn["bos_token_id"] != begin["bos_token_id"]
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
        if generated_offsets != begin["emitted_offsets"]:
            raise ValueError("generated boundaries differ from captured token pieces")
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
        labeller = SpanLabeller("agentic")
        generated_labels = [
            labeller.feed(turn["generated_text"][a:b]) for a, b in generated_offsets
        ]
        if generated_labels != begin["emitted_spans"]:
            raise ValueError("generated boundary labels differ from captured SpanLabeller facets")
        spans.extend(generated_labels[:consumed])
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
            block = [i for i in remaining[: max_tokens // 2] if i < first + max_tokens // 2]
            end = block[-1] + 1
            start = max(0, end - max_tokens)
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
            or row["original_position_range"] != [start, start + len(ids)]
            or row["replay_position_range"] != [0, len(ids)]
            or row["bos"] != _bos_evidence(ids, turn["bos_token_id"])
            or not 0 < len(ids) <= max_tokens
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
    if (
        transcript_acceptance(rows, manifest["turns"], max_tokens=max_tokens)
        != manifest["acceptance"]
    ):
        raise ValueError("transcript acceptance accounting mismatch")
    if _counts(rows, max_tokens=max_tokens) != manifest["counts"]:
        raise ValueError("transcript corpus counts mismatch")
    return rows
