"""Gated, episode-separated §3.5 profiles. No model execution or array loading.

The public CLI consumes a JSON specification with ``sets``. Each set names model,
domain, kind (regression/jacobian/hosted-jacobian), lens, records (directories),
identity (Task4 hosted identity directory), and instrument {plan, corpus, proof,
runtime}. Missing cross-read cells are reported, never substituted with chat.
"""

from __future__ import annotations

import json
import math
import re
import runpy
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus
from local_llm_lab.pipeline.lens_fitting.jacobian import read_plan, require_self_check
from local_llm_lab.pipeline.lens_fitting.replay import first_difference, read_source
from local_llm_lab.pipeline.lens_fitting.runtime import snapshot_identity
from local_llm_lab.pipeline.live_lens.instruments import file_sha256
from local_llm_lab.project import PROJECT_ROOT

PILOT = PROJECT_ROOT / "research/records/LIVE-LENS-PILOT-2026-09-07"
ATLAS_SCRIPT = PROJECT_ROOT / "scripts/live_lens_atlas.py"
OVERLAP = "Held prose windows selected the ridge weight among five values; this is not a disjoint prose test. Pilot episodes are disjoint generalisation material."


class Inputs:
    """Remember consumed bytes, including upstream descriptors; recheck before output."""

    def __init__(self):
        self.hashes = {}
        self.record_sets = {}
        self.snapshots = []

    def file(self, path, expected=None):
        path = Path(path).absolute()
        sha = file_sha256(path)
        if (expected is not None and sha != expected) or (
            str(path) in self.hashes and self.hashes[str(path)] != sha
        ):
            raise ValueError(f"changed input: {path}")
        self.hashes[str(path)] = sha
        return path

    def json(self, path, expected=None):
        path = self.file(path, expected)
        result = json.loads(
            path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
        )
        self.file(path)
        return result

    def refs(self, value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self.file(value["path"], value["sha256"])
            for item in value.values():
                self.refs(item)
        elif isinstance(value, list):
            for item in value:
                self.refs(item)

    def record_set(self, directory, names):
        _record_set(directory, names)
        key = str(Path(directory).absolute())
        if key in self.record_sets and self.record_sets[key] != set(names):
            raise ValueError("record membership changed")
        self.record_sets[key] = set(names)

    def recheck(self):
        for snapshot in self.snapshots:
            if (
                snapshot_identity(Path(snapshot["snapshot_path"]), hf_id=snapshot["hf_id"])
                != snapshot
            ):
                raise ValueError("snapshot asset set changed")
        for directory, names in self.record_sets.items():
            _record_set(directory, names)
        for path, sha in list(self.hashes.items()):
            self.file(path, sha)


@lru_cache(maxsize=1)
def _legacy_helpers(script_sha):
    # Actual legacy implementation, with main disabled; no duplicate span algorithm.
    helpers = runpy.run_path(str(ATLAS_SCRIPT))
    if file_sha256(ATLAS_SCRIPT) != script_sha:
        raise ValueError("legacy script changed during loading")
    return helpers


def legacy_helpers():
    return _legacy_helpers(file_sha256(ATLAS_SCRIPT))


def prompt_spans(tokenizer, begin, kind):
    prompt, ids = begin["prompt"], begin["prompt_ids"]
    bos = getattr(tokenizer, "bos_token", None)
    add = (bos is None or not prompt.startswith(bos)) if kind == "prose" else False
    encoded = tokenizer(prompt, add_special_tokens=add, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if list(encoded["input_ids"]) != ids or len(offsets) != len(ids):
        raise ValueError("tokenizer offsets do not align with exact recorded prompt IDs")
    if any(not 0 <= a <= b <= len(prompt) for a, b in offsets):
        raise ValueError("invalid tokenizer offsets")
    if kind == "prose":
        return ["chat"] * len(ids)
    labels = legacy_helpers()["spans_for_prompt"](
        tokenizer, prompt, begin["context"].get("messages", []), kind
    )
    if len(labels) != len(ids):
        raise ValueError("legacy span labels do not align with prompt IDs")
    return labels


def statistic(values):
    n, numerator = len(values), sum(values)
    return {"numerator": numerator, "n": n, "share": numerator / n if n else None}


def summarize_episode(events, episode, tokenizer, *, layers, lens_kind, validation):
    """Pure statistics seam; production callers first validate record chains and gates."""
    if layers != list(range(1, max(layers) + 1)) or len(layers) < 2:
        raise ValueError("profiles require all nonfinal layers and final identity")
    final, kind = layers[-1], episode["kind"]
    helpers = legacy_helpers()
    if kind not in {"agentic", "chat", "prose"}:
        raise ValueError("unknown episode kind")
    prompts, texts, emitted, rank = {}, defaultdict(str), {}, {}
    agreement, prompt_labels, target_labels = defaultdict(list), set(), set()
    for event in events:
        tag = event["kind"]
        if tag == "begin_turn":
            turn = event["turn"]
            if turn in prompts or event["layers"] != layers:
                raise ValueError("duplicate turn or inconsistent layers")
            labels = prompt_spans(tokenizer, event, kind)
            prompts[turn] = (event["prompt_ids"], labels)
            prompt_labels.update(labels[:-1])
        elif tag == "reading":
            ids, labels = prompts[event["turn"]]
            position = event["position"]
            if set(event["top"]) != {str(layer) for layer in layers}:
                raise ValueError("reading layer set differs")
            if position + 1 < len(ids):
                if position < 0:
                    raise ValueError("negative reading position")
                for layer in layers:
                    agreement[layer, labels[position]].append(
                        event["top"][str(layer)][0] == ids[position + 1]
                    )
        elif tag == "emitted":
            turn, position = event["turn"], event["position"]
            if (
                turn not in prompts
                or position < len(prompts[turn][0])
                or (turn, position) in emitted
            ):
                raise ValueError("invalid or duplicate emitted target")
            texts[turn] += tokenizer.decode([event["token_id"]])
            span = helpers["emitted_span"](texts[turn], kind)
            emitted[turn, position] = (event["token_id"], span)
            target_labels.add(span)
        elif tag == "rank":
            turn, position, horizon = event["turn"], event["position"], event["horizon"]
            target = position + horizon
            target_event = emitted.get((turn, target))
            if (
                horizon not in (1, 4, 8)
                or position < 0
                or target_event is None
                or target_event[0] != event["token_id"]
            ):
                raise ValueError("rank does not target an actual emitted event")
            key = (turn, position, target, event["token_id"], horizon)
            layer = event["layer"]
            if (
                layer not in layers
                or (layer, key) in rank
                or type(event["rank"]) is not int
                or event["rank"] < 1
            ):
                raise ValueError("invalid or duplicate rank pair")
            rank[layer, key] = event["rank"]
    keys = {key for layer, key in rank if layer == final}
    for layer in layers[:-1]:
        if {key for observed, key in rank if observed == layer} != keys:
            raise ValueError("missing or extra final/nonfinal rank pairs")
    if not target_labels:
        target_labels.add({"prose": "continuation", "chat": "chat", "agentic": "note"}[kind])
    result = []
    for layer in layers[:-1]:
        map_check = (
            validation.get(str(layer), {}).get("map_check", {}).get("outcome", "inconclusive")
        )
        if map_check not in {"pass", "fail", "inconclusive"}:
            raise ValueError("invalid recorded map check")
        common = {
            "episode": episode["label"],
            "episode_kind": kind,
            "layer": layer,
            "difficulty": episode.get("difficulty"),
            "map_check": map_check,
            "validation": validation.get(str(layer), {}).get("verdict"),
            "final_layer": final,
        }
        for span in sorted(target_labels):
            for horizon in (1, 4, 8):
                selected = sorted(
                    key for key in keys if key[-1] == horizon and emitted[key[0], key[2]][1] == span
                )
                biased = lens_kind == "regression" and horizon == 1
                result.append(
                    common
                    | {
                        "summary": "foreknowledge",
                        "span": span,
                        "span_role": "emitted_target",
                        "target_role": "actual_emitted_token",
                        "horizon": horizon,
                        **statistic([rank[layer, key] <= 10 for key in selected]),
                        "base_rate": statistic([rank[final, key] <= 10 for key in selected]),
                        "biased_by_construction": biased,
                        "interpretive_eligible": not biased
                        and (lens_kind == "regression" or map_check == "pass"),
                        "interpretation": None,
                    }
                )
        for span in sorted(prompt_labels):
            result.append(
                common
                | {
                    "summary": "prompt_agreement",
                    "span": span,
                    "span_role": "source_prompt_position",
                    "target_role": "next_authored_prompt_token",
                    "horizon": None,
                    **statistic(agreement[layer, span]),
                    "base_rate": statistic(agreement[final, span]),
                    "biased_by_construction": False,
                    "interpretive_eligible": lens_kind == "regression" or map_check == "pass",
                    "interpretation": None,
                }
            )
    return result


def _record(path, inputs, *, sha=None):
    path = inputs.file(path, sha)
    events, identity = read_source(path)
    inputs.file(path)
    return events, identity


def _labels(entries):
    labels = [entry["label"] for entry in entries]
    if len(labels) != len(set(labels)) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in labels
    ):
        raise ValueError("invalid or duplicate episode labels")
    return set(labels)


def _record_set(directory, expected):
    actual = {p.name for p in Path(directory).iterdir() if p.name.endswith((".jsonl", ".jsonl.gz"))}
    if actual != set(expected):
        raise ValueError("complete output record set differs")


def replay_records(directory, inputs):
    """Current replay-complete hashes, never historical record hashes, own outputs."""
    directory = Path(directory)
    historical = inputs.json(directory / "manifest.json")
    current = inputs.json(directory / "replay-manifest.json")
    complete = inputs.json(directory / "replay-complete.json")
    if (
        complete["status"] != "complete"
        or current["source_manifest_sha256"]
        != inputs.hashes[str((directory / "manifest.json").absolute())]
    ):
        raise ValueError("incomplete replay or historical manifest drift")
    labels = _labels(historical["episodes"])
    if _labels(complete["outputs"]) != labels or _labels(current["sources"]) != labels:
        raise ValueError("replay source/output membership differs")
    inputs.record_set(directory, [label + ".jsonl" for label in labels])
    output = []
    for episode in historical["episodes"]:
        label = episode["label"]
        source = next(row for row in current["sources"] if row["label"] == label)
        original, identity = _record(source["identity"]["path"], inputs)
        if (
            identity != source["identity"]
            or identity["decompressed_sha256"] != episode["record_sha256"]
        ):
            raise ValueError("replay source identity differs")
        result = next(row for row in complete["outputs"] if row["label"] == label)
        events, _ = _record(directory / (label + ".jsonl"), inputs, sha=result["sha256"])
        provenance = events[0]["provenance"]
        if provenance != current | {"episode": label}:
            raise ValueError("current record provenance differs from replay manifest")

        def controls(rows):
            return [
                {k: v for k, v in row.items() if k != "layers"}
                for row in rows
                if row["kind"] in {"begin_turn", "forward", "emitted", "end_turn"}
            ]

        if first_difference(controls(original), controls(events)):
            raise ValueError("native replay controls differ from actual source")
        if result["forwards"] != sum(row["kind"] == "forward" for row in events) or result[
            "tokens"
        ] != sum(len(row["input_ids"]) for row in events if row["kind"] == "forward"):
            raise ValueError("replay completion counts differ")
        if any(row["layers"] != current["layers"] for row in events if row["kind"] == "begin_turn"):
            raise ValueError("replay layers differ")
        output.append((episode, events))
    return historical, current, output


def verify_identity(directory, snapshot, inputs):
    """Anchor to the repository's actual hosted pilot, not a caller-provided fixture."""
    directory = Path(directory)
    proof = inputs.json(directory / "identity.json")
    source_atlas = inputs.json(PILOT / "atlas.json", proof["source_atlas_sha256"])
    output_atlas = inputs.json(directory / "atlas.json", proof["output_atlas_sha256"])
    canonical = inputs.json(PILOT / "manifest.json")
    inputs.file(ATLAS_SCRIPT, proof["summarizer_sha256"])
    inputs.file(directory / "manifest.json", proof["legacy_manifest_sha256"])
    inputs.file(directory / "replay-manifest.json", proof["replay_manifest_sha256"])
    if (
        first_difference(source_atlas, output_atlas)
        or proof.get("exact_json_equal") is not True
        or proof.get("first_difference") is not None
    ):
        raise ValueError("actual legacy atlas JSON identity failed")
    historical, current, records = replay_records(directory, inputs)
    if (
        historical != canonical
        or current["snapshot"] != snapshot
        or current["lens_sha256"] != canonical["lens_sha256"]
        or current["layers"] != canonical["layers"]
    ):
        raise ValueError("identity is not the actual hosted pilot on this snapshot")
    if canonical["model"] != snapshot["hf_id"]:
        raise ValueError("identity model differs")
    if proof["output_records"] != {
        episode["label"] + ".jsonl": file_sha256(directory / (episode["label"] + ".jsonl"))
        for episode, _ in records
    }:
        raise ValueError("identity output record hashes differ")
    for episode, events in records:
        path = PILOT / episode["record"]
        if not path.exists() and path.suffix == ".jsonl":
            path = path.with_suffix(".jsonl.gz")
        original, identity = _record(path, inputs)
        if identity["decompressed_sha256"] != episode["record_sha256"]:
            raise ValueError("canonical pilot record hash differs")
        # Exact native observations are independently bound, not just a claimed atlas flag.
        if first_difference(original[1:-1], events[1:-1]):
            raise ValueError("hosted identity observations differ from canonical pilot")
    return {
        "identity": str(directory / "identity.json"),
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "status": "verified_actual_pilot",
    }


def verify_instrument(config, inputs):
    corpus = inputs.json(config["corpus"])
    inputs.refs(corpus)
    rows = read_corpus(Path(config["corpus"]))
    inputs.file(config["plan"])
    plan = read_plan(Path(config["plan"]), rows)
    proof = inputs.json(config["proof"])
    runtime = inputs.json(config["runtime"])
    snapshot = runtime["snapshot"]
    actual = snapshot_identity(Path(snapshot["snapshot_path"]), hf_id=snapshot["hf_id"])
    if (
        actual != snapshot
        or plan["snapshot_sha256"] != snapshot["snapshot_sha256"]
        or plan["corpus_sha256"] != file_sha256(Path(config["corpus"]))
    ):
        raise ValueError("instrument corpus or snapshot differs")
    inputs.snapshots.append(snapshot)
    for asset in snapshot["files"]:
        inputs.file(Path(snapshot["snapshot_path"]) / asset["name"], asset["sha256"])
    architecture = inputs.json(Path(snapshot["snapshot_path"]) / "config.json")
    architecture = architecture.get("text_config", architecture)
    if (
        plan["layers"] != list(range(1, architecture["num_hidden_layers"]))
        or plan["hidden_size"] != architecture["hidden_size"]
    ):
        raise ValueError("instrument architecture differs")
    require_self_check(plan, proof)
    if proof.get("direction_seed") != plan["seeds"]["self_directions"]:
        raise ValueError("instrument direction seed differs")
    for check in proof["checks"]:
        for measured, bounds in [
            (check, plan["self_bounds"]),
            (check["reference_stability"], plan["stability_bounds"]),
        ]:
            if (
                measured["outcome"] != "pass"
                or any(measured[k] != bounds[k] for k in ("atol", "rtol"))
                or not isinstance(measured["max_error"], (int, float))
                or not math.isfinite(measured["max_error"])
                or measured["max_error"] < 0
            ):
                raise ValueError("instrument numeric evidence missing or mismatched")
        if not math.isfinite(check["epsilon"]) or check["epsilon"] <= 0:
            raise ValueError("invalid instrument epsilon")
    inputs.recheck()
    return plan, snapshot


def prose_records(directory, inputs, tokenizer):
    """Validate §14 accepted/dropped coverage; only accepted root records are returned."""
    from local_llm_lab.pipeline.lens_fitting.prose import OVERLAP as CAPTURE_OVERLAP
    from local_llm_lab.pipeline.lens_fitting.prose import _bytes, _sha

    directory = Path(directory)
    plan = inputs.json(directory / "plan.json")
    if _sha(_bytes({k: v for k, v in plan.items() if k != "plan_sha256"})) != plan["plan_sha256"]:
        raise ValueError("prose frozen plan digest differs")
    manifest = inputs.json(directory / "manifest.json")
    corpus = inputs.json(plan["corpus"]["path"], plan["corpus"]["sha256"])
    inputs.refs(corpus)
    rows = read_corpus(Path(plan["corpus"]["path"]))
    held = [row for row in rows if row["split"] == "held"]
    if (
        len(held) != 51
        or len(plan["windows"]) != 51
        or corpus["domain"] != "prose"
        or corpus["model_hf_id"] != plan["model"]
    ):
        raise ValueError("invalid held prose corpus")
    if (
        plan["min_tokens"],
        plan["max_tokens"],
        plan["top_k"],
        plan["cache_strategy"],
        plan["sampler"],
        plan["emitted_count_includes_eos"],
        plan["overlap"],
    ) != (32, 200, 10, "none", {"temperature": 0.0, "kind": "greedy"}, True, CAPTURE_OVERLAP):
        raise ValueError("prose protocol differs")
    for key in ("tokenizer", "sequences", "sources"):
        if plan[key] != corpus[key]:
            raise ValueError("prose plan corpus descriptor differs")
    for row, window in zip(held, plan["windows"], strict=True):
        if (
            window["kind"] != "prose"
            or window["max_tokens"] != 200
            or window["window_index"] != row["index"]
            or window["prompt_ids"] != row["ids"][:824]
            or window["window_sha256"] != _sha(_bytes(row["ids"]))
            or window["prompt_tokens"] != 824
            or window["prompt_sha256"] != _sha(window["prompt"].encode())
            or tokenizer.decode(window["prompt_ids"]) != window["prompt"]
        ):
            raise ValueError("prose held window or prefix differs")
    if manifest["status"] != "complete" or manifest["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("incomplete prose capture")
    for key in (
        "model",
        "snapshot",
        "layers",
        "lens_sha256",
        "top_k",
        "min_tokens",
        "max_tokens",
        "sampler",
        "cache_strategy",
        "emitted_count_includes_eos",
        "overlap",
        "seed",
    ):
        if manifest[key] != plan[key]:
            raise ValueError(f"prose manifest {key} differs")
    accepted, dropped = manifest["episodes"], manifest["dropped"]
    if (
        manifest["accepted_count"] != len(accepted)
        or manifest["dropped_count"] != len(dropped)
        or _labels(accepted + dropped) != _labels(plan["windows"])
    ):
        raise ValueError("prose accepted/dropped coverage differs")
    inputs.record_set(directory, [entry["label"] + ".jsonl" for entry in accepted])
    inputs.record_set(directory / "dropped", [entry["label"] + ".jsonl" for entry in dropped])
    result = []
    for entry in accepted + dropped:
        is_dropped = entry in dropped
        name = ("dropped/" if is_dropped else "") + entry["label"] + ".jsonl"
        if entry["record"] != name:
            raise ValueError("prose accepted/dropped path differs")
        window = next(w for w in plan["windows"] if w["label"] == entry["label"])
        for key in (
            "window_index",
            "window_sha256",
            "prompt_sha256",
            "prompt_tokens",
            "kind",
            "max_tokens",
        ):
            if entry[key] != window[key]:
                raise ValueError("prose episode metadata differs")
        events, _ = _record(directory / name, inputs, sha=entry["record_sha256"])
        provenance = events[0]["provenance"]
        if provenance != {
            key: manifest[key] for key in ("model", "lens_sha256", "layers", "snapshot")
        } | {"episode": window, "plan_sha256": plan["plan_sha256"]}:
            raise ValueError("prose record provenance differs")
        begins = [row for row in events if row["kind"] == "begin_turn"]
        emissions = [row for row in events if row["kind"] == "emitted"]
        if (
            len(begins) != 1
            or begins[0]["prompt_ids"] != window["prompt_ids"]
            or begins[0]["prompt"] != window["prompt"]
            or begins[0]["layers"] != plan["layers"]
            or begins[0]["context"].get("kind") != "prose"
        ):
            raise ValueError("prose recorded prompt differs")
        prompt_spans(tokenizer, begins[0], "prose")
        n = len(emissions)
        if n != entry["generated_tokens"] or n > 200 or (n < 32) != is_dropped:
            raise ValueError("prose continuation retention/count differs")
        if is_dropped and entry.get("reason") != "continuation_under_32":
            raise ValueError("prose dropped reason differs")
        if entry["finish_reason"] == "length":
            if n != 200:
                raise ValueError("prose cap count differs")
        elif entry["finish_reason"] == "stop":
            eos = getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id]
            if not emissions or emissions[-1]["token_id"] not in eos:
                raise ValueError("prose stop was not EOS")
        else:
            raise ValueError("prose unsupported stop")
        if not is_dropped:
            result.append((entry, events))
    return manifest, plan, result


def lens_metadata(item, snapshot, plan, inputs):
    lens = inputs.file(item["lens"])
    metadata = inputs.json(lens.with_suffix(".json"))
    inputs.file(lens, metadata["npz_sha256"])
    kind, domain = item["kind"], item["domain"]
    if domain not in {"agentic", "prose"} or kind not in {
        "regression",
        "jacobian",
        "hosted-jacobian",
    }:
        raise ValueError("unsupported fitted domain/kind")
    depth = len(plan["layers"]) + 1
    if kind == "hosted-jacobian":
        canonical = inputs.json(PILOT / "manifest.json")
        if (
            domain != "prose"
            or metadata["npz_sha256"] != canonical["lens_sha256"]
            or metadata["d_model"] != plan["hidden_size"]
            or metadata["layers_in_file"] != list(range(depth - 1))
        ):
            raise ValueError("hosted lens does not match pilot")
        validation = {}
    else:
        model = {k: v for k, v in metadata["model"].items() if k != "name"}
        if (
            metadata["kind"] != kind
            or metadata["domain"] != domain
            or model != snapshot
            or metadata["layers"] != plan["layers"]
            or metadata["num_layers"] != depth
            or metadata["hidden_size"] != plan["hidden_size"]
        ):
            raise ValueError("fitted lens model/domain/kind/layers differ")
        validation = {}
        if kind == "jacobian":
            report = metadata["validation"]
            if (
                metadata["plan_sha256"] != plan["plan_sha256"]
                or metadata["run_plan"] != plan
                or report["plan_sha256"] != plan["plan_sha256"]
                or report["directions"] != 32
                or report["direction_seed"] != plan["seeds"]["validation_directions"]
            ):
                raise ValueError("Jacobian lens plan/validation differs")
            validation = report["per_layer"]
            if set(validation) != {str(x) for x in plan["layers"]}:
                raise ValueError("missing per-layer map validation")
            for check in validation.values():
                from local_llm_lab.pipeline.live_lens.validation import layer_verdict

                verdict = check["verdict"]
                agreement = check["map_check"]
                expected = layer_verdict(agreement["outcome"], "inconclusive", None)
                if verdict != expected or any(
                    agreement[k] != plan["response_bounds"][k] for k in ("atol", "rtol")
                ):
                    raise ValueError("map validation hooks differ")
    return metadata, validation


def specificity(sets):
    """Availability cells, never pooled numeric claims."""
    sets = list(sets)
    for model, kind in sorted({(item["model"], item["kind"]) for item in sets}):
        domains = ["prose"] if kind == "hosted-jacobian" else ["agentic", "prose"]
        for domain in domains:
            if not any(
                item["model"] == model and item["kind"] == kind and item["domain"] == domain
                for item in sets
            ):
                sets.append(
                    {
                        "name": "-".join((model, domain, kind)),
                        "domain": domain,
                        "model": model,
                        "kind": kind,
                        "rows": [],
                    }
                )
    return [
        {
            "lens_set": item["name"],
            "training_domain": item["domain"],
            "evaluation_material": material,
            "status": "available"
            if any(row["episode_kind"] == kind for row in item["rows"])
            else "unavailable",
            "episodes": sorted(
                {row["episode"] for row in item["rows"] if row["episode_kind"] == kind}
            ),
            "overlap": OVERLAP
            if kind == "prose"
            else "Disjoint pilot evaluation; never fit material.",
        }
        for item in sets
        for material, kind in [
            ("pilot_agentic", "agentic"),
            ("held_prose_continuations", "prose"),
            ("pilot_chat", "chat"),
        ]
    ]


def build_profiles(config_path, output):
    """All real evidence gates precede scientific aggregation; output is exclusive."""
    inputs = Inputs()
    inputs.file(Path(__file__))
    inputs.file(ATLAS_SCRIPT)
    config = inputs.json(config_path)
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    items = config["sets"]
    if not items:
        raise ValueError("at least one named lens set required")
    prepared, names = [], set()
    for item in items:
        name = "-".join(item[key] for key in ("model", "domain", "kind"))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("invalid or duplicate lens set name")
        names.add(name)
        plan, snapshot = verify_instrument(item["instrument"], inputs)
        from local_llm_lab.models import load_model_spec

        if load_model_spec(item["model"]).hf_id != snapshot["hf_id"]:
            raise ValueError("selected model differs from evidence")
        identity = verify_identity(item["identity"], snapshot, inputs)
        metadata, validation = lens_metadata(item, snapshot, plan, inputs)
        prepared.append((item, name, plan, snapshot, identity, metadata, validation))
    # No record summary above this boundary. Loading a local tokenizer is not a model load.
    from transformers import AutoTokenizer

    products = []
    for item, name, plan, snapshot, identity, metadata, validation in prepared:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot["snapshot_path"], local_files_only=True, trust_remote_code=False, use_fast=True
        )
        rows, records, seen = [], [], set()
        for directory in item["records"]:
            directory = Path(directory).absolute()
            if (directory / "replay-manifest.json").exists():
                historical, current, episodes = replay_records(directory, inputs)
                if current["domain"] != item["domain"] or current["kind"] != item["kind"]:
                    raise ValueError("replay lens domain/kind differs")
                # Prose replays must trace to a completed accepted §14 capture.
                source_dirs = {
                    str(Path(row["identity"]["path"]).parent) for row in current["sources"]
                }
                if any(ep["kind"] == "prose" for ep, _ in episodes):
                    if len(source_dirs) != 1:
                        raise ValueError("prose replay must use one accepted capture set")
                    source_manifest, _, accepted = prose_records(
                        next(iter(source_dirs)), inputs, tokenizer
                    )
                    if source_manifest != historical or _labels(
                        [ep for ep, _ in accepted]
                    ) != _labels([ep for ep, _ in episodes]):
                        raise ValueError("prose replay includes unaccepted source episodes")
                elif historical != inputs.json(PILOT / "manifest.json"):
                    raise ValueError("evaluation is not the disjoint pilot")
            else:
                historical, current, episodes = prose_records(directory, inputs, tokenizer)
                if item["kind"] != "hosted-jacobian":
                    raise ValueError("original capture is hosted; fitted lenses require replay")
            if (
                current["snapshot"] != snapshot
                or current["lens_sha256"] != metadata["npz_sha256"]
                or current["layers"] != plan["layers"] + [len(plan["layers"]) + 1]
            ):
                raise ValueError("current evaluation lens/snapshot/layers differ")
            for episode, events in episodes:
                if episode["label"] in seen:
                    raise ValueError("episode appears in multiple evaluation sets")
                seen.add(episode["label"])
                rows.extend(
                    summarize_episode(
                        events,
                        episode,
                        tokenizer,
                        layers=current["layers"],
                        lens_kind=item["kind"],
                        validation=validation,
                    )
                )
            records.append(
                {
                    "directory": str(directory),
                    "metadata": current,
                    "excluded_dropped": historical.get("dropped", []),
                    "emitted_count_convention": "producer emitted events, including EOS when recorded",
                }
            )
        products.append(
            {
                "name": name,
                "model": item["model"],
                "domain": item["domain"],
                "kind": item["kind"],
                "rows": rows,
                "lens_metadata": metadata,
                "records": records,
                "identity": identity,
                "instrument": item["instrument"],
                "instrument_scope": "Cached/reference instrument path on this snapshot; not regression Jacobian map agreement.",
                "overlap": OVERLAP,
            }
        )
    inputs.file(ATLAS_SCRIPT)
    inputs.file(Path(__file__))
    from local_llm_lab.spawn import run

    commit = run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    conventions = {
        "foreknowledge": "rank <= 10; source t, target t+h; exact episode/turn/source/target/token/horizon final pairs",
        "prompt_agreement": "top1 at source t equals prompt_ids[t+1], only t+1 inside prompt; source-position span",
        "pooling": "none across episodes or layers",
        "primary_layer": None,
        "interpretation": "No aggregate interpretation emitted. Requirement §5.2 references h4/h8 explicitly; h1/h4/h8 remain separate.",
        "regression_h1": "retained, biased_by_construction, excluded from reading rule",
    }
    table = specificity(products)
    inputs.recheck()
    provenance = {
        "source_commit": commit,
        "inputs": dict(inputs.hashes),
        "metric_conventions": conventions,
    }
    # Construct JSON first so nonfinite data cannot leave a nominally complete output.
    documents = {
        item["name"] + ".json": json.dumps(item | provenance, indent=2, allow_nan=False) + "\n"
        for item in products
    }
    page = render_page({"sets": products, "specificity": table, **provenance})
    inputs.recheck()
    output.mkdir(parents=True, exist_ok=False)
    for filename, document in documents.items():
        with (output / filename).open("x") as stream:
            stream.write(document)
    with (output / "profiles.html").open("x") as stream:
        stream.write(page)
    with (output / "specificity.json").open("x") as stream:
        json.dump({"cells": table, **provenance}, stream, indent=2, allow_nan=False)
    inputs.recheck()
    output_hashes = {p.name: file_sha256(p) for p in sorted(output.iterdir())}
    with (output / "complete.json").open("x") as stream:
        json.dump(
            {
                "status": "complete",
                "outputs": output_hashes,
                **provenance,
            },
            stream,
            indent=2,
            allow_nan=False,
        )
    return {"status": "complete", "lens_sets": len(products), "output": str(output)}


def render_page(data):
    """Standalone, safe embedded JSON; every curve is one episode/span/horizon."""
    payload = (
        json.dumps(data, allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lens layer profiles</title><style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px system-ui;color:#142333;background:#f7f9fc}label{display:inline-flex;flex-direction:column;margin:0 1rem 1rem 0;gap:.4rem}select{padding:.5rem;font:inherit}svg{width:100%;background:white;border:1px solid #ccd5df}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:.6rem;border-bottom:1px solid #ccd5df}a{color:#194ba8}:focus-visible{outline:3px solid #d87500}pre{white-space:pre-wrap}</style>
<h1>Lens layer profiles</h1><p>Every curve is a single episode, kind, span and summary. All nonfinal layers are shown; no primary layer and no pooled default. Exported JSON is authoritative.</p>
<p>Regression horizon 1 is biased by construction and excluded from the reading rule. Held prose windows selected the ridge weight among five values; pilot episodes are disjoint evaluation. Missing cross-reads remain unavailable.</p>
<div id="controls"></div><p id="status" role="status"></p><p id="export"></p><svg id="plot" viewBox="0 0 960 340" role="img" aria-label="Layer profile and final-distribution base rate"></svg>
<p>Blue: selected summary. Orange dashed: its matched final-distribution control. Null values indicate zero denominator.</p>
<table><thead><tr><th>Layer</th><th>Numerator / n</th><th>Share</th><th>Final numerator / n</th><th>Final share</th><th>Map check</th><th>Eligible</th></tr></thead><tbody id="rows"></tbody></table>
<h2>Specificity availability</h2><table><thead><tr><th>Lens set / training domain</th><th>Evaluation material</th><th>Status</th></tr></thead><tbody id="cells"></tbody></table>
<script id="data" type="application/json">"""
        + payload
        + """</script><script>
'use strict';const data=JSON.parse(document.getElementById('data').textContent), controls=document.getElementById('controls'), choices={};
const dims=['lens set','episode kind','episode','summary','span','horizon'];
for(const d of dims){const label=document.createElement('label');label.append(document.createTextNode(d));const s=document.createElement('select');s.setAttribute('aria-label',d);label.append(s);controls.append(label);choices[d]=s;s.onchange=()=>refresh(dims.indexOf(d)+1);}
function options(d,values,reset){const select=choices[d],old=select.value;select.replaceChildren();for(const value of [...new Set(values)]){const o=document.createElement('option');o.value=String(value);o.textContent=String(value);select.append(o);}if(!reset&&values.map(String).includes(old))select.value=old;}
function refresh(start=0){options('lens set',data.sets.map(x=>x.name),start===0);const set=data.sets.find(x=>x.name===choices['lens set'].value);let rows=set?set.rows:[];
const fields=['episode_kind','episode','summary','span','horizon'];for(let i=1;i<dims.length;i++){const field=fields[i-1];options(dims[i],rows.map(r=>r[field]===null?'prompt':r[field]),i>=start);rows=rows.filter(r=>String(r[field]===null?'prompt':r[field])===choices[dims[i]].value);}
rows=rows.slice().sort((a,b)=>a.layer-b.layer);const svg=document.getElementById('plot');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';function el(tag,attrs,text){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;svg.append(e);return e;}
const max=Math.max(2,...(set?set.rows.map(r=>r.final_layer-1):[2])),x=l=>55+(l-1)/(max-1)*850,y=v=>290-v*250;el('line',{x1:55,y1:290,x2:905,y2:290,stroke:'#566'});for(const v of [0,.25,.5,.75,1]){el('line',{x1:55,y1:y(v),x2:905,y2:y(v),stroke:'#e4e8ec'});el('text',{x:10,y:y(v)+5,'font-size':12},String(v));}el('text',{x:440,y:328,'font-size':14},'Residual layer');
let points=[];for(const r of rows){if(r.share===null){if(points.length)el('polyline',{points:points.join(' '),fill:'none',stroke:'#245dc1','stroke-width':3});points=[];continue;}points.push(x(r.layer)+','+y(r.share));el('circle',{cx:x(r.layer),cy:y(r.share),r:4,fill:'#245dc1'});el('text',{x:x(r.layer)-3,y:310,'font-size':11},String(r.layer));}if(points.length)el('polyline',{points:points.join(' '),fill:'none',stroke:'#245dc1','stroke-width':3});
const base=rows.length?rows[0].base_rate.share:null;if(base!==null&&rows.every(r=>r.base_rate.share===base&&r.base_rate.n===rows[0].base_rate.n))el('line',{x1:55,x2:905,y1:y(base),y2:y(base),stroke:'#a9540b','stroke-width':2,'stroke-dasharray':'7 5'});
const body=document.getElementById('rows');body.replaceChildren();for(const r of rows){const tr=document.createElement('tr');for(const value of[r.layer,r.numerator+' / '+r.n,r.share,r.base_rate.numerator+' / '+r.base_rate.n,r.base_rate.share,r.map_check,r.interpretive_eligible]){const td=document.createElement('td');td.textContent=value===null?'null':String(value);tr.append(td);}body.append(tr);}
document.getElementById('status').textContent=rows.length?`${rows[0].episode}: ${rows[0].span_role}; ${rows[0].target_role}. ${rows[0].biased_by_construction?'Biased by construction; excluded.':''} Per-layer validation appears below.`:'Unavailable: no observations for this selection.';
const exportNode=document.getElementById('export');exportNode.replaceChildren();if(set){const a=document.createElement('a');a.href=encodeURIComponent(set.name)+'.json';a.textContent='Open authoritative '+set.name+' JSON';exportNode.append(a);}}
for(const c of data.specificity){const tr=document.createElement('tr');for(const v of[c.lens_set+' / '+c.training_domain,c.evaluation_material,c.status]){const td=document.createElement('td');td.textContent=v;tr.append(td);}document.getElementById('cells').append(tr);}refresh();
</script></html>"""
    )
