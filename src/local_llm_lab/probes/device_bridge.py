"""Offline A1/A2 runner. Admission finishes for the entire request before numeric work.

All inputs are local, explicit paths. The positions document seals the capture files and names
cells; token indices are independently reconstructed from the rendered corpus and local tokenizer.
The capture schema is workspace_capture_v2's completed manifest/index/NumPy memmaps. No model is
loaded and no cache resolver, network client, accelerator, or device launcher is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from local_llm_lab import hf_text
from local_llm_lab.models import base_of_artifact
from local_llm_lab.probes import sae_bridge as B
from local_llm_lab.probes.device_lens_admission import load_admitted_device_lens

INTERPRETATION_LIMITS = [
    "A large feature contribution to a lens score is a contribution to that score, at that layer, "
    "under that averaging convention. It is not a token probability, not a causal derivative, "
    "and not evidence of a workspace, a maintained state, or flexible access.",
    "A description we generate is not a published label and no artifact may present it as one.",
]
CAPTURE_FILES = (
    "manifest.json",
    "index.jsonl",
    "progress.jsonl",
    "residual_note.npy",
    "residual_act.npy",
)


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _hash(path, algorithm="sha256"):
    path = Path(path)
    if algorithm not in ("sha256", "blob"):
        raise ValueError(f"dictionary digest algorithm {algorithm!r} is unsupported")
    h = (
        hashlib.sha256()
        if algorithm == "sha256"
        else hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    )
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _equal(actual, expected, name):
    if actual != expected:
        raise ValueError(f"{name} mismatch: actual {actual!r}; declared {expected!r}")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _threshold(value, name, *, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def _config(path, stage):
    conf = _json(path)
    _equal(conf.get("schema_version"), 1, "config schema_version")
    for name in (
        "model_base",
        "checkpoint_sha256",
        "lens_sha256",
        "dictionary_repo",
        "dictionary_folder",
        "error_budget",
    ):
        if not conf.get(name):
            raise ValueError(f"config missing {name}")
    try:
        budget = B.ErrorBudget.from_dict(conf["error_budget"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"config error_budget: {exc}") from exc
    for section in {"a1", "a2"} if stage == "both" else {stage}:
        if section not in conf:
            raise ValueError(f"config missing {section}")
        cfg = conf[section]
        _integer(cfg.get("k"), f"{section}.k")
        if section == "a1":
            _integer(cfg.get("chunk"), "a1.chunk")
            _integer(cfg.get("control_layer"), "a1.control_layer")
            features = cfg.get("check_features")
            if (
                not isinstance(features, list)
                or not features
                or len(set(features)) != len(features)
            ):
                raise ValueError("a1.check_features must be a nonempty unique list")
            for feature in features:
                _integer(feature, "a1.check_features", 0)
            _threshold(cfg.get("two_product_tolerance"), "a1.two_product_tolerance")
            _threshold(cfg.get("maximum_control_overlap"), "a1.maximum_control_overlap", maximum=1)
            if cfg["maximum_control_overlap"] >= 1:
                raise ValueError("a1.maximum_control_overlap must be below 1")
            _threshold(cfg.get("convention_threshold"), "a1.convention_threshold", maximum=1)
            if cfg["convention_threshold"] == 0:
                raise ValueError("a1.convention_threshold must be positive")
        else:
            _threshold(cfg.get("identity_tolerance"), "a2.identity_tolerance")
    return conf, budget


def _dictionary(directory, receipt_path, conf, hidden):
    receipt = _json(receipt_path)
    for key in ("repo", "folder"):
        _equal(receipt.get(key), conf[f"dictionary_{key}"], f"dictionary receipt {key}")
    config = _json(directory / "config.json")
    _equal(receipt.get("config"), config, "dictionary receipt config")
    for name in ("config.json", "params.safetensors"):
        _receipt_file(directory / name, receipt)
    _equal(
        base_of_artifact(config.get("model_name", "")), conf["model_base"], "dictionary model_name"
    )
    _equal(config.get("architecture"), "jump_relu", "dictionary architecture")
    _equal(config.get("hf_hook_point_in"), config.get("hf_hook_point_out"), "dictionary hook")
    layer = B.layer_for_hook(config.get("hf_hook_point_in", ""))
    folder_layer = re.search(r"(?:^|/)layer_(\d+)_width_", receipt["folder"])
    if folder_layer is None or int(folder_layer[1]) + 1 != layer:
        raise ValueError("dictionary receipt folder does not identify the configured hook layer")
    width = _integer(config.get("width"), "dictionary width")
    shapes = {
        "w_enc": [hidden, width],
        "b_enc": [width],
        "threshold": [width],
        "w_dec": [width, hidden],
        "b_dec": [hidden],
    }
    header, _ = B.safetensors_header(directory / "params.safetensors")
    _equal(set(header), set(shapes), "dictionary tensor names")
    for name, shape in shapes.items():
        _equal(header[name]["shape"], shape, f"dictionary {name} shape")
        if header[name]["dtype"] not in ("BF16", "F16", "F32", "F64"):
            raise ValueError(f"dictionary {name} dtype is unsupported")
    # A metadata view suffices for hook/pairing admission; weights are loaded only afterwards.
    view = SimpleNamespace(
        config=config, hidden_size=hidden, width=width, hook_point=config["hf_hook_point_in"]
    )
    return view, receipt


def _receipt_file(path, receipt):
    item = receipt.get("files", {}).get(path.name)
    if not isinstance(item, dict):
        raise ValueError(f"dictionary digest receipt missing {path.name}")
    _equal(path.stat().st_size, item.get("bytes"), f"dictionary {path.name} digest byte count")
    _equal(_hash(path, item.get("algorithm")), item.get("digest"), f"dictionary {path.name} digest")


def _checkpoint_index(snapshot, checkpoint):
    """The numerical loader's index may only name verified tensor owners."""
    index = snapshot / "model.safetensors.index.json"
    owners = {}
    for name in checkpoint["sha256"]:
        if name.endswith(".safetensors"):
            header, _ = B.safetensors_header(snapshot / name)
            owners.update({tensor: name for tensor in header})
    if index.exists():
        _equal(_json(index).get("weight_map"), owners, "checkpoint index weight_map")
        return _hash(index)
    if set(owners.values()) != {"model.safetensors"}:
        raise ValueError("checkpoint index is required for sharded readout")
    return None


def _examples(directory, receipt, width, vocab, k):
    from safetensors import safe_open

    path = directory / "examples.safetensors"
    _receipt_file(path, receipt)
    with safe_open(path, framework="np") as archive:
        if "top_tokens" not in archive.keys():  # noqa: SIM118 -- safe_open is not iterable
            raise ValueError("examples.safetensors missing top_tokens")
        tokens = archive.get_tensor("top_tokens")
    if tokens.ndim != 2 or tokens.shape[0] != width or tokens.shape[1] < k:
        raise ValueError("examples top_tokens shape disagrees with dictionary width or requested k")
    if tokens.dtype.kind not in "iu" or np.any(tokens < 0) or np.any(tokens >= vocab):
        raise ValueError("examples top_tokens must carry integer vocabulary token ids")
    return tokens[:, :k]


def _pairing_tables(path):
    """Accept the existing one-registration table, or a list at a layer for distinct cells."""
    raw = _json(path)
    if not isinstance(raw, dict):
        raise ValueError("pairing registration must be a model-keyed table")
    out = {}
    for base, entry in raw.items():
        if base.startswith("_"):
            continue
        shell = {k: v for k, v in entry.items() if k != "measured_pairings"}
        B._validate_anchor_table({base: shell})
        for layer, registrations in entry.get("measured_pairings", {}).items():
            _integer(int(layer), "pairing layer")
            records = registrations if isinstance(registrations, list) else [registrations]
            for record in records:
                B._validate_anchor_table({base: {**shell, "measured_pairings": {layer: record}}})
            out[base, layer] = (shell, records)
    return out


def _cell_pairing(tables, base, layer, identity):
    shell, records = tables.get((base, str(layer)), ({}, []))
    exact = [r for r in records if r["pair"] == identity]
    if len(exact) > 1:
        raise ValueError(f"pairing registration is ambiguous at layer {layer}")
    if records and not exact:
        differences = [
            {
                k: {"measured": r["pair"][k], "reading": identity[k]}
                for k in B.PAIR_FIELDS
                if r["pair"][k] != identity[k]
            }
            for r in records
        ]
        raise ValueError(f"pairing registration mismatch at layer {layer}: {differences}")
    # Retain a nonmatching record for the existing guard to explain the exact differing fields.
    record = exact[0] if exact else (records[0] if records else None)
    return {base: {**shell, "measured_pairings": {} if record is None else {str(layer): record}}}


def _capture(
    directory, positions_path, corpus_path, snapshot, metadata, lens, conf, dictionary, pairings
):
    from tokenizers import Tokenizer

    seal = _json(positions_path)
    _equal(seal.get("schema_version"), 1, "positions schema_version")
    hashes = seal.get("capture_files_sha256", {})
    _equal(set(hashes), set(CAPTURE_FILES), "capture hash manifest files")
    for name in CAPTURE_FILES:
        _equal(_hash(directory / name), hashes[name], f"capture {name} hash")
    _equal(_hash(corpus_path), seal.get("corpus_sha256"), "corpus sha256")
    _equal(_hash(snapshot / "tokenizer.json"), seal.get("tokenizer_sha256"), "tokenizer sha256")
    manifest = _json(directory / "manifest.json")
    _equal(manifest.get("schema_version"), 1, "capture schema_version")
    _equal(
        manifest.get("load_report_sha256"),
        conf["checkpoint_sha256"],
        "capture checkpoint hash manifest",
    )
    if not conf.get("capture_model"):
        raise ValueError("config missing capture_model label; checkpoint hashes establish the base")
    _equal(manifest.get("model"), conf["capture_model"], "capture model label")
    _equal(manifest.get("precision"), "float32", "capture precision for float32 residual memmaps")
    _integer(manifest.get("width"), "capture width")
    _equal(manifest.get("memmap_row"), "global index minus rows_from", "capture memmap_row")
    _equal(
        manifest.get("positions"),
        ["P_note: last prompt token", "P_act: token before the tool-name token"],
        "capture positions convention",
    )
    _equal(
        manifest.get("tf32"),
        {"matmul_allow_tf32": False, "float32_matmul_precision": "highest"},
        "capture tf32",
    )
    count = _integer(manifest.get("decisions"), "capture decisions")
    low = _integer(manifest.get("rows_from"), "capture rows_from", 0)
    _equal(manifest.get("rows_to"), low + count, "capture row interval")
    layers = manifest.get("repo_layers")
    if (
        not isinstance(layers, list)
        or not layers
        or layers != sorted(set(layers))
        or any(type(x) is not int or not 1 <= x <= lens.num_layers for x in layers)
    ):
        raise ValueError("capture repo_layers must be unique ascending repository layers")
    layer = B.layer_for_hook(dictionary.hook_point)
    if layer not in layers:
        raise ValueError(f"capture repo_layers missing dictionary layer {layer}")
    arrays = {}
    for name in ("note", "act"):
        array = np.load(directory / f"residual_{name}.npy", mmap_mode="r", allow_pickle=False)
        _equal(
            array.shape, (count, len(layers), lens.hidden_size), f"capture residual_{name} shape"
        )
        _equal(array.dtype, np.dtype("float32"), f"capture residual_{name} dtype")
        arrays[f"P_{name}"] = array
    events = _rows(directory / "progress.jsonl")
    for event, field, value in (
        ("corpus", "corpus_sha256", seal["corpus_sha256"]),
        ("loaded", "maps_sha256", metadata["source_archive_sha256"]),
        ("done", "decisions", count),
    ):
        selected = [e for e in events if e.get("event") == event]
        if len(selected) != 1:
            raise ValueError(f"capture progress must contain exactly one {event} event")
        _equal(selected[0].get(field), value, f"capture {event}.{field}")
    index = _rows(directory / "index.jsonl")
    _equal([r.get("i") for r in index], list(range(low, low + count)), "capture index rows")
    corpus, seen = [], set()
    for row in _rows(corpus_path):
        m = row.get("metadata", {})
        if "task_id" not in m:
            continue
        key = (m["task_id"], m["step"])
        if key not in seen:
            seen.add(key)
            corpus.append(row)
    _equal(manifest.get("corpus_decisions"), len(corpus), "capture corpus_decisions")
    if low + count > len(corpus):
        raise ValueError("capture row interval exceeds corpus")
    _equal(seal.get("reduction"), "per-position, no reduction", "positions reduction")
    _equal(seal.get("endpoint"), "pre-final-norm", "positions endpoint")
    cells = seal.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("positions cells must be nonempty")
    tok = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tables, readings, selected_residuals, used = _pairing_tables(pairings), [], [], set()
    for cell in cells:
        i, position = cell.get("row"), cell.get("position")
        _integer(i, "positions row", 0)
        if not low <= i < low + count or position not in arrays or (i, position) in used:
            raise ValueError(
                "positions cell is duplicate or outside the capture row/position domain"
            )
        used.add((i, position))
        for field in ("token_index", "token_id", "context_tokens"):
            _integer(cell.get(field), f"positions {field}", 1 if field == "context_tokens" else 0)
        row, ix = corpus[i], index[i - low]
        for field in ("task_id", "step", "family", "variant", "recovery"):
            _equal(ix.get(field), row["metadata"].get(field), f"capture index {field}")
        prompt = tok.encode(row["prompt"], add_special_tokens=False)
        completion = tok.encode(row["completion"], add_special_tokens=False)
        match = re.search(r'"name":\s*"([a-z_]+)"', row["completion"])
        if match is None:
            raise ValueError(f"corpus row {i} has no tool-name token")
        target = next(
            (j for j, (a, b) in enumerate(completion.offsets) if a <= match.start(1) < b), None
        )
        if target is None or not prompt.ids or not completion.ids:
            raise ValueError(f"corpus row {i} cannot reproduce token offsets")
        actual = {
            "P_note": len(prompt.ids) - 1,
            "P_act": len(prompt.ids) + target - 1,
            "n_prompt_tokens": len(prompt.ids),
            "n_note_tokens": target,
        }
        for field, value in actual.items():
            _equal(ix.get(field), value, f"capture index {field}")
        _equal(cell.get("token_index"), actual[position], "positions token_index")
        _equal(
            cell.get("token_id"),
            completion.ids[0 if position == "P_note" else target],
            "positions token_id",
        )
        _equal(cell.get("context_tokens"), actual["P_act"] + 2, "positions context_tokens")
        reading = {
            "positions": [actual[position]],
            "reduction": seal["reduction"],
            "endpoint": seal["endpoint"],
            "context_tokens": cell["context_tokens"],
        }
        identity = B.reading_identity(
            metadata["nu"],
            lens_sha256=lens.sha256,
            capture_dtype=manifest["precision"],
            capture_batch=manifest["width"],
            reading=reading,
        )
        table = _cell_pairing(tables, conf["model_base"], layer, identity)
        B.hook_alignment(
            dictionary,
            lens,
            base=conf["model_base"],
            nu=metadata["nu"],
            capture_dtype=manifest["precision"],
            capture_batch=manifest["width"],
            reading=reading,
            pairing_table=table,
        )
        fit_record = B.fit_precision_record(
            metadata["nu"],
            declared=None,
            layer=layer,
            base=conf["model_base"],
            lens_sha256=lens.sha256,
            capture_dtype=manifest["precision"],
            capture_batch=manifest["width"],
            reading=reading,
            pairing_table=table,
        )
        residual = arrays[position][i - low, layers.index(layer)]
        if not np.isfinite(residual).all():
            raise ValueError(f"capture row {i} {position} residual is not finite")
        readings.append(
            {**cell, "target_source": "expert corpus completion", "fit_precision": fit_record}
        )
        selected_residuals.append(residual)
    outside = [
        r for r in readings if not 16 <= r["token_index"] <= 126 or r["context_tokens"] != 128
    ]
    domain = seal.get("domain_of_validity")
    if outside:
        if (
            not isinstance(domain, dict)
            or not isinstance(domain.get("statement"), str)
            or not domain["statement"].strip()
        ):
            raise ValueError(
                "positions outside fit domain require the lens owner's domain_of_validity statement"
            )
        for field, expected in (
            ("lens_sha256", lens.sha256),
            ("nu_sha256", B.nu_digest(metadata["nu"])),
            ("positions", sorted({r["token_index"] for r in outside})),
            ("context_tokens", sorted({r["context_tokens"] for r in outside})),
        ):
            _equal(domain.get(field), expected, f"domain_of_validity {field}")
    return (
        selected_residuals,
        readings,
        {
            "manifest": manifest,
            "files_sha256": hashes,
            "positions_sha256": _hash(positions_path),
            "corpus_sha256": seal["corpus_sha256"],
            "tokenizer_sha256": seal["tokenizer_sha256"],
            "pairings_sha256": _hash(pairings),
            "domain_of_validity": domain,
        },
    )


def run_bridge(
    *,
    model_snapshot,
    dictionary_dir,
    dictionary_receipt,
    lens_archive,
    config,
    output_dir,
    stage,
    capture_dir=None,
    positions=None,
    corpus=None,
    pairings=None,
    dry_run=False,
):
    """Admit every input first; dry-run emits only the admission receipt, otherwise run A1/A2."""
    if stage not in ("a1", "a2", "both"):
        raise ValueError("stage must be a1, a2 or both")
    snapshot, directory, archive, out = map(
        Path, (model_snapshot, dictionary_dir, lens_archive, output_dir)
    )
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    conf, budget = _config(config, stage)
    checked = ["config"]
    checkpoint = hf_text.checkpoint_metadata(snapshot)
    _equal(checkpoint["sha256"], conf["checkpoint_sha256"], "checkpoint hash manifest")
    index_sha256 = _checkpoint_index(snapshot, checkpoint)
    checked.append("checkpoint")
    lens, metadata = load_admitted_device_lens(
        archive,
        checkpoint=snapshot,
        model_base=conf["model_base"],
        expected_sha256=conf["lens_sha256"],
    )
    checked.append("lens")
    dictionary, receipt = _dictionary(directory, dictionary_receipt, conf, lens.hidden_size)
    checked.append("dictionary")
    layer = B.hook_alignment(dictionary, lens, base=conf["model_base"], nu=metadata["nu"])
    # Verify the selected readout's shape from headers before loading any numeric bridge tensors.
    text = checkpoint["tensors"]
    heads = [k for k in text if k.endswith("lm_head.weight")]
    embeds = [k for k in text if k.endswith("embed_tokens.weight")]
    norms = [k for k in text if k.endswith(".norm.weight") and ".layers." not in k]
    head = heads if heads else embeds
    if len(head) != 1 or len(norms) != 1:
        raise ValueError("checkpoint must identify exactly one readout and final norm")
    shape = text[head[0]]["shape"]
    if (
        len(shape) != 2
        or shape[1] != lens.hidden_size
        or text[norms[0]]["shape"] != [lens.hidden_size]
    ):
        raise ValueError("checkpoint readout/norm shapes disagree with lens")
    for name in (head[0], norms[0]):
        if text[name]["dtype"] not in ("BF16", "F16", "F32", "F64"):
            raise ValueError(f"checkpoint {name} readout dtype is unsupported")
    vocab = shape[0]
    shipped = None
    if stage in ("a1", "both"):
        a1 = conf["a1"]
        if a1["k"] > vocab or max(a1["check_features"]) >= dictionary.width:
            raise ValueError("a1 k/check_features exceeds vocabulary/dictionary")
        B.lens_map_for_layer(lens, a1["control_layer"])
        if a1["control_layer"] == layer:
            raise ValueError("a1 control_layer must differ from dictionary layer")
        shipped = _examples(directory, receipt, dictionary.width, vocab, a1["k"])
        checked.append("examples")
    capture_inputs = (capture_dir, positions, corpus, pairings)
    residuals, readings, capture_provenance = [], [], None
    if stage in ("a2", "both") or any(x is not None for x in capture_inputs):
        if any(x is None for x in capture_inputs):
            raise ValueError(
                "capture_dir, positions, corpus and pairings are required together for A2"
            )
        residuals, readings, capture_provenance = _capture(
            Path(capture_dir),
            Path(positions),
            Path(corpus),
            snapshot,
            metadata,
            lens,
            conf,
            dictionary,
            Path(pairings),
        )
        checked.extend(["capture", "positions", "pairings"])
    provenance = {
        "checkpoint": {
            "path": str(snapshot),
            "sha256": checkpoint["sha256"],
            "index_sha256": index_sha256,
        },
        "lens": metadata,
        "dictionary": receipt,
        "readings": readings,
        "capture": capture_provenance,
        "config_sha256": _hash(config),
        "labels": "unlabelled",
        "labels_reason": B.UNLABELLED_REASON,
    }
    result = {
        "schema_version": 1,
        "stage": stage,
        "status": "admitted-dry-run" if dry_run else "complete",
        "checked": checked,
        "config": conf,
        "error_budget": budget.as_dict(),
        "provenance": provenance,
        "interpretation_limits": list(INTERPRETATION_LIMITS),
    }
    # This is the numeric boundary. Every requested cell, receipt and pairing has been checked.
    if not dry_run:
        dictionary = B.load_dictionary(
            directory / "params.safetensors",
            directory / "config.json",
            source=f"{receipt['repo']}/{receipt['folder']}",
        )
        unembedding = B.load_unembedding(snapshot)
        if not np.isfinite(unembedding.weight).all() or not np.isfinite(unembedding.gain).all():
            raise ValueError("checkpoint readout contains nonfinite values")
        provenance["bridge"] = B.bridge_provenance(
            dictionary,
            lens,
            unembedding,
            layer,
            dictionary_repo=receipt["repo"],
            dictionary_folder=receipt["folder"],
            fit_precision=B.fit_precision_record(metadata["nu"], declared=None),
        )
        lens_map = B.lens_map_for_layer(lens, layer)
        if stage in ("a1", "both"):
            a1 = conf["a1"]
            raw, _ = B.feature_scores(
                dictionary, unembedding, None, k=a1["k"], gain=False, chunk=a1["chunk"]
            )
            raw_overlap = float(B.overlap_at_k(shipped, raw).mean())
            gained, _ = B.feature_scores(
                dictionary, unembedding, None, k=a1["k"], gain=True, chunk=a1["chunk"]
            )
            discriminator = B.convention_check(
                shipped, raw, gained, threshold=a1["convention_threshold"]
            )
            if discriminator["stop"]:
                result["status"] = "refused-numeric-check"
                result["a1"] = {
                    "readout_heading": f"Our top-token readout at repository layer {layer}",
                    "raw_arm_overlap_recorded_first": raw_overlap,
                    "convention_check": discriminator,
                }
            else:
                ids, scores = B.feature_scores(
                    dictionary, unembedding, lens_map, k=a1["k"], chunk=a1["chunk"]
                )
                gaps, membership = [], []
                for feature in a1["check_features"]:
                    column = B.feature_score_column(dictionary, unembedding, lens_map, feature)
                    direct_ids, _ = B.top_tokens(column, a1["k"])
                    membership.append(set(direct_ids[0]) == set(ids[feature]))
                    gaps.append(float(np.max(np.abs(column[ids[feature]] - scores[feature]))))
                control = B.negative_control(
                    dictionary,
                    unembedding,
                    lens,
                    layer,
                    a1["control_layer"],
                    k=a1["k"],
                    features=np.array(a1["check_features"]),
                )
                control["maximum_control_overlap"] = a1["maximum_control_overlap"]
                control["passed"] = control["mean_overlap"] <= a1["maximum_control_overlap"]
                two = {
                    "features": a1["check_features"],
                    "max_absolute_gap": max(gaps),
                    "tolerance": a1["two_product_tolerance"],
                    "top_token_membership_matches": all(membership),
                    "passed": all(membership) and max(gaps) <= a1["two_product_tolerance"],
                }
                result["a1"] = {
                    "readout_heading": f"Our top-token readout at repository layer {layer}",
                    "raw_arm_overlap_recorded_first": raw_overlap,
                    "convention_check": discriminator,
                    "two_products": two,
                    "negative_control": control,
                    "top_tokens": ids.tolist(),
                    "top_scores": scores.tolist(),
                }
                if discriminator["stop"] or not control["passed"] or not two["passed"]:
                    result["status"] = "refused-numeric-check"
        if stage in ("a2", "both") and result["status"] == "complete":
            result["a2"] = [
                {
                    **reading,
                    **B.decompose_position(
                        dictionary,
                        unembedding,
                        lens_map,
                        h,
                        reading["token_id"],
                        error_budget=budget,
                        k=conf["a2"]["k"],
                        tolerance=conf["a2"]["identity_tolerance"],
                    ),
                }
                for reading, h in zip(readings, residuals, strict=True)
            ]
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    out.mkdir(parents=True)
    (out / ("admission.json" if dry_run else "result.json")).write_text(payload, encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model-snapshot",
        "dictionary-dir",
        "dictionary-receipt",
        "lens-archive",
        "config",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("capture-dir", "positions", "corpus", "pairings"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--stage", required=True, choices=("a1", "a2", "both"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_bridge(**vars(args))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"refused: {exc}\n")
    print(json.dumps({"status": result["status"], "checked": result["checked"]}))
    return 0 if result["status"] in ("complete", "admitted-dry-run") else 2
