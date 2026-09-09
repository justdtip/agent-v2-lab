"""Requirements §14: frozen held-prefix plans and EOS/cap-only prose capture.

Planning imports no MLX. Execution is explicit, exclusive and never resumes a run.
Authored tails are not passed to capture or scored as emitted foreknowledge.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from local_llm_lab.forward import encode_prompt
from local_llm_lab.pipeline.lens_fitting.corpus import TOKENIZER_PATTERNS, read_corpus
from local_llm_lab.pipeline.lens_fitting.runtime import (
    configure_allocator_cache,
    load_runtime,
    resolve_snapshot,
    resource_snapshot,
    snapshot_identity,
)
from local_llm_lab.pipeline.live_lens.instruments import (
    LensIdentity,
    LensMaps,
    file_sha256,
)
from local_llm_lab.pipeline.live_lens.session import CaptureSession, LensReadout, RecordWriter

PREFIX_TOKENS = 824
MAX_TOKENS = 200
MIN_TOKENS = 32
HELD_WINDOWS = 51
SEED = 20260902
OVERLAP = "These held windows selected the ridge weight among five values; no other fit overlap."
README = """# Held prose capture — requirements §14

Registration precedes checkpoint loading. plan.json freezes all 51 held windows in corpus
order, their 824-token raw prefixes and hashes, model snapshot, tokenizer and hosted lens.
No chat template. Greedy continuation: 200 emitted tokens maximum, EOS or cap only.
The emitted count includes a terminal EOS response, matching the pilot. Continuations
under 32 emitted tokens are counted and retained under dropped/. Authored trailing
window tokens are never scored as foreknowledge. All layers include final identity.

These held windows selected the ridge weight among five values; this overlap is disclosed.
No disjoint test-shard download is authorised. No raw logits or activations are retained.
The native ForwardLedger preserves partitions and un-emitted lookahead independently
of emitted count. A complete manifest requires exactly one outcome for every window.
Partial runs cannot resume or certify completion. Allocator caching is disabled; byte
measurements do not establish buffer-count safety. Native acceptance remains separate.
"""


def _bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_bytes(value) + b"\n")


def corpus_tokenizer(corpus_path, spec, *, revision="main"):
    """Recover filename roles from one verified snapshot, including HF blob symlinks.

    Corpus descriptors resolve symlinks to opaque blobs. Every tokenizer/config asset
    in the selected offline snapshot must map back to exactly one descriptor file by
    both resolved path and content hash; no newest-snapshot search or download occurs.
    """
    read_corpus(corpus_path)
    manifest = json.loads(Path(corpus_path).read_bytes())
    if manifest["model_hf_id"] != spec.hf_id:
        raise ValueError("corpus tokenizer model mismatch")
    snapshot = resolve_snapshot(spec, revision=revision)
    directory = Path(snapshot["snapshot_path"])
    files = sorted(
        {
            path
            for pattern in TOKENIZER_PATTERNS
            for path in directory.glob(pattern)
            if path.is_file()
        }
    )
    bound = {
        (str(Path(row["path"]).resolve(strict=True)), row["sha256"])
        for row in manifest["tokenizer"]["files"]
    }
    actual = {(str(path.resolve(strict=True)), file_sha256(path)) for path in files}
    roles = {path.relative_to(directory).as_posix() for path in files}
    if actual != bound or len(files) != len(bound) or "tokenizer_config.json" not in roles:
        raise ValueError("snapshot tokenizer assets do not match frozen corpus descriptor")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if snapshot_identity(directory, hf_id=spec.hf_id) != snapshot:
        raise ValueError("snapshot changed while loading corpus tokenizer")
    return tokenizer


def make_plan(corpus_path, spec, lens_path, *, lens_sha256, tokenizer, revision="main"):
    """Pure preflight binds exact corpus, snapshot, raw prompt and hosted lens bytes."""
    corpus_path, lens_path = Path(corpus_path).absolute(), Path(lens_path).absolute()
    before = file_sha256(corpus_path)
    rows = read_corpus(corpus_path)
    manifest = json.loads(corpus_path.read_bytes())
    if before != file_sha256(corpus_path):
        raise ValueError("corpus changed during planning")
    if manifest["domain"] != "prose" or manifest["model_hf_id"] != spec.hf_id:
        raise ValueError("prose corpus/model mismatch")
    descriptor = manifest["tokenizer"]
    if (
        descriptor["bos_policy"]
        != "add specials unless text already starts with tokenizer.bos_token"
        or descriptor["bos_token"] != tokenizer.bos_token
        or descriptor["class"] != f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    ):
        raise ValueError("corpus tokenizer descriptor drift")
    held = [row for row in rows if row["split"] == "held"]
    if len(held) != HELD_WINDOWS:
        raise ValueError("section 14 requires exactly 51 held windows")
    snapshot = resolve_snapshot(spec, revision=revision)
    config = json.loads((Path(snapshot["snapshot_path"]) / "config.json").read_bytes())
    config = config.get("text_config", config)
    dimensions = {k: config[k] for k in ("num_hidden_layers", "hidden_size", "vocab_size")}
    if any(type(n) is not int or n <= 0 for n in dimensions.values()):
        raise ValueError("invalid snapshot dimensions")
    lens = LensMaps.load(
        lens_path,
        expected_sha256=lens_sha256,
        hidden_size=dimensions["hidden_size"],
        num_layers=dimensions["num_hidden_layers"],
        # Issue 99: the snapshot says which model this is; the lens has to agree. A digest
        # proves the file is the file the caller named, not that the caller named the right one.
        identity=LensIdentity(
            base=spec.base,
            num_layers=dimensions["num_hidden_layers"],
            training=spec.training,
        ),
    )
    if set(lens.maps) != set(range(1, dimensions["num_hidden_layers"])):
        raise ValueError("hosted lens must cover every intermediate layer")
    windows = []
    for row in held:
        ids = row["ids"][:PREFIX_TOKENS]
        prompt = tokenizer.decode(ids)
        if encode_prompt(tokenizer, prompt) != ids:
            raise ValueError(f"exact prefix roundtrip drift at window {row['index']}")
        if any(token >= dimensions["vocab_size"] for token in row["ids"]):
            raise ValueError("window token outside snapshot vocabulary")
        windows.append(
            {
                "label": f"prose-window-{row['index']:04d}",
                "kind": "prose",
                "window_index": row["index"],
                "window_sha256": _sha(_bytes(row["ids"])),
                "prompt_ids": ids,
                "prompt": prompt,
                "prompt_sha256": _sha(prompt.encode()),
                "prompt_tokens": PREFIX_TOKENS,
                "max_tokens": MAX_TOKENS,
            }
        )
    plan = {
        "schema_version": 1,
        "requirements": "14",
        "spec": asdict(spec),
        "model": spec.hf_id,
        "snapshot": snapshot,
        "dimensions": dimensions,
        "corpus": {"path": str(corpus_path), "sha256": before},
        "sequences": manifest["sequences"],
        "sources": manifest["sources"],
        "tokenizer": descriptor,
        "lens_path": str(lens_path),
        "lens_sha256": lens_sha256,
        "layers": list(range(1, dimensions["num_hidden_layers"] + 1)),
        "top_k": 10,
        "cache_strategy": "none",
        "sampler": {"temperature": 0.0, "kind": "greedy"},
        "seed": SEED,
        "min_tokens": MIN_TOKENS,
        "max_tokens": MAX_TOKENS,
        "emitted_count_includes_eos": True,
        "stop_policy": "EOS or cap only",
        "overlap": OVERLAP,
        "windows": windows,
    }
    plan["plan_sha256"] = _sha(_bytes(plan))
    return plan


def register_plan(output, plan):
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / "README.md").write_text(README)
    _write(output / "plan.json", plan)
    return output


def validate_plan(plan, spec, tokenizer):
    """Rebuild every invariant at consumption, including all upstream bound hashes."""
    actual = make_plan(
        plan["corpus"]["path"],
        spec,
        plan["lens_path"],
        lens_sha256=plan["lens_sha256"],
        tokenizer=tokenizer,
        revision=plan["snapshot"]["resolved_revision"] or "main",
    )
    if _bytes(actual) != _bytes(plan):
        raise ValueError("frozen plan/input drift")


def generate_prose(session, model, tokenizer, window, *, stream_generate, sampler):
    """Public session seam: native cache, every yielded token, no text-based stopping."""
    prompt = window["prompt"]
    if encode_prompt(tokenizer, prompt) != window["prompt_ids"]:
        raise ValueError("runtime tokenizer exact prefix drift")
    session.set_context(
        kind="prose",
        label=window["label"],
        window_index=window["window_index"],
        turn=0,
        messages=[{"role": "user", "content": prompt}],
    )
    count, finish_reason = 0, None
    with session.generation(model, tokenizer, prompt, turn_cache=None) as captured:
        stream = stream_generate(
            captured, tokenizer, prompt, sampler=sampler, max_tokens=MAX_TOKENS
        )
        try:
            for response in stream:
                if finish_reason is not None or count >= MAX_TOKENS:
                    raise ValueError("native stream exceeded EOS/cap boundary")
                session.emitted(response.token)
                count += 1
                finish_reason = response.finish_reason
            if finish_reason not in ("stop", "length"):
                raise ValueError("native stream ended without EOS/cap finish reason")
            if finish_reason == "length" and count != MAX_TOKENS:
                raise ValueError("native cap count mismatch")
            if finish_reason == "stop" and int(response.token) not in tokenizer.eos_token_ids:
                raise ValueError("native stop was not EOS")
        finally:
            stream.close()
    return {"generated_tokens": count, "finish_reason": finish_reason, "turns": 1}


def capture_windows(
    output,
    plan,
    loaded,
    lens,
    *,
    stream_generate,
    sampler,
    session_factory=CaptureSession,
    readout_factory=LensReadout,
    resources=resource_snapshot,
    progress=print,
):
    """Write pilot-compatible records and a non-complete manifest until full coverage."""
    from local_llm_lab.pipeline.lens_fitting.replay import read_source

    output = Path(output)
    (output / "dropped").mkdir(exist_ok=False)
    manifest = {
        k: plan[k]
        for k in (
            "model",
            "lens_sha256",
            "layers",
            "top_k",
            "seed",
            "max_tokens",
            "min_tokens",
            "cache_strategy",
            "sampler",
            "emitted_count_includes_eos",
            "overlap",
            "snapshot",
        )
    }
    manifest.update(
        plan_sha256=plan["plan_sha256"],
        lock_path=str(loaded.lock_path),
        episodes=[],
        dropped=[],
        status="partial",
    )
    _write(output / "manifest.json", manifest)
    reader = readout_factory(loaded.view, lens)
    for window in plan["windows"]:
        start = time.monotonic()
        path = output / (window["label"] + ".jsonl")
        provenance = {k: manifest[k] for k in ("model", "lens_sha256", "layers", "snapshot")}
        provenance.update(episode=window, plan_sha256=plan["plan_sha256"])
        with RecordWriter(path, provenance) as write:
            session = session_factory(
                loaded.view,
                reader,
                write,
                layers=plan["layers"],
                attention_blocks=(),
                top_k=plan["top_k"],
            )
            result = generate_prose(
                session,
                loaded.model,
                loaded.tokenizer,
                window,
                stream_generate=stream_generate,
                sampler=sampler,
            )
        events, identity = read_source(path)
        ends = [e for e in events if e["kind"] == "end_turn"]
        if len(ends) != 1 or ends[0]["emitted_count"] != result["generated_tokens"]:
            raise ValueError("capture emission accounting differs from generator")
        entry = {
            **window,
            **result,
            "seconds": time.monotonic() - start,
            "record": path.name,
            "record_sha256": identity["sha256"],
            **resources(),
        }
        # Prefix data remains in frozen plan and record; manifest retains its hashes/length.
        entry.pop("prompt")
        entry.pop("prompt_ids")
        if result["generated_tokens"] < MIN_TOKENS:
            dropped_path = output / "dropped" / path.name
            path.rename(dropped_path)
            entry.update(record="dropped/" + path.name, reason="continuation_under_32")
            manifest["dropped"].append(entry)
        else:
            manifest["episodes"].append(entry)
        (output / "manifest.json").write_bytes(_bytes(manifest) + b"\n")
        progress(json.dumps({"event": "episode", **entry}))
    observed = [e["window_index"] for e in manifest["episodes"] + manifest["dropped"]]
    if sorted(observed) != sorted(w["window_index"] for w in plan["windows"]):
        raise ValueError("planned held-window coverage mismatch")
    manifest.update(
        status="complete",
        accepted_count=len(manifest["episodes"]),
        dropped_count=len(manifest["dropped"]),
    )
    (output / "manifest.json").write_bytes(_bytes(manifest) + b"\n")
    return manifest


def execute_plan(output, spec):
    """The only checkpoint entry: registration + revalidation precede locked loading."""
    output = Path(output).absolute()
    plan = json.loads((output / "plan.json").read_bytes())
    if (output / "README.md").read_text() != README:
        raise ValueError("required prewritten registration changed")
    tokenizer = corpus_tokenizer(
        plan["corpus"]["path"], spec, revision=plan["snapshot"]["resolved_revision"] or "main"
    )
    validate_plan(plan, spec, tokenizer)
    if set(p.name for p in output.iterdir()) != {"README.md", "plan.json"}:
        raise FileExistsError("execution requires an unused registered plan directory")
    _write(output / "execution.json", {"plan_sha256": plan["plan_sha256"], "status": "started"})
    loaded = load_runtime(SimpleNamespace(spec=spec, snapshot=plan["snapshot"]), capture=True)
    if loaded.resolved.cache_strategy != "none":
        raise ValueError("prose capture requires resolved none")
    dimensions = {
        "num_hidden_layers": loaded.view.num_layers,
        "hidden_size": loaded.view.hidden_size,
        "vocab_size": loaded.view.vocab_size,
    }
    if dimensions != plan["dimensions"]:
        raise ValueError("runtime ArchitectureView differs from snapshot dimensions")
    if (
        snapshot_identity(Path(plan["snapshot"]["snapshot_path"]), hf_id=spec.hf_id)
        != plan["snapshot"]
    ):
        raise ValueError("snapshot drift after load")
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    allocator = configure_allocator_cache()
    mx.random.seed(plan["seed"])
    loaded.model.eval()
    lens = LensMaps.load(
        Path(plan["lens_path"]),
        expected_sha256=plan["lens_sha256"],
        hidden_size=loaded.view.hidden_size,
        num_layers=loaded.view.num_layers,
        identity=LensIdentity(
            base=loaded.spec.base,
            num_layers=loaded.view.num_layers,
            training=loaded.spec.training,
        ),
    )
    _write(
        output / "runtime.json", {"lock_path": str(loaded.lock_path), "allocator_cache": allocator}
    )
    return capture_windows(
        output, plan, loaded, lens, stream_generate=stream_generate, sampler=make_sampler(temp=0.0)
    )
