"""Requirements §3.1 and §10: immutable, leakage-free fitting corpora."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from local_llm_lab.agent_protocol import Action
from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.protocol import (
    SYSTEM_PROMPT,
    assistant_message,
    build_prompt,
    generation_suffix,
    tool_message,
)


class OffsetTokenizer:
    bos_token = None
    chat_template = "synthetic canonical chat v1"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        body = "".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>\n" for m in messages)
        return body + (
            generation_suffix(load_model_spec("qwen35-4b")) if add_generation_prompt else ""
        )

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        self.calls.append((text, add_special_tokens))
        prefix = [1] if add_special_tokens else []
        return {
            "input_ids": prefix + [ord(c) + 2 for c in text],
            "offset_mapping": ([(0, 0)] if prefix else []) + [(i, i + 1) for i in range(len(text))],
        }


@pytest.fixture
def setup(tmp_path):
    spec = load_model_spec("qwen35-4b")
    tokenizer = OffsetTokenizer()
    asset = tmp_path / "tokenizer.json"
    asset.write_text('{"synthetic": true}')
    return spec, tokenizer, [asset]


def api():
    return importlib.import_module("local_llm_lab.pipeline.lens_fitting.corpus")


def evaluation(path, spec, *, count=5, task_id="train-read-0000-clean"):
    steps = []
    for i in range(count):
        raw = f'Note {i}.\n```json\n{{"name":"read_file","arguments":{{"path":"/a"}}}}\n```'
        steps.append(
            {
                "index": i,
                "raw": raw,
                "thought": f"Note {i}.",
                "action": {"name": "read_file", "arguments": {"path": "/a"}},
                "observation": f"unique observation {i}",
            }
        )
    data = {
        "summary": {
            "model": {"spec": {"hf_id": spec.hf_id}},
            "split": task_id.split("-")[0],
            "keep_last": 2,
        },
        "trajectories": [{"task_id": task_id, "prompt": "Inspect the workspace.", "steps": steps}],
    }
    path.write_text(json.dumps(data))
    return data


def build(tmp_path, setup, data=None, **kwargs):
    spec, tokenizer, assets = setup
    source = tmp_path / "eval.json"
    if data is None:
        evaluation(source, spec)
    else:
        source.write_text(json.dumps(data))
    manifest = tmp_path / "corpus.json"
    api().build_agentic_corpus(
        [source],
        tokenizer,
        spec,
        manifest,
        max_tokens=kwargs.pop("max_tokens", 20000),
        tokenizer_files=assets,
        **kwargs,
    )
    return manifest, api().read_corpus(manifest)


def test_runner_prompt_spans_and_raw_completion(tmp_path, setup):
    """§3.1: canonical windowing, raw output and BOS must share one aligned tokenization."""
    manifest, rows = build(tmp_path, setup)
    spec, tokenizer, _ = setup
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Inspect the workspace."},
    ]
    for i in range(4):
        messages += [
            assistant_message(f"Note {i}.", Action("read_file", {"path": "/a"})),
            tool_message("read_file", f"unique observation {i}"),
        ]
    assert rows[4]["prompt"] == build_prompt(tokenizer, messages, spec=spec, keep_last=2)
    assert "unique observation 0" not in rows[4]["prompt"]
    row = rows[4]
    hidden = row["text"].index("[earlier read_file result hidden:")
    assert row["spans"][hidden + 1] == "observation"
    assert row["spans"][0] == "template"  # BOS has no character extent.
    assert row["ids"][0] == 1
    assert row["n_prompt"] == len(row["prompt"]) + 1
    assert row["spans"][row["n_prompt"]] == "note"
    assert row["spans"][-1] == "call"
    assert len(tokenizer.calls) == 5
    assert all(len(r["ids"]) == len(r["spans"]) for r in rows)
    assert [r["split"] for r in rows] == ["fit", "fit", "fit", "fit", "held"]
    assert json.loads(manifest.read_text())["counts"]["held"]["sequences"] == 1


def test_parse_failure_is_included_and_finish_stops(tmp_path, setup):
    """§3.1: final malformed raw output survives; replay stops at the runner's terminal turn."""
    spec, _, _ = setup
    data = evaluation(tmp_path / "original.json", spec, count=3)
    data["trajectories"][0]["steps"][1] = {
        "index": 1,
        "raw": "unparsed final output",
        "parse_error": "no call",
    }
    _, rows = build(tmp_path, setup, data)
    assert len(rows) == 2
    assert rows[-1]["text"].endswith("unparsed final output")


def test_dropped_source_step_does_not_move_held_membership(tmp_path, setup):
    """§3.1: source-step indexing precedes length filtering, including dropped rows."""
    spec, _, _ = setup
    data = evaluation(tmp_path / "original.json", spec)
    data["trajectories"][0]["steps"][0]["raw"] = (
        "x" * 30000 + data["trajectories"][0]["steps"][0]["raw"]
    )
    # Keep canonical history consistent with the raw completion.
    data["trajectories"][0]["steps"][0]["thought"] = "x" * 30000 + "Note 0."
    # An independent next trajectory avoids carrying the intentionally enormous note.
    steps = data["trajectories"][0]["steps"]
    data["trajectories"] = [dict(data["trajectories"][0], steps=steps[:1])]
    for i, step in enumerate(steps[1:], 1):
        data["trajectories"].append(
            {
                "task_id": f"train-read-{i:04d}-clean",
                "prompt": "Task",
                "steps": [dict(step, index=0)],
            }
        )
    manifest, rows = build(tmp_path, setup, data)
    assert [r["index"] for r in rows] == [1, 2, 3, 4]
    assert rows[-1]["split"] == "held"
    dropped = json.loads(manifest.read_text())["dropped_rows"]
    assert dropped[0]["index"] == 0
    assert dropped[0]["tokens"] > 20000
    assert dropped[0]["source"] == str((tmp_path / "eval.json").resolve())


@pytest.mark.parametrize(
    "task_id",
    [
        "test-read-0000-clean",
        "valid-read-0000-clean",
        "validation-read-0000-clean",
        "pilot-read-0000-clean",
        "train-pilot-0000-clean",
        "trainish-read-0000-clean",
    ],
)
def test_reject_nontraining_and_pilot_ids(tmp_path, setup, task_id):
    """§3.1/§10: neither held evaluation IDs nor malformed training lookalikes may leak."""
    data = evaluation(tmp_path / "original.json", setup[0], task_id=task_id)
    with pytest.raises(ValueError, match="training|pilot|split"):
        build(tmp_path, setup, data)


@pytest.mark.parametrize("mutation", ["mixed", "missing", "window", "split"])
def test_fail_closed_on_source_provenance(tmp_path, setup, mutation):
    """§3.1/§10: source identity and prompt-window policy cannot be guessed or mixed."""
    data = evaluation(tmp_path / "original.json", setup[0])
    if mutation == "mixed":
        data["trajectories"][0]["model"] = {"spec": {"hf_id": "other/model"}}
    elif mutation == "missing":
        data["summary"].pop("model")
    elif mutation == "window":
        data["summary"]["keep_last"] = 9
    else:
        data["summary"]["split"] = "train2"
    with pytest.raises(ValueError):
        build(tmp_path, setup, data)


@pytest.mark.parametrize("target", ["source", "sequences", "manifest", "tokenizer"])
def test_hash_bound_artifacts_reject_corruption(tmp_path, setup, target):
    """§10: modified input, token stream, manifest or tokenizer assets are detected on read."""
    manifest, _ = build(tmp_path, setup)
    content = json.loads(manifest.read_text())
    path = {
        "source": tmp_path / "eval.json",
        "sequences": Path(content["sequences"]["path"]),
        "manifest": manifest,
        "tokenizer": setup[2][0],
    }[target]
    path.write_text(path.read_text() + " ")
    if target == "manifest":
        content["domain"] = "prose"
        path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="hash|SHA|digest"):
        api().read_corpus(manifest)


def test_duplicate_sources_and_trajectories_are_rejected(tmp_path, setup):
    """§10: aliases or overlapping reports must not double-count trajectories."""
    spec, tok, assets = setup
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    evaluation(a, spec)
    b.write_bytes(a.read_bytes())
    for paths in ([a, a], [a, b]):
        with pytest.raises(ValueError, match="duplicate"):
            api().build_agentic_corpus(
                paths, tok, spec, tmp_path / "out.json", max_tokens=20000, tokenizer_files=assets
            )


def test_prose_nonoverlap_tail_and_immutable_outputs(tmp_path, setup):
    """§3.1/§10: fixed prose windows count the tail and forbid overwriting existing artifacts."""
    spec, tok, assets = setup
    path = tmp_path / "prose.txt"
    path.write_text("abcd" * 1300)
    out = tmp_path / "prose.json"
    api().build_prose_corpus([path], tok, spec, out, tokenizer_files=assets)
    rows = api().read_corpus(out)
    assert len(rows) == 5
    assert all(len(row["ids"]) == 1024 and row["n_prompt"] == 0 for row in rows)
    assert all(set(row["spans"]) <= {"chat", "template"} for row in rows)
    assert rows[-1]["split"] == "held"
    assert json.loads(out.read_text())["discarded_trailing_tokens"] == 81
    flattened = [token for row in rows for token in row["ids"]]
    assert flattened == [1] + [ord(c) + 2 for c in path.read_text()[:5119]]
    with pytest.raises(FileExistsError):
        api().build_prose_corpus([path], tok, spec, out, tokenizer_files=assets)


def test_finish_does_not_append_or_render_subsequent_steps(tmp_path, setup):
    """§3.1: finish terminates rendering even if an input artifact contains trailing steps."""
    data = evaluation(tmp_path / "original.json", setup[0], count=3)
    data["trajectories"][0]["steps"][0].update(
        raw='Done.\n```json\n{"name":"finish","arguments":{"answer":"ok"}}\n```',
        thought="Done.",
        action={"name": "finish", "arguments": {"answer": "ok"}},
    )
    _, rows = build(tmp_path, setup, data)
    assert len(rows) == 1
    assert rows[0]["text"].endswith('"ok"}}\n```')


def test_boundary_crossing_token_is_template_without_second_encoding(tmp_path, setup):
    """§3.1: a merged prompt/raw token is template and excluded from the prompt count."""
    spec, _, assets = setup

    class MergingTokenizer(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            boundary = text.index("Note 0.")
            result["input_ids"][boundary : boundary + 2] = [999]
            result["offset_mapping"][boundary : boundary + 2] = [(boundary - 1, boundary + 1)]
            return result

    tok = MergingTokenizer()
    data = evaluation(tmp_path / "original.json", spec, count=1)
    _, rows = build(tmp_path, (spec, tok, assets), data)
    row = rows[0]
    assert row["spans"][row["n_prompt"]] == "template"
    assert row["ids"][row["n_prompt"]] == 999
    assert len(tok.calls) == 1


def test_existing_bos_suppresses_added_bos(tmp_path, setup):
    """§3.1: BOS already in rendered text must follow the generator's no-double-BOS policy."""
    spec, _, assets = setup

    class BosTokenizer(OffsetTokenizer):
        bos_token = "<BOS>"

        def apply_chat_template(self, *args, **kwargs):
            return self.bos_token + super().apply_chat_template(*args, **kwargs)

    tok = BosTokenizer()
    _, rows = build(tmp_path, (spec, tok, assets))
    assert all(not add_special for _, add_special in tok.calls)
    assert rows[0]["ids"][0] == ord("<") + 2


@pytest.mark.parametrize(
    "broken", ["missing_content", "missing_offsets", "reversed_offsets", "zero_offsets"]
)
def test_alignment_failure_is_not_silently_template(tmp_path, setup, broken):
    """§3.1/§10: missing semantic alignment or invalid token offsets fail closed."""
    spec, _, assets = setup

    class BrokenTokenizer(OffsetTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            prompt = super().apply_chat_template(*args, **kwargs)
            return (
                prompt.replace("Inspect the workspace.", "rewritten")
                if broken == "missing_content"
                else prompt
            )

        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            if broken == "missing_offsets":
                result.pop("offset_mapping")
            elif broken == "zero_offsets":
                result["offset_mapping"] = [(0, 0)] * len(result["input_ids"])
            elif broken == "reversed_offsets":
                result["offset_mapping"][3] = (10, 1)
            return result

    with pytest.raises(ValueError, match="alignment"):
        build(tmp_path, (spec, BrokenTokenizer(), assets))
    assert not (tmp_path / "corpus.json").exists()


def test_raw_history_contradiction_rejected(tmp_path, setup):
    """§3.1: canonical past actions must be the actions parsed from the model's own raw turn."""
    data = evaluation(tmp_path / "original.json", setup[0])
    data["trajectories"][0]["steps"][0]["thought"] = "invented note"
    with pytest.raises(ValueError, match="contradict"):
        build(tmp_path, setup, data)


def _validation_remote(tmp_path, monkeypatch, texts):
    from types import SimpleNamespace

    import datasets
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard = tmp_path / "fake-remote.parquet"
    pq.write_table(pa.table({"text": texts}), shard)
    calls = []

    class Hub:
        def dataset_info(self, repo_id, *, revision):
            assert repo_id == "Salesforce/wikitext"
            assert revision == "main"
            return SimpleNamespace(sha="a" * 40)

        def list_repo_files(self, repo_id, *, repo_type, revision):
            assert (repo_id, repo_type, revision) == ("Salesforce/wikitext", "dataset", "a" * 40)
            return [
                "README.md",
                "wikitext-103-raw-v1/train-00000-of-00001.parquet",
                "wikitext-103-raw-v1/test-00000-of-00001.parquet",
                "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
                "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
            ]

    def download(repo_id, filename, *, repo_type, revision):
        calls.append((repo_id, filename, repo_type, revision))
        assert filename == "wikitext-103-raw-v1/validation-00000-of-00001.parquet"
        return str(shard)

    def reject_builder(*args, **kwargs):
        raise AssertionError(
            "Dataset builder may download excluded splits before selecting validation"
        )

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(datasets, "load_dataset", reject_builder)
    return calls


def test_download_is_validation_only_pinned_and_hash_verified(tmp_path, monkeypatch):
    """§3.1/§10: only authorised validation shard bytes may cross the download boundary."""
    calls = _validation_remote(
        tmp_path, monkeypatch, ["first validation passage", "second validation passage"]
    )
    text, descriptor = api().download_prose(tmp_path, revision="main")
    assert calls == [
        (
            "Salesforce/wikitext",
            "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
            "dataset",
            "a" * 40,
        )
    ]
    assert text.read_text() == "first validation passage\nsecond validation passage\n"
    record = json.loads(descriptor.read_text())
    assert record["revision"] == "a" * 40
    assert record["text"]["sha256"] == hashlib.sha256(text.read_bytes()).hexdigest()
    assert (
        record["files"][0]["sha256"]
        == hashlib.sha256((tmp_path / "fake-remote.parquet").read_bytes()).hexdigest()
    )
    assert api().download_prose(tmp_path, revision="main") == (text, descriptor)
    assert len(calls) == 1
    text.write_text("corruption")
    with pytest.raises(ValueError, match="hash|SHA"):
        api().download_prose(tmp_path, revision="main")
    assert len(calls) == 1


def test_download_rejects_unresolved_revision_and_wrong_reuse(tmp_path, monkeypatch):
    """§10: a moving revision cannot enter provenance; a changed download descriptor is rejected."""
    import huggingface_hub

    class Hub:
        def dataset_info(self, *args, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(sha="main")

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    with pytest.raises(ValueError, match="revision"):
        api().download_prose(tmp_path, revision="main")
    assert not list(tmp_path.iterdir())


def test_tokenizer_loading_never_requests_weights(tmp_path, monkeypatch, setup):
    """§3.1/§10: tokenizer loading restricts the snapshot to tokenizer/config assets."""
    import fnmatch

    import huggingface_hub
    import transformers

    calls = []

    def snapshot_download(repo_id, *, allow_patterns, local_files_only):
        assert repo_id == setup[0].hf_id
        assert not any(
            fnmatch.fnmatch(name, pattern)
            for name in (
                "model.safetensors",
                "model-00001-of-00002.safetensors",
                "pytorch_model.bin",
                "model.safetensors.index.json",
            )
            for pattern in allow_patterns
        )
        calls.append(allow_patterns)
        (tmp_path / "tokenizer.json").write_text("{}")
        (tmp_path / "tokenizer_config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("must not be opened or hashed")
        return str(tmp_path)

    def from_pretrained(path, **kwargs):
        assert kwargs == {"local_files_only": True, "trust_remote_code": False, "use_fast": True}
        return setup[1]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", from_pretrained)
    _, files = api().load_corpus_tokenizer(setup[0])
    assert {p.name for p in files} == {"tokenizer.json", "tokenizer_config.json"}
    assert len(calls) == 1


def test_cli_builds_manifest_with_fixed_corpus_budget(tmp_path, setup, monkeypatch):
    """§3.1/§10: CLI defaults apply the corpus data cap and use the tokenizer-only loader."""
    import runpy

    spec, tok, assets = setup
    source = tmp_path / "eval.json"
    data = evaluation(source, spec, count=1)
    data["trajectories"][0]["prompt"] += "x" * 1000
    source.write_text(json.dumps(data))
    monkeypatch.setattr(api(), "load_corpus_tokenizer", lambda *a, **kw: (tok, assets))
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/lens_corpus.py"))[
        "main"
    ]
    output = tmp_path / "cli.json"
    main(["--corpus", "agentic", "--evals", str(source), "--out", str(output)])
    manifest = json.loads(output.read_text())
    assert manifest["max_tokens"] == 2048
    assert len(manifest["dropped_rows"]) == 1  # Character tokenizer exceeds the actual token cap.
    assert api().read_corpus(output) == []


def _rewrite_manifest(path, mutate):
    manifest = json.loads(path.read_text())
    mutate(manifest)
    manifest.pop("manifest_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("mutation", ["spans", "n_prompt", "ids", "order", "split", "source_step"])
def test_reader_checks_rows_beyond_file_hash(tmp_path, setup, mutation):
    """§10: invalid typed rows and split/order invariants fail even with matching file hashes."""
    manifest, _ = build(tmp_path, setup)
    content = json.loads(manifest.read_text())
    path = Path(content["sequences"]["path"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "spans":
        rows[0]["spans"].pop()
    elif mutation == "n_prompt":
        rows[0]["n_prompt"] = len(rows[0]["ids"]) + 1
    elif mutation == "ids":
        rows[0]["ids"][0] = True
    elif mutation == "order":
        rows.reverse()
    elif mutation == "source_step":
        rows[0]["step_index"] = 999
    else:
        rows[0]["split"] = "held"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _rewrite_manifest(
        manifest,
        lambda m: m["sequences"].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    with pytest.raises(ValueError):
        api().read_corpus(manifest)


def test_download_descriptor_cannot_be_reused_for_other_split(tmp_path, monkeypatch):
    """§10: even a rehashed download descriptor cannot authorize a train/test split."""
    _validation_remote(tmp_path, monkeypatch, ["validation"])
    _, descriptor = api().download_prose(tmp_path)
    data = json.loads(descriptor.read_text())
    data["split"] = "test"
    data.pop("descriptor_sha256")
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    data["descriptor_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    descriptor.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="split"):
        api().download_prose(tmp_path)


def test_prose_manifest_binds_download_descriptor(tmp_path, setup, monkeypatch):
    """§3.1/§10: a downloaded corpus binds both downloaded text and the pinned descriptor."""
    _validation_remote(tmp_path, monkeypatch, ["prose " * 200])
    source, descriptor = api().download_prose(tmp_path / "download")
    spec, tok, assets = setup
    manifest = tmp_path / "downloaded.json"
    api().build_prose_corpus(
        [source], tok, spec, manifest, tokenizer_files=assets, download_descriptor=descriptor
    )
    assert len(api().read_corpus(manifest)) == 1
    descriptor.write_text(descriptor.read_text() + " ")
    with pytest.raises(ValueError, match="hash|SHA"):
        api().read_corpus(manifest)


def test_source_order_is_explicit_and_global(tmp_path, setup):
    """§3.1/§10: source ordering controls stable indices across training splits."""
    spec, tok, assets = setup
    first, second = tmp_path / "z-train.json", tmp_path / "a-train1.json"
    evaluation(first, spec, count=3)
    evaluation(second, spec, count=3, task_id="train1-read-0000-clean")
    out = tmp_path / "ordered.json"
    metadata = api().build_agentic_corpus(
        [first, second], tok, spec, out, max_tokens=20000, tokenizer_files=assets
    )
    rows = api().read_corpus(out)
    assert [source["path"] for source in metadata["sources"]] == [str(first), str(second)]
    assert rows[3]["task_id"] == "train1-read-0000-clean"
    assert rows[4]["step_index"] == 1
    assert rows[4]["split"] == "held"


def _rewrite_rows(manifest, mutate):
    sequence_path = Path(json.loads(manifest.read_text())["sequences"]["path"])
    rows = [json.loads(line) for line in sequence_path.read_text().splitlines()]
    mutate(rows)
    sequence_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _rewrite_manifest(
        manifest,
        lambda m: m["sequences"].update(
            sha256=hashlib.sha256(sequence_path.read_bytes()).hexdigest()
        ),
    )


def _prose_manifest(tmp_path, setup):
    spec, tok, assets = setup
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("a" * 2050)  # BOS + text -> two chunks and three trailing tokens.
    second.write_text("b" * 1024)  # BOS + text -> one chunk and one trailing token.
    manifest = tmp_path / "prose.json"
    api().build_prose_corpus([first, second], tok, spec, manifest, tokenizer_files=assets)
    return manifest


def test_reader_rejects_rehashed_wrong_prose_step(tmp_path, setup):
    """§3.1/§10: internally consistent token_start cannot legitimize an absent source chunk."""
    manifest = _prose_manifest(tmp_path, setup)
    _rewrite_rows(manifest, lambda rows: rows[1].update(step_index=999, token_start=1022976))
    with pytest.raises(ValueError, match="prose|source"):
        api().read_corpus(manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        "source_order",
        "tokens",
        "source_tail",
        "total_tail",
        "total_count",
        "chunk_size",
        "typed_tail",
    ],
)
def test_reader_reconciles_prose_sources_chunks_and_tails(tmp_path, setup, mutation):
    """§3.1/§10: ordered source chunks and declared token/tail budgets must reconcile."""
    manifest = _prose_manifest(tmp_path, setup)
    if mutation == "source_order":
        _rewrite_rows(manifest, lambda rows: rows[0].update(source=rows[2]["source"]))
    else:

        def change(metadata):
            if mutation == "tokens":
                metadata["sources"][0]["tokens"] += 1024
            elif mutation == "source_tail":
                metadata["sources"][0]["discarded_trailing_tokens"] = 1024
            elif mutation == "total_tail":
                metadata["discarded_trailing_tokens"] += 1
            elif mutation == "total_count":
                metadata["source_sequence_count"] += 1
            elif mutation == "chunk_size":
                metadata["chunk_tokens"] = 512
            else:
                metadata["sources"][1]["discarded_trailing_tokens"] = True

        _rewrite_manifest(manifest, change)
    with pytest.raises(ValueError, match="prose|source|chunk|tail"):
        api().read_corpus(manifest)


@pytest.mark.parametrize(
    "mutation", ["negative", "reversed", "out_of_text", "nonmonotonic", "boolean", "wrong_shape"]
)
def test_reader_rejects_rehashed_invalid_offsets(tmp_path, setup, mutation):
    """§3.1/§10: reader enforces typed, bounded, monotonic offsets independently of hashes."""
    manifest, _ = build(tmp_path, setup)

    def change(rows):
        row = rows[0]
        if mutation == "negative":
            row["offsets"][0] = [-100, 0]
        elif mutation == "reversed":
            row["offsets"][0] = [1, 0]
        elif mutation == "out_of_text":
            row["offsets"][-1][1] = len(row["text"]) + 1
        elif mutation == "nonmonotonic":
            row["offsets"][3] = [0, 1]
        elif mutation == "boolean":
            row["offsets"][0] = [False, 0]
        else:
            row["offsets"][0] = []

    _rewrite_rows(manifest, change)
    with pytest.raises(ValueError, match="alignment"):
        api().read_corpus(manifest)


@pytest.mark.parametrize("domain", ["agentic", "prose"])
def test_pilot_rule_scope_and_legacy_manifest_compatibility(tmp_path, setup, domain):
    """§3.1/§10: task-ID exclusions must not describe authorised prose validation as excluded."""
    manifest = (
        build(tmp_path, setup)[0] if domain == "agentic" else _prose_manifest(tmp_path, setup)
    )
    metadata = json.loads(manifest.read_text())
    assert metadata["pilot_exclusion"]["applies_to"] == "agentic_task_ids"
    expected = api().read_corpus(manifest)
    _rewrite_manifest(manifest, lambda m: m["pilot_exclusion"].pop("applies_to"))
    assert api().read_corpus(manifest) == expected


@pytest.mark.parametrize("chunk_tokens", [128, 2048])
def test_prose_explicit_window_length_roundtrips(tmp_path, setup, chunk_tokens):
    """Gemma pivot: fit lengths on both sides of the window keep exact partition accounting."""
    spec, tok, assets = setup
    source = tmp_path / "text.txt"
    source.write_text("a" * (5 * chunk_tokens + 6))
    out = tmp_path / "manifest.json"
    metadata = api().build_prose_corpus(
        [source], tok, spec, out, tokenizer_files=assets, chunk_tokens=chunk_tokens
    )
    rows = api().read_corpus(out)
    assert metadata["chunk_tokens"] == chunk_tokens
    assert len(rows) == 5
    assert [r["split"] for r in rows] == ["fit"] * 4 + ["held"]
    assert [r["token_start"] for r in rows] == [i * chunk_tokens for i in range(5)]
    assert all(len(r["ids"]) == chunk_tokens for r in rows)
    assert [token for row in rows for token in row["ids"]] == [1] + [99] * (5 * chunk_tokens - 1)
    assert metadata["discarded_trailing_tokens"] == 7
    assert metadata["counts"]["held"]["tokens"] == chunk_tokens


@pytest.mark.parametrize("chunk_tokens", [0, -1, True, 128.0, "128"])
def test_prose_refuses_invalid_window_before_tokenizing(tmp_path, setup, chunk_tokens):
    spec, tok, assets = setup
    source = tmp_path / "text.txt"
    source.write_text("sample")
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="chunk"):
        api().build_prose_corpus(
            [source], tok, spec, out, tokenizer_files=assets, chunk_tokens=chunk_tokens
        )
    assert not tok.calls
    assert not out.exists() and not out.with_suffix(".jsonl").exists()


def test_prose_cli_forwards_explicit_window(tmp_path, setup, monkeypatch):
    import importlib.util

    spec, tok, assets = setup
    module_spec = importlib.util.spec_from_file_location(
        "corpus_cli", Path(__file__).resolve().parents[1] / "scripts/lens_corpus.py"
    )
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_corpus_tokenizer", lambda *a, **kw: (tok, assets))
    source = tmp_path / "text.txt"
    source.write_text("a" * 650)
    out = tmp_path / "manifest.json"
    cli.main(
        [
            "--corpus",
            "prose",
            "--files",
            str(source),
            "--out",
            str(out),
            "--model",
            spec.name,
            "--prose-chunk-tokens",
            "128",
            "--local-files-only",
        ]
    )
    assert json.loads(out.read_text())["chunk_tokens"] == 128
    assert len(api().read_corpus(out)) == 5
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--corpus",
                "agentic",
                "--evals",
                str(source),
                "--out",
                str(tmp_path / "x.json"),
                "--prose-chunk-tokens",
                "128",
            ]
        )


@pytest.mark.parametrize("chunk_tokens", [None, 0, -1, True, 1024.0, "1024"])
def test_reader_rejects_rehashed_invalid_window_type(tmp_path, setup, chunk_tokens):
    """A valid self-digest cannot make an invalid partition definition readable."""
    manifest = _prose_manifest(tmp_path, setup)
    metadata = json.loads(manifest.read_bytes())
    metadata.pop("manifest_sha256")
    metadata["chunk_tokens"] = chunk_tokens
    metadata["manifest_sha256"] = api()._sha(api()._json_bytes(metadata))
    manifest.write_bytes(api()._json_bytes(metadata))
    with pytest.raises(ValueError, match="chunk"):
        api().read_corpus(manifest)
