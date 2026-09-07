"""Pure replay integrity and refusal contracts; no native model import."""

import copy
import gzip
import importlib
import json
from pathlib import Path

import pytest

from local_llm_lab.pipeline.live_lens.session import RecordWriter


def api():
    return importlib.import_module("local_llm_lab.pipeline.lens_fitting.replay")


def rows():
    return [
        dict(
            kind="begin_turn",
            turn=0,
            prompt="p",
            prompt_ids=[1, 2],
            context={},
            layers=[1, 2],
            attention_blocks=[],
            top_k=1,
            audit_modulus=23,
            audit_seed=0,
            rank_horizons=[1, 4, 8],
            distribution_capacity=16,
        ),
        dict(kind="reading", turn=0, position=0, top={"1": [1], "2": [1]}),
        dict(kind="reading", turn=0, position=1, top={"1": [1], "2": [1]}),
        dict(
            kind="forward",
            turn=0,
            offset=0,
            input_ids=[1, 2],
            logits_sha256="a" * 64,
            logits_shape=[1, 2, 5],
            argmax=[[1, 1]],
        ),
        # Generator lookahead consumes an as-yet un-emitted token.
        dict(kind="reading", turn=0, position=2, top={"1": [1], "2": [1]}),
        dict(
            kind="forward",
            turn=0,
            offset=2,
            input_ids=[3],
            logits_sha256="b" * 64,
            logits_shape=[1, 1, 5],
            argmax=[[1]],
        ),
        dict(kind="emitted", turn=0, position=2, token_id=3),
        dict(
            kind="rank", turn=0, position=1, layer=1, horizon=1, token_id=3, rank=1, probability=0.5
        ),
        dict(
            kind="rank", turn=0, position=1, layer=2, horizon=1, token_id=3, rank=1, probability=0.5
        ),
        dict(kind="end_turn", turn=0, emitted_count=1, forwarded_count=3, status="complete"),
    ]


def write(path, events=None):
    with RecordWriter(path, {"model": "example/tiny", "layers": [1, 2]}) as out:
        for row in rows() if events is None else events:
            out(row)


def test_gzip_uses_same_chain_validator_and_binds_both_identities(tmp_path):
    path = tmp_path / "ep.jsonl"
    write(path)
    compressed = path.with_suffix(".jsonl.gz")
    compressed.write_bytes(gzip.compress(path.read_bytes()))
    plain, identity = api().read_source(path)
    decoded, compressed_identity = api().read_source(compressed)
    assert plain == decoded
    assert identity["decompressed_sha256"] == compressed_identity["decompressed_sha256"]
    assert identity["sha256"] != compressed_identity["sha256"]
    corrupt = path.read_text().replace("example/tiny", "example/evil")
    compressed.write_bytes(gzip.compress(corrupt.encode()))
    with pytest.raises(ValueError, match="hash chain"):
        api().read_source(compressed)


@pytest.mark.parametrize(
    "mutation", ["offset", "emission", "turn", "aborted", "incomplete", "shape", "reading"]
)
def test_structural_corruption_refused_before_runtime(tmp_path, mutation):
    events = rows()
    if mutation == "offset":
        events[5]["offset"] = 3
    if mutation == "emission":
        events[6]["token_id"] = 4
    if mutation == "turn":
        events[5]["turn"] = 1
    if mutation == "aborted":
        events[-1]["status"] = "aborted"
    if mutation == "incomplete":
        events.pop()
    if mutation == "shape":
        events[5]["logits_shape"] = [1, 2, 5]
    if mutation == "reading":
        events[4]["position"] = 9
    path = tmp_path / "ep.jsonl"
    write(path, events)
    with pytest.raises(ValueError):
        api().read_source(path)


def test_content_revalidated_at_consumption(tmp_path):
    path = tmp_path / "ep.jsonl"
    write(path)
    _, identity = api().read_source(path)
    path.unlink()
    changed = rows()
    changed[0]["context"] = {"changed": True}
    write(path, changed)
    with pytest.raises(ValueError, match="changed"):
        api().read_source(path, expected=identity)


def test_identity_comparison_reports_first_specific_field():
    expected = {"episodes": [{"seconds": 1.5}], "foreknowledge": {"L1": {"n": 2}}}
    assert api().first_difference(expected, copy.deepcopy(expected)) is None
    changed = copy.deepcopy(expected)
    changed["episodes"][0]["seconds"] = 2
    assert api().first_difference(expected, changed) == "$.episodes[0].seconds: 1.5 != 2"


def test_public_capture_driver_preserves_lookahead_and_fresh_turn_cache():
    """A fake public session checks orchestration without substituting native evidence."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    import numpy as np

    module = api()
    events = [dict(kind="manifest"), *rows(), dict(kind="end_record", status="complete")]
    second = copy.deepcopy(rows())
    for row in second:
        row["turn"] = 1
    events[-1:-1] = second
    calls, caches, written = [], [], []
    model = object()

    def make_cache():
        cache = object()
        caches.append(cache)
        return cache

    view = SimpleNamespace(model=model, make_cache=make_cache)
    tok = SimpleNamespace(bos_token=None, encode=lambda *a, **kw: [1, 2])

    class Session:
        def __init__(self, v, readout, emit, **kw):
            self.emit, self.turn = emit, -1
            assert kw["retain_logits"] is False

        def set_context(self, **kwargs):
            pass

        @contextmanager
        def generation(self, m, tokenizer, prompt, *, turn_cache):
            assert m is model and turn_cache is None
            self.turn += 1
            self.emit(rows()[0] | {"turn": self.turn})
            self.offset = 0

            def captured(ids, *, cache):
                assert cache is caches[self.turn]
                calls.append(("forward", ids.tolist()))
                row = rows()[3 if self.offset == 0 else 5] | {"turn": self.turn}
                self.offset += len(row["input_ids"])
                self.emit(row)

            yield captured
            self.emit(rows()[-1] | {"turn": self.turn})

        def emitted(self, token):
            calls.append(("emitted", token))
            self.emit(rows()[6] | {"turn": self.turn})

    result = module.replay_record(
        view,
        tok,
        None,
        events,
        written.append,
        layers=[1, 2],
        array_api=np,
        session_factory=Session,
    )
    assert calls == [("forward", [[1, 2]]), ("forward", [[3]]), ("emitted", 3)] * 2
    assert len(caches) == 2 and caches[0] is not caches[1]
    assert result == {"forwards": 4, "tokens": 6}
    tok.encode = lambda *a, **kw: [2, 1]
    with pytest.raises(ValueError, match="tokenizer prompt IDs"):
        module.replay_record(
            view,
            tok,
            None,
            events,
            written.append,
            layers=[1, 2],
            array_api=np,
            session_factory=Session,
        )
    assert len(caches) == 2


def test_actual_legacy_summarizer_identity_and_statistics_mutation(tmp_path, monkeypatch):
    """Invoke actual atlas main; altering historical seconds must turn identity red."""
    import huggingface_hub
    import transformers

    module = api()

    class Tokenizer:
        def __call__(self, *a, **kw):
            return {"input_ids": [1, 2], "offset_mapping": [(0, 1), (1, 2)]}

        def decode(self, ids):
            return "x"

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    original = huggingface_hub.snapshot_download
    reference = tmp_path / "reference"
    reference.mkdir()
    write(reference / "ep.jsonl")
    manifest = {"model": "test", "episodes": [{"label": "ep", "kind": "chat", "seconds": 1.5}]}
    (reference / "manifest.json").write_text(json.dumps(manifest))
    # First use the actual main to obtain an independent baseline output.
    import runpy
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts/live_lens_atlas.py"
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: str(tmp_path))
    with monkeypatch.context() as patch:
        patch.setattr(
            sys,
            "argv",
            [
                str(script),
                "--records",
                str(reference),
                "--out",
                str(tmp_path / "source-atlas.json"),
            ],
        )
        runpy.run_path(str(script), run_name="__main__")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", original)
    expected = tmp_path / "source-atlas.json"
    result = module.legacy_identity(
        reference,
        expected,
        model="qwen35-4b",
        snapshot=tmp_path,
        expected_atlas_sha256=module.file_sha256(expected),
    )
    assert result["exact_json_equal"] is True
    assert huggingface_hub.snapshot_download is original
    mutated = tmp_path / "mutated"
    mutated.mkdir()
    write(mutated / "ep.jsonl")
    manifest["episodes"][0]["seconds"] = 2
    (mutated / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"episodes\[0\].seconds"):
        module.legacy_identity(
            mutated,
            expected,
            model="qwen35-4b",
            snapshot=tmp_path,
            expected_atlas_sha256=module.file_sha256(expected),
        )
    assert json.loads((mutated / "identity.json").read_text())["exact_json_equal"] is False
    assert huggingface_hub.snapshot_download is original


@pytest.mark.parametrize("arguments,code", [(["--help"], 0), (["--model", "qwen35-4b"], 2)])
def test_cli_refusals_do_not_import_mlx(arguments, code):
    import sys

    from local_llm_lab.spawn import run

    script = Path(__file__).resolve().parents[1] / "scripts/lens_replay.py"
    program = (
        "import runpy,sys; sys.argv=" + repr([str(script), *arguments]) + ";\n"
        "try: runpy.run_path(" + repr(str(script)) + ", run_name='__main__')\n"
        f"except SystemExit as e: assert e.code == {code}\n"
        "assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)\n"
    )
    result = run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def prepared_fixture(tmp_path, monkeypatch, *, identity=False):
    from dataclasses import replace

    import numpy as np

    from local_llm_lab.models import load_model_spec

    module = api()
    source = tmp_path / "source"
    source.mkdir()
    write(source / "ep.jsonl")
    manifest = {
        "model": "example/tiny",
        "layers": [1, 2],
        "episodes": [
            {
                "label": "ep",
                "record": "ep.jsonl",
                "record_sha256": module.file_sha256(source / "ep.jsonl"),
            }
        ],
    }
    lens = tmp_path / "lens.npz"
    np.savez(lens, J0=np.eye(2))
    manifest["lens_sha256"] = module.file_sha256(lens)
    (source / "manifest.json").write_text(json.dumps(manifest))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"hidden_size": 2, "num_hidden_layers": 2, "vocab_size": 5})
    )
    monkeypatch.setattr(
        module, "resolve_snapshot", lambda *a, **kw: {"snapshot_path": str(snapshot)}
    )
    spec = replace(load_model_spec("qwen35-4b"), hf_id="example/tiny")
    return module.prepare_replay(
        source,
        spec,
        tmp_path / "qwen35-4b-prose-jacobian",
        lens,
        lens_sha256=manifest["lens_sha256"],
        domain="prose",
        kind="jacobian",
    )


def test_preparation_validates_layers_lens_sources_and_exclusive_output(tmp_path, monkeypatch):
    module = api()
    prepared = prepared_fixture(tmp_path, monkeypatch)
    assert prepared.layers == (1, 2)
    assert not prepared.output.exists()
    assert prepared.spec.cache_strategy == "history"
    kwargs = dict(lens_sha256=prepared.lens_sha256, domain="prose", kind="jacobian")
    with pytest.raises(ValueError, match="final identity"):
        module.prepare_replay(
            prepared.source,
            prepared.spec,
            prepared.output,
            prepared.lens_path,
            layers=[1],
            **kwargs,
        )
    with pytest.raises(ValueError, match="hash"):
        module.prepare_replay(
            prepared.source,
            prepared.spec,
            prepared.output,
            prepared.lens_path,
            **(kwargs | {"lens_sha256": "0" * 64}),
        )
    prepared.output.mkdir()
    with pytest.raises(FileExistsError):
        module.prepare_replay(
            prepared.source, prepared.spec, prepared.output, prepared.lens_path, **kwargs
        )


def test_output_records_verified_and_manifest_is_separate(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = api()
    prepared = prepared_fixture(tmp_path, monkeypatch)
    loaded = SimpleNamespace(
        view=SimpleNamespace(hidden_size=2, num_layers=2),
        tokenizer=None,
        snapshot=prepared.snapshot,
        spec=prepared.spec,
        resolved=SimpleNamespace(cache_strategy="none"),
        lock_path=tmp_path / "lock",
    )

    def replay(view, tokenizer, lens, events, emit, **kwargs):
        for event in events[1:-1]:
            emit(event)
        return {"forwards": 2, "tokens": 3}

    monkeypatch.setattr(module, "replay_record", replay)
    result = module.run_replay(prepared, loaded, allocator_cache={"limit_bytes": 0})
    assert result["status"] == "complete"
    assert (prepared.output / "manifest.json").read_bytes() == (
        prepared.source / "manifest.json"
    ).read_bytes()
    provenance = json.loads((prepared.output / "replay-manifest.json").read_bytes())
    assert provenance["snapshot"] == prepared.snapshot
    assert provenance["retain_logits"] is False
    assert provenance["sources"] == list(prepared.records)
    module.read_source(prepared.output / "ep.jsonl")


def test_manifest_mutation_refused_before_output(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = api()
    prepared = prepared_fixture(tmp_path, monkeypatch)
    loaded = SimpleNamespace(
        view=SimpleNamespace(hidden_size=2, num_layers=2),
        snapshot=prepared.snapshot,
        spec=prepared.spec,
        resolved=SimpleNamespace(cache_strategy="none"),
    )
    (prepared.source / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest changed"):
        module.run_replay(prepared, loaded)
    assert not prepared.output.exists()
