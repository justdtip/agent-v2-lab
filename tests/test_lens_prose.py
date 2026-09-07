"""Requirements §14 pure contracts; fake forwards are not native acceptance evidence."""

from __future__ import annotations

import builtins
import copy
import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from local_llm_lab.forward import ForwardLedger
from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.lens_fitting import prose
from local_llm_lab.pipeline.lens_fitting.corpus import build_prose_corpus
from local_llm_lab.pipeline.lens_fitting.replay import read_source
from local_llm_lab.pipeline.live_lens.instruments import file_sha256
from local_llm_lab.pipeline.live_lens.session import read_record


@pytest.fixture(autouse=True)
def no_mlx(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"mlx", "mlx_lm"}, "pure §14 test imported MLX"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    yield
    assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)


class Tokenizer:
    bos_token = None
    eos_token_ids = {0}
    chat_template = "unused chat template"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        return {
            "input_ids": self.encode(text, add_special_tokens=add_special_tokens),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("section 14 forbids chat templates")


@pytest.fixture
def prepared(tmp_path):
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"hidden_size": 2, "num_hidden_layers": 2, "vocab_size": 128})
    )
    (snapshot / "model.safetensors").write_bytes(b"fake pure fixture, never loaded")
    spec = replace(load_model_spec("qwen35-4b"), hf_id=str(snapshot))
    tokenizer = Tokenizer()
    source = tmp_path / "source.txt"
    source.write_text("a" * (255 * 1024))
    asset = snapshot / "tokenizer_config.json"
    asset.write_text("{}")
    corpus = tmp_path / "corpus.json"
    build_prose_corpus([source], tokenizer, spec, corpus, tokenizer_files=[asset])
    lens = tmp_path / "lens.npz"
    np.savez(lens, J0=np.eye(2))
    plan = prose.make_plan(corpus, spec, lens, lens_sha256=file_sha256(lens), tokenizer=tokenizer)
    return SimpleNamespace(
        spec=spec,
        tokenizer=tokenizer,
        corpus=corpus,
        lens=lens,
        plan=plan,
        source=source,
        asset=asset,
        root=tmp_path,
    )


def test_section14_held_only_exact_order_raw_prefix_all_layers(prepared):
    p = prepared.plan
    assert [w["window_index"] for w in p["windows"]] == list(range(4, 255, 5))
    assert len(p["windows"]) == 51
    assert all(
        w["prompt_ids"] == [97] * 824 and w["prompt"] == "a" * 824 and w["max_tokens"] == 200
        for w in p["windows"]
    )
    assert p["sampler"] == {"temperature": 0.0, "kind": "greedy"}
    assert p["min_tokens"] == 32 and p["layers"] == [1, 2]
    assert p["cache_strategy"] == "none" and "ridge" in p["overlap"]
    assert "authored_continuation" not in p["windows"][0]
    prose.validate_plan(p, prepared.spec, prepared.tokenizer)


@pytest.mark.parametrize("target", ["source", "asset", "lens", "corpus"])
def test_section14_bound_file_hash_drift_refused(prepared, target):
    path = getattr(prepared, target)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        prose.validate_plan(prepared.plan, prepared.spec, prepared.tokenizer)


@pytest.mark.parametrize("target", ["window_index", "prompt_ids", "max_tokens", "plan_sha256"])
def test_section14_frozen_plan_drift_refused(prepared, target):
    plan = copy.deepcopy(prepared.plan)
    if target == "plan_sha256":
        plan[target] = "0" * 64
    else:
        plan["windows"][0][target] = [5] if target == "prompt_ids" else 5
    with pytest.raises(ValueError, match="drift"):
        prose.validate_plan(plan, prepared.spec, prepared.tokenizer)


def test_section14_tokenizer_roundtrip_drift_refused(prepared, monkeypatch):
    monkeypatch.setattr(prepared.tokenizer, "decode", lambda ids: "b" * len(ids))
    with pytest.raises(ValueError, match="roundtrip"):
        prose.validate_plan(prepared.plan, prepared.spec, prepared.tokenizer)


class Session:
    """Pure record-producing fake using the real ledger and public generation seam."""

    def __init__(self, view=None, reader=None, write=None, *, layers=(1, 2), **kwargs):
        self.write = write or (lambda row: None)
        self.layers = layers
        self.context = None
        self.generated = []

    def set_context(self, **metadata):
        self.context = metadata

    def event(self, kind, **kwargs):
        self.write({"kind": kind, "turn": 0, **kwargs})

    @contextmanager
    def generation(self, model, tokenizer, prompt, *, turn_cache):
        assert turn_cache is None
        assert self.context["kind"] == "prose"
        assert self.context["messages"] == [{"role": "user", "content": prompt}]
        self.ledger = ForwardLedger(tokenizer.encode(prompt, add_special_tokens=True))
        self.event(
            "begin_turn",
            prompt=prompt,
            prompt_ids=self.ledger.prompt_ids,
            context=self.context,
            layers=list(self.layers),
            attention_blocks=[],
            top_k=1,
            audit_modulus=23,
            audit_seed=0,
            rank_horizons=[1, 4, 8],
            distribution_capacity=16,
        )
        self.forward(self.ledger.prompt_ids)
        try:
            yield self
        except BaseException:
            self.event(
                "end_turn",
                emitted_count=len(self.generated),
                forwarded_count=self.ledger.offset,
                status="aborted",
            )
            raise
        else:
            self.event(
                "end_turn",
                emitted_count=len(self.generated),
                forwarded_count=self.ledger.offset,
                status="complete",
            )

    def forward(self, ids):
        offset = self.ledger.offset
        for local in range(len(ids)):
            self.event(
                "reading", position=offset + local, top={str(layer): [1] for layer in self.layers}
            )
        self.event(
            "forward",
            offset=offset,
            input_ids=ids,
            logits_sha256="a" * 64,
            logits_shape=[1, len(ids), 128],
            argmax=[[1] * len(ids)],
        )
        self.ledger.record(offset, ids)

    def emitted(self, token):
        position = len(self.ledger.prompt_ids) + len(self.generated)
        self.ledger.emitted(token)
        self.generated.append(token)
        self.event("emitted", position=position, token_id=token)
        for h in (1, 4, 8):
            if max(0, self.ledger.offset - 16) <= position - h < self.ledger.offset:
                for layer in self.layers:
                    self.event(
                        "rank",
                        position=position - h,
                        layer=layer,
                        horizon=h,
                        token_id=token,
                        rank=1,
                        probability=0.5,
                    )


def stream_factory(count, *, fail=False, close_log=None, finish=None):
    def generate(captured, tokenizer, prompt, *, sampler, max_tokens):
        assert sampler == "greedy" and max_tokens == 200
        try:
            for index in range(count):
                token = 0 if index == count - 1 and count < 200 else 1
                # Simulate the real library's forwarded but not-yet-emitted lookahead.
                captured.forward([token])
                yield SimpleNamespace(
                    token=token,
                    text='```json\n{"name":"finish"}\n```',
                    finish_reason=(finish or ("stop" if count < 200 else "length"))
                    if index == count - 1
                    else None,
                )
                if fail:
                    raise RuntimeError("injected generator failure")
        finally:
            if close_log is not None:
                close_log.append(True)

    return generate


@pytest.mark.parametrize("count", [1, 31, 32, 200])
def test_section14_greedy_cap_eos_public_capture_and_tool_close_ignored(prepared, count):
    closed = []
    session = Session()
    result = prose.generate_prose(
        session,
        object(),
        prepared.tokenizer,
        prepared.plan["windows"][0],
        sampler="greedy",
        stream_generate=stream_factory(count, close_log=closed),
    )
    assert result["generated_tokens"] == count
    assert len(session.generated) == count
    assert session.ledger.offset == 824 + count
    assert result["finish_reason"] == ("length" if count == 200 else "stop")
    assert closed == [True]


def test_section14_stream_closed_and_aborted_on_error(prepared):
    closed, events = [], []
    with pytest.raises(RuntimeError, match="injected"):
        prose.generate_prose(
            Session(write=events.append),
            object(),
            prepared.tokenizer,
            prepared.plan["windows"][0],
            sampler="greedy",
            stream_generate=stream_factory(32, fail=True, close_log=closed),
        )
    assert closed == [True] and events[-1]["status"] == "aborted"


@pytest.mark.parametrize("count,finish", [(31, "length"), (200, "stop"), (201, "length")])
def test_section14_invalid_native_terminal_cannot_complete(prepared, count, finish):
    with pytest.raises(ValueError):
        prose.generate_prose(
            Session(),
            object(),
            prepared.tokenizer,
            prepared.plan["windows"][0],
            sampler="greedy",
            stream_generate=stream_factory(count, finish=finish),
        )


def capture(prepared, output, counts, *, fail=False):
    plan = copy.deepcopy(prepared.plan)
    plan["windows"] = plan["windows"][: len(counts)]  # Small pure driver fixture only.
    counts = iter(counts)

    def generate(*args, **kwargs):
        return stream_factory(next(counts), fail=fail)(*args, **kwargs)

    loaded = SimpleNamespace(
        view=None,
        model=object(),
        tokenizer=prepared.tokenizer,
        lock_path=prepared.root / "primary.lock",
    )
    return prose.capture_windows(
        output,
        plan,
        loaded,
        None,
        stream_generate=generate,
        sampler="greedy",
        session_factory=Session,
        readout_factory=lambda *_: None,
        resources=lambda: {},
        progress=lambda _: None,
    )


def test_section14_minimum_drops_audit_and_unchanged_readers(prepared):
    output = prose.register_plan(prepared.root / "run", prepared.plan)
    manifest = capture(prepared, output, [31, 32])
    assert manifest["status"] == "complete"
    assert manifest["accepted_count"] == manifest["dropped_count"] == 1
    assert manifest["episodes"][0]["window_index"] == 9
    dropped = manifest["dropped"][0]
    assert dropped["window_index"] == 4 and dropped["reason"] == "continuation_under_32"
    assert len(list(output.glob("*.jsonl"))) == 1
    for entry in manifest["episodes"] + manifest["dropped"]:
        path = output / entry["record"]
        assert read_record(path) == read_source(path)[0]
        assert file_sha256(path) == entry["record_sha256"]
        assert entry["kind"] == "prose" and entry["seconds"] >= 0


def test_section14_partial_output_exclusive_no_resume(prepared):
    output = prose.register_plan(prepared.root / "run", prepared.plan)
    with pytest.raises(FileExistsError):
        prose.register_plan(output, prepared.plan)
    with pytest.raises(RuntimeError, match="injected"):
        capture(prepared, output, [32], fail=True)
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["status"] == "partial" and not manifest["episodes"]
    path = next(output.glob("*.jsonl"))
    with pytest.raises(ValueError, match="incomplete"):
        read_record(path)
    with pytest.raises(FileExistsError):
        capture(prepared, output, [32])


def test_section14_registration_and_capture_true_precede_loader(prepared, monkeypatch):
    output = prose.register_plan(prepared.root / "run", prepared.plan)
    monkeypatch.setattr(prose, "corpus_tokenizer", lambda *a, **k: prepared.tokenizer)
    calls = []

    def loader(preflight, *, capture):
        assert capture is True and preflight.snapshot == prepared.plan["snapshot"]
        assert (output / "README.md").exists() and (output / "plan.json").exists()
        calls.append(preflight.spec.cache_strategy)
        raise RuntimeError("fake loader boundary; no checkpoint")

    monkeypatch.setattr(prose, "load_runtime", loader)
    with pytest.raises(RuntimeError, match="fake loader"):
        prose.execute_plan(output, prepared.spec)
    assert calls == [prepared.spec.cache_strategy]
    with pytest.raises(FileExistsError):
        prose.execute_plan(output, prepared.spec)
    assert len(calls) == 1


def test_section14_plan_drift_never_reaches_loader(prepared, monkeypatch):
    output = prose.register_plan(prepared.root / "run", prepared.plan)
    monkeypatch.setattr(prose, "corpus_tokenizer", lambda *a, **k: prepared.tokenizer)
    monkeypatch.setattr(prose, "load_runtime", lambda *a, **k: pytest.fail("loader reached"))
    prepared.source.write_text("changed")
    with pytest.raises(ValueError):
        prose.execute_plan(output, prepared.spec)


def test_section14_history_default_resolves_none_before_load_policy(prepared, monkeypatch):
    from local_llm_lab import runlock
    from local_llm_lab.pipeline.lens_fitting import runtime

    calls = []
    monkeypatch.setattr(runtime, "primary_worktree", lambda: prepared.root)
    monkeypatch.setattr(runlock, "hold_model_run_lock", lambda **kw: calls.append("lock"))
    monkeypatch.setattr(runlock, "PROJECT_ROOT", prepared.root)

    def policy(spec, adapter):
        assert calls == ["lock"]
        assert spec.cache_strategy == "none" and adapter is None
        raise RuntimeError("pure policy boundary")

    monkeypatch.setattr(runtime, "_load_policy", policy)
    spec = replace(prepared.spec, cache_strategy="history")
    with pytest.raises(RuntimeError, match="pure policy"):
        runtime.load_runtime(
            SimpleNamespace(spec=spec, snapshot=prepared.plan["snapshot"]), capture=True
        )


def test_section14_missing_held_window_refused(prepared, monkeypatch):
    original = prose.read_corpus
    monkeypatch.setattr(prose, "read_corpus", lambda path: original(path)[5:])
    with pytest.raises(ValueError, match="51 held"):
        prose.validate_plan(prepared.plan, prepared.spec, prepared.tokenizer)


def test_section14_capture_emission_error_closes_suspended_stream(prepared):
    closed = []

    class BrokenSession(Session):
        def emitted(self, token):
            raise RuntimeError("capture emission failed")

    with pytest.raises(RuntimeError, match="capture emission"):
        prose.generate_prose(
            BrokenSession(),
            object(),
            prepared.tokenizer,
            prepared.plan["windows"][0],
            sampler="greedy",
            stream_generate=stream_factory(200, close_log=closed),
        )
    assert closed == [True]


def test_section14_runtime_tokenizer_drift_precedes_session(prepared, monkeypatch):
    monkeypatch.setattr(prepared.tokenizer, "encode", lambda *a, **k: [1])
    with pytest.raises(ValueError, match="runtime tokenizer"):
        prose.generate_prose(
            None,
            object(),
            prepared.tokenizer,
            prepared.plan["windows"][0],
            sampler="greedy",
            stream_generate=lambda *a, **k: pytest.fail("stream reached"),
        )


@pytest.fixture
def opaque_descriptor(prepared):
    """The frozen corpus stores resolved blob paths, losing filename roles."""
    snapshot = prepared.asset.parent
    blobs = prepared.root / "blobs"
    blobs.mkdir()
    assets = [snapshot / "config.json", prepared.asset]
    for number, path in enumerate(assets):
        blob = blobs / (str(number) * 40)
        path.rename(blob)
        path.symlink_to(blob)
    corpus = prepared.root / "opaque-corpus.json"
    build_prose_corpus(
        [prepared.source], prepared.tokenizer, prepared.spec, corpus, tokenizer_files=assets
    )
    descriptor = json.loads(corpus.read_bytes())["tokenizer"]["files"]
    assert all(Path(row["path"]).parent == blobs for row in descriptor)
    return corpus


def test_section14_opaque_blob_roles_resolved_from_exact_snapshot(
    prepared, opaque_descriptor, monkeypatch
):
    calls = []

    def load(directory, **kwargs):
        calls.append((directory, kwargs))
        return prepared.tokenizer

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)),
    )
    assert prose.corpus_tokenizer(opaque_descriptor, prepared.spec) is prepared.tokenizer
    assert calls == [
        (
            str(prepared.asset.parent),
            {"local_files_only": True, "trust_remote_code": False, "use_fast": True},
        )
    ]


def test_section14_mismatched_snapshot_asset_refused_before_tokenizer(
    prepared, opaque_descriptor, monkeypatch
):
    # Even equal bytes at a different resolved path are not the bound corpus asset.
    replacement = prepared.root / "different-blob"
    replacement.write_bytes(prepared.asset.read_bytes())
    prepared.asset.unlink()
    prepared.asset.symlink_to(replacement)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *a, **k: pytest.fail("loader reached")
            )
        ),
    )
    with pytest.raises(ValueError, match="snapshot tokenizer assets"):
        prose.corpus_tokenizer(opaque_descriptor, prepared.spec)
