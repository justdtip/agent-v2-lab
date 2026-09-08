"""Source-only launch boundaries: tests never import MLX or load checkpoints."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from local_llm_lab.models import load_model_spec


def driver():
    path = Path(__file__).parents[1] / "scripts/gemma_transcript_lens.py"
    spec = importlib.util.spec_from_file_location("gemma_transcript_lens_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def registered(tmp_path, monkeypatch):
    mod = driver()
    specs = {}
    snapshots = {}
    for name in mod.FIT_MODELS:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"num_hidden_layers": 34}))
        (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": "template"}))
        (directory / "tokenizer.json").write_text("{}")
        specs[name] = replace(load_model_spec(name), hf_id=str(directory))
        files = [
            {"name": p.name, "sha256": mod.file_record(p)["sha256"], "bytes": p.stat().st_size}
            for p in sorted(directory.iterdir())
        ]
        snapshots[name] = {
            "hf_id": str(directory),
            "snapshot_path": str(directory),
            "snapshot_sha256": "a" * 64,
            "files": files,
        }
    monkeypatch.setattr(mod, "model_spec", lambda name: specs[name])
    monkeypatch.setattr(mod, "verify_rendering_gate", lambda _: None)
    tokenizer = mod.tokenizer_identity_from_directory(Path(specs["gemma3-4b"].hf_id))
    registration = {
        "schema_version": 1,
        "fitting_context_tokens": 2048,
        "format": mod.FORMAT,
        "producer_model": "gemma3-4b",
        "fit_models": list(mod.FIT_MODELS),
        "model_identity": {"base": specs["gemma3-4b"].base, "training": None, "num_layers": 34},
        "tokenizer": tokenizer,
        "tokenizer_directory": specs["gemma3-4b"].hf_id,
        "snapshots": snapshots,
        "cohorts": mod.plan_cohorts(),
        "evaluation": dict(mod.EVALUATION),
        "rendering_gate": {},
        "capture_projection": {"peak_gib": 8.0, "basis": "source buffer geometry; unmeasured"},
        "capture_limits": {"max_prompt_tokens": 8192, "max_forward_tokens": 8393},
    }
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration))
    return mod, path, registration


def test_inspect_is_default_and_never_hashes_weights(registered, monkeypatch, capsys):
    mod, path, _ = registered
    monkeypatch.setattr(mod, "verify_snapshot", lambda *_: pytest.fail("weights touched"))
    assert mod.main(["--registration", str(path)]) == 0
    assert "inspected" in capsys.readouterr().out


def test_changed_task_fingerprint_refused(registered):
    mod, path, data = registered
    data["cohorts"][0]["tasks"][0]["fingerprint"] = "bad"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="cohort"):
        mod.read_registration(path)


def test_capture_requires_execute_and_owned_window_before_snapshot(
    registered, monkeypatch, tmp_path
):
    mod, path, _ = registered
    monkeypatch.setattr(mod, "verify_snapshot", lambda *_: pytest.fail("weights touched"))
    monkeypatch.setattr(
        mod,
        "require_owned_window",
        lambda: (_ for _ in ()).throw(ValueError("owned window required")),
    )
    with pytest.raises(ValueError, match="execute"):
        mod.capture(
            path,
            split="train",
            record=tmp_path / "capture.jsonl",
            output=tmp_path / "evaluation.json",
            execute=False,
        )
    with pytest.raises(ValueError, match="owned window"):
        mod.capture(
            path,
            split="train",
            record=tmp_path / "capture.jsonl",
            output=tmp_path / "evaluation.json",
            execute=True,
        )


def test_capture_passes_exact_evaluation_cohort(registered, monkeypatch, tmp_path):
    mod, path, data = registered
    monkeypatch.setattr(mod, "require_owned_window", lambda: tmp_path)
    monkeypatch.setattr(mod, "verify_snapshot", lambda *_: None)
    seen = []

    def stage(**kwargs):
        seen.append(kwargs)
        raise LookupError("runner reached")

    monkeypatch.setattr(mod, "run_evaluation", stage)
    with pytest.raises(LookupError, match="runner reached"):
        mod.capture(
            path,
            split="train1",
            record=tmp_path / "capture.jsonl",
            output=tmp_path / "evaluation.json",
            execute=True,
        )
    kw = seen[0]
    assert (kw["split"], kw["difficulty"], kw["limit"], kw["seed"]) == ("train1", 1, 24, 20260902)
    assert (kw["temperature"], kw["max_tokens"], kw["max_steps"], kw["keep_last"]) == (
        0.0,
        200,
        24,
        2,
    )
    assert kw["use_cache"] is False and kw["spec"].cache_strategy == "none"
    assert kw["capture"].max_prompt_tokens == 8192


def test_oversized_prompt_refused_before_forward(registered):
    mod, _, _ = registered
    capture = mod.BoundedTranscriptCapture(
        lambda _: None, max_prompt_tokens=3, max_forward_tokens=204
    )
    tokenizer = type(
        "Tokenizer",
        (),
        {"bos_token": None, "encode": lambda self, text, **kw: list(range(len(text)))},
    )()
    with (
        pytest.raises(ValueError, match="prompt exceeds"),
        capture.generation(None, tokenizer, "four", turn_cache=None),
    ):
        pytest.fail("forward boundary entered")


def test_owned_window_is_primary_and_refuses_missing_foreign_expired(
    registered, monkeypatch, tmp_path
):
    from types import SimpleNamespace

    mod, _, _ = registered
    monkeypatch.setattr(mod, "primary_worktree", lambda: tmp_path)
    monkeypatch.setenv(mod.WINDOW_HOLDER_ENV, "ours")
    paths = []
    window = SimpleNamespace(
        nonce="ours", holder_state="running", expected_end_epoch=mod.time.time() + 300
    )

    def read(path):
        paths.append(path)
        return window

    monkeypatch.setattr(mod, "read_window", read)
    assert mod.require_owned_window() == tmp_path
    assert paths == [tmp_path / mod.WINDOW_RELATIVE_PATH]
    window.nonce = "foreign"
    with pytest.raises(ValueError, match="current owned"):
        mod.require_owned_window()
    window.nonce = "ours"
    window.expected_end_epoch = mod.time.time() - 1
    with pytest.raises(ValueError, match="current owned"):
        mod.require_owned_window()
    monkeypatch.setattr(mod, "read_window", lambda _: None)
    with pytest.raises(ValueError, match="current owned"):
        mod.require_owned_window()


def test_freeze_uses_only_local_tokenizer_and_keeps_ruling_manifest(
    registered, monkeypatch, tmp_path
):
    mod, path, data = registered
    binding = mod.file_record(path)
    captures = []
    events = {}
    for cohort in data["cohorts"]:
        source = tmp_path / (cohort["split"] + ".jsonl")
        source.write_text("fixture")
        captures.append(source)
        provenance = {
            "registration": binding,
            "fitting_context_tokens": 2048,
            "cohort": cohort,
            "producer_snapshot": data["snapshots"]["gemma3-4b"],
            "evaluation": data["evaluation"],
            "capture_limits": data["capture_limits"],
            "rendering_gate": data["rendering_gate"],
        }
        turns = [
            {
                "kind": "begin_turn",
                "context": {
                    "task_id": task.task_id,
                    "step": 0,
                    "keep_last": 2,
                    "messages": [{}, {"role": "user", "content": task.prompt}],
                },
            }
            for task in mod.make_tasks(
                cohort["split"], 24, 20260902, difficulty=cohort["difficulty"]
            )
        ]
        events[source] = [{"kind": "manifest", "provenance": provenance}, *turns]
    monkeypatch.setattr(mod, "read_transcript", lambda p: events[p])
    monkeypatch.setattr(mod, "verify_snapshot", lambda *_: pytest.fail("weights touched"))
    seen = []
    tokenizer = type("Tokenizer", (), {"is_fast": True, "chat_template": "template"})()
    monkeypatch.setattr(mod, "load_local_tokenizer", lambda directory: tokenizer)

    def build(*args, **kwargs):
        seen.append((args, kwargs))
        return {"manifest_sha256": "f" * 64, "acceptance": {"status": "ruling_required"}}

    monkeypatch.setattr(mod, "build_transcript_corpus", build)
    result = mod.freeze(path, captures=list(reversed(captures)), output=tmp_path / "corpus.json")
    assert result["status"] == "ruling_required"
    assert seen[0][0][0] == captures
    assert seen[0][0][1] is tokenizer
    assert seen[0][1]["max_tokens"] == 2048


def test_captured_task_prompt_must_match_registered_fingerprint(registered):
    mod, _, data = registered
    cohort = data["cohorts"][0]
    event = {
        "kind": "begin_turn",
        "context": {
            "task_id": cohort["tasks"][0]["task_id"],
            "step": 0,
            "keep_last": 2,
            "messages": [{}, {"role": "user", "content": "changed task"}],
        },
    }
    with pytest.raises(ValueError, match="fingerprint"):
        mod.validate_captured_cohort([event], cohort)
