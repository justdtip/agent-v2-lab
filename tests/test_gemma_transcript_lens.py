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


@pytest.fixture
def committed_rendering_gate(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    mod = driver()
    primary, running = tmp_path / "primary", tmp_path / "running"
    primary.mkdir()
    running.mkdir()
    monkeypatch.setattr(mod, "primary_worktree", lambda: primary)
    monkeypatch.setattr(mod, "__file__", str(running / "scripts" / "driver.py"))
    landed, ruling = "7d2a18a" + "0" * 33, "a1ff0b0" + "0" * 33
    environment, amendment, span = "020aa89" + "0" * 33, "ec4b9d1" + "0" * 33, "2edad6e" + "0" * 33
    paths = [
        "design_specifications/pending/CODEX-TASKS-2026-09-08.md",
        "research/records/GEMMA3-JSPACE-MAP-2026-09-08/DIAGNOSTIC-RERUN.md",
        "src/local_llm_lab/pipeline/protocol.py",
        "src/local_llm_lab/pipeline/env.py",
        "src/local_llm_lab/agent_tasks.py",
        "src/local_llm_lab/pipeline/live_lens/session.py",
        "configs/models/gemma3-4b.yaml",
        "configs/models/gemma3-4b-bf16.yaml",
    ]
    blobs, records = {}, []
    for relative in paths:
        commit = (
            amendment
            if relative.startswith("design_specifications/")
            else environment
            if relative in ("src/local_llm_lab/pipeline/env.py", "src/local_llm_lab/agent_tasks.py")
            else span
            if relative.endswith("live_lens/session.py")
            else landed
        )
        blob = ("committed " + relative).encode()
        blobs[commit + ":" + relative] = blob
        for root in (primary, running):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        records.append(
            {
                "path": str(primary / relative),
                "commit": commit,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "storage": "git_blob",
            }
        )

    def git(argv, **kwargs):
        if argv[3] == "rev-parse":
            return SimpleNamespace(stdout=argv[4].removesuffix("^{commit}") + "\n")
        if argv[3] == "merge-base":
            return SimpleNamespace(returncode=0)
        assert argv[3] == "show"
        return SimpleNamespace(stdout=blobs[argv[4]])

    monkeypatch.setattr(mod, "run", git)
    return (
        mod,
        {
            "landed_commit": landed,
            "ruling_commit": ruling,
            "files": records,
            "environment_commit": environment,
            "second_amendment_commit": amendment,
            "span_commit": span,
        },
        primary,
        running,
    )


def test_historical_rendering_evidence_survives_later_document_amendment(committed_rendering_gate):
    mod, gate, primary, _ = committed_rendering_gate
    report = primary / "research/records/GEMMA3-JSPACE-MAP-2026-09-08/DIAGNOSTIC-RERUN.md"
    report.write_bytes(report.read_bytes() + b"\nLater analysis appended.")
    mod.verify_rendering_gate(gate)


def test_rendering_evidence_rejects_wrong_committed_blob_hash(committed_rendering_gate):
    mod, gate, _, _ = committed_rendering_gate
    gate["files"][1]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="committed blob hash"):
        mod.verify_rendering_gate(gate)


@pytest.mark.parametrize("checkout", ["primary", "running"])
def test_rendering_source_must_still_match_committed_fix(committed_rendering_gate, checkout):
    mod, gate, primary, running = committed_rendering_gate
    root = primary if checkout == "primary" else running
    (root / "src/local_llm_lab/pipeline/protocol.py").write_text("changed source")
    with pytest.raises(ValueError, match="checkout differs"):
        mod.verify_rendering_gate(gate)


@pytest.mark.parametrize("changed", ["snapshot", "window"])
def test_capture_rechecks_identity_and_window_before_complete_footer(
    registered, monkeypatch, tmp_path, changed
):
    mod, path, data = registered
    calls = []
    finished = False

    def window():
        calls.append("window")
        if finished and changed == "window":
            raise ValueError("window ownership changed during run")
        return tmp_path

    def snapshot(*args):
        calls.append("snapshot")
        if finished and changed == "snapshot":
            raise ValueError("snapshot changed during run")

    def stage(**kwargs):
        nonlocal finished
        calls.append("run")
        kwargs["output"].write_text(
            json.dumps(
                {"trajectories": [{"task_id": t["task_id"]} for t in data["cohorts"][0]["tasks"]]}
            )
        )
        finished = True

    monkeypatch.setattr(mod, "require_owned_window", window)
    monkeypatch.setattr(mod, "verify_snapshot", snapshot)
    monkeypatch.setattr(mod, "run_evaluation", stage)
    record = tmp_path / "capture.jsonl"
    with pytest.raises(ValueError, match="changed during run"):
        mod.capture(
            path, split="train", record=record, output=tmp_path / "evaluation.json", execute=True
        )
    events = [json.loads(line)["event"] for line in record.read_text().splitlines()]
    assert events[-1] == {"kind": "end_record", "status": "aborted"}
    assert calls[:4] == ["window", "snapshot", "window", "run"]


def test_environment_and_second_amendment_gate_cannot_be_omitted(committed_rendering_gate):
    mod, gate, _, _ = committed_rendering_gate
    gate.pop("environment_commit")
    with pytest.raises(ValueError, match="landed rendering fix"):
        mod.verify_rendering_gate(gate)


def test_environment_evidence_must_bind_environment_fix_revision(committed_rendering_gate):
    mod, gate, _, _ = committed_rendering_gate
    record = next(row for row in gate["files"] if row["path"].endswith("pipeline/env.py"))
    record["commit"] = gate["landed_commit"]
    with pytest.raises(ValueError, match="required revision"):
        mod.verify_rendering_gate(gate)
