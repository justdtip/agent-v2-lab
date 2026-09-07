"""Requirements §3.5/5/14: tiny synthetic statistics, never native evidence."""

import copy
import importlib

import pytest


def api():
    return importlib.import_module("local_llm_lab.pipeline.lens_fitting.profiles")


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 1), (1, 2), (2, 3)]}

    def decode(self, ids):
        return "x"


def events():
    return [
        {
            "kind": "begin_turn",
            "turn": 0,
            "prompt": "abc",
            "prompt_ids": [1, 2, 3],
            "context": {},
            "layers": [1, 2, 3],
        },
        {"kind": "reading", "turn": 0, "position": 0, "top": {"1": [2], "2": [3], "3": [2]}},
        {"kind": "reading", "turn": 0, "position": 2, "top": {"1": [4], "2": [4], "3": [4]}},
        {"kind": "emitted", "turn": 0, "position": 3, "token_id": 4},
        *[
            {
                "kind": "rank",
                "turn": 0,
                "position": 2,
                "horizon": 1,
                "layer": layer,
                "token_id": 4,
                "rank": rank,
            }
            for layer, rank in [(1, 5), (2, 15), (3, 20)]
        ],
    ]


def summarize(rows=None, label="test"):
    return api().summarize_episode(
        rows or events(),
        {"label": label, "kind": "prose"},
        Tokenizer(),
        layers=[1, 2, 3],
        lens_kind="regression",
        validation={},
    )


def test_exact_pairs_and_prompt_boundary():
    """§3.5/14: emitted targets own base rate; prompt boundary is excluded."""
    rows = summarize()
    row = next(
        r for r in rows if r["summary"] == "foreknowledge" and r["layer"] == 1 and r["horizon"] == 1
    )
    assert (row["numerator"], row["n"], row["share"], row["base_rate"]["share"]) == (1, 1, 1, 0)
    assert row["span"] == "continuation" and row["span_role"] == "emitted_target"
    assert row["biased_by_construction"] and not row["interpretive_eligible"]
    prompt = next(r for r in rows if r["summary"] == "prompt_agreement" and r["layer"] == 1)
    assert (prompt["numerator"], prompt["n"]) == (1, 1)


@pytest.mark.parametrize("change", ["missing", "duplicate", "horizon", "turn", "target"])
def test_pair_mismatch_fails(change):
    """§2: exact final counterpart, no filtering or coordinate substitution."""
    rows = events()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    elif change == "horizon":
        rows[-1]["horizon"] = 4
    elif change == "turn":
        rows[-1]["turn"] = 1
    else:
        rows[-1]["token_id"] = 5
    with pytest.raises(ValueError):
        summarize(rows)


def test_episode_isolation_and_zero_denominator():
    """§5: changing another episode cannot alter this episode; n=0 is null."""
    before = summarize(label="one")
    other = events()
    other[-1]["rank"] = 1
    changed = summarize(other, label="two")
    assert summarize(label="one") == before
    assert changed != before
    empty = next(r for r in before if r["horizon"] == 8)
    assert empty["n"] == 0 and empty["share"] is None and empty["base_rate"]["share"] is None


def test_authored_tail_is_never_ranked_and_alignment_fails_closed():
    """§14: descriptive authored tails cannot create emitted targets."""
    rows = events()
    rows[0]["authored_continuation"] = [4, 5, 6]
    assert summarize(rows) == summarize()
    rows[-1]["position"] = 3
    with pytest.raises(ValueError, match="actual emitted"):
        summarize(rows)
    rows = events()
    rows[0]["prompt_ids"] = [9, 8, 7]
    with pytest.raises(ValueError, match="exact recorded"):
        summarize(rows)


def test_map_status_is_visible_and_controls_eligibility():
    """§3.6: failed/inconclusive map rows remain data, not refit inventions."""
    for status in ("pass", "fail", "inconclusive"):
        rows = api().summarize_episode(
            events(),
            {"label": "fixture", "kind": "prose"},
            Tokenizer(),
            layers=[1, 2, 3],
            lens_kind="jacobian",
            validation={"1": {"map_check": {"outcome": status}}},
        )
        row = next(r for r in rows if r["layer"] == 1 and r["horizon"] == 1)
        assert row["map_check"] == status and row["n"] == 1
        assert row["interpretive_eligible"] == (status == "pass")
        assert row["interpretation"] is None


def test_specificity_distinguishes_missing_domain_and_chat():
    """§3.5/14: missing prose cells remain unavailable; chat cannot fill them."""
    sets = [
        {
            "name": "tiny-agentic-regression",
            "model": "tiny",
            "kind": "regression",
            "domain": "agentic",
            "rows": [{"episode": "chat", "episode_kind": "chat"}],
        }
    ]
    cells = api().specificity(sets)
    assert len(cells) == 6
    assert [c["evaluation_material"] for c in cells if c["status"] == "available"] == ["pilot_chat"]
    assert all(
        "ridge weight" in c["overlap"]
        for c in cells
        if c["evaluation_material"] == "held_prose_continuations"
    )


def test_input_bytes_and_complete_membership_rechecked(tmp_path):
    """§3.5: changed content or newly added records invalidate consumption."""
    path = tmp_path / "value.json"
    path.write_text("{}")
    inputs = api().Inputs()
    inputs.json(path)
    inputs.record_set(tmp_path, [])
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed input"):
        inputs.recheck()
    inputs = api().Inputs()
    inputs.record_set(tmp_path, [])
    (tmp_path / "extra.jsonl").write_text("")
    with pytest.raises(ValueError, match="record set"):
        inputs.recheck()


def test_html_escapes_script_termination_and_is_standalone():
    """§3.5: data is inert; no remote resources or implicit episode pooling."""
    page = api().render_page(
        {"sets": [], "specificity": [], "test": "</script><script>alert(1)</script>&"}
    )
    assert "<script>alert(1)" not in page
    assert "\\u003c/script\\u003e" in page
    assert "https://" not in page and "http://www.w3.org/2000/svg" in page
    assert "aria-label" in page and "source_prompt_position" not in page


def record_fixture(path, provenance=None):
    """Tiny valid record, explicitly synthetic, not acceptable native evidence."""
    from local_llm_lab.pipeline.live_lens.session import RecordWriter

    rows = [
        dict(
            kind="begin_turn",
            turn=0,
            prompt="abc",
            prompt_ids=[1, 2, 3],
            context={},
            layers=[1, 2, 3],
            attention_blocks=[],
            top_k=1,
            audit_modulus=23,
            audit_seed=0,
            rank_horizons=[1, 4, 8],
            distribution_capacity=16,
        ),
        *[
            dict(kind="reading", turn=0, position=p, top={"1": [2], "2": [2], "3": [2]})
            for p in range(3)
        ],
        dict(
            kind="forward",
            turn=0,
            offset=0,
            input_ids=[1, 2, 3],
            logits_sha256="a" * 64,
            logits_shape=[1, 3, 5],
            argmax=[[2, 2, 2]],
        ),
        dict(kind="emitted", turn=0, position=3, token_id=4),
        *[
            dict(
                kind="rank",
                turn=0,
                position=2,
                layer=layer,
                horizon=1,
                token_id=4,
                rank=1,
                probability=0.5,
            )
            for layer in [1, 2, 3]
        ],
        dict(kind="end_turn", turn=0, emitted_count=1, forwarded_count=3, status="complete"),
    ]
    with RecordWriter(path, provenance or {"synthetic_test_fixture": True}) as writer:
        for row in rows:
            writer(row)


def replay_fixture(tmp_path):
    import json

    module = api()
    source, output = tmp_path / "source", tmp_path / "replay"
    source.mkdir()
    output.mkdir()
    record_fixture(source / "test.jsonl")
    _, identity = module.read_source(source / "test.jsonl")
    historical = {
        "model": "synthetic-test",
        "lens_sha256": "old-hosted-hash",
        "layers": [1, 2, 3],
        "episodes": [
            {
                "label": "test",
                "kind": "prose",
                "record": "test.jsonl",
                "record_sha256": identity["sha256"],
            }
        ],
    }
    (output / "manifest.json").write_text(json.dumps(historical))
    current = {
        "lens_sha256": "new-lens-hash",
        "domain": "agentic",
        "kind": "regression",
        "layers": [1, 2, 3],
        "snapshot": {"synthetic_test_fixture": True},
        "source_manifest_sha256": module.file_sha256(output / "manifest.json"),
        "sources": [{"label": "test", "identity": identity}],
    }
    (output / "replay-manifest.json").write_text(json.dumps(current))
    record_fixture(output / "test.jsonl", current | {"episode": "test"})
    complete = {
        "status": "complete",
        "outputs": [
            {
                "label": "test",
                "sha256": module.file_sha256(output / "test.jsonl"),
                "forwards": 1,
                "tokens": 3,
            }
        ],
    }
    (output / "replay-complete.json").write_text(json.dumps(complete))
    return output


def test_current_replay_hashes_override_historical_hashes(tmp_path):
    """Task4: immutable historical manifest never authenticates new lens outputs."""
    directory = replay_fixture(tmp_path)
    historical, current, records = api().replay_records(directory, api().Inputs())
    assert historical["lens_sha256"] != current["lens_sha256"]
    assert len(records) == 1
    assert api().file_sha256(directory / "test.jsonl") != historical["episodes"][0]["record_sha256"]


@pytest.mark.parametrize("target", ["test.jsonl", "replay-manifest.json", "source"])
def test_changed_replay_inputs_refused(tmp_path, target):
    """§3.4/3.5: new output, source and current lens hashes are all binding."""
    import json

    directory = replay_fixture(tmp_path)
    if target == "source":
        path = tmp_path / "source/test.jsonl"
        path.write_text(path.read_text() + "\n")
    elif target == "replay-manifest.json":
        path = directory / target
        value = json.loads(path.read_text())
        value["lens_sha256"] = "changed"
        path.write_text(json.dumps(value))
    else:
        path = directory / target
        path.write_text(path.read_text() + "\n")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        api().replay_records(directory, api().Inputs())


def test_forged_identity_flag_cannot_accept_synthetic_records(tmp_path):
    """R52: a fabricated success flag cannot substitute for actual pilot evidence."""
    import json

    directory = replay_fixture(tmp_path)
    (directory / "identity.json").write_text(
        json.dumps({"exact_json_equal": True, "source_atlas_sha256": "fake"})
    )
    with pytest.raises(ValueError, match="changed input"):
        api().verify_identity(directory, {"synthetic_test_fixture": True}, api().Inputs())


def test_missing_evidence_refused_before_aggregation(tmp_path, monkeypatch):
    """R52: scientific summary cannot execute before instrument proof exists."""
    import json

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "sets": [
                    {
                        "model": "tiny",
                        "domain": "agentic",
                        "kind": "regression",
                        "instrument": {"corpus": str(tmp_path / "missing")},
                    }
                ]
            }
        )
    )

    def forbidden(*a, **k):
        pytest.fail("aggregation before evidence")

    monkeypatch.setattr(api(), "summarize_episode", forbidden)
    with pytest.raises(FileNotFoundError):
        api().build_profiles(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_cli_help_and_invalid_inputs_import_no_mlx(tmp_path):
    """§4: public CLI stays pure, including refusal paths."""
    import sys

    from local_llm_lab.spawn import run

    script = api().PROJECT_ROOT / "scripts/lens_profiles.py"
    for argv, code in [
        (["--help"], 0),
        (["--config", str(tmp_path / "missing"), "--out", str(tmp_path / "out")], 1),
    ]:
        program = f"""import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*a,**k):
    if name=='mlx' or name.startswith('mlx.') or name=='mlx_lm' or name.startswith('mlx_lm.'):
        raise AssertionError('native import forbidden')
    return original(name,*a,**k)
builtins.__import__=guarded
sys.argv={[str(script), *argv]!r}
runpy.run_path({str(script)!r},run_name='__main__')
"""
        result = run([sys.executable, "-c", program], capture_output=True, text=True)
        assert result.returncode == code, result.stderr
        assert "native import forbidden" not in result.stderr


@pytest.fixture(autouse=True)
def no_native_imports(monkeypatch):
    """All tests are pure; numerical fixtures cannot claim native scientific acceptance."""
    import builtins
    import sys

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if (
            name == "mlx"
            or name.startswith("mlx.")
            or name == "mlx_lm"
            or name.startswith("mlx_lm.")
        ):
            raise AssertionError("native import forbidden in profile tests")
        return original(name, *args, **kwargs)

    # The process may already hold mlx from other files' collection; what this fixture proves
    # is that *this test* imported none, so the assertion is on the difference, not the set.
    native = lambda name: name == "mlx" or name.startswith("mlx.")  # noqa: E731
    before = {name for name in sys.modules if native(name)}
    monkeypatch.setattr(builtins, "__import__", guarded)
    yield
    assert {name for name in sys.modules if native(name)} == before


def prose_adapter_fixture(tmp_path, monkeypatch):
    """Explicit adapter-only fixture: bypass corpus/chain readers tested separately."""
    import json

    from local_llm_lab.pipeline.lens_fitting.prose import OVERLAP, _bytes, _sha

    module = api()
    root = tmp_path / "prose"
    root.mkdir()
    (root / "dropped").mkdir()
    ids = [1] * 1024
    rows = [{"split": "held", "index": index * 5 + 4, "ids": ids} for index in range(51)]
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "domain": "prose",
                "model_hf_id": "synthetic-test",
                "tokenizer": {},
                "sequences": {},
                "sources": [],
            }
        )
    )
    monkeypatch.setattr(module, "read_corpus", lambda path: rows)
    windows = [
        {
            "label": f"test-{row['index']}",
            "kind": "prose",
            "window_index": row["index"],
            "window_sha256": _sha(_bytes(ids)),
            "prompt_ids": ids[:824],
            "prompt": "x" * 824,
            "prompt_sha256": _sha(("x" * 824).encode()),
            "prompt_tokens": 824,
            "max_tokens": 200,
        }
        for row in rows
    ]
    plan = {
        "corpus": {"path": str(corpus), "sha256": module.file_sha256(corpus)},
        "tokenizer": {},
        "sequences": {},
        "sources": [],
        "windows": windows,
        "model": "synthetic-test",
        "snapshot": {"fixture": True},
        "layers": [1, 2, 3],
        "lens_sha256": "fixture",
        "min_tokens": 32,
        "max_tokens": 200,
        "top_k": 10,
        "cache_strategy": "none",
        "sampler": {"kind": "greedy", "temperature": 0.0},
        "emitted_count_includes_eos": True,
        "overlap": OVERLAP,
        "seed": 20260902,
    }
    plan["plan_sha256"] = _sha(_bytes(plan))
    (root / "plan.json").write_text(json.dumps(plan))
    manifest = {
        key: plan[key]
        for key in (
            "model",
            "snapshot",
            "layers",
            "lens_sha256",
            "min_tokens",
            "max_tokens",
            "top_k",
            "cache_strategy",
            "sampler",
            "emitted_count_includes_eos",
            "overlap",
            "seed",
            "plan_sha256",
        )
    }
    manifest.update(status="complete", episodes=[], dropped=[], accepted_count=1, dropped_count=50)
    records = {}
    for index, window in enumerate(windows):
        dropped = index != 0
        name = ("dropped/" if dropped else "") + window["label"] + ".jsonl"
        (root / name).write_text("explicit adapter fixture; not a capture record")
        n = 31 if dropped else 32
        entry = {key: value for key, value in window.items() if key not in {"prompt", "prompt_ids"}}
        entry.update(
            record=name,
            record_sha256=module.file_sha256(root / name),
            generated_tokens=n,
            finish_reason="stop",
        )
        if dropped:
            entry["reason"] = "continuation_under_32"
        manifest["dropped" if dropped else "episodes"].append(entry)
        provenance = {key: plan[key] for key in ("model", "lens_sha256", "layers", "snapshot")}
        provenance.update(episode=window, plan_sha256=plan["plan_sha256"])
        records[str(root / name)] = [
            dict(kind="manifest", provenance=provenance),
            dict(
                kind="begin_turn",
                layers=plan["layers"],
                prompt_ids=window["prompt_ids"],
                prompt=window["prompt"],
                context={"kind": "prose"},
            ),
            *[dict(kind="emitted", token_id=1) for _ in range(n)],
        ]
    (root / "manifest.json").write_text(json.dumps(manifest))

    def adapter_record(path, inputs, *, sha=None):
        inputs.file(path, sha)
        return records[str(path)], {}

    monkeypatch.setattr(module, "_record", adapter_record)

    class ProseTokenizer:
        eos_token_ids = [1]

        def decode(self, tokens):
            return "x" * len(tokens)

        def __call__(self, prompt, **kwargs):
            return {
                "input_ids": [1] * len(prompt),
                "offset_mapping": [(i, i + 1) for i in range(len(prompt))],
            }

    return root, ProseTokenizer(), manifest


def test_prose_adapter_only_accepts_root_episodes(tmp_path, monkeypatch):
    """§14: all 51 outcomes are checked; dropped continuations never reach profiles."""
    root, tokenizer, _ = prose_adapter_fixture(tmp_path, monkeypatch)
    manifest, _, accepted = api().prose_records(root, api().Inputs(), tokenizer)
    assert len(accepted) == 1 and len(manifest["dropped"]) == 50
    assert accepted[0][0]["generated_tokens"] == 32


@pytest.mark.parametrize(
    "mutation", ["partial", "missing", "duplicate", "retention", "root_extra", "changed_plan"]
)
def test_prose_adapter_refuses_invalid_completion(tmp_path, monkeypatch, mutation):
    """§14: no partial or mislabeled held-prose set can acquire availability."""
    import json

    root, tokenizer, manifest = prose_adapter_fixture(tmp_path, monkeypatch)
    if mutation == "partial":
        manifest["status"] = "partial"
    elif mutation == "missing":
        manifest["dropped"].pop()
    elif mutation == "duplicate":
        manifest["dropped"][-1]["label"] = manifest["dropped"][0]["label"]
    elif mutation == "retention":
        manifest["episodes"][0]["generated_tokens"] = 31
    elif mutation == "root_extra":
        (root / "extra.jsonl").write_text("")
    else:
        plan = json.loads((root / "plan.json").read_text())
        plan["windows"][0]["prompt_ids"][0] = 3
        (root / "plan.json").write_text(json.dumps(plan))
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        api().prose_records(root, api().Inputs(), tokenizer)


def test_actual_original_pilot_atlas_unchanged(tmp_path, monkeypatch):
    """Legacy compatibility: actual 13 pilot records, cached tokenizer and real atlas main."""
    import json
    import runpy
    import sys

    import huggingface_hub

    module = api()
    import gzip

    control = tmp_path / "pilot-control"
    control.mkdir()
    for name in ("manifest.json", "atlas.json"):
        (control / name).write_bytes((module.PILOT / name).read_bytes())
    manifest = json.loads((control / "manifest.json").read_text())
    for episode in manifest["episodes"]:
        source = module.PILOT / episode["record"]
        if not source.exists():
            source = source.with_suffix(".jsonl.gz")
        raw = source.read_bytes()
        if source.suffix == ".gz":
            raw = gzip.decompress(raw)
        (control / (episode["label"] + ".jsonl")).write_bytes(raw)
    from huggingface_hub.errors import LocalEntryNotFoundError

    from local_llm_lab.pipeline.lens_fitting.runtime import primary_worktree

    try:
        snapshot = huggingface_hub.snapshot_download(
            manifest["model"],
            cache_dir=primary_worktree() / ".cache/huggingface/hub",
            local_files_only=True,
            allow_patterns=["tokenizer*", "*.json"],
        )
    except LocalEntryNotFoundError:
        pytest.skip("offline pilot tokenizer is not cached on this host")
    before = {path.name: module.file_sha256(path) for path in control.iterdir()}

    def offline(*a, **kwargs):
        assert kwargs.get("local_files_only") is True
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    out = tmp_path / "new-atlas.json"
    monkeypatch.setattr(
        sys, "argv", [str(module.ATLAS_SCRIPT), "--records", str(control), "--out", str(out)]
    )
    runpy.run_path(str(module.ATLAS_SCRIPT), run_name="__main__")
    assert json.loads(out.read_text()) == json.loads((control / "atlas.json").read_text())
    assert before == {path.name: module.file_sha256(path) for path in control.iterdir()}


def instrument_fixture(tmp_path, monkeypatch):
    """Tiny frozen numerical proof fixture; cannot pass canonical pilot identity gate."""
    import json

    from local_llm_lab.pipeline.lens_fitting.jacobian import freeze_plan, make_plan
    from local_llm_lab.pipeline.lens_fitting.runtime import snapshot_identity

    module = api()
    snapshot_dir = tmp_path / "synthetic-snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text(
        json.dumps({"hidden_size": 2, "num_hidden_layers": 4})
    )
    (snapshot_dir / "model.safetensors").write_bytes(
        b"explicit synthetic fixture; not model weights"
    )
    snapshot = snapshot_identity(snapshot_dir, hf_id=str(snapshot_dir))
    rows = [
        {
            "index": i,
            "source": "synthetic-fixture",
            "split": split,
            "ids": [1, 2],
            "spans": ["chat", "chat"],
        }
        for i, split in enumerate(["fit", "held"])
    ]
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text("{}")
    monkeypatch.setattr(module, "read_corpus", lambda path: rows)
    plan = make_plan(
        rows,
        layers=[1, 2, 3],
        hidden_size=2,
        corpus_sha256=module.file_sha256(corpus_path),
        snapshot_sha256=snapshot["snapshot_sha256"],
        seed=1,
        self_bounds={"atol": 0.1, "rtol": 0.01},
        response_bounds={"atol": 0.2, "rtol": 0.02},
        stability_bounds={"atol": 0.3, "rtol": 0.03},
        held_count=150,
        working_set_bytes=1000000,
        initial_peak_bytes=1,
    )
    plan_path = tmp_path / "plan.json"
    plan = freeze_plan(plan_path, plan)
    proof = {
        "plan_sha256": plan["plan_sha256"],
        "directions": 16,
        "batch_size": 8,
        "direction_seed": plan["seeds"]["self_directions"],
        "checks": [
            {
                "layer": layer,
                "mode": mode,
                "outcome": "pass",
                "max_error": 0.0,
                **plan["self_bounds"],
                "epsilon": 0.01,
                "candidate_rows": 16,
                "actual_batch_sizes": [1] if mode == "restore" else [8],
                "reference_stability": {
                    "outcome": "pass",
                    "max_error": 0.0,
                    **plan["stability_bounds"],
                },
            }
            for layer in plan["self_layers"]
            for mode in ["restore", "broadcast"]
        ],
    }
    proof_path, runtime_path = tmp_path / "self-check.json", tmp_path / "runtime.json"
    proof_path.write_text(json.dumps(proof))
    runtime_path.write_text(json.dumps({"snapshot": snapshot, "model_run_lock": "fixture-only"}))
    return (
        {
            "corpus": str(corpus_path),
            "plan": str(plan_path),
            "proof": str(proof_path),
            "runtime": str(runtime_path),
        },
        plan,
        snapshot,
    )


def test_source_bound_instrument_plan_and_numbers(tmp_path, monkeypatch):
    """R52: correct bound fixture passes only the pure instrument seam, never native acceptance."""
    config, plan, snapshot = instrument_fixture(tmp_path, monkeypatch)
    actual_plan, actual_snapshot = api().verify_instrument(config, api().Inputs())
    assert (actual_plan, actual_snapshot) == (plan, snapshot)


@pytest.mark.parametrize("mutation", ["plan", "proof", "bounds", "snapshot", "epsilon", "seed"])
def test_instrument_drift_is_refused(tmp_path, monkeypatch, mutation):
    """R52: source-bound plan, numeric evidence and physical snapshot bytes must agree."""
    import json
    from pathlib import Path

    config, _, snapshot = instrument_fixture(tmp_path, monkeypatch)
    if mutation == "snapshot":
        (Path(snapshot["snapshot_path"]) / "model.safetensors").write_bytes(b"changed fixture")
    else:
        path = Path(config["plan" if mutation == "plan" else "proof"])
        value = json.loads(path.read_text())
        if mutation == "plan":
            value["hidden_size"] = 3
        elif mutation == "proof":
            value["checks"].pop()
        elif mutation == "bounds":
            value["checks"][0]["atol"] = 99
        elif mutation == "epsilon":
            value["checks"][0]["epsilon"] = 0
        else:
            value["direction_seed"] += 1
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        api().verify_instrument(config, api().Inputs())


def test_lens_sidecar_and_snapshot_are_bound_without_npz_loading(tmp_path, monkeypatch):
    """§4: hashing a fitted artifact suffices for summaries; matrices are never loaded."""
    import json

    config, plan, snapshot = instrument_fixture(tmp_path, monkeypatch)
    lens = tmp_path / "fixture.npz"
    lens.write_bytes(b"explicit hash-only fixture; not scientific matrices")
    metadata = {
        "kind": "regression",
        "domain": "agentic",
        "model": snapshot | {"name": "tiny"},
        "npz_sha256": api().file_sha256(lens),
        "layers": [1, 2, 3],
        "num_layers": 4,
        "hidden_size": 2,
    }
    lens.with_suffix(".json").write_text(json.dumps(metadata))
    item = {"kind": "regression", "domain": "agentic", "lens": str(lens)}
    assert api().lens_metadata(item, snapshot, plan, api().Inputs())[0] == metadata
    item["domain"] = "prose"
    with pytest.raises(ValueError, match="fitted lens"):
        api().lens_metadata(item, snapshot, plan, api().Inputs())
    item["domain"] = "agentic"
    lens.write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="changed input"):
        api().lens_metadata(item, snapshot, plan, api().Inputs())


def test_prompt_span_is_source_not_next_target():
    """§3.5: a boundary target remains agreement under the source-position span."""

    class BoundaryTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2], "offset_mapping": [(0, 1), (1, 2)]}

    data = [
        {
            "kind": "begin_turn",
            "turn": 0,
            "prompt": "AB",
            "prompt_ids": [1, 2],
            "layers": [1, 2],
            "context": {
                "messages": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                ]
            },
        },
        {"kind": "reading", "turn": 0, "position": 0, "top": {"1": [2], "2": [2]}},
    ]
    rows = api().summarize_episode(
        data,
        {"label": "boundary", "kind": "agentic"},
        BoundaryTokenizer(),
        layers=[1, 2],
        lens_kind="regression",
        validation={},
    )
    row = next(r for r in rows if r["summary"] == "prompt_agreement")
    assert row["span"] == "task" and row["n"] == row["numerator"] == 1
    assert row["span_role"] == "source_prompt_position"


def test_profile_output_is_exclusive_with_explicit_missing_cells(tmp_path, monkeypatch):
    """§3.5: fixture-only output seam writes named JSON and a completion hash set."""
    import json

    module = api()
    item = {
        "model": "qwen35-4b",
        "domain": "agentic",
        "kind": "regression",
        "instrument": {},
        "identity": "synthetic-fixture",
        "lens": "synthetic-fixture",
        "records": [],
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"sets": [item]}))
    from local_llm_lab.models import load_model_spec

    snapshot = {"hf_id": load_model_spec(item["model"]).hf_id, "snapshot_path": str(tmp_path)}
    monkeypatch.setattr(module, "verify_instrument", lambda *a: ({"layers": [1, 2]}, snapshot))
    monkeypatch.setattr(module, "verify_identity", lambda *a: {"synthetic_fixture_only": True})
    monkeypatch.setattr(module, "lens_metadata", lambda *a: ({"synthetic_fixture_only": True}, {}))
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    output = tmp_path / "out"
    module.build_profiles(config, output)
    complete = json.loads((output / "complete.json").read_text())
    assert set(complete["outputs"]) == {
        "qwen35-4b-agentic-regression.json",
        "profiles.html",
        "specificity.json",
    }
    assert all(
        module.file_sha256(output / name) == sha for name, sha in complete["outputs"].items()
    )
    assert all(
        cell["status"] == "unavailable"
        for cell in json.loads((output / "specificity.json").read_text())["cells"]
    )
    before = {path.name: module.file_sha256(path) for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        module.build_profiles(config, output)
    assert before == {path.name: module.file_sha256(path) for path in output.iterdir()}


def test_html_javascript_syntax_and_filter_behavior():
    """§3.5 page: the actual embedded script selects one episode and reacts to changes."""
    import json
    import shutil

    from local_llm_lab.spawn import run

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed for the pure DOM fixture")
    one = summarize(label="one")
    two = summarize(label="two")
    data = {"sets": [{"name": "synthetic-test", "rows": one + two}], "specificity": []}
    page = api().render_page(data)
    script = page.split("<script>")[1].split("</script>")[0]
    # Minimal standard-DOM fixture exercises the exported page's real script, not a copied filter.
    harness = """const assert=require('node:assert/strict');
class Element {
 constructor(tag){this.tag=tag;this.children=[];this._value='';this.textContent='';}
 append(...items){this.children.push(...items);if(this.tag==='select'&&!this._value&&items.length)this._value=items[0].value;}
 replaceChildren(){this.children=[];this._value='';}
 setAttribute(k,v){this[k]=v;}
 get value(){return this._value;}
 set value(v){this._value=String(v);}
}
const elements={};global.document={
 getElementById(id){return elements[id]??=(new Element(id));},
 createElement(tag){return new Element(tag);},
 createTextNode(t){return {textContent:t};},
 createElementNS(ns,tag){return new Element(tag);}
};
"""
    harness += 'document.getElementById("data").textContent=' + json.dumps(json.dumps(data)) + ";\n"
    harness += (
        script
        + """\nassert.equal(choices.episode.value,'one');
assert.ok(document.getElementById('status').textContent.startsWith('one:'));
choices.episode.value='two';choices.episode.onchange();
assert.ok(document.getElementById('status').textContent.startsWith('two:'));
assert.equal(document.getElementById('rows').children.length,2);
assert.ok(document.getElementById('plot').children.some(x=>x.tag==='line'&&x['stroke-dasharray']));
"""
    )
    result = run([node, "-"], input=harness, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
