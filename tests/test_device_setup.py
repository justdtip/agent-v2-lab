"""`lab-device`: the device's step zero, tested with no network and no model.

The hub is stubbed, the checkpoint is a tiny fixture in the snapshot's on-disk shape, the dataset
is a tmp directory with the manifest the train stage requires, and the token never leaves the
prompt for anything but the hub's own store, which is stubbed to prove it.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from local_llm_lab import device_setup

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


@pytest.fixture
def tiny_snapshot(tmp_path):
    """A checkpoint in the official snapshot's declared shape, tiny (plan §16.9)."""
    from transformers import AutoModelForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
        sliding_window=8, tie_word_embeddings=True,
    )  # fmt: skip
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    state = {
        "language_model." + k: v for k, v in model.state_dict().items() if k != "lm_head.weight"
    }
    state["vision_tower.patch.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    root = tmp_path / "snapshot"
    root.mkdir()
    safetensors_torch.save_file(state, root / "model.safetensors", metadata={"format": "pt"})
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma3",
                "architectures": ["Gemma3ForConditionalGeneration"],
                "text_config": json.loads(config.to_json_string()),
                "vision_config": {"model_type": "siglip_vision_model"},
            }
        )
    )
    return root


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data" / "agent_tiny"
    root.mkdir(parents=True)
    (root / "train.jsonl").write_text('{"text": "a"}\n{"text": "b"}\n')
    (root / "valid.jsonl").write_text('{"text": "c"}\n')
    (root / "manifest.json").write_text(json.dumps({"rows": 3, "splits": ["train", "valid"]}))
    return root


def test_preflight_reports_every_row_with_a_basis_and_fails_on_a_missing_checkpoint(
    tmp_path, monkeypatch, tiny_snapshot, dataset, capsys
):
    monkeypatch.setenv("LLL_BACKEND", "torch")
    monkeypatch.setenv("LLL_DEVICE", "cpu")
    monkeypatch.setattr(device_setup, "_torch_models", lambda: [("tiny", "org/tiny")])
    monkeypatch.setattr(device_setup, "_snapshot_dir", lambda repo_id: tiny_snapshot)
    report = tmp_path / "preflight.json"
    code = device_setup.main(
        ["preflight", "--allow-cpu", "--json", str(report), "--data", str(dataset)]
    )
    out = capsys.readouterr().out
    data = json.loads(report.read_text())
    checks = {row["check"]: row for row in data["rows"]}
    assert all("basis" in row for row in data["rows"])
    assert checks["cuda"]["ok"] is True and "allowed" in checks["cuda"]["detail"]
    assert checks["determinism pinned, workspace set before first CUDA use"]["ok"] is True
    assert (
        checks["weights tiny"]["ok"] is True and "format=['pt']" in checks["weights tiny"]["detail"]
    )
    assert checks[f"data {dataset.name}"]["ok"] is True
    assert checks["box window"]["check"] == "box window"
    assert data["device"]["determinism"] == "pinned"
    # On this laptop MLX is installed and peft may not be; the report says so and fails honestly.
    assert (code == 0) == data["ok"]
    assert "preflight:" in out

    monkeypatch.setattr(device_setup, "_snapshot_dir", lambda repo_id: None)
    code = device_setup.main(["preflight", "--allow-cpu", "--json", str(report)])
    data = json.loads(report.read_text())
    weights = next(r for r in data["rows"] if r["check"] == "weights tiny")
    assert code == 1 and weights["ok"] is False and "lab-device fetch" in weights["detail"]


def test_preflight_without_allow_cpu_fails_the_cuda_row_on_a_box_without_cuda(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LLL_BACKEND", "torch")
    monkeypatch.setenv("LLL_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_setup, "_torch_models", lambda: [])
    report = tmp_path / "p.json"
    assert device_setup.main(["preflight", "--json", str(report)]) == 1
    cuda = next(r for r in json.loads(report.read_text())["rows"] if r["check"] == "cuda")
    assert cuda["ok"] is False


def test_login_hands_the_token_to_the_hub_store_and_keeps_nothing(monkeypatch, capsys):
    calls = {}
    hub = types.SimpleNamespace(
        login=lambda token, add_to_git_credential: calls.update(
            token=token, git=add_to_git_credential
        ),
        HfApi=lambda: types.SimpleNamespace(whoami=lambda: {"name": "daniel"}),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("hf_secret_token\n"))
    assert device_setup.main(["login", "--stdin"]) == 0
    assert calls == {"token": "hf_secret_token", "git": False}
    out = capsys.readouterr().out
    assert "logged in as daniel" in out and "hf_secret_token" not in out
    assert "do not copy" in out


def test_login_refuses_without_a_terminal_unless_told_to_read_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    assert device_setup.main(["login"]) == 2
    assert "needs a terminal" in capsys.readouterr().err


def test_fetch_dry_run_gives_the_verdict_from_the_mode_arithmetic(monkeypatch, capsys):
    sizes = {"org/small": (4.3e9, 8.6e9), "org/big": (27e9, 54e9), "org/unknown": (None, None)}
    monkeypatch.setattr(device_setup, "_hub_size", lambda repo_id: sizes[repo_id])
    code = device_setup.main(
        ["fetch", "--dry-run", "--budget-gib", "48", "--mode", "inference", *sizes]
    )
    out = capsys.readouterr().out
    assert code == 0
    lines = {
        line.split()[1]: line for line in out.splitlines() if line.startswith(("FETCH", "skip"))
    }
    assert lines["org/small"].startswith("FETCH") and "full=no" in lines["org/small"]
    assert lines["org/big"].startswith("skip") and "inference=no" in lines["org/big"]
    assert lines["org/unknown"].startswith("skip") and "inference=?" in lines["org/unknown"]
    # 4.3B at 16 bytes is 68.8 GB against a 48 GiB budget: full fine-tuning does not fit.
    assert device_setup.feasibility("x", 48 * device_setup.GIB, 4.3e9) == {
        "inference": True, "lora": True, "full": False,
    }  # fmt: skip


def test_pack_and_verify_round_trip_and_a_corrupted_archive_is_refused(tmp_path, dataset, capsys):
    import tarfile

    archive = tmp_path / "out" / "agent_tiny.tar.gz"
    assert device_setup.main(["pack-data", str(dataset), "--out", str(archive)]) == 0
    dest = tmp_path / "device"
    assert device_setup.main(["verify-data", str(archive), "--dest", str(dest)]) == 0
    assert (dest / "agent_tiny" / "manifest.json").exists()
    assert not (dest / "agent_tiny" / "SHA256SUMS.json").exists()
    assert "digests match" in capsys.readouterr().out

    # A row altered in transit, packed under the original digests: refused by digest, before
    # the manifest gate, so a corrupted dataset never reaches the train stage as the one packed.
    tampered = tmp_path / "tampered"
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(tampered, filter="data")
    (tampered / "agent_tiny" / "train.jsonl").write_text('{"text": "not what was packed"}\n')
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        tar.add(tampered / "agent_tiny", arcname="agent_tiny")
    assert device_setup.main(["verify-data", str(bad), "--dest", str(tmp_path / "device2")]) == 1
    assert "digest mismatch" in capsys.readouterr().err


def test_pack_refuses_a_dataset_without_a_manifest(tmp_path):
    root = tmp_path / "data" / "unstamped"
    root.mkdir(parents=True)
    (root / "train.jsonl").write_text("{}\n")
    with pytest.raises(Exception, match="manifest"):
        device_setup.main(["pack-data", str(root)])


def _blob_sha1(data: bytes) -> str:
    import hashlib

    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def test_fetch_dictionary_names_files_and_bytes_and_refuses_an_absent_layer(
    monkeypatch, capsys, tmp_path
):
    import hashlib

    folder = "resid_post_all/layer_17_width_16k_l0_small"
    config = json.dumps(  # the real config's keys
        {
            "hf_hook_point_in": "model.layers.17.output",
            "model_name": "google/x",
            "l0": 20,
            "width": 16384,
        }
    ).encode()
    params = b"\x00" * 4096
    listing = {
        f"{folder}/config.json": (248, {"blob_id": _blob_sha1(config), "lfs": None}),
        f"{folder}/params.safetensors": (
            335_700_000,
            {"blob_id": "pointer", "lfs": {"sha256": hashlib.sha256(params).hexdigest()}},
        ),
        f"{folder}/examples.safetensors": (900_000_000, {"blob_id": "p", "lfs": {"sha256": "0"}}),
    }
    info = types.SimpleNamespace(
        siblings=[types.SimpleNamespace(rfilename=k, size=v[0], **v[1]) for k, v in listing.items()]
    )
    root = tmp_path / "snap"
    (root / folder).mkdir(parents=True)
    (root / folder / "config.json").write_bytes(config)
    (root / folder / "params.safetensors").write_bytes(params)
    fetched = {}

    def hf_hub_download(repo, path, local_files_only):
        if not (root / path).is_file():
            raise FileNotFoundError(path)
        return str(root / path)

    hub = types.SimpleNamespace(
        HfApi=lambda: types.SimpleNamespace(model_info=lambda repo, files_metadata: info),
        snapshot_download=lambda repo, allow_patterns: (
            fetched.update(patterns=allow_patterns) or str(root)
        ),
        hf_hub_download=hf_hub_download,
    )
    # The sidecar lands under this cache, never under the real one.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert device_setup.main(["fetch-dictionary", "google/x", "--layer", "17", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "params.safetensors" in out and "examples" not in out and "0.31 GiB" in out
    assert device_setup.main(["fetch-dictionary", "google/x", "--layer", "17"]) == 0
    assert fetched["patterns"] == [
        "resid_post_all/layer_17_width_16k_l0_small/config.json",
        "resid_post_all/layer_17_width_16k_l0_small/params.safetensors",
    ]
    out = capsys.readouterr().out
    assert "model='google/x' hook='model.layers.17.output'" in out and out.count("verified") == 2
    assert device_setup.main(["fetch-dictionary", "google/x", "--layer", "18"]) == 2
    assert "no such files" in capsys.readouterr().err
    assert device_setup.main(["fetch-dictionary", "google/x"]) == 2
    assert "--all-layers" in capsys.readouterr().err
    # Every layer the listing holds, in order, read from the listing rather than assumed.
    info.siblings.extend(
        types.SimpleNamespace(
            rfilename=f"resid_post_all/layer_{n}_width_16k_l0_small/{name}",
            size=1,
            blob_id="b",
            lfs=None,
        )
        for n in (3, 0)
        for name in ("config.json", "params.safetensors")
    )
    assert device_setup.main(["fetch-dictionary", "google/x", "--all-layers", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "3 layers listed: 0-17" in out and "6 files" in out
    # The trap: a file of the right name and length whose bytes are another suite's.
    (root / folder / "params.safetensors").write_bytes(b"\x01" + b"\x00" * 4095)
    assert device_setup.main(["fetch-dictionary", "google/x", "--layer", "17"]) == 1
    assert "!= declared" in capsys.readouterr().err


def test_fetch_dictionary_records_digests_and_the_offline_check_reads_them(
    monkeypatch, capsys, tmp_path
):
    import hashlib

    folder = "resid_post_all/layer_17_width_16k_l0_small"
    config = json.dumps(
        {"hf_hook_point_in": "model.layers.17.output", "model_name": "google/x"}
    ).encode()
    params = b"\x00" * 4096
    root = tmp_path / "snap"
    (root / folder).mkdir(parents=True)
    (root / folder / "config.json").write_bytes(config)
    (root / folder / "params.safetensors").write_bytes(params)
    info = types.SimpleNamespace(
        siblings=[
            types.SimpleNamespace(
                rfilename=f"{folder}/config.json", size=248, blob_id=_blob_sha1(config), lfs=None
            ),
            types.SimpleNamespace(
                rfilename=f"{folder}/params.safetensors",
                size=4096,
                blob_id="p",
                lfs={"sha256": hashlib.sha256(params).hexdigest()},
            ),
        ]
    )

    def hf_hub_download(repo, path, local_files_only):
        if not (root / path).is_file():
            raise FileNotFoundError(path)
        return str(root / path)

    hub = types.SimpleNamespace(
        HfApi=lambda: types.SimpleNamespace(model_info=lambda repo, files_metadata: info),
        snapshot_download=lambda repo, allow_patterns: str(root),
        hf_hub_download=hf_hub_download,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    assert device_setup.main(["fetch-dictionary", "google/x", "--layer", "17"]) == 0
    sidecar = device_setup.dictionary_sidecar(tmp_path / "cache", "google/x", folder)
    recorded = json.loads(sidecar.read_text())
    assert set(recorded["files"]) == {"config.json", "params.safetensors"}
    assert recorded["files"]["params.safetensors"]["algorithm"] == "sha256"
    assert recorded["config"]["model_name"] == "google/x"
    capsys.readouterr()
    # Offline: the hub is never asked; the recorded digests are the comparison.
    hub.HfApi = lambda: (_ for _ in ()).throw(AssertionError("hub asked offline"))
    assert device_setup.main(["fetch-dictionary", "google/x", "--layer", "17", "--offline"]) == 0
    assert capsys.readouterr().out.count("verified") == 2
    result = device_setup.verify_cached_dictionary("google/x", folder, offline=True)
    assert result["basis"] == "recorded-at-fetch"
    assert all(f["ok"] for f in result["files"].values())
    (root / folder / "params.safetensors").write_bytes(b"\x01" + b"\x00" * 4095)
    assert device_setup.main(["fetch-dictionary", "google/x", "--all-layers", "--offline"]) == 1
    assert "MISMATCH" in capsys.readouterr().out
    # Without a sidecar and without the hub, undecided is the only honest reading.
    sidecar.unlink()
    result = device_setup.verify_cached_dictionary("google/x", folder, offline=True)
    assert result["basis"] is None and result["files"]["params.safetensors"]["ok"] is None


# ------------------------------------------------------ dictionary rows and the render row


def _dictionary_fixture(root, model_name="google/gemma-3-4b-it"):
    """A dictionary folder in the cache's shape: params bytes and the config that names it."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "params.safetensors").write_bytes(b"\x00" * 64 + b"dictionary-bytes")
    (root / "config.json").write_text(
        json.dumps(
            {
                "hf_hook_point_in": "model.layers.17.output",
                "width": 16384,
                "model_name": model_name,
                "architecture": "jump_relu",
                "l0": 20,
            }
        )
    )
    return root


def _verified(fixture, folder, *, ok, basis="hub-declared", present=True, note=None):
    """What verify_cached_dictionary returns, in its own shape, for one stubbed folder."""
    files = {}
    for name in device_setup.DICTIONARY_FILES:
        path = fixture / name
        entry = {"present": present, "path": str(path) if present else None}
        if present and ok is None:
            entry.update(ok=None, note="undecided: no hub answer and no digest recorded at fetch")
        elif present:
            actual = device_setup._file_digest(path, "sha256")
            entry.update(
                algorithm="sha256", expected=actual if ok else "0" * 64, actual=actual, ok=ok
            )
        files[name] = entry
    return {"repo": "google/scope", "folder": folder, "basis": basis, "note": note, "files": files}


FOLDER = "resid_post_all/layer_17_width_16k_l0_small"
ENTRY, BASE = "gemma3-4b-cuda-bf16", "google/gemma-3-4b-it"


def _stub_verify(monkeypatch, fixture, **kw):
    monkeypatch.setattr(
        device_setup,
        "verify_cached_dictionary",
        lambda repo, folder, *, offline: _verified(fixture, folder, **kw),
    )


def test_dictionary_rows_read_the_verification_and_check_the_model(monkeypatch, tmp_path):
    fixture = _dictionary_fixture(tmp_path / "dict")
    monkeypatch.setattr(device_setup, "configure_local_cache", lambda: tmp_path / "cache")
    _stub_verify(monkeypatch, fixture, ok=True)
    rows = device_setup.dictionary_rows(ENTRY, BASE, [("google/scope", FOLDER)], offline=False)
    checks = {r["check"].split(" dictionary ")[1]: r for r in rows}
    assert checks[f"{FOLDER} present"]["ok"] is True
    assert checks[f"{FOLDER} digest"]["ok"] is True
    assert checks[f"{FOLDER} digest"]["basis"] == "hub-declared"
    assert checks[f"{FOLDER} model"]["ok"] is True
    assert checks[f"{FOLDER} model"]["basis"] == "registry"

    # The verification found different bytes: the digest row fails with both digests shown.
    _stub_verify(monkeypatch, fixture, ok=False)
    rows = device_setup.dictionary_rows(ENTRY, BASE, [("google/scope", FOLDER)], offline=False)
    digest = [r for r in rows if r["check"].endswith("digest")][0]
    assert digest["ok"] is False and "!=" in digest["detail"]

    # Undecided stays undecided, and the verification's note travels with it.
    _stub_verify(monkeypatch, fixture, ok=None, basis=None, note="hub not reachable: no route")
    rows = device_setup.dictionary_rows(ENTRY, BASE, [("google/scope", FOLDER)], offline=True)
    digest = [r for r in rows if r["check"].endswith("digest")][0]
    assert digest["ok"] is None and "undecided" in digest["detail"]
    assert "hub not reachable" in digest["detail"]


def test_dictionary_of_another_checkpoint_fails_the_model_row(monkeypatch, tmp_path):
    fixture = _dictionary_fixture(tmp_path / "dict", model_name="google/gemma-3-4b-pt")
    _stub_verify(monkeypatch, fixture, ok=True)
    rows = device_setup.dictionary_rows(ENTRY, BASE, [("google/scope", FOLDER)], offline=False)
    model = [r for r in rows if r["check"].endswith("model")][0]
    assert model["ok"] is False
    assert "gemma-3-4b-pt" in model["detail"] and "gemma-3-4b-it" in model["detail"]


def test_a_missing_dictionary_names_the_fetch_command_with_its_layer(monkeypatch, tmp_path):
    _stub_verify(monkeypatch, tmp_path, ok=None, present=False)
    rows = device_setup.dictionary_rows(ENTRY, BASE, [("google/scope", FOLDER)], offline=True)
    assert len(rows) == 1 and rows[0]["ok"] is False
    assert (
        "lab-device fetch-dictionary google/scope --layer 17 --site resid_post_all "
        "--width 16k --l0 small" in rows[0]["detail"]
    )
    assert "not the suite's layout" in device_setup._fetch_command("google/scope", "odd/folder")


def test_entry_dictionaries_are_read_from_the_registry_file(monkeypatch, tmp_path):
    from local_llm_lab import models

    (tmp_path / "e.yaml").write_text(
        "name: e\ndictionaries:\n  - repo: google/scope\n    folder: resid_post_all/layer_1_w\n"
    )
    (tmp_path / "bad.yaml").write_text("name: bad\ndictionaries:\n  - repo: google/scope\n")
    monkeypatch.setattr(models, "_REGISTRY_DIR", tmp_path)
    assert device_setup._entry_dictionaries("e") == [("google/scope", "resid_post_all/layer_1_w")]
    assert device_setup._entry_dictionaries("absent") == []
    with pytest.raises(ValueError, match="names a repo and a folder"):
        device_setup._entry_dictionaries("bad")
    assert device_setup._parse_dictionary_arg("google/scope:site/layer_2") == (
        "google/scope",
        "site/layer_2",
    )
    with pytest.raises(ValueError, match="owner/repo:folder"):
        device_setup._parse_dictionary_arg("scope")


class _StubTokenizer:
    """Answers the one call the render makes, deterministically, ending in the entry's suffix."""

    def __init__(self, suffix, salt=""):
        self.suffix, self.salt = suffix, salt

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return body + self.salt + (self.suffix if add_generation_prompt else "")


@pytest.fixture
def corpus(tmp_path):
    """A source corpus in the render's shape: three roles of replay rows and a manifest."""
    root = tmp_path / "source"
    root.mkdir()
    for role, n in (("train", 3), ("valid", 2), ("test", 1)):
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": f"{role} {i}"},
                    {"role": "assistant", "content": f"reply {i}"},
                ],
                "metadata": {"source": "pre-expansion-policy-replay"},
            }
            for i in range(n)
        ]
        (root / f"{role}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "manifest.json").write_text(json.dumps({"seed": 1, "generator_version": 4}))
    return root


def test_render_row_reproduces_the_laptop_manifest_and_names_a_differing_split(
    monkeypatch, tmp_path, corpus
):
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.data import generation_suffix, render_dataset

    spec = load_model_spec(ENTRY)
    suffix = generation_suffix(spec)
    laptop = tmp_path / "laptop"
    recorded = render_dataset(corpus, laptop, _StubTokenizer(suffix), spec=spec)
    assert set(recorded["outputs"]) == {"train", "valid", "test"}

    monkeypatch.setattr(device_setup, "_render_tokenizer", lambda hf_id: _StubTokenizer(suffix))
    row = device_setup.render_row(ENTRY, corpus, laptop / "manifest.json")
    assert row["ok"] is True and "train=same" in row["detail"]

    # A tokenizer that renders differently: the same corpus, a different digest, named by split.
    monkeypatch.setattr(
        device_setup, "_render_tokenizer", lambda hf_id: _StubTokenizer(suffix, salt="x")
    )
    row = device_setup.render_row(ENTRY, corpus, laptop / "manifest.json")
    assert row["ok"] is False and "train=DIFFERENT" in row["detail"]

    # A different source corpus is refused before any render, as the wrong comparison.
    (corpus / "valid.jsonl").write_text((corpus / "valid.jsonl").read_text() + "\n")
    monkeypatch.setattr(
        device_setup,
        "_render_tokenizer",
        lambda hf_id: (_ for _ in ()).throw(AssertionError("no render on a different source")),
    )
    row = device_setup.render_row(ENTRY, corpus, laptop / "manifest.json")
    assert row["ok"] is False and "['valid']" in row["detail"]


def test_preflight_carries_the_dictionary_and_render_rows_with_a_basis(
    monkeypatch, tmp_path, tiny_snapshot, corpus, capsys
):
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.data import generation_suffix, render_dataset

    monkeypatch.setenv("LLL_BACKEND", "torch")
    monkeypatch.setenv("LLL_DEVICE", "cpu")
    monkeypatch.setattr(device_setup, "_torch_models", lambda: [(ENTRY, BASE)])
    monkeypatch.setattr(device_setup, "_snapshot_dir", lambda repo_id: tiny_snapshot)
    _stub_verify(monkeypatch, _dictionary_fixture(tmp_path / "dict"), ok=True)
    spec = load_model_spec(ENTRY)
    suffix = generation_suffix(spec)
    laptop = tmp_path / "laptop"
    render_dataset(corpus, laptop, _StubTokenizer(suffix), spec=spec)
    monkeypatch.setattr(device_setup, "_render_tokenizer", lambda hf_id: _StubTokenizer(suffix))
    report = tmp_path / "report.json"
    code = device_setup.main(
        [
            "preflight", "--allow-cpu", "--json", str(report),
            "--model", ENTRY,
            "--dictionary", f"google/scope:{FOLDER}",
            "--render-source", str(corpus), "--render-manifest", str(laptop / "manifest.json"),
        ]
    )  # fmt: skip
    rows = {r["check"]: r for r in json.loads(report.read_text())["rows"]}
    # The exit code reports the laptop's own environment rows too (MLX present, peft absent);
    # what this test owns is every row the dictionary and render checks added.
    assert code in (0, 1)
    added = [r for k, r in rows.items() if " dictionary " in k or k.startswith("render ")]
    assert len(added) == 4 and all(r["ok"] is True for r in added), added
    assert rows[f"{ENTRY} dictionary {FOLDER} digest"]["basis"] == "hub-declared"
    assert rows[f"{ENTRY} dictionary {FOLDER} model"]["basis"] == "registry"
    assert all(r["basis"] for r in rows.values())
    assert f"render {ENTRY}" in capsys.readouterr().out


# ---------------------------------------------------------------------------------- bootstrap


def _bootstrap_stubs(monkeypatch, tmp_path, *, free_gib=250.0, logged_in="daniel", dictionary_ok=0):
    import shutil

    calls = []
    cache = tmp_path / "cache"
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setattr(device_setup, "_logged_in", lambda: logged_in)
    monkeypatch.setattr(device_setup, "login", lambda args: calls.append(("login",)) or 0)
    monkeypatch.setattr(device_setup, "_hub_size", lambda repo: (4e9, 8 * device_setup.GIB))
    monkeypatch.setattr(
        device_setup, "_dictionary_plan", lambda repo, site, width, l0: (34, 11 * device_setup.GIB)
    )
    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: types.SimpleNamespace(free=free_gib * device_setup.GIB)
    )
    monkeypatch.setattr(
        device_setup, "fetch", lambda args: calls.append(("fetch", tuple(args.ids), args.all)) or 0
    )

    def fetch_dictionary(args):
        calls.append(("dictionary", args.repo, args.all_layers))
        for model in ("google/gemma-3-4b-it", "google/gemma-3-12b-it"):
            if model.split("-")[2] in args.repo:  # "4b" / "12b"
                folder = f"{args.site}/layer_17_width_{args.width}_l0_{args.l0}"
                side = device_setup.dictionary_sidecar(cache, args.repo, folder)
                side.parent.mkdir(parents=True, exist_ok=True)
                side.write_text(json.dumps({"config": {"model_name": model}, "files": {}}))
        return dictionary_ok

    monkeypatch.setattr(device_setup, "fetch_dictionary", fetch_dictionary)
    monkeypatch.setattr(
        device_setup, "verify_data", lambda args: calls.append(("data", args.archive)) or 0
    )
    monkeypatch.setattr(
        device_setup,
        "_entries_by_base",
        lambda: {
            "google/gemma-3-4b-it": "gemma3-4b-cuda-bf16",
            "google/gemma-3-12b-it": "gemma3-12b-cuda-bf16",
        },
    )
    monkeypatch.setattr(
        device_setup,
        "preflight",
        lambda args: (
            calls.append(
                ("preflight", args.model[0], tuple(args.dictionary or ()), args.render_source)
            )
            or 0
        ),
    )
    return calls, cache


def test_bootstrap_dry_run_logs_in_plans_the_disk_and_downloads_nothing(
    monkeypatch, tmp_path, capsys
):
    calls, _ = _bootstrap_stubs(monkeypatch, tmp_path)
    report = tmp_path / "report.json"
    code = device_setup.main(["bootstrap", "--dry-run", "--report", str(report)])
    out = capsys.readouterr().out
    assert code == 0 and calls == [], "a dry run fetches nothing"
    assert "already logged in as daniel" in out and "GiB to fetch plus 40 GiB reserve" in out
    rows = json.loads(report.read_text())["rows"]
    assert [r["step"] for r in rows] == ["login", "plan", "dry-run"]
    assert "8.00 GiB  google/gemma-3-4b-it" in out and "(34 layers)" in out


def test_bootstrap_refuses_a_disk_the_plan_does_not_fit_with_the_numbers(
    monkeypatch, tmp_path, capsys
):
    _bootstrap_stubs(monkeypatch, tmp_path, free_gib=50.0)
    code = device_setup.main(["bootstrap", "--report", str(tmp_path / "r.json")])
    assert code == 1
    out = capsys.readouterr().out
    assert (
        "FAIL plan" in out and "38.0 GiB to fetch plus 40 GiB reserve, against 50.0 GiB free" in out
    )


def test_bootstrap_runs_every_step_in_order_and_pairs_each_dictionary_with_its_entry(
    monkeypatch, tmp_path, capsys
):
    calls, _ = _bootstrap_stubs(monkeypatch, tmp_path)
    source = tmp_path / "data" / "agent_v2e"
    source.mkdir(parents=True)
    rendered = tmp_path / "data" / "agent_v2e-gemma3-4b"
    rendered.mkdir()
    (rendered / "manifest.json").write_text("{}")
    report = tmp_path / "report.json"
    code = device_setup.main([
        "bootstrap", "--report", str(report), "--allow-cpu",
        "--data-archive", "x/agent_v2e.tar.gz", "--data-archive", "x/agent_v2e-gemma3-4b.tar.gz",
        "--data-dest", str(tmp_path / "data"),
        "--render-source", str(source), "--render-manifest", str(rendered / "manifest.json"),
    ])  # fmt: skip
    assert code == 0, capsys.readouterr().out
    assert calls == [
        ("fetch", ("google/gemma-3-4b-it", "google/gemma-3-12b-it"), True),
        ("dictionary", "google/gemma-scope-2-4b-it", True),
        ("dictionary", "google/gemma-scope-2-12b-it", True),
        ("data", "x/agent_v2e.tar.gz"),
        ("data", "x/agent_v2e-gemma3-4b.tar.gz"),
        ("preflight", "gemma3-4b-cuda-bf16",
         ("google/gemma-scope-2-4b-it:resid_post_all/layer_17_width_16k_l0_small",), None),
        ("preflight", "gemma3-12b-cuda-bf16",
         ("google/gemma-scope-2-12b-it:resid_post_all/layer_17_width_16k_l0_small",), str(source)),
    ]  # fmt: skip
    rows = json.loads(report.read_text())
    assert rows["exit"] == 0 and [r["step"] for r in rows["rows"]][-2:] == [
        "preflight gemma3-4b-cuda-bf16",
        "preflight gemma3-12b-cuda-bf16",
    ]


def test_bootstrap_stops_at_the_first_failure_and_still_writes_the_report(monkeypatch, tmp_path):
    calls, _ = _bootstrap_stubs(monkeypatch, tmp_path, dictionary_ok=1)
    report = tmp_path / "report.json"
    code = device_setup.main(
        ["bootstrap", "--report", str(report), "--data-archive", "x/agent_v2e.tar.gz"]
    )
    assert code == 1
    assert [c[0] for c in calls] == ["fetch", "dictionary"], "nothing after the failed step"
    rows = json.loads(report.read_text())
    assert rows["exit"] == 1 and rows["rows"][-1]["ok"] is False
    assert rows["rows"][-1]["step"] == "dictionary google/gemma-scope-2-4b-it"


def test_bootstrap_without_a_token_prompts_or_fails_by_flag(monkeypatch, tmp_path):
    calls, _ = _bootstrap_stubs(monkeypatch, tmp_path, logged_in=None)
    assert (
        device_setup.main(["bootstrap", "--skip-login", "--report", str(tmp_path / "a.json")]) == 2
    )
    assert calls == []
    assert device_setup.main(["bootstrap", "--dry-run", "--report", str(tmp_path / "b.json")]) == 0
    assert calls == [("login",)], "the prompt ran once, then the plan"


def test_bootstrap_writes_its_report_when_a_step_raises(monkeypatch, tmp_path, capsys):
    calls, _ = _bootstrap_stubs(monkeypatch, tmp_path)

    def explode(args):
        raise RuntimeError("CUDA is already initialised")

    monkeypatch.setattr(device_setup, "fetch", explode)
    report = tmp_path / "report.json"
    assert device_setup.main(["bootstrap", "--report", str(report)]) == 1
    rows = json.loads(report.read_text())
    assert rows["exit"] == 1 and rows["rows"][-1]["step"] == "exception"
    assert "CUDA is already initialised" in rows["rows"][-1]["detail"]
    assert "Traceback" in capsys.readouterr().err


def test_pack_data_leaves_macos_sidecars_out_of_the_archive_and_the_digests(tmp_path, dataset):
    """AppleDouble files match a *.jsonl glob on Linux and are never data."""
    (dataset / "._train.jsonl").write_bytes(b"\x00\x05\x16\x07")
    (dataset / ".DS_Store").write_bytes(b"junk")
    out = tmp_path / "packed.tar.gz"
    assert device_setup.main(["pack-data", str(dataset), "--out", str(out)]) == 0
    import tarfile

    with tarfile.open(out) as tar:
        names = [m.name.split("/", 1)[1] for m in tar.getmembers() if "/" in m.name]
    assert "._train.jsonl" not in names and ".DS_Store" not in names
    assert "train.jsonl" in names and "SHA256SUMS.json" in names
    assert "._train.jsonl" not in device_setup._digests(dataset)
    dest = tmp_path / "dest"
    assert device_setup.main(["verify-data", str(out), "--dest", str(dest)]) == 0
    assert not list((dest / dataset.name).glob("._*"))
