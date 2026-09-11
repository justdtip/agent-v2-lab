"""Frozen workspace adapter: token identity and episode exclusions, without a model."""

import hashlib
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus
from local_llm_lab.pipeline.lens_fitting.workspace_corpus import build_workspace_corpus


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    tok = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "hello": 1, "note": 2, "name": 3, "read_file": 4, "extra": 5},
            unk_token="[UNK]",
        )
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = tmp_path / "tokenizer.json"
    tok.save(str(tokenizer))
    corpus, index, cells = [], [], []
    for i, task in enumerate(
        [
            "train-read-0000-clean",
            "train-read-0001-clean",
            "train-read-0001-clean",
            "test-read-0002-clean",
        ]
    ):
        prompt, completion = "hello " * (i + 1), 'note {"name": "read_file"}'
        meta = dict(task_id=task, step=i, family="read", variant="clean", recovery=False)
        corpus.append(dict(prompt=prompt, completion=completion, metadata=meta))
        p, c = tok.encode(prompt), tok.encode(completion)
        target = next(
            j for j, (a, b) in enumerate(c.offsets) if a <= completion.index("read_file") < b
        )
        ix = dict(
            i=i,
            **meta,
            P_note=len(p.ids) - 1,
            P_act=len(p.ids) + target - 1,
            n_prompt_tokens=len(p.ids),
            n_note_tokens=target,
        )
        index.append(ix)
        if i == 0:
            for pos in ("P_note", "P_act"):
                cells.append(
                    dict(
                        row=i,
                        position=pos,
                        token_index=ix[pos],
                        token_id=c.ids[0 if pos == "P_note" else target],
                        context_tokens=ix["P_act"] + 2,
                    )
                )
    cp, ip, pp = [tmp_path / x for x in ("corpus.jsonl", "index.jsonl", "positions.json")]
    cp.write_text("".join(json.dumps(x) + "\n" for x in corpus))
    ip.write_text("".join(json.dumps(x) + "\n" for x in index))
    pp.write_text(
        json.dumps(
            dict(
                schema_version=1,
                corpus_sha256=sha(cp),
                tokenizer_sha256=sha(tokenizer),
                capture_files_sha256={"index.jsonl": sha(ip)},
                reduction="per-position, no reduction",
                endpoint="pre-final-norm",
                cells=cells,
            )
        )
    )
    return cp, ip, pp, tokenizer


def test_exact_frozen_context_and_exclusion(inputs, tmp_path):
    m = build_workspace_corpus(*inputs, tmp_path / "out")
    rows = read_corpus(tmp_path / "out/manifest.json")
    assert len(rows) == 1
    assert rows[0]["task_id"] == "train-read-0001-clean"
    assert len(rows[0]["ids"]) == rows[0]["P_act"] + 2
    assert rows[0]["P_note"] == rows[0]["n_prompt_tokens"] - 1
    assert m["excluded_episode_ids"] == ["train-read-0000-clean"]
    assert m["counts"]["eligible_rows"] == 2
    assert m["counts"]["fit"] == 1
    assert rows[0]["split"] == "fit"
    assert len(rows[0]["offsets"]) == len(rows[0]["spans"]) == len(rows[0]["ids"])


@pytest.mark.parametrize("which", range(4))
def test_bound_source_changes_refused(inputs, tmp_path, which):
    build_workspace_corpus(*inputs, tmp_path / "out")
    inputs[which].write_bytes(inputs[which].read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash|digest"):
        read_corpus(tmp_path / "out/manifest.json")


def test_index_position_wrong_refuses_even_rehashed(inputs, tmp_path):
    cp, ip, pp, tp = inputs
    rows = [json.loads(x) for x in ip.read_text().splitlines()]
    rows[1]["P_act"] += 1
    ip.write_text("".join(json.dumps(x) + "\n" for x in rows))
    p = json.loads(pp.read_text())
    p["capture_files_sha256"]["index.jsonl"] = sha(ip)
    pp.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="P_act"):
        build_workspace_corpus(*inputs, tmp_path / "out")


def test_no_silent_truncation(inputs, tmp_path):
    with pytest.raises(ValueError, match="max_tokens"):
        build_workspace_corpus(*inputs, tmp_path / "out", max_tokens=1)


def test_rows_rehashed_mutation_still_refused(inputs, tmp_path):
    m = build_workspace_corpus(*inputs, tmp_path / "out")
    rp = tmp_path / "out/rows.jsonl"
    rows = [json.loads(x) for x in rp.read_text().splitlines()]
    rows[0]["ids"][0] += 1
    rp.write_text("".join(json.dumps(x) + "\n" for x in rows))
    m["rows"]["sha256"] = sha(rp)
    m.pop("manifest_sha256")
    payload = json.dumps(
        m, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    m["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    (tmp_path / "out/manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="reconstruct"):
        read_corpus(tmp_path / "out/manifest.json")


def test_comparison_target_refuses(inputs, tmp_path):
    pp = inputs[2]
    seal = json.loads(pp.read_text())
    seal["cells"][0]["token_id"] += 1
    pp.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="target token"):
        build_workspace_corpus(*inputs, tmp_path / "out")


def test_comparison_duplicate_refuses(inputs, tmp_path):
    pp = inputs[2]
    seal = json.loads(pp.read_text())
    seal["cells"].append(seal["cells"][0])
    pp.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="duplicate"):
        build_workspace_corpus(*inputs, tmp_path / "out")


def test_first_duplicate_wins_without_re_render(inputs, tmp_path):
    cp, ip, pp, tp = inputs
    original = cp.read_text()
    duplicate = json.loads(original.splitlines()[1])
    duplicate["prompt"] = "different later duplicate"
    cp.write_text(original + json.dumps(duplicate) + "\n")
    seal = json.loads(pp.read_text())
    seal["corpus_sha256"] = sha(cp)
    pp.write_text(json.dumps(seal))
    m = build_workspace_corpus(*inputs, tmp_path / "out")
    assert m["counts"]["capture_rows"] == 4
    assert "different" not in read_corpus(tmp_path / "out/manifest.json")[0]["prompt"]


def test_whole_episode_excluded_and_immutable_output(inputs, tmp_path):
    cp, ip, pp, tp = inputs
    raw = [json.loads(x) for x in cp.read_text().splitlines()]
    ix = [json.loads(x) for x in ip.read_text().splitlines()]
    raw[2]["metadata"]["task_id"] = raw[0]["metadata"]["task_id"]
    ix[2]["task_id"] = ix[0]["task_id"]
    cp.write_text("".join(json.dumps(x) + "\n" for x in raw))
    ip.write_text("".join(json.dumps(x) + "\n" for x in ix))
    seal = json.loads(pp.read_text())
    seal["corpus_sha256"], seal["capture_files_sha256"]["index.jsonl"] = sha(cp), sha(ip)
    pp.write_text(json.dumps(seal))
    manifest = build_workspace_corpus(*inputs, tmp_path / "out")
    assert manifest["counts"]["eligible_rows"] == 1
    with pytest.raises(FileExistsError):
        build_workspace_corpus(*inputs, tmp_path / "out")


def test_cli_workspace_no_registry_model_load(inputs, tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("corpus_cli", Path("scripts/lens_corpus.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "load_model_spec", lambda *args: pytest.fail("model registry loaded")
    )
    module.main(
        [
            "--corpus",
            "agentic",
            "--workspace-corpus",
            str(inputs[0]),
            "--capture-index",
            str(inputs[1]),
            "--positions",
            str(inputs[2]),
            "--tokenizer-json",
            str(inputs[3]),
            "--out",
            str(tmp_path / "frozen.json"),
        ]
    )
    assert len(read_corpus(tmp_path / "frozen.json")) == 1


def test_malformed_training_id_is_not_silently_excluded(inputs, tmp_path):
    cp, ip, pp, tp = inputs
    raw = [json.loads(x) for x in cp.read_text().splitlines()]
    ix = [json.loads(x) for x in ip.read_text().splitlines()]
    raw[1]["metadata"]["task_id"] = "train-madeup-0001-clean"
    ix[1]["task_id"] = "train-madeup-0001-clean"
    cp.write_text("".join(json.dumps(x) + "\n" for x in raw))
    ip.write_text("".join(json.dumps(x) + "\n" for x in ix))
    seal = json.loads(pp.read_text())
    seal["corpus_sha256"] = sha(cp)
    seal["capture_files_sha256"]["index.jsonl"] = sha(ip)
    pp.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="task ID"):
        build_workspace_corpus(*inputs, tmp_path / "out")
