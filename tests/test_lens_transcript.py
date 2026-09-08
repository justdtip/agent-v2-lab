"""Tiny source-only transcript tests: no MLX or checkpoint boundary is called."""

import json

import pytest

from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.lens_fitting import transcript as t


class Tokenizer:
    bos_token = None

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(token) for token in ids)

    def __call__(self, text, **kwargs):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


class Inputs:
    ndim = 2
    shape = (1, 1)

    def __init__(self, values):
        self.values = values

    def __getitem__(self, _):
        return self

    def tolist(self):
        return self.values


def identity(tmp_path):
    asset = tmp_path / "tokenizer.json"
    asset.write_text("{}")
    return {
        "files": [t.file_record(asset)],
        "assets": [{"name": "tokenizer.json", "sha256": t.file_record(asset)["sha256"]}],
        "chat_template_sha256": "a" * 64,
    }


def make_record(tmp_path, *, turns=5, long=False):
    spec = load_model_spec("gemma3-4b")
    ident = {"base": spec.base, "training": None, "num_layers": 34}
    tok_id = identity(tmp_path)
    path = tmp_path / "capture.jsonl"
    with t.TranscriptWriter(path, {"model_identity": ident, "tokenizer": tok_id}) as writer:
        capture = t.TranscriptCapture(writer, materialize=lambda _: None)
        for step in range(turns):
            messages = [
                {"role": "system", "content": t.system_prompt(spec=spec)},
                {"role": "user", "content": " task "},
            ]
            if step:
                messages += [
                    {"role": "assistant", "content": "old note"},
                    {"role": "tool", "name": "read_file", "content": "O" * (2500 if long else 8)},
                ]
            displayed = t._as_declared_roles(messages, spec)
            prompt = "|".join(m["content"].strip() for m in displayed) + "|"
            capture.set_context(
                task_id="train-read-0000-clean", step=step, keep_last=2, messages=messages
            )
            with capture.generation(
                lambda *_a, **_k: None, Tokenizer(), prompt, turn_cache=None
            ) as model:
                model(Inputs(Tokenizer().encode(prompt)))
                capture.emitted(ord("n"))
                model(Inputs([ord("n")]))
                capture.emitted(ord("!"))  # emitted, not consumed
    return path, spec, ident, tok_id


def test_capture_roundtrip_and_unique_long_context(tmp_path):
    source, spec, ident, tok_id = make_record(tmp_path, long=True)
    path = tmp_path / "corpus.json"
    manifest = t.build_transcript_corpus(
        [source], Tokenizer(), spec, path, tokenizer_identity=tok_id, model_identity=ident
    )
    rows = t.read_transcript_corpus(path)
    assert {r["split"] for r in rows} == {"fit", "held"}
    assert max(map(lambda r: len(r["ids"]), rows)) == 2048
    assert sum(c["positions_beyond_1024"] for c in manifest["counts"].values()) > 0
    assert sum(c["generated"] for c in manifest["counts"].values()) == 5
    assert sum(turn["emitted_unconsumed"] for turn in manifest["turns"]) == 5
    assert manifest["turns"][0]["prompt_boundary_audit"][1]["trim_left"] == 1
    assert sum(c["spans"]["task"] for c in manifest["counts"].values()) == 4
    assert all(
        not r["generated_mask"][i] or r["spans"][i] == "note"
        for r in rows
        for i in r["score_positions"]
    )


def test_hash_chain_refuses_missing_footer(tmp_path):
    source, *_ = make_record(tmp_path)
    source.write_bytes(b"\n".join(source.read_bytes().splitlines()[:-1]) + b"\n")
    with pytest.raises(ValueError, match="incomplete"):
        t.read_transcript(source)


def test_failed_forward_not_recorded(tmp_path):
    events = []
    capture = t.TranscriptCapture(events.append, materialize=lambda _: None)

    def fail(*args, **kwargs):
        raise RuntimeError("native failed")

    with (
        pytest.raises(RuntimeError),
        capture.generation(fail, Tokenizer(), "x", turn_cache=None) as model,
    ):
        model(Inputs([120]))
    assert not any(e["kind"] == "forward" for e in events)
    assert events[-1]["status"] == "aborted"


def test_lookahead_count_and_capture_rejects_reuse(tmp_path):
    events = []
    capture = t.TranscriptCapture(events.append, materialize=lambda _: None)
    with (
        pytest.raises(ValueError, match="no cache reuse"),
        capture.generation(None, Tokenizer(), "x", turn_cache=object()),
    ):
        pass
    capture.set_context(task_id="train-read-0000-clean", step=0, messages=[])
    with capture.generation(lambda *_a, **_k: None, Tokenizer(), "x", turn_cache=None) as model:
        model(Inputs([120, 121]))
    assert events[-1]["forwarded_count"] == 2 and events[-1]["emitted_count"] == 0


def test_source_score_tampering_rejected_even_with_rehashed_rows(tmp_path):
    source, spec, ident, tok_id = make_record(tmp_path)
    path = tmp_path / "corpus.json"
    t.build_transcript_corpus(
        [source], Tokenizer(), spec, path, tokenizer_identity=tok_id, model_identity=ident
    )
    manifest = json.loads(path.read_text())
    seq = tmp_path / "corpus.jsonl"
    rows = [json.loads(line) for line in seq.read_text().splitlines()]
    rows[0]["score_positions"].append(rows[0]["score_positions"][0])
    seq.write_bytes(b"".join(t.encoded(r) + b"\n" for r in rows))
    manifest["sequences"] = t.file_record(seq)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = t.digest(manifest)
    path.write_bytes(t.encoded(manifest))
    with pytest.raises(ValueError, match="window"):
        t.read_transcript_corpus(path)


def test_missing_observation_convention_refused():
    spec = load_model_spec("gemma3-4b")
    begin = {
        "prompt": "plain|task",
        "prompt_ids": list(range(10)),
        "context": {
            "step": 0,
            "messages": [
                {"role": "system", "content": "plain"},
                {"role": "user", "content": "task"},
            ],
        },
    }
    with pytest.raises(ValueError, match="observation convention"):
        t._prompt_semantics(spec, begin, [[i, i + 1] for i in range(10)])


def test_rehashed_semantic_span_tampering_is_refused(tmp_path):
    source, spec, ident, tok_id = make_record(tmp_path)
    path = tmp_path / "corpus.json"
    t.build_transcript_corpus(
        [source], Tokenizer(), spec, path, tokenizer_identity=tok_id, model_identity=ident
    )
    manifest = json.loads(path.read_text())
    seq = tmp_path / "corpus.jsonl"
    rows = [json.loads(line) for line in seq.read_text().splitlines()]
    rows[0]["spans"][0] = "call"
    seq.write_bytes(b"".join(t.encoded(r) + b"\n" for r in rows))
    manifest["sequences"] = t.file_record(seq)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = t.digest(manifest)
    path.write_bytes(t.encoded(manifest))
    with pytest.raises(ValueError, match="invalid transcript window"):
        t.read_transcript_corpus(path)


def test_concentration_counts_member_generation_and_following_observations():
    same = {"name": "read_file", "arguments": {"path": "x"}}
    turns = [
        {"task_id": "train-read-0000-clean", "step": step, "canonical_action": action}
        for step, action in enumerate([same, same, {"name": "finish", "arguments": {}}, None])
    ]
    rows = [
        {
            "split": "fit",
            "turn_index": i,
            "ids": [0] * 4,
            "score_positions": [0, 1, 2, 3],
            "generated_mask": [False, False, True, True],
            "spans": ["observation", "observation", "note", "call"],
        }
        for i in range(4)
    ]
    result = t.transcript_acceptance(rows, turns)
    fit = result["concentration"]["fit"]
    assert result["repeated_runs"] == [[0, 1]]
    assert fit["repeated_run_positions"] == 8  # gen0+gen1, obs1+obs2; scaffold0 excluded
    assert fit["positions"] == 16
    assert fit["invalid_action_turns"] == 1
    assert fit["invalid_action_generated_positions"] == 2
    assert fit["largest_episode_share"] == 1
    assert result["status"] == "ruling_required"


def test_reader_retains_a_complete_corpus_that_requires_ruling(tmp_path):
    source, spec, ident, tok_id = make_record(tmp_path)
    path = tmp_path / "corpus.json"
    manifest = t.build_transcript_corpus(
        [source], Tokenizer(), spec, path, tokenizer_identity=tok_id, model_identity=ident
    )
    assert manifest["acceptance"]["status"] == "ruling_required"
    assert manifest["acceptance"]["concentration"]["fit"]["largest_episode_share"] == 1
    assert t.read_transcript_corpus(path)
