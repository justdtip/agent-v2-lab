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


def make_record(
    tmp_path, *, turns=5, long=False, tokenizer=None, context_tokens=2048, position_support=None
):
    tokenizer = Tokenizer() if tokenizer is None else tokenizer
    spec = load_model_spec("gemma3-4b")
    ident = {"base": spec.base, "training": None, "num_layers": 34}
    tok_id = identity(tmp_path)
    path = tmp_path / "capture.jsonl"
    provenance = {
        "model_identity": ident,
        "tokenizer": tok_id,
        "fitting_context_tokens": context_tokens,
    }
    if position_support is not None:
        provenance["position_support"] = position_support
    with t.TranscriptWriter(path, provenance) as writer:
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
                lambda *_a, **_k: None, tokenizer, prompt, turn_cache=None
            ) as model:
                model(Inputs(tokenizer.encode(prompt)))
                capture.emitted(ord("n"))
                model(Inputs([ord("n")]))
                capture.emitted(ord("!"))  # emitted, not consumed
    return path, spec, ident, tok_id


def test_capture_roundtrip_and_unique_long_context(tmp_path):
    source, spec, ident, tok_id = make_record(tmp_path, long=True)
    path = tmp_path / "corpus.json"
    manifest = t.build_transcript_corpus(
        [source],
        Tokenizer(),
        spec,
        path,
        tokenizer_identity=tok_id,
        model_identity=ident,
        max_tokens=2048,
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
        [source],
        Tokenizer(),
        spec,
        path,
        tokenizer_identity=tok_id,
        model_identity=ident,
        max_tokens=2048,
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
        [source],
        Tokenizer(),
        spec,
        path,
        tokenizer_identity=tok_id,
        model_identity=ident,
        max_tokens=2048,
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
            "spans": ["observation", "observation", "note", "call_skeleton"],
            "window_start": 0,
            "bos": t._bos_evidence([0] * 4, None),
        }
        for i in range(4)
    ]
    result = t.transcript_acceptance(rows, turns, max_tokens=2048)
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
        [source],
        Tokenizer(),
        spec,
        path,
        tokenizer_identity=tok_id,
        model_identity=ident,
        max_tokens=2048,
    )
    assert manifest["acceptance"]["status"] == "ruling_required"
    assert manifest["acceptance"]["concentration"]["fit"]["largest_episode_share"] == 1
    assert t.read_transcript_corpus(path)


def test_boolean_and_integer_arguments_are_not_identical_repeated_calls():
    turns = [
        {
            "task_id": "train-read-0000-clean",
            "step": i,
            "canonical_action": {"name": "read_file", "arguments": {"offset": value}},
        }
        for i, value in enumerate([True, 1, False, 0])
    ]
    assert t.transcript_acceptance([], turns, max_tokens=2048)["repeated_runs"] == []


def test_reader_rejects_same_episode_in_different_source_files(tmp_path):
    source, spec, ident, tok_id = make_record(tmp_path)
    path = tmp_path / "corpus.json"
    t.build_transcript_corpus(
        [source],
        Tokenizer(),
        spec,
        path,
        tokenizer_identity=tok_id,
        model_identity=ident,
        max_tokens=2048,
    )
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_bytes(source.read_bytes())
    manifest = json.loads(path.read_text())
    manifest["sources"].append(t.file_record(duplicate))
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = t.digest(manifest)
    path.write_bytes(t.encoded(manifest))
    with pytest.raises(ValueError, match="duplicate trajectory"):
        t.read_transcript_corpus(path)


def test_context_length_must_match_frozen_source_registration(tmp_path):
    source, spec, ident, tok_id = make_record(tmp_path)
    with pytest.raises(ValueError, match="capture identity"):
        t.build_transcript_corpus(
            [source],
            Tokenizer(),
            spec,
            tmp_path / "wrong.json",
            tokenizer_identity=tok_id,
            model_identity=ident,
            max_tokens=4096,
        )
    path = tmp_path / "corpus.json"
    t.build_transcript_corpus(
        [source],
        Tokenizer(),
        spec,
        path,
        tokenizer_identity=tok_id,
        model_identity=ident,
        max_tokens=2048,
    )
    manifest = json.loads(path.read_text())
    manifest["max_tokens"] = 4096
    manifest["window_rule"] = t._window_rule(4096)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = t.digest(manifest)
    path.write_bytes(t.encoded(manifest))
    with pytest.raises(ValueError, match="source identity"):
        t.read_transcript_corpus(path)


def test_generated_facets_reuse_shared_span_labeller_at_capture_and_read(tmp_path):
    pieces = [
        "Inspect first.",
        '```json\n{"name":"read_file","arguments":{',
        '"path":',
        ' "/work',
        'space"}}\n```',
    ]

    class PieceTokenizer:
        bos_token = None
        bos_token_id = None

        def encode(self, text, **kwargs):
            return [999]

        def decode(self, ids):
            return "".join(pieces[i] for i in ids)

    tokenizer = PieceTokenizer()
    labels, _, _ = t._generated_labels(tokenizer, list(range(len(pieces))))
    assert labels == ["note", "call_skeleton", "call_skeleton", "call_argument", "call_argument"]
    chat, _, _ = t._generated_labels(tokenizer, list(range(len(pieces))), kind="chat")
    assert chat == ["chat_prose"] * len(pieces)
    record = tmp_path / "facets.jsonl"
    with t.TranscriptWriter(record, {}) as writer:
        capture = t.TranscriptCapture(writer, materialize=lambda _: None)
        capture.set_context(task_id="train-read-0000-clean", step=0, messages=[])
        with capture.generation(
            lambda *_a, **_k: None, tokenizer, "prompt", turn_cache=None
        ) as model:
            model(Inputs([999]))
            for token in range(len(pieces)):
                capture.emitted(token)
                model(Inputs([token]))
    events = t.read_transcript(record)
    assert [e["span"] for e in events if e["kind"] == "emitted"] == labels
    assert list(t._turns(events))[0][0]["emitted_spans"] == labels
    malformed, _, _ = t._generated_labels(tokenizer, [0, 1, 2, 3])
    assert malformed[-1] == "call_argument"


def test_position_histograms_and_bos_distinguish_capture_from_fresh_replay(tmp_path):
    class BosTokenizer(Tokenizer):
        bos_token_id = 1

        def encode(self, text, **kwargs):
            return [1] + super().encode(text, **kwargs)

        def __call__(self, text, **kwargs):
            return {
                "input_ids": self.encode(text),
                "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(text))],
            }

    tokenizer = BosTokenizer()
    source, spec, ident, tok_id = make_record(tmp_path, long=True, tokenizer=tokenizer)
    path = tmp_path / "corpus.json"
    manifest = t.build_transcript_corpus(
        [source],
        tokenizer,
        spec,
        path,
        tokenizer_identity=tok_id,
        model_identity=ident,
        max_tokens=2048,
    )
    rows = t.read_transcript_corpus(path)
    assert any(
        row["window_start"] > 0 and not row["bos"]["starts_with_captured_bos"] for row in rows
    )
    assert all(row["bos"]["inserted_bos_tokens"] == 0 for row in rows)
    assert all(
        row["original_position_range"]
        == [row["window_start"], row["window_start"] + len(row["ids"])]
        for row in rows
    )
    assert all(row["replay_position_range"] == [0, len(row["ids"])] for row in rows)
    for group in manifest["counts"].values():
        for scope, denominator in [("scored", group["positions"]), ("all_input", group["tokens"])]:
            hist = group["position_distributions"][scope]
            assert sum(hist["original_captured_offsets"].values()) == denominator
            assert sum(hist["actual_replay_positions"].values()) == denominator
        assert sum(group["generated_spans"].values()) == group["generated"]
        assert sum(group["input_spans"].values()) + group["generated"] == group["positions"]
        assert "observation" not in group["generated_spans"]
    fit = manifest["counts"]["fit"]
    assert fit["bos"]["scored_bos_positions"] == 1
    assert (
        max(map(int, fit["position_distributions"]["scored"]["original_captured_offsets"])) > 2047
    )
    assert max(map(int, fit["position_distributions"]["scored"]["actual_replay_positions"])) <= 2047
    assert "not asserted native RoPE" in manifest["position_policy"]
    assert (
        manifest["acceptance"]["non_prose_span_check"]["denominator"]
        == "all fitted scored positions"
    )
    assert "note" in manifest["acceptance"]["non_prose_span_check"]["eligible_counts"]


def support_request():
    return {
        "target_max_position": 2749,
        "reference_max_position": 2047,
        "source": "Director review 2026-09-08; reported maximum across held map episodes",
    }


def support_fixture_rows(scored_position):
    rows, turns = [], []
    for i, split in enumerate(("fit", "fit", "fit", "held")):
        turns.append({"task_id": f"train-read-{i:04d}-clean", "step": 0, "canonical_action": None})
        rows.append(
            {
                "split": split,
                "turn_index": i,
                "ids": [0] * 2816,
                "score_positions": [scored_position if split == "fit" else 2815],
                "generated_mask": [True] * 2816,
                "spans": ["call_skeleton"] * 2816,
                "window_start": 9000,
                "bos": t._bos_evidence([0] * 2816, None),
            }
        )
    return rows, turns


@pytest.mark.parametrize(("actual", "expected"), [(2748, "ruling_required"), (2749, "passed")])
def test_position_support_uses_scored_replay_max_not_declared_cap_or_input(actual, expected):
    rows, turns = support_fixture_rows(actual)
    acceptance = t.transcript_acceptance(
        rows, turns, max_tokens=2816, position_support=support_request()
    )
    assert acceptance["status"] == expected
    report = acceptance["position_support"]
    assert report["fit_scored_max_position"] == actual
    assert report["fit_input_max_position"] == 2815
    assert report["held_scored_max_position"] == 2815
    assert report["held_input_max_position"] == 2815
    assert report["fit_scored_at_or_above_target"] == (3 if actual == 2749 else 0)
    assert report["fit_scored_above_reference"] == {
        "reference_max_position": 2047,
        "count": 3,
        "denominator": 3,
        "fraction": 1.0,
    }


def test_position_support_zero_based_last_index_and_strict_integer_validation():
    request = support_request()
    for bad in (True, 2816, -1):
        with pytest.raises(ValueError, match="position support"):
            t.validate_position_support({**request, "target_max_position": bad}, max_tokens=2816)
    assert t.validate_position_support({**request, "target_max_position": 2815}, max_tokens=2816)


def test_requested_position_support_roundtrip_and_lowered_target_tamper(tmp_path):
    request = support_request()
    source, spec, ident, tok_id = make_record(
        tmp_path, long=True, context_tokens=2816, position_support=request
    )
    path = tmp_path / "corpus.json"
    manifest = t.build_transcript_corpus(
        [source],
        Tokenizer(),
        spec,
        path,
        tokenizer_identity=tok_id,
        model_identity=ident,
        max_tokens=2816,
    )
    rows = t.read_transcript_corpus(path)
    assert manifest["position_support"] == request
    report = manifest["acceptance"]["position_support"]
    assert report["fit_scored_max_position"] == 2815
    assert report["fit_scored_at_or_above_target"] > 0
    assert max(len(row["ids"]) for row in rows) == 2816
    changed = json.loads(path.read_text())
    changed["position_support"]["target_max_position"] = 2748
    changed.pop("manifest_sha256")
    changed["manifest_sha256"] = t.digest(changed)
    path.write_bytes(t.encoded(changed))
    with pytest.raises(ValueError, match="source position support"):
        t.read_transcript_corpus(path)
    with pytest.raises(ValueError, match="source position support"):
        t.build_transcript_corpus(
            [source],
            Tokenizer(),
            spec,
            tmp_path / "changed.json",
            tokenizer_identity=tok_id,
            model_identity=ident,
            max_tokens=2816,
            position_support={**request, "target_max_position": 2748},
        )


def test_unrequested_legacy_position_support_leaves_output_unchanged(tmp_path):
    source, spec, ident, tok_id = make_record(tmp_path)
    path = tmp_path / "legacy.json"
    manifest = t.build_transcript_corpus(
        [source],
        Tokenizer(),
        spec,
        path,
        tokenizer_identity=tok_id,
        model_identity=ident,
        max_tokens=2048,
    )
    assert "position_support" not in manifest
    assert "position_support" not in manifest["acceptance"]
    assert t.read_transcript_corpus(path)
