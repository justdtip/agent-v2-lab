"""The capture contract, exercised: every refusal fires, and the happy path writes what it promises.

Each test here corresponds to a way a capture can be silently useless later — a width nobody
recorded, a promoted path nobody declared, a position stated as a rule rather than as an integer, a
corpus that moved after the set was fixed. The point of the tests is that those are refusals now
rather than confusions six weeks from now, when the only symptom is a lens reading that disagrees.
"""

from __future__ import annotations

import json

import pytest
import torch

from local_llm_lab.pipeline.state_programme import capture


def _identity(config="cfg", weights=("aa", "bb")) -> dict:
    """A checkpoint identity of the shape the loader's digest manifest produces."""
    manifest = {"config.json": config}
    manifest.update({f"model-{i:05d}.safetensors": w for i, w in enumerate(weights)})
    return capture.checkpoint_identity(manifest)


def _decision(**overrides) -> dict:
    base = {
        "task_id": "test-read-0000-clean", "step": 1, "split": "test", "family": "read",
        "variant": "clean", "difficulty": 2, "recovery": False, "rendered_rows": 1,
        "messages_sha256": "", "row_ordinals": [0],
    }
    return {**base, **overrides}


def _row(task="test-read-0000-clean", step=1, text="a") -> dict:
    """A row of the shape the **agent** corpus actually has: a rendered prompt string, no ids."""
    return {
        "metadata": {"task_id": task, "step": step, "family": "read", "variant": "clean"},
        "messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "b"}],
        "prompt": f"<bos>{text}<start_of_turn>model\n",
        "completion": "b<end_of_turn>",
    }


def _cell(**overrides) -> dict:
    base = {
        "task_id": "t", "step": 0, "split": "test", "family": "read", "variant": "clean",
        "difficulty": 2, "recovery": False, "rendered_rows": 1, "row_ordinals": [0],
        "messages_sha256": "m", "rendered_prompt_sha256": "r", "token_ids_sha256": "t",
        "token_ids_length": 4, "dtype": "torch.bfloat16",
        "token_index": 3, "seq_len": 4, "forward_batch": 1, "anchor_batch": 1,
        "capture_dtype": "native", "checkpoint_sha256": "c", "layers": 2, "d_model": 4,
        "device": "cpu", "decoding": "greedy", "shard": 0, "index_in_shard": 0,
        "config_sha256": "cfg", "weight_files": 2, "basis": "measured-here",
    }
    return {**base, **overrides}


def _prepare(**override):
    """The preparation seam: the exact model input, before the reuse decision."""
    import hashlib

    def run(row):
        ids = list(range(max(1, len(row["prompt"]))))
        return {
            "token_ids": ids,
            "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "token_ids_length": len(ids),
            "rendered_prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
        } | override
    return run


def _forward(layers=2, d_model=4, **override):
    """A seam that attests to a conforming pass. Overrides let a test make it attest otherwise."""
    import hashlib

    def run(row, prepared=None):
        ids = (prepared or _prepare()(row))["token_ids"]
        attestation = {
            "residuals": torch.zeros(layers, d_model, dtype=torch.bfloat16),
            "seq_len": len(ids), "token_index": len(ids) - 1, "layers": layers,
            "d_model": d_model, "device": "cpu", "dtype": "torch.bfloat16",
            "forward_batch": 1, "anchor_batch": 1, "capture_dtype": "native",
            "rendered_prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
            "token_ids_sha256": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest(),
            "token_ids_length": len(ids),
        }
        return {**attestation, **override}
    return run


def test_the_module_reads_only_fields_a_real_corpus_row_has() -> None:
    """The check that would have caught it: assert the assumption against the corpus, not a fixture.

    The first version of this module read `row["ids"]`, which is the *lens* corpus's shape. The
    agent corpus has `messages`, `prompt`, `completion` and `metadata` and no ids. Every test passed,
    because every test built its own row and helpfully supplied one. A fixture that constructs its
    own input validates the code against a corpus that does not exist, so this pins the declared
    fields against a real row and skips only when the corpus is genuinely absent.
    """
    import json
    from pathlib import Path

    corpus = Path("data/agent_v2e-gemma3-4b/test.jsonl")
    if not corpus.exists():
        pytest.skip(f"{corpus} is not on this machine")
    with corpus.open(encoding="utf-8") as stream:
        real = json.loads(next(line for line in stream if line.strip()))
    missing = [field for field in capture.REQUIRED_ROW_FIELDS if field not in real]
    assert not missing, f"the module requires {missing}, which a real corpus row does not carry"
    assert "ids" not in real, "this corpus has no token ids; the seam must own tokenization"


# ------------------------------------------------------------------- the contract's own refusals

def test_a_cell_missing_any_required_field_is_refused_naming_all_of_them() -> None:
    cell = _cell()
    del cell["token_index"], cell["decoding"]
    with pytest.raises(capture.ContractViolation, match="declares no"):
        capture.assert_contract(cell)


@pytest.mark.parametrize("field", capture.REQUIRED_CELL_FIELDS)
def test_every_required_field_is_actually_required(field: str) -> None:
    """A field listed in the contract and not checked would be documentation, not a contract."""
    cell = _cell()
    del cell[field]
    with pytest.raises(capture.ContractViolation):
        capture.assert_contract(cell)


@pytest.mark.parametrize("width", [2, 64, 256])
def test_a_capture_at_any_other_width_is_refused(width: int) -> None:
    """The forward is not batch-invariant, so a residual at another width is another function."""
    with pytest.raises(capture.ContractViolation, match="batch-invariant"):
        capture.assert_contract(_cell(forward_batch=width))
    with pytest.raises(capture.ContractViolation, match="batch-invariant"):
        capture.assert_contract(_cell(anchor_batch=width))


def test_a_promoted_capture_is_refused_by_name() -> None:
    with pytest.raises(capture.ContractViolation, match="residual_precision_probe"):
        capture.assert_contract(_cell(capture_dtype="promoted-float32"))


def test_the_position_must_be_an_integer_inside_the_sequence() -> None:
    with pytest.raises(capture.ContractViolation, match="rule for finding it"):
        capture.assert_contract(_cell(token_index="last"))
    with pytest.raises(capture.ContractViolation, match="rule for finding it"):
        capture.assert_contract(_cell(token_index=-1))
    with pytest.raises(capture.ContractViolation, match="outside seq_len"):
        capture.assert_contract(_cell(token_index=4, seq_len=4))


# ------------------------------------------------------- the corpus and the set must still agree

def test_a_corpus_row_whose_prompt_digest_moved_is_refused(tmp_path) -> None:
    row = _row()
    decision = _decision(messages_sha256=capture.prompt_digest(row["messages"]))
    moved = _row(text="something else")
    target = capture.CaptureTarget(directory=tmp_path, entry="e", identity=_identity(),
                                   decoding="greedy")
    with pytest.raises(capture.ContractViolation, match="re-enumerate rather than capture"):
        capture.capture_decisions(
            decisions=[decision], corpus=capture.rows_by_decision([moved]),
            forward=_forward(), target=target,
        )


def test_a_decision_absent_from_the_corpus_is_refused(tmp_path) -> None:
    target = capture.CaptureTarget(directory=tmp_path, entry="e", identity=_identity(),
                                   decoding="greedy")
    with pytest.raises(capture.ContractViolation, match="drifted apart"):
        capture.capture_decisions(decisions=[_decision()], corpus={}, forward=_forward(),
                                  target=target)


def test_repeated_keys_index_to_one_decision_because_the_copies_are_identical() -> None:
    rows = [_row(), _row(), _row()]
    index = capture.rows_by_decision(rows)
    assert len(index) == 1
    assert index[("test-read-0000-clean", 1)] is rows[0]


# --------------------------------------------------------------------------- what a run produces

def test_a_run_writes_its_shards_and_a_manifest_line_per_decision(tmp_path) -> None:
    rows = [_row(step=i) for i in range(5)]
    decisions = [
        _decision(step=i, messages_sha256=capture.prompt_digest(r["messages"]))
        for i, r in enumerate(rows)
    ]
    target = capture.CaptureTarget(directory=tmp_path, entry="e", identity=_identity(),
                                   decoding="greedy", shard_size=2)
    summary = capture.capture_decisions(
        decisions=decisions, corpus=capture.rows_by_decision(rows),
        forward=_forward(), target=target,
    )

    assert summary["captured"] == 5
    lines = [json.loads(l) for l in (tmp_path / "manifest.jsonl").read_text().splitlines() if l]
    assert len(lines) == 5
    for cell in lines:
        capture.assert_contract(cell)          # the manifest itself meets the contract
        assert cell["checkpoint_sha256"] == _identity()["checkpoint_sha256"]
        assert cell["shard_sha256"]
        assert (tmp_path / f"residuals-{cell['shard']:05d}.pt").exists()
    # Three shards at size two, and the residuals keep their native dtype rather than promoting.
    assert sorted({c["shard"] for c in lines}) == [0, 1, 2]
    assert torch.load(tmp_path / "residuals-00000.pt").dtype is torch.bfloat16


def test_a_second_run_resumes_from_the_manifest_and_does_not_recapture(tmp_path) -> None:
    """An interrupted capture pass costs the shard it was writing, never the pass."""
    rows = [_row(step=i) for i in range(4)]
    decisions = [
        _decision(step=i, messages_sha256=capture.prompt_digest(r["messages"]))
        for i, r in enumerate(rows)
    ]
    corpus = capture.rows_by_decision(rows)
    target = capture.CaptureTarget(directory=tmp_path, entry="e", identity=_identity(),
                                   decoding="greedy", shard_size=2)
    capture.capture_decisions(decisions=decisions[:2], corpus=corpus, forward=_forward(),
                              target=target)

    calls = []

    def counting(ids):
        calls.append(ids)
        return _forward()(ids)

    summary = capture.capture_decisions(decisions=decisions, corpus=corpus, forward=counting,
                                        target=target)
    assert summary["already_present"] == 2
    assert summary["captured"] == 2
    assert len(calls) == 2, "a resumed run forwarded a decision it had already captured"


def test_resuming_after_a_partial_shard_does_not_overwrite_it(tmp_path) -> None:
    """The bug this pins: `len(done) // shard_size` hands the next run the partial shard's number.

    A run that ends on a partial shard leaves a manifest whose count is not a multiple of the shard
    size. Deriving the next shard by division then reuses that number, the file is overwritten, and
    the `shard_sha256` beside the already-written manifest lines silently stops matching its file.
    Detectable afterwards, which is not the same as prevented.
    """
    rows = [_row(step=i) for i in range(7)]
    decisions = [
        _decision(step=i, messages_sha256=capture.prompt_digest(r["messages"]))
        for i, r in enumerate(rows)
    ]
    corpus = capture.rows_by_decision(rows)
    target = capture.CaptureTarget(directory=tmp_path, entry="e", identity=_identity(),
                                   decoding="greedy", shard_size=2)

    capture.capture_decisions(decisions=decisions[:5], corpus=corpus, forward=_forward(),
                              target=target)   # shards 0 and 1 full, shard 2 holds one
    first = [json.loads(l) for l in (tmp_path / "manifest.jsonl").read_text().splitlines() if l]
    partial = next(c for c in first if c["shard"] == 2)
    digest_before = partial["shard_sha256"]

    capture.capture_decisions(decisions=decisions, corpus=corpus, forward=_forward(),
                              target=target)
    after = [json.loads(l) for l in (tmp_path / "manifest.jsonl").read_text().splitlines() if l]

    assert len(after) == 7, "every decision has exactly one manifest line"
    # The partial shard is untouched: same file, same digest, and nothing else claims its number.
    import hashlib
    on_disk = hashlib.sha256((tmp_path / "residuals-00002.pt").read_bytes()).hexdigest()
    assert on_disk == digest_before, "the partial shard was overwritten by a later run"
    assert {c["shard"] for c in after if c["step"] in (5, 6)} == {3}
    # And every manifest line still describes the file it names.
    for cell in after:
        path = tmp_path / f"residuals-{cell['shard']:05d}.pt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == cell["shard_sha256"]


def test_the_summary_reports_what_was_asked_for_and_not_only_what_was_written(tmp_path) -> None:
    """Method entry thirty-four's coverage rule, applied to this module's own verdict.

    A summary saying how many cells were written, and not how many were asked for, cannot tell a
    complete pass from a truncated one — and both read as success. The capture pass is the place
    that matters most, because a partial capture set produces a probe fitted on a subset nobody
    declared.
    """
    rows = [_row(step=i) for i in range(3)]
    decisions = [
        _decision(step=i, messages_sha256=capture.prompt_digest(r["messages"]))
        for i, r in enumerate(rows)
    ]
    corpus = capture.rows_by_decision(rows)
    target = capture.CaptureTarget(directory=tmp_path, entry="e", identity=_identity(),
                                   decoding="greedy", shard_size=2)

    partial = capture.capture_decisions(decisions=decisions[:2], corpus=corpus,
                                        forward=_forward(), target=target)
    assert partial == {**partial, "requested": 2, "captured": 2, "outstanding": 0, "complete": True}

    # Asked for all three against a directory holding two: complete only once the third is written.
    full = capture.capture_decisions(decisions=decisions, corpus=corpus, forward=_forward(),
                                     target=target)
    assert full["requested"] == 3
    assert full["already_present"] == 2
    assert full["captured"] == 1
    assert full["outstanding"] == 0 and full["complete"] is True


# ----------------------------------------- F2: an identity that cannot tell checkpoints apart

def test_two_checkpoints_sharing_a_config_do_not_share_an_identity() -> None:
    """The defect: `sha256["config.json"]` as the checkpoint's identity.

    Same config, every weight different, identical identity — an identity that cannot distinguish
    the thing it identifies. The captures would then be attributed to a checkpoint that did not
    produce them, and nothing downstream could tell.
    """
    a = _identity(config="same", weights=("w1", "w2"))
    b = _identity(config="same", weights=("DIFFERENT", "ALSO-DIFFERENT"))
    assert a["config_sha256"] == b["config_sha256"], "the fixture must share a config"
    assert a["checkpoint_sha256"] != b["checkpoint_sha256"]


def test_an_identity_without_weights_or_without_a_config_is_refused() -> None:
    with pytest.raises(capture.ContractViolation, match="no config.json"):
        capture.checkpoint_identity({"model-00000.safetensors": "w"})
    with pytest.raises(capture.ContractViolation, match="different weights would then share"):
        capture.checkpoint_identity({"config.json": "c"})
    for empty in ({}, None, "abc"):
        with pytest.raises(capture.ContractViolation, match="no file digests"):
            capture.checkpoint_identity(empty)


# --------------------------------------------- F3: a resume that verifies rather than trusts

def _started(tmp_path, shard_size=2, identity=None):
    rows = [_row(step=i) for i in range(3)]
    decisions = [
        _decision(step=i, messages_sha256=capture.prompt_digest(r["messages"]))
        for i, r in enumerate(rows)
    ]
    corpus = capture.rows_by_decision(rows)
    target = capture.CaptureTarget(directory=tmp_path, entry="e",
                                   identity=identity or _identity(), decoding="greedy",
                                   shard_size=shard_size)
    capture.capture_decisions(decisions=decisions[:2], corpus=corpus, forward=_forward(),
                              target=target)
    return rows, decisions, corpus, target


def test_resuming_under_a_different_checkpoint_is_refused(tmp_path) -> None:
    _, decisions, corpus, _ = _started(tmp_path)
    other = capture.CaptureTarget(directory=tmp_path, entry="e",
                                  identity=_identity(weights=("zz", "yy")), decoding="greedy",
                                  shard_size=2)
    with pytest.raises(capture.ResumeUnverified, match="mix two checkpoints"):
        capture.capture_decisions(decisions=decisions, corpus=corpus, forward=_forward(),
                                  target=other)


def test_resuming_when_the_messages_have_moved_is_refused(tmp_path) -> None:
    """The corpus moved and the request did not: refused before the reuse branch is reached.

    The request's own digest is checked first, so a corpus that has drifted from the capture set is
    a `ContractViolation` — re-enumerate — and only a corpus agreeing with the request but not with
    the manifest reaches `ResumeUnverified`.
    """
    rows, decisions, _, target = _started(tmp_path)
    moved = capture.rows_by_decision([_row(step=i, text=f"changed {i}") for i in range(3)])
    with pytest.raises(capture.ContractViolation, match="re-enumerate rather than capture"):
        capture.capture_decisions(decisions=decisions, corpus=moved, forward=_forward(),
                                  target=target)


def test_resuming_when_only_the_rendered_bytes_moved_is_refused(tmp_path) -> None:
    """The case the semantic digest cannot see, which is the whole reason both are recorded.

    Same `messages`, different rendered prompt: the message record hashes identically and the bytes
    the model would read do not. A resume checking only the semantic digest accepts a capture of an
    input this run would never produce.
    """
    rows, decisions, _, target = _started(tmp_path)
    restyled = []
    for i in range(3):
        row = _row(step=i)
        row["prompt"] = row["prompt"].replace("<bos>", "<bos> ")   # same messages, other bytes
        restyled.append(row)
    corpus = capture.rows_by_decision(restyled)
    assert capture.prompt_digest(restyled[0]["messages"]) == capture.prompt_digest(
        _row(step=0)["messages"]
    ), "the fixture must leave the message record identical"
    with pytest.raises(capture.ResumeUnverified, match="rendered prompt has moved"):
        capture.capture_decisions(decisions=decisions, corpus=corpus, forward=_forward(),
                                  target=target)


def test_resuming_with_a_missing_shard_is_refused(tmp_path) -> None:
    _, decisions, corpus, target = _started(tmp_path)
    (tmp_path / "residuals-00000.pt").unlink()
    with pytest.raises(capture.ResumeUnverified, match="not on disk"):
        capture.capture_decisions(decisions=decisions, corpus=corpus, forward=_forward(),
                                  target=target)


def test_resuming_with_altered_shard_bytes_is_refused(tmp_path) -> None:
    """The one a membership check can never see: the file is there and it is not the file."""
    _, decisions, corpus, target = _started(tmp_path)
    path = tmp_path / "residuals-00000.pt"
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(capture.ResumeUnverified, match="bytes have changed"):
        capture.capture_decisions(decisions=decisions, corpus=corpus, forward=_forward(),
                                  target=target)


def test_a_clean_resume_reports_what_it_verified(tmp_path) -> None:
    _, decisions, corpus, target = _started(tmp_path)
    summary = capture.capture_decisions(decisions=decisions, corpus=corpus, forward=_forward(),
                                        prepare=_prepare(), target=target)
    assert summary["complete"] is True
    assert summary["reused"] == 2, "the two already captured were checked and kept"
    assert summary["captured"] == 1
    assert summary["consumed_ids_verified_on_reuse"] is True
    assert "the consumed ids" in summary["verified_scope"]


def test_without_a_prepare_seam_the_summary_says_the_ids_were_not_verified(tmp_path) -> None:
    """A verification not performed is named, not omitted — the coverage rule, applied here."""
    _, decisions, corpus, target = _started(tmp_path)
    summary = capture.capture_decisions(decisions=decisions, corpus=corpus, forward=_forward(),
                                        target=target)
    assert summary["consumed_ids_verified_on_reuse"] is False
    assert "not** the consumed ids" in summary["verified_scope"]


def test_a_reused_cell_whose_ids_this_tokenizer_would_not_produce_is_refused(tmp_path) -> None:
    """Codex R2: two tokenizers give different ids for the same bytes, and the checkpoint identity
    covers no tokenizer asset, so nothing else in the chain can see this."""
    _, decisions, corpus, target = _started(tmp_path)
    with pytest.raises(capture.ResumeUnverified, match="no tokenizer asset"):
        capture.capture_decisions(
            decisions=decisions, corpus=corpus, forward=_forward(),
            prepare=_prepare(token_ids_sha256="0" * 64), target=target,
        )


def test_a_reused_cell_the_request_no_longer_describes_is_refused(tmp_path) -> None:
    """Codex R1: the capture set re-enumerated after the pass began.

    The manifest and the corpus still agree with each other, so every earlier check passes; only a
    comparison against the *request* can see that the decision has been re-labelled.
    """
    _, decisions, corpus, target = _started(tmp_path)
    reenumerated = [{**d, "variant": "wrong_path"} for d in decisions]
    with pytest.raises(capture.ResumeUnverified, match="re-enumerated since the pass began"):
        capture.capture_decisions(decisions=reenumerated, corpus=corpus, forward=_forward(),
                                  prepare=_prepare(), target=target)


# ------------- C1: the writer must check the seam's report, not stamp the contract's constants

def _one(tmp_path, **override):
    row = _row()
    decision = _decision(messages_sha256=capture.prompt_digest(row["messages"]))
    target = capture.CaptureTarget(directory=tmp_path, entry="e", identity=_identity(),
                                   decoding="greedy")
    return lambda: capture.capture_decisions(
        decisions=[decision], corpus=capture.rows_by_decision([row]),
        forward=_forward(**override), target=target,
    )


def test_a_seam_that_ran_at_another_width_is_refused_not_relabelled(tmp_path) -> None:
    """The defect: the writer wrote `forward_batch = FORWARD_BATCH` and then asserted it was.

    A seam that expanded a batch was recorded as width 1 — the guard erased the evidence it existed
    to check. Mutating the cell after the fact was refused, which is why the old tests passed; a
    seam reporting the truth was accepted and overwritten, which is what a real one would do.
    """
    with pytest.raises(capture.ContractViolation, match="not batch-invariant"):
        _one(tmp_path, forward_batch=64)()
    with pytest.raises(capture.ContractViolation, match="not batch-invariant"):
        _one(tmp_path, anchor_batch=64)()


def test_a_seam_that_ran_a_promoted_path_is_refused_not_relabelled(tmp_path) -> None:
    with pytest.raises(capture.ContractViolation, match="arithmetic path"):
        _one(tmp_path, capture_dtype="promoted-float32")()


def test_a_non_integer_position_is_refused_before_it_is_coerced(tmp_path) -> None:
    """`int(1.9)` is 1, and a writer that coerces before validating validates its own coercion."""
    with pytest.raises(capture.ContractViolation, match="before any coercion"):
        _one(tmp_path, token_index=1.9)()
    with pytest.raises(capture.ContractViolation, match="before any coercion"):
        _one(tmp_path, token_index=True)()


def test_a_seam_that_tokenized_other_bytes_is_refused(tmp_path) -> None:
    """The check `messages_sha256` cannot perform: the ids consumed are not the row's prompt."""
    with pytest.raises(capture.ContractViolation, match="an input this row does not carry"):
        _one(tmp_path, rendered_prompt_sha256="0" * 64)()


def test_a_seam_whose_residuals_do_not_match_its_declared_shape_is_refused(tmp_path) -> None:
    with pytest.raises(capture.ContractViolation, match="cannot be indexed by layer"):
        _one(tmp_path, residuals=torch.zeros(7, 3, dtype=torch.bfloat16))()


def test_a_seam_that_attests_to_nothing_is_refused_naming_the_fields(tmp_path) -> None:
    row = _row()
    decision = _decision(messages_sha256=capture.prompt_digest(row["messages"]))
    target = capture.CaptureTarget(directory=tmp_path, entry="e", identity=_identity(),
                                   decoding="greedy")
    with pytest.raises(capture.ContractViolation, match="attest to the pass it actually ran"):
        capture.capture_decisions(
            decisions=[decision], corpus=capture.rows_by_decision([row]),
            forward=lambda row: {"residuals": torch.zeros(2, 4), "seq_len": 1, "token_index": 0,
                                 "layers": 2, "d_model": 4, "device": "cpu", "dtype": "x"},
            target=target,
        )


def test_a_conforming_pass_is_recorded_from_the_seam_and_not_from_the_constants(tmp_path) -> None:
    """The positive half: the cell's values come from the seam's report, traceably."""
    row = _row()
    decision = _decision(messages_sha256=capture.prompt_digest(row["messages"]))
    target = capture.CaptureTarget(directory=tmp_path, entry="e", identity=_identity(),
                                   decoding="greedy")
    capture.capture_decisions(decisions=[decision], corpus=capture.rows_by_decision([row]),
                              forward=_forward(), target=target)
    cell = json.loads((tmp_path / "manifest.jsonl").read_text().splitlines()[0])

    import hashlib
    assert cell["rendered_prompt_sha256"] == hashlib.sha256(row["prompt"].encode()).hexdigest()
    assert cell["token_ids_length"] == cell["seq_len"] == len(row["prompt"])
    assert cell["token_index"] == cell["seq_len"] - 1
    assert cell["messages_sha256"] == capture.prompt_digest(row["messages"])
    assert cell["row_ordinals"] == [0]
    assert cell["dtype"] == "torch.bfloat16"


def test_the_progress_callback_is_actually_called_and_its_fields_are_right(tmp_path) -> None:
    """The callback path, exercised. It had never been, and it raised NameError on the device.

    `progress({... "captured": written + len(done)})` referred to a name that does not exist: the
    variable was renamed in the summary and the callback was missed. Every test passed because none
    supplied a callback, so the first thing to reach the line was the capture pass on the card,
    which would have died after shard 0 with one shard on disk and the manifest naming it.

    The point of this test is not the arithmetic. It is that a seam nothing calls is a seam nothing
    checks, and the fixture must call it.
    """
    rows = [_row(step=i) for i in range(5)]
    decisions = [
        _decision(step=i, messages_sha256=capture.prompt_digest(r["messages"]))
        for i, r in enumerate(rows)
    ]
    corpus = capture.rows_by_decision(rows)
    target = capture.CaptureTarget(directory=tmp_path, entry="e", identity=_identity(),
                                   decoding="greedy", shard_size=2)

    seen: list[dict] = []
    capture.capture_decisions(decisions=decisions, corpus=corpus, forward=_forward(),
                              prepare=_prepare(), target=target, progress=seen.append)

    # Two full shards of two; the trailing partial shard flushes without a progress event.
    assert [e["shard"] for e in seen] == [0, 1]
    assert [e["captured"] for e in seen] == [2, 4]
    for event in seen:
        assert event["event"] == "shard"
        assert event["requested"] == 5
        assert event["reused"] == 0
        assert event["written_this_pass"] == event["captured"]


def test_the_progress_callback_counts_reused_cells_in_its_running_total(tmp_path) -> None:
    """`captured` is what the request now has, not what this pass happened to write."""
    rows = [_row(step=i) for i in range(5)]
    decisions = [
        _decision(step=i, messages_sha256=capture.prompt_digest(r["messages"]))
        for i, r in enumerate(rows)
    ]
    corpus = capture.rows_by_decision(rows)
    target = capture.CaptureTarget(directory=tmp_path, entry="e", identity=_identity(),
                                   decoding="greedy", shard_size=2)
    capture.capture_decisions(decisions=decisions[:2], corpus=corpus, forward=_forward(),
                              prepare=_prepare(), target=target)

    seen: list[dict] = []
    capture.capture_decisions(decisions=decisions, corpus=corpus, forward=_forward(),
                              prepare=_prepare(), target=target, progress=seen.append)
    assert seen, "a resumed pass with three still to capture must still report a shard"
    assert seen[0]["reused"] == 2
    assert seen[0]["written_this_pass"] == 2
    assert seen[0]["captured"] == 4
