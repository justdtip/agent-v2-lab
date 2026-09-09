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


def _decision(**overrides) -> dict:
    base = {
        "task_id": "test-read-0000-clean", "step": 1, "split": "test", "family": "read",
        "variant": "clean", "difficulty": 2, "recovery": False, "rendered_rows": 1,
        "prompt_sha256": "", "row_ordinals": [0],
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
        "difficulty": 2, "recovery": False, "rendered_rows": 1, "prompt_sha256": "d",
        "token_index": 3, "seq_len": 4, "forward_batch": 1, "anchor_batch": 1,
        "capture_dtype": "native", "checkpoint_sha256": "c", "layers": 2, "d_model": 4,
        "device": "cpu", "decoding": "greedy", "shard": 0, "index_in_shard": 0,
        "basis": "measured-here",
    }
    return {**base, **overrides}


def _forward(layers=2, d_model=4):
    def run(row):
        ids = list(range(len(row["prompt"].split())))or [0]
        return {
            "residuals": torch.zeros(layers + 1, d_model, dtype=torch.bfloat16),
            "seq_len": len(ids), "token_index": len(ids) - 1, "layers": layers,
            "d_model": d_model, "device": "cpu", "dtype": "torch.bfloat16",
        }
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
    decision = _decision(prompt_sha256=capture.prompt_digest(row["messages"]))
    moved = _row(text="something else")
    target = capture.CaptureTarget(directory=tmp_path, entry="e", checkpoint_sha256="c",
                                   decoding="greedy")
    with pytest.raises(capture.ContractViolation, match="re-enumerate rather than capture"):
        capture.capture_decisions(
            decisions=[decision], corpus=capture.rows_by_decision([moved]),
            forward=_forward(), target=target,
        )


def test_a_decision_absent_from_the_corpus_is_refused(tmp_path) -> None:
    target = capture.CaptureTarget(directory=tmp_path, entry="e", checkpoint_sha256="c",
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
        _decision(step=i, prompt_sha256=capture.prompt_digest(r["messages"]))
        for i, r in enumerate(rows)
    ]
    target = capture.CaptureTarget(directory=tmp_path, entry="e", checkpoint_sha256="c0ffee",
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
        assert cell["checkpoint_sha256"] == "c0ffee"
        assert cell["shard_sha256"]
        assert (tmp_path / f"residuals-{cell['shard']:05d}.pt").exists()
    # Three shards at size two, and the residuals keep their native dtype rather than promoting.
    assert sorted({c["shard"] for c in lines}) == [0, 1, 2]
    assert torch.load(tmp_path / "residuals-00000.pt").dtype is torch.bfloat16


def test_a_second_run_resumes_from_the_manifest_and_does_not_recapture(tmp_path) -> None:
    """An interrupted capture pass costs the shard it was writing, never the pass."""
    rows = [_row(step=i) for i in range(4)]
    decisions = [
        _decision(step=i, prompt_sha256=capture.prompt_digest(r["messages"]))
        for i, r in enumerate(rows)
    ]
    corpus = capture.rows_by_decision(rows)
    target = capture.CaptureTarget(directory=tmp_path, entry="e", checkpoint_sha256="c",
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
        _decision(step=i, prompt_sha256=capture.prompt_digest(r["messages"]))
        for i, r in enumerate(rows)
    ]
    corpus = capture.rows_by_decision(rows)
    target = capture.CaptureTarget(directory=tmp_path, entry="e", checkpoint_sha256="c",
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
        _decision(step=i, prompt_sha256=capture.prompt_digest(r["messages"]))
        for i, r in enumerate(rows)
    ]
    corpus = capture.rows_by_decision(rows)
    target = capture.CaptureTarget(directory=tmp_path, entry="e", checkpoint_sha256="c",
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
