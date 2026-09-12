"""The rank a replay reports, and the tie it used to break by accident.

Method entry 36, 2026-09-12. `clean_rank_of_steered` counts the tokens strictly above the steered
one. It used to be that token's index in a full `argsort`, which for a tie broke in unspecified
order -- so one of two tokens the clean model valued identically was reported as displaced.

The fixture has to CONTAIN a tie. Two rank rules agree on every row where nothing is tied, so a
test built from ordinary rows would pass under either and establish nothing (method entry 35).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class _Tok:
    """Just enough tokenizer for the row builder."""

    def batch_decode(self, rows):
        return [f"<{r[0]}>" for r in rows]


def _chat():
    import steer_chat
    chat = steer_chat.Chat.__new__(steer_chat.Chat)
    chat.tokenizer = _Tok()
    return chat


def _old_rank(step, token):
    """What the code did before: position in a descending argsort."""
    order = step.argsort(descending=True)
    return int((order == token).nonzero()[0])


def test_a_tie_for_first_is_rank_zero_and_the_old_rule_disagreed():
    chat = _chat()
    logits = torch.tensor([[[5.0, 5.0, 5.0, 1.0, 0.0]]])
    step = logits[0, 0].float().log_softmax(-1)

    for token in (0, 1, 2):
        rows = chat._rows_from([chat._score(logits, [token])], [token])
        assert rows[0]["clean_rank_of_steered"] == 0, token
    # and the rule it replaced did not agree, which is what makes this fixture worth having
    assert [_old_rank(step, t) for t in (0, 1, 2)] == [0, 1, 2]


def test_an_untied_row_is_unchanged_by_the_new_rule():
    chat = _chat()
    logits = torch.tensor([[[9.0, 4.0, 1.0, 0.5, 0.0]]])
    step = logits[0, 0].float().log_softmax(-1)
    for token in range(5):
        rows = chat._rows_from([chat._score(logits, [token])], [token])
        assert rows[0]["clean_rank_of_steered"] == _old_rank(step, token) == token


def test_a_flat_tail_counts_only_what_is_strictly_above():
    chat = _chat()
    logits = torch.full((1, 1, 500), -30.0)
    logits[0, 0, :7] = torch.linspace(8.0, 2.0, 7)
    rows = chat._rows_from([chat._score(logits, [400])], [400])
    assert rows[0]["clean_rank_of_steered"] == 7
    # the old rule counted the 493 tokens tied at the floor that happened to sort first
    assert _old_rank(logits[0, 0].float().log_softmax(-1), 400) > 7


def test_the_batched_and_one_at_a_time_paths_agree_on_untied_rows():
    chat = _chat()
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 40) * 3
    tokens = [3, 11, 0, 27, 5, 39]
    batched = chat._rows_from([chat._score(logits, tokens)], tokens)
    singly = []
    for i, t in enumerate(tokens):
        singly += chat._rows_from([chat._score(logits[:, i:i + 1, :], [t])], [t], start=i)
    assert batched == singly


def test_positions_are_contiguous_across_chunks():
    chat = _chat()
    torch.manual_seed(1)
    a, b = torch.randn(1, 3, 20), torch.randn(1, 4, 20)
    tokens = [1, 2, 3, 4, 5, 6, 7]
    rows = chat._rows_from([chat._score(a, tokens[:3]), chat._score(b, tokens[3:])], tokens)
    assert [r["position"] for r in rows] == list(range(7))
    assert [r["steered_token"] for r in rows] == [f"<{t}>" for t in tokens]
