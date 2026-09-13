"""The function that assigns every label in the experiment had no test at all.

Two defects lived here undetected through a full 13-minute training run and a 995-second
evaluation: `D_negative_extra` injected the genuine concept vector and supervised "NO. I do not
detect an injected thought", under a prompt drawn from the same eighteen the positives use; and
`D_mismatch` answered "is the injected thought about X?" by denying detection outright rather than
denying X. Neither was visible in the loss, which fell to 0.14.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_introspect_data import (  # noqa: E402
    HELD_OUT_DAMAGE, LADDER, MISMATCH_TARGET, NEGATIVE_TARGET, POSITIVE_TARGET,
)
from local_llm_lab.introspect.vocabulary import all_concepts, split  # noqa: E402
from train_introspect import make_rows  # noqa: E402

LAYERS = [2, 5]


@pytest.fixture
def plan():
    train_words = sorted(split(seed=0)[0])
    words = sorted(all_concepts())
    scales = {w: {str(l): {str(k): {"scale": 0.5 + i * 0.01, "measured": k}
                           for i, k in enumerate(LADDER + HELD_OUT_DAMAGE)}
                  for l in LAYERS} for w in words}
    bank = {l: torch.randn(len(words), 16) for l in LAYERS}
    rows = make_rows(bank, scales, words, train_words, LAYERS, random.Random(0), 2000)
    return rows, {w: i for i, w in enumerate(words)}


def test_no_row_injects_a_real_concept_and_denies_detection(plan):
    """The defect, stated as the property it violates."""
    rows, _index = plan
    bad = [r for r in rows if r["bank_index"] >= 0 and r["target"] == NEGATIVE_TARGET]
    assert not bad, f"{len(bad)} rows inject a concept and deny detection: {bad[:2]}"


def test_the_negative_classes_use_sentinels_and_carry_their_source(plan):
    rows, index = plan
    for kind, sentinel in (("D_noise", -2), ("D_common", -3), ("D_negative_extra", -4)):
        at = [r for r in rows if r["cls"] == kind]
        assert at, kind
        assert {r["bank_index"] for r in at} == {sentinel}, kind
        # the scale was calibrated on a concept, so the row has to say which one
        assert all(isinstance(r["source_index"], int) and r["source_index"] >= 0 for r in at)
        assert all(r["measured"] != r["measured"] for r in at), \
            f"{kind} stores the concept's achieved damage as its own"


def test_a_negative_prompt_is_not_separable_from_a_positive_by_its_text(plan):
    """D_negative_extra shares all eighteen prompt strings with D_positive, which is the point:
    it must be a different VECTOR, not a different question."""
    rows, _index = plan
    pos = {r["prompt"] for r in rows if r["cls"] == "D_positive"}
    neg = {r["prompt"] for r in rows if r["cls"] == "D_negative_extra"}
    assert neg <= pos and len(neg) > 1


def test_mismatch_rows_name_a_distractor_and_are_positives_until_rendered(plan):
    rows, _index = plan
    at = [r for r in rows if r["cls"] == "D_mismatch"]
    assert at
    for r in at:
        assert r["distractor"] != r["concept"]
        assert r["bank_index"] >= 0
        rendered = MISMATCH_TARGET.format(name=r["distractor"])
        assert rendered != NEGATIVE_TARGET
        assert r["distractor"] in rendered
        # A concept IS present on these rows, and the evaluation counts the first token. Opening
        # with NO would train 1,320 denials on the token the headline is read from.
        assert rendered.startswith("YES.")
        assert r["concept"] not in rendered, "the mismatch row must not leak the real concept"


def test_positives_name_the_concept_that_was_actually_injected(plan):
    rows, index = plan
    for r in rows:
        if r["cls"] in ("D_positive", "D_heldstrength"):
            assert r["target"] == POSITIVE_TARGET.format(name=r["concept"])
            assert r["bank_index"] == index[r["concept"]]


def test_held_out_concepts_never_reach_the_training_rows(plan):
    rows, _index = plan
    held = set(split(seed=0)[1])
    used = {r["concept"] for r in rows if r["concept"]} | \
           {r["distractor"] for r in rows if r.get("distractor")}
    assert not (used & held), sorted(used & held)[:5]
