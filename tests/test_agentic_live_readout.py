"""Readout uses the full model distribution and byte-checked source residuals."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_lens_upstream import TinyLensModel, make_rows


def test_summary_does_not_confuse_six_winner_with_model_winner():
    from local_llm_lab.pipeline.lens_fitting.agentic_readout import summarize_logits

    scores = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0])
    result = summarize_logits(scores, list(range(6)))
    assert result["top_token_id"] == 6
    assert result["top_margin"] == 15
    assert result["mass"] < 0.001
    assert np.argmax(result["six"]) == 5


def test_greedy_tie_is_lowest_id_not_topk_order(monkeypatch):
    from local_llm_lab.pipeline.lens_fitting.agentic_readout import summarize_logits

    scores = torch.ones(7)
    # topk does not promise stable index ordering for ties (CUDA may choose either).
    monkeypatch.setattr(
        torch,
        "topk",
        lambda *_: SimpleNamespace(values=torch.ones(2), indices=torch.tensor([5, 4])),
    )
    assert summarize_logits(scores, list(range(6)))["top_token_id"] == 0


def test_tool_ids_are_rederived_from_tokenizer_and_capture(tmp_path):
    from tokenizers import Tokenizer, models

    from local_llm_lab.pipeline.lens_fitting.agentic_readout import validated_tool_ids

    names = [f"tool{i}" for i in range(6)]
    tok = Tokenizer(models.WordLevel(dict(zip(names, range(6), strict=True))))
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    manifest = {"tools": names, "tool_first_tokens": dict(zip(names, range(6), strict=True))}
    assert validated_tool_ids(manifest, path, list(range(6))) == list(range(6))
    with pytest.raises(ValueError, match="tool"):
        validated_tool_ids(manifest, path, list(reversed(range(6))))


def test_live_scores_cross_capture_and_full_unembed_boundaries():
    from local_llm_lab.pipeline.lens_fitting.agentic_readout import score_cells
    from local_llm_lab.pipeline.lens_fitting.upstream import load_upstream

    model = TinyLensModel()
    ids = make_rows(1)[0]["ids"]
    up = load_upstream()
    with (
        up.fitting.ActivationRecorder(model.layers, at=[1, 2], start_graph_at=None) as rec,
        torch.no_grad(),
    ):
        h = model(torch.tensor([ids]))
    positions = [20, len(ids) - 2]
    cells = [
        {
            "cell": {"row": 0, "position": site, "token_index": pos, "token_id": ids[pos + 1]},
            "input_ids": ids,
            "prompt_identity": {"task_id": "train-read-0-clean"},
            "residuals": {
                layer: rec.activations[layer - 1][0, pos].detach().numpy().copy()
                for layer in [2, 3]
            },
            "capture_forward": {"output_attentions": False, "logits_to_keep": positions},
        }
        for site, pos in zip(["P_note", "P_act"], positions, strict=True)
    ]

    def outer(*, input_ids, logits_to_keep, output_attentions):
        return SimpleNamespace(logits=model.unembed(model(input_ids))[:, logits_to_keep])

    lens = SimpleNamespace(maps={2: np.eye(6, dtype=np.float32), 3: np.eye(6, dtype=np.float32)})
    rows = score_cells(
        outer, model, cells, {"prose": lens, "agentic": lens}, list(range(6)), [2, 3]
    )
    assert len(rows) == 4
    assert all(r["prose"] == r["agentic"] for r in rows)
    assert rows[0]["model"]["top_token_id"] == model.unembed(h[0, 20]).argmax().item()
    cells[0]["residuals"][2][0] += 1
    with pytest.raises(ValueError, match="capture residual"):
        score_cells(outer, model, cells, {"prose": lens, "agentic": lens}, list(range(6)), [2, 3])


def test_near_unit_mass_is_admitted_by_comparison_reader():
    from local_llm_lab.pipeline.lens_fitting.agentic_compare import _readout
    from local_llm_lab.pipeline.lens_fitting.agentic_readout import summarize_logits

    torch.manual_seed(3)
    scores = torch.cat([torch.randn(6), torch.full((30,), -100.0)])
    tokens = list(range(6))
    result = summarize_logits(scores, tokens)
    assert 0 <= result["mass"] <= 1
    _readout(result, tokens)
