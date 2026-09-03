from __future__ import annotations

from pathlib import Path


def test_banned_model_constants_are_limited_to_approved_or_legacy_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = ("36", "2048", "35", "<|im_end|>", "model.model.layers")
    modern = {
        root / "src/local_llm_lab/probes/capture.py",
        root / "src/local_llm_lab/probes/adapter_delta.py",
        root / "src/local_llm_lab/pipeline/jlens.py",
    }
    offenders = []
    for path in modern:
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == []
