from __future__ import annotations

import ast
from pathlib import Path

_PROJECTION_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


def _banned_literals(paths: set[Path], forbidden: tuple[str, ...]) -> list[str]:
    """Report literal model constants, never incidental identifier substrings."""
    findings: list[str] = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and str(node.value) in forbidden:
                findings.append(f"{path.name}:{node.lineno}:{node.value}")
            if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                names = {
                    item.value
                    for item in node.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
                if len(names & _PROJECTION_NAMES) >= 3:
                    findings.append(f"{path.name}:{node.lineno}:projection-list")
    return findings


def test_banned_model_constants_are_limited_to_approved_or_legacy_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = ("36", "2048", "35", "<|im_end|>", "model.model.layers")
    modern = {
        root / "src/local_llm_lab/probes/capture.py",
        root / "src/local_llm_lab/probes/adapter_delta.py",
        root / "src/local_llm_lab/pipeline/jlens.py",
        root / "research/jspace_sweep.py",
    }
    assert _banned_literals(modern, forbidden) == []


def test_banned_literal_scanner_ignores_identifiers_and_reports_literals(tmp_path) -> None:
    source = tmp_path / "example.py"
    source.write_text(
        'result_2048 = 1\nsize = 2048\nkeys = ("q_proj", "k_proj", "v_proj")\n',
        encoding="utf-8",
    )

    assert _banned_literals({source}, ("2048",)) == [
        "example.py:2:2048",
        "example.py:3:projection-list",
    ]
