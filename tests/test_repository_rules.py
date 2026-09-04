from __future__ import annotations

import ast
from pathlib import Path

_PROJECTION_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)

_SANCTIONED_SOURCE_PATHS = frozenset(
    {
        "src/local_llm_lab/arch.py",
        "src/local_llm_lab/models.py",
    }
)
_LEGACY_SOURCE_PATHS = frozenset(
    {
        # Pre-SPEC-001 model-specific training/data/task paths, owned by later migrations.
        "src/local_llm_lab/agent_tasks.py",
        "src/local_llm_lab/depth_expansion.py",
        "src/local_llm_lab/generate_complex_data.py",
        "src/local_llm_lab/pipeline/protocol.py",
        "src/local_llm_lab/pipeline/tasks.py",
        "src/local_llm_lab/pipeline/transcript.py",
        "src/local_llm_lab/train_expanded.py",
        "src/local_llm_lab/train_grpo.py",
    }
)


def _discover_modern_python_sources(root: Path) -> set[Path]:
    """Find source files covered by the modern-model-agnostic rule."""
    candidates = set((root / "src/local_llm_lab").rglob("*.py"))
    candidates.update((root / "research").glob("*.py"))
    excluded = _SANCTIONED_SOURCE_PATHS | _LEGACY_SOURCE_PATHS
    return {path for path in candidates if path.relative_to(root).as_posix() not in excluded}


def _attribute_chain(node: ast.AST) -> str | None:
    """Return a dotted attribute chain, without treating source text as a match."""
    names: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        names.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        names.append(current.id)
        return ".".join(reversed(names))
    return None


def _constant_string(node: ast.AST) -> str | None:
    """Evaluate a string literal or a string-only constant concatenation."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_string(node.left)
        right = _constant_string(node.right)
        return left + right if left is not None and right is not None else None
    return None


def _banned_model_assumptions(paths: set[Path], forbidden: tuple[str, ...]) -> list[str]:
    """Report model assumptions while ignoring incidental identifier substrings."""
    findings: list[str] = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and str(node.value) in forbidden:
                findings.append(f"{path.name}:{node.lineno}:{node.value}")
            if isinstance(node, ast.Attribute):
                chain = _attribute_chain(node)
                if chain in forbidden:
                    findings.append(f"{path.name}:{node.lineno}:{chain}")
            if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                names = {
                    value
                    for item in node.elts
                    if (value := _constant_string(item)) is not None
                }
                if names & _PROJECTION_NAMES:
                    findings.append(f"{path.name}:{node.lineno}:projection-list")
    return findings


def _hard_coded_probe_layer_defaults(paths: set[Path]) -> list[str]:
    """Report literal numeric defaults for probe and J-lens layer-list arguments."""
    findings: list[str] = []
    layer_flags = {"--layers", "--readout-layers"}
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            flag = _constant_string(node.args[0])
            if flag not in layer_flags:
                continue
            default = next(
                (
                    _constant_string(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg == "default"
                ),
                None,
            )
            if default is None or "," not in default:
                continue
            cells = [cell.strip() for cell in default.split(",")]
            try:
                numeric = bool(cells) and all(cell and float(cell) == float(cell) for cell in cells)
            except ValueError:
                numeric = False
            if numeric:
                findings.append(f"{path.name}:{node.lineno}:{flag}={default}")
    return findings


def test_banned_model_constants_are_limited_to_approved_or_legacy_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = ("36", "2048", "35", "<|im_end|>", "model.model.layers")
    assert _banned_model_assumptions(_discover_modern_python_sources(root), forbidden) == []


def test_banned_model_scanner_discovers_repository_wide_modern_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = _discover_modern_python_sources(root)

    assert root / "src/local_llm_lab/probes/capture.py" in paths
    assert root / "src/local_llm_lab/pipeline/jlens.py" in paths
    assert root / "research/jspace_sweep.py" in paths
    assert root / "src/local_llm_lab/arch.py" not in paths


def test_banned_model_scanner_reports_ast_assumptions_and_ignores_lookalikes(tmp_path) -> None:
    source = tmp_path / "example.py"
    source.write_text(
        "\n".join(
            (
                "result_2048 = 1",
                "size = 2048",
                "layers = model.model.layers",
                'keys = ("q_proj",)',
                'hidden_keys = ("down" + "_proj",)',
                'lookalike = "projection_q_proj_suffix"',
                'plain = "model layers"',
            )
        ),
        encoding="utf-8",
    )

    assert _banned_model_assumptions({source}, ("2048", "model.model.layers")) == [
        "example.py:2:2048",
        "example.py:3:model.model.layers",
        "example.py:4:projection-list",
        "example.py:5:projection-list",
    ]


def test_probe_layer_defaults_come_from_runtime_registry_metadata() -> None:
    """Catches model-specific numeric layer lists returning to probe or J-lens parsers."""
    root = Path(__file__).resolve().parents[1]

    assert _hard_coded_probe_layer_defaults(_discover_modern_python_sources(root)) == []


def test_probe_layer_default_scanner_ignores_nonliteral_and_unrelated_defaults(tmp_path) -> None:
    """Catches the AST rule widening beyond literal layer-list parser defaults."""
    source = tmp_path / "example.py"
    source.write_text(
        "\n".join(
            (
                'parser.add_argument("--layers", default="2,4,6")',
                'parser.add_argument("--readout-layers", default="0.5,1.0")',
                'parser.add_argument("--layers", default=None)',
                'parser.add_argument("--other", default="2,4,6")',
                'computed = "2,4,6"',
                'parser.add_argument("--layers", default=computed)',
            )
        ),
        encoding="utf-8",
    )

    assert _hard_coded_probe_layer_defaults({source}) == [
        "example.py:1:--layers=2,4,6",
        "example.py:2:--readout-layers=0.5,1.0",
    ]
