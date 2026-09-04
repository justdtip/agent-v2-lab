from __future__ import annotations

import ast
from pathlib import Path

_PROJECTION_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)

# Briefing §1.7. Shared by the AST pass and the substring pass so the two cannot drift.
_FORBIDDEN_MODEL_CONSTANTS = ("36", "2048", "35", "<|im_end|>", "model.model.layers")

# Characters that may sit next to a banned number without changing what it means: list
# separators, brackets, quotes, whitespace, and the dash of a range spec such as "1-35" —
# a real evasion, since _hard_coded_probe_layer_defaults only sees comma-separated defaults.
# Anything else (digits, letters, ``_``, ``.``, ``:``, ``/``, ``%``, ``\``) marks the number
# as part of a longer token; dates ("2035-09-04") stay protected by their digit neighbours,
# and ":" and "/" stay excluded so timestamps ("12:35:07") and paths ("run/35/x") never fire.
_NUMERIC_TOKEN_SEPARATORS = frozenset(",;=()[]{}'\"| \t\n\r-")

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


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """Identify docstring nodes, whose prose cannot become a runtime model assumption."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def _contains_number_token(text: str, number: str) -> bool:
    """Report a banned number only where it stands as a whole number inside the text."""
    start = 0
    while (index := text.find(number, start)) != -1:
        before = text[index - 1] if index else ""
        after = text[index + len(number)] if index + len(number) < len(text) else ""
        if before in ("", *_NUMERIC_TOKEN_SEPARATORS) and after in (
            "",
            *_NUMERIC_TOKEN_SEPARATORS,
        ):
            return True
        start = index + 1
    return False


def _is_projection_name_list(text: str) -> bool:
    """Recognise a projection name standing as a cell of a comma-separated list.

    Any exact cell counts, mirroring the AST pass's any-member rule, so one unknown cell
    (``"q_proj,k_proj,lm_head"``) cannot launder the list. Prose stays safe on cell shape:
    a comma-free sentence is a single cell, and a prose fragment between commas is never
    *exactly* a projection name.
    """
    cells = [cell.strip() for cell in text.split(",")]
    return len(cells) > 1 and any(cell in _PROJECTION_NAMES for cell in cells)


def _banned_model_substrings(paths: set[Path], forbidden: tuple[str, ...]) -> list[str]:
    """Report banned model constants hidden inside longer string literals.

    The AST pass above only sees a banned constant when the whole literal *is* that constant,
    so ``--layers`` defaulting to ``"6,12,18,24,30,35"`` passed the guard while pinning the
    36-layer 3B model's last block. This pass restores what wiring-map §6 check 2 assumed a
    plain grep would do, under a rule chosen to stay silent on legitimate text:

    * Only string literals are examined, and docstrings are skipped: prose describing a
      measurement ("a few hundred rows against 2048 dimensions") is not a hard-coded constant,
      and a docstring can never be read as a value.
    * A literal that *equals* a banned constant is left to the AST pass, which reports it with
      more context and covers non-string constants too.
    * A banned **number** counts only where both neighbours are list separators, brackets,
      quotes, whitespace, a range dash or the string boundary (`_NUMERIC_TOKEN_SEPARATORS`).
      Any other neighbour means the digits belong to a longer token, which is what keeps
      version strings (``0.35.1``), hex digests (``a35f…``), ANSI colour codes (``\\x1b[35m``),
      timestamps (``12:35:07``), dates (``2035-09-04``, whose digits guard each other), paths
      (``run/35/x``), fractions (``0.35``), longer numbers (``20480``) and identifiers
      (``layer35``) out of the findings. The dash is a separator because a range default such
      as ``"1-35"`` has no comma, so `_hard_coded_probe_layer_defaults` cannot see it.
    * A banned **non-numeric** constant (``<|im_end|>``, ``model.model.layers``) is already
      self-delimiting, so a plain substring match applies.
    * A projection name is reported when it stands as an exact cell of a comma-separated,
      multi-cell literal — any cell counts, as in the AST pass, so an unknown cell cannot
      launder the list. Prose survives on cell shape, not on an all-cells rule: a comma-free
      sentence is a single cell, and a prose fragment between commas is never exactly a
      projection name.
    """
    findings: list[str] = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = _docstring_constant_ids(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue
            text = node.value
            for banned in forbidden:
                if banned == text or banned not in text:
                    continue
                if banned.isdigit() and not _contains_number_token(text, banned):
                    continue
                findings.append(f"{path.name}:{node.lineno}:{banned}")
            if _is_projection_name_list(text):
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
    assert (
        _banned_model_assumptions(_discover_modern_python_sources(root), _FORBIDDEN_MODEL_CONSTANTS)
        == []
    )


def test_banned_model_constants_are_not_hidden_inside_string_literals() -> None:
    """R17: the AST pass misses a banned constant embedded in a longer literal."""
    root = Path(__file__).resolve().parents[1]
    assert (
        _banned_model_substrings(_discover_modern_python_sources(root), _FORBIDDEN_MODEL_CONSTANTS)
        == []
    )


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


def test_banned_substring_scanner_catches_the_layer_list_default(tmp_path) -> None:
    """The blind spot R17 names: the escaped `--layers` default the AST pass cannot see."""
    source = tmp_path / "example.py"
    source.write_text(
        'parser.add_argument("--layers", default="6,12,18,24,30,35")\n',
        encoding="utf-8",
    )

    assert _banned_model_assumptions({source}, _FORBIDDEN_MODEL_CONSTANTS) == []
    assert _banned_model_substrings({source}, _FORBIDDEN_MODEL_CONSTANTS) == ["example.py:1:35"]


def test_banned_substring_scanner_catches_the_dash_range_default(tmp_path) -> None:
    """A range default has no comma, so the layer-defaults guard above cannot see it."""
    source = tmp_path / "example.py"
    source.write_text('parser.add_argument("--layers", default="1-35")\n', encoding="utf-8")

    assert _hard_coded_probe_layer_defaults({source}) == []
    assert _banned_model_substrings({source}, _FORBIDDEN_MODEL_CONSTANTS) == ["example.py:1:35"]


def test_banned_substring_scanner_reports_every_embedded_constant_kind(tmp_path) -> None:
    source = tmp_path / "example.py"
    source.write_text(
        "\n".join(
            (
                'depths = "6,12,18,24,30,36"',
                'width = "hidden=2048"',
                'stop = "note<|im_end|>"',
                'path = "getattr(model.model.layers, name)"',
                'keys = "q_proj,k_proj,v_proj"',
                'padded = "q_proj,k_proj,v_proj,o_proj,lm_head"',
            )
        ),
        encoding="utf-8",
    )

    assert _banned_model_substrings({source}, _FORBIDDEN_MODEL_CONSTANTS) == [
        "example.py:1:36",
        "example.py:2:2048",
        "example.py:3:<|im_end|>",
        "example.py:4:model.model.layers",
        "example.py:5:projection-list",
        "example.py:6:projection-list",
    ]


def test_banned_substring_scanner_ignores_legitimate_text(tmp_path) -> None:
    """Catches the substring rule widening onto versions, digests, dates, prose and words."""
    source = tmp_path / "example.py"
    source.write_text(
        "\n".join(
            (
                '"""A few hundred rows against 2048 dimensions; 36 blocks."""',
                'version = "mlx-lm 0.31.3 needs 0.35.1"',
                'digest = "a35f36c2048deadbeef1234567890abcdef2048ab"',
                'colour = "\\x1b[35m\\x1b[36mwarn\\x1b[0m"',
                'stamp = "2035-09-04T12:35:07"',
                'counts = "20480 tokens, 1350 rows, 2360 steps"',
                'names = "layer35 layer36 x2048y _35"',
                'where = "outputs/probes/run/35/x"',
                'share = "35% of rows"',
                'fractions = "0.35,0.5"',
                'readout = "Blocks whose down_proj/o_proj directions are read out."',
                'listed = "gate_proj is adapted everywhere, and the rest is frozen"',
                'bare = "2048"',
            )
        ),
        encoding="utf-8",
    )

    assert _banned_model_substrings({source}, _FORBIDDEN_MODEL_CONSTANTS) == []
    # The bare literal on the last line is still a violation; the AST pass owns it.
    assert _banned_model_assumptions({source}, _FORBIDDEN_MODEL_CONSTANTS) == ["example.py:13:2048"]


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
