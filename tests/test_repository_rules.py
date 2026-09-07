from __future__ import annotations

import ast
import os
import sys
import sysconfig
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import (
    PRE_EXISTING_FORK_SITES,
    SPAWN_SANCTIONED_PATHS,
    TESTS_THAT_CAN_REACH_MLX,
    TESTS_THAT_LOAD_MLX,
)

from local_llm_lab import spawn

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


# ------------------------------------------------------------------ arm configuration files
#
# Briefing §1.7 bans hard-coded model constants "outside `configs/models/` and
# `src/local_llm_lab/arch.py`", but every scanner above walks Python only, so `configs/*.yaml`
# sat outside the guard's universe entirely. That is how `train.num_layers: 36` — the 3B
# layer count, never read, since `cli.py:450` writes `resolved.num_layers` from the
# architecture view — survived in four arm configs. This pass closes the hole.

# The registry: the one directory where a model's own constants belong.
_SANCTIONED_CONFIG_DIRS = ("configs/models",)

# Pre-SPEC-001 recipe files consumed by mlx-lm/mlx-tune directly (`model:` is an HF id,
# `fine_tune_type`, `mask_prompt`), not by `pipeline/cli.py`. Their top-level `num_layers` is
# the library's "adapt the last N blocks" argument — a recipe choice, not a claim about the
# architecture — so the architecture-key rule does not apply to them.
_LEGACY_CONFIG_PATHS = frozenset(
    {
        "configs/agent_lora.yaml",
        "configs/lora.yaml",
    }
)

# Owned by the uncommitted R32 stage-2 slice at the Chief's gate (`gated_delta_mode` lands in
# `agent_v2b_qwen35_4b.yaml`, and `test_run_d_configs_are_literal_pairwise_recipes` compares
# the B pair literally, so the two must lose the dead key together). Delete both entries with
# that slice; the sweep below then covers all six arm configs.
_DEFERRED_CONFIG_PATHS = frozenset(
    {
        "configs/agent_v2b.yaml",
        "configs/agent_v2b_qwen35_4b.yaml",
    }
)

# Key names that state the model's architecture. An arm config may not carry them at any
# value: the architecture comes from the registry through `ArchitectureView`, so a config that
# names one is duplicating a fact it does not own — whether or not the number is still 36.
_MODEL_ARCHITECTURE_KEYS = frozenset(
    {
        "num_layers",
        "n_layers",
        "num_hidden_layers",
        "hidden_size",
        "head_dim",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
    }
)

# Blocks whose values are task counts by definition (SPEC-003 §3's split table). A count that
# happens to equal a banned number — `tasks.valid: 36` in the run A/B/C configs — is a dataset
# size, and re-pinning run C's generator hashes to dodge a scanner would be the real defect.
_TASK_COUNT_BLOCKS = frozenset({"tasks", "splits"})


def _discover_arm_config_sources(root: Path) -> set[Path]:
    """Find the run-configuration YAML covered by the model-agnostic rule."""
    candidates = set((root / "configs").rglob("*.yaml"))
    excluded = _DEFERRED_CONFIG_PATHS | _LEGACY_CONFIG_PATHS
    return {
        path
        for path in candidates
        if (relative := path.relative_to(root).as_posix()) not in excluded
        and not relative.startswith(tuple(f"{name}/" for name in _SANCTIONED_CONFIG_DIRS))
    }


def _config_scalars(
    node: Any, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield every scalar of a parsed YAML document with the key path that reaches it."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _config_scalars(value, (*prefix, str(key)))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _config_scalars(value, (*prefix, str(index)))
    else:
        yield prefix, node


def _banned_config_model_assumptions(paths: set[Path], forbidden: tuple[str, ...]) -> list[str]:
    """Report model assumptions written into run configuration.

    Two rules, both keyed on what a value *means* rather than on where it sits in the file:

    * A key from `_MODEL_ARCHITECTURE_KEYS` is a finding at any value, because the fact it
      states belongs to the registry. This is the rule that cannot be dodged by editing 36 to
      32 and leaving a second, silently stale source of truth behind.
    * A banned constant appearing as a scalar value is a finding, under the same whole-number
      rule the Python substring pass uses (`_contains_number_token`), so `qwen35-4b` and
      `data/agent_v2b-qwen35-4b` stay quiet while `"6,12,18,24,30,36"` does not. Values inside
      a `tasks:`/`splits:` table are exempt: those are task counts.

    Comments are not read, exactly as docstrings are skipped in the Python pass: a comment can
    never be loaded as a value.
    """
    findings: list[str] = []
    for path in sorted(paths):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key_path, value in _config_scalars(document):
            dotted = ".".join(key_path)
            for part in key_path:
                if part in _MODEL_ARCHITECTURE_KEYS:
                    findings.append(f"{path.name}:{dotted}:architecture-key")
                    break
            if value is None or isinstance(value, bool):
                continue
            if key_path and key_path[0] in _TASK_COUNT_BLOCKS:
                continue
            text = str(value)
            for banned in forbidden:
                if banned == text:
                    findings.append(f"{path.name}:{dotted}:{banned}")
                elif banned not in text:
                    continue
                elif banned.isdigit():
                    if _contains_number_token(text, banned):
                        findings.append(f"{path.name}:{dotted}:{banned}")
                else:
                    findings.append(f"{path.name}:{dotted}:{banned}")
    return findings


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


def test_banned_model_constants_are_absent_from_run_configuration() -> None:
    """Briefing §1.7 applied to `configs/*.yaml`, which no Python scanner can reach."""
    root = Path(__file__).resolve().parents[1]

    assert (
        _banned_config_model_assumptions(
            _discover_arm_config_sources(root), _FORBIDDEN_MODEL_CONSTANTS
        )
        == []
    )


def test_config_scanner_sweeps_every_arm_config_and_exempts_only_the_registry() -> None:
    """Catch an arm config joining the repository outside the sweep, or a stale exemption."""
    root = Path(__file__).resolve().parents[1]
    paths = _discover_arm_config_sources(root)

    assert root / "configs/agent_v2d.yaml" in paths
    assert root / "configs/agent_v2d_qwen35_4b.yaml" in paths
    assert root / "configs/agent_v2.yaml" in paths
    assert root / "configs/agent_v2c.yaml" in paths
    assert root / "configs/models/qwen35-4b.yaml" not in paths
    assert root / "configs/agent_lora.yaml" not in paths
    # An exemption names a real file, so a rename cannot leave one silently in force.
    for relative in _DEFERRED_CONFIG_PATHS | _LEGACY_CONFIG_PATHS:
        assert (root / relative).is_file(), relative


def test_config_scanner_reports_architecture_keys_and_embedded_constants(tmp_path) -> None:
    """The class this pass exists for: a layer count written into an arm config."""
    source = tmp_path / "arm.yaml"
    source.write_text(
        "\n".join(
            (
                "train:",
                "  num_layers: 36",
                "  iters: 400",
                "probes:",
                "  layers: '6,12,18,24,30,36'",
                "protocol:",
                "  stop: 'note<|im_end|>'",
                "capture:",
                "  hidden: 2048",
            )
        ),
        encoding="utf-8",
    )

    assert _banned_config_model_assumptions({source}, _FORBIDDEN_MODEL_CONSTANTS) == [
        "arm.yaml:train.num_layers:architecture-key",
        "arm.yaml:train.num_layers:36",
        "arm.yaml:probes.layers:36",
        "arm.yaml:protocol.stop:<|im_end|>",
        "arm.yaml:capture.hidden:2048",
    ]


def test_config_scanner_keeps_a_renamed_layer_count_from_laundering_the_rule(tmp_path) -> None:
    """Editing 36 to the 4B's own count leaves the same duplicated fact behind."""
    source = tmp_path / "arm.yaml"
    source.write_text("train:\n  num_layers: 32\n", encoding="utf-8")

    assert _banned_config_model_assumptions({source}, _FORBIDDEN_MODEL_CONSTANTS) == [
        "arm.yaml:train.num_layers:architecture-key",
    ]


def test_config_scanner_ignores_task_counts_model_names_and_comments(tmp_path) -> None:
    """Catches the config rule widening onto legitimate run configuration."""
    source = tmp_path / "arm.yaml"
    source.write_text(
        "\n".join(
            (
                "# Adapt two-thirds of the 36 transformer layers.",
                "model: qwen35-4b",
                "data: 'data/agent_v2b-qwen35-4b'",
                "output: 'outputs/agent-v2b-qwen35-4b'",
                "seed: 20260902",
                "tasks:",
                "  valid: 36",
                "splits:",
                "  valid: {count: 36, difficulty: 1}",
                "train:",
                "  max_seq_length: 2688",
                "  iters: 400",
                "eval:",
                "  limit: 180",
                "criteria:",
                "  criterion: 'total >= 150/180 with McNemar p < 0.05'",
            )
        ),
        encoding="utf-8",
    )

    assert _banned_config_model_assumptions({source}, _FORBIDDEN_MODEL_CONSTANTS) == []


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


# ------------------------------------------------------------- the model-run lock (issue 83)
#
# The Director's one constraint on a research run is that two model loads never happen at once
# on this 24 GiB machine, and `runlock` is the whole mechanism. Wiring every entry point by hand
# is a guard the next entry point forgets, so the rule is structural instead: on the v2 surface
# there is exactly one door to `mlx_lm`'s loader, and it is the one that takes the lock first.

#: The v2 surface. Legacy top-level scripts are excluded on purpose, and the exclusion is not a
#: convenience: briefing §3 keeps `compare_agent.py`, `evaluate_agent.py`, `chat_replay.py`,
#: `compare_chat.py`, `depth_expansion.py`, `evaluate_complex_agent.py` and `train_grpo.py` "for
#: reference and not on the v2 path", the runs the concurrency clause names are all v2 entry
#: points (`agent-pipeline`, `agent-v2-*`), and rewriting modules the briefing says not to
#: refactor would put an untested change next to the mechanism that gates every run. They are
#: reported in the issue-83 hand-off as observed-not-wired rather than silently swept in.
_RUN_LOCK_SURFACE = ("src/local_llm_lab/pipeline", "src/local_llm_lab/probes")

#: The module that owns the lock, and the wrapper the surface must call instead.
_RUN_LOCK_MODULE = "src/local_llm_lab/runlock.py"
_RUN_LOCK_WRAPPER = "load_weights"

#: Weight loaders that are not `mlx_lm.load`. `FastLanguageModel.from_pretrained` reaches
#: `mlx_lm.load` from inside mlx-tune, where no wrapper of ours can sit, so a module that calls
#: one of these must at least name the lock module itself.
_INDIRECT_WEIGHT_LOADERS = frozenset({"FastLanguageModel.from_pretrained"})


def _discover_run_lock_surface(root: Path) -> set[Path]:
    """Every Python file on the v2 surface the single-door rule covers."""
    found: set[Path] = set()
    for relative in _RUN_LOCK_SURFACE:
        found.update((root / relative).rglob("*.py"))
    return found


def _direct_weight_loader_uses(paths: set[Path]) -> list[str]:
    """Report code that reaches `mlx_lm`'s loader without the run lock.

    Two forms, because both have appeared in this tree: `from mlx_lm import load` (the import
    every call site used before the lock existed) and a dotted `mlx_lm.load(...)`. Other
    `mlx_lm` imports are untouched — `stream_generate`, `sample_utils.make_sampler` and the
    version check in `cli.py` load no weights, and banning the package outright would make the
    rule noisy enough to be waived.
    """
    findings: list[str] = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # `ast.walk` is breadth-first, so a file's findings are sorted by line here: a report
        # that jumps around the file is read as two separate defects rather than one.
        in_file: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {"mlx_lm", "mlx_lm.utils"}:
                for alias in node.names:
                    if alias.name == "load":
                        in_file.append((node.lineno, f"from {node.module} import load"))
            if isinstance(node, ast.Attribute) and (chain := _attribute_chain(node)) in {
                "mlx_lm.load",
                "mlx_lm.utils.load",
            }:
                in_file.append((node.lineno, chain))
        findings.extend(f"{path.name}:{line}:{what}" for line, what in sorted(in_file))
    return findings


def _unlocked_indirect_loaders(paths: set[Path]) -> list[str]:
    """Report a module that loads weights through a library wrapper without naming the lock."""
    findings: list[str] = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = [
            (node.lineno, chain)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (chain := _attribute_chain(node.func)) in _INDIRECT_WEIGHT_LOADERS
        ]
        if not calls:
            continue
        imports_lock = any(
            isinstance(node, ast.ImportFrom) and node.module == "local_llm_lab.runlock"
            for node in ast.walk(tree)
        )
        if imports_lock:
            continue
        findings.extend(f"{path.name}:{line}:{chain}" for line, chain in calls)
    return findings


def test_the_v2_surface_reaches_model_weights_only_through_the_run_lock() -> None:
    """Issue 83: one door to `mlx_lm.load`, and it takes the model-run lock before it opens.

    This is the rule that makes the lock a mechanism rather than a helper callers remember.
    `runlock.load_weights` acquires before the import, so a refusal costs no weights; every
    stage and probe reaches weights through `evaluate.load_policy` or `cli._load_training_base`,
    and both of those now go through the wrapper.

    Legacy top-level scripts are out of scope — see `_RUN_LOCK_SURFACE` for why that exclusion
    is deliberate and not a convenience.
    """
    root = Path(__file__).resolve().parents[1]

    assert _direct_weight_loader_uses(_discover_run_lock_surface(root)) == []


def test_the_run_lock_surface_covers_every_stage_and_probe() -> None:
    """Catches a new pipeline or probe module joining the tree outside the rule's sweep."""
    root = Path(__file__).resolve().parents[1]
    paths = _discover_run_lock_surface(root)

    assert root / "src/local_llm_lab/pipeline/evaluate.py" in paths
    assert root / "src/local_llm_lab/pipeline/cli.py" in paths
    assert root / "src/local_llm_lab/pipeline/prefer.py" in paths
    assert root / "src/local_llm_lab/probes/patch.py" in paths
    assert root / "src/local_llm_lab/probes/state_probe.py" in paths
    # The wrapper's own module is not on the surface: it is the door, so it holds the import.
    assert root / _RUN_LOCK_MODULE not in paths


def test_the_run_lock_wrapper_is_the_module_that_holds_the_import() -> None:
    """The single door exists, imports the loader, and takes the lock before it does."""
    root = Path(__file__).resolve().parents[1]
    source = (root / _RUN_LOCK_MODULE).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=_RUN_LOCK_MODULE)

    wrapper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == _RUN_LOCK_WRAPPER
    )
    body = list(ast.walk(wrapper))
    acquires = [
        node.lineno
        for node in body
        if isinstance(node, ast.Call) and _attribute_chain(node.func) == "hold_model_run_lock"
    ]
    imports = [
        node.lineno
        for node in body
        if isinstance(node, ast.ImportFrom) and node.module == "mlx_lm"
    ]

    assert acquires and imports, "the wrapper must both acquire and import the loader"
    # Order matters: a refusal must cost no weights and no download.
    assert min(acquires) < min(imports)


def test_indirect_weight_loaders_on_the_surface_name_the_lock() -> None:
    """`prefer.py` reaches `mlx_lm.load` from inside mlx-tune; the AST rule cannot see it."""
    root = Path(__file__).resolve().parents[1]

    assert _unlocked_indirect_loaders(_discover_run_lock_surface(root)) == []


def test_direct_loader_scanner_reports_both_forms_and_ignores_the_rest(tmp_path) -> None:
    """Catches the rule widening to every `mlx_lm` import, or narrowing to only the `from` form."""
    source = tmp_path / "example.py"
    source.write_text(
        "\n".join(
            (
                "from mlx_lm import load",
                "from mlx_lm import stream_generate",
                "from mlx_lm.sample_utils import make_sampler",
                "import mlx_lm",
                "model, tokenizer = mlx_lm.load(hf_id)",
                "from mlx_lm import generate, load",
            )
        ),
        encoding="utf-8",
    )

    assert _direct_weight_loader_uses({source}) == [
        "example.py:1:from mlx_lm import load",
        "example.py:5:mlx_lm.load",
        "example.py:6:from mlx_lm import load",
    ]


def test_indirect_loader_scanner_fires_only_without_the_lock_import(tmp_path) -> None:
    """Catches the indirect rule passing a module that dropped its lock import."""
    unlocked = tmp_path / "unlocked.py"
    unlocked.write_text(
        "model, tokenizer = FastLanguageModel.from_pretrained(name)\n", encoding="utf-8"
    )
    locked = tmp_path / "locked.py"
    locked.write_text(
        "\n".join(
            (
                "from local_llm_lab.runlock import hold_model_run_lock",
                "hold_model_run_lock()",
                "model, tokenizer = FastLanguageModel.from_pretrained(name)",
            )
        ),
        encoding="utf-8",
    )

    assert _unlocked_indirect_loaders({unlocked, locked}) == [
        "unlocked.py:1:FastLanguageModel.from_pretrained"
    ]


# ---------------------------------------------------------- starting processes (issue 83)
#
# Forking is not dangerous in itself. Forking is dangerous *in this repository's interpreters*,
# where MLX has been imported and Metal is up: the parent's fork handler can meet a lock the
# Metal driver's memory-pool-decay thread holds, and libplatform aborts the process rather than
# raising. It killed a full suite run on 2026-09-05 (macOS crash report
# `Python-2026-09-05-202845.ips`), from a `subprocess.Popen` in a lock fixture.
#
# `local_llm_lab.spawn` is therefore the only sanctioned way to start a child: it resolves the
# program to an absolute path, sets `close_fds=False`, and refuses `cwd`, `preexec_fn`,
# `start_new_session` and the rest — which together are exactly CPython's conditions for taking
# `posix_spawn` instead of `fork`. See that module's docstring for the mechanism.

#: Dotted call chains that start a process.
_PROCESS_STARTERS = frozenset(
    {
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.getoutput",
        "subprocess.getstatusoutput",
        "os.system",
        "os.fork",
        "os.forkpty",
        "os.popen",
        "os.posix_spawn",
        "os.posix_spawnp",
        "multiprocessing.Process",
        "multiprocessing.Pool",
    }
)
#: Prefixes covering the `os.spawn*` and `os.exec*` families without listing eighteen names.
_PROCESS_STARTER_PREFIXES = ("os.spawn", "os.exec")

# The two exemption sets live in `conftest.py` and are imported here, because the runtime fork
# guard there and this static rule must never disagree about which files may fork. One list,
# two enforcers: a site that leaves the list is covered by both at once.
#
# `PRE_EXISTING_FORK_SITES` predated the helper. It was **reported, not fixed** in the issue-83
# slice, on the ground that seven untested edits next to the mechanism gating every
# model-loading run was the wrong trade, and a guard failing 114 tests on its first run is the
# shape of a guard people switch off. Issue 84 then emptied it, `provenance.py` first because it
# was the only entry that forked with a model already resident. The list is now empty and the
# rule below refuses additions to it.


def _discover_spawn_surface(root: Path) -> set[Path]:
    """Every Python file the single-spawn-helper rule covers."""
    found = set((root / "src/local_llm_lab").rglob("*.py"))
    found.update((root / "tests").rglob("*.py"))
    excluded = SPAWN_SANCTIONED_PATHS | PRE_EXISTING_FORK_SITES
    return {path for path in found if path.relative_to(root).as_posix() not in excluded}


def _is_process_starter(chain: str) -> bool:
    return chain in _PROCESS_STARTERS or chain.startswith(_PROCESS_STARTER_PREFIXES)


def _direct_process_starts(paths: set[Path], root: Path) -> list[str]:
    """Report a process started without going through `local_llm_lab.spawn`.

    Two call shapes, because a rule that only knows one is a rule with a documented way
    around it: the dotted `subprocess.run(...)`, and a bare `run(...)` after
    `from subprocess import run`. The bare form is resolved through the file's own imports, so
    `spawn.run` and any unrelated `run` method stay quiet -- only a name actually bound to one
    of the starter functions counts.
    """
    findings: list[str] = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bound: dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module.split(".")[0] not in {"subprocess", "os", "multiprocessing"}:
                continue
            for alias in node.names:
                chain = f"{node.module}.{alias.name}"
                if _is_process_starter(chain):
                    bound[alias.asname or alias.name] = chain
        in_file: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                if (chain := bound.get(node.func.id)) is not None:
                    in_file.append((node.lineno, chain))
                continue
            chain = _attribute_chain(node.func)
            if chain is not None and _is_process_starter(chain):
                in_file.append((node.lineno, chain))
        relative = path.relative_to(root).as_posix()
        findings.extend(f"{relative}:{line}:{chain}" for line, chain in sorted(in_file))
    return findings


def test_the_fork_exemption_list_is_empty_and_stays_empty() -> None:
    """Issue 84's acceptance: no file is exempt from the spawn helper any more.

    Stated as its own test rather than left implicit in an empty set, because the failure this
    guards against is not a fork -- it is somebody adding a name to
    ``PRE_EXISTING_FORK_SITES`` to make the rule above go green. That edit makes one file
    invisible to *both* enforcers at once, the static rule here and the runtime guard in
    ``conftest``, which is exactly how the seven accumulated. The list earned its exemptions by
    predating the helper; nothing can earn one now.
    """
    assert frozenset() == PRE_EXISTING_FORK_SITES, (
        "the fork exemption list is closed. A file that needs to start a process goes through "
        "local_llm_lab.spawn; if it cannot, that is a finding about the file, not an entry "
        f"here. Added: {sorted(PRE_EXISTING_FORK_SITES)}"
    )


def test_processes_are_started_only_through_the_spawn_helper() -> None:
    """Issue 83: a fork in an interpreter with Metal up aborts, so nothing here may fork.

    The rule is structural for the same reason the run-lock rule is: remembering to pass four
    keyword arguments correctly at every call site is a discipline the next call site skips,
    and the failure it causes is a kill in the middle of an unrelated test rather than an
    error at the offending line.
    """
    root = Path(__file__).resolve().parents[1]

    assert _direct_process_starts(_discover_spawn_surface(root), root) == []


def test_the_spawn_exemptions_are_all_still_load_bearing() -> None:
    """A stale exemption is a hole: a fixed site must leave this list, not sit in it.

    Every exempted path must still exist and still start a process directly. When one of the
    pre-existing sites is migrated to the helper, this test fails until its entry is removed.
    """
    root = Path(__file__).resolve().parents[1]

    for relative in sorted(PRE_EXISTING_FORK_SITES | SPAWN_SANCTIONED_PATHS):
        path = root / relative
        assert path.is_file(), f"{relative} is exempted but does not exist"
        assert _direct_process_starts({path}, root), (
            f"{relative} no longer starts a process directly; drop its exemption"
        )


def test_the_spawn_surface_covers_the_lock_and_its_fixtures() -> None:
    """Catches the sweep missing the files this slice actually added."""
    root = Path(__file__).resolve().parents[1]
    paths = _discover_spawn_surface(root)

    assert root / "src/local_llm_lab/runlock.py" in paths
    assert root / "tests/test_runlock.py" in paths
    assert root / "tests/conftest.py" in paths
    assert root / "src/local_llm_lab/spawn.py" not in paths


def test_the_process_start_scanner_reports_the_families_and_ignores_lookalikes(tmp_path) -> None:
    """Catches the rule narrowing to `subprocess.run`, or widening to any `.run(` call."""
    source = tmp_path / "example.py"
    source.write_text(
        "\n".join(
            (
                "subprocess.run(argv)",
                "subprocess.Popen(argv)",
                "os.execvp(program, argv)",
                "os.spawnv(mode, program, argv)",
                "os.fork()",
                "multiprocessing.Process(target=work)",
                "from subprocess import Popen as launch",
                "launch(argv)",
                "from subprocess import run",
                "run(argv)",
                "from local_llm_lab.spawn import run as safe_run",
                "safe_run(argv)",
                "spawn.run(argv)",
                "runner.run(argv)",
                "trainer.Popen(argv)",
                "os.path.join(a, b)",
            )
        ),
        encoding="utf-8",
    )

    assert _direct_process_starts({source}, tmp_path) == [
        "example.py:1:subprocess.run",
        "example.py:2:subprocess.Popen",
        "example.py:3:os.execvp",
        "example.py:4:os.spawnv",
        "example.py:5:os.fork",
        "example.py:6:multiprocessing.Process",
        # The bare forms: `from subprocess import run` then `run(argv)` passed the dotted-only
        # scanner, which made the rule opt-out by import style.
        "example.py:8:subprocess.Popen",
        "example.py:10:subprocess.run",
    ]


# --- The MLX import closure (heartbeat 2026-09-07 18:05 and 18:20) -------------------------
#
# Two sessions ran pytest beside a held model-run lock on the same afternoon, each having
# checked something that did not answer the question. The rule that followed -- grep the named
# test files for an mlx import and run only the empty ones -- is not sufficient, because pytest
# imports each file's whole closure. `test_preflight.py` imports no mlx and loads it through
# `local_llm_lab.training.gated_delta_chunked`; `test_tuner_data.py` imports `mlx_lm`, whose
# package `__init__` imports `mlx.core` through `mlx_lm.utils`. A grep clears both.
#
# So the set is pinned here and computed by walking the closure, and the walk crosses into
# third-party packages: stopping at them is what misses `test_tuner_data.py`, and a false
# negative here is the dangerous direction -- it clears a file to run beside a live model.
#: Imported from ``conftest`` rather than kept here: the collector that skips these files
#: while another seat holds the box window reads the same tuple (issue 95), and two copies of
#: one list is how they drift. The pin's *check* stays here, where the closure walker is.
_TESTS_THAT_LOAD_MLX = TESTS_THAT_LOAD_MLX

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_roots(repo_root: Path) -> tuple[Path, ...]:
    """Where a dotted module name may resolve: this repository's sources, then the
    **running interpreter's** site-packages.

    Not ``repo_root / ".venv" / ...``. A worktree without its own ``.venv`` has no such
    directory, so a hard-coded path silently resolves nothing there, every third-party name
    becomes unreachable, and the closure tests fail for a reason that has nothing to do with
    imports -- which is what the reviewer hit in a scratch worktree. The interpreter running the
    suite always has a site-packages, wherever the checkout is.
    """
    return (repo_root / "src", Path(sysconfig.get_paths()["purelib"]))


_IMPORT_ROOTS = _import_roots(_REPO_ROOT)


def _module_path(dotted: str, roots: tuple[Path, ...]) -> Path | None:
    """The file a dotted module name resolves to, or None for a name we cannot see."""
    for root in roots:
        module = root / (dotted.replace(".", "/") + ".py")
        if module.exists():
            return module
        package = root / dotted.replace(".", "/") / "__init__.py"
        if package.exists():
            return package
    return None


def _imports_that_execute(path: Path) -> list[str]:
    """Every module imported when this file is imported.

    Module scope only: an import inside a function or a class body does not run at import
    time, and `pipeline/coherence.py` deliberately imports the runner that way. `try:`, `if:`,
    `with:` and loop bodies at module scope *do* run and are followed. A `TYPE_CHECKING` block
    never runs, so its body is skipped and its `else:` is not.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError):
        return []
    found: list[str] = []

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                # Not descended into: a function-local import does not run at import time, and
                # `pipeline/coherence.py` relies on that. Making this branch descend is the
                # mutation `test_the_walker_ignores_an_import_that_does_not_run_at_import_time`
                # exists to catch; it does, on three tests.
                continue
            if isinstance(node, ast.If):
                test = node.test
                type_checking = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                    isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                )
                walk(node.orelse)
                if not type_checking:
                    walk(node.body)
                continue
            if isinstance(node, ast.Try):
                walk(node.body)
                walk(node.orelse)
                walk(node.finalbody)
                for handler in node.handlers:
                    walk(handler.body)
                continue
            if isinstance(node, ast.With | ast.AsyncWith):
                walk(node.body)
                continue
            if isinstance(node, ast.For | ast.AsyncFor | ast.While):
                walk(node.body)
                walk(node.orelse)
                continue
            if isinstance(node, ast.Import):
                found.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.append(node.module)
                # ``from pkg import name`` also imports ``pkg.name`` when that is a submodule,
                # even if ``pkg/__init__.py`` never mentions it; the resolver below drops the
                # names that are plain attributes. Reviewer's addition at landing: a submodule
                # reached only this way would otherwise be a false negative, the dangerous
                # direction.
                found.extend(f"{node.module}.{alias.name}" for alias in node.names)

    walk(tree.body)
    return found


def _loads_mlx(entry: Path, roots: tuple[Path, ...]) -> str | None:
    """The first module on a path from ``entry`` to ``mlx``, or None if it never reaches it."""
    seen: set[str] = set()
    stack: list[tuple[str | None, Path]] = [(None, entry)]
    while stack:
        via, path = stack.pop()
        for name in _imports_that_execute(path):
            if name.split(".")[0] == "mlx":
                return via or "<the file itself>"
            if name in seen:
                continue
            seen.add(name)
            resolved = _module_path(name, roots)
            if resolved is not None:
                stack.append((via or name, resolved))
    return None


def test_the_tests_that_load_mlx_are_the_pinned_set() -> None:
    """Pinned so a new import cannot quietly put a test file beside a live model.

    The rule this enforces is operational: before running pytest while a model-run lock is
    held, run only files outside this set. It is pinned rather than merely computed so that a
    new direct or transitive mlx import has to be added here deliberately, and so that a file
    that stops loading mlx has to be removed -- a stale name is as wrong as a missing one,
    because it keeps a runnable file off the list forever.
    """
    computed = tuple(
        path.name
        for path in sorted(Path(__file__).parent.glob("*.py"))
        if _loads_mlx(path, _IMPORT_ROOTS) is not None
    )
    assert computed == _TESTS_THAT_LOAD_MLX


def test_the_grep_that_this_rule_replaces_is_wrong_in_both_directions() -> None:
    """Why the closure exists, pinned against the rule it replaces.

    ``grep -l "import mlx"`` over the named files, the rule proposed after the two crossings of
    2026-09-07, disagrees with the closure on eight files here. One disagreement
    is dangerous and seven are merely wasteful, and the test asserts both so that neither can
    quietly change.

    ``test_tuner_data.py`` is the dangerous one: it says ``from mlx_lm.tuner...``, which does not
    contain the substring ``import mlx``, and ``mlx_lm``'s package ``__init__`` loads ``mlx.core``
    through ``mlx_lm.utils``. The grep clears it and pytest loads mlx.

    The seven wasteful ones carry an ``import mlx`` inside a function, so the grep holds them and
    importing them costs nothing. ``test_preflight.py`` is deliberately *not* among either group:
    it has such a lazy import at line 1996 **and** loads mlx transitively through
    ``local_llm_lab.training.gated_delta_chunked``, so the grep happens to hold it for a reason
    unrelated to why it must be held. It was cited as the counter-example when this was first
    written; it is not one, and the correction is the point of naming it here.
    """
    tests = Path(__file__).parent
    cleared_but_loads, held_but_clean = [], []
    for path in sorted(tests.glob("*.py")):
        grep_hit = "import mlx" in path.read_text(encoding="utf-8", errors="replace")
        closure_hit = _loads_mlx(path, _IMPORT_ROOTS) is not None
        if closure_hit and not grep_hit:
            cleared_but_loads.append(path.name)
        elif grep_hit and not closure_hit:
            held_but_clean.append(path.name)

    assert cleared_but_loads == ["test_tuner_data.py"]
    assert _loads_mlx(tests / "test_tuner_data.py", _IMPORT_ROOTS) == "mlx_lm.tuner.trainer"
    assert held_but_clean == [
        # Added 2026-09-08 and it strengthens the point rather than weakening it: `conftest.py`
        # now *discusses* function-local mlx imports, in the comment on the any-scope set the
        # window collector keys on (issue 95). It imports nothing. The grep holds a file for
        # talking about the thing, which is the failure mode a substring rule always has and a
        # closure never does.
        "conftest.py",
        "test_cache_equivalence.py",
        "test_lens_jacobian.py",
        "test_metal_cache_limit.py",
        "test_repository_rules.py",
        "test_runner.py",
        "test_state_probe.py",
    ]
    assert _loads_mlx(tests / "test_preflight.py", _IMPORT_ROOTS) == (
        "local_llm_lab.training.gated_delta_chunked"
    )


def test_the_walker_ignores_an_import_that_does_not_run_at_import_time(tmp_path) -> None:
    """A function-local or TYPE_CHECKING import must not count, or the set over-reports.

    This is not hypothetical: ``pipeline/coherence.py`` imports the runner inside ``loop_step``
    precisely so importing it costs nothing, and an earlier hand-rolled walk of mine counted
    that and wrongly called two clean files dirty. A fixture with only module-scope imports
    could not show the difference.
    """
    (tmp_path / "lazy.py").write_text("def go():\n    import mlx.core as mx\n    return mx\n")
    (tmp_path / "guarded.py").write_text(
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import mlx.core as mx\n"
    )
    (tmp_path / "eager.py").write_text("import mlx.core as mx\n")
    roots = (tmp_path,)
    assert _loads_mlx(tmp_path / "lazy.py", roots) is None
    assert _loads_mlx(tmp_path / "guarded.py", roots) is None
    assert _loads_mlx(tmp_path / "eager.py", roots) == "<the file itself>"


def test_the_walker_follows_transitively_and_across_third_party(tmp_path) -> None:
    """The two failures a file-local grep has, in one fixture.

    ``leaf`` imports mlx; ``middle`` imports ``leaf``; ``entry`` imports ``middle`` and nothing
    else. And ``via_vendor`` imports a package that is not ours, which must still be followed:
    stopping at third-party names is exactly what would miss ``test_tuner_data.py``.
    """
    (tmp_path / "leaf.py").write_text("import mlx.core\n")
    (tmp_path / "middle.py").write_text("import leaf\n")
    (tmp_path / "entry.py").write_text("import middle\n")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "__init__.py").write_text("from vendor.inner import thing\n")
    (vendor / "inner.py").write_text("import mlx.core\nthing = 1\n")
    (tmp_path / "via_vendor.py").write_text("import vendor\n")
    roots = (tmp_path,)
    assert _loads_mlx(tmp_path / "entry.py", roots) == "middle"
    assert _loads_mlx(tmp_path / "via_vendor.py", roots) == "vendor"


def test_the_walker_does_not_confuse_mlx_lm_with_mlx_by_prefix(tmp_path) -> None:
    """``mlx_lm`` is a hit because it *loads* mlx, never because its name starts with ``mlx``.

    A prefix match gets the right answer on ``mlx_lm`` for the wrong reason and would also
    claim ``mlx_audio`` and ``mlx_embeddings``, which are installed here and which nothing
    imports. The fixture pins the distinction with a package that is mlx-prefixed and clean.
    """
    (tmp_path / "mlx_clean.py").write_text("value = 1\n")
    (tmp_path / "user.py").write_text("import mlx_clean\n")
    roots = (tmp_path,)
    assert _loads_mlx(tmp_path / "user.py", roots) is None


def test_the_walker_follows_a_submodule_imported_from_its_package(tmp_path) -> None:
    """``from pkg import sub`` imports ``pkg.sub`` whether or not ``pkg/__init__.py`` does.

    A package whose init is clean and whose submodule loads mlx, reached only through this form,
    must count; recording only the package half of the statement would clear it. Reviewer's
    fixture at landing."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "sub.py").write_text("import mlx.core\n")
    (pkg / "attr.py").write_text("value = 1\n")
    (tmp_path / "uses_sub.py").write_text("from pkg import sub\n")
    (tmp_path / "uses_attr.py").write_text("from pkg.attr import value\n")
    roots = (tmp_path,)
    assert _loads_mlx(tmp_path / "uses_sub.py", roots) == "pkg.sub"
    assert _loads_mlx(tmp_path / "uses_attr.py", roots) is None


def test_import_roots_come_from_the_interpreter_not_a_venv_under_the_repo(tmp_path) -> None:
    """A worktree with no ``.venv`` must still resolve third-party modules.

    The first version hard-coded ``repo_root/.venv/lib/python3.13/site-packages``. In a checkout
    without one, every third-party name resolves to nothing, so the closure stops at the first
    such import and reports files clean that are not — a **false negative**, which is the
    direction that clears a file to run beside a live model. The reviewer hit it in a scratch
    worktree and the symptom was two failing tests, not a wrong answer, but the mechanism is the
    same one.

    ``tmp_path`` stands in for that checkout: it has no ``.venv`` at all, and resolution must
    still work, because it comes from the interpreter running the suite.
    """
    assert not (tmp_path / ".venv").exists()
    roots = _import_roots(tmp_path)

    # The property, stated directly: site-packages is not derived from the root passed in.
    assert Path(sysconfig.get_paths()["purelib"]) in roots
    assert tmp_path not in roots[1].parents

    # And the behaviour: a third-party name still resolves under those roots.
    (tmp_path / "entry.py").write_text("from mlx_lm.tuner import trainer\n")
    assert _loads_mlx(tmp_path / "entry.py", roots) == "mlx_lm.tuner.trainer"


# --- Issue 89: a records script that reaches the model refuses to run ---------------------
#
# Records under `research/records/` include the scripts that produced them. Those scripts load
# the model and take no model-run lock, so a committed copy is a working launcher for code the
# repository does not own. Three were guarded by hand, and one of those guards was silently
# overwritten by a later copy of the same record, caught only because a reviewer reads every
# records commit. This is that reviewer, mechanised.

_RECORD_SENTINEL = "--i-am-a-record"
_RECORDS_ROOT = _REPO_ROOT / "research" / "records"

#: Raised by the stub packages the executed child sees instead of the real ones. If this string
#: reaches a child's stderr, its guard did not fire and the stub is the only reason no model was
#: loaded -- so the assertion that it is absent is load-bearing, not decorative.
_STUB_MARKER = "stub package: the record guard did not fire"


def _model_reaching_sites(tree: ast.Module) -> list[tuple[int, str]]:
    """Every place a records script reaches the model, decided by AST and never by substring.

    Two kinds of site:

    * an ``mlx`` or ``mlx_lm`` import, **at any scope** -- unlike the closure walker above, which
      deliberately ignores function-local imports because it asks what runs on import. Here the
      question is whether running the file can reach the model, and ``main()``'s body runs.
      ``HISTORY-CACHE-2026-09-07/prepare_corpus.py`` hides its only import inside ``main``;
      a module-scope-only rule would clear it.
    * a call to ``load_policy``, which takes the model-run lock.

    ``mlx`` is included alongside the ``mlx_lm`` the issue names, because
    ``LIVE-LENS-INFRASTRUCTURE-2026-09-07/benchmark-v1.py`` loads the checkpoint at module scope
    while importing only ``mlx.core``: an ``mlx_lm``-only rule misses the worst file in the tree.
    The hazard being guarded is Metal initialisation and the lock, and ``import mlx`` is both.
    """
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in {"mlx", "mlx_lm"}:
                    sites.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and module.split(".")[0] in {"mlx", "mlx_lm"}:
                sites.append((node.lineno, f"from {module} import ..."))
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if name == "load_policy":
                sites.append((node.lineno, "call load_policy"))
    return sorted(sites)


def _record_guard_line(tree: ast.Module) -> int | None:
    """The line of the module-scope refusal guard, or None.

    Structural, not textual: a module-level ``if`` whose test carries the sentinel **as a string
    constant** and whose body leaves the interpreter. A file that merely prints the sentinel, or
    mentions it in a comment, has no guard.
    """
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        carries = any(
            isinstance(sub, ast.Constant) and sub.value == _RECORD_SENTINEL
            for sub in ast.walk(node.test)
        )
        if not carries:
            continue
        exits = any(
            (isinstance(sub, ast.Raise) and "SystemExit" in ast.dump(sub))
            or (
                isinstance(sub, ast.Call)
                and (
                    (isinstance(sub.func, ast.Name) and sub.func.id == "exit")
                    or (isinstance(sub.func, ast.Attribute) and sub.func.attr == "exit")
                )
            )
            for statement in node.body
            for sub in ast.walk(statement)
        )
        if exits:
            return node.lineno
    return None


def _record_guard_faults(path: Path) -> list[str]:
    """Why ``path`` fails the rule, or an empty list. Never executes the file."""
    source = path.read_text()
    tree = ast.parse(source)
    sites = _model_reaching_sites(tree)
    if not sites:
        return []
    faults: list[str] = []
    guard = _record_guard_line(tree)
    if guard is None:
        first_line, first_kind = sites[0]
        return [
            f"reaches the model at line {first_line} ({first_kind}) and carries no refusal guard; "
            f"running it by hand starts a model process outside R47, R48 and the issue-83 lock"
        ]
    late = [f"line {line} ({kind})" for line, kind in sites if line < guard]
    if late:
        faults.append(
            f"the guard is at line {guard}, after {', '.join(late)}; a guard that follows the "
            f"import it guards refuses only once the model is already resident"
        )
    try:
        # `ast.parse` is not enough. A guard placed above `from __future__` parses cleanly and
        # raises SyntaxError only under `compile`, which is how the first hand-written guard
        # passed review while being broken.
        compile(source, str(path), "exec")
    except SyntaxError as error:
        faults.append(f"does not compile: {error.msg} (line {error.lineno})")
    return faults


#: The as-run transcripts. Renamed from ``.py`` so a record copy cannot be *imported* -- which
#: it did fix -- but ``python whatever.as-run.py.txt`` runs the file as happily as a ``.py``,
#: and nineteen of the twenty in the tree reach the model. The rename closed the import
#: accident and left the launcher open, so the rule covers both suffixes.
_RECORD_TRANSCRIPT_SUFFIX = ".as-run.py.txt"


def _record_sources(root: Path) -> list[Path]:
    """Every file under a records tree that Python will run: ``.py`` and the transcripts."""
    return sorted(
        {*root.rglob("*.py"), *root.rglob(f"*{_RECORD_TRANSCRIPT_SUFFIX}")},
        key=lambda path: path.as_posix(),
    )


def _guarded_record_scripts() -> list[Path]:
    """Records sources the rule applies to: those that reach the model."""
    return [
        path
        for path in _record_sources(_RECORDS_ROOT)
        if _model_reaching_sites(ast.parse(path.read_text()))
    ]


@pytest.fixture(scope="module")
def stub_import_root(tmp_path_factory) -> Path:
    """A ``PYTHONPATH`` entry whose ``mlx``, ``mlx_lm`` and ``local_llm_lab`` refuse to import.

    The execution check below runs real records scripts. If one of their guards does not fire,
    the very next thing the file does is load a 4B checkpoint or take the primary model-run lock,
    on the Director's laptop, possibly beside another run. These stubs make that impossible: the
    child dies on the import instead, and the test reports the guard that failed.
    """
    root = tmp_path_factory.mktemp("stub-imports")
    for package in ("mlx", "mlx_lm", "local_llm_lab"):
        directory = root / package
        directory.mkdir()
        (directory / "__init__.py").write_text(
            f'raise RuntimeError("{_STUB_MARKER}: {package} was imported")\n'
        )
    return root


def test_every_records_script_that_reaches_the_model_carries_a_refusal_guard() -> None:
    """The rule of issue 89, by AST.

    Guarding is not optional politeness. These files hard-code absolute paths to the checkpoint
    and to the primary checkout, so a copy that runs, runs against the real model.
    """
    faults = {
        path.relative_to(_REPO_ROOT).as_posix(): reasons
        for path in _guarded_record_scripts()
        if (reasons := _record_guard_faults(path))
    }
    detail = "\n".join(
        f"  {name}: {'; '.join(reasons)}" for name, reasons in sorted(faults.items())
    )
    assert not faults, f"records scripts that reach the model without a working guard:\n{detail}"


def test_the_rule_reads_imports_and_calls_not_the_words_mlx_lm_and_load_policy() -> None:
    """The negative fixture the rule would otherwise cry wolf on.

    ``TRAIN-COST-2026-09-05/build_page.py`` writes an HTML page whose prose names both
    ``mlx_lm`` and ``evaluate.load_policy``. It imports neither and calls neither. A substring
    scanner flags it, someone switches the rule off, and the rule protects nothing after that.
    """
    page = _RECORDS_ROOT / "TRAIN-COST-2026-09-05" / "build_page.py"
    source = page.read_text()

    assert "mlx_lm" in source and "load_policy" in source
    assert _model_reaching_sites(ast.parse(source)) == []
    assert _record_guard_faults(page) == []


def test_a_records_script_that_imports_mlx_lm_without_a_guard_fails_the_rule(tmp_path) -> None:
    """The positive fixture, and the overwrite case that motivated the issue.

    The second half is the one with teeth: it starts from a guard that passes and deletes it,
    which is what a later copy of a record did by hand and nobody noticed.
    """
    unguarded = tmp_path / "probe.py"
    unguarded.write_text('"""A probe."""\nimport mlx_lm\n\nmlx_lm.load("model")\n')
    (faults,) = (_record_guard_faults(unguarded),)
    assert faults and "carries no refusal guard" in faults[0]

    guard = (
        'import sys as _sys\n'
        f'if __name__ == "__main__" and "{_RECORD_SENTINEL}" not in _sys.argv:\n'
        '    _sys.exit("refusing to run: a record, not a launcher")\n'
    )
    guarded = tmp_path / "guarded.py"
    guarded.write_text(f'"""A probe."""\n{guard}import mlx_lm\n')
    assert _record_guard_faults(guarded) == []

    # The overwrite: the same file, the guard gone.
    overwritten = tmp_path / "overwritten.py"
    overwritten.write_text(guarded.read_text().replace(guard, ""))
    assert _record_guard_faults(overwritten)


def test_a_guard_placed_after_the_import_it_guards_fails_the_rule(tmp_path) -> None:
    """Order is the whole point: a refusal that runs after ``import mlx`` refuses too late."""
    late = tmp_path / "late.py"
    late.write_text(
        '"""A probe."""\n'
        "import mlx.core as mx\n"
        "import sys as _sys\n"
        f'if __name__ == "__main__" and "{_RECORD_SENTINEL}" not in _sys.argv:\n'
        '    _sys.exit("refusing to run")\n'
    )
    (fault,) = _record_guard_faults(late)
    assert "after line 2" in fault


def test_a_guard_above_from_future_parses_but_does_not_compile(tmp_path) -> None:
    """Why the placement check compiles rather than parses.

    ``from __future__`` must be the first statement after the docstring. A guard inserted above
    it satisfies ``ast.parse`` and raises ``SyntaxError`` only under ``compile`` -- exactly how
    the first hand-written guard passed review while being broken. The rule must see it.
    """
    broken = tmp_path / "broken.py"
    broken.write_text(
        '"""A probe."""\n'
        "import sys as _sys\n"
        f'if __name__ == "__main__" and "{_RECORD_SENTINEL}" not in _sys.argv:\n'
        '    _sys.exit("refusing to run")\n'
        "from __future__ import annotations\n"
        "import mlx_lm\n"
    )
    ast.parse(broken.read_text())  # parses, which is the trap

    (fault,) = _record_guard_faults(broken)
    assert "does not compile" in fault


def test_every_guarded_records_script_actually_refuses_when_run(stub_import_root: Path) -> None:
    """The guards are verified by running them, not by reading them.

    Every records script the rule applies to is started as a subprocess with no sentinel
    argument. Each must exit non-zero saying it refuses, before importing anything that would
    reach the model -- proved by the stub packages, which raise if reached and whose marker must
    not appear.

    Started through ``local_llm_lab.spawn`` so the child arrives by ``posix_spawn`` (R45).
    """
    environment = {**os.environ, "PYTHONPATH": str(stub_import_root)}
    for path in _guarded_record_scripts():
        name = path.relative_to(_REPO_ROOT).as_posix()
        completed = spawn.run(
            [sys.executable, str(path)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode != 0, f"{name} ran to completion instead of refusing"
        assert "refusing to run" in completed.stderr, (
            f"{name} exited {completed.returncode} without the refusal: "
            f"{completed.stderr.strip()[-400:]}"
        )
        assert _STUB_MARKER not in completed.stderr, (
            f"{name} reached a model import before refusing; only the stub stopped it"
        )


def test_the_rule_covers_the_as_run_transcripts_because_python_runs_them_too(tmp_path) -> None:
    """The hole the ``.py.txt`` rename left open.

    The rename was made so a record copy could not be *imported*, and it did fix that. It did
    nothing about running: ``python probe.as-run.py.txt`` executes the file exactly as a ``.py``
    would, and when the rule first covered only ``*.py`` it reported the records tree clean while
    nineteen transcripts reached the model and one carried a guard.

    Both halves are asserted. The discovery finds a transcript, and it finds it *for the same
    reason* it finds a ``.py`` -- the model-reaching sites, not the suffix -- so a transcript with
    no such site stays out.
    """
    (tmp_path / "reaches.as-run.py.txt").write_text('"""A transcript."""\nimport mlx.core as mx\n')
    (tmp_path / "clean.as-run.py.txt").write_text('"""A transcript."""\nimport json\n')
    (tmp_path / "reaches.py").write_text('"""A script."""\nimport mlx.core as mx\n')

    found = {path.name for path in _record_sources(tmp_path)}
    assert found == {"reaches.as-run.py.txt", "clean.as-run.py.txt", "reaches.py"}

    reaching = {
        path.name
        for path in _record_sources(tmp_path)
        if _model_reaching_sites(ast.parse(path.read_text()))
    }
    assert reaching == {"reaches.as-run.py.txt", "reaches.py"}
    assert _record_guard_faults(tmp_path / "reaches.as-run.py.txt")
    assert _record_guard_faults(tmp_path / "clean.as-run.py.txt") == []


def test_the_any_scope_mlx_set_is_the_pinned_superset_the_collector_needs() -> None:
    """Two questions, two pinned lists, and the collector needs the wider one (issue 95).

    `TESTS_THAT_LOAD_MLX` answers "what maps MLX **on import**", which is what the closure walker
    above computes and what a person needs before running a suite beside a live model.
    `TESTS_THAT_CAN_REACH_MLX` answers "what can map MLX **at all**", which is what a window has
    to exclude: a function-local `import mlx` maps the library exactly as surely when the test
    runs, and the exclusive-handoff terms of 2026-09-08 say "at any scope" for that reason.

    Five files sit between the two: they import mlx or mlx_lm inside a function or fixture. A
    collector keyed on the narrower list would have let them run inside somebody else's window
    and refused that seat's launch, which is the failure the window exists to stop.
    """

    def reaches_at_any_scope(path: Path) -> bool:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name.split(".")[0] in {"mlx", "mlx_lm"} for alias in node.names
            ):
                return True
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {
                "mlx",
                "mlx_lm",
            }:
                return True
        return False

    computed = tuple(
        sorted(
            path.name
            for path in Path(__file__).parent.glob("*.py")
            if reaches_at_any_scope(path)
        )
    )
    assert computed == TESTS_THAT_CAN_REACH_MLX, (
        "the any-scope set is pinned so a new function-local mlx import has to be added here "
        "deliberately, and so a file that stops reaching mlx has to be removed"
    )
    assert set(TESTS_THAT_LOAD_MLX) < set(TESTS_THAT_CAN_REACH_MLX), (
        "import-time loading is a strict subset of reaching at any scope; if they are equal the "
        "wider list has stopped being computed from a wider question"
    )
