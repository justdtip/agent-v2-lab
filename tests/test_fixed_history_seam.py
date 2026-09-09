"""The seam check that replaced an assertion which could not fail.

`scripts/fixed_history_lens.py` produced the log-probability figures in
`research/records/ARM-A-DIVERGENCE-2026-09-07`, and every one of them slices a joint tokenization at
a count taken from the prompt tokenized alone. That slice is the turn's own tokens only if the join
does not re-tokenize. The guard that stood beside this was `assert <comparison> or True`, which is
`(<comparison>) or True` and true for every input; it never tested anything.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


def _script():
    file = Path(__file__).resolve().parents[1] / "scripts/fixed_history_lens.py"
    spec = importlib.util.spec_from_file_location("fixed_history_lens_under_test", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_clean_seam_passes_and_a_longer_joint_sequence_is_the_normal_case() -> None:
    script = _script()
    script.refuse_unless_prompt_is_a_prefix(
        [10, 11, 12], [10, 11, 12, 90, 91], task_id="t", label="adapter"
    )


def test_a_merged_seam_refuses_and_prints_both_tokenizations() -> None:
    """The failure this exists for: the join merges the prompt's last token with the turn's first,
    so the slice at the prompt's own count starts one token inside the prompt."""
    script = _script()
    with pytest.raises(script.SeamError) as raised:
        script.refuse_unless_prompt_is_a_prefix(
            [10, 11, 12], [10, 11, 77, 91], task_id="test-update-0004", label="adapter"
        )
    message = str(raised.value)
    assert "test-update-0004" in message
    assert "adapter" in message
    # The consequence, not only the fact: a reader must not have to work out why it matters.
    assert "summed over the wrong window" in message
    # Both sides, around the divergence — a count cannot say which token merged.
    assert "prompt alone" in message and "prompt+turn" in message
    assert "[10, 11, 12]" in message and "[10, 11, 77, 91]" in message


def test_a_truncated_joint_sequence_refuses_rather_than_indexing_past_the_end() -> None:
    script = _script()
    with pytest.raises(script.SeamError):
        script.refuse_unless_prompt_is_a_prefix([10, 11, 12], [10, 11], task_id="t", label="base")


def test_the_guard_that_could_not_fail_is_gone_from_the_script() -> None:
    """Pinned here as well as in the repository-rule scanner, because this is the file it stood in
    and the record it stood over is published.

    Parsed rather than grepped, and the reason is this test's own first draft: it searched the
    source for ``or True`` and failed on the docstring that explains the shape. A scanner for
    inert guards cannot be a substring search, because the prose describing one is not one.
    """
    source = (Path(__file__).resolve().parents[1] / "scripts/fixed_history_lens.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.BoolOp):
            constants = [v for v in node.test.values if isinstance(v, ast.Constant)]
            assert not constants, f"line {node.lineno}: assert short-circuits on a constant"
    # And the replacement is reached on both turns, not only the adapter's.
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "refuse_unless_prompt_is_a_prefix"
    ]
    assert len(calls) == 2, "the adapter turn and the base turn are both checked"
