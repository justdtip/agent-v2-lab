from __future__ import annotations

import pytest

from local_llm_lab.models import load_model_spec
from local_llm_lab.probes.policies import (
    policy_names,
    resolve_layers,
    resolve_policy,
    validate_layer_syntax,
)


def test_registry_policy_resolution_is_per_model_and_preserves_explicit_directories(
    tmp_path,
) -> None:
    """Catches a global policy table leaking qwen25 adapters into qwen35."""
    qwen25 = load_model_spec("qwen25-coder-3b")
    qwen35 = load_model_spec("qwen35-4b")

    assert policy_names(qwen25) == ("base", "A", "B", "C")
    assert policy_names(qwen35) == ("base",)
    assert resolve_policy("base", qwen35) is None
    assert resolve_policy("B", qwen25).as_posix().endswith(
        "outputs/agent-v2b/best-adapter"
    )
    assert resolve_policy(str(tmp_path), qwen35) == tmp_path.resolve()
    with pytest.raises(ValueError, match="unknown policy"):
        resolve_policy("B", qwen35)


def test_qwen35_default_layers_resolve_without_dtype_or_model_runtime() -> None:
    """Catches defaults remaining tied to qwen25's 36-layer indices."""
    selection = resolve_layers(None, load_model_spec("qwen35-4b"), 32)

    assert selection.indices == (5, 11, 16, 21, 27, 32)
    assert selection.fractions == (0.167, 0.333, 0.5, 0.667, 0.833, 1.0)
    assert selection.as_dict() == {
        "source": "registry-default",
        "requested": ["0.167", "0.333", "0.5", "0.667", "0.833", "1.0"],
        "fractions": [0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
        "indices": [5, 11, 16, 21, 27, 32],
        "num_layers": 32,
    }


@pytest.mark.parametrize(
    ("raw", "indices", "fractions", "requested"),
    [
        ("1,1.0", (1, 32), (0.03125, 1.0), ["1", "1.0"]),
        ("1,0.5,1.0", (1, 16, 32), (0.03125, 0.5, 1.0), ["1", "0.5", "1.0"]),
        ("0.5,16,0.500", (16,), (0.5,), ["0.5"]),
        ("1e0,1", (32, 1), (1.0, 0.03125), ["1e0", "1"]),
        ("0.01,01", (1,), (0.01,), ["0.01"]),
    ],
)
def test_cli_layer_tokens_preserve_lexical_kind_and_first_resolved_occurrence(
    raw, indices, fractions, requested
) -> None:
    """Catches decimal tokens becoming indices or later collisions replacing the first."""
    selection = resolve_layers(raw, load_model_spec("qwen35-4b"), 32)

    assert selection.indices == indices
    assert selection.fractions == fractions
    assert selection.as_dict() == {
        "source": "cli",
        "requested": requested,
        "fractions": list(fractions),
        "indices": list(indices),
        "num_layers": 32,
    }


@pytest.mark.parametrize(
    "raw",
    ["", ",1", "1,", "1,,2", "0", "-1", "0.0", "-0.1", "nan", "inf", "-inf", "1.1"],
)
def test_layer_syntax_rejects_empty_nonpositive_nonfinite_and_large_fractions(raw) -> None:
    """Catches malformed layer requests reaching a model-loading boundary."""
    with pytest.raises(ValueError, match="layers"):
        validate_layer_syntax(raw)


def test_layer_syntax_accepts_registry_default_and_mixed_valid_tokens() -> None:
    """Catches validation rejecting the omitted default or legal mixed requests."""
    validate_layer_syntax(None)
    validate_layer_syntax("1,0.5,1.0,2e-1")


@pytest.mark.parametrize("num_layers", [0, -1, True])
def test_layer_resolution_rejects_invalid_model_depth(num_layers) -> None:
    """Catches invalid architecture depths producing misleading selections."""
    with pytest.raises(ValueError, match="num_layers"):
        resolve_layers(None, load_model_spec("qwen35-4b"), num_layers)


def test_layer_resolution_rejects_explicit_indices_beyond_actual_depth() -> None:
    """Catches a syntactically valid index reaching a probe for the wrong model depth."""
    with pytest.raises(ValueError, match="1.*32"):
        resolve_layers("33", load_model_spec("qwen35-4b"), 32)
