"""The lens-fit CLI reads the backend once and refuses the one it cannot fit on.

Fixture-tested here because that is where it can be: this box has MLX, the rented device will not,
and the branch that matters is the one this box never takes. Both directions are driven through the
real script, so the seam is exercised where it sits rather than reimplemented in the test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _cli():
    """Load `scripts/lens_fit.py` as a module, the way the other CLI tests here do."""
    file = Path(__file__).resolve().parents[1] / "scripts/lens_fit.py"
    spec = importlib.util.spec_from_file_location("fit_cli_backend", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARGV = ["--kind", "regression", "--model", "gemma3-4b", "--corpus", "corpus", "--out", "out"]


class _Reached(RuntimeError):
    """Raised past the seam, so a test can prove the MLX path was entered rather than skipped."""


def test_a_torch_box_is_refused_before_anything_reaches_mlx(monkeypatch, capsys) -> None:
    from local_llm_lab import device, models
    from local_llm_lab.pipeline.lens_fitting import runtime

    monkeypatch.setattr(device, "backend", lambda: "torch")
    monkeypatch.delenv(device.BACKEND_ENV, raising=False)

    def refuse(*args, **kwargs):  # pragma: no cover - reaching these is the failure
        raise AssertionError("the seam let a torch box through to the MLX path")

    monkeypatch.setattr(models, "load_model_spec", refuse)
    monkeypatch.setattr(runtime, "prepare_fit", refuse)

    assert _cli().main(ARGV) == 3
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "refused"
    assert event["backend"] == "torch"
    assert event["backend_source"] == "first installed, MLX first"
    assert event["kind"] == "regression"
    # The load-bearing half: the refusal says the two backends are different instruments, so a
    # reader cannot take it as a mere packaging gap to be worked around.
    assert "finite difference" in event["estimator_here"]
    assert "exact autograd" in event["estimator_there"]
    assert "neither backend may stand in for the other" in event["not_interchangeable"]
    assert event["override"] == f"{device.BACKEND_ENV}=mlx if MLX is installed on this box"


def test_the_refusal_names_the_environment_variable_when_that_is_what_chose(monkeypatch, capsys):
    """A backend chosen by `$LLL_BACKEND` and one chosen by what is installed are different
    situations for whoever reads the refusal, so the message must not report them the same way."""
    from local_llm_lab import device

    monkeypatch.setattr(device, "backend", lambda: "torch")
    monkeypatch.setenv(device.BACKEND_ENV, "torch")

    assert _cli().main(ARGV) == 3
    assert json.loads(capsys.readouterr().out)["backend_source"] == f"${device.BACKEND_ENV}"


def test_an_mlx_box_takes_exactly_the_path_it_always_did(monkeypatch) -> None:
    """The seam must be inert on MLX: the proof is that execution reaches the first real call."""
    from local_llm_lab import device, models
    from local_llm_lab.pipeline.lens_fitting import runtime

    monkeypatch.setattr(device, "backend", lambda: "mlx")
    monkeypatch.setattr(models, "load_model_spec", lambda name: name)

    def reached(*args, **kwargs):
        raise _Reached

    monkeypatch.setattr(runtime, "prepare_fit", reached)
    with pytest.raises(_Reached):
        _cli().main(ARGV)


def test_the_seam_is_read_once_and_before_the_mlx_imports() -> None:
    """Position is the property under test, not behaviour: a backend read after the imports would
    pass both tests above on this box and fail on the device, which is the only box that matters."""
    source = (Path(__file__).resolve().parents[1] / "scripts/lens_fit.py").read_text()
    assert source.count("device.backend()") == 1
    seam = source.index("device.backend()")
    for reaches_mlx in (
        "from local_llm_lab.pipeline.lens_fitting.runtime import",
        "from local_llm_lab.pipeline.lens_fitting.artifacts import",
        "from local_llm_lab.pipeline.lens_fitting.regression import",
        "from local_llm_lab.pipeline.lens_fitting.jacobian import",
    ):
        assert seam < source.index(reaches_mlx), f"{reaches_mlx} precedes the backend read"


def test_the_module_imports_nothing_heavy_at_module_scope() -> None:
    """The refusal has to work on a box with neither MLX nor the fitting package importable."""
    before = set(sys.modules)
    _cli()
    added = set(sys.modules) - before
    assert not [name for name in added if name.split(".")[0] in ("mlx", "torch")]
