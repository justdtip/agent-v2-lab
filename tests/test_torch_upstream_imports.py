"""The torch seam must resolve upstream once, without depending on ambient sys.path.

Fresh interpreters matter here: a module already in sys.modules can hide both the absent-clone
failure and a caller accidentally reaching a second upstream copy. These tests load no model.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

from local_llm_lab import spawn

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("local_llm_lab.arch_torch", "local_llm_lab.torch_capture")


def run_child(source, *args, env=None):
    child_env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if env:
        child_env.update(env)
    return spawn.run(
        [sys.executable, "-c", textwrap.dedent(source), *map(str, args)],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("module", MODULES)
def test_a_missing_named_clone_cannot_fall_back_to_an_installed_copy(tmp_path, module):
    pytest.importorskip("torch")
    missing = tmp_path / "missing-reference"
    result = run_child(
        """
        import importlib
        import sys
        from local_llm_lab.upstream_ref import UpstreamUnavailable
        try:
            importlib.import_module(sys.argv[1])
        except UpstreamUnavailable as error:
            assert sys.argv[2] in str(error), error
            assert "JLENS_PATH" in str(error), error
        else:
            raise AssertionError("a named missing upstream clone was silently bypassed")
        """,
        module,
        missing,
        env={"JLENS_PATH": str(missing)},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_both_modules_reach_the_shared_loader_and_use_its_upstream():
    pytest.importorskip("torch")
    result = run_child(
        """
        import inspect
        import sys
        from pathlib import Path
        from local_llm_lab import upstream_ref
        try:
            expected = upstream_ref.load_upstream()
        except upstream_ref.UpstreamUnavailable as error:
            print(error)
            sys.exit(77)
        calls = []
        original = upstream_ref.load_upstream
        def record(*args, **kwargs):
            calls.append(inspect.currentframe().f_back.f_globals["__name__"])
            return original(*args, **kwargs)
        upstream_ref.load_upstream = record
        from local_llm_lab import arch_torch, torch_capture
        assert set(calls) == {arch_torch.__name__, torch_capture.__name__}, calls
        assert torch_capture.ActivationRecorder is expected.fitting.ActivationRecorder
        assert issubclass(torch_capture.TorchCapture, expected.fitting.ActivationRecorder)
        assert Path(arch_torch.hf.__file__).resolve().parents[1] == expected.path
        """,
    )
    if result.returncode == 77:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, result.stdout + result.stderr


def test_an_already_imported_different_clone_is_refused(tmp_path):
    pytest.importorskip("torch")
    # Only a package/import identity is needed. No transcription of upstream behavior.
    first, second = tmp_path / "first", tmp_path / "second"
    for root in (first, second):
        package = root / "jlens"
        package.mkdir(parents=True)
        for name in ("__init__.py", "fitting.py", "lens.py"):
            (package / name).write_text("")
    result = run_child(
        """
        import importlib
        import os
        import sys
        from local_llm_lab.upstream_ref import UpstreamUnavailable, load_upstream
        load_upstream(sys.argv[1])
        os.environ["JLENS_PATH"] = sys.argv[2]
        for module in sys.argv[3:]:
            try:
                importlib.import_module(module)
            except UpstreamUnavailable as error:
                assert "already imported" in str(error), error
                assert sys.argv[1] in str(error) and sys.argv[2] in str(error), error
            else:
                raise AssertionError(f"{module} silently accepted a different upstream clone")
        """,
        first,
        second,
        *MODULES,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_clone_skips_both_model_test_modules_at_collection(tmp_path):
    pytest.importorskip("torch")
    missing = tmp_path / "missing-reference"
    result = run_child(
        """
        import sys
        import pytest
        sys.exit(pytest.main([
            "--collect-only", "-q", "-rs", "-p", "no:cacheprovider", *sys.argv[1:]
        ]))
        """,
        ROOT / "tests/test_arch_torch.py",
        ROOT / "tests/test_torch_capture.py",
        env={"JLENS_PATH": str(missing)},
    )
    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, output
    assert "SKIPPED [2]" in output or output.count("SKIPPED [1]") == 2, output
    assert str(missing) in output, output
    assert "ERROR" not in output, output
