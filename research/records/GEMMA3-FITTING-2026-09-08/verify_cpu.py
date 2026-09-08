"""Run the focused corpus tests while making native MLX imports impossible."""

import importlib.abc
import importlib.util
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
sys.dont_write_bytecode = True
_original_find = importlib.util.find_spec


def package_available(name, package=None):
    if name.split(".")[0] in {"mlx", "mlx_lm"}:
        return None
    return _original_find(name, package)


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx", "mlx_lm"}:
            raise ImportError("Native imports blocked: " + fullname)


if __name__ == "__main__":
    importlib.util.find_spec = package_available
    sys.meta_path.insert(0, NoNative())
    import pytest

    raise SystemExit(
        pytest.main(
            [
                "--noconftest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--tb=short",
                str(PROJECT / "tests/test_lens_corpus.py"),
            ]
        )
    )
