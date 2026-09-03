from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ("mlx", "mlx-lm", "mlx-tune", "datasets")


def _memory_gib() -> float | None:
    try:
        raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(raw.strip()) / 1024**3
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def main() -> None:
    import mlx.core as mx

    print(f"Python:   {platform.python_version()} ({platform.machine()})")
    print(f"macOS:    {platform.mac_ver()[0]}")
    memory = _memory_gib()
    print(f"Memory:   {memory:.0f} GiB unified" if memory is not None else "Memory:   unknown")
    print(f"MLX:      default device is {mx.default_device()}")

    result = mx.sum(mx.array([1.0, 2.0, 3.0]))
    mx.eval(result)
    if result.item() != 6.0:
        raise RuntimeError("MLX compute smoke test returned an unexpected result")
    print("Compute:  MLX smoke test passed")

    for package in PACKAGES:
        try:
            print(f"Package:  {package} {version(package)}")
        except PackageNotFoundError:
            raise SystemExit(f"Missing package: {package}. Run `uv sync`.") from None
