"""One name for the device, so every seat asks the same question and records the same answer.

Two implementations of the six calls the box discipline makes — ``working_set``, ``peak``,
``reset_peak``, ``clear_cache``, ``set_cache_limit``, ``device_info`` — selected by :func:`backend`,
plus :func:`budget` (the R47 fraction of what the device will grant), :func:`select` (which torch
device), :func:`pin` (the determinism the golden tests need, set before CUDA initialises) and
:func:`describe` (what is *actually* set, read back from the runtime, never what was intended).

Nothing here imports ``mlx`` or ``torch`` at module load: the module is importable on a box with
neither, ``describe`` names which is present without importing it, and the MLX runtime is only
touched by a call that asks for it. Every memory figure is in bytes, as the MLX calls report them.
On CUDA each call takes a device index, ``"cuda:N"``, ``None`` for the selected device, or
``"all"`` for one figure per visible device.
"""

from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import os
import platform
import resource
import sys
from typing import Any

__all__ = [
    "BACKENDS",
    "BACKEND_ENV",
    "CUBLAS_DETERMINISTIC",
    "CUBLAS_ENV",
    "DEVICE_ENV",
    "R47_FRACTION",
    "backend",
    "budget",
    "clear_cache",
    "describe",
    "device_info",
    "peak",
    "pin",
    "reset_peak",
    "select",
    "set_cache_limit",
    "working_set",
]

BACKEND_ENV = "LLL_BACKEND"
DEVICE_ENV = "LLL_DEVICE"
CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_DETERMINISTIC = ":4096:8"
BACKENDS = ("mlx", "torch")
#: The fraction of the device's grantable memory a workload may plan for: R47, the lens guard's 0.6.
R47_FRACTION = 0.6

_pinned: dict[str, Any] | None = None


def _importable(name: str) -> bool:
    """Whether ``name`` can be imported, including when something has already put it there.

    The membership test is not an optimisation and cannot be dropped. ``find_spec`` raises
    ``ValueError`` — not a miss, an exception — for a module that is in ``sys.modules`` with
    ``__spec__`` set to ``None``, which is what every hand-built stub is; twelve test files in this
    suite install one. A module that is already loaded is importable by definition, so answering
    from ``sys.modules`` first is also the answer that is *correct*, not merely the one that avoids
    the exception.

    Found because the lens-fit CLI's backend seam (`scripts/lens_fit.py`) reads the backend before
    the fitting imports, which put it downstream of a stub for the first time.
    """
    if name in sys.modules:
        return True
    return importlib.util.find_spec(name) is not None


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def backend() -> str:
    """``mlx`` or ``torch``: ``$LLL_BACKEND`` if set, else the first installed, MLX first."""
    named = os.environ.get(BACKEND_ENV)
    if named:
        if named not in BACKENDS:
            raise ValueError(f"${BACKEND_ENV}={named!r}; expected one of {BACKENDS}")
        if not _importable(named):
            raise RuntimeError(f"${BACKEND_ENV}={named!r} but {named} is not installed")
        return named
    for candidate in BACKENDS:
        if _importable(candidate):
            return candidate
    raise RuntimeError("neither mlx nor torch is installed; install the [mlx] or [cuda] extra")


def select(prefer: str | None = None) -> str:
    """The torch device string: explicit, else ``$LLL_DEVICE``, else availability; MPS if named."""
    import torch

    named = prefer or os.environ.get(DEVICE_ENV)
    if not named:
        return "cuda" if torch.cuda.is_available() else "cpu"
    kind = named.split(":", 1)[0]
    if kind not in ("cpu", "cuda", "mps"):
        raise ValueError(f"unknown device {named!r}; expected cpu, cuda, cuda:N or mps")
    if kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{named} requested but CUDA is not available")
    if kind == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"{named} requested but MPS is not available")
    return named


def pin(seed: int = 0, *, deterministic: bool = True, attention: str = "eager") -> dict[str, Any]:
    """Set the determinism the golden tests need; the return is :func:`describe`, a reading.

    ``CUBLAS_WORKSPACE_CONFIG`` is read by cuBLAS at first use, so this must run before torch
    touches the device; a pin after initialisation with the variable unset is refused rather than
    reported as pinned. ``attention`` is not set here — it is the ``attn_implementation`` the caller
    passes at model load, recorded so :func:`describe` can say what the model was loaded with.
    """
    global _pinned
    import torch

    if deterministic:
        if torch.cuda.is_initialized() and os.environ.get(CUBLAS_ENV) != CUBLAS_DETERMINISTIC:
            raise RuntimeError(
                f"CUDA is already initialised with {CUBLAS_ENV}={os.environ.get(CUBLAS_ENV)!r}; "
                f"set it to {CUBLAS_DETERMINISTIC!r} or call pin() before the first CUDA use"
            )
        os.environ[CUBLAS_ENV] = CUBLAS_DETERMINISTIC
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    torch.manual_seed(seed)
    _pinned = {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "attn_implementation": attention,
    }
    return describe()


def describe() -> dict[str, Any]:
    """What is set, read back. Imports torch only as the backend or if already loaded; no mlx."""
    out: dict[str, Any] = {
        "python": platform.python_version(),
        # system/release/machine come from os.uname(); platform.platform() and .processor() shell
        # out on macOS, and a fork beside Metal aborts the interpreter (conftest, spawn.py).
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "mlx": _version("mlx"),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "pinned": _pinned,
        "determinism": "pinned" if _pinned and _pinned["deterministic"] else "UNPINNED",
    }
    try:
        out["backend"] = backend()
    except RuntimeError as error:
        out["backend"] = None
        out["backend_error"] = str(error)
    if out["backend"] == "torch" or "torch" in sys.modules:
        import torch

        out.update(
            device=os.environ.get(DEVICE_ENV) or ("cuda" if torch.cuda.is_available() else "cpu"),
            cuda_available=torch.cuda.is_available(),
            device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            device_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else [],
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            threads=torch.get_num_threads(),
        )
        out[CUBLAS_ENV] = os.environ.get(CUBLAS_ENV)
    return out


# ------------------------------------------------------------------ the six calls, two ways


def _cuda_indices(device: Any) -> list[int] | None:
    """CUDA indices for ``device``, or ``None`` when the device is not CUDA."""
    import torch

    if device == "all":
        return list(range(torch.cuda.device_count()))
    if device is None:
        device = select()
    if isinstance(device, int):
        return [device]
    if isinstance(device, str) and device.startswith("cuda"):
        _, _, index = device.partition(":")
        return [int(index) if index else torch.cuda.current_device()]
    return None


def _per_device(device: Any, cuda_call, cpu_call):
    indices = _cuda_indices(device)
    if indices is None:
        return cpu_call()
    if device == "all":
        return {index: cuda_call(index) for index in indices}
    return cuda_call(indices[0])


def _rss() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _maxrss() -> int:
    figure = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(figure if sys.platform == "darwin" else figure * 1024)


def _host_memory() -> int:
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        import psutil

        return int(psutil.virtual_memory().total)


def working_set(device: Any = None) -> int | dict[int, int] | None:
    """Bytes live now. CPU torch has no allocator counter: process RSS, or ``None`` sans psutil."""
    if backend() == "mlx":
        import mlx.core as mx

        return int(mx.get_active_memory())
    import torch

    return _per_device(device, lambda i: int(torch.cuda.memory_allocated(i)), _rss)


def peak(device: Any = None) -> int | dict[int, int]:
    """Peak bytes since the last :func:`reset_peak`; on CPU the process high-water mark, unreset."""
    if backend() == "mlx":
        import mlx.core as mx

        return int(mx.get_peak_memory())
    import torch

    return _per_device(device, lambda i: int(torch.cuda.max_memory_allocated(i)), _maxrss)


def reset_peak(device: Any = None) -> bool:
    """``True`` when the peak counter was reset; ``False`` on CPU, whose high-water mark cannot."""
    if backend() == "mlx":
        import mlx.core as mx

        mx.reset_peak_memory()
        return True
    import torch

    def reset(index: int) -> bool:
        torch.cuda.reset_peak_memory_stats(index)
        return True

    result = _per_device(device, reset, lambda: False)
    return all(result.values()) if isinstance(result, dict) else result


def clear_cache(device: Any = None) -> None:
    if backend() == "mlx":
        import mlx.core as mx

        mx.clear_cache()
        return
    import torch

    gc.collect()
    if _cuda_indices(device) is not None:
        torch.cuda.empty_cache()


def set_cache_limit(limit_bytes: int, device: Any = None) -> int | dict[int, int] | None:
    """Cap what the allocator may hold. MLX returns the previous limit; CUDA applies a per-process
    fraction of the device and returns the bytes it names; CPU cannot and returns ``None``."""
    if backend() == "mlx":
        import mlx.core as mx

        return int(mx.set_cache_limit(int(limit_bytes)))
    import torch

    def cap(index: int) -> int:
        total = torch.cuda.mem_get_info(index)[1]
        torch.cuda.set_per_process_memory_fraction(min(1.0, int(limit_bytes) / total), index)
        return int(limit_bytes)

    return _per_device(device, cap, lambda: None)


def device_info(device: Any = None) -> dict[str, Any] | dict[int, dict[str, Any]]:
    """What the device will grant, with ``memory_size`` in bytes on every backend."""
    if backend() == "mlx":
        import mlx.core as mx

        return {"backend": "mlx", **dict(mx.device_info())}
    import torch

    def cuda(index: int) -> dict[str, Any]:
        properties = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        return {
            "backend": "torch",
            "device": f"cuda:{index}",
            "name": properties.name,
            "memory_size": int(total),
            "memory_free": int(free),
            "capability": f"{properties.major}.{properties.minor}",
        }

    def cpu() -> dict[str, Any]:
        return {
            "backend": "torch",
            "device": "cpu",
            "name": platform.machine(),
            "memory_size": _host_memory(),
            "threads": torch.get_num_threads(),
        }

    return _per_device(device, cuda, cpu)


def budget(fraction: float = R47_FRACTION, device: Any = None) -> int | dict[int, int]:
    """The R47 planning cap: ``fraction`` of what the device grants, on MLX its recommended working
    set, on CUDA the device total, on CPU host memory. What §10.2 sizes a batch under."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must lie in (0, 1]; got {fraction!r}")
    info = device_info(device)
    if device == "all":
        return {index: int(fraction * entry["memory_size"]) for index, entry in info.items()}
    grant = info.get("max_recommended_working_set_size") or info["memory_size"]
    return int(fraction * grant)
