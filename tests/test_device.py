"""`device.py`: one name for the device, two implementations of the box calls, readings not intents.

No model, no MLX import: the MLX branch is exercised through a stub ``mlx.core`` so this file stays
outside the MLX import closure, and the CUDA branch through stubbed ``torch.cuda`` calls so the
multi-device form is tested on a box with no device.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from local_llm_lab import device

torch = pytest.importorskip("torch")


@pytest.fixture
def restore_torch_flags():
    """`pin()` is global state; hand it back the way it was found."""
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
        os.environ.get(device.CUBLAS_ENV),
        device._pinned,
    )
    yield
    torch.use_deterministic_algorithms(saved[0])
    torch.backends.cudnn.deterministic = saved[1]
    torch.backends.cudnn.benchmark = saved[2]
    torch.backends.cuda.matmul.allow_tf32 = saved[3]
    torch.backends.cudnn.allow_tf32 = saved[4]
    torch.set_float32_matmul_precision(saved[5])
    if saved[6] is None:
        os.environ.pop(device.CUBLAS_ENV, None)
    else:
        os.environ[device.CUBLAS_ENV] = saved[6]
    device._pinned = saved[7]


@pytest.fixture
def torch_cpu(monkeypatch):
    monkeypatch.setenv(device.BACKEND_ENV, "torch")
    monkeypatch.setenv(device.DEVICE_ENV, "cpu")


@pytest.fixture
def stub_mlx(monkeypatch):
    """A fake ``mlx.core`` with the six calls, so dispatch is tested without loading MLX."""
    calls: list[tuple] = []
    core = types.SimpleNamespace(
        get_active_memory=lambda: 111,
        get_peak_memory=lambda: 222,
        reset_peak_memory=lambda: calls.append(("reset",)),
        clear_cache=lambda: calls.append(("clear",)),
        set_cache_limit=lambda limit: calls.append(("limit", limit)) or 333,
        device_info=lambda: {"memory_size": 1000, "max_recommended_working_set_size": 800},
    )
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(device, "_importable", lambda name: name == "mlx")
    monkeypatch.setenv(device.BACKEND_ENV, "mlx")
    return calls


# ---------------------------------------------------------------------------- backend, select


def test_backend_honours_the_variable_and_refuses_what_is_not_installed(monkeypatch):
    monkeypatch.setattr(device, "_importable", lambda name: name == "torch")
    monkeypatch.delenv(device.BACKEND_ENV, raising=False)
    assert device.backend() == "torch"
    monkeypatch.setenv(device.BACKEND_ENV, "mlx")
    with pytest.raises(RuntimeError, match="not installed"):
        device.backend()
    monkeypatch.setenv(device.BACKEND_ENV, "cuda")
    with pytest.raises(ValueError, match="expected one of"):
        device.backend()
    monkeypatch.setattr(device, "_importable", lambda name: False)
    monkeypatch.delenv(device.BACKEND_ENV)
    with pytest.raises(RuntimeError, match="neither"):
        device.backend()


def test_backend_prefers_mlx_when_both_are_present_so_the_laptop_path_survives(monkeypatch):
    monkeypatch.setattr(device, "_importable", lambda name: True)
    monkeypatch.delenv(device.BACKEND_ENV, raising=False)
    assert device.backend() == "mlx"


def test_select_is_explicit_then_variable_then_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv(device.DEVICE_ENV, raising=False)
    assert device.select() == "cpu"
    assert device.select("cpu") == "cpu"
    monkeypatch.setenv(device.DEVICE_ENV, "cuda:1")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        device.select()
    assert device.select("cpu") == "cpu"
    with pytest.raises(ValueError, match="unknown device"):
        device.select("tpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv(device.DEVICE_ENV)
    assert device.select() == "cuda"
    assert device.select("cuda:1") == "cuda:1"


# ---------------------------------------------------------------------------- pin, describe


def test_describe_is_a_reading_and_says_unpinned_until_pin_runs(torch_cpu, restore_torch_flags):
    device._pinned = None
    before = device.describe()
    assert before["backend"] == "torch"
    assert before["determinism"] == "UNPINNED"
    assert before["pinned"] is None
    assert before["torch"] == torch.__version__
    assert isinstance(before["deterministic_algorithms"], bool)
    after = device.pin(seed=7, attention="eager")
    assert after["determinism"] == "pinned"
    assert after["pinned"] == {"seed": 7, "deterministic": True, "attn_implementation": "eager"}
    # Read back from torch, not echoed from the arguments.
    assert torch.are_deterministic_algorithms_enabled() is True
    assert after["deterministic_algorithms"] is True
    assert after["tf32_matmul"] is False and after["tf32_cudnn"] is False
    assert after["cudnn_benchmark"] is False and after["cudnn_deterministic"] is True
    assert after["float32_matmul_precision"] == "highest"
    assert after[device.CUBLAS_ENV] == device.CUBLAS_DETERMINISTIC
    assert torch.initial_seed() == 7


def test_pin_refuses_after_cuda_initialised_without_the_workspace_variable(
    torch_cpu, restore_torch_flags, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.delenv(device.CUBLAS_ENV, raising=False)
    with pytest.raises(RuntimeError, match="already initialised"):
        device.pin()
    monkeypatch.setenv(device.CUBLAS_ENV, device.CUBLAS_DETERMINISTIC)
    assert device.pin()["determinism"] == "pinned"


def test_describe_does_not_import_mlx(torch_cpu, monkeypatch):
    def refuse(name, *args, **kwargs):
        if name.split(".")[0] == "mlx":
            raise AssertionError("describe() imported mlx")
        return real_import(name, *args, **kwargs)

    import builtins

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", refuse)
    assert "backend" in device.describe()


# ---------------------------------------------------------------------------- the six calls


def test_the_six_calls_on_cpu_torch_say_what_they_cannot_do(torch_cpu):
    assert device.working_set() is None or device.working_set() > 0
    assert device.peak() > 0
    assert device.reset_peak() is False
    assert device.clear_cache() is None
    assert device.set_cache_limit(10**9) is None
    info = device.device_info()
    assert info["backend"] == "torch" and info["device"] == "cpu"
    assert info["memory_size"] > 0
    assert device.budget() == int(device.R47_FRACTION * info["memory_size"])
    assert device.budget(0.5) == int(0.5 * info["memory_size"])
    with pytest.raises(ValueError, match="fraction"):
        device.budget(0)


def test_the_six_calls_dispatch_to_mlx_and_budget_uses_the_recommended_working_set(stub_mlx):
    assert device.working_set() == 111
    assert device.peak() == 222
    assert device.reset_peak() is True
    device.clear_cache()
    assert device.set_cache_limit(444) == 333
    assert device.device_info() == {
        "backend": "mlx",
        "memory_size": 1000,
        "max_recommended_working_set_size": 800,
    }
    assert device.budget() == int(0.6 * 800)
    assert stub_mlx == [("reset",), ("clear",), ("limit", 444)]


def test_the_cuda_form_takes_an_index_a_string_none_or_all(torch_cpu, monkeypatch):
    monkeypatch.setenv(device.DEVICE_ENV, "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i: 100 + i)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda i: 200 + i)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (500 + i, 1000 + i))
    fractions: dict[int, float] = {}
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda f, i: fractions.__setitem__(i, f)
    )
    resets: list[int] = []
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda i: resets.append(i))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: types.SimpleNamespace(name=f"gpu{i}", major=9, minor=0),
    )
    assert device.working_set() == 100
    assert device.working_set(1) == 101
    assert device.working_set("cuda:1") == 101
    assert device.working_set("all") == {0: 100, 1: 101}
    assert device.peak("all") == {0: 200, 1: 201}
    assert device.reset_peak("all") is True and resets == [0, 1]
    assert device.set_cache_limit(500, "all") == {0: 500, 1: 500}
    assert fractions == {0: 0.5, 1: 500 / 1001}
    info = device.device_info("all")
    assert info[1] == {
        "backend": "torch",
        "device": "cuda:1",
        "name": "gpu1",
        "memory_size": 1001,
        "memory_free": 501,
        "capability": "9.0",
    }
    assert device.budget(device="all") == {0: 600, 1: int(0.6 * 1001)}
    assert device.budget() == 600
    device.clear_cache()


def test_importable_answers_for_a_module_that_is_present_without_a_spec(monkeypatch):
    """`find_spec` raises rather than returning None for a stub, and this suite is full of stubs.

    Twelve test files install a hand-built module into `sys.modules` to keep MLX out of a test.
    Such a module has `__spec__ is None`, and `importlib.util.find_spec` answers that with
    `ValueError`. Any code that reads the backend downstream of one of those stubs would fail with
    an error about a spec rather than about a backend. The lens-fit CLI's seam is the first caller
    in that position, which is how this was found.
    """
    stub = types.ModuleType("mlx")
    assert stub.__spec__ is None, "the premise: a hand-built module carries no spec"
    monkeypatch.setitem(sys.modules, "mlx", stub)

    assert device._importable("mlx") is True
    monkeypatch.delenv(device.BACKEND_ENV, raising=False)
    assert device.backend() == "mlx"

    # A name that is neither loaded nor installed still goes to `find_spec` and still answers False.
    assert device._importable("a_module_that_is_not_installed_anywhere") is False


def test_the_shims_readers_set_the_workspace_default_so_a_later_pin_is_not_refused(monkeypatch):
    """The first device run: preflight read the device before it pinned, and pin refused by its
    own rule. The shim's readers now set the cuBLAS default before creating a context."""
    torch = pytest.importorskip("torch")
    monkeypatch.delenv(device.CUBLAS_ENV, raising=False)
    monkeypatch.setattr(device, "_pinned", None)
    monkeypatch.setattr(device, "_workspace_before_cuda", None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    device._before_cuda()
    assert os.environ[device.CUBLAS_ENV] == device.CUBLAS_DETERMINISTIC
    # A context that now exists, created after the default was set: pin proceeds and the reading
    # says the workspace was set before the context, which is what preflight's row reads.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    reading = device.pin(seed=3)
    assert reading["determinism"] == "pinned"
    assert reading["cublas_workspace_before_cuda"] is True
    # A context that exists with the variable unset, created outside the shim: the readers set
    # nothing, the fact is recorded as false, and pin still refuses.
    monkeypatch.delenv(device.CUBLAS_ENV)
    monkeypatch.setattr(device, "_workspace_before_cuda", None)
    device._before_cuda()
    assert device.CUBLAS_ENV not in os.environ
    assert device.describe()["cublas_workspace_before_cuda"] is False
    with pytest.raises(RuntimeError, match="before the first CUDA use"):
        device.pin(seed=3)
