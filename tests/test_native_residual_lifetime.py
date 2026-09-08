"""Residual ownership checks using only Python fakes, with cyclic GC disabled."""

import gc
import sys
import types
import weakref

import pytest

from local_llm_lab import arch


class FakeIds:
    ndim = 2
    shape = (1, 2)

    def astype(self, _dtype):
        return self


class Residual:
    pass


@pytest.mark.parametrize("failure", [None, "forward", "missing"])
def test_native_residuals_release_without_cyclic_collection(monkeypatch, failure):
    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    fake_core.array = lambda _ids: FakeIds()
    fake_core.int32 = object()
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    references = []
    calls = []

    class FakeCapture:
        def __init__(self, _view, sink, *, layers):
            self.sink = sink
            assert layers == (1, 2)

        def __enter__(self):
            return self

        def __call__(self, _ids):
            calls.append("forward")
            for layer in (1, 2):
                if failure == "missing" and layer == 2:
                    break
                value = Residual()
                references.append(weakref.ref(value))
                self.sink.residual(layer, 0, value)
            if failure == "forward":
                raise RuntimeError("forward failed")

        def __exit__(self, *_exc):
            calls.append("exit")

    monkeypatch.setattr(arch, "NativeCapture", FakeCapture)
    view = object.__new__(arch.ArchitectureView)
    view.num_layers = 2
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        if failure is None:
            result = view.native_residuals([3, 4], [1, 2])
            assert set(result) == {1, 2}
            assert all(
                reference() is result[layer] for layer, reference in enumerate(references, 1)
            )
            del result
        else:
            expected = RuntimeError if failure == "forward" else ValueError
            # Do not keep pytest's exception info: its traceback owns local arrays.
            with pytest.raises(expected):
                view.native_residuals([3, 4], [1, 2])
        assert calls == ["forward", "exit"]
        assert references
        assert all(reference() is None for reference in references)
    finally:
        if was_enabled:
            gc.enable()
