"""Requirements §§2, 3.3, 3.6: frozen plans and same-position derivatives."""

import importlib
from collections import Counter

import numpy as np
import pytest


def api():
    return importlib.import_module("local_llm_lab.pipeline.lens_fitting.jacobian")


def rows():
    return [
        dict(
            index=i,
            source=f"source-{i}",
            step_index=i,
            split=split,
            ids=[1, 2, 3],
            spans=["task", "task", "call"],
        )
        for i, split in enumerate(("fit", "fit", "held", "held"))
    ]


def test_sampling_is_reproducible_balanced_and_split_bound():
    """§3.3/Task3: one position per draw, equal spans, never held rows in fit."""
    j = api()
    a = j.sample_positions(rows(), split="fit", count=400, seed=7)
    assert a == j.sample_positions(rows(), split="fit", count=400, seed=7)
    assert Counter(p["span"] for p in a) == {"task": 200, "call": 200}
    assert {p["row"] for p in a} <= {0, 1}
    assert all(p["source"] == rows()[p["row"]]["source"] for p in a)
    assert len({(p["row"], p["position"]) for p in a}) < len(a)
    assert [p["draw"] for p in a] == list(range(400))


def test_convergence_floor_and_ceiling():
    """§3.3: diagnostics each ten, two stable changes, acceptance at 150 minimum."""
    j = api()
    c = j.Convergence()
    for n in range(1, 151):
        done = c.add(np.eye(2, dtype=np.float32))
        assert done == (n == 150)
    assert len(c.curve) == 15
    assert c.reason == "converged"
    assert c.curve[8]["decision_eligible"] is False
    c = j.Convergence()
    for n in range(400):
        c.add(np.eye(2, dtype=np.float32) * (2.0 ** (n // 10)))
    assert c.reason == "ceiling_without_convergence"
    assert not c.accepted


def test_plan_is_exclusive_persisted_and_tamper_checked(tmp_path):
    """§3.3/3.6: all identities, seeds and bounds freeze before model execution."""
    j = api()
    bounds = dict(atol=0.001, rtol=0.01)
    plan = j.make_plan(
        rows(),
        layers=[1, 2, 3],
        hidden_size=4,
        corpus_sha256="corpus",
        snapshot_sha256="snapshot",
        seed=9,
        self_bounds=bounds,
        response_bounds=bounds,
        stability_bounds=bounds,
        held_count=150,
        working_set_bytes=10000,
        initial_peak_bytes=1000,
    )
    path = tmp_path / "plan.json"
    frozen = j.freeze_plan(path, plan)
    assert j.read_plan(path, rows()) == frozen
    assert len(frozen["fit_positions"]) == 400
    assert len(frozen["held_positions"]) == 150
    with pytest.raises(FileExistsError):
        j.freeze_plan(path, plan)
    changed = rows()
    changed[0]["ids"][0] = 4
    with pytest.raises(ValueError, match="corpus"):
        j.read_plan(path, changed)


def test_resource_and_time_gates_fail_closed():
    """§3.3/R47: project before larger batch; bytes never certify buffer safety."""
    j = api()
    with pytest.raises(ValueError, match="memory"):
        j.memory_gate(601, 1000)
    assert j.memory_gate(600, 1000) is None
    with pytest.raises(ValueError, match="Director"):
        j.time_gate({1: 1501})
    assert j.time_gate({1: 1500}) is None


def test_validation_compares_held_mean_and_instability_is_inconclusive():
    """§3.6: response_agreement uses mean-to-mean; only map failure requests refit."""
    v = importlib.import_module("local_llm_lab.pipeline.lens_fitting.validation")
    samples = [dict(row=2, span="task"), dict(row=3, span="call")]
    directions = np.random.default_rng(5).normal(size=(32, 2)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    def measured(sample, directions, scale):
        return directions * (1 if sample["row"] == 2 else 3)

    report = v.validate_layer(
        np.eye(2, dtype=np.float32) * 2,
        samples,
        directions,
        measured,
        response_bounds=dict(atol=1e-5, rtol=1e-5),
        stability_bounds=dict(atol=1e-5, rtol=1e-5),
    )
    assert report["map_check"]["outcome"] == "pass"
    assert report["verdict"]["refit_required"] is False
    assert report["verdict"]["concept_check"] == "inconclusive"
    bad = v.validate_layer(
        np.eye(2),
        samples,
        directions,
        lambda s, d, scale: d * scale,
        response_bounds=dict(atol=0, rtol=0),
        stability_bounds=dict(atol=0, rtol=0),
    )
    assert bad["map_check"]["outcome"] == "inconclusive"
    assert not bad["verdict"]["refit_required"]


def test_native_linear_tail_full_basis_and_epsilon():
    """§2/3.3: full-basis J is output-by-input and uses full primal norm in every batch."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    j = api()
    weight = mx.array([[1.0, 3.0], [-2.0, 0.5]], dtype=mx.float32)

    class Linear:
        hidden_size, num_layers = 2, 4

        def make_cache(self):
            return [KVCache() for _ in range(4)]

        def embed(self, ids):
            return mx.array([[[float(i), 1.0] for i in ids]], dtype=mx.float32)

        def masks(self, h, cache):
            return {}

        def run_block(self, index, h, masks, cache_i):
            return h @ weight if index == 3 else h

    view = Linear()
    for p in (0, 2):
        state = j.prepare_position(view, [1, 2, 3, 4], 1, p)
        assert state.epsilon == pytest.approx(0.01 * np.linalg.norm(np.array(state.full_primal)))
        for mode in ("restore", "broadcast"):
            got = j.full_jacobian(state, mode=mode, batch_size=8)
            np.testing.assert_allclose(got, np.array(weight).T, atol=2e-5)
            np.testing.assert_allclose(
                j.reference_responses(state, np.eye(2, dtype=np.float32)),
                np.array(weight),
                atol=2e-5,
            )


def test_native_cache_metadata_and_snapshot_immutable():
    """§3.3/SOURCE-NOTES-05: clone KV/SSM state and temporal metadata without mutation."""
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    j = api()
    kv, ssm = KVCache(), ArraysCache(2)
    kv.update_and_fetch(mx.ones((1, 1, 3, 2)), mx.ones((1, 1, 3, 2)))
    ssm[0], ssm[1] = mx.ones((1, 3, 2)), mx.ones((1, 1, 2, 2))
    ssm.lengths, ssm.left_padding = mx.array([7]), mx.array([2])
    copied = j.copy_cache([kv, ssm], batch_size=8)
    assert copied[0].offset == 3
    assert copied[0].keys.shape[0] == 8
    np.testing.assert_array_equal(copied[1].lengths, [7] * 8)
    np.testing.assert_array_equal(copied[1].left_padding, [2] * 8)
    copied[0].keys[0, 0, 0, 0] = 9
    copied[1][0][0, 0, 0] = 9
    assert kv.keys[0, 0, 0, 0].item() == 1
    assert ssm[0][0, 0, 0].item() == 1
    with pytest.raises(ValueError, match="unsupported"):
        j.copy_cache([object()], batch_size=1)


def test_native_tiny_cached_reference_three_layers():
    """§3.3: 16 directions at three tiny Qwen3.5 layers, both paths, position zero too."""
    from test_live_lens_native import tiny_model

    from local_llm_lab.arch import ArchitectureView

    j = api()
    view = ArchitectureView.from_model(tiny_model())
    directions = j.unit_directions(16, view.hidden_size, 12)
    for position in (0, 3):
        for layer in (1, 2, 3):
            state = j.prepare_position(view, [1, 2, 3, 4], layer, position)
            reference = j.reference_responses(state, directions)
            for mode in ("restore", "broadcast"):
                candidate = j.cached_responses(state, directions, mode=mode, batch_size=8)
                np.testing.assert_allclose(candidate, reference, atol=3e-4, rtol=3e-3)


def test_proof_requires_both_paths_all_layers_and_frozen_plan(tmp_path):
    """§3.3/R52: missing/failed path proof cannot authorize benchmark or fit."""
    j = api()
    plan = dict(plan_sha256="frozen", self_layers=[1, 2, 3])
    with pytest.raises(ValueError, match="proof"):
        j.require_self_check(plan, {})
    checks = [
        dict(layer=layer, mode=m, outcome="pass", actual_batch_sizes=[1] if m == "restore" else [8])
        for layer in (1, 2, 3)
        for m in ("restore", "broadcast")
    ]
    proof = dict(plan_sha256="frozen", checks=checks, directions=16, batch_size=8)
    assert j.require_self_check(plan, proof) is None
    proof["checks"][0]["outcome"] = "fail"
    with pytest.raises(ValueError, match="proof"):
        j.require_self_check(plan, proof)


def test_benchmark_projects_before_measure_and_stops_on_cost():
    """§3.3/R47: batch8 first, project larger before execution, no >25min fit grant."""
    j = api()
    calls = []

    def measure(layer, mode, batch):
        calls.append((layer, mode, batch))
        return dict(
            elapsed_s=1, preparation_s=2, peak_bytes=100, directions=8, reference_passed=True
        )

    plan = dict(
        plan_sha256="frozen",
        layers=[1, 2, 3],
        hidden_size=16,
        working_set_bytes=1000,
        initial_peak_bytes=100,
        fit_positions=[{}] * 400,
        batches=[8, 16, 32, 64],
    )
    proof = dict(
        plan_sha256="frozen",
        directions=16,
        batch_size=8,
        checks=[
            dict(
                layer=layer,
                mode=m,
                outcome="pass",
                actual_batch_sizes=[1] if m == "restore" else [8],
            )
            for layer in (1, 2, 3)
            for m in ("restore", "broadcast")
        ],
    )
    plan["self_layers"] = [1, 2, 3]
    report = j.benchmark_plan(plan, proof, measure)
    assert calls[:6] == [(layer, m, 8) for layer in (1, 2, 3) for m in ("restore", "broadcast")]
    assert not any(batch == 64 for _, _, batch in calls)
    assert report["status"] == "stop_required"
    assert report["buffer_count_safety"] == "not established by byte measurements"
    with pytest.raises(ValueError, match="benchmark"):
        j.require_benchmark(plan, report)


def test_equal_span_fit_mean_not_token_weighted():
    """§3.3/3.6: running fit mean matches validator's equal span averaging."""
    c = api().Convergence()
    c.add(np.eye(2), span="task")
    c.add(np.eye(2), span="task")
    c.add(np.eye(2) * 3, span="call")
    np.testing.assert_array_equal(c.mean, np.eye(2) * 2)


def test_plan_preparation_never_loads_runtime(tmp_path):
    """§3.3/R52: pure plan preparation binds snapshot architecture and explicit bounds."""
    import json
    from types import SimpleNamespace

    j = api()
    (tmp_path / "config.json").write_text(json.dumps(dict(hidden_size=2, num_hidden_layers=4)))
    prepared = SimpleNamespace(
        rows=rows(),
        corpus_manifest_sha256="corpus",
        snapshot=dict(snapshot_path=str(tmp_path), snapshot_sha256="snapshot"),
    )
    config = dict(
        seed=9,
        held_count=150,
        working_set_bytes=10000,
        initial_peak_bytes=1000,
        self_bounds=dict(atol=0.001, rtol=0.01),
        response_bounds=dict(atol=0.001, rtol=0.01),
        stability_bounds=dict(atol=0.001, rtol=0.01),
    )
    cfg = tmp_path / "bounds.json"
    cfg.write_text(json.dumps(config))
    frozen = j.prepare_plan(prepared, tmp_path / "plan.json", config_path=cfg)
    assert frozen["hidden_size"] == 2
    assert frozen["layers"] == [1, 2, 3]
    assert frozen["allocator_cache_limit_bytes"] == 0
    assert j.prepare_plan(prepared, tmp_path / "plan.json") == frozen
    prepared.snapshot["snapshot_sha256"] = "different"
    with pytest.raises(ValueError, match="snapshot"):
        j.prepare_plan(prepared, tmp_path / "plan.json")


def test_failed_stage_persists_proof_and_stops_before_timing(tmp_path, monkeypatch):
    """§3.3/R52: failed measured proof is saved; benchmark cannot run afterwards."""
    j = api()
    plan = dict(plan_sha256="frozen", self_layers=[1, 2, 3])
    monkeypatch.setattr(j, "self_check", lambda *a: dict(plan_sha256="frozen", checks=[]))

    def forbidden(*args, **kwargs):
        pytest.fail("benchmark reached despite failed proof")

    monkeypatch.setattr(j, "benchmark", forbidden)
    with pytest.raises(ValueError, match="proof"):
        j.run_jacobian_stage(None, [], plan, stage="fit", record_dir=tmp_path)
    assert (tmp_path / "self-check.json").exists()
    assert (tmp_path / "stop.json").exists()


def test_native_fitted_analytic_tail_and_validation(tmp_path):
    """§3.3/3.6: installed tiny stand-in fits 150 draws/layer and validates held means."""
    from types import SimpleNamespace

    import mlx.core as mx
    from test_live_lens_native import tiny_model

    from local_llm_lab.arch import ArchitectureView

    j = api()
    native = ArchitectureView.from_model(tiny_model())
    weight = np.eye(native.hidden_size, dtype=np.float32) * 2
    weight[0, 1] = 0.5
    w = mx.array(weight)
    view = SimpleNamespace(
        hidden_size=native.hidden_size,
        num_layers=native.num_layers,
        embed=native.embed,
        make_cache=native.make_cache,
        masks=native.masks,
        run_block=lambda index, h, masks, cache: h @ w if index == 3 else h,
    )
    bounds = dict(atol=1e-3, rtol=1e-3)
    plan = j.freeze_plan(
        tmp_path / "plan.json",
        j.make_plan(
            rows(),
            layers=[1, 2, 3],
            hidden_size=64,
            corpus_sha256="corpus",
            snapshot_sha256="snapshot",
            seed=4,
            self_bounds=bounds,
            response_bounds=bounds,
            stability_bounds=bounds,
            held_count=150,
            working_set_bytes=2**30,
            initial_peak_bytes=2**28,
        ),
    )
    result, validation = j.run_jacobian_stage(view, rows(), plan, stage="fit", record_dir=tmp_path)
    assert set(result.maps) == {1, 2, 3}
    for layer, matrix in result.maps.items():
        np.testing.assert_allclose(matrix, weight.T, atol=1e-4)
        assert result.per_layer[str(layer)]["positions"] == 150
        assert validation["per_layer"][str(layer)]["map_check"]["outcome"] == "pass"
    assert (tmp_path / "self-check.json").exists()
    assert (tmp_path / "benchmark.json").exists()
    assert (tmp_path / "validation.json").exists()


def test_benchmark_refuses_falsified_memory_projection_before_next_measurement():
    """§3.3/R47: a measured peak above its declared projection invalidates the next run."""
    j = api()
    plan = dict(
        plan_sha256="frozen",
        self_layers=[1, 2, 3],
        layers=[1, 2, 3],
        hidden_size=2,
        working_set_bytes=1000,
        initial_peak_bytes=100,
        fit_positions=[{}] * 400,
        batches=[8, 16, 32, 64],
    )
    proof = dict(
        plan_sha256="frozen",
        directions=16,
        batch_size=8,
        checks=[
            dict(
                layer=layer,
                mode=mode,
                outcome="pass",
                actual_batch_sizes=[1] if mode == "restore" else [8],
            )
            for layer in (1, 2, 3)
            for mode in ("restore", "broadcast")
        ],
    )
    calls = []

    def measure(*args):
        calls.append(args)
        return dict(
            elapsed_s=0.1, preparation_s=0.1, peak_bytes=140, directions=2, reference_passed=True
        )

    with pytest.raises(ValueError, match="projection"):
        j.benchmark_plan(plan, proof, measure)
    assert len(calls) == 1


def test_zero_primal_step_fallback_does_not_divide_by_tangent_norm():
    """§3.3/jlens rule: zero primal uses epsilon0.01 even for a nonunit tangent."""
    j = api()
    np.testing.assert_allclose(j.finite_difference_steps(0, np.array([2.0])), [0.01])
    np.testing.assert_allclose(j.finite_difference_steps(5, np.array([2.0])), [0.025])


def fake_array_backend(monkeypatch):
    """Pure array/runtime double: importing it never initializes Metal."""
    import sys
    from types import ModuleType

    backend = ModuleType("mlx.core")
    backend.peak = 10
    backend.get_peak_memory = lambda: backend.peak
    backend.device_info = lambda: {"max_recommended_working_set_size": 10000}
    backend.array = np.asarray
    backend.float32 = np.float32
    backend.zeros_like = np.zeros_like
    backend.broadcast_to = np.broadcast_to
    backend.eval = lambda *args: None
    backend.linalg = np.linalg
    package = ModuleType("mlx")
    package.core = backend
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", backend)
    return backend


def check_plan():
    return dict(
        plan_sha256="frozen",
        self_layers=[1, 2, 3],
        self_row=0,
        self_position=0,
        working_set_bytes=10000,
        initial_peak_bytes=100,
        self_bounds=dict(atol=1e-5, rtol=1e-5),
        stability_bounds=dict(atol=1e-5, rtol=1e-5),
        seeds=dict(self_directions=7),
    )


@pytest.mark.parametrize("breach_at", ["prepare", "reference", "half", "restore"])
@pytest.mark.parametrize("peak", [140, 7000])
def test_self_check_stops_between_internal_workloads(monkeypatch, tmp_path, breach_at, peak):
    """§3.3/R47 review R1: no later reference/candidate after any breached workload."""
    import json
    from types import SimpleNamespace

    j = api()
    backend = fake_array_backend(monkeypatch)
    calls = []

    def mark(name):
        calls.append(name)
        if name == breach_at:
            backend.peak = peak

    def prepare(*args, **kwargs):
        mark("prepare")
        return SimpleNamespace(epsilon=0.01)

    def reference(state, directions, *, epsilon_scale=1.0, **kwargs):
        mark("reference" if epsilon_scale == 1 else "half")
        return directions

    def candidate(state, directions, *, mode, **kwargs):
        mark(mode)
        return directions

    monkeypatch.setattr(j, "prepare_position", prepare)
    monkeypatch.setattr(j, "reference_responses", reference)
    monkeypatch.setattr(j, "cached_responses", candidate)
    with pytest.raises(ValueError):
        j.run_jacobian_stage(
            SimpleNamespace(hidden_size=2),
            [dict(ids=[1])],
            check_plan(),
            stage="check",
            record_dir=tmp_path,
        )
    expected = ["prepare", "reference", "half", "restore"]
    assert calls == expected[: expected.index(breach_at) + 1]
    stopped = json.loads((tmp_path / "stop.json").read_text())
    assert stopped["peak_bytes"] == peak
    assert stopped["projected_peak_bytes"] == 100
    assert stopped["working_set_bytes"] == 10000


@pytest.mark.parametrize("breach_at", ["prepare", "candidate"])
def test_benchmark_checks_first_extreme_before_second_prepare(monkeypatch, breach_at):
    """§3.3/R47 review R1: internal first-extreme breach prevents second extreme."""
    from types import SimpleNamespace

    j = api()
    backend = fake_array_backend(monkeypatch)
    calls = []

    def prepare(*args, **kwargs):
        calls.append("prepare")
        if breach_at == "prepare":
            backend.peak = 140
        return None

    def candidate(*args, **kwargs):
        calls.append("candidate")
        backend.peak = 140
        return np.ones((2, 2))

    monkeypatch.setattr(j, "prepare_position", prepare)
    monkeypatch.setattr(j, "cached_responses", candidate)
    plan = check_plan() | dict(
        layers=[1, 2, 3],
        hidden_size=2,
        batches=[8],
        fit_positions=[dict(row=0, position=0)] * 400,
        held_positions=[dict(row=1, position=1)],
    )
    proof = dict(
        plan_sha256="frozen",
        directions=16,
        batch_size=8,
        checks=[
            dict(
                layer=layer,
                mode=mode,
                outcome="pass",
                actual_batch_sizes=[1] if mode == "restore" else [8],
            )
            for layer in (1, 2, 3)
            for mode in ("restore", "broadcast")
        ],
    )
    with pytest.raises(ValueError):
        j.benchmark(SimpleNamespace(hidden_size=2), [dict(ids=[1]), dict(ids=[1, 2])], plan, proof)
    assert calls == (["prepare"] if breach_at == "prepare" else ["prepare", "candidate"])


@pytest.mark.parametrize("batch_size", [32, 64])
def test_self_check_exercises_actual_broadcast_width(monkeypatch, batch_size):
    """§3.3/R52 review R2: 16 unique directions must exercise actual batch32/64 arrays."""
    from types import SimpleNamespace

    j = api()
    fake_array_backend(monkeypatch)
    state = SimpleNamespace(
        view=SimpleNamespace(hidden_size=2),
        primal=np.ones((1, 1, 2)),
        primal_norm=2.0,
        prefix_cache=[],
        layer=1,
        epsilon=0.02,
    )
    monkeypatch.setattr(j, "prepare_position", lambda *a, **k: state)
    reference_rows = []

    def reference(state, directions, **kwargs):
        reference_rows.append(len(directions))
        return directions

    monkeypatch.setattr(j, "reference_responses", reference)
    cache_widths, tail_widths = [], []

    def cache(entries, *, batch_size):
        cache_widths.append(batch_size)
        return []

    def tail(view, layer, h, cache):
        tail_widths.append(h.shape[0])
        return h

    monkeypatch.setattr(j, "copy_cache", cache)
    monkeypatch.setattr(j, "pre_norm_tail", tail)
    proof = j.self_check(
        SimpleNamespace(hidden_size=2), [dict(ids=[1])], check_plan(), batch_size=batch_size
    )
    assert reference_rows == [16] * 6
    assert max(cache_widths) == batch_size
    assert max(tail_widths) == batch_size
    assert proof["directions"] == 16
    assert {c["candidate_rows"] for c in proof["checks"]} == {batch_size}
    j.require_self_check(check_plan(), proof, batch_size=batch_size)
    proof["checks"][-1]["actual_batch_sizes"] = [16]
    with pytest.raises(ValueError, match="proof"):
        j.require_self_check(check_plan(), proof, batch_size=batch_size)
    assert {c["outcome"] for c in proof["checks"]} == {"pass"}


@pytest.mark.parametrize("route", ["reference", "cached_copy", "cached_plus"])
def test_derivative_guard_stops_before_next_sign_or_cache(monkeypatch, route):
    """§3.3/R47 review R1: the guard also runs inside the FD direction/sign loops."""
    from types import SimpleNamespace

    j = api()
    backend = fake_array_backend(monkeypatch)
    state = SimpleNamespace(
        view=SimpleNamespace(hidden_size=2),
        primal=np.ones((1, 1, 2)),
        full_primal=np.ones((1, 2, 2)),
        position=0,
        primal_norm=2.0,
        prefix_cache=[],
        layer=1,
        epsilon=0.02,
    )
    calls = []

    def copy(entries, *, batch_size):
        calls.append("copy")
        if route == "cached_copy":
            backend.peak = 140
        return []

    def tail(view, layer, h, cache=None):
        calls.append("tail")
        backend.peak = 140
        return h

    monkeypatch.setattr(j, "copy_cache", copy)
    monkeypatch.setattr(j, "pre_norm_tail", tail)
    guard = j.WorkloadMemoryGuard(100, 10000, backend.get_peak_memory)
    with pytest.raises(ValueError):
        if route == "reference":
            j.reference_responses(state, np.eye(2), guard=guard)
        else:
            j.cached_responses(state, np.eye(2), mode="broadcast", guard=guard)
    assert (
        calls
        == {"reference": ["tail"], "cached_copy": ["copy"], "cached_plus": ["copy", "tail"]}[route]
    )


def test_prepare_guard_stops_before_prefix_cache_allocation(monkeypatch):
    """§3.3/R47 review R1: full-primal breach stops before prefix preparation starts."""
    from types import SimpleNamespace

    j = api()
    backend = fake_array_backend(monkeypatch)
    calls = []

    def block(*args):
        calls.append("block")
        backend.peak = 140
        return np.ones((1, 2, 2))

    def cache():
        pytest.fail("prefix cache allocated after full-primal breach")

    view = SimpleNamespace(
        num_layers=3,
        embed=lambda ids: np.ones((1, len(ids), 2)),
        masks=lambda *a: {},
        run_block=block,
        make_cache=cache,
    )
    guard = j.WorkloadMemoryGuard(100, 10000, backend.get_peak_memory)
    with pytest.raises(ValueError):
        j.prepare_position(view, [1, 2], 1, 1, guard=guard)
    assert calls == ["block"]


def test_validation_guard_stops_before_half_epsilon_measurement(monkeypatch):
    """§3.6/R47 review R1: held validation shares the bounded derivative guard."""
    from types import SimpleNamespace

    j = api()
    v = importlib.import_module("local_llm_lab.pipeline.lens_fitting.validation")
    backend = fake_array_backend(monkeypatch)
    calls = []

    def prepare(*args, **kwargs):
        calls.append("prepare")
        return None

    def candidate(*args, **kwargs):
        calls.append("candidate")
        backend.peak = 140
        return np.ones((32, 2))

    monkeypatch.setattr(j, "prepare_position", prepare)
    monkeypatch.setattr(j, "cached_responses", candidate)
    choice = dict(
        layer=1,
        mode="broadcast",
        batch_size=8,
        reference_passed=True,
        peak_bytes=10,
        projected_peak_bytes=100,
        projected_seconds=1,
    )
    plan = dict(
        plan_sha256="frozen",
        layers=[1],
        rows_sha256=j.digest(rows()),
        held_positions=[dict(row=2, split="held", span="task", position=0)],
        fit_positions=[dict(row=0, split="fit", span="task", position=0)],
        paths=["restore", "broadcast"],
        batches=[8],
        working_set_bytes=10000,
        seeds=dict(validation_directions=7),
        response_bounds=dict(atol=0.1, rtol=0.1),
        stability_bounds=dict(atol=0.1, rtol=0.1),
    )
    report = dict(
        plan_sha256="frozen",
        status="ready",
        selected={"1": choice},
        observations=[choice],
        projected_positions=400,
    )
    with pytest.raises(ValueError, match="projection"):
        v.validate_maps(SimpleNamespace(hidden_size=2), rows(), {1: np.eye(2)}, plan, report)
    assert calls == ["prepare", "candidate"]


def test_fit_guard_stops_before_first_basis_after_prepare_breach(monkeypatch):
    """§3.3/R47 review R1: a fit preparation breach cannot launch its basis tail."""
    from types import SimpleNamespace

    j = api()
    backend = fake_array_backend(monkeypatch)
    calls = []

    def prepare(*args, **kwargs):
        calls.append("prepare")
        backend.peak = 140
        return None

    def basis(*args, **kwargs):
        calls.append("basis")
        return np.eye(2)

    monkeypatch.setattr(j, "prepare_position", prepare)
    monkeypatch.setattr(j, "full_jacobian", basis)
    plan = check_plan() | dict(
        layers=[1, 2, 3],
        hidden_size=2,
        rows_sha256=j.digest(rows()),
        paths=["restore", "broadcast"],
        batches=[8],
        fit_positions=[dict(row=0, position=0, span="task")] * 400,
    )
    checks = [
        dict(
            layer=layer,
            mode=mode,
            outcome="pass",
            actual_batch_sizes=[1] if mode == "restore" else [8],
        )
        for layer in (1, 2, 3)
        for mode in ("restore", "broadcast")
    ]
    proof = dict(plan_sha256="frozen", checks=checks, directions=16, batch_size=8)
    choices = {
        str(layer): dict(
            layer=layer,
            mode="broadcast",
            batch_size=8,
            reference_passed=True,
            peak_bytes=10,
            projected_peak_bytes=100,
            projected_seconds=1,
        )
        for layer in (1, 2, 3)
    }
    report = dict(
        plan_sha256="frozen",
        status="ready",
        selected=choices,
        observations=list(choices.values()),
        projected_positions=400,
    )
    with pytest.raises(ValueError, match="projection"):
        j.fit_jacobian(SimpleNamespace(hidden_size=2, num_layers=4), rows(), plan, proof, report)
    assert calls == ["prepare"]


@pytest.mark.parametrize("batch_size", [32, 64])
def test_native_self_check_large_batch_dimensions(monkeypatch, batch_size):
    """§3.3/R52 review R2: actual tiny cache and tail arrays reach batch32/64."""
    from test_live_lens_native import tiny_model

    from local_llm_lab.arch import ArchitectureView

    j = api()
    view = ArchitectureView.from_model(tiny_model())
    plan = check_plan() | dict(
        working_set_bytes=4 * 2**30,
        initial_peak_bytes=2**28,
        self_position=3,
        self_bounds=dict(atol=3e-3, rtol=3e-2),
        stability_bounds=dict(atol=3e-3, rtol=3e-2),
    )
    original_copy, original_tail = j.copy_cache, j.pre_norm_tail
    cache_widths, tail_widths = [], []

    def copy(cache, *, batch_size=1):
        entries = original_copy(cache, batch_size=batch_size)
        for entry in entries:
            if getattr(entry, "keys", None) is not None:
                cache_widths.append(entry.keys.shape[0])
            elif hasattr(entry, "cache"):
                cache_widths.extend(a.shape[0] for a in entry.cache if a is not None)
        return entries

    def tail(view, layer, h, cache=None):
        tail_widths.append(h.shape[0])
        return original_tail(view, layer, h, cache)

    monkeypatch.setattr(j, "copy_cache", copy)
    monkeypatch.setattr(j, "pre_norm_tail", tail)
    proof = j.self_check(view, [dict(ids=[1, 2, 3, 4])], plan, batch_size=batch_size)
    j.require_self_check(plan, proof, batch_size=batch_size)
    assert max(cache_widths) == batch_size
    assert max(tail_widths) == batch_size
