"""Graph-once Jacobian acceptance on frozen real HF CPU fixtures, never a checkpoint.

The 64-token exactness test covers every source block. Timing uses Research's separate
48-token, width-64, six-block fixture and sources 1/3 to target 4; both paths are measured
end to end after warmup, best of three, under eager and sdpa. CUDA remains unexecuted.
"""

from __future__ import annotations

import json
import time
import weakref
from importlib import import_module
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from local_llm_lab.upstream_ref import UpstreamUnavailable, load_upstream  # noqa: E402

try:
    UPSTREAM = load_upstream()
except UpstreamUnavailable as error:
    pytest.skip(str(error), allow_module_level=True)

from transformers import Gemma3ForCausalLM, Gemma3TextConfig  # noqa: E402

HF = import_module("jlens.hf")  # The shared loader has already selected/validated this copy.
WIDTH, BLOCKS, VOCAB = 64, 6, 128
SEQ, TIMING_SEQ, DIM_BATCH, REPEATS = 64, 48, 16, 3
PROMPT = "fixed fixture token IDs"
EPSILON = torch.finfo(torch.float32).eps


@pytest.fixture
def estimator():
    # Missing implementation is a failed test, not an optional-dependency skip.
    from local_llm_lab.torch_jacobian import jacobian_for_prompt_vjp

    return jacobian_for_prompt_vjp


def make_model(attention="eager", *, seq_len=SEQ):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = Gemma3TextConfig(
        num_hidden_layers=BLOCKS,
        hidden_size=WIDTH,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=128,
        vocab_size=VOCAB,
        sliding_window=16,
        sliding_window_pattern=2,
        attn_implementation=attention,
    )
    model = Gemma3ForCausalLM(config).float().eval().requires_grad_(False)
    frozen_ids = torch.randint(0, VOCAB, (1, seq_len))

    class FixedTokenizer:
        def __call__(self, prompt, *, return_tensors, truncation, max_length):
            assert prompt == PROMPT and return_tensors == "pt" and truncation is True
            return SimpleNamespace(input_ids=frozen_ids[:, :max_length].clone())

    return HF.HFLensModel(model, FixedTokenizer(), force_bos=False)


def assert_maps_close(actual, expected):
    assert actual.keys() == expected.keys()
    errors = {}
    for layer in actual:
        ours, reference = actual[layer], expected[layer]
        assert ours.dtype == torch.float32 and ours.device.type == "cpu"
        assert ours.shape == (WIDTH, WIDTH) and not ours.requires_grad
        assert bool(torch.isfinite(ours).all())
        difference = float((ours - reference).abs().max())
        scale = float(reference.abs().max())
        errors[layer] = {
            "max_abs": difference,
            "max_norm_relative": difference / scale if scale else 0.0,
        }
        torch.testing.assert_close(
            ours, reference, atol=EPSILON, rtol=EPSILON,
            msg=lambda message, layer=layer: f"source block {layer}: {message}",
        )
    return errors


def hook_counts(model):
    return [(len(block._forward_hooks), len(block._forward_pre_hooks)) for block in model.layers]


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_every_source_layer_matches_unmodified_upstream_at_float32_epsilon(
    estimator, attention, capsys
):
    model = make_model(attention)
    sources = list(range(BLOCKS - 1))
    expected, seq_len, selected = UPSTREAM.fitting.jacobian_for_prompt(
        model, PROMPT, sources, dim_batch=1, max_seq_len=SEQ
    )
    actual, actual_len, actual_selected = estimator(
        model, PROMPT, sources, dim_batch=DIM_BATCH, max_seq_len=SEQ
    )
    assert (actual_len, actual_selected) == (seq_len, selected)
    assert seq_len == SEQ
    assert selected == int(UPSTREAM.valid_position_mask(SEQ).sum())
    errors = assert_maps_close(actual, expected)
    with capsys.disabled():
        print("\nJACOBIAN_EXACTNESS " + json.dumps({
            "attention": attention, "tokens": SEQ, "source_blocks": sources,
            "target_block": BLOCKS - 1, "atol": EPSILON, "rtol": EPSILON,
            "errors_by_layer": errors,
        }, sort_keys=True))


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("dim_batch", [7, WIDTH + 1])
def test_remainder_and_oversized_batches_preserve_all_rows_and_negative_layer_indices(
    estimator, attention, dim_batch
):
    model = make_model(attention)
    expected, length, count = UPSTREAM.fitting.jacobian_for_prompt(
        model, PROMPT, [1, 3], target_layer=4, dim_batch=1, max_seq_len=SEQ
    )
    actual, actual_len, actual_count = estimator(
        model, PROMPT, [-3, -5, 1], target_layer=-2, dim_batch=dim_batch, max_seq_len=SEQ
    )
    assert (actual_len, actual_count) == (length, count)
    assert list(actual) == [1, 3]
    assert_maps_close(actual, expected)


def test_one_forward_batch_one_and_model_state_hook_identity_are_preserved(estimator):
    model = make_model()
    before = hook_counts(model)
    weights = tuple(model._hf_model.parameters())
    values = [weight.clone() for weight in weights]
    forwards = []

    def observe(_module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        forwards.append(tuple(ids.shape))
        assert not kwargs.get("output_hidden_states", False)
        assert kwargs.get("use_cache") is False

    spy = model._text_module.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        with torch.no_grad():
            actual, _, _ = estimator(
                model, PROMPT, [1, 3], target_layer=4, dim_batch=DIM_BATCH, max_seq_len=SEQ
            )
    finally:
        spy.remove()
    assert forwards == [(1, SEQ)]
    assert hook_counts(model) == before
    assert all(weight is original for weight, original in zip(
        model._hf_model.parameters(), weights, strict=True
    ))
    assert all(weight.grad is None and not weight.requires_grad for weight in weights)
    assert all(torch.equal(weight, saved) for weight, saved in zip(weights, values, strict=True))
    assert all(not value.requires_grad for value in actual.values())


def scalar_selector_oracle(model, selected, *, source_layers=(1, 3), target_layer=4):
    """Independent oracle: sum separate scalar-target derivatives, then average source sites.

    Unlike the implementation's batched cotangents this differentiates one target coordinate
    at one position at a time. Three output rows suffice to expose either support being wrong.
    """
    output_dimensions = (0, 1, 7)
    positions = selected.nonzero(as_tuple=True)[0]
    rows = {layer: torch.zeros(len(output_dimensions), WIDTH) for layer in source_layers}
    with UPSTREAM.fitting.ActivationRecorder(
        model.layers, [*source_layers, target_layer], start_graph_at=min(source_layers)
    ) as recorder, torch.enable_grad():
        model.forward(model.encode(PROMPT, max_length=SEQ))
        target = recorder.activations[target_layer]
        inputs = [recorder.activations[layer] for layer in source_layers]
        for row, dimension in enumerate(output_dimensions):
            for target_position in positions:
                derivatives = torch.autograd.grad(
                    target[0, target_position, dimension], inputs, retain_graph=True
                )
                for layer, derivative in zip(source_layers, derivatives, strict=True):
                    rows[layer][row] += derivative[0, positions].float().mean(dim=0)
    return output_dimensions, rows


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_selector_replaces_both_target_sum_and_source_mean_support(estimator, attention):
    model = make_model(attention)
    selected = torch.zeros(SEQ, dtype=torch.bool)
    selected[[0, 8, 33, SEQ - 1]] = True  # Includes sites excluded by upstream's default rule.
    saved = selected.clone()
    calls = []

    def selector(length):
        calls.append(length)
        return selected

    actual, length, count = estimator(
        model, PROMPT, [1, 3], target_layer=4, dim_batch=DIM_BATCH,
        max_seq_len=SEQ, skip_first=SEQ, position_selector=selector,
    )
    assert (length, count) == (SEQ, int(selected.sum()))
    assert calls == [SEQ] and torch.equal(selected, saved)
    dimensions, oracle = scalar_selector_oracle(model, selected)
    for layer, expected in oracle.items():
        torch.testing.assert_close(
            actual[layer][list(dimensions)], expected, atol=EPSILON, rtol=EPSILON
        )


def test_selector_changes_are_local_and_never_replace_upstream_global_rule(estimator):
    model = make_model()
    original_rule = UPSTREAM.fitting.valid_position_mask
    selected = original_rule(SEQ)
    expected, _, original_count = UPSTREAM.fitting.jacobian_for_prompt(
        model, PROMPT, [1], target_layer=4, dim_batch=1, max_seq_len=SEQ
    )
    actual, _, count = estimator(
        model, PROMPT, [1], target_layer=4, dim_batch=7, max_seq_len=SEQ,
        position_selector=lambda _: selected,
    )
    assert count == original_count
    assert_maps_close(actual, expected)
    selected.zero_()
    selected[[0, SEQ - 1]] = True
    changed, _, changed_count = estimator(
        model, PROMPT, [1], target_layer=4, dim_batch=7, max_seq_len=SEQ,
        position_selector=lambda _: selected,
    )
    assert changed_count == 2
    assert not torch.allclose(changed[1], actual[1], atol=EPSILON, rtol=EPSILON)
    assert UPSTREAM.fitting.valid_position_mask is original_rule
    assert int(original_rule(SEQ).sum()) == original_count


@pytest.mark.parametrize("bad_selector", [
    lambda length: torch.zeros(length, dtype=torch.bool),
    lambda length: torch.ones(length, dtype=torch.int64),
    lambda length: torch.ones(1, length, dtype=torch.bool),
    lambda length: torch.ones(length - 1, dtype=torch.bool),
    lambda length: [True] * length,
    lambda length: None,
])
def test_invalid_selector_is_a_runtime_error_not_a_short_prompt_skip(estimator, bad_selector):
    model = make_model()
    before = hook_counts(model)
    with pytest.raises(RuntimeError, match="position_selector"):
        estimator(model, PROMPT, [1], position_selector=bad_selector)
    assert hook_counts(model) == before


def test_selector_failure_is_not_misreported_as_an_unusable_short_prompt(estimator):
    def failed_selector(length):
        raise ValueError("selector data is corrupt")

    with pytest.raises(RuntimeError, match="position_selector"):
        estimator(make_model(), PROMPT, [1], position_selector=failed_selector)


@pytest.mark.parametrize("kwargs", [
    {"dim_batch": 0}, {"dim_batch": -1}, {"dim_batch": True},
    {"max_seq_len": 0}, {"skip_first": -1},
    {"target_layer": 1}, {"target_layer": -BLOCKS - 1},
])
def test_invalid_arguments_fail_before_leaking_hooks(estimator, kwargs):
    model = make_model()
    before = hook_counts(model)
    with pytest.raises(ValueError):
        estimator(model, PROMPT, [1], **kwargs)
    assert hook_counts(model) == before


@pytest.mark.parametrize("sources", [[], [-BLOCKS - 1], [BLOCKS]])
def test_invalid_source_layers_follow_upstream_validation(estimator, sources):
    model = make_model()
    with pytest.raises(ValueError):
        estimator(model, PROMPT, sources)
    assert hook_counts(model) == [(0, 0)] * BLOCKS


def test_explicit_position_rule_and_truncation_match_upstream(estimator):
    model = make_model()
    expected = UPSTREAM.fitting.jacobian_for_prompt(
        model, PROMPT, [1], target_layer=4, dim_batch=1, max_seq_len=18, skip_first=0
    )
    actual = estimator(
        model, PROMPT, [1], target_layer=4, dim_batch=7, max_seq_len=18, skip_first=0
    )
    assert actual[1:] == expected[1:] == (18, 17)
    assert_maps_close(actual[0], expected[0])


def test_native_forward_failure_removes_every_owned_recorder_hook(estimator, monkeypatch):
    model = make_model()
    before = hook_counts(model)
    forward = model.forward

    def fail_after_forward(ids):
        forward(ids)
        raise RuntimeError("fixture forward failed")

    monkeypatch.setattr(model, "forward", fail_after_forward)
    with pytest.raises(RuntimeError, match="fixture forward failed"):
        estimator(model, PROMPT, [1, 3], target_layer=4)
    assert hook_counts(model) == before
    assert all(parameter.grad is None for parameter in model._hf_model.parameters())


def test_backward_failure_removes_every_owned_recorder_hook(estimator, monkeypatch):
    model = make_model()
    before = hook_counts(model)

    def failed_backward(*args, **kwargs):
        raise RuntimeError("fixture backward failed")

    monkeypatch.setattr(torch.autograd, "grad", failed_backward)
    with pytest.raises(RuntimeError, match="fixture backward failed"):
        estimator(model, PROMPT, [1, 3], target_layer=4)
    assert hook_counts(model) == before
    assert all(parameter.grad is None for parameter in model._hf_model.parameters())


def timed_runs(operation):
    times, result = [], None
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = operation()
        times.append(time.perf_counter() - start)
    return times, result


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_warmed_end_to_end_batching_ratio_is_at_least_one(estimator, attention, capsys):
    model = make_model(attention, seq_len=TIMING_SEQ)
    arguments = {"target_layer": 4, "max_seq_len": TIMING_SEQ}

    def batched():
        return estimator(model, PROMPT, [1, 3], dim_batch=DIM_BATCH, **arguments)

    def sequential():
        return UPSTREAM.fitting.jacobian_for_prompt(
            model, PROMPT, [1, 3], dim_batch=1, **arguments
        )

    # Warm both complete paths, including first-use autograd/vmap work, before timing either.
    batched()
    sequential()
    batched_seconds, actual = timed_runs(batched)
    sequential_seconds, expected = timed_runs(sequential)
    assert actual[1:] == expected[1:]
    errors = assert_maps_close(actual[0], expected[0])
    ratio = min(sequential_seconds) / min(batched_seconds)
    with capsys.disabled():
        print("\nJACOBIAN_TIMING " + json.dumps({
            "attention": attention, "tokens": TIMING_SEQ, "width": WIDTH, "blocks": BLOCKS,
            "source_blocks": [1, 3], "target_block": 4, "dim_batch": DIM_BATCH,
            "batched_seconds": batched_seconds, "sequential_seconds": sequential_seconds,
            "best_of": REPEATS, "sequential_over_batched": ratio,
            "errors_by_layer": errors,
        }, sort_keys=True))
    assert ratio >= 1.0, f"{attention} end-to-end batching ratio regressed: {ratio:.6f}x"


@pytest.mark.parametrize("failure", ["encoded_shape", "residual_shape"])
def test_model_runtime_shape_failures_are_not_short_prompt_skips(estimator, monkeypatch, failure):
    model = make_model()
    if failure == "encoded_shape":
        monkeypatch.setattr(model, "encode", lambda *args, **kwargs: torch.arange(SEQ))
        message = "model.encode"
    else:
        # The declared width differs from the real model output, after a genuine HF forward.
        model.d_model = WIDTH + 1
        message = "recorded residual shapes"
    with pytest.raises(RuntimeError, match=message):
        estimator(model, PROMPT, [1, 3], target_layer=4)
    assert hook_counts(model) == [(0, 0)] * BLOCKS


def test_nonfinite_gradient_is_a_runtime_failure_not_a_short_prompt_skip(estimator, monkeypatch):
    model = make_model()
    gradient = torch.autograd.grad

    def nonfinite_gradient(*args, **kwargs):
        values = gradient(*args, **kwargs)
        return (values[0] * float("nan"), *values[1:])

    monkeypatch.setattr(torch.autograd, "grad", nonfinite_gradient)
    with pytest.raises(RuntimeError, match="non-finite Jacobian"):
        estimator(model, PROMPT, [1, 3], target_layer=4)
    assert hook_counts(model) == [(0, 0)] * BLOCKS


def tensor_paths(value, path):
    """Inspect owned local tensors/containers, never attributes of the caller's model."""
    if torch.is_tensor(value):
        return [path]
    if isinstance(value, dict):
        return [
            found for key, item in value.items()
            for found in tensor_paths(item, f"{path}[{key}]")
        ]
    if isinstance(value, (list, tuple)):
        return [
            found for index, item in enumerate(value)
            for found in tensor_paths(item, f"{path}[{index}]")
        ]
    return []


@pytest.mark.parametrize("failure", ["backward", "nonfinite_result"])
def test_retained_exception_releases_only_the_estimators_owned_tensor_locals(
    estimator, monkeypatch, failure
):
    """An exception may remain logged. Its estimator frame must not retain our tape/workspace.

    Deeper torch and caller frames can retain their own references; this test deliberately makes
    no assertion about those. The caller-supplied model itself remains live and unchanged.
    """
    model = make_model()
    gradient = torch.autograd.grad

    def fail(*args, **kwargs):
        if failure == "backward":
            raise RuntimeError("retained fixture backward exception")
        values = gradient(*args, **kwargs)
        return tuple(value * float("nan") for value in values)

    monkeypatch.setattr(torch.autograd, "grad", fail)
    with pytest.raises(RuntimeError) as caught:
        estimator(model, PROMPT, [1, 3], target_layer=4, dim_batch=7)
    traceback = caught.value.__traceback__  # Deliberately retain the exception and its frames.
    owned_frame = None
    while traceback is not None:
        if traceback.tb_frame.f_code is estimator.__code__:
            owned_frame = traceback.tb_frame
            break
        traceback = traceback.tb_next
    assert owned_frame is not None
    owned_locals = owned_frame.f_locals
    retained = [
        path for name, value in owned_locals.items() if name != "model"
        for path in tensor_paths(value, name)
    ]
    assert retained == [], f"estimator exception frame retained owned tensors: {retained}"
    recorder = owned_locals.get("recorder")
    assert recorder is None or recorder.activations == {}
    assert hook_counts(model) == [(0, 0)] * BLOCKS
    assert all(parameter.grad is None for parameter in model._hf_model.parameters())


def test_previous_source_gradients_are_released_before_the_next_backward(estimator, monkeypatch):
    model = make_model()
    gradient = torch.autograd.grad
    previous, checked = [], []

    def watched_gradient(*args, **kwargs):
        if previous:
            alive = [index for index, reference in enumerate(previous) if reference() is not None]
            checked.append(alive)
            assert alive == [], f"source gradients remained live into the next backward: {alive}"
        values = gradient(*args, **kwargs)
        previous[:] = [weakref.ref(value) for value in values]
        return values

    monkeypatch.setattr(torch.autograd, "grad", watched_gradient)
    maps, _, _ = estimator(model, PROMPT, [1, 3], target_layer=4, dim_batch=7)
    assert checked and all(not alive for alive in checked)
    assert all(reference() is None for reference in previous)
    assert all(bool(torch.isfinite(value).all()) for value in maps.values())
