"""The decoder edit preserves reconstruction error; re-encoding need not attain it."""

import json

import pytest

torch = pytest.importorskip("torch")
from local_llm_lab.sae_intervention import SAEIntervention  # noqa: E402


def dictionary():
    # JumpReLU, with a deliberately non-inverse decoder. Feature zero's direction
    # also excites feature one. Powers of two make the counterexample exact in fp32.
    bias = torch.tensor([0.5, -0.5, 1.0])
    decoder = torch.tensor([[2.0, 0.0], [1.0, 1.0], [0.0, 0.0]])

    def encoder(h):
        pre = (h - bias)[:2]
        return pre * (pre > 0.25).to(pre.dtype)

    return encoder, decoder, bias


def test_preserves_full_base_reconstruction_error_and_unselected_code():
    encoder, decoder, bias = dictionary()
    h = torch.tensor([1.5, 1.5, 4.0])
    original = h.clone()
    z = encoder(h)
    replaced_code = torch.tensor([3.0, 2.0])
    edit = SAEIntervention(encoder, decoder, bias, features=(0,), target_values=torch.tensor([3.0]))
    actual = edit(h)
    assert torch.equal(actual, torch.tensor([5.5, 3.5, 4.0]))
    epsilon = h - bias - decoder @ z
    assert torch.equal(actual - bias - decoder @ replaced_code, epsilon)
    assert torch.equal(h, original)
    assert not torch.equal(actual, bias + decoder @ replaced_code)


def test_reencoding_reports_missed_target_and_off_target_change_as_numbers():
    encoder, decoder, bias = dictionary()
    edit = SAEIntervention(encoder, decoder, bias, features=(0,), target_values=torch.tensor([3.0]))
    edit(torch.tensor([1.5, 1.5, 4.0]))
    record = edit.diagnostic_record()
    assert record["features"] == [0]
    assert record["before"] == [1.0]
    assert record["target"] == [3.0]
    assert record["achieved"] == [5.0]
    assert record["target_error"] == [2.0]
    assert record["off_target_features"] == [1]
    assert record["off_target_change"] == [2.0]
    json.dumps(record, allow_nan=False)
    record["achieved"][0] = -100
    assert edit.diagnostic_record()["achieved"] == [5.0]


def test_linear_callable_decoder_matches_matrix_and_zero_edit_is_identity():
    encoder, decoder, bias = dictionary()
    h = torch.tensor([1.5, 1.5, 4.0])
    for decode in (decoder, lambda z: decoder @ z):
        edit = SAEIntervention(
            encoder, decode, bias, features=(1, 0), target_values=encoder(h).flip(0)
        )
        assert torch.equal(edit(h), h)
        assert edit.diagnostic_record()["off_target_change"] == []


def test_jump_threshold_crossing_is_measured_and_never_claimed_as_attainment():
    encoder, decoder, bias = dictionary()
    edit = SAEIntervention(encoder, decoder, bias, features=(0,), target_values=torch.tensor([0.2]))
    # Feature zero initially below threshold, and the intervention crosses it.
    h = torch.tensor([0.625, -0.5, 1.0])
    actual = edit(h)
    diagnostic = edit.diagnostic_record()
    assert diagnostic["before"] == [0.0]
    assert diagnostic["achieved"][0] != diagnostic["target"][0]
    assert torch.equal(encoder(actual)[0], torch.tensor(diagnostic["achieved"][0]))


def test_encoder_cannot_mutate_base_and_diagnostics_do_not_retain_graph():
    _, decoder, bias = dictionary()

    def mutating_encoder(h):
        h.sub_(bias)
        return h[:2].relu()

    h = torch.tensor([1.5, 1.5, 4.0], requires_grad=True)
    target = torch.tensor([3.0], requires_grad=True)
    edit = SAEIntervention(mutating_encoder, decoder, bias, features=(0,), target_values=target)
    out = edit(h)
    assert torch.equal(h, torch.tensor([1.5, 1.5, 4.0]))
    out.sum().backward()
    assert target.grad.item() == 3.0
    assert h.grad is not None
    json.dumps(edit.diagnostic_record(), allow_nan=False)


@pytest.mark.parametrize("features", [(), (True,), (-1,), (0, 0), (2,)])
def test_invalid_feature_selection_fails_closed(features):
    encoder, decoder, bias = dictionary()
    with pytest.raises(ValueError, match="feature"):
        edit = SAEIntervention(
            encoder, decoder, bias, features=features, target_values=torch.ones(len(features))
        )
        edit(torch.ones(3))


@pytest.mark.parametrize(
    "fault",
    [
        "transpose",
        "bias_shape",
        "target_shape",
        "dtype",
        "nonfinite",
        "encoder_shape",
        "decoder_shape",
    ],
)
def test_shape_dtype_and_nonfinite_contracts(fault):
    encoder, decoder, bias = dictionary()
    target = torch.tensor([3.0])
    if fault == "transpose":
        decoder = decoder.T
    elif fault == "bias_shape":
        bias = torch.zeros(2)
    elif fault == "target_shape":
        target = torch.ones(2)
    elif fault == "dtype":
        target = target.double()
    elif fault == "nonfinite":
        target[0] = float("nan")
    elif fault == "encoder_shape":

        def encoder(h):
            return h.unsqueeze(0)
    elif fault == "decoder_shape":

        def decoder(z):
            return z

    with pytest.raises(ValueError):
        edit = SAEIntervention(encoder, decoder, bias, features=(0,), target_values=target)
        edit(torch.ones(3))


def test_failed_application_cannot_report_stale_diagnostic():
    encoder, decoder, bias = dictionary()
    edit = SAEIntervention(encoder, decoder, bias, features=(0,), target_values=torch.tensor([3.0]))
    with pytest.raises(RuntimeError, match="successful"):
        edit.diagnostic_record()
    edit(torch.ones(3))
    with pytest.raises(ValueError):
        edit(torch.full((3,), float("nan")))
    with pytest.raises(RuntimeError, match="successful"):
        edit.diagnostic_record()


def test_capture_applies_decoder_edit_at_requested_layer_and_position():
    from test_torch_capture import TorchCapture, fixture

    view, sink, ids = fixture()
    encoder, decoder, bias = dictionary()
    target = torch.tensor([3.0])
    edit = SAEIntervention(encoder, decoder, bias, features=(0,), target_values=target)
    baseline = view.model(ids).logits
    base = view.model.embed(ids)[0, 1] * 2
    expected = (base + decoder[:, 0] * (target[0] - encoder(base)[0])) * 3
    with TorchCapture(view, sink, layers=(1, 2)) as capture:
        capture.intervene(1, 1, edit)
        actual = capture(ids).logits
        torch.testing.assert_close(actual[0, 1], expected, rtol=0, atol=0)
        assert torch.equal(actual[0, (0, 2)], baseline[0, (0, 2)])


def test_reused_encoder_output_buffer_cannot_rewrite_before_diagnostic():
    encoder, decoder, bias = dictionary()
    buffer = torch.empty(2)

    def buffered_encoder(h):
        buffer.copy_(encoder(h))
        return buffer

    edit = SAEIntervention(
        buffered_encoder, decoder, bias, features=(0,), target_values=torch.tensor([3.0])
    )
    edit(torch.tensor([1.5, 1.5, 4.0]))
    diagnostic = edit.diagnostic_record()
    assert diagnostic["before"] == [1.0]
    assert diagnostic["achieved"] == [5.0]
    assert diagnostic["off_target_change"] == [2.0]


def test_sae_clamp_and_one_shot_through_real_hf_dynamic_cache():
    from test_arch_torch import make_model
    from test_torch_capture import Sink
    from transformers.cache_utils import DynamicCache

    from local_llm_lab.arch_torch import TorchArchitectureView
    from local_llm_lab.torch_capture import TorchCapture

    view = TorchArchitectureView.from_model(make_model())
    sink = Sink()
    cache = DynamicCache(config=view.model.config)
    width = view.hidden_size
    edit = SAEIntervention(
        lambda h: h,
        torch.eye(width),
        torch.zeros(width),
        features=(0, 1),
        target_values=torch.tensor([1.0, 2.0]),
    )
    ids = torch.tensor([[1, 2, 3]])
    with TorchCapture(view, sink, layers=(0, 1)) as capture:
        capture.intervene(1, 1, lambda h: h + 0.25)
        handle = capture.clamp(0, "emitted", edit)
        capture(ids, past_key_values=cache, emitted_positions=[])
        for absolute in (3, 4):
            assert cache.get_seq_length() == absolute
            capture(ids[:, :1], past_key_values=cache, emitted_positions=[absolute])
            torch.testing.assert_close(sink.rows[0][1][0, 0, :2], torch.tensor([1.0, 2.0]))
        record = capture.intervention_record
        assert [len(row["applications"]) for row in record] == [1, 2]
        assert all("diagnostic" in event for event in record[1]["applications"])
        assert [event["absolute_position"] for event in record[1]["applications"]] == [3, 4]
        handle.release()
        capture(ids[:, :1], past_key_values=cache)
        assert cache.get_seq_length() == 6
        assert len(capture.intervention_record[1]["applications"]) == 2
        # Release stops editing input residuals; it does not erase effects in the cache.
        assert torch.equal(sink.rows[0][1], view.embed(ids[:, :1]))
        handle = capture.clamp(0, 4, edit)
        with pytest.raises(ValueError, match="already cached"):
            capture(ids[:, :1], past_key_values=cache)
        assert cache.get_seq_length() == 6
        handle.release()


def test_diagnostic_differences_do_not_overflow_finite_fp32_readings():
    def encoder(h):
        return torch.where(h > 0, torch.full_like(h, -3e38), torch.zeros_like(h))

    edit = SAEIntervention(
        encoder,
        torch.tensor([[1e-38]]),
        torch.zeros(1),
        features=(0,),
        target_values=torch.tensor([3e38]),
    )
    edit(torch.zeros(1))
    record = edit.diagnostic_record()
    assert record["target_error"][0] == record["achieved"][0] - record["target"][0]
    json.dumps(record, allow_nan=False)
