"""`hf_text`: a checkpoint's text decoder, plain or wrapped, loaded fail-closed, no class named.

Tiny random models on CPU, no MLX, no registry, no lock: this file is not in the MLX closure and
runs beside a seat's window without mapping Metal.
"""

from __future__ import annotations

import json

import pytest

from local_llm_lab import hf_text

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _tiny_config():
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )


@pytest.fixture
def plain(tmp_path):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(_tiny_config())
    root = tmp_path / "plain"
    model.save_pretrained(root, safe_serialization=True)
    return root, model


@pytest.fixture
def wrapped(plain, tmp_path):
    """The same tensors under `language_model.`, two foreign towers beside them, wrapper config."""
    root, _ = plain
    tensors = safetensors_torch.load_file(root / "model.safetensors")
    rekeyed = {hf_text.TEXT_PREFIX + name: tensor for name, tensor in tensors.items()}
    rekeyed["vision_tower.patch.weight"] = torch.zeros(3, 3)
    rekeyed["multi_modal_projector.linear.weight"] = torch.zeros(4, 4)
    out = tmp_path / "wrapped"
    out.mkdir()
    safetensors_torch.save_file(rekeyed, out / "model.safetensors")
    text_config = json.loads((root / "config.json").read_text())
    (out / "config.json").write_text(
        json.dumps({"model_type": "some-wrapper", "text_config": text_config})
    )
    return out, tensors


def test_a_plain_checkpoint_loads_as_itself_with_nothing_unexpected(plain):
    root, saved = plain
    model, report = hf_text.load_text_causal_lm(root, dtype="float32")
    assert report["wrapper"] is False and report["key_mapping"] is None
    assert report["unexpected_keys"] == [] and report["other_prefixes"] == []
    assert report["architecture"] == type(saved).__name__ and report["model_type"] == "llama"
    assert report["device"] == "cpu" and report["dtype"] == "float32"
    assert report["storage_dtypes"] == ["F32"]
    # The tied head is one tensor with two names, restored as such and not saved twice.
    assert report["tied_keys"] == {"lm_head.weight": "model.embed_tokens.weight"}
    assert "lm_head.weight" not in report["loading_info"].get("missing_keys", [])
    assert not any(p.requires_grad for p in model.parameters()) and not model.training
    ids = torch.tensor([[1, 2, 3]])
    assert torch.allclose(model(ids).logits, saved(ids).logits)


def test_a_wrapper_checkpoint_loads_its_text_tower_and_reports_the_other_towers_exactly(wrapped):
    root, tensors = wrapped
    model, report = hf_text.load_text_causal_lm(root, dtype="float32")
    assert report["wrapper"] is True
    assert report["key_mapping"] == {r"^language_model\.": ""}
    assert report["other_prefixes"] == ["multi_modal_projector", "vision_tower"]
    assert report["unexpected_keys"] == [
        "multi_modal_projector.linear.weight",
        "vision_tower.patch.weight",
    ]
    assert report["checkpoint_text_tensors"] == len(tensors)
    state = model.state_dict()
    for name, tensor in tensors.items():
        assert torch.equal(state[name], tensor), name
    meta = hf_text.checkpoint_metadata(root)
    assert meta["text_bytes"] == sum(t.numel() * 4 for t in tensors.values())


def test_dtype_and_attention_are_read_back_not_echoed(plain):
    root, _ = plain
    model, report = hf_text.load_text_causal_lm(root, dtype="bfloat16", attn_implementation="sdpa")
    assert report["dtype"] == "bfloat16" and report["attn_implementation"] == "sdpa"
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}


def test_a_missing_text_tensor_is_refused_not_zero_filled(wrapped):
    root, _ = wrapped
    tensors = safetensors_torch.load_file(root / "model.safetensors")
    del tensors[hf_text.TEXT_PREFIX + "model.layers.1.mlp.down_proj.weight"]
    safetensors_torch.save_file(tensors, root / "model.safetensors")
    with pytest.raises(ValueError, match="missing_keys"):
        hf_text.load_text_causal_lm(root, dtype="float32")


def test_a_shape_that_changed_in_transit_is_refused(wrapped):
    root, _ = wrapped
    tensors = safetensors_torch.load_file(root / "model.safetensors")
    key = hf_text.TEXT_PREFIX + "model.layers.0.mlp.up_proj.weight"
    tensors[key] = torch.zeros(tensors[key].shape[0] + 1, tensors[key].shape[1])
    safetensors_torch.save_file(tensors, root / "model.safetensors")
    with pytest.raises(ValueError, match="mismatch|shape"):
        hf_text.load_text_causal_lm(root, dtype="float32")


def test_a_wrapper_config_with_no_text_tensors_is_refused_before_loading(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"text_config": {"model_type": "llama"}}))
    safetensors_torch.save_file({"vision_tower.w": torch.zeros(2)}, root / "model.safetensors")
    with pytest.raises(ValueError, match="no tensor is under"):
        hf_text.checkpoint_metadata(root)


def test_no_shards_and_no_directory_are_refused(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="no local safetensors"):
        hf_text.checkpoint_metadata(empty)
    with pytest.raises(FileNotFoundError):
        hf_text.checkpoint_metadata(tmp_path / "absent")
