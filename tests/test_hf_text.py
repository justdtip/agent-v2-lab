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


@pytest.fixture
def official_layout(tmp_path):
    """The real snapshot's shape, mirrored from its headers rather than a stand-in's.

    On disk `google/gemma-3-4b-it` declares `model_type: gemma3`, `architectures:
    [Gemma3ForConditionalGeneration]`, `text_config.model_type: gemma3_text`, keeps the text
    tower under `language_model.model.*` beside `vision_tower.*` and `multi_modal_projector.*`,
    and stamps `__metadata__: {format: pt}`. A fixture that saved a `Gemma3TextConfig` model
    would declare `Gemma3ForCausalLM` instead, a config shape no registered checkpoint has, and
    the loader path would be untested by construction (SWE-2, 2026-09-09). The tiny sizes are
    the only departure, and `AutoModelForCausalLM` on the wrapper config would still build the
    multimodal class, which is exactly what the loader must not do.
    """
    from transformers import AutoModelForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    text_config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=8,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(text_config)
    state = {
        hf_text.TEXT_PREFIX + name: tensor
        for name, tensor in model.state_dict().items()
        if name != "lm_head.weight"  # tied: the real snapshot does not store it either
    }
    state["vision_tower.vision_model.embeddings.patch_embedding.weight"] = torch.zeros(4, 3, 2, 2)
    state["multi_modal_projector.mm_input_projection_weight"] = torch.zeros(4, 16)
    root = tmp_path / "official-layout"
    root.mkdir()
    safetensors_torch.save_file(state, root / "model.safetensors", metadata={"format": "pt"})
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma3",
                "architectures": ["Gemma3ForConditionalGeneration"],
                "text_config": json.loads(text_config.to_json_string()),
                "vision_config": {"model_type": "siglip_vision_model", "hidden_size": 4},
            }
        )
    )
    return root, model


def test_the_official_layout_loads_the_text_tower_as_the_causal_lm_not_the_wrapper(
    official_layout,
):
    root, saved = official_layout
    model, report = hf_text.load_text_causal_lm(root, dtype="float32")
    assert report["architecture"] == "Gemma3ForCausalLM"
    assert report["model_type"] == "gemma3_text"
    assert report["wrapper"] is True and report["safetensors_format"] == ["pt"]
    assert report["other_prefixes"] == ["multi_modal_projector", "vision_tower"]
    assert report["tied_keys"] == {"lm_head.weight": "model.embed_tokens.weight"}
    ids = torch.tensor([[1, 2, 3, 4]])
    assert torch.allclose(model(ids).logits, saved(ids).logits)


def test_the_repositorys_mlx_conversion_is_refused_by_its_own_format(official_layout, tmp_path):
    """Same architecture, same text_config, same prefix: only `__metadata__` tells them apart."""
    root, _ = official_layout
    tensors = safetensors_torch.load_file(root / "model.safetensors")
    text_only = {k: v for k, v in tensors.items() if k.startswith(hf_text.TEXT_PREFIX)}
    mlx = tmp_path / "mlx-conversion"
    mlx.mkdir()
    safetensors_torch.save_file(text_only, mlx / "model.safetensors", metadata={"format": "mlx"})
    config = json.loads((root / "config.json").read_text())
    del config["vision_config"]
    (mlx / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="format \\['mlx'\\].*another runtime"):
        hf_text.checkpoint_metadata(mlx)
