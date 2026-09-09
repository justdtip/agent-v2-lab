"""The `train` stage's torch branch, end to end behind the device shim.

Builds a real (tiny) checkpoint on disk and real rendered rows, so the path under test is the one a
run takes: `device.backend()` dispatches, `device.pin()` runs before the model loads, the manifest
carries the determinism reading and every departure, and the cadence is the converted one.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from local_llm_lab import device, runlock  # noqa: E402
from local_llm_lab.pipeline.train_torch import MANIFEST_NAME  # noqa: E402

#: `models/` is gitignored and exists only in the primary checkout, while `project.PROJECT_ROOT` is
#: the *worktree* root. The registry's own comment says a stage running from a worktree "finds the
#: same file as one running from here", and for `hf_id: models/...` that is not true. This test
#: therefore reaches the shared checkout the way `runlock` does, and the mismatch is reported rather
#: than worked around in the resolver, which is WS-E's.
TOKENIZER_SOURCE = runlock.box_state_root() / "models" / "gemma-3-4b-it-bf16"


@pytest.fixture
def tiny_checkpoint(tmp_path: Path):
    """A Gemma 3 whose vocabulary matches the real tokenizer, so real rows tokenize into it."""
    from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3TextConfig

    if not (TOKENIZER_SOURCE / "tokenizer.json").is_file():
        pytest.skip(f"no tokenizer at {TOKENIZER_SOURCE}")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_SOURCE))

    torch.manual_seed(0)
    model = Gemma3ForCausalLM(
        Gemma3TextConfig(
            vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, sliding_window=8,
            rms_norm_eps=1e-6, use_cache=False,
        )
    ).to(torch.bfloat16)
    where = tmp_path / "checkpoint"
    model.save_pretrained(where)
    tokenizer.save_pretrained(where)
    return where


def _dataset(directory: Path, rows: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", rows), ("valid", 4)):
        with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(count):
                handle.write(json.dumps({
                    "prompt": f"user asks about item {index}. ",
                    "completion": f"the answer for item {index} is {index * 7}.",
                }) + "\n")
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    return directory


def _config(checkpoint: Path, data: Path, output: Path) -> dict:
    return {
        "model": "tiny-gemma3",
        "data": str(data),
        "output": output,
        "seed": 20260902,
        "train": {
            # mlx-lm units: 8 micro-batches at accumulation 2 is 4 optimizer steps.
            "iters": 8,
            "iters_unit": "batches",
            "grad_accumulation_steps": 2,
            "batch_size": 1,
            "learning_rate": 1e-4,
            "max_seq_length": 128,
            "val_batches": 0,
            "steps_per_report": 2,
            "steps_per_eval": 8,
            "save_every": 4,
            "lora_layers": 2,
            "rank": 16,
            "scale": 32.0,
            "gated_delta_chunk": 64,
        },
    }


def test_the_stage_dispatches_to_torch_and_records_what_it_did(
    tiny_checkpoint: Path, tmp_path: Path, monkeypatch
) -> None:
    import local_llm_lab.models as models
    from local_llm_lab.pipeline import cli

    monkeypatch.setenv("LLL_BACKEND", "torch")
    assert device.backend() == "torch"
    monkeypatch.setattr(
        models, "load_model_spec",
        lambda name: SimpleNamespace(name=name, hf_id=str(tiny_checkpoint)),
    )

    output = tmp_path / "run"
    config = _config(tiny_checkpoint, _dataset(tmp_path / "rendered", rows=8), output)

    cli.stage_train(config, iters=None)  # the CLI entry, so the branch itself is under test

    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert manifest["backend"] == "torch"

    # The cadence is the converted one: 8 micro-batches at accumulation 2 is 4 optimizer steps.
    assert manifest["recipe"]["source_iters"] == 8
    assert manifest["recipe"]["source_units"] == "batches"
    assert manifest["recipe"]["max_steps"] == 4
    assert manifest["recipe"]["save_steps"] == 2
    assert manifest["result"]["global_step"] == 4

    # device.pin ran, and what it returned is a reading rather than an echo of the request.
    assert isinstance(manifest["determinism"], dict) and manifest["determinism"]
    assert manifest["device"] == device.select()

    # Only the top two blocks train, and only they were upcast.
    assert 0 < manifest["parameters"]["trainable"] < manifest["parameters"]["total"]
    assert manifest["parameters"]["upcast_to_float32"] == manifest["parameters"]["trainable"]

    # Every way this run is not the MLX run of the same config is carried, not absorbed.
    departures = " | ".join(manifest["departures_from_the_mlx_record"])
    assert "num_items_in_batch" in departures
    assert "length-sorts" in departures
    assert "rank" in departures
    assert "gated_delta_chunk" in departures

    assert (output / "checkpoints" / "checkpoint-4" / "model.safetensors").is_file()


def test_the_mlx_backend_does_not_reach_the_torch_branch(monkeypatch) -> None:
    """The laptop's path must be the one it always was, reached by the same code."""
    monkeypatch.setenv("LLL_BACKEND", "mlx")
    assert device.backend() == "mlx"

    from local_llm_lab.pipeline import cli

    called = False

    def _fail(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("the torch branch was entered on the mlx backend")

    monkeypatch.setattr("local_llm_lab.pipeline.train_torch.stage_train_torch", _fail)
    with pytest.raises(Exception):  # it proceeds into the MLX path and fails on a bare config
        cli.stage_train({"train": {}, "model": "x", "output": Path("/nonexistent")}, iters=None)
    assert not called


def test_a_multi_process_run_is_refused_rather_than_distributed_by_default(monkeypatch) -> None:
    """The gap is named in code, because the alternative is invisible.

    FSDP2 is validated per parameter on CPU in the two-device record, and is not wired into this
    stage. Handed more than one process, `Trainer` and `accelerate` distribute under their own
    default: the loss would fall, a checkpoint would be written, and the memory arithmetic every
    device decision rests on would describe a configuration that never ran. A refusal in the first
    hour on rented hardware is the cheapest possible version of finding that out.
    """
    from local_llm_lab.pipeline.train_torch import require_supported_distribution

    require_supported_distribution(1)  # the single-device path is the one that runs
    with pytest.raises(NotImplementedError, match="world size 2"):
        require_supported_distribution(2)

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LLL_BACKEND", "torch")
    from local_llm_lab.pipeline import cli

    with pytest.raises(NotImplementedError, match="not wired here"):
        cli.stage_train({"train": {}, "model": "x", "output": Path("/nonexistent"), "seed": 1}, None)
