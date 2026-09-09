"""The golden test the order asks for: forty rows, on CPU, through the whole loop.

Loop, cadence, checkpoint and manifest end to end against a tiny randomly-initialised Gemma 3.
Nothing here loads a checkpoint, takes the model-run lock, or needs an accelerator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from local_llm_lab.pipeline.data import (  # noqa: E402
    DatasetManifestMissingError,
    require_dataset_manifest,
)
from local_llm_lab.training.collator import IGNORE_INDEX, CausalCollator  # noqa: E402
from local_llm_lab.training.torch_full import (  # noqa: E402
    build_adamw,
    freeze_all_but_top_layers,
    make_chunked_loss_trainer_class,
)

ROWS, VOCAB, LAYERS = 40, 64, 4


def _tiny_model():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    return Gemma3ForCausalLM(
        Gemma3TextConfig(
            vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=LAYERS,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=8,
            rms_norm_eps=1e-6, use_cache=False,
        )
    )


def _rows(seed: int = 0) -> list[dict]:
    """Forty rows of uneven length, because equal-length rows hide the reduction that matters."""
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for index in range(ROWS):
        length = int(torch.randint(12, 40, (1,), generator=generator))
        ids = torch.randint(1, VOCAB, (length,), generator=generator).tolist()
        rows.append({"input_ids": ids, "prompt_length": length // 3})
    return rows


def _dataset_dir(tmp_path: Path, rows: list[dict], *, stamp: bool = True) -> Path:
    directory = tmp_path / "rendered"
    directory.mkdir()
    with (directory / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    if stamp:
        (directory / "manifest.json").write_text("{}", encoding="utf-8")
    return directory


def test_an_unstamped_dataset_directory_stops_the_run_before_it_starts(tmp_path: Path) -> None:
    """Issue #73's commit point: rows without a manifest are a write that died mid-way."""
    with pytest.raises(DatasetManifestMissingError):
        require_dataset_manifest(_dataset_dir(tmp_path, _rows(), stamp=False))


def test_the_collator_masks_pads_out_of_both_attention_and_the_loss() -> None:
    """mlx-lm attended over its pads and discarded the logits; Gemma's masks make that unsafe."""
    batch = CausalCollator(pad_token_id=0)([
        {"input_ids": [5, 6, 7, 8], "prompt_length": 2},
        {"input_ids": [9, 10], "prompt_length": 1},
    ])
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch["labels"].tolist() == [
        [IGNORE_INDEX, IGNORE_INDEX, 7, 8],
        [IGNORE_INDEX, 10, IGNORE_INDEX, IGNORE_INDEX],
    ]


def test_the_mlx_pad_artefact_is_reproducible_and_off_by_default() -> None:
    """Every untruncated mlx-lm row supervises one pad token; that is inherited only on purpose."""
    rows = [{"input_ids": [5, 6, 7, 8], "prompt_length": 1}]
    default = CausalCollator()(rows)
    assert IGNORE_INDEX not in default["labels"][0, 1:].tolist()

    reproduced = CausalCollator(supervise_one_pad=True, pad_strategy="mlx")(rows)
    assert reproduced["input_ids"].shape[1] == 33  # 1 + 32*ceil(4/32)
    assert reproduced["labels"][0, 4].item() == 0  # the supervised pad column
    assert reproduced["labels"][0, 5].item() == IGNORE_INDEX


def test_forty_rows_run_the_loop_cadence_and_checkpoints_end_to_end(tmp_path: Path) -> None:
    from transformers import TrainingArguments

    rows = _rows()
    data = _dataset_dir(tmp_path, rows)
    require_dataset_manifest(data)  # the run refuses to start without this

    model = _tiny_model()
    opened = freeze_all_but_top_layers(model, 2)
    assert opened < sum(p.numel() for p in model.parameters())
    build_adamw(model, learning_rate=1e-4)  # refuses a dtype whose moments would be bfloat16

    output = tmp_path / "run"
    arguments = TrainingArguments(
        output_dir=str(output),
        max_steps=4,                      # optimizer steps, already converted from micro-batches
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        max_grad_norm=1.0,                # Trainer clips through the accelerator, not per-shard
        logging_steps=1,
        save_steps=2,
        save_strategy="steps",
        eval_strategy="no",
        report_to=[],
        use_cpu=True,
        seed=20260902,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = make_chunked_loss_trainer_class(chunk_size=16)(
        model=model,
        args=arguments,
        train_dataset=rows,
        data_collator=CausalCollator(pad_token_id=0),
    )
    result = trainer.train()

    assert result.global_step == 4
    assert result.training_loss > 0
    # The cadence actually fired: two checkpoints at save_steps=2 over four steps.
    checkpoints = sorted(p.name for p in output.glob("checkpoint-*"))
    assert checkpoints == ["checkpoint-2", "checkpoint-4"]
    assert (output / "checkpoint-4" / "model.safetensors").is_file()
    # And the loss was logged every step.
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    assert len(losses) == 4
    assert all(loss == loss for loss in losses)  # no NaN


def test_only_the_opened_layers_moved(tmp_path: Path) -> None:
    """A freeze that does not hold is the failure that trains a different experiment."""
    from transformers import TrainingArguments

    rows = _rows()
    model = _tiny_model()
    freeze_all_but_top_layers(model, 1)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    trainer = make_chunked_loss_trainer_class(chunk_size=16)(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "run"), max_steps=2, per_device_train_batch_size=2,
            learning_rate=1e-2, logging_steps=1, save_strategy="no", eval_strategy="no",
            report_to=[], use_cpu=True, seed=1,
        ),
        train_dataset=rows,
        data_collator=CausalCollator(pad_token_id=0),
    )
    trainer.train()

    moved = {n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])}
    assert moved, "nothing moved at all; the run did not train"
    assert all(name.startswith(f"model.layers.{LAYERS - 1}.") for name in moved), sorted(moved)
