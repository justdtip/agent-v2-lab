"""The unit conversion no other test in the tree catches.

Every arm config counts iterations in micro-batches and says so in `iters_unit`, which nothing
reads. Handing those numbers to `transformers`, which counts optimizer steps, trains an arm
`grad_accumulation_steps` times too long and evaluates it that many times too often -- and both
runs finish, report a loss, and write a checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from local_llm_lab.training.torch_config import (
    TrainingConfigError,
    load_full_finetune_config,
)

ARM_1 = Path("configs/agent_v2e_qwen35_4b_top8.yaml")


def _arm(**overrides):
    payload = yaml.safe_load(ARM_1.read_text(encoding="utf-8"))
    payload["train"].update(overrides)
    return payload


def test_the_real_arm_converts_to_the_steps_the_survey_names() -> None:
    """1,200 micro-batches at accumulation 4 is 300 optimizer steps, and the record says 1,200."""
    config = load_full_finetune_config(_arm(), root=Path("."))
    assert (config.source_iters, config.source_units) == (1200, "batches")
    assert config.max_steps == 300
    assert config.eval_steps == 100
    assert config.save_steps == 100
    # The full-fine-tuning analogue of `lora_layers: 8`, which is the arm this stream re-measures.
    assert config.trainable_top_layers == 8


def test_steps_are_taken_at_face_value_when_the_config_says_steps() -> None:
    config = load_full_finetune_config(_arm(iters_unit="steps"), root=Path("."))
    assert config.max_steps == 1200
    assert config.eval_steps == 400


def test_an_unknown_unit_is_refused_rather_than_assumed() -> None:
    """The field exists to mark a unit this reader may not know how to convert."""
    with pytest.raises(TrainingConfigError, match="iters_unit"):
        load_full_finetune_config(_arm(iters_unit="epochs"), root=Path("."))


def test_a_length_that_is_not_whole_in_steps_is_refused() -> None:
    with pytest.raises(TrainingConfigError, match="train.iters=1201"):
        load_full_finetune_config(_arm(iters=1201), root=Path("."))


@pytest.mark.parametrize("key", ["steps_per_eval", "save_every"])
def test_evaluation_and_checkpoint_cadence_must_be_whole(key: str) -> None:
    """`ckpt 800` names a checkpoint; a save cadence that lands elsewhere is a different run."""
    with pytest.raises(TrainingConfigError, match=key):
        load_full_finetune_config(_arm(**{key: 401}), root=Path("."))


def test_reporting_cadence_is_rounded_and_the_rounding_is_recorded() -> None:
    """Logging every 2.5 steps is not a thing; logging is also not part of the experiment."""
    config = load_full_finetune_config(_arm(), root=Path("."))
    assert config.logging_steps == 2
    assert any("reporting cadence rounded" in line for line in config.departures)


def test_every_departure_from_the_mlx_record_is_stated() -> None:
    """R60-shaped: the run manifest carries these, so a comparison is never made silently."""
    departures = " | ".join(load_full_finetune_config(_arm(), root=Path(".")).departures)
    assert "num_items_in_batch" in departures      # token-weighted against unweighted
    assert "length-sorts" in departures            # batch order and padding
    assert "rank" in departures                    # adapter fields full FT ignores
    assert "gated_delta_chunk" in departures       # a dense model has no recurrence


def test_a_config_without_a_train_section_is_refused() -> None:
    with pytest.raises(TrainingConfigError, match="train"):
        load_full_finetune_config({"model": "x"}, root=Path("."))
