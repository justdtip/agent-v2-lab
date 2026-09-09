"""The *joined* gate: the train stage driving FSDP2, one process against two.

`two_device_agreement.py` checked the sharded path on its own. This checks the path the stage
actually takes -- `stage_train_torch` building the wrapper, the FSDP2 configuration and `Trainer`
-- because those are two claims and only the second one lifts the stage's multi-process refusal.

It runs the real stage rather than reassembling its pieces beside it. A gate on a reassembly is a
gate on a different object, which is the whole reason the refusal exists. Two things are therefore
overridden and both are named here rather than hidden:

* ``require_supported_distribution`` is neutralised. It is the refusal this gate exists to lift,
  and the only way to produce the number that lifts it is to run the thing it refuses.
* ``load_model_spec`` is pointed at a tiny checkpoint built here, saved in the shape every real
  registry entry has -- a `Gemma3ForConditionalGeneration` wrapper, vision tower included -- so the
  loader path under test is the one a real run takes.

The comparison is the Chief's amended gate: **per parameter**, gradients at step zero before any
update, and values after N steps. Not a norm, because a wrong slice hides inside a right norm; not
the loss, because at step zero on identical weights it is identical by construction in every arm,
including a broken one.

    python  ...stage_agreement.py --mode plain --out one.json --dump one.pt
    torchrun --nproc_per_node=2 ...stage_agreement.py --mode fsdp2 --out two.json --dump two.pt
    python  ...stage_agreement.py --compare one.pt two.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _gathered(model: Any, attribute: str) -> dict[str, Any]:
    """Per-parameter tensors, sharded ones gathered, keyed by name."""
    out = {}
    for name, parameter in model.named_parameters():
        tensor = getattr(parameter, attribute, None)
        if tensor is None:
            continue
        tensor = tensor.detach()
        tensor = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor
        out[name] = tensor.double().clone()
    return out


def _make_callback(store: dict[str, Any]) -> Any:
    """A `Trainer` callback that captures gradients once, before the first optimizer step."""
    from transformers import TrainerCallback

    class StepZeroGradients(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            if "grad_step0" not in store and model is not None:
                store["grad_step0"] = _gathered(model, "grad")
            return control

    return StepZeroGradients()


def _tiny_wrapper_checkpoint(where: Path, tokenizer_source: Path, seed: int) -> Path:
    import torch
    from transformers import AutoTokenizer, Gemma3Config, Gemma3ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
    torch.manual_seed(seed)
    config = Gemma3Config(
        text_config=dict(
            vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=6,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=8,
            rms_norm_eps=1e-6, use_cache=False,
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
            image_size=16, patch_size=8, num_channels=3,
        ),
    )
    Gemma3ForConditionalGeneration(config).to(torch.bfloat16).save_pretrained(where)
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


def compare(one: Path, two: Path) -> dict[str, Any]:
    import torch

    a, b = torch.load(one, weights_only=False), torch.load(two, weights_only=False)
    result: dict[str, Any] = {}
    for key in ("grad_step0", "value_final"):
        if key not in a or key not in b:
            result[key] = {"status": "absent in one arm"}
            continue
        rows = []
        for name, left in a[key].items():
            right = b[key][name]
            scale = left.abs().max().item()
            rows.append((name, (left - right).abs().max().item() / (scale if scale > 0 else 1.0)))
        worst_name, worst = max(rows, key=lambda r: r[1])
        result[key] = {
            "parameters": len(rows),
            "worst_relative": worst,
            "worst_parameter": worst_name,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plain", "fsdp2"))
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--tokenizer-source", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()

    if args.compare:
        print(json.dumps(compare(*args.compare), indent=2))
        return

    import torch

    import local_llm_lab.models as models
    import local_llm_lab.pipeline.train_torch as train_torch

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "checkpoint"
    if rank == 0 and not checkpoint.exists():
        _tiny_wrapper_checkpoint(checkpoint, args.tokenizer_source, args.seed)
    if world > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group("gloo", rank=rank, world_size=world)
        dist.barrier()  # every rank reads the checkpoint rank 0 wrote
    data = _dataset(work / "rendered", args.rows) if rank == 0 else work / "rendered"

    # Named overrides, for the reasons in the module docstring.
    train_torch.require_supported_distribution = lambda world_size: None

    # Delegate for every name but ours, and for ours return a *real* spec with only `hf_id`
    # replaced. A bare stand-in object breaks `pipeline.protocol`, which resolves a spec at import
    # and reads `.chat` off it -- the same lesson as the fixture rule, one level up: a stand-in that
    # lacks the real object's shape is not a substitute for it.
    import dataclasses

    _real_load = models.load_model_spec
    _template = _real_load("gemma3-4b-bf16")

    def _load(name: str):
        if name != "tiny-wrapper":
            return _real_load(name)
        return dataclasses.replace(_template, name=name, hf_id=str(checkpoint))

    models.load_model_spec = _load

    store: dict[str, Any] = {}
    output = work / f"run-{args.mode}"
    config = {
        "model": "tiny-wrapper", "data": str(data), "output": output, "seed": args.seed,
        "train": {
            "iters": 8, "iters_unit": "batches", "grad_accumulation_steps": 2, "batch_size": 1,
            "learning_rate": 1e-4, "max_seq_length": 128, "val_batches": 0,
            "steps_per_report": 2, "steps_per_eval": 8, "save_every": 4, "lora_layers": 2,
        },
    }
    manifest = train_torch.stage_train_torch(config, callbacks=[_make_callback(store)])

    trained = train_torch  # keep the module referenced for clarity in tracebacks
    del trained
    from local_llm_lab.hf_text import load_text_causal_lm

    model, _ = load_text_causal_lm(output / "checkpoints" / "checkpoint-4", device="cpu")
    store["value_final"] = {
        name: p.detach().double().clone() for name, p in model.named_parameters()
    }
    if rank == 0:
        torch.save(store, args.dump)
        args.out.write_text(json.dumps({
            "mode": args.mode, "world_size": world, "torch": torch.__version__,
            "fsdp": manifest["recipe"], "load": manifest["load"],
            "result": manifest["result"],
        }, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"mode": args.mode, "world": world,
                          "step": manifest["result"]["global_step"]}, indent=2))
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
