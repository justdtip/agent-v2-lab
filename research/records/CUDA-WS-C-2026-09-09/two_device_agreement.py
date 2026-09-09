"""Do one device and two devices produce the same loss curve on the same rows?

The order's golden test, run on CPU because it can be. FSDP2 at world size one is not a valid rung
-- it silently zeros gradients for some parameter shapes in non-root units (pytorch #144045) -- so
this compares **plain single-process training against two-process FSDP2**, which is the comparison
the single-device path actually makes, and the record says so rather than implying one code path.

**Comparability is the whole design.** Both arms must optimise the same objective on the same rows:

* The global accumulation window is `accum` micro-batches either way. At world size `W` each rank
  takes `accum / W` of them, interleaved by rank, so together the ranks cover exactly the window the
  single process covers, in the same order.
* Each micro-batch contributes a **summed** token loss. At the end of the window the sum is divided
  by the window's **global** supervised-token count, all-reduced across ranks. A per-rank mean would
  weight a rank holding short rows equally with one holding long rows.
* FSDP2 reduce-scatters gradients with a **mean** over ranks, so each rank's loss is scaled by `W`
  to recover the sum. Without it the two-device gradient is `1/W` of the one-device gradient and the
  curves diverge in a way that looks like a sharding bug and is arithmetic.

**The root unit stays replicated, and that is a finding rather than a convenience.** Sharding the
root as well (`--shard-root`) fails, and the failure is a consequence of this stream's own loss
design: `causal_lm_chunked_loss` reaches for `lm_head.weight` and calls `model.model` directly, to
avoid materialising a 262,208-wide logit tensor. Both bypass the pre-forward hooks FSDP2 uses to
unshard a unit's parameters, so the embedding is still a `DTensor` when `F.embedding` meets a plain
`input_ids`:

    RuntimeError: aten.embedding.default got mixed torch.Tensor and DTensor, need to convert
    all torch.Tensor to DTensor before calling distributed operators!
    (transformers/models/gemma3/modeling_gemma3.py:117, in Gemma3TextModel.forward)

So the two designs trade against each other: chunked loss wants raw parameters, FSDP2 wants every
access to go through a module call. Sharding only the transformer blocks and leaving the embedding,
final norm and tied head replicated resolves it, and is the same "gathered embedding root unit" the
Research Division's memory arithmetic already carries as a per-device cost. The alternative -- call
`lm_head` as a module per chunk -- unshards and reshards the largest parameter in the model once per
chunk, and is why this is recorded as a decision rather than taken silently.

Gradient clipping is deliberately **off**: the gate exists to attribute any deviation to sharding,
and a clip is a second place the two arms could differ. The shape of the model is read from a JSON
file rather than written here, so this names no layer count and no width (plan 10.1).

    .venv/bin/python  research/records/.../two_device_agreement.py --mode plain  --out one.json
    .venv/bin/torchrun --nproc_per_node=2 ...two_device_agreement.py --mode fsdp2 --out two.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _rows(count: int, vocab: int, seed: int) -> list[dict]:
    """Rows of uneven length. Equal lengths would hide the token-weighting the design turns on."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(count):
        length = int(torch.randint(12, 40, (1,), generator=generator))
        ids = torch.randint(1, vocab, (length,), generator=generator).tolist()
        rows.append({"input_ids": ids, "prompt_length": length // 3})
    return rows


def _gathered(model: Any, attribute: str) -> dict[str, Any]:
    """Per-parameter tensors, sharded ones gathered, keyed by name.

    The amended gate compares **per parameter**, not a norm over all of them: a wrong slice hides
    inside a right norm, which is the same failure as a wrong gradient hiding inside a right loss.
    """
    out = {}
    for name, parameter in model.named_parameters():
        tensor = getattr(parameter, attribute, None)
        if tensor is None:
            continue
        tensor = tensor.detach()
        tensor = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor
        out[name] = tensor.double().clone()
    return out


def _build_model(config_path: Path, seed: int):
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    payload = {k: v for k, v in json.loads(config_path.read_text()).items() if not k.startswith("_")}
    torch.manual_seed(seed)
    return Gemma3ForCausalLM(Gemma3TextConfig(**payload)), payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("plain", "fsdp2", "fsdp2-hand-reduce", "fsdp2-unreduced"),
        required=True,
    )
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--accum", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--loss-chunk", type=int, required=True)
    parser.add_argument(
        "--dump", type=Path, required=True,
        help="Where to write per-parameter step-0 gradients and final values, for the comparison "
             "the amended gate makes.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    from local_llm_lab.training.collator import CausalCollator
    from local_llm_lab.training.torch_full import causal_lm_chunked_loss

    from local_llm_lab.training.torch_full import wrap_with_chunked_loss

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = args.mode.startswith("fsdp2")
    #: Shard the root unit too, holding the tied embedding, the final norm and the head together.
    shard_root = args.mode == "fsdp2"
    #: Root outside every unit, reduced by hand. Correct, and a second reduction path.
    hand_reduce = args.mode == "fsdp2-hand-reduce"
    #: THE NEGATIVE CONTROL. Root outside every unit and reduced by nothing: the configuration the
    #: old loss-only gate passed while its root gradients were 40% apart.
    if args.mode == "fsdp2-unreduced" and rank == 0:
        print("NEGATIVE CONTROL: the root unit is reduced by nothing and is expected to disagree")
    failure: dict[str, str] | None = None

    if distributed:
        if world < 2:
            raise SystemExit(
                "fsdp2 mode needs at least two processes: world size one is not a valid FSDP "
                "development rung (pytorch #144045). Launch with torchrun --nproc_per_node=2."
            )
        dist.init_process_group("gloo", rank=rank, world_size=world)
    if args.accum % world:
        raise SystemExit(f"accum {args.accum} must divide by world size {world}")

    inner, shape = _build_model(args.model_config, args.seed)
    # Both arms train the wrapper, so parameter names match between them and the sharded arm has a
    # unit boundary at the only place the chunked loss can be inside one.
    model = wrap_with_chunked_loss(inner, chunk_size=args.loss_chunk)
    mesh_repr = None
    try:
        if distributed:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard

            mesh = init_device_mesh("cpu", (world,))
            mesh_repr = str(mesh)
            for layer in inner.model.layers:
                fully_shard(layer, mesh=mesh)
            if shard_root:
                # The root unit: the tied embedding, the final norm and the head, together.
                fully_shard(model, mesh=mesh)
    except Exception as error:  # the record must say exactly where, with the error
        failure = {
            "stage": "fully_shard on a cpu mesh",
            "type": f"{type(error).__module__}.{type(error).__name__}",
            "message": str(error),
        }

    curve: list[dict] = []
    dumped: dict[str, dict[str, Any]] = {}
    if failure is None:
        rows = _rows(args.rows, shape["vocab_size"], args.seed)
        collate = CausalCollator(pad_token_id=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, foreach=False)
        per_rank = args.accum // world
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            # The window is a pure function of the step, so both arms build the identical list of
            # micro-batches without communicating; the ranks then interleave over it.
            start = step * args.accum * args.batch
            window = [
                [rows[(start + i * args.batch + j) % len(rows)] for j in range(args.batch)]
                for i in range(args.accum)
            ]
            mine = [window[i] for i in range(rank, args.accum, world)]
            assert len(mine) == per_rank, (len(mine), per_rank)

            token_total = torch.zeros((), dtype=torch.float64)
            for micro in mine:
                token_total += (collate(micro)["labels"] != -100).sum()
            if distributed:
                dist.all_reduce(token_total)

            reported = torch.zeros((), dtype=torch.float64)
            for micro in mine:
                batch = collate(micro)
                summed = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                    num_items_in_batch=1,
                )
                reported += summed.detach().double()
                (summed * world / token_total.item()).backward()
            if distributed:
                dist.all_reduce(reported)
            # The gradient, before any update, is what says whether the two arms optimise the same
            # objective. A loss curve can agree at step 0 and drift for either of two reasons; only
            # the gradient separates "the reduction is right and float summation reordered" from
            # "the reduction is wrong by a factor". Sharded gradients are gathered to compare.
            # FSDP2 communicates gradients only for parameters inside a unit. The root unit --
            # embedding, final norm, tied head -- is deliberately left replicated here, so nothing
            # reduces its gradient and each rank holds only its own half of the window. Reducing it
            # by hand is what makes the two arms the same optimiser. Mean, not sum, to match the
            # reduction FSDP applies to the sharded parameters against the same `* world` scaling.
            if hand_reduce:
                for name, parameter in model.named_parameters():
                    if ".layers." not in name and parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                        parameter.grad /= world

            # The amended gate's quantity, captured before the first update: per-parameter
            # gradients, sharded ones gathered. Step zero on identical weights is where a wrong
            # reduction is unambiguous, because nothing has diverged yet for any other reason.
            if step == 0:
                dumped["grad_step0"] = _gathered(model, "grad")

            # Split by whether the parameter lives inside a sharded unit. FSDP2 communicates
            # gradients only for parameters it manages; anything outside every unit keeps a purely
            # local gradient and is never reduced, which is invisible in a total norm.
            groups = {"sharded_blocks": 0.0, "root_unit": 0.0}
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                gradient = gradient.full_tensor() if hasattr(gradient, "full_tensor") else gradient
                bucket = "sharded_blocks" if ".layers." in name else "root_unit"
                groups[bucket] += float((gradient.double() ** 2).sum())
            optimizer.step()
            curve.append({
                "step": step,
                "loss": float(reported / token_total),
                "grad_norm": sum(groups.values()) ** 0.5,
                "grad_norm_blocks": groups["sharded_blocks"] ** 0.5,
                "grad_norm_root": groups["root_unit"] ** 0.5,
                "tokens": int(token_total),
            })

    if failure is None:
        dumped["value_final"] = _gathered(model, "data")
        if rank == 0:
            torch.save(dumped, args.dump)

    payload = {
        "mode": args.mode,
        "world_size": world,
        "backend": "gloo" if distributed else None,
        "mesh": mesh_repr,
        "torch": torch.__version__,
        "model_shape": shape,
        "settings": {
            "rows": args.rows, "steps": args.steps, "accum": args.accum, "batch": args.batch,
            "learning_rate": args.learning_rate, "seed": args.seed, "loss_chunk": args.loss_chunk,
        },
        "gradient_clipping": "off, so any deviation is attributable to sharding alone",
        "sharded": {
            "plain": "nothing; single process",
            "fsdp2": "blocks and the root unit (tied embedding, final norm, head)",
            "fsdp2-hand-reduce": "blocks only; root outside every unit, reduced by hand",
            "fsdp2-unreduced": "blocks only; root reduced by NOTHING (negative control: the "
                               "configuration the old loss-only gate passed)",
        }[args.mode],
        "failure": failure,
        "curve": curve,
    }
    if rank == 0:
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"mode": args.mode, "world": world, "failure": failure,
                          "steps": len(curve)}, indent=2))
    if distributed:
        dist.destroy_process_group()
    if failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
