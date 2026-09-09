"""How close were the two bfloat16 frameworks at the positions where they disagreed?

The precision-matched arm resolved twenty-one of the twenty-four gating flips to quantisation
and left three where MLX bfloat16 still produces the recorded token and torch bfloat16 does not.
"Torch differs from MLX at matched precision" is not yet "torch is wrong": two implementations
of the same arithmetic land either side of a tie all the time, and a tie is not a defect.

So this reads the **margin** at those positions in each framework: the top two tokens and the
gap between them, as probabilities. The reading is:

* **both frameworks near-tied** — the disagreement is arithmetic. Each is picking a coin-flip
  and they disagree about the coin. Nothing to fix, and the hard rule was never about this.
* **one framework confident and the other confident the other way** — that is a defect, and the
  position is worth taking apart.

The recorded probability cannot answer this: it belongs to the 4-bit model, which is the one
model here already known to disagree with both.

Box discipline: this loads two 7.3 GiB checkpoints in turn and refuses without a window.

    python scripts/flip_margin.py --records <dir> --positions <mlx-bf16-arm.json> --json <out>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "research" / "acceptance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import golden_trajectories as golden  # noqa: E402
import provenance  # noqa: E402
import tolerance as _tolerance  # noqa: E402

#: The ruled tie band, taken from the module that owns it so the two cannot drift.
TIE_ULPS = _tolerance.TIE_ULPS
#: One definition of the grid, in the module that owns the rule. Three copies of a device
#: comparison had already accepted a wrong device between them; a grid is the same hazard.
bf16_ulp = _tolerance.bf16_ulp


def _require_own_window() -> None:
    from local_llm_lab import runlock

    blocking = runlock.blocking_window()
    if blocking is not None:
        raise SystemExit(runlock.refusal_for_window(blocking))
    if runlock.read_window() is None:
        raise SystemExit(
            "refusing to load weights: no box window is announced. Announce one with "
            "`runlock run` and read `runlock status` back until it says running."
        )


def _sequences(episodes: list[Any], wanted: list[dict]) -> dict[tuple[str, int], list[int]]:
    """The teacher-forced sequence for every turn that carries one of these positions."""
    by_label = {episode.label: episode for episode in episodes}
    out: dict[tuple[str, int], list[int]] = {}
    for row in wanted:
        episode = by_label[row["episode"]]
        for turn in episode.turns:
            if turn.index == row["turn"]:
                out[(row["episode"], row["turn"])] = list(turn.prompt_ids) + list(turn.token_ids)
    return out


def _reading(top1: int, p1: float, top2: int, p2: float, logit1: float) -> dict:
    """One framework's answer at one position, with the grid taken from the value itself.

    ``ln(p1/p2)`` is exactly the logit gap, and the grid it must be read against is bfloat16's
    spacing **at the magnitude of the logit that set it**. Fixing that magnitude once was the
    defect this replaces: it made a 1.5-logit gap read as six ULPs at an assumed |v| ~ 32 and
    three at the true |v| in [64, 128), and "every gap is an exact multiple of the grid" cannot
    tell 0.25 from 0.5, because a multiple of the finer grid is a multiple of the coarser.
    """

    gap = math.log(p1 / p2) if p2 > 0 else float("inf")
    ulp = bf16_ulp(logit1)
    return {
        "top1": top1,
        "p1": p1,
        "top2": top2,
        "p2": p2,
        "logit_gap": gap,
        "top_logit": logit1,
        "exponent": None if logit1 == 0 else math.floor(math.log2(abs(logit1))),
        "bf16_ulp": ulp,
        "gap_ulps": gap / ulp,
    }


def mlx_margins(checkpoint: Path, sequences: dict, rows: list[dict]) -> dict[tuple, tuple]:
    import mlx.core as mx
    from mlx_lm import load

    model, _ = load(str(checkpoint))
    out = {}
    for row in rows:
        sequence = sequences[(row["episode"], row["turn"])]
        logits = model(mx.array(sequence)[None])
        # The prediction *of* position p is read at p-1, the same join the tolerance arm uses.
        raw = logits[0, row["position"] - 1]
        probabilities = mx.softmax(raw.astype(mx.float32))
        order = mx.argsort(-probabilities)[:2]
        mx.eval(probabilities, order, raw)
        ids = [int(i) for i in order.tolist()]
        out[(row["episode"], row["turn"], row["position"])] = _reading(
            ids[0],
            float(probabilities[ids[0]].item()),
            ids[1],
            float(probabilities[ids[1]].item()),
            float(raw[ids[0]].item()),
        )
    del model
    return out


def torch_margins(checkpoint: Path, sequences: dict, rows: list[dict]) -> dict[tuple, tuple]:
    import torch

    from local_llm_lab import device
    from local_llm_lab.hf_text import load_text_causal_lm
    from local_llm_lab.pipeline.lens_fitting.upstream import same_device

    device.pin(attention="eager")
    target = device.select()
    model, report = load_text_causal_lm(
        checkpoint, dtype="bfloat16", attn_implementation="eager", device=target
    )
    if not same_device(target, str(report.get("device"))):
        raise SystemExit(f"asked for {target}, loader reports {report.get('device')!r}")
    out = {}
    with torch.no_grad():
        for row in rows:
            sequence = sequences[(row["episode"], row["turn"])]
            ids = torch.tensor([sequence], device=target)
            logits = model(ids).logits
            raw = logits[0, row["position"] - 1]
            probabilities = torch.softmax(raw.float(), dim=-1)
            top = torch.topk(probabilities, 2)
            out[(row["episode"], row["turn"], row["position"])] = _reading(
                int(top.indices[0]),
                float(top.values[0]),
                int(top.indices[1]),
                float(top.values[1]),
                float(raw[int(top.indices[0])].item()),
            )
    del model
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--verdict", default="port", help="which verdict's rows to read")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="the port's snapshot directory; required off the laptop, because the weights cache "
        "is resolved from the primary checkout and the device keeps its under $HF_HOME",
    )
    parser.add_argument(
        "--torch-only",
        action="store_true",
        help="read the port only, for a box with no MLX: the card, whose own margin at a "
        "position is the second measurement the below-resolution class needs",
    )
    arguments = parser.parse_args(argv)

    from local_llm_lab.models import load_model_spec

    payload = json.loads(arguments.positions.read_text())
    rows = [row for row in payload["positions"] if row["verdict"] == arguments.verdict]
    if not rows:
        parser.error(f"no positions with verdict {arguments.verdict!r}")
    print(f"{len(rows)} position(s) with verdict {arguments.verdict!r}")

    episodes = golden.load_episodes(arguments.records)
    sequences = _sequences(episodes, rows)

    _require_own_window()
    started = time.monotonic()
    # Two containers of the same bfloat16 model, because neither framework reads the other's.
    # `gemma3-4b-bf16` is the MLX conversion and torch's loader refuses it (its safetensors
    # header declares `format: mlx`); `gemma3-4b-cuda-bf16` is the upstream HF repo the MLX
    # copy was converted from. Same weights and same dtype, which is what this arm needs.
    from huggingface_hub import snapshot_download

    from local_llm_lab import device as device_module
    from local_llm_lab.runlock import primary_checkout_root

    device_name = device_module.select()
    mlx_checkpoint = Path(load_model_spec("gemma3-4b-bf16").hf_id)
    if arguments.checkpoint is not None:
        torch_checkpoint = arguments.checkpoint
    else:
        cache = primary_checkout_root() / ".cache" / "huggingface" / "hub"
        torch_checkpoint = Path(
            snapshot_download(
                load_model_spec("gemma3-4b-cuda-bf16").hf_id,
                cache_dir=str(cache),
                local_files_only=True,
            )
        )
    if not arguments.torch_only:
        print(f"MLX bfloat16:   {mlx_checkpoint}")
    print(f"torch bfloat16: {torch_checkpoint}")

    mlx_top = {} if arguments.torch_only else mlx_margins(mlx_checkpoint, sequences, rows)
    torch_top = torch_margins(torch_checkpoint, sequences, rows)

    out = []
    for row in rows:
        key = (row["episode"], row["turn"], row["position"])
        reference = mlx_top.get(key)
        port = torch_top[key]
        # The port's margin **toward the token the reference prefers**: positive when the port
        # agrees, negative when it prefers the other. Signed, because the cross-device spread
        # this feeds is the distance between the two devices' preferences, not of two magnitudes.
        toward = row["recorded_token"] if reference is None else reference["top1"]
        signed = port["logit_gap"] if port["top1"] == toward else -port["logit_gap"]
        entry = {
            **row,
            "device": device_name,
            "reference": reference,
            "port": port,
            "port_signed_gap_toward_reference": signed,
            "port_signed_gap_ulps": signed / port["bf16_ulp"],
        }
        if reference is not None:
            entry["tie"] = reference["gap_ulps"] <= TIE_ULPS
            entry["reading"] = (
                f"tie: the reference's own top two are within {TIE_ULPS} ULP"
                if entry["tie"]
                else f"the reference is {reference['gap_ulps']:.1f} ULP clear"
            )
        out.append(entry)
        head = (
            f"\n{row['episode']} turn {row['turn']} position {row['position']}"
            f"\n  recorded (4-bit) {row['recorded_token']} at P={row['recorded_probability']:.6f}"
        )
        if reference is not None:
            head += (
                f"\n  mlx   bf16: {reference['top1']} | {reference['top2']} | gap "
                f"{reference['logit_gap']:.4f} at |v|={reference['top_logit']:.2f} "
                f"(2^{reference['exponent']}, ULP {reference['bf16_ulp']}) "
                f"= {reference['gap_ulps']:.1f} ULP"
            )
        head += (
            f"\n  torch bf16 on {device_name}: {port['top1']} | {port['top2']} | gap "
            f"{port['logit_gap']:.4f} at |v|={port['top_logit']:.2f} "
            f"(2^{port['exponent']}, ULP {port['bf16_ulp']}) = {port['gap_ulps']:.1f} ULP"
            f"\n  signed toward the reference: {signed:+.4f} "
            f"({entry['port_signed_gap_ulps']:+.1f} ULP)"
        )
        print(head + (f"\n  {entry['reading']}" if reference is not None else ""))

    print("\n" + "=" * 78)
    print(f"{len(out)} position(s) read on {device_name}")
    if arguments.json:
        arguments.json.write_text(
            json.dumps(
                {
                    "what": "top-two margins in each bfloat16 framework at the positions where "
                    "they disagreed, to separate a defect from a rounding tie",
                    "tie_ulps": TIE_ULPS,
                    "device": device_name,
                    "read": provenance.Measured(len(out), provenance.MEASURED_HERE).as_dict(),
                    "seconds": provenance.Measured(
                        round(time.monotonic() - started, 1),
                        provenance.MEASURED_HERE,
                        unit="s",
                    ).as_dict(),
                    "positions": out,
                },
                indent=1,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
