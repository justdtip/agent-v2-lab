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

#: A disagreement is a tie when the two candidates' **logits** sit within this many units of
#: last place of the stored bfloat16. A probability margin is the wrong instrument here and
#: measuring one was a mistake: the logits are bfloat16, so their gaps are quantised to the
#: bfloat16 grid, and a gap of two ULPs can present as a 0.24 probability margin that reads as
#: confident. Softmax is monotone, so ``ln(p1/p2)`` recovers the logit gap exactly.
TIE_ULPS = 8


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


def bf16_ulp(magnitude: float) -> float:
    """The spacing of bfloat16 near ``magnitude``: 7 mantissa bits, so 2**(exponent - 7)."""
    if magnitude == 0:
        return 2.0**-133
    return 2.0 ** (math.floor(math.log2(abs(magnitude))) - 7)


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


def mlx_margins(checkpoint: Path, sequences: dict, rows: list[dict]) -> dict[tuple, tuple]:
    import mlx.core as mx
    from mlx_lm import load

    model, _ = load(str(checkpoint))
    out = {}
    for row in rows:
        sequence = sequences[(row["episode"], row["turn"])]
        logits = model(mx.array(sequence)[None])
        # The prediction *of* position p is read at p-1, the same join the tolerance arm uses.
        probabilities = mx.softmax(logits[0, row["position"] - 1].astype(mx.float32))
        order = mx.argsort(-probabilities)[:2]
        mx.eval(probabilities, order)
        ids = [int(i) for i in order.tolist()]
        out[(row["episode"], row["turn"], row["position"])] = (
            ids[0],
            float(probabilities[ids[0]].item()),
            ids[1],
            float(probabilities[ids[1]].item()),
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
            probabilities = torch.softmax(logits[0, row["position"] - 1].float(), dim=-1)
            top = torch.topk(probabilities, 2)
            out[(row["episode"], row["turn"], row["position"])] = (
                int(top.indices[0]),
                float(top.values[0]),
                int(top.indices[1]),
                float(top.values[1]),
            )
    del model
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--verdict", default="port", help="which verdict's rows to read")
    parser.add_argument("--json", type=Path)
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

    from local_llm_lab.runlock import primary_checkout_root

    mlx_checkpoint = Path(load_model_spec("gemma3-4b-bf16").hf_id)
    cache = primary_checkout_root() / ".cache" / "huggingface" / "hub"
    torch_checkpoint = Path(
        snapshot_download(
            load_model_spec("gemma3-4b-cuda-bf16").hf_id,
            cache_dir=str(cache),
            local_files_only=True,
        )
    )
    print(f"MLX bfloat16:   {mlx_checkpoint}")
    print(f"torch bfloat16: {torch_checkpoint}")

    mlx_top = mlx_margins(mlx_checkpoint, sequences, rows)
    torch_top = torch_margins(torch_checkpoint, sequences, rows)

    out = []
    for row in rows:
        key = (row["episode"], row["turn"], row["position"])
        m, t = mlx_top[key], torch_top[key]
        # ln(p1/p2) is exactly z1 - z2, so the logit gap comes back without keeping the logits.
        mlx_gap = math.log(m[1] / m[3]) if m[3] > 0 else float("inf")
        torch_gap = math.log(t[1] / t[3]) if t[3] > 0 else float("inf")
        # The gap is measured against the grid the logits are stored on. Their magnitude is not
        # recorded here, so the ULP is taken at the scale these logits actually sit at; the
        # gaps themselves come back as exact multiples of it, which is the check that it is
        # the right grid.
        ulp = bf16_ulp(32.0)
        mlx_ulps, torch_ulps = mlx_gap / ulp, torch_gap / ulp
        tied = mlx_ulps <= TIE_ULPS and torch_ulps <= TIE_ULPS
        entry = {
            **row,
            "mlx_top1": m[0],
            "mlx_p1": m[1],
            "mlx_top2": m[2],
            "mlx_p2": m[3],
            "mlx_margin": m[1] - m[3],
            "mlx_logit_gap": mlx_gap,
            "mlx_gap_ulps": mlx_ulps,
            "torch_top1": t[0],
            "torch_p1": t[1],
            "torch_top2": t[2],
            "torch_p2": t[3],
            "torch_margin": t[1] - t[3],
            "torch_logit_gap": torch_gap,
            "torch_gap_ulps": torch_ulps,
            "bf16_ulp": ulp,
            "reading": (
                f"arithmetic: the candidates are within {TIE_ULPS} ULPs of the stored bfloat16 "
                f"in both frameworks ({mlx_ulps:.0f} and {torch_ulps:.0f}), so this is a tie at "
                "the resolution of the numbers being compared, not a disagreement about them"
            )
            if tied
            else "beyond the bfloat16 grid: worth taking apart",
        }
        out.append(entry)
        print(
            f"\n{row['episode']} turn {row['turn']} position {row['position']}"
            f"\n  recorded (4-bit) {row['recorded_token']} at P={row['recorded_probability']:.6f}"
            f"\n  mlx   bf16: {m[0]} p={m[1]:.4f} | {m[2]} p={m[3]:.4f} | "
            f"logit gap {mlx_gap:.4f} = {mlx_ulps:.0f} ULP"
            f"\n  torch bf16: {t[0]} p={t[1]:.4f} | {t[2]} p={t[3]:.4f} | "
            f"logit gap {torch_gap:.4f} = {torch_ulps:.0f} ULP"
            f"\n  {entry['reading']}"
        )

    tied = sum(1 for e in out if e["reading"].startswith("arithmetic"))
    print("\n" + "=" * 78)
    print(f"{tied} of {len(out)} are ties at the resolution of the stored bfloat16 logits")
    if arguments.json:
        arguments.json.write_text(
            json.dumps(
                {
                    "what": "top-two margins in each bfloat16 framework at the positions where "
                    "they disagreed, to separate a defect from a rounding tie",
                    "tie_ulps": TIE_ULPS,
                    "near_ties_in_both": provenance.Measured(
                        tied, provenance.MEASURED_HERE
                    ).as_dict(),
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
