"""G-2(b) at ``cache_strategy: none``: the first number in this migration not true by construction.

Loads the bf16 checkpoint on CPU, reads every deciding position of the recorded episodes with
the recorded prefix in front of it, and compares the argmax the port produces against the one
MLX recorded. Teacher-forced, so a disagreement at one position cannot cascade into the next
and every position is an independent comparison.

**What a disagreement means depends entirely on the recorded confidence.** A flip where the
model's own probability was at least 0.99 fails the run: quantisation between MLX 4-bit and CPU
bf16 does not move an argmax that confident, so such a flip is a mask, position, entry or norm
defect. A flip at a near-tie is arithmetic and is reported, not gated. Four fifths of this
corpus clears the 0.99 bar, so the rule governs most of it.

**This is not a reproduction test.** Free-running token-for-token agreement across backends is
struck from the gates: one flipped argmax separates two trajectories completely, and over
thousands of tokens the probability of zero flips is effectively zero. Exact reproduction is
asked of one backend against itself, which is a different run.

Box discipline: this loads 7.3 GiB of weights and refuses without an announced window.

    python scripts/tolerance_baseline.py --records <dir> --episode calculate-0158
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "research" / "acceptance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import golden_trajectories as golden  # noqa: E402
import tolerance  # noqa: E402

#: The registry entry whose weights torch can load. The pilot ran MLX 4-bit; this is the same
#: base at a different precision, and that difference is part of what is being measured. The
#: path comes from the registry rather than from this file, so a run inside a worktree resolves
#: to the same weights as one in the primary checkout.
CHECKPOINT_ENTRY = "gemma3-4b-bf16"


def _require_own_window() -> None:
    """Refuse unless *this* process holds the window, not merely that one is open.

    An earlier draft asked only whether a window existed, which would have loaded 7.3 GiB
    inside another seat's slot: the check passed precisely because somebody else was holding
    it. `blocking_window` is the right question, because it returns None both when nothing is
    open and when the open one is ours, so the two cases have to be separated.
    """
    from local_llm_lab import runlock

    blocking = runlock.blocking_window()
    if blocking is not None:
        raise SystemExit(runlock.refusal_for_window(blocking))
    held = runlock.read_window()
    if held is None:
        raise SystemExit(
            "refusing to load 7.3 GiB of weights: no box window is announced. Announce one, "
            "read `runlock status` back until it says running, and re-run. Announcing is not "
            "holding."
        )
    print(f"box window (ours): {held.seat} -- {held.purpose}")


def _load(checkpoint: Path):
    import torch
    from transformers import AutoTokenizer, Gemma3ForCausalLM

    from local_llm_lab import device
    from local_llm_lab.arch_torch import TorchArchitectureView

    reading = device.pin(attention="eager")
    print(f"determinism: {reading.get('determinism')}, attention: eager, dtype bfloat16 on cpu")

    started = time.monotonic()
    model = Gemma3ForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    view = TorchArchitectureView.from_model(model)
    print(
        f"loaded in {time.monotonic() - started:.1f}s: {view.num_layers} layers, "
        f"hidden {view.hidden_size}, vocab {view.vocab_size}"
    )
    return model, tokenizer, view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--episode", action="append", help="restrict to these episode labels")
    parser.add_argument("--checkpoint", type=Path, help="overrides the registry entry")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", type=Path, help="write the report here")
    arguments = parser.parse_args(argv)

    episodes = golden.load_episodes(arguments.records)
    if arguments.episode:
        wanted = set(arguments.episode)
        episodes = [episode for episode in episodes if episode.label in wanted]
    if not episodes:
        parser.error("no episodes selected")

    _require_own_window()
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.torch_capture import TorchCapture

    checkpoint = arguments.checkpoint or Path(load_model_spec(CHECKPOINT_ENTRY).hf_id)
    model, _, view = _load(checkpoint)

    class _Sink:
        """The capture needs a sink; teacher forcing reads logits and no residuals."""

        def residual(self, layer, offset, h) -> None:
            pass

        def output(self, offset, ids, logits) -> None:
            pass

    reports = []
    with TorchCapture(view, _Sink(), layers=(view.num_layers,)) as wrapped:
        forward = tolerance.torch_forward_rows(wrapped, view, top_k=arguments.top_k)
        for episode in episodes:
            started = time.monotonic()
            report = tolerance.run_tolerance(episode, forward, top_k=arguments.top_k)
            elapsed = time.monotonic() - started
            print(f"\n{episode.label}  ({elapsed:.1f}s)")
            print(f"  {report.agreement.describe()}")
            print(f"  {report.jaccard.describe()}")
            print(f"  gate: {'PASS' if report.passed else 'FAIL'}")
            reports.append((episode, report, elapsed))

    compared = sum(report.agreement.compared for _, report, _ in reports)
    agreed = sum(report.agreement.agreed for _, report, _ in reports)
    hard = sum(len(report.agreement.hard_flips) for _, report, _ in reports)
    print("\n" + "=" * 78)
    print(
        f"teacher-forced argmax agreement {agreed}/{compared} "
        f"({agreed / compared if compared else 0:.6f}); "
        f"{hard} flips at P >= {tolerance.HARD_CONFIDENCE}; "
        f"gate {'PASS' if hard == 0 else 'FAIL'}"
    )

    if arguments.json:
        arguments.json.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "records_precision": "mlx 4-bit",
                    "port_precision": "cpu bfloat16",
                    "compared": compared,
                    "agreed": agreed,
                    "hard_flips": hard,
                    "episodes": [
                        {
                            "label": episode.label,
                            "compared": report.agreement.compared,
                            "agreed": report.agreement.agreed,
                            "hard_flips": len(report.agreement.hard_flips),
                            "jaccard_mean": report.jaccard.mean,
                            "seconds": round(elapsed, 1),
                        }
                        for episode, report, elapsed in reports
                    ],
                },
                indent=1,
            )
            + "\n"
        )
    return 0 if hard == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
