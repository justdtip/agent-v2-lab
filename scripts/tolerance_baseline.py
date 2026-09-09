"""G-2(b) at ``cache_strategy: none``: the first number in this migration not true by construction.

Loads the bf16 checkpoint on the device ``LLL_DEVICE`` selects, reads every deciding position
of the recorded episodes with the recorded prefix in front of it, and compares the argmax the
port produces against the one MLX recorded. Teacher-forced, so a disagreement at one position
cannot cascade into the next and every position is an independent comparison.

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
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "research" / "acceptance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gate_records  # noqa: E402
import golden_trajectories as golden  # noqa: E402
import provenance  # noqa: E402
import tolerance  # noqa: E402

#: The registry entry torch can load. **Not** ``gemma3-4b-bf16``: that one's weights are the
#: MLX conversion, whose safetensors header says ``format: mlx`` outright, and handing it to
#: ``from_pretrained`` would refuse or load something else. This entry is the upstream HF
#: checkpoint, already in the local cache, and it is the same bf16 tensors the MLX conversion
#: was made from, which is why the golden records correspond to it at all.
CHECKPOINT_ENTRY = "gemma3-4b-cuda-bf16"

#: The load goes through ``local_llm_lab.hf_text.load_text_causal_lm``, the package's one
#: text-only loader. This script previously carried its own copy of the key mapping, the vision
#: prefixes and the fail-closed checks; that copy is deleted rather than kept beside the
#: package's, and naming a model class here would breach section 16.3 in a script meant to
#: outlive this checkpoint.


def _episode_row(episode, report, elapsed: float) -> dict:
    """One episode's result, with every number saying what kind of number it is.

    ``confident_positions`` is the count that makes the hard rule readable: "0 flips at
    P >= 0.99" and "0 of 78 confident positions flipped" are the same fact and only the second
    can be read. It comes from the records rather than from this run, so it is a laptop basis
    even when the agreement beside it was measured on the device.
    """
    here = provenance.MEASURED_HERE
    confident = sum(
        1
        for turn in episode.turns
        for emission in turn.emissions
        if (turn.emitted_confidence(emission.position) or 0) >= tolerance.HARD_CONFIDENCE
    )
    return {
        "label": episode.label,
        "compared": provenance.Measured(report.agreement.compared, here).as_dict(),
        "agreed": provenance.Measured(report.agreement.agreed, here).as_dict(),
        "hard_flips": provenance.Measured(len(report.agreement.hard_flips), here).as_dict(),
        "flip_confidences": provenance.Measured(
            sorted(report.agreement.flip_confidences),
            provenance.LAPTOP_BASIS,
            basis="the MLX recording's own probability at each flipped position",
        ).as_dict(),
        "confident_positions": provenance.Measured(
            confident,
            provenance.LAPTOP_BASIS,
            basis=f"emissions recorded at P >= {tolerance.HARD_CONFIDENCE} in the MLX run",
        ).as_dict(),
        "jaccard_mean": provenance.Measured(report.jaccard.mean, here).as_dict(),
        "seconds": provenance.Measured(round(elapsed, 1), here, unit="s").as_dict(),
    }


def run_episodes(episodes, forward, *, top_k: int = 5, per_episode: Path | None = None):
    """Read every episode, writing each result the moment it is finished.

    The first version of this accumulated everything and wrote once at the end, with stdout
    block-buffered because it was not a terminal. An interrupt at ten minutes therefore
    recovered nothing: no JSON, and not one episode line. That was survivable on a laptop and
    is not on a rented device, where an interrupted hour is an hour paid for and this kit's
    whole purpose is to say what failed and where.

    So the contract is: whatever finished is on disk, whatever the run does next.
    """
    reports = []
    for index, episode in enumerate(episodes, 1):
        print(f"\n[{index}/{len(episodes)}] {episode.label} ...", flush=True)
        started = time.monotonic()
        report = tolerance.run_tolerance(episode, forward, top_k=top_k)
        elapsed = time.monotonic() - started
        print(f"{episode.label}  ({elapsed:.1f}s)")
        print(f"  {report.agreement.describe()}")
        print(f"  {report.jaccard.describe()}")
        print(f"  gate: {'PASS' if report.passed else 'FAIL'}", flush=True)
        reports.append((episode, report, elapsed))
        if per_episode is not None:
            with per_episode.open("a") as stream:
                stream.write(json.dumps(_episode_row(episode, report, elapsed)) + "\n")
                stream.flush()
    return reports


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


def _snapshot(repo_id: str) -> Path:
    """The cached snapshot for ``repo_id``, from the primary checkout, without the network.

    ``configure_local_cache`` points the cache at ``PROJECT_ROOT``, which inside a worktree is
    the worktree, whose ``.cache`` is empty. The weights are a shared artefact like the lock,
    the window and ``models/``: they exist once, in the primary checkout. This is the same
    failure ``_resolve_checkpoint`` documents for local checkpoints, arriving through a
    different reader, so it is fixed here the same way.

    ``primary_checkout_root`` is the git-derived reader, split out of ``box_state_root`` for
    exactly this: the box-state half may be redirected by ``$AGENT_V2_BOX_STATE_DIR`` so an
    isolated run cannot take the machine's lock, and weights must never follow that redirect.
    This ran against ``box_state_root`` with the redirect refused at the call until the split
    landed; that interim is deleted rather than left beside the real reader.
    """
    from huggingface_hub import snapshot_download

    from local_llm_lab.runlock import primary_checkout_root

    cache = primary_checkout_root() / ".cache" / "huggingface" / "hub"
    # Passed rather than exported: huggingface_hub reads its environment at import time, so
    # setting HF_HOME here is a no-op once anything has already imported it.
    print(f"weights cache: {cache}")
    return Path(snapshot_download(repo_id, local_files_only=True, cache_dir=str(cache)))


def _peak_gib(target: str) -> float | None:
    """Peak allocator memory on a CUDA device, or None where there is nothing to read.

    The checklist asks for the peak measured on the device rather than projected; nothing in
    this script measured one, and the laptop's 7.88 GiB came from outside it.
    """
    if not target.startswith("cuda"):
        return None
    import torch

    return torch.cuda.max_memory_allocated(target) / (1024**3)


def _load(checkpoint: Path) -> tuple[Any, Any, str]:
    """Text-only load through the package's loader, which fails closed on every gap.

    **The device is selected, never assumed.** This ran ``device="cpu"`` as a literal, written
    on a laptop that had no CUDA, and on the rented card it therefore loaded onto the CPU and
    produced a CPU number while every environment variable said ``cuda:0``. The load report
    said ``device: cpu`` and nothing else did; a reader taking the agreement figure at face
    value would have recorded a CPU measurement as the port's CUDA result. So the target comes
    from ``device.select``, which reads ``LLL_DEVICE``, and is returned so the record can name
    what actually ran rather than what the script was written on.
    """
    from local_llm_lab import device
    from local_llm_lab.arch_torch import TorchArchitectureView
    from local_llm_lab.hf_text import load_text_causal_lm

    reading = device.pin(attention="eager")
    target = device.select()
    pinned = reading.get("determinism")
    print(f"determinism: {pinned}, attention: eager, dtype bfloat16 on {target}")

    started = time.monotonic()
    model, report = load_text_causal_lm(
        checkpoint, dtype="bfloat16", attn_implementation="eager", device=target
    )
    if str(report.get("device")) != target:
        raise RuntimeError(
            f"asked for {target} and the loader reports {report.get('device')!r}; refusing to "
            "measure, because a number taken on another device would be recorded as this one's"
        )
    view = TorchArchitectureView.from_model(model)
    print(
        f"loaded in {time.monotonic() - started:.1f}s: {view.num_layers} layers, "
        f"hidden {view.hidden_size}, vocab {view.vocab_size}; load report {report}"
    )
    return model, view, target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--episode", action="append", help="restrict to these episode labels")
    parser.add_argument("--checkpoint", type=Path, help="overrides the registry entry")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", type=Path, help="write the report here")
    parser.add_argument(
        "--decoding",
        choices=("greedy", "sampled"),
        default="greedy",
        help="greedy only: this compares against MLX records made greedy, so 'sampled' is "
        "refused by name (plan §16.16); the sampled path is the state programme's",
    )
    arguments = parser.parse_args(argv)
    if arguments.decoding != "greedy":
        from local_llm_lab.pipeline.sampled_decode import require_greedy

        try:
            require_greedy(arguments.decoding, where="the tolerance runner")
        except ValueError as error:
            parser.error(str(error))

    episodes = golden.load_episodes(arguments.records)
    if arguments.episode:
        wanted = set(arguments.episode)
        episodes = [episode for episode in episodes if episode.label in wanted]
    if not episodes:
        parser.error("no episodes selected")

    _require_own_window()
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.torch_capture import TorchCapture

    checkpoint = arguments.checkpoint or _snapshot(load_model_spec(CHECKPOINT_ENTRY).hf_id)
    model, view, target = _load(checkpoint)

    class _Sink:
        """The capture needs a sink; teacher forcing reads logits and no residuals."""

        def residual(self, layer, offset, h) -> None:
            pass

        def output(self, offset, ids, logits) -> None:
            pass

    # Each episode's result is written the moment it finishes, and the stream is flushed.
    #
    # The first version of this accumulated everything and wrote once at the end, with stdout
    # block-buffered because it was not a terminal. An interrupt at ten minutes therefore
    # recovered nothing: no JSON, and not one episode line. That was survivable on a laptop
    # and is not on a rented device, where an interrupted hour is an hour paid for and the
    # kit's whole purpose is to say what failed and where.
    per_episode = arguments.json.with_suffix(".jsonl") if arguments.json else None
    with TorchCapture(view, _Sink(), layers=(view.num_layers,)) as wrapped:
        forward = tolerance.torch_forward_rows(wrapped, view, top_k=arguments.top_k)
        reports = run_episodes(episodes, forward, top_k=arguments.top_k, per_episode=per_episode)

    peak = _peak_gib(target)
    if peak is not None:
        print(f"peak {target} memory: {peak:.2f} GiB")

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
                    "head": provenance.head(
                        gate_records.current_identity(checkpoint),
                        records=str(arguments.records),
                        records_precision="mlx 4-bit",
                        port_precision=f"{target} bfloat16",
                    ),
                    "compared": provenance.Measured(compared, provenance.MEASURED_HERE).as_dict(),
                    "agreed": provenance.Measured(agreed, provenance.MEASURED_HERE).as_dict(),
                    "hard_flips": provenance.Measured(hard, provenance.MEASURED_HERE).as_dict(),
                    **(
                        {}
                        if peak is None
                        else {
                            "peak_memory": provenance.Measured(
                                round(peak, 3), provenance.MEASURED_HERE, unit="GiB"
                            ).as_dict()
                        }
                    ),
                    "episodes": [
                        _episode_row(episode, report, elapsed)
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
