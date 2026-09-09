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

#: The registry entry torch can load. **Not** ``gemma3-4b-bf16``: that one's weights are the
#: MLX conversion, whose safetensors header says ``format: mlx`` outright, and handing it to
#: ``from_pretrained`` would refuse or load something else. This entry is the upstream HF
#: checkpoint, already in the local cache, and it is the same bf16 tensors the MLX conversion
#: was made from, which is why the golden records correspond to it at all.
CHECKPOINT_ENTRY = "gemma3-4b-cuda-bf16"

#: The snapshot is the multimodal checkpoint, so the text weights carry a ``language_model.``
#: prefix and the vision tower rides along. Both from WS-A's ``cpu_gates.py``; lifting that
#: loader into the package is WS-E's and this mirrors it rather than diverging from it.
KEY_MAPPING = {r"^language_model\.": ""}
VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.")


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

    ``box_state_root`` is that reader and it carries a known flaw: ``$AGENT_V2_BOX_STATE_DIR``
    redirects it so an isolated run cannot take the machine's lock, and weights must not follow
    that redirect. Plan §16.8 has SWE-2 splitting the git-derived half out as
    ``primary_checkout_root``; when it lands this call moves to it. The redirect is unset here
    and the run records which path it read, so the caveat is visible rather than assumed away.
    """
    import os

    from huggingface_hub import snapshot_download

    from local_llm_lab.runlock import box_state_root

    root = box_state_root()
    if os.environ.get("AGENT_V2_BOX_STATE_DIR"):
        raise SystemExit(
            "AGENT_V2_BOX_STATE_DIR is set, so the shared-state reader is redirected and the "
            "weights would be looked for in scratch. Unset it, or wait for "
            "primary_checkout_root (plan section 16.8)."
        )
    cache = root / ".cache" / "huggingface" / "hub"
    # Passed rather than exported: huggingface_hub reads its environment at import time, so
    # setting HF_HOME here is a no-op once anything has already imported it.
    print(f"weights cache: {cache}")
    return Path(snapshot_download(repo_id, local_files_only=True, cache_dir=str(cache)))


def _load(checkpoint: Path):
    """Text-only load that fails closed, mirroring WS-A's ``cpu_gates.load_text_model``.

    The vision tower is expected to be left behind and every *other* gap is an error: a
    missing text key, a shape change or a conversion failure would otherwise load a model that
    runs and is not the one the records were made from.
    """
    import json

    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    from local_llm_lab import device
    from local_llm_lab.arch_torch import TorchArchitectureView

    reading = device.pin(attention="eager")
    print(f"determinism: {reading.get('determinism')}, attention: eager, dtype bfloat16 on cpu")

    raw = json.loads((checkpoint / "config.json").read_text())
    config = Gemma3TextConfig(**raw["text_config"])
    started = time.monotonic()
    model, info = Gemma3ForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
        key_mapping=KEY_MAPPING,
    )
    for field in ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors"):
        if info.get(field):
            raise SystemExit(f"text checkpoint load failed: {field}={info[field]}")
    unexpected = set(info.get("unexpected_keys", ()))
    if any(not key.startswith(VISION_PREFIXES) for key in unexpected):
        raise SystemExit(
            "the load left behind keys that are not the vision tower: "
            f"{sorted(key for key in unexpected if not key.startswith(VISION_PREFIXES))[:5]}"
        )
    model = model.eval()
    model.requires_grad_(False)
    view = TorchArchitectureView.from_model(model)
    print(
        f"loaded in {time.monotonic() - started:.1f}s: {view.num_layers} layers, "
        f"hidden {view.hidden_size}, vocab {view.vocab_size}; "
        f"{len(unexpected)} vision keys left behind"
    )
    return model, view


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

    checkpoint = arguments.checkpoint or _snapshot(load_model_spec(CHECKPOINT_ENTRY).hf_id)
    model, view = _load(checkpoint)

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
