"""The precision-matched arm: torch bfloat16 against MLX bfloat16, at the gating positions.

The device run left one question open and this script is the whole of it. Twenty-four positions
failed the hard rule: the MLX 4-bit recording was at least 0.99 confident and the torch bfloat16
port produced a different argmax. Two explanations survive that result, and it cannot separate
them:

* **the port** — a defect in the torch implementation that CPU and CUDA share; or
* **the premise** — the rule assumed quantisation cannot move a confident argmax, and the
  comparison is 4-bit against bfloat16, which is not a small perturbation. The recorded
  probability is the *4-bit model's* confidence in its own preference, not shared ground truth.

Removing quantisation from the comparison separates them. MLX loads the **bfloat16** weights and
reads the same positions. Then, at each one:

``mlx bf16 == torch bf16``   the two bfloat16 implementations agree and the 4-bit record is the
                             outlier. The premise was wrong and the records need re-basing.
``mlx bf16 == recorded``     MLX still says what it said at matched precision, so torch differs
                             from MLX with quantisation excluded. That is the port.
``all three differ``         unresolved, and named as such rather than assigned.

Teacher-forced, one forward per turn, reading the argmax at each recorded position with the
recorded prefix ahead of it -- the same shape as ``tolerance.teacher_forced_run``, which is
reused rather than reimplemented so the two arms cannot drift in how they index.

Only the turns that carry a gating flip are run. The full corpus is available behind ``--all``
and is not the first thing to reach for: the question lives at twenty-four positions.

Box discipline: this loads 7.3 GiB of weights and refuses without an announced window.

    python scripts/mlx_reference.py --records <dir> --flips <confident-flips.json> --json <out>
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

import golden_trajectories as golden  # noqa: E402
import provenance  # noqa: E402

#: The registry entry whose weights are MLX bfloat16 -- the same model as the records at a
#: different precision, which is the one thing this comparison may not get wrong.
REFERENCE_ENTRY = "gemma3-4b-bf16"

#: What each verdict means, carried in the record so a reader need not reconstruct the argument.
VERDICTS = {
    "quantisation": (
        "the two bfloat16 implementations agree and the 4-bit record is the outlier: the hard "
        "rule's premise was wrong at this position"
    ),
    "port": (
        "MLX at matched precision still produces the recorded token, so torch differs from MLX "
        "with quantisation excluded: this position is the port"
    ),
    "unresolved": "all three differ; this position assigns to neither explanation",
}


def _require_own_window() -> None:
    """Refuse unless *this* process holds the window, not merely that one is open."""
    from local_llm_lab import runlock

    blocking = runlock.blocking_window()
    if blocking is not None:
        raise SystemExit(runlock.refusal_for_window(blocking))
    if runlock.read_window() is None:
        raise SystemExit(
            "refusing to load 7.3 GiB of weights: no box window is announced. Announce one with "
            "`runlock run` and read `runlock status` back until it says running."
        )


def _load(checkpoint: Path) -> tuple[Any, Any]:
    """MLX bfloat16, through the same loader that produced the records at 4-bit."""
    from mlx_lm import load

    started = time.monotonic()
    model, tokenizer = load(str(checkpoint))
    print(f"loaded MLX in {time.monotonic() - started:.1f}s from {checkpoint}", flush=True)
    return model, tokenizer


def mlx_argmax_rows(model: Any) -> Any:
    """A ``forward`` for :func:`tolerance.teacher_forced_run`, backed by an MLX model.

    Returns one ``(argmax, ())`` row per input position. The top-k is empty on purpose: this
    arm compares argmax only, and an empty tuple is visibly nothing rather than a plausible
    ranking nobody computed.
    """
    import mlx.core as mx

    def forward(sequence):
        logits = model(mx.array(list(sequence))[None])
        argmax = mx.argmax(logits[0], axis=-1)
        mx.eval(argmax)
        return [(int(token), ()) for token in argmax.tolist()]

    return forward


def _verdict(recorded: int, torch_token: int, mlx_token: int) -> str:
    if mlx_token == torch_token:
        return "quantisation"
    if mlx_token == recorded:
        return "port"
    return "unresolved"


def compare(flips: list[dict], produced: dict[tuple[int, int], int]) -> list[dict]:
    """Join the recorded, torch and MLX tokens at each gating position."""
    rows = []
    for flip in flips:
        key = (flip["turn"], flip["position"])
        if key not in produced:
            rows.append({**flip, "mlx_bf16_token": None, "verdict": "not reached"})
            continue
        mlx_token = produced[key]
        rows.append(
            {
                **flip,
                "mlx_bf16_token": mlx_token,
                "verdict": _verdict(flip["recorded_token"], flip["produced_token"], mlx_token),
            }
        )
    return rows


def run(episodes, flips_by_episode, forward, *, per_position: Path | None) -> list[dict]:
    """Read each episode's flagged turns, writing every position the moment it lands."""
    import tolerance

    rows: list[dict] = []
    for index, episode in enumerate(episodes, 1):
        flips = flips_by_episode.get(episode.label, [])
        if not flips:
            continue
        wanted = {flip["turn"] for flip in flips}
        subset = [turn for turn in episode.turns if turn.index in wanted]
        print(
            f"\n[{index}] {episode.label}: {len(flips)} positions in {len(subset)} turn(s) ...",
            flush=True,
        )
        started = time.monotonic()
        produced, _ = tolerance.teacher_forced_run(_EpisodeView(episode, subset), forward, top_k=0)
        episode_rows = compare(flips, produced)
        for row in episode_rows:
            print(
                f"  turn {row['turn']} position {row['position']}: recorded "
                f"{row['recorded_token']} (P={row['recorded_probability']:.6f}), torch "
                f"{row['produced_token']}, mlx {row['mlx_bf16_token']} -> {row['verdict']}",
                flush=True,
            )
        print(f"  ({time.monotonic() - started:.1f}s)", flush=True)
        rows.extend(episode_rows)
        if per_position is not None:
            with per_position.open("a") as stream:
                for row in episode_rows:
                    stream.write(json.dumps(row) + "\n")
                stream.flush()
    return rows


class _EpisodeView:
    """An episode restricted to the turns that carry a gating flip.

    ``teacher_forced_run`` walks ``episode.turns``; handing it a narrowed view runs only the
    turns in question without touching its indexing, which is the part that must not drift
    between the two arms.
    """

    def __init__(self, episode: Any, turns: list[Any]) -> None:
        self._episode = episode
        self.turns = turns
        self.label = episode.label

    def __getattr__(self, name: str) -> Any:
        return getattr(self._episode, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--flips",
        type=Path,
        required=True,
        help="confident-flips.json from the device run: the positions to resolve",
    )
    parser.add_argument("--checkpoint", type=Path, help="overrides the registry entry")
    parser.add_argument("--json", type=Path, help="write the report here")
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every turn of every episode rather than only the flagged turns",
    )
    arguments = parser.parse_args(argv)

    flips = json.loads(arguments.flips.read_text())["positions"]
    by_episode: dict[str, list[dict]] = {}
    for flip in flips:
        by_episode.setdefault(flip["episode"], []).append(flip)
    print(f"{len(flips)} gating positions across {len(by_episode)} episodes")

    episodes = golden.load_episodes(arguments.records)
    if not arguments.all:
        episodes = [episode for episode in episodes if episode.label in by_episode]
    if not episodes:
        parser.error("no episodes selected")

    # Declared before the load, with what it rests on, so the reading can be compared with it.
    projection = provenance.Measured(
        9.0,
        provenance.EXPECTED,
        basis="7.3 GiB MLX bfloat16 weights on disk, plus ~1.4 GiB of logits for the longest "
        "turn (2,607 positions x 262,208 vocab at bf16), plus overhead",
        unit="GiB",
    )
    print(f"projected peak: {projection.as_dict()}")

    _require_own_window()
    from local_llm_lab.models import load_model_spec

    checkpoint = arguments.checkpoint or Path(load_model_spec(REFERENCE_ENTRY).hf_id)
    model, _ = _load(checkpoint)

    per_position = arguments.json.with_suffix(".jsonl") if arguments.json else None
    started = time.monotonic()
    rows = run(episodes, by_episode, mlx_argmax_rows(model), per_position=per_position)
    elapsed = time.monotonic() - started

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print("\n" + "=" * 78)
    for verdict in ("quantisation", "port", "unresolved", "not reached"):
        if counts.get(verdict):
            print(f"{verdict:14} {counts[verdict]:3} of {len(rows)}")
    print(
        "\nreading: "
        + (
            "the premise was 4-bit's; the records re-base on MLX bfloat16"
            if counts.get("quantisation") == len(rows)
            else "at least one position is not explained by quantisation; see the rows"
        )
    )

    if arguments.json:
        arguments.json.write_text(
            json.dumps(
                {
                    "what": "torch bfloat16 against MLX bfloat16 at the positions that failed "
                    "the hard rule, to separate a port defect from the 4-bit/bfloat16 gap",
                    "verdicts": VERDICTS,
                    "reference": str(checkpoint),
                    "projected_peak": projection.as_dict(),
                    "seconds": provenance.Measured(
                        round(elapsed, 1), provenance.MEASURED_HERE, unit="s"
                    ).as_dict(),
                    "counts": {
                        k: provenance.Measured(v, provenance.MEASURED_HERE).as_dict()
                        for k, v in counts.items()
                    },
                    "positions": rows,
                },
                indent=1,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
