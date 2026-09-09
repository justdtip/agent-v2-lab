"""The seven acceptance gates as one script, for the first hour on rented hardware.

Plan §7 fixes the order and §12.1 fixes the purpose: no CUDA number enters a record until
these pass, and the first hour on a paid device runs this rather than an experiment. The kit
stops at the first failing gate and prints the number it saw beside the number it expected, so
a failure on the remote is attributable to the device change and to nothing else.

Two rules this file exists to enforce, both of which cost this programme something to learn:

**A gate that has not run is never a pass.** Every gate reports one of ``pass``, ``fail`` or
``unavailable``, and ``unavailable`` names the workstream that owns it. An unimplemented gate
that returned success would be the most expensive line in the repository, because the whole
point of the kit is that a green run authorises spending.

**A check whose answer is known in advance is labelled as bookkeeping.** Gate 5 carries one:
the record's own forward-to-emission join. It cannot fail on a correct reader, it validates no
backend whatsoever, and it is the only thing that catches a position convention conflated
across two coordinate systems. It is reported under its own heading and never counted toward
the gate's verdict.

Usage::

    python scripts/acceptance_gates.py --records <stage-two record dir>

Gates 1-4 belong to WS-A, gate 5 and 6 to WS-B, gate 7 to WS-D.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPOSITORY_ROOT / "src", _REPOSITORY_ROOT / "research" / "acceptance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gate_records  # noqa: E402
import golden_trajectories as golden  # noqa: E402
import tolerance  # noqa: E402

PASS = "pass"
RESUMED = "resumed"
FAIL = "fail"
UNAVAILABLE = "unavailable"


@dataclass
class GateResult:
    status: str
    saw: str = ""
    expected: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return self.status == FAIL


@dataclass
class Gate:
    number: int
    name: str
    owner: str
    run: Callable[[argparse.Namespace], GateResult]


def _not_this_workstream(number: int, owner: str) -> Callable[[argparse.Namespace], GateResult]:
    def run(arguments: argparse.Namespace) -> GateResult:
        return GateResult(
            status=UNAVAILABLE,
            saw="not implemented",
            expected=f"{owner} implements gate {number}",
            notes=[
                "reported unavailable rather than skipped: a gate that has not run is not a "
                "pass, and this kit's green run is what authorises spending on hardware"
            ],
        )

    return run


def _selected_episodes(arguments: argparse.Namespace):
    """The episodes a gate reads, and the shortest one alone under ``--smoke``.

    Smoke exists so an hour that dies at minute fifty has touched every gate once and can say
    which of them was never going to work. The reduced input is part of the resume key, so a
    later full pass cannot resume past a smoke record; see gate_records.
    """
    episodes = golden.load_episodes(arguments.records)
    if not arguments.smoke or not episodes:
        return episodes
    shortest = min(episodes, key=lambda episode: episode.emission_count)
    return [shortest]


def gate_5_golden_trajectories(arguments: argparse.Namespace) -> GateResult:
    """Every episode token-for-token from its recorded prompt ids; one under --smoke."""
    episodes = _selected_episodes(arguments)
    if not episodes:
        return GateResult(FAIL, saw="0 records", expected=f"records under {arguments.records}")

    notes: list[str] = []

    # Bookkeeping first, and labelled as such. This asserts that the reader and the writer
    # share one position convention; it asserts nothing about any backend.
    agreements = 0
    disagreements = 0
    missing = 0
    for episode in episodes:
        consistency = golden.record_consistency(episode)
        agreements += consistency.agreements
        disagreements += consistency.disagreements
        missing += consistency.missing_forwards
    notes.append(
        f"bookkeeping (not validation): forward-to-emission join {agreements} agree, "
        f"{disagreements} disagree, {missing} missing across {len(episodes)} episodes"
    )
    if disagreements or missing:
        return GateResult(
            FAIL,
            saw=f"{disagreements} disagreements, {missing} missing forwards",
            expected="0 and 0; the reader disagrees with the record's own coordinates",
            notes=notes,
        )

    backend = _load_backend(arguments)
    if backend is None:
        notes.append(
            "no backend loaded, so no reproduction was attempted; pass --model to run the "
            "torch loop against these records"
        )
        return GateResult(
            UNAVAILABLE,
            saw="records read and self-consistent; nothing regenerated",
            expected=(
                f"{len(episodes)} episode{'' if len(episodes) == 1 else 's'} "
                "token-for-token from a loaded backend"
            ),
            notes=notes,
        )

    model, view, tokenizer, spec = backend
    generate = golden.torch_generator(model, view, tokenizer, spec)
    diverged = []
    tokens = 0
    for episode in episodes:
        result = golden.reproduce(episode, generate)
        tokens += result.tokens_compared
        if not result.reproduced:
            diverged.append(result)
    if diverged:
        first = diverged[0].first_divergence
        return GateResult(
            FAIL,
            saw=f"{len(diverged)} of {len(episodes)} episodes diverged",
            expected=f"0 of {len(episodes)}",
            notes=[*notes, first.describe() if first else "divergence without a first index"],
        )
    return GateResult(
        PASS,
        saw=f"{len(episodes)} episodes, {tokens} tokens, all identical",
        expected=f"{len(episodes)} episodes identical",
        notes=notes,
    )


def gate_6_golden_lens_reads(arguments: argparse.Namespace) -> GateResult:
    """Rank rows at every layer for the emitted tokens, against the recorded ones."""
    episodes = _selected_episodes(arguments)
    emitted = sum(episode.emission_count for episode in episodes)
    agentic = sum(episode.emission_count for episode in episodes if episode.kind == "agentic")
    # Computable from the records alone, and it sizes the tolerance gate rather than
    # describing it: the hard rule only bites where the model was already confident.
    confident = 0
    scored = 0
    for episode in episodes:
        for turn in episode.turns:
            for emission in turn.emissions:
                probability = turn.emitted_confidence(emission.position)
                if probability is None:
                    continue
                scored += 1
                if probability >= tolerance.HARD_CONFIDENCE:
                    confident += 1
    share = confident / scored if scored else 0.0
    notes = [
        f"corpus: {emitted} emitted tokens across {len(episodes)} episodes, "
        f"{agentic} of them agentic",
        f"tolerance coverage: {confident} of {scored} emissions were recorded at "
        f"P >= {tolerance.HARD_CONFIDENCE} ({share:.4f}), so the hard flip rule governs that "
        "share of the decisions; the remainder can flip on precision alone and is reported "
        "rather than gated",
        "the recorded ranks come from MLX 4-bit through the hosted lens; a comparison against "
        "another precision confounds the port with quantisation, and the gate reports the "
        "difference rather than attributing it",
    ]
    if _load_backend(arguments) is None:
        return GateResult(
            UNAVAILABLE,
            saw="corpus read; no lens reads produced",
            expected=f"rank rows at every layer for {agentic} agentic emitted tokens",
            notes=[
                *notes,
                "needs a loaded backend and the hosted lens; the comparison side is built and "
                "the producing side is WS-A's view",
            ],
        )
    return GateResult(
        UNAVAILABLE,
        saw="backend loaded, but the lens read path is not ported",
        expected=f"rank rows at every layer for {agentic} agentic emitted tokens",
        notes=notes,
    )


def _load_backend(arguments: argparse.Namespace) -> tuple[Any, Any, Any, Any] | None:
    """Load a model only when asked, and only inside an announced box window.

    Loading weights without a window is the one thing box discipline forbids outright, so the
    refusal is here rather than in a comment.
    """
    if not arguments.model:
        return None
    from local_llm_lab import runlock

    if runlock.read_window() is None:
        raise SystemExit(
            "refusing to load a model: no box window is announced. Announce one and read "
            "`runlock status` back until it says running; announcing is not holding."
        )
    raise SystemExit(
        "--model is wired but the torch architecture view does not exist yet (WS-A). "
        "Remove --model to run the gates that read records only."
    )


GATES = [
    Gate(1, "structural discovery of the decoder", "WS-A", _not_this_workstream(1, "WS-A")),
    Gate(2, "residual_source_agreement at 64 and 1,400", "WS-A", _not_this_workstream(2, "WS-A")),
    Gate(
        3,
        "final-layer identity against the emitted token",
        "WS-A",
        _not_this_workstream(3, "WS-A"),
    ),
    Gate(4, "readout gate with both negative controls", "WS-A", _not_this_workstream(4, "WS-A")),
    Gate(5, "golden trajectories", "WS-B", gate_5_golden_trajectories),
    Gate(6, "golden lens reads", "WS-B", gate_6_golden_lens_reads),
    Gate(
        7, "lens un-port refitted and compared per layer", "WS-D", _not_this_workstream(7, "WS-D")
    ),
]


def _print_environment() -> None:
    print("Environment")
    try:
        import torch

        print(f"  torch {torch.__version__}")
        print(f"  cuda available: {torch.cuda.is_available()}")
        print(f"  mps available: {torch.backends.mps.is_available()}")
        print(f"  default dtype: {torch.get_default_dtype()}")
    except ImportError:
        print("  torch is not installed")
    # Pin before the first gate, then print what pin() read back rather than what it was asked
    # for. `CUBLAS_WORKSPACE_CONFIG` is read by cuBLAS at first use, so a pin after CUDA has
    # initialised is refused; that refusal reaching the operator is the point of doing it here.
    from local_llm_lab import device

    try:
        reading = device.pin(attention="eager")
    except RuntimeError as error:
        print(f"  determinism: REFUSED -- {error}")
        print("  every number below was taken unpinned and carries that.")
        print()
        return
    for key in sorted(reading):
        print(f"  {key}: {reading[key]}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--records", type=Path, required=True, help="directory of stage-two capture records"
    )
    parser.add_argument("--model", help="model to load; requires an announced box window")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run every gate on its smallest input, so an hour that dies late has touched "
        "each one once and can say which was never going to work",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="directory of gate records; a gate whose record matches this run's commit, "
        "checkpoint and device is not re-run",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="run every gate instead of stopping at the first failure",
    )
    arguments = parser.parse_args(argv)

    _print_environment()
    if arguments.smoke:
        selected = _selected_episodes(arguments)
        print(
            f"SMOKE PASS: every gate on its smallest input "
            f"({', '.join(episode.label for episode in selected)}).\n"
            "A pass here is not a pass on the full input, and the resume key says so.\n"
        )

    store = None
    if arguments.results:
        identity = gate_records.current_identity(
            arguments.model,
            smoke=arguments.smoke,
            episodes=sorted(episode.label for episode in _selected_episodes(arguments)),
        )
        store = gate_records.GateRecords(arguments.results, identity)
        usable, why = identity.usable
        print(f"resume store: {arguments.results}")
        note = "" if usable else f" -- {why}"
        print(f"  commit {identity.source_commit[:12]}, resumable: {usable}{note}")
        print()

    results: list[tuple[Gate, GateResult]] = []
    for gate in GATES:
        stored = store.completed(gate.number) if store is not None else None
        if stored is not None and stored.get("status") == PASS:
            result = GateResult(
                status=RESUMED,
                saw=stored.get("saw", ""),
                expected=stored.get("expected", ""),
                notes=[f"resumed from {store.directory}; not re-run"],
            )
        else:
            if store is not None and gate.number in store.refusals:
                reason = store.refusals[gate.number]
                print(f"         note:     gate {gate.number} not resumed: {reason}")
            result = gate.run(arguments)
            if store is not None:
                store.write(
                    gate.number,
                    {"status": result.status, "saw": result.saw, "expected": result.expected},
                )
        results.append((gate, result))
        marker = {PASS: "PASS", FAIL: "FAIL", UNAVAILABLE: "----", RESUMED: "SKIP"}[result.status]
        print(f"[{marker}] gate {gate.number}: {gate.name} ({gate.owner})")
        if result.saw:
            print(f"         saw:      {result.saw}")
        if result.expected:
            print(f"         expected: {result.expected}")
        for note in result.notes:
            print(f"         note:     {note}")
        print()
        if result.blocking and not arguments.keep_going:
            print(f"stopping at gate {gate.number}; later gates were not run")
            break

    passed = sum(1 for _, result in results if result.status in (PASS, RESUMED))
    resumed = [gate.number for gate, result in results if result.status == RESUMED]
    if resumed:
        print(f"gates resumed rather than re-run: {resumed}")
    print(f"{passed} of {len(GATES)} gates passed.")
    unavailable = [gate.number for gate, result in results if result.status == UNAVAILABLE]
    if unavailable:
        print(f"gates not yet runnable: {unavailable}. None of these is a pass.")
    return 0 if passed == len(GATES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
