"""Regenerate stage two's episodes from their recorded prompts and compare token for token.

Stage two's capture records are the migration's golden baseline: fifteen episodes produced by
MLX on a 4-bit checkpoint, hash-chained, with every prompt, every emitted token and every
forward's argmax written down. A torch port reproduces them or it does not, and the useful
output when it does not is *where* it first stopped agreeing, not that it disagreed.

Three checks live here and they assert different things, which is the whole point of keeping
them apart:

``record_consistency`` is a check whose answer is known in advance. Each forward's last argmax
must equal the next emitted token, because generation was greedy and the record wrote both.
It asserts bookkeeping -- that this module's join between forwards, emissions and positions
matches the writer's convention -- and it asserts nothing whatever about any backend. It runs
without a model and it cannot fail on a correct reader. That is exactly why it catches the
one error class no later check can: a position convention conflated across two coordinate
systems. Label it as bookkeeping wherever it is reported and never let it stand in for
validation it does not perform.

``reproduce`` is the golden test. It regenerates each turn greedily from the recorded
``prompt_ids`` and compares to the recorded emissions, reporting the first divergence with
both tokens and both layer-34 top-k lists at the deciding position.

``readout_agreement`` compares the generated argmax against the recorded argmax per forward.
The record stores only a SHA-256 of each logit tensor and never the tensor, so a full max-abs
logit error is not computable from the record alone; what is computable is argmax agreement and
exact digest identity, and this module reports those two and says which is which. A caller
holding both backends live can attribute the difference to quantisation; this module cannot,
and does not claim to.

Run against the records with the replay generator, which exercises the harness and not any
port::

    python research/acceptance/golden_trajectories.py --records <dir>

The replay generator reads its answers from the record under test, so it passes by
construction. It is here to prove the comparison machinery reports what it should, and
``--self-test`` runs the mutation controls that give that claim teeth.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from local_llm_lab.pipeline.live_lens.session import read_record  # noqa: E402

__all__ = [
    "Consistency",
    "Divergence",
    "Emission",
    "Episode",
    "EpisodeResult",
    "Forward",
    "GeneratedTurn",
    "ReadoutAgreement",
    "Turn",
    "TurnResult",
    "load_episode",
    "load_episodes",
    "main",
    "readout_agreement",
    "record_consistency",
    "recorded_generator",
    "reproduce",
]

#: The pilot's generation cap, recorded in its ``plan.json``. A turn that emits exactly this
#: many tokens hit the cap rather than stopping on the turn's own stop rule.
DEFAULT_MAX_TOKENS = 200


@dataclass(frozen=True)
class Forward:
    """One recorded native forward: what went in, and the argmax that came out."""

    offset: int
    input_ids: tuple[int, ...]
    argmax: tuple[int, ...]
    logits_sha256: str
    logits_shape: tuple[int, ...]
    final_readout_max_abs_error: float | None

    @property
    def prediction(self) -> int:
        """Argmax at the final input position, which is the token this forward predicts."""
        return self.argmax[-1]


@dataclass(frozen=True)
class Emission:
    """One generated token, at its absolute position in the turn's sequence."""

    position: int
    token_id: int
    span: str | None


@dataclass
class Turn:
    """One assistant turn: the prompt it was given and the tokens it produced."""

    index: int
    prompt_ids: tuple[int, ...]
    emissions: tuple[Emission, ...]
    forwards: dict[int, Forward]
    readings: dict[int, dict[str, list[int]]]
    confidence: dict[int, float]
    final_layer: int
    emitted_count: int
    forwarded_count: int
    status: str

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(emission.token_id for emission in self.emissions)

    @property
    def hit_token_cap(self) -> bool:
        """Whether the turn stopped because it ran out of budget rather than by its stop rule."""
        return len(self.emissions) >= DEFAULT_MAX_TOKENS

    def final_top(self, position: int) -> list[int]:
        """Recorded final-layer top-k at ``position``, which is the model's own distribution.

        The layer is read from the record's own declared list, never assumed. Gemma 3 4B has 34
        and the GPU is expected to bring larger models, so a literal here would be scaffolding
        that silently returns nothing on the first model with a different depth.
        """
        return list(self.readings.get(position, {}).get(str(self.final_layer), []))

    def emitted_confidence(self, position: int) -> float | None:
        """The model's own probability for the token emitted at ``position``.

        The final layer's readout is the model's own softmax, so at horizon 1 this is the
        probability of the greedy choice. It is the quantity the P >= 0.99 rule reads:
        quantisation cannot move an argmax that confident, so a flip there is a mask, position,
        entry or norm defect.
        """
        return self.confidence.get(position)


@dataclass
class Episode:
    """One captured episode, its turns in order, with the provenance the record carries."""

    label: str
    kind: str
    path: Path
    turns: tuple[Turn, ...]
    provenance: dict[str, Any]

    @property
    def emission_count(self) -> int:
        return sum(len(turn.emissions) for turn in self.turns)


@dataclass
class Consistency:
    """Result of the known-answer bookkeeping check. See this module's docstring."""

    label: str
    agreements: int
    disagreements: int
    missing_forwards: int
    detail: tuple[str, ...] = ()

    ASSERTS = (
        "decoding was greedy; the record's forward offsets and emitted positions share one "
        "convention; this reader joins them the way the writer wrote them"
    )

    @property
    def passed(self) -> bool:
        return self.disagreements == 0 and self.missing_forwards == 0


@dataclass(frozen=True)
class Divergence:
    """The first position at which regeneration stopped matching the record."""

    turn: int
    index: int
    position: int
    expected_token: int
    actual_token: int
    expected_final_top: tuple[int, ...]
    actual_final_top: tuple[int, ...]

    def describe(self) -> str:
        return (
            f"turn {self.turn} token {self.index} (position {self.position}): "
            f"recorded {self.expected_token}, generated {self.actual_token}\n"
            f"    recorded final-layer top: {list(self.expected_final_top)}\n"
            f"    generated final-layer top: {list(self.actual_final_top)}"
        )


@dataclass(frozen=True)
class GeneratedTurn:
    """What a generator returns for one turn.

    ``final_top`` is the generated side's final-layer top-k per emitted token, used only to
    describe a divergence. A generator that cannot supply it returns an empty mapping and the
    divergence report says so rather than inventing one.
    """

    token_ids: tuple[int, ...]
    final_top: dict[int, tuple[int, ...]] = field(default_factory=dict)
    argmax_by_offset: dict[int, int] = field(default_factory=dict)
    logits_sha256_by_offset: dict[int, str] = field(default_factory=dict)
    stop_reason: str = "unrecorded"


class TurnGenerator(Protocol):
    """The one thing this harness needs from a backend."""

    def __call__(self, prompt_ids: tuple[int, ...], *, max_tokens: int) -> GeneratedTurn: ...


@dataclass
class TurnResult:
    index: int
    expected: tuple[int, ...]
    actual: tuple[int, ...]
    divergence: Divergence | None
    stop_reason: str

    @property
    def reproduced(self) -> bool:
        return self.divergence is None and self.expected == self.actual


@dataclass
class EpisodeResult:
    label: str
    turns: tuple[TurnResult, ...]
    consistency: Consistency

    @property
    def reproduced(self) -> bool:
        return all(turn.reproduced for turn in self.turns)

    @property
    def first_divergence(self) -> Divergence | None:
        for turn in self.turns:
            if turn.divergence is not None:
                return turn.divergence
        return None

    @property
    def tokens_compared(self) -> int:
        return sum(len(turn.expected) for turn in self.turns)


@dataclass
class ReadoutAgreement:
    """Argmax agreement between a generator's forwards and the record's."""

    label: str
    compared: int
    agreed: int
    digests_compared: int
    digests_identical: int

    @property
    def rate(self) -> float:
        return self.agreed / self.compared if self.compared else 0.0

    @property
    def digest_note(self) -> str:
        """Bit-identity is only a meaningful column when the generator supplies digests."""
        if self.digests_compared == 0:
            return "generator supplied no logit digests; bit-identity not assessed"
        return f"{self.digests_identical} of {self.digests_compared} forwards bit-identical"


def _events(path: Path) -> list[dict[str, Any]]:
    """Read and hash-verify a capture record, reusing the writer's own reader."""
    return read_record(path)


def load_episode(path: Path) -> Episode:
    """Parse one capture record into turns, verifying its hash chain on the way in."""
    events = _events(path)
    manifest = events[0]
    provenance = dict(manifest.get("provenance", {}))
    episode_meta = dict(provenance.get("episode", {}))

    # The final layer is whatever the record declares, per turn. Deriving it is the difference
    # between a reader that works on the next model and one that quietly returns nothing.
    final_layers = {
        event["turn"]: max(event["layers"])
        for event in events
        if event.get("kind") == "begin_turn" and event.get("layers")
    }

    begins: dict[int, dict[str, Any]] = {}
    ends: dict[int, dict[str, Any]] = {}
    emissions: dict[int, list[Emission]] = {}
    forwards: dict[int, dict[int, Forward]] = {}
    readings: dict[int, dict[int, dict[str, list[int]]]] = {}
    confidence: dict[int, dict[int, float]] = {}

    for event in events:
        kind = event.get("kind")
        if kind == "begin_turn":
            begins[event["turn"]] = event
        elif kind == "end_turn":
            ends[event["turn"]] = event
        elif kind == "emitted":
            emissions.setdefault(event["turn"], []).append(
                Emission(
                    position=event["position"],
                    token_id=event["token_id"],
                    span=event.get("span"),
                )
            )
        elif kind == "forward":
            argmax = event["argmax"]
            flat = tuple(argmax[0]) if argmax and isinstance(argmax[0], list) else tuple(argmax)
            forwards.setdefault(event["turn"], {})[event["offset"]] = Forward(
                offset=event["offset"],
                input_ids=tuple(event["input_ids"]),
                argmax=flat,
                logits_sha256=event["logits_sha256"],
                logits_shape=tuple(event["logits_shape"]),
                final_readout_max_abs_error=event.get("final_readout_max_abs_error"),
            )
        elif kind == "reading":
            readings.setdefault(event["turn"], {})[event["position"]] = event["top"]
        elif (
            kind == "rank"
            and event.get("layer") == final_layers.get(event["turn"])
            and event.get("horizon") == 1
        ):
            # A rank row is keyed by the *reading* position, whose forward predicts the token
            # one position later. Storing it under the emission's own position keeps every
            # consumer on one convention; the join is checked by record_consistency.
            confidence.setdefault(event["turn"], {})[event["position"] + 1] = event["probability"]

    turns = tuple(
        Turn(
            index=index,
            prompt_ids=tuple(begins[index]["prompt_ids"]),
            emissions=tuple(sorted(emissions.get(index, []), key=lambda item: item.position)),
            forwards=forwards.get(index, {}),
            readings=readings.get(index, {}),
            confidence=confidence.get(index, {}),
            final_layer=final_layers.get(index, 0),
            emitted_count=ends.get(index, {}).get("emitted_count", len(emissions.get(index, []))),
            forwarded_count=ends.get(index, {}).get("forwarded_count", 0),
            status=ends.get(index, {}).get("status", "unrecorded"),
        )
        for index in sorted(begins)
    )
    return Episode(
        label=episode_meta.get("label", path.stem),
        kind=episode_meta.get("kind", "unknown"),
        path=path,
        turns=turns,
        provenance=provenance,
    )


def load_episodes(directory: Path) -> list[Episode]:
    """Every capture record in ``directory``, in filename order."""
    return [load_episode(path) for path in sorted(directory.glob("*.jsonl"))]


def record_consistency(episode: Episode) -> Consistency:
    """Assert the record's own forward-to-emission join. Bookkeeping, not validation.

    The forward at offset ``p`` sees positions up to and including ``p`` and its final argmax
    is therefore the greedy prediction for position ``p + 1``. Every emitted token must equal
    that prediction. A failure here means this reader disagrees with the writer about
    coordinates, which would make every other number in this module meaningless.
    """
    agreements = 0
    disagreements = 0
    missing = 0
    detail: list[str] = []
    for turn in episode.turns:
        for emission in turn.emissions:
            forward = turn.forwards.get(emission.position - 1)
            if forward is None:
                missing += 1
                if len(detail) < 8:
                    detail.append(
                        f"turn {turn.index}: no forward at offset {emission.position - 1}"
                    )
                continue
            if forward.prediction == emission.token_id:
                agreements += 1
            else:
                disagreements += 1
                if len(detail) < 8:
                    detail.append(
                        f"turn {turn.index} position {emission.position}: "
                        f"forward predicted {forward.prediction}, record emitted "
                        f"{emission.token_id}"
                    )
    return Consistency(
        label=episode.label,
        agreements=agreements,
        disagreements=disagreements,
        missing_forwards=missing,
        detail=tuple(detail),
    )


def recorded_generator(episode: Episode) -> TurnGenerator:
    """A generator that replays the record it is given.

    It passes by construction and validates the harness rather than any backend. Keeping it
    here, named for what it is, is cheaper than a caller inventing one that quietly becomes a
    green test of nothing.
    """
    by_prompt = {turn.prompt_ids: turn for turn in episode.turns}

    def generate(prompt_ids: tuple[int, ...], *, max_tokens: int) -> GeneratedTurn:
        turn = by_prompt[tuple(prompt_ids)]
        token_ids = turn.token_ids[:max_tokens]
        return GeneratedTurn(
            token_ids=token_ids,
            final_top={
                emission.position: tuple(turn.final_top(emission.position - 1))
                for emission in turn.emissions
            },
            argmax_by_offset={
                offset: forward.prediction for offset, forward in turn.forwards.items()
            },
            stop_reason="replayed",
        )

    return generate


def torch_generator(model: Any, view: Any, tokenizer: Any, spec: Any) -> TurnGenerator:
    """The real generator: the runner's torch greedy loop, driven from recorded prompt ids.

    UNEXECUTED. Nothing has run through this, because it needs WS-A's architecture view and no
    torch view exists yet. It is wired rather than sketched so that swapping the view in is the
    only remaining step, and so the call shape it assumes is inspectable now rather than
    discovered on rented hardware.

    It supplies no ``final_top`` and no logit digests. A divergence report will therefore say
    the generated side's layer-34 top is empty, which is true, rather than filling it with the
    recorded one and making a comparison look like it happened.
    """
    from local_llm_lab.pipeline.runner import generate_turn_tokens

    def generate(prompt_ids: tuple[int, ...], *, max_tokens: int) -> GeneratedTurn:
        token_ids, reason = generate_turn_tokens(
            model, tokenizer, list(prompt_ids), max_tokens, view=view, spec=spec
        )
        return GeneratedTurn(token_ids=tuple(token_ids), stop_reason=reason)

    return generate


def _divergence(
    turn: Turn, generated: GeneratedTurn, index: int, actual_token: int | None
) -> Divergence:
    expected = turn.token_ids
    position = turn.prompt_ids.__len__() + index
    return Divergence(
        turn=turn.index,
        index=index,
        position=position,
        expected_token=expected[index] if index < len(expected) else -1,
        actual_token=actual_token if actual_token is not None else -1,
        expected_final_top=tuple(turn.final_top(position - 1)),
        actual_final_top=tuple(generated.final_top.get(position, ())),
    )


def reproduce(
    episode: Episode, generate: TurnGenerator, *, max_tokens: int = DEFAULT_MAX_TOKENS
) -> EpisodeResult:
    """Regenerate every turn from its recorded prompt and report the first divergence."""
    results: list[TurnResult] = []
    for turn in episode.turns:
        generated = generate(turn.prompt_ids, max_tokens=max_tokens)
        expected = turn.token_ids
        actual = tuple(generated.token_ids)
        divergence: Divergence | None = None
        for index in range(max(len(expected), len(actual))):
            expected_token = expected[index] if index < len(expected) else None
            actual_token = actual[index] if index < len(actual) else None
            if expected_token != actual_token:
                divergence = _divergence(turn, generated, index, actual_token)
                break
        results.append(
            TurnResult(
                index=turn.index,
                expected=expected,
                actual=actual,
                divergence=divergence,
                stop_reason=generated.stop_reason,
            )
        )
    return EpisodeResult(
        label=episode.label,
        turns=tuple(results),
        consistency=record_consistency(episode),
    )


def readout_agreement(episode: Episode, generate: TurnGenerator) -> ReadoutAgreement:
    """Per-forward argmax agreement, plus how many logit digests matched exactly.

    Digest identity is a strict check that a different backend is expected to fail; it is
    reported as a count rather than a gate so that a reader can see whether a run was
    bit-identical or merely argmax-identical.
    """
    compared = 0
    agreed = 0
    digests_compared = 0
    digests_identical = 0
    for turn in episode.turns:
        generated = generate(turn.prompt_ids, max_tokens=DEFAULT_MAX_TOKENS)
        for offset, forward in sorted(turn.forwards.items()):
            if offset in generated.argmax_by_offset:
                compared += 1
                if generated.argmax_by_offset[offset] == forward.prediction:
                    agreed += 1
            digest = generated.logits_sha256_by_offset.get(offset)
            if digest is not None:
                digests_compared += 1
                if digest == forward.logits_sha256:
                    digests_identical += 1
    return ReadoutAgreement(
        label=episode.label,
        compared=compared,
        agreed=agreed,
        digests_compared=digests_compared,
        digests_identical=digests_identical,
    )


def _self_test(episodes: list[Episode]) -> int:
    """Mutation controls: prove the comparison reports a divergence where one was planted.

    A harness whose only exercise is a generator that replays the record is a test that cannot
    fail. These controls make it fail on purpose, at a known index, which is the smallest thing
    that turns the replay run into evidence about the comparison.
    """
    episode = next((item for item in episodes if len(item.turns[0].emissions) > 5), episodes[0])
    turn = episode.turns[0]
    honest = recorded_generator(episode)
    failures: list[str] = []

    baseline = reproduce(episode, honest)
    if not baseline.reproduced:
        failures.append("replay of the record did not reproduce the record")

    for planted in (0, 3, len(turn.token_ids) - 1):

        def mutated(prompt_ids, *, max_tokens, _at=planted):
            generated = honest(prompt_ids, max_tokens=max_tokens)
            if tuple(prompt_ids) != turn.prompt_ids:
                return generated
            tokens = list(generated.token_ids)
            tokens[_at] = tokens[_at] + 1
            return GeneratedTurn(
                token_ids=tuple(tokens),
                final_top=generated.final_top,
                argmax_by_offset=generated.argmax_by_offset,
                stop_reason="mutated",
            )

        result = reproduce(episode, mutated)
        divergence = result.first_divergence
        if result.reproduced:
            failures.append(f"a token flipped at index {planted} was not detected")
        elif divergence is None or divergence.index != planted:
            reported = "none" if divergence is None else str(divergence.index)
            failures.append(f"token flipped at index {planted}, divergence reported at {reported}")

    def truncated(prompt_ids, *, max_tokens):
        generated = honest(prompt_ids, max_tokens=max_tokens)
        if tuple(prompt_ids) != turn.prompt_ids:
            return generated
        return GeneratedTurn(
            token_ids=generated.token_ids[:-1],
            final_top=generated.final_top,
            argmax_by_offset=generated.argmax_by_offset,
            stop_reason="truncated",
        )

    short = reproduce(episode, truncated)
    if short.reproduced:
        failures.append("a turn one token short was not detected")
    elif short.first_divergence.index != len(turn.token_ids) - 1:
        failures.append("a short turn diverged at the wrong index")

    consistency = record_consistency(episode)
    if not consistency.passed:
        failures.append("the record's own forward-to-emission join does not hold")

    print(f"self-test on {episode.label}")
    for failure in failures:
        print(f"  FAIL {failure}")
    if not failures:
        print("  all mutation controls reported the planted divergence at its exact index")
    return 1 if failures else 0


def _report(
    results: list[EpisodeResult],
    agreements: list[ReadoutAgreement],
    *,
    generator: str,
    generator_note: str,
) -> None:
    print()
    print(f"Generator: {generator}")
    print(f"  {generator_note}")
    print()
    print("Bookkeeping: each forward's last argmax against the next emitted token")
    print(f"  asserts: {Consistency.ASSERTS}")
    print(f"  {'episode':40s} {'agree':>7s} {'disagree':>9s} {'missing':>8s}")
    for result in results:
        consistency = result.consistency
        print(
            f"  {consistency.label:40s} {consistency.agreements:7d} "
            f"{consistency.disagreements:9d} {consistency.missing_forwards:8d}"
        )
    total_agree = sum(item.consistency.agreements for item in results)
    total_disagree = sum(item.consistency.disagreements for item in results)
    print(f"  {'total':40s} {total_agree:7d} {total_disagree:9d}")

    print()
    print("Golden trajectories: regeneration against the record")
    print(f"  {'episode':40s} {'turns':>6s} {'tokens':>7s}  result")
    for result in results:
        status = "reproduced" if result.reproduced else "DIVERGED"
        print(f"  {result.label:40s} {len(result.turns):6d} {result.tokens_compared:7d}  {status}")
        if result.first_divergence is not None:
            print(f"    first divergence: {result.first_divergence.describe()}")

    if agreements:
        print()
        print("Final-layer readout: generated argmax against recorded argmax")
        print(f"  {'episode':40s} {'compared':>9s} {'agreed':>7s} {'rate':>7s}")
        for agreement in agreements:
            print(
                f"  {agreement.label:40s} {agreement.compared:9d} "
                f"{agreement.agreed:7d} {agreement.rate:7.4f}"
            )
        print(f"  bit-identity: {agreements[0].digest_note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--records",
        type=Path,
        required=True,
        help="directory of stage-two capture records (*.jsonl)",
    )
    parser.add_argument("--episode", action="append", help="restrict to these episode labels")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the mutation controls on the comparison itself and exit",
    )
    parser.add_argument(
        "--consistency-only",
        action="store_true",
        help="run only the bookkeeping check, which needs no generator",
    )
    parser.add_argument("--json", type=Path, help="write a machine-readable summary here")
    arguments = parser.parse_args(argv)

    episodes = load_episodes(arguments.records)
    if arguments.episode:
        wanted = set(arguments.episode)
        episodes = [episode for episode in episodes if episode.label in wanted]
    if not episodes:
        parser.error(f"no capture records found under {arguments.records}")

    if arguments.self_test:
        return _self_test(episodes)

    results: list[EpisodeResult] = []
    agreements: list[ReadoutAgreement] = []
    for episode in episodes:
        if arguments.consistency_only:
            results.append(
                EpisodeResult(
                    label=episode.label, turns=(), consistency=record_consistency(episode)
                )
            )
            continue
        generate = recorded_generator(episode)
        results.append(reproduce(episode, generate))
        agreements.append(readout_agreement(episode, generate))

    _report(
        results,
        agreements,
        generator="recorded_replay",
        generator_note=(
            "replays the record under test, so the trajectory and readout columns pass by "
            "construction. This run is evidence about the harness, not about any backend. "
            "Run --self-test for the mutation controls that give that claim teeth."
        ),
    )

    if arguments.json:
        arguments.json.write_text(
            json.dumps(
                {
                    "generator": "recorded_replay",
                    "generator_note": (
                        "replays the record under test; exercises the harness, not a backend"
                    ),
                    "episodes": [
                        {
                            "label": result.label,
                            "reproduced": result.reproduced,
                            "tokens_compared": result.tokens_compared,
                            "consistency_agreements": result.consistency.agreements,
                            "consistency_disagreements": result.consistency.disagreements,
                            "first_divergence": (
                                None
                                if result.first_divergence is None
                                else {
                                    "turn": result.first_divergence.turn,
                                    "index": result.first_divergence.index,
                                    "position": result.first_divergence.position,
                                    "expected_token": result.first_divergence.expected_token,
                                    "actual_token": result.first_divergence.actual_token,
                                }
                            ),
                        }
                        for result in results
                    ],
                },
                indent=1,
            )
            + "\n"
        )

    failed = [result for result in results if result.turns and not result.reproduced]
    broken = [result for result in results if not result.consistency.passed]
    return 1 if failed or broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
