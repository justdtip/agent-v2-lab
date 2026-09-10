"""Sampled decoding on torch: the ruling as code, beside the greedy loop and never inside it.

The Director lifted the temperature refusal on 2026-09-10 (plan §16.16; the WS-B order's
amendment of that date), scoped exactly as the Chief advised, and this module is that scope
with nothing added:

- **Only for runs declared as the state programme.** A caller declares one by handing a
  :class:`SampledDecoding` to the runner as its sampler. Every golden gate, the tolerance runner
  and every comparison against an MLX record stay greedy, because their records were made
  greedy, and they refuse the sampled mode by name through :func:`require_greedy`.
- **Full softmax at temperature 1.0, no top-p, no top-k.** The estimands are the model's own
  laws, and any tuning measures a different distribution. The fields exist so that a caller who
  wants them has to say so and be refused by name; an argument for another value is a change to
  what the device's runs measure and is a separate ruling, not a keyword.
- **Seeded through the device shim's pinned seed.** The draw comes from a ``torch.Generator``
  of its own, seeded from ``device.pin`` unless the caller states a seed, so the model's own
  forward never consumes it and the global seed stays what the pin said.
- **Provenance in every manifest.** :func:`decoding_manifest` gives mode, temperature,
  truncation, the sampler's name and backend, the seed and where it came from, and the note
  that sampled draws reproduce only within one backend and kernel set. A sampled record is
  compared only with a sampled record from the same backend and kernel set.

The loop itself is :func:`local_llm_lab.pipeline.runner.torch_sampled_stream`, a sibling of the
greedy loop that shares its prefill partition, its lookahead and its stop rule and differs in
one line. The greedy function is untouched, and a test holds it to that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

__all__ = [
    "GREEDY",
    "SAMPLED",
    "MODES",
    "RULED_TEMPERATURE",
    "TRUNCATION_NONE",
    "BACKEND",
    "REPRODUCIBILITY_NOTE",
    "SampledDecoding",
    "decoding_mode",
    "decoding_manifest",
    "require_greedy",
]

GREEDY = "greedy"
SAMPLED = "sampled"
MODES = (GREEDY, SAMPLED)

#: The one temperature the ruling allows. Not a default to be overridden: the constructor
#: refuses every other value by name.
RULED_TEMPERATURE = 1.0
TRUNCATION_NONE = "none"
BACKEND = "torch"

#: Names, so the manifest can say which function drew and which chose.
SAMPLED_STREAM = "torch_sampled_stream"
GREEDY_STREAM = "torch_greedy_stream"
SAMPLED_FUNCTION = "torch.multinomial over the full softmax"
GREEDY_FUNCTION = "argmax in float32"

REPRODUCIBILITY_NOTE = (
    "sampled draws reproduce only within one backend and kernel set: the same seed on another "
    "backend, or on the same backend with different kernels, is a different sequence. A sampled "
    "record is compared only with a sampled record from the same backend and kernel set, and "
    "never with a greedy record."
)

_RULING = "plan §16.16, the WS-B order's amendment of 2026-09-10"


@dataclass
class SampledDecoding:
    """A declaration that this run is the state programme's and draws rather than chooses.

    Construct it with no arguments to get what was ruled. Every field exists so that asking for
    something else is a refusal with a name rather than a silent difference in what the device
    measured.

    One instance is one random stream: the generator is built on the first draw, seeded once,
    and continues across turns for as long as the same instance is handed in. A fresh instance
    with the same seed reproduces the draws; that is the whole of the reproducibility claim, and
    it holds only within one backend and kernel set.
    """

    mode: ClassVar[str] = SAMPLED

    temperature: float = RULED_TEMPERATURE
    top_p: float | None = None
    top_k: int | None = None
    #: ``None`` reads the device shim's pinned seed when the seed is first needed.
    seed: int | None = None
    #: How many draws this instance has made, for the manifest of a run that was interrupted.
    draws: int = field(default=0, compare=False)
    _seed_source: str = field(default="", repr=False, compare=False)
    _generators: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if float(self.temperature) != RULED_TEMPERATURE:
            raise ValueError(
                f"sampled decoding is ruled at temperature {RULED_TEMPERATURE} and refuses "
                f"temperature {self.temperature}: another value changes what the device's runs "
                f"measure and is a separate ruling ({_RULING})"
            )
        self.temperature = float(self.temperature)
        if self.top_p is not None:
            raise ValueError(
                f"sampled decoding refuses top-p truncation (top_p={self.top_p}): the estimand "
                f"is the model's own distribution, and any truncation measures a different one "
                f"({_RULING})"
            )
        if self.top_k is not None:
            raise ValueError(
                f"sampled decoding refuses top-k truncation (top_k={self.top_k}): the estimand "
                f"is the model's own distribution, and any truncation measures a different one "
                f"({_RULING})"
            )
        if self.seed is not None:
            self.seed = int(self.seed)
            self._seed_source = "explicit"

    @property
    def truncation(self) -> str:
        return TRUNCATION_NONE

    def resolved_seed(self) -> int:
        """The seed the draws use, read from the device shim's pin the first time it is asked.

        Refuses when nothing is pinned rather than inventing one: an unpinned process has no
        seed to record, and a manifest with no seed is a draw nobody can repeat.
        """
        if self.seed is None:
            from local_llm_lab import device

            pinned = device.describe().get("pinned")
            if not pinned or "seed" not in pinned:
                raise RuntimeError(
                    "the sampled decoding's generator is seeded through the device shim's "
                    "pinned seed and nothing is pinned; call device.pin() before the first draw, "
                    "or state a seed"
                )
            self.seed = int(pinned["seed"])
            self._seed_source = "device.pin"
        return self.seed

    @property
    def seed_source(self) -> str:
        """``device.pin`` or ``explicit``; empty until the seed has been resolved."""
        return self._seed_source

    def generator(self, device: Any) -> Any:
        """The generator for tensors on ``device``, built and seeded once per device."""
        import torch

        key = str(device)
        generator = self._generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.resolved_seed())
            self._generators[key] = generator
        return generator

    def draw(self, logits: Any) -> int:
        """One token from the full softmax of one row of logits, at the ruled temperature."""
        import torch

        row = logits.float()
        probabilities = torch.softmax(row / self.temperature, dim=-1)
        token = torch.multinomial(probabilities, 1, generator=self.generator(row.device))
        self.draws += 1
        return int(token.item())


def decoding_mode(decoding: Any) -> str:
    """The mode a decoding argument names: ``None`` and ``"greedy"`` are greedy."""
    if decoding is None:
        return GREEDY
    if isinstance(decoding, SampledDecoding):
        return SAMPLED
    if isinstance(decoding, str) and decoding in MODES:
        return decoding
    raise ValueError(
        f"unknown decoding {decoding!r}: one of {MODES}, or a SampledDecoding for the sampled "
        f"mode, which carries its own seed"
    )


def require_greedy(decoding: Any, *, where: str) -> str:
    """Refuse the sampled mode by name, for everything that compares against an MLX record.

    Returns ``"greedy"`` so a caller can record what it ran under. The refusal names the
    caller, because "sampled is refused" at minute fifty says nothing about which of the
    kit's three entry points said it.
    """
    mode = decoding_mode(decoding)
    if mode == SAMPLED:
        raise ValueError(
            f"{where} stays greedy and refuses the sampled mode: the MLX records it compares "
            f"against were made greedy, and a record made in one mode is not compared with one "
            f"made in the other ({_RULING}). The sampled path is the state programme's."
        )
    return mode


def _kernel_set() -> dict[str, Any]:
    """What 'the same backend and kernel set' means here, read back rather than assumed."""
    from local_llm_lab import device

    reading = device.describe()
    kernels: dict[str, Any] = {"torch": reading.get("torch"), "device": reading.get("device")}
    if reading.get("device_names"):
        kernels["device_names"] = list(reading["device_names"])
    try:
        import torch

        kernels["cuda"] = torch.version.cuda
    except ImportError:  # pragma: no cover - torch is the backend this module serves
        kernels["cuda"] = None
    return kernels


def decoding_manifest(decoding: Any) -> dict[str, Any]:
    """The ``decoding`` block a run's manifest carries, typed, for either mode.

    A greedy block carries no temperature and no seed, because greedy has neither. A sampled
    block carries everything the ruling asked for, and resolving it reads the pinned seed, so a
    manifest written before the first draw still records the seed the draws will use.
    """
    mode = decoding_mode(decoding)
    if mode == GREEDY:
        return {
            "mode": GREEDY,
            "sampler": {"name": GREEDY_STREAM, "function": GREEDY_FUNCTION, "backend": BACKEND},
        }
    assert isinstance(decoding, SampledDecoding)
    return {
        "mode": SAMPLED,
        "temperature": decoding.temperature,
        "truncation": decoding.truncation,
        "sampler": {"name": SAMPLED_STREAM, "function": SAMPLED_FUNCTION, "backend": BACKEND},
        "seed": decoding.resolved_seed(),
        "seed_source": decoding.seed_source,
        "draws": decoding.draws,
        "kernel_set": _kernel_set(),
        "note": REPRODUCIBILITY_NOTE,
    }
