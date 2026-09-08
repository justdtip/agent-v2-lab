"""Immutable model declarations and the registry used by the agent-v2 pipeline."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "ChatSpec",
    "LoraSpec",
    "ModelSpec",
    "ProbesSpec",
    "ResolvedSpec",
    "load_model_spec",
    "registered_models",
]

_THINKING_MODES = frozenset({"unsupported", "off", "inference", "trained"})
_CACHE_STRATEGIES = frozenset({"auto", "trim", "snapshot", "history", "none"})
# The installed generator's default prefill cadence (``mlx_lm.generate.generate_step``,
# ``prefill_step_size``). The history cache plans its forward partition with this number so that
# reuse matches the ordinary schedule; ``tests/test_history_cache.py`` pins it to the library's
# signature default so an upgrade cannot move one without the other being noticed.
NATIVE_PREFILL_STEP_SIZE = 2048
_CAPTURE_DTYPES = frozenset({"native", "float32"})
_DEFAULT_PROBE_FRACTIONS = (0.167, 0.333, 0.5, 0.667, 0.833, 1.0)
_REGISTRY_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"
#: A registry `hf_id` starting with this names a checkpoint **in this repository** rather than a
#: Hugging Face repo, and is resolved against the project root when the spec is loaded.
#:
#: An explicit prefix rather than "does this path happen to exist": a spec whose meaning depends on
#: the filesystem it is read on is the same class of defect as behaviour that depends on where a
#: process is standing, and every stage that resolved a bare `models/...` from its own working
#: directory would work from the repository root and fail from a worktree or a scratch directory.
_LOCAL_CHECKPOINT_PREFIX = "models/"


def _resolve_checkpoint(hf_id: str) -> str:
    """Absolute path for a local checkpoint; a Hugging Face repo id unchanged."""
    if not hf_id.startswith(_LOCAL_CHECKPOINT_PREFIX):
        return hf_id
    return str((Path(__file__).resolve().parents[2] / hf_id).resolve())


@dataclass(frozen=True)
class ChatSpec:
    thinking: Literal["unsupported", "off", "inference", "trained"]
    template_kwargs: dict[str, Any]
    end_of_turn: str
    extra_stop_tokens: tuple[str, ...]
    #: The marker the chat template emits to open an assistant turn under
    #: ``add_generation_prompt``. Declared rather than assumed, because `protocol` hard-coded
    #: ChatML's and asserted it, so every generation-side render raised on a model whose template
    #: opens a turn any other way. Gemma's is ``"<start_of_turn>model\n"``.
    #:
    #: **Required, and keyword-only.** It had a ChatML default for one revision, so that the
    #: dozen positional constructors in the tests kept working, and the Chief was right to
    #: refuse it: a ChatML default on the one field whose purpose is to stop a ChatML value
    #: being assumed is the same defect one level down. The parser's ``_required_string`` covers
    #: registry files and nothing else, and ``_default_spec`` constructs this class directly,
    #: which is precisely where ``<|im_end|>`` had been hiding.
    #:
    #: Keyword-only because removing a default from the middle of a signature would let every
    #: positional constructor silently mis-assign, and the symptom would be a turn ending that
    #: is some other field's value. Named, they fail to construct instead, which is a list of
    #: sites rather than a bug.
    generation_prefix: str = field(kw_only=True)
    #: The role a tool observation renders under. ``"tool"`` where the template has that role,
    #: which is every ChatML model here; ``"user"`` where it does not.
    #:
    #: Gemma 3's template has branches for user, assistant and system only, and enforces strict
    #: user/model alternation with an explicit `raise_exception`, so an observation at an even
    #: loop index stops the render. **Provisional, for reading only** (the Chief's ruling of
    #: 2026-09-08): re-roling keeps a base-model reading inside the model's own distribution, and
    #: the only thing at stake is prompt rendering. It reverts to provisional the day anyone
    #: builds a supervised target, because there the choice changes what is trained on, and the
    #: run's manifest has to say which role it used.
    observation_role: str = field(default="tool", kw_only=True)
    max_think_tokens: int = 512


@dataclass(frozen=True)
class LoraSpec:
    keys: Literal["auto", "attention+mlp", "all-linear"] | tuple[str, ...]
    rank: int
    scale: float
    dropout: float


@dataclass(frozen=True)
class ProbesSpec:
    """The registry's ``probes:`` block, under the names the specs use for it.

    SPEC-004 §2 and ruling R17 both write ``spec.probes.layer_fractions``, and the R18b
    assignment writes ``spec.probes.capture_dtype``. Those names live here rather than on
    ``ModelSpec`` so the stored fields (``probe_layer_fractions``, ``probe_capture_dtype``) and
    every caller reading them stay exactly as they are.
    """

    layer_fractions: tuple[float, ...]
    capture_dtype: Literal["native", "float32"]
    #: The declared residual band, as 1-based layer pairs (R41e). Empty when the registry
    #: declares none. `live_lens.instruments.read_band` is the *validating* reader, used where
    #: the installed block kinds are known; this is the declaration itself, for readers that
    #: only need to say which pairs a thing covers (issue 88).
    live_lens_pairs: tuple[tuple[int, int], ...]
    #: Recorded answers to equal-distance partner ties, in-band layer to its partner (issue 86).
    #: A ruling, not a rule: the code consults it and reports any tie it does not cover.
    partner_tie_breaks: dict[int, int]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    #: The **model** this checkpoint is, as opposed to `hf_id`, which is where its weights live.
    #: They differ whenever a checkpoint is converted: `models/gemma-3-4b-it-4bit` and
    #: `models/gemma-3-4b-it-bf16` are two precisions of one model, `google/gemma-3-4b-it`.
    #:
    #: The distinction is not bookkeeping. A Jacobian lens is fitted on a **model**, and the
    #: pivot's own ruling is that the pilot runs 4-bit while the hosted lens was fitted on bf16,
    #: with the precision mismatch disclosed rather than avoided. An identity built from `hf_id`
    #: refuses that pairing, which is a true statement about the files and a false one about the
    #: experiment. Defaults to `hf_id`, which is right for every checkpoint we did not convert.
    source: str
    family: str
    chat: ChatSpec
    lora: LoraSpec
    train: dict[str, Any]
    cache_strategy: Literal["auto", "trim", "snapshot", "history", "none"]
    probe_layer_fractions: tuple[float, ...]
    memory_budget_gib: float
    policies: dict[str, str]
    #: The declared residual band as 1-based pairs (R41e). Defaulted because every existing
    #: caller constructs a spec without it and a band is not required of a model.
    probe_live_lens_pairs: tuple[tuple[int, int], ...] = ()
    #: Recorded partner tie-breaks (issue 86). Defaulted for the same reason as the band: a
    #: model with no tie, or no ruling on one, declares none and the family reports the tie.
    probe_partner_tie_breaks: dict[int, int] = field(default_factory=dict)
    cache_equivalence_verified: dict[str, str] | None = None
    # R18: native block execution during capture is the default; the float32 block path is for
    # the J-lens tail and JVP, where the deviation is measured by the preflight and recorded.
    probe_capture_dtype: Literal["native", "float32"] = "native"

    @property
    def probes(self) -> ProbesSpec:
        """The ``probes:`` block under the specs' own names; a view, not stored state."""
        return ProbesSpec(
            layer_fractions=self.probe_layer_fractions,
            live_lens_pairs=self.probe_live_lens_pairs,
            partner_tie_breaks=dict(self.probe_partner_tie_breaks),
            capture_dtype=self.probe_capture_dtype,
        )

    def resolve(self, model: Any, tokenizer: Any) -> ResolvedSpec:
        """Combine this declaration with facts exposed by ``ArchitectureView``.

        Architecture traversal belongs exclusively to ``ArchitectureView``.  The import is
        deliberately lazy so registry-only operations never import model runtime code.
        """
        from local_llm_lab.arch import ArchitectureView

        view = ArchitectureView.from_model(model)
        layer_types = tuple(view.layer_kind(index) for index in range(view.num_layers))
        lora_keys = view.lora_targets(self.lora.keys)
        cache_strategy: Literal["trim", "snapshot", "history", "none"]
        if self.cache_strategy == "auto":
            if view.cache_trimmable:
                cache_strategy = "trim"
                cache_strategy_reason = "auto:trimmable"
            elif self.cache_equivalence_verified is not None:
                cache_strategy = "snapshot"
                cache_strategy_reason = "auto:equivalence_verified"
            else:
                cache_strategy = "none"
                cache_strategy_reason = "auto:equivalence_unverified"
        else:
            cache_strategy = self.cache_strategy
            cache_strategy_reason = f"explicit:{self.cache_strategy}"
        return ResolvedSpec(
            spec=self,
            num_layers=view.num_layers,
            hidden_size=view.hidden_size,
            vocab_size=view.vocab_size,
            tie_word_embeddings=view.tie_word_embeddings,
            layer_types=layer_types,
            lora_keys=lora_keys,
            trainable_parameters=view.lora_parameter_count(lora_keys, self.lora.rank),
            probe_layers=tuple(
                max(1, round(fraction * view.num_layers)) for fraction in self.probe_layer_fractions
            ),
            cache_strategy=cache_strategy,
            cache_strategy_reason=cache_strategy_reason,
            snapshot_revision=_snapshot_revision(model, tokenizer),
            jvp_method="untested",
        )


@dataclass(frozen=True)
class ResolvedSpec:
    spec: ModelSpec
    num_layers: int
    hidden_size: int
    vocab_size: int
    tie_word_embeddings: bool
    layer_types: tuple[str, ...]
    lora_keys: tuple[str, ...]
    trainable_parameters: int
    probe_layers: tuple[int, ...]
    cache_strategy: Literal["trim", "snapshot", "history", "none"]
    cache_strategy_reason: str
    snapshot_revision: str | None
    jvp_method: Literal["forward", "finite_difference", "untested"]

    def as_dict(self) -> dict[str, Any]:
        """Return stable JSON-ready metadata for manifests and result records."""
        return {
            "spec": asdict(self.spec),
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "tie_word_embeddings": self.tie_word_embeddings,
            "layer_types": list(self.layer_types),
            "lora_keys": list(self.lora_keys),
            "trainable_parameters": self.trainable_parameters,
            "probe_layers": list(self.probe_layers),
            "cache_strategy": self.cache_strategy,
            "cache_strategy_reason": self.cache_strategy_reason,
            "snapshot_revision": self.snapshot_revision,
            "jvp_method": self.jvp_method,
        }


def registered_models() -> list[str]:
    """Return registry names in deterministic lexicographic order."""
    return sorted(path.stem for path in _REGISTRY_DIR.glob("*.yaml"))


def load_model_spec(name_or_hf_id: str) -> ModelSpec:
    """Load a registered declaration, or safe non-thinking defaults for an unknown HF id."""
    registry_path = _REGISTRY_DIR / f"{name_or_hf_id}.yaml"
    if registry_path.is_file():
        return _load_registry_file(registry_path)
    for name in registered_models():
        spec = _load_registry_file(_REGISTRY_DIR / f"{name}.yaml")
        if spec.hf_id == name_or_hf_id:
            return spec
    return _default_spec(name_or_hf_id)


def _load_registry_file(path: Path) -> ModelSpec:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"model config {path} must contain a mapping")
    return _model_spec_from_mapping(raw, source=str(path))


def _model_spec_from_mapping(raw: dict[str, Any], *, source: str) -> ModelSpec:
    from local_llm_lab.arch import LORA_POLICIES

    chat = _mapping(raw, "chat", source)
    lora = _mapping(raw, "lora", source)
    train = _mapping(raw, "train", source)
    cache = _mapping(raw, "cache", source)
    probes = _mapping(raw, "probes", source)
    memory = _mapping(raw, "memory", source)
    policies = _mapping(raw, "policies", source)

    thinking = _required_string(chat, "thinking", source)
    if thinking not in _THINKING_MODES:
        raise ValueError(f"{source}: chat.thinking must be one of {sorted(_THINKING_MODES)}")
    cache_strategy = _required_string(cache, "strategy", source)
    if cache_strategy not in _CACHE_STRATEGIES:
        raise ValueError(f"{source}: cache.strategy must be one of {sorted(_CACHE_STRATEGIES)}")
    cache_equivalence_verified = cache.get("equivalence_verified")
    if cache_equivalence_verified is not None:
        valid_keys = isinstance(cache_equivalence_verified, dict) and set(
            cache_equivalence_verified
        ) == {"date", "sha256"}
        if not valid_keys or not all(
            isinstance(value, str) and value for value in cache_equivalence_verified.values()
        ):
            raise ValueError(
                f"{source}: cache.equivalence_verified must be null or a mapping with "
                "non-empty date and sha256 strings"
            )
        if re.fullmatch(r"[0-9a-fA-F]{64}", cache_equivalence_verified["sha256"]) is None:
            raise ValueError(
                f"{source}: cache.equivalence_verified.sha256 must be 64 hexadecimal characters"
            )
    keys = lora.get("keys")
    if isinstance(keys, list):
        if not all(isinstance(key, str) and key for key in keys):
            raise ValueError(f"{source}: lora.keys entries must be non-empty strings")
        lora_keys: Literal["auto", "attention+mlp", "all-linear"] | tuple[str, ...] = tuple(keys)
    elif isinstance(keys, str) and keys in LORA_POLICIES:
        lora_keys = keys
    else:
        raise ValueError(
            f"{source}: lora.keys must be one of {sorted(LORA_POLICIES)} or a list of names"
        )
    fractions_raw = probes.get("layer_fractions")
    if not isinstance(fractions_raw, list) or not fractions_raw:
        raise ValueError(f"{source}: probes.layer_fractions must be a non-empty list")
    if any(
        isinstance(fraction, bool) or not isinstance(fraction, (int, float))
        for fraction in fractions_raw
    ):
        raise ValueError(f"{source}: probes.layer_fractions must contain numeric values")
    fractions = tuple(float(fraction) for fraction in fractions_raw)
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError(f"{source}: probes.layer_fractions must be within (0, 1]")
    tie_breaks_raw = probes.get("partner_tie_breaks", {})
    if not isinstance(tie_breaks_raw, dict):
        raise ValueError(f"{source}: probes.partner_tie_breaks must be a mapping")
    if any(
        isinstance(key, bool) or not isinstance(key, int) or isinstance(value, bool)
        or not isinstance(value, int)
        for key, value in tie_breaks_raw.items()
    ):
        raise ValueError(f"{source}: probes.partner_tie_breaks must map layers to layers")
    tie_breaks = {int(key): int(value) for key, value in tie_breaks_raw.items()}
    capture_dtype = probes.get("capture_dtype", "native")
    if capture_dtype not in _CAPTURE_DTYPES:
        raise ValueError(f"{source}: probes.capture_dtype must be one of {sorted(_CAPTURE_DTYPES)}")
    template_kwargs = chat.get("template_kwargs")
    if not isinstance(template_kwargs, dict):
        raise ValueError(f"{source}: chat.template_kwargs must be a mapping")
    stops = chat.get("extra_stop_tokens")
    if not isinstance(stops, list) or not all(isinstance(token, str) for token in stops):
        raise ValueError(f"{source}: chat.extra_stop_tokens must be a list of strings")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in policies.items()):
        raise ValueError(f"{source}: policies must map strings to strings")

    return ModelSpec(
        name=_required_string(raw, "name", source),
        hf_id=_resolve_checkpoint(_required_string(raw, "hf_id", source)),
        source=str(raw.get("source") or _required_string(raw, "hf_id", source)),
        family=_required_string(raw, "family", source),
        chat=ChatSpec(
            thinking=thinking,
            template_kwargs=dict(template_kwargs),
            end_of_turn=_required_string(chat, "end_of_turn", source),
            generation_prefix=_required_string(chat, "generation_prefix", source),
            observation_role=str(chat.get("observation_role", "tool")),
            extra_stop_tokens=tuple(stops),
            max_think_tokens=_positive_int(
                chat.get("max_think_tokens", 512), "chat.max_think_tokens", source
            ),
        ),
        lora=LoraSpec(
            keys=lora_keys,
            rank=_positive_int(lora.get("rank"), "lora.rank", source),
            scale=float(lora.get("scale")),
            dropout=float(lora.get("dropout")),
        ),
        train=dict(train),
        cache_strategy=cache_strategy,
        cache_equivalence_verified=None
        if cache_equivalence_verified is None
        else dict(cache_equivalence_verified),
        probe_layer_fractions=fractions,
        probe_live_lens_pairs=tuple(
            (int(pair[0]), int(pair[1])) for pair in probes.get("live_lens_pairs", ())
        ),
        probe_partner_tie_breaks=tie_breaks,
        probe_capture_dtype=capture_dtype,
        memory_budget_gib=float(memory.get("budget_gib")),
        policies=dict(policies),
    )


def _default_spec(hf_id: str) -> ModelSpec:
    """Return the non-thinking, no-cache declaration used for unregistered models."""
    return ModelSpec(
        name=hf_id,
        hf_id=hf_id,
        source=hf_id,
        family="unknown",
        # ChatML's markers, for a model nobody declared. They are a guess and the only honest
        # thing to say about them is that they are Qwen's; a run that reaches this path and is
        # not a ChatML model will render turns with another vocabulary's tokens.
        # ChatML's markers, for a model nobody declared. They are a guess, and the only
        # honest thing to say about them is that they are Qwen's: a run that reaches this
        # path on any other family renders turns with another vocabulary's tokens. Named
        # rather than defaulted so this line is where a reader finds them.
        chat=ChatSpec(
            "unsupported", {}, "<|im_end|>", (), generation_prefix="<|im_start|>assistant\n"
        ),
        lora=LoraSpec("attention+mlp", 16, 32.0, 0.0),
        train={},
        cache_strategy="none",
        cache_equivalence_verified=None,
        probe_layer_fractions=_DEFAULT_PROBE_FRACTIONS,
        probe_capture_dtype="native",
        memory_budget_gib=22.0,
        policies={},
    )


def _mapping(raw: dict[str, Any], key: str, source: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source}: {key} must be a mapping")
    return value


def _required_string(raw: dict[str, Any], key: str, source: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: {key} must be a non-empty string")
    return value


def _positive_int(value: Any, key: str, source: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{source}: {key} must be a positive integer")
    return value


def _snapshot_revision(model: Any, tokenizer: Any) -> str | None:
    """Return available tokenizer or model revision metadata without inspecting decoder state."""
    for source in (tokenizer, model):
        for attribute in ("snapshot_revision", "revision", "_commit_hash"):
            value = getattr(source, attribute, None)
            if isinstance(value, str) and value:
                return value
        init_kwargs = getattr(source, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            for key in ("snapshot_revision", "revision", "_commit_hash"):
                value = init_kwargs.get(key)
                if isinstance(value, str) and value:
                    return value
    return None
