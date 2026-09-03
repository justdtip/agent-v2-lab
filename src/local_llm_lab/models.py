"""Immutable model declarations and the registry used by the agent-v2 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "ChatSpec",
    "LoraSpec",
    "ModelSpec",
    "ResolvedSpec",
    "load_model_spec",
    "registered_models",
]

_THINKING_MODES = frozenset({"unsupported", "off", "inference", "trained"})
_CACHE_STRATEGIES = frozenset({"auto", "trim", "snapshot", "none"})
_LORA_POLICIES = frozenset({"auto", "attention+mlp", "all-linear"})
_DEFAULT_PROBE_FRACTIONS = (0.167, 0.333, 0.5, 0.667, 0.833, 1.0)
_REGISTRY_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"


@dataclass(frozen=True)
class ChatSpec:
    thinking: Literal["unsupported", "off", "inference", "trained"]
    template_kwargs: dict[str, Any]
    end_of_turn: str
    extra_stop_tokens: tuple[str, ...]
    max_think_tokens: int = 512


@dataclass(frozen=True)
class LoraSpec:
    keys: Literal["auto", "attention+mlp", "all-linear"] | tuple[str, ...]
    rank: int
    scale: float
    dropout: float


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    family: str
    chat: ChatSpec
    lora: LoraSpec
    train: dict[str, Any]
    cache_strategy: Literal["auto", "trim", "snapshot", "none"]
    probe_layer_fractions: tuple[float, ...]
    memory_budget_gib: float
    policies: dict[str, str]

    def resolve(self, model: Any, tokenizer: Any) -> ResolvedSpec:
        """Combine this declaration with facts exposed by ``ArchitectureView``.

        Architecture traversal belongs exclusively to ``ArchitectureView``.  The import is
        deliberately lazy so registry-only operations never import model runtime code.
        """
        from local_llm_lab.arch import ArchitectureView

        view = ArchitectureView.from_model(model)
        layer_types = tuple(view.layer_kind(index) for index in range(view.num_layers))
        lora_keys = view.lora_targets(self.lora.keys)
        cache_strategy: Literal["trim", "snapshot", "none"]
        if self.cache_strategy == "auto":
            cache_strategy = "trim" if view.cache_trimmable else "none"
        else:
            cache_strategy = self.cache_strategy
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
                max(1, round(fraction * view.num_layers))
                for fraction in self.probe_layer_fractions
            ),
            cache_strategy=cache_strategy,
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
    cache_strategy: Literal["trim", "snapshot", "none"]
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
    keys = lora.get("keys")
    if isinstance(keys, list):
        if not all(isinstance(key, str) and key for key in keys):
            raise ValueError(f"{source}: lora.keys entries must be non-empty strings")
        lora_keys: Literal["auto", "attention+mlp", "all-linear"] | tuple[str, ...] = tuple(keys)
    elif isinstance(keys, str) and keys in _LORA_POLICIES:
        lora_keys = keys
    else:
        raise ValueError(
            f"{source}: lora.keys must be one of {sorted(_LORA_POLICIES)} or a list of names"
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
        hf_id=_required_string(raw, "hf_id", source),
        family=_required_string(raw, "family", source),
        chat=ChatSpec(
            thinking=thinking,
            template_kwargs=dict(template_kwargs),
            end_of_turn=_required_string(chat, "end_of_turn", source),
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
        probe_layer_fractions=fractions,
        memory_budget_gib=float(memory.get("budget_gib")),
        policies=dict(policies),
    )


def _default_spec(hf_id: str) -> ModelSpec:
    """Return the non-thinking, no-cache declaration used for unregistered models."""
    return ModelSpec(
        name=hf_id,
        hf_id=hf_id,
        family="unknown",
        chat=ChatSpec("unsupported", {}, "<|im_end|>", ()),
        lora=LoraSpec("attention+mlp", 16, 32.0, 0.0),
        train={},
        cache_strategy="none",
        probe_layer_fractions=_DEFAULT_PROBE_FRACTIONS,
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
