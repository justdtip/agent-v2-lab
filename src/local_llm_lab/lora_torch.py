"""Low-rank adapters by in-place module swap, so the model stays the class it was.

Not `peft`, and that is a decision rather than laziness. It is in the `cuda` extra only, so nothing
written against it could be tested on the laptop before it ran on a rented card. And a `PeftModel`
answers `hasattr` for both `model` and `lm_head` through its delegating `__getattr__`, so this
repository's `base_model_of` walks into it, finds a causal LM with no `.layers`, and raises the
multimodal-wrapper error -- a wrong diagnosis pointing at the loader.

Swapping `nn.Linear` for a wrapper in place keeps the object a `Gemma4ForCausalLM`, so everything
that reaches for `.model.layers` or the decoder layer's class name keeps working by construction.
"""

from __future__ import annotations

import math
from typing import Iterable, Iterator

import torch
from torch import nn

#: The projections the paper adapts. Deliberately not `lm_head`: the chunked loss applies that
#: weight itself, so an adapter there would be bypassed and never trained. Deliberately not the
#: per-layer-input path either -- it is outside the paper's scope and it sits behind the injection
#: site, so adapting it would change the geometry the experiment is about.
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """A frozen base projection plus a trainable low-rank correction.

    `lora_B` starts at zero, so the adapter is exactly the identity at initialisation: the first
    step's loss equals the base model's, which is what makes the replay term start where it should
    and what lets a test assert that applying adapters changed nothing yet.
    """

    def __init__(self, base: nn.Linear, *, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        # On the base layer's own device. `nn.Parameter(torch.empty(...))` lands on the CPU, and
        # the model is already resident on the card by the time adapters are applied -- the loader
        # moves it before returning. A CPU adapter fails on the first matmul, which is the good
        # case; a silently-migrated one would be worse.
        where = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=where))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=where))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        side = self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.T @ self.lora_B.T
        return out + (side * self.scaling).to(out.dtype)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def apply_lora(model: nn.Module, *, targets: Iterable[str] = DEFAULT_TARGETS,
               r: int = 32, alpha: int = 64, dropout: float = 0.0) -> int:
    """Swap every matching `nn.Linear` for an adapted one, in place. Returns how many were swapped.

    Refuses a target that is not an `nn.Linear` rather than skipping it: a fused expert weight
    under a familiar name would otherwise be silently left unadapted.
    """
    wanted = set(targets)
    swapped = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name not in wanted:
                continue
            if isinstance(child, LoRALinear):
                continue
            if not isinstance(child, nn.Linear):
                raise TypeError(f"{name}.{child_name} is {type(child).__name__}, not nn.Linear; "
                                "refusing to skip it silently")
            setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            swapped += 1
    return swapped


def lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for _name, module in lora_modules(model):
        yield module.lora_A
        yield module.lora_B


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the adapter. A full save_pretrained here writes 56 GiB of base weights per checkpoint."""
    out = {}
    for name, module in lora_modules(model):
        out[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        out[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return out


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> int:
    loaded = 0
    for name, module in lora_modules(model):
        a, b = state.get(f"{name}.lora_A"), state.get(f"{name}.lora_B")
        if a is None or b is None:
            raise KeyError(f"no adapter in the file for {name}")
        with torch.no_grad():
            module.lora_A.copy_(a.to(module.lora_A.dtype))
            module.lora_B.copy_(b.to(module.lora_B.dtype))
        loaded += 1
    return loaded


def merge_lora(model: nn.Module) -> int:
    """Fold every adapter into its base weight and put the plain Linear back."""
    merged = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            with torch.no_grad():
                delta = (child.lora_B @ child.lora_A) * child.scaling
                child.base.weight.add_(delta.to(child.base.weight.dtype))
            setattr(module, child_name, child.base)
            merged += 1
    return merged


def count_lora_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))
