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


def coverage(model: nn.Module, *, targets: Iterable[str] = DEFAULT_TARGETS,
             blocks: Iterable[nn.Module] | None = None) -> dict:
    """What a name-based target list would reach, and what it would leave behind.

    Names like `q_proj` are a Llama-lineage convention, not a fact about transformers, and matching
    on them fails two ways. A name that is absent is skipped in silence. A name that is present on
    something other than an `nn.Linear` -- a fused expert weight, say -- would be skipped just as
    quietly if `apply_lora` did not refuse it.

    Measured on Gemma 4 31B: 410 of an expected 420, because the ten `full_attention` layers carry
    no `v_proj` at all while the fifty sliding ones do. The outcome was right -- there is genuinely
    nothing there to adapt -- but nothing said so, and on a model where the missing name sat on a
    differently-spelled module the same silence would be a real partial adaptation.
    """
    wanted = set(targets)
    from .residual_patch import decoder_blocks
    blocks = list(blocks) if blocks is not None else list(decoder_blocks(model))
    per_block, linears_missed = [], []
    for index, block in enumerate(blocks):
        hit, present = set(), set()
        for name, module in block.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if isinstance(module, (nn.Linear, LoRALinear)):
                present.add(leaf)
                if leaf in wanted:
                    hit.add(leaf)
                elif isinstance(module, nn.Linear):
                    linears_missed.append(f"block {index}: {name}")
        per_block.append({"block": index, "hit": sorted(hit),
                          "absent": sorted(wanted - present)})
    absent = {}
    for entry in per_block:
        for name in entry["absent"]:
            absent.setdefault(name, []).append(entry["block"])
    return {"blocks": len(blocks), "expected": len(blocks) * len(wanted),
            "reached": sum(len(e["hit"]) for e in per_block),
            "absent_by_name": absent,
            "linears_not_targeted": sorted(set(linears_missed))}


def apply_lora(model: nn.Module, *, targets: Iterable[str] = DEFAULT_TARGETS,
               r: int = 32, alpha: int = 64, dropout: float = 0.0) -> int:
    """Swap every matching `nn.Linear` for an adapted one, in place. Returns how many were swapped.

    Refuses a target that is not an `nn.Linear` rather than skipping it: a fused expert weight
    under a familiar name would otherwise be silently left unadapted. What it CANNOT refuse is a
    name that is simply absent -- `coverage()` exists to report that, and the caller should print
    it rather than trusting the count.
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


#: Key for the rank and alpha, carried in the file so a reload cannot silently use the wrong ones.
META = "_lora_meta"


def lora_state_dict(model: nn.Module) -> dict:
    """Just the adapter. A full save_pretrained here writes 56 GiB of base weights per checkpoint.

    Rank and alpha travel with it. A rank mismatch on reload raises when the copy_ finds the wrong
    shape, but an ALPHA mismatch cannot: it loads every module cleanly and applies a uniformly
    rescaled adapter, so the file would produce a different model with nothing said.
    """
    out: dict = {}
    ranks, alphas = set(), set()
    for name, module in lora_modules(model):
        out[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        out[f"{name}.lora_B"] = module.lora_B.detach().cpu()
        ranks.add(int(module.lora_A.shape[0]))
        alphas.add(float(module.scaling) * int(module.lora_A.shape[0]))
    if out:
        out[META] = {"rank": sorted(ranks), "alpha": sorted(alphas), "modules": len(ranks and out)}
    return out


def adapter_meta(state: dict) -> dict | None:
    """The rank and alpha a file was saved with, or None for a file saved before they travelled."""
    meta = state.get(META)
    return meta if isinstance(meta, dict) else None


def load_lora_state_dict(model: nn.Module, state: dict) -> int:
    meta = adapter_meta(state)
    loaded = 0
    for name, module in lora_modules(model):
        a, b = state.get(f"{name}.lora_A"), state.get(f"{name}.lora_B")
        if a is None or b is None:
            raise KeyError(f"no adapter in the file for {name}")
        with torch.no_grad():
            module.lora_A.copy_(a.to(module.lora_A.dtype))
            module.lora_B.copy_(b.to(module.lora_B.dtype))
        loaded += 1
    if meta:
        want = [float(module.scaling) * int(module.lora_A.shape[0])
                for _n, module in lora_modules(model)]
        if want and sorted(set(want)) != sorted(meta.get("alpha", sorted(set(want)))):
            raise ValueError(
                f"adapter was saved with alpha {meta['alpha']} and this model was built with "
                f"{sorted(set(want))}: the same weights would be applied at a different strength")
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
