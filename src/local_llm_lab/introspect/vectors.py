"""Concept directions, and the generators that are deliberately NOT used for training.

Every concept vector in this literature comes from one recipe: the residual for a carrier sentence
about X, minus the mean over a fixed pool of unrelated words. So a model trained on vectors from
that recipe only ever has to learn to recognise the RECIPE, and holding out a concept does not test
otherwise, because a held-out concept is built the same way.

Training therefore uses G0 alone. The rest exist to be held out.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

#: The fifty words the G0 baseline is averaged over. Frozen here rather than imported from the
#: bench, so a later edit to the console cannot silently change what a trained model was trained on.
#: No concept in the training vocabulary may appear in this pool -- a concept whose own baseline
#: contains it has a direction pointing at nothing.
BASELINE_POOL: tuple[str, ...] = (
    "table", "window", "river", "pencil", "ladder", "carpet", "engine", "meadow", "bottle",
    "marble", "candle", "harbour", "tunnel", "basket", "curtain", "pillow", "saddle", "anchor",
    "lantern", "kettle", "ribbon", "gravel", "timber", "compass", "barrel", "mitten", "trolley",
    "cabinet", "furnace", "hammock", "parcel", "satchel", "thimble", "trumpet", "wardrobe",
    "zipper", "bracket", "canteen", "drawer", "funnel", "girder", "hinge", "jacket", "kerbstone",
    "lantern-post", "napkin", "ottoman", "plank", "quilt", "rafter",
)

CARRIER = "Tell me about {word}."
G0 = "tell_me_about_minus_fixed50_v1"

#: The alternative carriers G1 uses. Same arithmetic, different sentence frame.
G1_CARRIERS = ("{word}.", "The topic is {word}.", "Consider {word}.", "A short note on {word}.")


def _residual_at(model: Any, blocks: Any, ids: list[int], layer: int, device) -> torch.Tensor:
    """The residual entering block `layer`, at the last position, in fp32 on the host.

    Read at the block INPUT, which is where the training patch adds, so extraction and injection
    agree by construction rather than by coincidence.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook(_m, args):
        captured["h"] = args[0].detach()[0, -1].float().cpu()

    handle = blocks[layer].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device=device), use_cache=False)
    finally:
        handle.remove()
    return captured["h"]


def concept_vector(model: Any, tokenizer: Any, blocks: Any, word: str, layer: int, *, device,
                   pool: tuple[str, ...] = BASELINE_POOL, carrier: str = CARRIER,
                   render: Callable[[Any, str], list[int]] | None = None) -> torch.Tensor:
    """G0: the carrier's residual for this word, minus the mean over the pool. fp32, on the host."""
    if render is None:
        from .render import render_prompt
        render = lambda tk, text: render_prompt(tk, text, thinking=False)  # noqa: E731
    if word.lower() in {w.lower() for w in pool}:
        raise ValueError(f"{word!r} is in the baseline pool; its direction would point at nothing")
    target = _residual_at(model, blocks, render(tokenizer, carrier.format(word=word)),
                          layer, device)
    rest = torch.stack([_residual_at(model, blocks, render(tokenizer, carrier.format(word=w)),
                                     layer, device) for w in pool])
    return target - rest.mean(dim=0)


def common_mode(model: Any, tokenizer: Any, blocks: Any, layer: int, *, device,
                pool: tuple[str, ...] = BASELINE_POOL, carrier: str = CARRIER) -> torch.Tensor:
    """What every concept vector at this layer has in common: the carrier's own direction.

    Injected as a negative class at matched damage, this detects a model that has learned the
    generator's signature rather than any concept. Nobody has run this control.
    """
    from .render import render_prompt
    rows = torch.stack([_residual_at(model, blocks,
                                     render_prompt(tokenizer, carrier.format(word=w),
                                                   thinking=False), layer, device)
                        for w in pool])
    return rows.mean(dim=0)


def spectrum_matched(bank: torch.Tensor, *, generator: torch.Generator | None = None,
                     add_mean: bool = False) -> torch.Tensor:
    """A random direction with the concept bank's own covariance: same norm, same per-mode energy.

    The only null that can be matched on norm AND on damage at once, which is what makes the
    decisive comparison decisive rather than a choice between two mismatched controls. A plain
    Gaussian is off-manifold and roughly fifty times more destructive per unit norm than a concept
    is -- measured -- so a norm-matched Gaussian was never a like-for-like control.
    """
    centred = bank - bank.mean(dim=0, keepdim=True)
    u, s, v = torch.linalg.svd(centred.double(), full_matrices=False)
    coefficients = torch.randn(s.shape[0], generator=generator, dtype=torch.float64)
    drawn = (v.T @ (coefficients * s / (centred.shape[0] ** 0.5))).float()
    if add_mean:
        # The bank is a sample from N(mu, Sigma), not N(0, Sigma), and centring without adding mu
        # back makes the arms separable EXACTLY rather than merely well. `drawn` lies in
        # rowspace(bank - mu) by construction, so let w be the unit vector in span(bank)
        # orthogonal to that rowspace: <v_i, w> = ||P.mu|| for every row including held-out ones,
        # and <drawn, w> = 0 identically. AUC 1.000 by algebra, with a margin of about 1e4 after
        # the bf16 cast build_masks applies. Projecting on the bank mean measures the same thing
        # with slack: 0.988 to 1.000 on the real banks at layers 20/32/40/48, falling to 0.510 to
        # 0.551 with the mean restored (BANK-GEOMETRY-2026-09-13; the 0.79 once quoted here was a
        # synthetic estimate and was too kind).
        #
        # Measured 2026-09-14: the trained adapter says YES to a mean-matched null on 98 to 100
        # per cent of trials at matched damage, the same as to a real concept, and to the bank
        # mean alone on 60 to 100 per cent. It is reading this scalar and nothing else.
        drawn = drawn + bank.mean(dim=0)
    scale = bank.norm(dim=-1).mean() / drawn.norm().clamp(min=1e-12)
    return drawn * scale


def permuted_in_basis(vector: torch.Tensor, bank: torch.Tensor,
                      *, generator: torch.Generator | None = None) -> torch.Tensor:
    """A concept's own coefficients in the bank's principal basis, shuffled.

    Exactly the same spectrum as the vector it came from, and no concept. Cheaper than fitting a
    covariance and immune to a rank-deficient bank.
    """
    centred = bank - bank.mean(dim=0, keepdim=True)
    _u, _s, v = torch.linalg.svd(centred.double(), full_matrices=False)
    coefficients = (v @ vector.double())
    shuffled = coefficients[torch.randperm(coefficients.shape[0], generator=generator)]
    out = (v.T @ shuffled).float()
    return out * (vector.norm() / out.norm().clamp(min=1e-12))
