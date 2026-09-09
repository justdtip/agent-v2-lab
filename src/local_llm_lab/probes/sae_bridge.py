"""The SAE-to-J-lens bridge, stage A: the exact score decomposition.

For a residual layer with lens map ``J``, the model's readout ``W`` (vocabulary by hidden), the
final norm's gain ``g``, and a sparse decoder ``D`` (features by hidden, one direction per row), the
lens score map is the **linear** map

    L x = W ((J x) * g)

and for every activation ``h = b + D^T z + e`` (bias, reconstruction, residual) the scores split
exactly:

    L h = L b + sum_i z_i (L d_i) + L e

That is an identity, not a model. Feature ``i``'s signed contribution to the score on token ``v``
is ``z_i (L d_i)_v``, and the whole of the approximation sits in ``L e``. Everything here either
computes a term of that identity or checks it.

**Why the gain and not the full norm.** Gemma's final RMSNorm is ``x * rsqrt(mean(x^2) + eps) *
(1 + w)``. The ``rsqrt`` factor is a positive scalar *per input vector*, so it changes no ranking
within one vector, and it makes the map nonlinear, so the decomposition above would stop being an
identity. The score map therefore applies only the gain ``g = 1 + w``, and the per-vector scalar is
declared dropped. Ranks and overlaps are unaffected; anyone comparing score *values* against a
model's logits must reintroduce it and is told so by the provenance block.

**Orientation, read from the shipped shapes and not assumed.** The dictionary stores
``w_enc (hidden, width)``, ``w_dec (width, hidden)``, ``b_enc (width,)``, ``threshold (width,)``,
``b_dec (hidden,)``. So a feature direction is a *row* of ``w_dec``, encoding is
``h @ w_enc + b_enc``, and decoding is ``z @ w_dec + b_dec``. The lens convention is the loader's:
``LensMaps.apply(h, L) == h @ J.T``, so ``J x`` on a column vector is ``x @ J.T`` on a row.

**What is never materialised.** ``L D^T`` is vocabulary by width and, at this model's dimensions,
tens of gigabytes. ``feature_scores`` forms ``U = D J^T`` first (width by hidden, small), then takes
``U W^T`` a chunk of features at a time, keeping only each feature's top-k tokens.

**How tensors are read.** Straight from the safetensors container: the header names the dtype,
shape and byte range of every tensor, and the bytes are widened to float32 here. This matters for
one reason: the bf16 checkpoint stores its embedding in bfloat16, which NumPy has no dtype for, so
the convenience reader that hands NumPy arrays back cannot read the readout at all. Widening
bfloat16 is a bit shift into the high half of a float32 and is exact.

**Hook alignment is a test, not a comment.** The dictionary names its hook as the output of block
``N``; this repository's capture convention (``probes/capture.py``) numbers the residual after block
``N`` as layer ``N + 1``, and the lens archive keys map ``J{L-1}`` to layer ``L``. So block ``N``
reads lens layer ``N + 1``. :func:`layer_for_hook` is the one place that arithmetic lives, and
:func:`hook_alignment` refuses a lens that has no map at that layer.

**Labels.** No published label maps to the every-layer suite this bridge reads, so every artefact
ships its label field as ``unlabelled`` with the reason verbatim (:data:`UNLABELLED_REASON`). A
readout this module computes is headed as ours and names the layer it was read from; it is never
presented as a published label.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "UNLABELLED_REASON",
    "JumpReLUDictionary",
    "Unembedding",
    "bridge_provenance",
    "convention_check",
    "decode",
    "decompose_position",
    "encode",
    "feature_score_column",
    "feature_scores",
    "hook_alignment",
    "layer_for_hook",
    "lens_scores",
    "load_dictionary",
    "load_unembedding",
    "negative_control",
    "overlap_at_k",
    "read_tensor",
    "safetensors_header",
    "top_tokens",
]

#: Recorded verbatim in every artefact, on the second amendment's rule. The dictionary this bridge
#: reads has no published labels at any layer; the labels that exist index a different training run
#: at a sparsity this suite never published.
UNLABELLED_REASON = (
    "no Neuronpedia source maps to resid_post_all; the residual labels index "
    "resid_post/layer_17_width_16k_l0_medium, a different training run at a sparsity this suite "
    "never published"
)

#: The dictionary's hook naming: the output of decoder block N.
_HOOK = re.compile(r"^model\.layers\.(\d+)\.output$")

#: Feature chunk for the batched readout. Sized so one chunk of scores stays well under a gigabyte
#: at this model's vocabulary; it changes no value.
DEFAULT_CHUNK = 512


# ------------------------------------------------------------------------- safetensors, raw


def safetensors_header(path: Path | str) -> tuple[dict[str, Any], int]:
    """The container's header and the byte offset at which tensor data begins."""
    path = Path(path)
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def read_tensor(path: Path | str, name: str) -> np.ndarray:
    """One tensor by name, widened to float32; bfloat16 by bit shift, which is exact.

    The container is little-endian by specification. A bfloat16 value is the high sixteen bits of
    the float32 with the same value, so placing those bits in the high half of a zeroed 32-bit word
    and viewing it as float32 reproduces the value with no rounding at all.
    """
    path = Path(path)
    header, base = safetensors_header(path)
    if name not in header:
        raise KeyError(f"{path.name} has no tensor {name!r}")
    meta = header[name]
    start, end = meta["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(base + start)
        raw = handle.read(end - start)
    dtype = meta["dtype"]
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << np.uint32(16)
        array = bits.view(np.float32)
    elif dtype == "F32":
        array = np.frombuffer(raw, dtype="<f4")
    elif dtype == "F16":
        array = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "F64":
        array = np.frombuffer(raw, dtype="<f8").astype(np.float32)
    else:
        raise ValueError(f"{path.name}:{name}: dtype {dtype} is not a float this reader widens")
    return np.ascontiguousarray(array.reshape(meta["shape"]), dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------------- the dictionary


@dataclass(frozen=True)
class JumpReLUDictionary:
    """One Gemma Scope residual dictionary, with what identifies it."""

    w_enc: np.ndarray
    b_enc: np.ndarray
    threshold: np.ndarray
    w_dec: np.ndarray
    b_dec: np.ndarray
    config: dict[str, Any]
    sha256: str
    source: str

    @property
    def hidden_size(self) -> int:
        return int(self.w_dec.shape[1])

    @property
    def width(self) -> int:
        return int(self.w_dec.shape[0])

    @property
    def hook_point(self) -> str:
        return str(self.config["hf_hook_point_in"])


def load_dictionary(params: Path | str, config: Path | str, *, source: str = "") -> JumpReLUDictionary:
    """Load a JumpReLU dictionary and refuse one whose shapes disagree with each other.

    Every dimension is read from the tensors. Nothing here names a width or a hidden size; a
    dictionary of any size loads, and one whose five tensors do not describe a single
    ``(hidden, width)`` pair is refused by name.
    """
    params, config = Path(params), Path(config)
    conf = json.loads(config.read_text(encoding="utf-8"))
    if conf.get("architecture") != "jump_relu":
        raise ValueError(f"{config}: architecture {conf.get('architecture')!r} is not jump_relu")
    header, _ = safetensors_header(params)
    need = {"w_enc", "b_enc", "threshold", "w_dec", "b_dec"}
    if set(header) != need:
        raise ValueError(f"{params}: tensors {sorted(header)} are not exactly {sorted(need)}")
    tensors = {name: read_tensor(params, name) for name in need}
    hidden, width = tensors["w_enc"].shape
    expect = {
        "w_enc": (hidden, width),
        "b_enc": (width,),
        "threshold": (width,),
        "w_dec": (width, hidden),
        "b_dec": (hidden,),
    }
    for name, shape in expect.items():
        if tensors[name].shape != shape:
            raise ValueError(f"{params}: {name} is {tensors[name].shape}, expected {shape}")
    if int(conf.get("width", width)) != width:
        raise ValueError(f"{config}: width {conf.get('width')} disagrees with w_dec {width}")
    for name, value in tensors.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{params}: {name} is not finite")
    return JumpReLUDictionary(
        w_enc=tensors["w_enc"],
        b_enc=tensors["b_enc"],
        threshold=tensors["threshold"],
        w_dec=tensors["w_dec"],
        b_dec=tensors["b_dec"],
        config=conf,
        sha256=_sha256(params),
        source=source or str(params),
    )


def encode(dictionary: JumpReLUDictionary, h: np.ndarray) -> np.ndarray:
    """JumpReLU: pass the pre-activation through where it exceeds its feature's threshold.

    The reference form is ``relu(pre) * (pre > threshold)``. With every threshold positive the
    ``relu`` is implied by the gate; it is kept so a dictionary with a non-positive threshold still
    encodes as the reference does rather than as a shortcut assumes.
    """
    h = np.asarray(h, dtype=np.float32)
    pre = h @ dictionary.w_enc + dictionary.b_enc
    return np.where(pre > dictionary.threshold, np.maximum(pre, 0.0), 0.0).astype(np.float32)


def decode(dictionary: JumpReLUDictionary, z: np.ndarray) -> np.ndarray:
    return np.asarray(z, dtype=np.float32) @ dictionary.w_dec + dictionary.b_dec


# ------------------------------------------------------------------------------------ budget


def reconstruction_budget(
    dictionary: JumpReLUDictionary, residuals: np.ndarray, *, dominance: float
) -> dict[str, Any]:
    """Per-site residual share ``|e| / |h|`` and active-feature count, against a declared budget.

    ``residuals`` is ``(sites, hidden)``: one row per position the bridge would decompose. For
    each, ``e = h - decode(encode(h))`` is what the dictionary did not explain, and its share of
    ``|h|`` is the quantity ``decompose_position`` refuses to rank on when it exceeds
    ``dominance``. This reports the distribution of that share over the set, and how many of the
    sites the bridge would decline, so a set of real activations can be judged before any
    per-site table is drawn. ``dominance`` is an input and is echoed, never chosen here.
    """
    if not 0.0 < float(dominance) <= 1.0:
        raise ValueError(f"dominance must be in (0, 1], not {dominance!r}")
    H = np.asarray(residuals, dtype=np.float32)
    if H.ndim != 2 or H.shape[1] != dictionary.hidden_size:
        raise ValueError(
            f"residuals must be (sites, {dictionary.hidden_size}), not {tuple(H.shape)}"
        )
    z = encode(dictionary, H)
    e = H - decode(dictionary, z)
    h_norm = np.linalg.norm(H, axis=1)
    if not np.all(h_norm > 0):
        raise ValueError("a zero residual has no share to report")
    share = np.linalg.norm(e, axis=1) / h_norm
    active = (z > 0).sum(axis=1)
    over = share > float(dominance)
    q = [0.0, 0.25, 0.5, 0.75, 1.0]
    return {
        "dominance": float(dominance),
        "sites": int(H.shape[0]),
        "residual_share": share,
        "active_features": active,
        "rankable": ~over,
        "share_over_dominance": float(over.mean()),
        "residual_share_quantiles": {str(x): float(v) for x, v in zip(q, np.quantile(share, q), strict=True)},
        "active_features_quantiles": {
            str(x): int(v) for x, v in zip(q, np.quantile(active, q), strict=True)
        },
    }


# ------------------------------------------------------------------------------ intervention


def intervention_parts(dictionary: JumpReLUDictionary, *, dtype: Any, device: Any = "cpu"):
    """The three things ``sae_intervention.SAEIntervention`` takes, in its own column convention.

    The wrapper's contract, from its docstring: ``decoder`` is ``[residual, feature]``, which for
    a stored ``w_dec`` of ``(width, hidden)`` is the transpose; ``bias`` is supplied separately and
    never folded in; and the encoder callable returns a vector in the residual's dtype and device.
    Dictionary precision is the caller's declared choice, made here: the encoder computes in
    ``dtype`` and casts its result back to the residual's, so a bfloat16 residual gets a bfloat16
    code and the record can say what precision the code was formed in. Torch is imported here and
    not at module level; everything else in this module is numpy.

    Returns ``(encoder, decoder, bias, declared)`` where ``declared`` is the provenance to record.
    """
    import torch

    def tensor(a: np.ndarray):
        # The raw reader hands back read-only views of the file's bytes; torch wants its own copy.
        return torch.as_tensor(np.array(a, dtype=np.float32, copy=True), dtype=dtype, device=device)

    w_enc, b_enc, threshold = tensor(dictionary.w_enc), tensor(dictionary.b_enc), tensor(dictionary.threshold)
    decoder, bias = tensor(dictionary.w_dec.T), tensor(dictionary.b_dec)

    def encoder(h):
        pre = h.to(dtype=dtype) @ w_enc + b_enc
        z = torch.relu(pre) * (pre > threshold).to(dtype)
        return z.to(dtype=h.dtype, device=h.device)

    declared = {
        "decoder_layout": "[residual, feature]: the stored w_dec (width, hidden) transposed",
        "bias": "b_dec, supplied separately, never folded into the decoder",
        "encoder": "jump_relu, relu(pre) * (pre > threshold)",
        "dictionary_precision": str(dtype).replace("torch.", ""),
        "device": str(device),
        "code_dtype": "the residual's, cast on return",
        "params_sha256": dictionary.sha256,
    }
    return encoder, decoder, bias, declared


# --------------------------------------------------------------------------------- alignment


def layer_for_hook(hook_point: str) -> int:
    """The repository layer a dictionary hook reads: block ``N``'s output is layer ``N + 1``.

    The one place the off-by-one lives. ``probes/capture.py``: layer ``L`` is the residual stream
    after block ``L - 1``; the lens archive: key ``J{L-1}`` is layer ``L``. A hook naming block
    ``N`` therefore reads layer ``N + 1`` on both.
    """
    found = _HOOK.match(hook_point)
    if not found:
        raise ValueError(
            f"hook {hook_point!r} does not name a decoder block's output; the bridge reads the "
            "residual stream and nothing else"
        )
    return int(found.group(1)) + 1


def dictionary_base(dictionary: JumpReLUDictionary) -> str:
    """The base checkpoint the dictionary's config says it was trained on, resolved as a lens is.

    Gemma Scope configs carry ``model_name``. It goes through the same registry resolution the
    lens identity went through, so a registry name, a local conversion path and an upstream id
    all compare as the base they descend from. A config that names no model is refused: nothing
    else in the dictionary says which model it fits, and width and depth agree between a base
    checkpoint and that checkpoint after further training.
    """
    from local_llm_lab.models import base_of_artifact

    named = dictionary.config.get("model_name")
    if not isinstance(named, str) or not named:
        raise ValueError(
            f"dictionary {dictionary.source or dictionary.sha256[:12]!r} names no model in its "
            "config (`model_name`); it cannot be aligned to a lens"
        )
    return base_of_artifact(named)


def hook_alignment(dictionary: JumpReLUDictionary, lens: Any, *, base: str | None = None) -> int:
    """The lens layer this dictionary reads, refusing a lens that is not of the same model.

    Three things are compared and any disagreement refuses: the dictionary's ``model_name``,
    the lens identity's base, and, when the bridge runs from a registry entry, that entry's
    ``base``. Layer and hidden size are checked too, but they are the checks a dictionary
    trained on a different checkpoint of the same architecture passes.
    """
    dict_base = dictionary_base(dictionary)
    identity = getattr(lens, "identity", None)
    if identity is None or not getattr(identity, "base", None):
        raise ValueError(
            f"lens carries no identity, so it cannot be confirmed to map {dict_base!r}, the "
            "model the dictionary names"
        )
    if identity.base != dict_base:
        raise ValueError(
            f"dictionary was trained on {dict_base!r} and the lens maps {identity.base!r}; "
            "same width and depth do not make them the same model"
        )
    if base is not None:
        from local_llm_lab.models import base_of_artifact

        entry_base = base_of_artifact(base)
        if entry_base != dict_base:
            raise ValueError(
                f"the registry entry descends from {entry_base!r}; dictionary and lens are of "
                f"{dict_base!r}"
            )
    layer = layer_for_hook(dictionary.hook_point)
    if layer not in lens.maps:
        raise ValueError(
            f"dictionary hook {dictionary.hook_point!r} reads layer {layer}, and the lens carries "
            f"no map at that layer (it has {min(lens.maps)}..{max(lens.maps)})"
        )
    if lens.hidden_size != dictionary.hidden_size:
        raise ValueError(
            f"lens width {lens.hidden_size} is not the dictionary's {dictionary.hidden_size}"
        )
    return layer


# ------------------------------------------------------------------------------ unembedding


@dataclass(frozen=True)
class Unembedding:
    """The readout matrix and the final norm's gain, read from a checkpoint without a model."""

    weight: np.ndarray
    gain: np.ndarray
    tied: bool
    checkpoint: str
    tensor_names: dict[str, str]

    @property
    def vocab_size(self) -> int:
        return int(self.weight.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.weight.shape[1])


def _weight_map(directory: Path) -> dict[str, str]:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        return dict(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    single = directory / "model.safetensors"
    if not single.is_file():
        raise ValueError(f"{directory} holds neither model.safetensors nor an index")
    header, _ = safetensors_header(single)
    return {name: single.name for name in header}


def _only(candidates: list[str], what: str) -> str:
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one {what} tensor, found {candidates}")
    return candidates[0]


def load_unembedding(checkpoint: Path | str) -> Unembedding:
    """Read ``W`` and ``1 + w`` by tensor name, never by loading the model.

    Names are matched by suffix so the wrapper prefix a multimodal snapshot puts in front of the
    text tower does not matter. A head that is a separate tensor is used if present; otherwise the
    embedding is the head, and ``tied`` records which. The gain is ``1 + w`` because Gemma's norm
    stores its scale as an offset from one; the stored vector applied raw would be near noise.
    """
    directory = Path(checkpoint)
    weight_map = _weight_map(directory)
    names = sorted(weight_map)
    heads = [n for n in names if n.endswith("lm_head.weight")]
    embeds = [n for n in names if n.endswith("embed_tokens.weight")]
    norms = [n for n in names if n.endswith(".norm.weight") and ".layers." not in n]
    norm_name = _only(norms, "final norm")
    head_name = heads[0] if heads else _only(embeds, "embedding")
    weight = read_tensor(directory / weight_map[head_name], head_name)
    norm = read_tensor(directory / weight_map[norm_name], norm_name)
    if weight.ndim != 2 or norm.shape != (weight.shape[1],):
        raise ValueError(f"head {weight.shape} and norm {norm.shape} do not describe one width")
    return Unembedding(
        weight=weight,
        gain=(1.0 + norm).astype(np.float32),
        tied=not heads,
        checkpoint=str(directory),
        tensor_names={"head": head_name, "norm": norm_name},
    )


# ------------------------------------------------------------------------------- the readout


def _through_lens(x: np.ndarray, lens_map: np.ndarray | None) -> np.ndarray:
    """``J x`` for rows of ``x``, in the loader's convention; identity when there is no map."""
    return x if lens_map is None else x @ lens_map.T


def lens_scores(
    unembedding: Unembedding, lens_map: np.ndarray | None, x: np.ndarray, *, gain: bool = True
) -> np.ndarray:
    """``L x`` for rows of ``x``: ``W ((J x) * g)``, or without the gain, or without the lens."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    u = _through_lens(x, lens_map)
    if gain:
        u = u * unembedding.gain
    return u @ unembedding.weight.T


def top_tokens(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` token ids and scores per row, highest first."""
    scores = np.atleast_2d(scores)
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-vals, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)


def feature_scores(
    dictionary: JumpReLUDictionary,
    unembedding: Unembedding,
    lens_map: np.ndarray | None,
    *,
    k: int,
    gain: bool = True,
    chunk: int = DEFAULT_CHUNK,
) -> tuple[np.ndarray, np.ndarray]:
    """A1, batched: top-``k`` tokens of ``L d_i`` for every feature, never forming ``L D^T``.

    ``U = D J^T`` (width by hidden) is formed once and is small; ``U W^T`` is taken ``chunk``
    features at a time and only each feature's top-``k`` survives.
    """
    if chunk < 1:
        raise ValueError("chunk must be positive")
    u = _through_lens(dictionary.w_dec, lens_map)
    if gain:
        u = u * unembedding.gain
    idx = np.empty((dictionary.width, k), dtype=np.int64)
    val = np.empty((dictionary.width, k), dtype=np.float32)
    for start in range(0, dictionary.width, chunk):
        stop = min(start + chunk, dictionary.width)
        i, v = top_tokens(u[start:stop] @ unembedding.weight.T, k)
        idx[start:stop], val[start:stop] = i, v
    return idx, val


def feature_score_column(
    dictionary: JumpReLUDictionary,
    unembedding: Unembedding,
    lens_map: np.ndarray | None,
    feature: int,
    *,
    gain: bool = True,
) -> np.ndarray:
    """A1, the other way: ``L d_i`` for one feature as a direct product, for the two-ways check."""
    d = dictionary.w_dec[feature]
    v = d if lens_map is None else lens_map @ d
    if gain:
        v = v * unembedding.gain
    return unembedding.weight @ v


def overlap_at_k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per row, the fraction of token ids the two top-``k`` sets share."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if a.shape != b.shape:
        raise ValueError(f"top-k sets differ in shape: {a.shape} vs {b.shape}")
    out = np.empty(a.shape[0], dtype=np.float64)
    for r in range(a.shape[0]):
        out[r] = len(set(a[r].tolist()) & set(b[r].tolist())) / a.shape[1]
    return out


def negative_control(
    dictionary: JumpReLUDictionary,
    unembedding: Unembedding,
    lens: Any,
    layer: int,
    other_layer: int,
    *,
    k: int,
    features: np.ndarray | None = None,
) -> dict[str, Any]:
    """A bridge that cannot tell layers apart is not reading one.

    Reads the same decoder through the lens at ``layer`` and at ``other_layer`` and reports how
    much of each feature's top-``k`` the two share. The check *bites* when the layers are
    different and the overlap is well below one. It is *shown to fail* by passing the same layer
    twice: the overlap is then exactly one, and ``distinguishes`` is false -- which is the
    behaviour a control must exhibit under a deliberate mismatch, or it is not a control.
    """
    sub = dictionary if features is None else _subset(dictionary, features)
    here, _ = feature_scores(sub, unembedding, lens.maps[layer], k=k)
    there, _ = feature_scores(sub, unembedding, lens.maps[other_layer], k=k)
    overlap = overlap_at_k(here, there)
    return {
        "layer": int(layer),
        "other_layer": int(other_layer),
        "k": int(k),
        "features": int(sub.width),
        "mean_overlap": float(overlap.mean()),
        "max_overlap": float(overlap.max()),
        "share_identical": float((overlap == 1.0).mean()),
        "distinguishes": bool(layer != other_layer and overlap.mean() < 1.0),
    }


def _subset(dictionary: JumpReLUDictionary, features: np.ndarray) -> JumpReLUDictionary:
    f = np.asarray(features, dtype=np.int64)
    return JumpReLUDictionary(
        w_enc=dictionary.w_enc[:, f],
        b_enc=dictionary.b_enc[f],
        threshold=dictionary.threshold[f],
        w_dec=dictionary.w_dec[f],
        b_dec=dictionary.b_dec,
        config=dictionary.config,
        sha256=dictionary.sha256,
        source=dictionary.source,
    )


def convention_check(
    shipped: np.ndarray, raw: np.ndarray, gained: np.ndarray, *, threshold: float = 0.5
) -> dict[str, Any]:
    """The second amendment's discriminator, as a decision and not a number.

    ``shipped`` is the dictionary's own published top-``k`` per feature; ``raw`` is ours with the
    lens replaced by the identity and no gain; ``gained`` the same with the gain applied. Run the
    raw arm first and record it before looking at the other, so the first number is not chosen
    after seeing which convention agrees. Only the fourth row is a stop.
    """
    r = float(overlap_at_k(shipped, raw).mean())
    g = float(overlap_at_k(shipped, gained).mean())
    raw_ok, gain_ok = r >= threshold, g >= threshold
    if raw_ok and not gain_ok:
        conclusion, stop = "shipped file is raw; convention settled, bridge proceeds", False
    elif gain_ok and not raw_ok:
        conclusion, stop = "shipped file applies the final gain; convention settled, bridge proceeds", False
    elif raw_ok and gain_ok:
        conclusion, stop = "uninformative: both arms agree, the check has no power here", False
    else:
        conclusion, stop = "not a convention problem: orientation or tokenizer indexing; stop", True
    return {
        "raw_overlap": r,
        "gain_overlap": g,
        "threshold": threshold,
        "conclusion": conclusion,
        "stop": stop,
    }


# --------------------------------------------------------------------------- decomposition


def decompose_position(
    dictionary: JumpReLUDictionary,
    unembedding: Unembedding,
    lens_map: np.ndarray | None,
    h: np.ndarray,
    emitted_token: int,
    *,
    k: int = 10,
    dominance: float = 0.5,
    gain: bool = True,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """A2 at one position: which features push the score toward the emitted token, and how much
    of the score the dictionary did not explain.

    Computes every term of ``L h = L b + sum_i z_i (L d_i) + L e`` and asserts the identity on the
    emitted token before reporting anything, so a wrong orientation cannot produce a ranked table.
    If ``|L e| / |L h|`` exceeds ``dominance`` the decomposition is describing the dictionary's
    failure and not the model, and the result says so instead of ranking under it.
    """
    h = np.asarray(h, dtype=np.float32).reshape(-1)
    if h.shape[0] != dictionary.hidden_size:
        raise ValueError(f"activation has width {h.shape[0]}, dictionary {dictionary.hidden_size}")
    z = encode(dictionary, h)
    e = h - decode(dictionary, z)

    def score(x: np.ndarray) -> np.ndarray:
        return lens_scores(unembedding, lens_map, x, gain=gain)[0]

    lh, lb, le = score(h), score(dictionary.b_dec), score(e)
    # Row v of L is (W[v] * g) J, so the per-feature term z_i (L d_i)_v is z * (D (L[v])).
    lv = unembedding.weight[emitted_token] * (unembedding.gain if gain else 1.0)
    lv = lv if lens_map is None else lv @ lens_map
    contributions = z * (dictionary.w_dec @ lv)

    identity_gap = float(
        abs(lh[emitted_token] - (lb[emitted_token] + contributions.sum() + le[emitted_token]))
    )
    scale = float(abs(lh[emitted_token])) + 1e-6
    if identity_gap > tolerance * max(1.0, scale):
        raise ValueError(
            f"the score decomposition is not exact at token {emitted_token}: gap {identity_gap:.3e} "
            f"against a score of {scale:.3e}; orientation or convention is wrong"
        )
    share = float(np.linalg.norm(le) / (np.linalg.norm(lh) + 1e-12))
    active = int((z > 0).sum())
    out: dict[str, Any] = {
        "emitted_token": int(emitted_token),
        "score": float(lh[emitted_token]),
        "bias_term": float(lb[emitted_token]),
        "residual_term": float(le[emitted_token]),
        "feature_sum": float(contributions.sum()),
        "identity_gap": identity_gap,
        "residual_share": share,
        "active_features": active,
        "dominance_threshold": dominance,
        "ranked": share <= dominance,
    }
    if share > dominance:
        out["reason"] = (
            "residual dominates: the decomposition describes the dictionary's failure at this "
            "position, not the model; features are not ranked under it"
        )
        return out
    order = np.argsort(-np.abs(contributions), kind="stable")[:k]
    out["top_features"] = [
        {"feature": int(i), "z": float(z[i]), "contribution": float(contributions[i])}
        for i in order
        if z[i] > 0
    ]
    return out


# -------------------------------------------------------------------------------- provenance


def bridge_provenance(
    dictionary: JumpReLUDictionary,
    lens: Any,
    unembedding: Unembedding,
    layer: int,
    *,
    dictionary_repo: str,
    dictionary_folder: str,
) -> dict[str, Any]:
    """Every artefact carries this block (R57, R60): what was read, from where, under which conventions."""
    return {
        "checkpoint": {
            "path": unembedding.checkpoint,
            "head_tensor": unembedding.tensor_names["head"],
            "norm_tensor": unembedding.tensor_names["norm"],
            "tied_embeddings": unembedding.tied,
        },
        "lens": {
            "sha256": lens.sha256,
            "identity": lens.identity.as_dict() if lens.identity is not None else None,
            "storage_dtype": list(lens.storage_dtype),
            "layer": int(layer),
            "convention": "maps[L] is the archive's J{L-1}; L x for a row is x @ J.T",
        },
        "dictionary": {
            "repo": dictionary_repo,
            "folder": dictionary_folder,
            "hook_point": dictionary.hook_point,
            "width": dictionary.width,
            "l0": dictionary.config.get("l0"),
            "architecture": dictionary.config.get("architecture"),
            "params_sha256": dictionary.sha256,
            "decoder_rows_unit_norm": bool(
                np.allclose(np.linalg.norm(dictionary.w_dec, axis=1), 1.0, atol=1e-2)
            ),
        },
        "residual_convention": (
            "the dictionary hook is the output of decoder block N; the capture convention numbers "
            "that residual as layer N + 1 and the lens keys it as J{N}; the bridge reads layer "
            f"{layer}"
        ),
        "score_map": (
            "L x = W ((J x) * (1 + w)); the RMS normaliser is a positive per-vector scalar and is "
            "dropped so the map is linear; ranks and overlaps are unaffected, score values are not "
            "logits"
        ),
        "tensor_reader": "safetensors header and bytes, bfloat16 widened to float32 by bit shift",
        "labels": "unlabelled",
        "labels_reason": UNLABELLED_REASON,
    }
