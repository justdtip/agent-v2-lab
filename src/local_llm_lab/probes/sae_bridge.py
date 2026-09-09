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


#: The cross-path term, **per layer**, and at every layer it is a refusal rather than a note
#: until the pairing is measured there (WS-D's displacement control `e771c2b`, as amended by
#: WS-A's map/anchor audit `ca396fb`).
#:
#: The width-rows ruling keeps captures native at width 1 while lenses are fitted in float32, so a
#: reading that puts the two together crosses paths. The displacement control then displaced the
#: anchor those derivatives were read at and measured how far they moved. In coherent float32 at
#: layer 1 the sampled scalar derivatives moved by a median 1.035 of their own size, against 0.471
#: natively, and an equal-norm random displacement moved them as much or more. **The sensitivity
#: is not the arithmetic's.** A Jacobian lens is a statement about the neighbourhood it was
#: fitted in, and that neighbourhood is a per-layer quantity.
#:
#: What that is **not**: a condition number. The audit withdrew the amplification ratios once
#: quoted here, because they divided a derivative change by a displacement norm measured at one
#: position while the intervention replaced the whole sequence; and a ratio of one scalar
#: projection over a finite displacement is a directional sensitivity in any case. So the per-layer
#: figures are the sampled distribution, median with its range and count, and the refusal cites
#: them as that. Nor does a median settle a layer: at layer 33 the float32 median is 0.125 and the
#: maximum over the same eighteen projections is 91.9, so a late layer is not shown benign by its
#: middle.
#:
#: Two consequences the code carries rather than a reader remembering them. A cross-path reading
#: is **refused by name** at any layer whose pairing has not been measured, citing that layer's
#: sampled sensitivity where it was measured, because at layer 1 the crossing is not an error bar
#: on the answer, it is the size of the answer. And the size of the crossing stays unmeasured
#: until a paired comparison exists **at that layer**: the two promoted-floor percentages below
#: are historical context about residuals at two lengths, not a bound on a readout through a lens,
#: for the reason W3 gives -- the readout's denominator may be small.
#:
#: A same-path reading carries no term and is not refused: a float32 generation read through a
#: float32 lens crosses nothing. Nor does a reading that touches no capture at all, such as the
#: dictionary-direction readout in :func:`feature_scores`, where there is no anchor to have moved.
LENS_PATH_TERM = {
    "term": "a lens read against a capture from a different width, precision or path",
    "measured": False,
    "status": "unmeasured at every layer; no paired comparison exists yet",
    "why": (
        "the lens is a statement about the neighbourhood it was fitted in: displacing the anchor "
        "moved the sampled scalar derivatives at layer 1 by a median 1.035 of their own size in "
        "float32, and by more than their own size at their worst, so the crossing is not a "
        "correction to the answer but potentially the whole of it. This is a sampled directional "
        "sensitivity, not a condition number, and it does not bound the readout"
    ),
    "to_measure": (
        "a paired comparison at that layer, at this reading's own positions and context: the same "
        "readout through the same lens on a capture from each path"
    ),
    "historical_context": [
        {
            "relative": 0.0124,
            "at_tokens": 64,
            "quantity": "promoted-float32 loop against bf16 native, residual-relative",
            "note": "1.2386% to four digits, and it agrees across CPU and CUDA",
        },
        {
            "relative": 0.694,
            "at_tokens": 1400,
            "quantity": "promoted-float32 loop against bf16 native, residual-relative",
            "note": "on CUDA; the same length reads 6.9% on the CPU, so it is not one number",
        },
    ],
    "context_basis": (
        "WS-A's first-hour promoted floor, recorded at two lengths on two backends; context for "
        "this crossing, not a calibrated term on any reading through a lens (width audit, W3)"
    ),
}


#: Anchor sensitivity and measured pairings, by model and by layer, held beside this module as
#: data. It is data and not code because it is a measurement of particular models at particular
#: layers: the quantity falls by more than two orders of magnitude across one stack, so it is no
#: more a property of this bridge than a checkpoint's weights are, and the model-constant rule is
#: right to keep such numbers out of a module that must work on the next model too.
ANCHOR_SENSITIVITY_PATH = Path(__file__).with_name("anchor_sensitivity.json")

_ANCHOR_TABLE: dict[str, Any] | None = None

#: The table's allowed shape, checked on load at **every** level. The point is not to ban numbers:
#: medians, ranges and counts are numbers, and a legitimate statistic may coincidentally equal a
#: withdrawn ratio, so a value blacklist would catch the wrong thing and miss the right one. What is
#: checked is the *quantity at each location*: a key nobody declared -- an ``amplification_ratio``
#: nested three levels down, say -- is refused wherever it appears, because a quantity this record
#: does not name is a quantity nobody ruled on.
_CELL_KEYS = frozenset({"median", "min", "max", "count"})
_ARM_KEYS = frozenset({"float32", "native"})
_LAYER_KEYS = frozenset(
    {"sampled_displacement", "equal_norm_random_displacement", "native_anchor_read_in_the_float32_tail"}
)
_MODEL_KEYS = frozenset(
    {"basis", "label", "quantity", "precision_convention", "layers", "reading_notes",
     "measured_pairings"}
)  # fmt: skip


def _check_keys(node: Any, allowed: frozenset[str], where: str, *, exact: bool = True) -> None:
    """Refuse a node carrying a key nobody declared, naming it and where it sits."""
    if not isinstance(node, dict):
        raise ValueError(
            f"{ANCHOR_SENSITIVITY_PATH.name}: {where} must be a mapping, not "
            f"{type(node).__name__}"
        )
    undeclared = sorted(set(node) - allowed)
    if undeclared:
        raise ValueError(
            f"{ANCHOR_SENSITIVITY_PATH.name}: {where} carries undeclared {undeclared}; a quantity "
            "this record does not name is a quantity nobody ruled on"
        )
    missing = sorted(allowed - set(node)) if exact else []
    if missing:
        raise ValueError(f"{ANCHOR_SENSITIVITY_PATH.name}: {where} is missing {missing}")


def _check_cell(cell: Any, where: str) -> None:
    _check_keys(cell, _CELL_KEYS, where)
    for key in ("median", "min", "max"):
        if isinstance(cell[key], bool) or not isinstance(cell[key], (int, float)):
            raise ValueError(f"{ANCHOR_SENSITIVITY_PATH.name}: {where}.{key} must be a number")
    if isinstance(cell["count"], bool) or not isinstance(cell["count"], int):
        raise ValueError(f"{ANCHOR_SENSITIVITY_PATH.name}: {where}.count must be an integer")


def _validate_anchor_table(table: dict[str, Any]) -> dict[str, Any]:
    """Refuse a table carrying any quantity this record does not name, at any depth."""
    for model, entry in table.items():
        _check_keys(entry, _MODEL_KEYS, repr(model), exact=False)
        for layer, measured in (entry.get("layers") or {}).items():
            where = f"{model}.layers[{layer}]"
            _check_keys(measured, _LAYER_KEYS, where)
            for arm in ("sampled_displacement", "equal_norm_random_displacement"):
                block = measured[arm]
                _check_keys(block, _ARM_KEYS, f"{where}.{arm}")
                for precision, cell in block.items():
                    _check_cell(cell, f"{where}.{arm}.{precision}")
            _check_cell(
                measured["native_anchor_read_in_the_float32_tail"],
                f"{where}.native_anchor_read_in_the_float32_tail",
            )
        for layer, pairing in (entry.get("measured_pairings") or {}).items():
            _validate_pairing(pairing, f"{model}.measured_pairings[{layer}]")
    return table


def anchor_table() -> dict[str, Any]:
    """The measurement table, read once from :data:`ANCHOR_SENSITIVITY_PATH` and validated."""
    global _ANCHOR_TABLE
    if _ANCHOR_TABLE is None:
        raw = json.loads(ANCHOR_SENSITIVITY_PATH.read_text(encoding="utf-8"))
        _ANCHOR_TABLE = _validate_anchor_table(
            {key: value for key, value in raw.items() if not key.startswith("_")}
        )
    return _ANCHOR_TABLE


def anchor_sensitivity(layer: int, *, base: str | None = None) -> dict[str, Any] | None:
    """What the displacement control measured at ``layer`` of ``base``, or ``None``.

    ``None`` covers three different absences on purpose, and none of them is filled in: a model
    nobody measured, a layer nobody measured on a model somebody did, and no model named at all.
    Returning a neighbouring layer's value instead would be a guess wearing a measurement's
    clothes.
    """
    entry = anchor_table().get(base or "", {})
    measured = (entry.get("layers") or {}).get(str(int(layer)))
    return dict(measured) if measured else None


def measured_layers(base: str | None = None) -> list[int]:
    """Which layers of ``base`` the control measured, in order; empty for an unmeasured model."""
    entry = anchor_table().get(base or "", {})
    return sorted(int(key) for key in (entry.get("layers") or {}))


#: What identifies the pair a cross-path measurement was taken on. A measurement is evidence about
#: *an experiment*, so lifting a refusal with it requires the reading to be that experiment. Model
#: and layer are necessary and nowhere near sufficient: the same layer at another capture
#: precision, another width, or with the paths reversed is a different object, which is the whole
#: finding of this week's controls.
#:
#: The lens's side comes from its own nu, so it cannot be asserted by the caller: ``fit_dtype`` and
#: ``fit_width`` are read from the precision block and ``nu_sha256`` digests the nu entire, which
#: pins the endpoint and the fit's positions with it. ``lens_side`` records which side of the pair
#: the lens was fitted on, so a measurement taken with the lens on the other side does not match a
#: reading with it on this one. The reading's own side the caller must declare, because only the
#: caller knows it: which positions, under which reduction, at which endpoint, at what context
#: length. A2's caller supplies its actual provenance or it gets no permission.
PAIR_FIELDS = (
    "fit_dtype",
    "fit_width",
    "capture_dtype",
    "capture_width",
    "lens_side",
    "nu_sha256",
    "positions",
    "reduction",
    "endpoint",
    "context_tokens",
)

#: The subset the caller declares about the reading; the rest is read off the lens.
READING_FIELDS = ("positions", "reduction", "endpoint", "context_tokens")


def _validate_pairing(pairing: Any, where: str) -> None:
    """A registered pairing carries the identity of the pair it measured, or it is not one."""
    if not isinstance(pairing, dict):
        raise ValueError(f"{ANCHOR_SENSITIVITY_PATH.name}: {where} must be a mapping")
    pair = pairing.get("pair")
    if not isinstance(pair, dict) or set(pair) != set(PAIR_FIELDS):
        missing = sorted(set(PAIR_FIELDS) - set(pair)) if isinstance(pair, dict) else list(PAIR_FIELDS)
        extra = sorted(set(pair) - set(PAIR_FIELDS)) if isinstance(pair, dict) else []
        raise ValueError(
            f"{ANCHOR_SENSITIVITY_PATH.name}: {where}.pair must carry exactly {list(PAIR_FIELDS)}"
            + (f"; missing {missing}" if missing else "")
            + (f"; undeclared {extra}" if extra else "")
        )


def nu_digest(nu: dict | None) -> str | None:
    """A digest of the lens's nu, which pins its endpoint and its fit's positions along with it."""
    if not isinstance(nu, dict):
        return None
    canonical = json.dumps(nu, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def reading_identity(
    nu: dict | None,
    *,
    capture_dtype: str | None,
    capture_batch: int | None,
    reading: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The identity of the pair a proposed reading would be, with ``None`` for what it cannot say.

    Nothing is defaulted. A field the caller did not declare stays ``None``, which cannot equal a
    registered value, so an undeclared reading is unmeasured rather than permitted -- the failure
    direction a guard must have.
    """
    fit = lens_fit_precision(nu)
    declared_reading = reading or {}
    identity: dict[str, Any] = {
        "fit_dtype": fit["fit_dtype"],
        "fit_width": fit["forward_batch"],
        "capture_dtype": capture_dtype,
        "capture_width": capture_batch,
        "lens_side": "fit",
        "nu_sha256": nu_digest(nu),
    }
    for field in READING_FIELDS:
        identity[field] = declared_reading.get(field)
    return identity


def path_pairing(
    layer: int, *, base: str | None = None, identity: dict[str, Any] | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """The measured pairing at ``layer`` of ``base`` **for this identity**, and why not if not.

    Returns ``(pairing, None)`` only when a registered measurement's pair matches the proposed
    reading in every bound field. Otherwise ``(None, reason)``: no measurement at that layer, one
    whose identity is incomplete, or one taken on a different pair, with the differing fields
    named. Model and layer alone never suffice, which was the defect this replaced.
    """
    entry = anchor_table().get(base or "", {})
    registered = (entry.get("measured_pairings") or {}).get(str(int(layer)))
    if not registered:
        return None, f"no cross-path pairing is registered at layer {layer} of {base!r}"
    try:
        _validate_pairing(registered, f"{base}.measured_pairings[{layer}]")
    except ValueError as error:
        return None, f"the registered pairing does not identify its pair, so it measures nothing: {error}"
    if identity is None:
        return None, "the proposed reading declared no identity, so it matches no measurement"
    pair = registered["pair"]
    undeclared = [field for field in PAIR_FIELDS if identity.get(field) is None]
    if undeclared:
        return None, (
            f"the reading does not declare {undeclared}, and an undeclared field cannot match a "
            "measured one; the caller supplies its own provenance"
        )
    differing = [field for field in PAIR_FIELDS if pair[field] != identity[field]]
    if differing:
        detail = "; ".join(
            f"{field}: measured {pair[field]!r}, this reading {identity[field]!r}"
            for field in differing
        )
        return None, (
            f"a pairing is registered at layer {layer}, but it measured a different pair -- {detail}"
        )
    return dict(registered), None


def path_term_for_layer(
    layer: int, *, base: str | None = None, identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The cross-path term as it stands for one layer and one proposed pair: the measurement if
    one was taken on that pair, else the standing statement that none was, with that layer's
    sensitivity attached and the reason no measurement applies."""
    measured, reason = path_pairing(layer, base=base, identity=identity)
    if measured is not None:
        return {"layer": int(layer), "base": base, "measured": True, **measured}
    term = dict(LENS_PATH_TERM)
    term["layer"] = int(layer)
    term["base"] = base
    term["unmeasured_because"] = reason
    term["proposed_pair"] = dict(identity) if identity else None
    term["anchor_sensitivity"] = anchor_sensitivity(layer, base=base)
    entry = anchor_table().get(base or "", {})
    term["anchor_sensitivity_basis"] = entry.get("basis")
    term["anchor_sensitivity_label"] = entry.get("label")
    term["anchor_sensitivity_quantity"] = entry.get("quantity")
    term["anchor_sensitivity_notes"] = list(entry.get("reading_notes") or [])
    term["historical_context"] = [dict(entry) for entry in LENS_PATH_TERM["historical_context"]]
    return term


#: The three fields the width-rows ruling requires a lens's nu to carry. Spelled as the fitter
#: spells them, in nu's own precision block, so there is one name per quantity in the tree.
FIT_PRECISION_KEYS = ("fit_dtype", "forward_batch", "anchor_batch")


def lens_fit_precision(nu: dict | None) -> dict[str, Any]:
    """The fit precision a lens declares, read from its nu; every field ``None`` when it declares
    none.

    Reads ``nu["precision"]`` and nothing else. A lens fitted before the ruling carries no such
    keys, and that is reported as an absence rather than filled in with a default: an assumed
    ``float32`` here would be the inert kind of check, passing exactly the case it exists to
    catch.
    """
    precision = ((nu or {}).get("precision") or {}) if isinstance(nu, dict) else {}
    return {key: precision.get(key) for key in FIT_PRECISION_KEYS}


def fit_precision_record(
    nu: dict | None,
    *,
    declared: str | None,
    layer: int | None = None,
    base: str | None = None,
    capture_dtype: str | None = None,
    capture_batch: int | None = None,
    reading: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare a lens's declared fit precision with the registry's, and record the path term.

    Refuses two ways on the fit precision, each naming the field and both sides: a registry that
    declares a fit precision against a lens whose nu does not say what it was fitted in, and a
    lens whose ``fit_dtype`` is not the declared one. Both silent is **not** a refusal: it is
    every lens fitted before the ruling, including upstream's hosted one, and the record says the
    fit precision is undeclared rather than pretending to have checked it. What a Jacobian is a
    Jacobian *of* is the fitter's gate to keep, so the widths are recorded here, not gated.

    And a third way, on the reading rather than the artefacts: ``capture_dtype`` and
    ``capture_batch`` describe the capture this reading will be taken on, and a capture from a
    different precision or width than the fit is a **cross-path reading**, refused by name at any
    layer whose pairing is unmeasured. ``capture_dtype=None`` means no capture is read at all,
    which is the dictionary-direction readout and crosses nothing; passing the capture's own
    dtype is what a caller does when a residual is involved.
    """
    fit = lens_fit_precision(nu)
    fit_dtype = fit["fit_dtype"]
    if declared is not None and fit_dtype is None:
        raise ValueError(
            f"the registry declares probes.lens_fit_dtype {declared!r} and this lens's nu carries "
            "no `fit_dtype`, so it does not say what arithmetic it was fitted in. In bfloat16 the "
            "derivative depends on the forward's batch width, so an undeclared fit is not "
            "assumed to be the declared one (WS-D, the width rows)"
        )
    if declared is not None and fit_dtype != declared:
        raise ValueError(
            f"this lens declares fit_dtype {fit_dtype!r} and the registry declares "
            f"probes.lens_fit_dtype {declared!r}; a map fitted in one arithmetic is not a map "
            "fitted in the other"
        )
    status = "undeclared" if declared is None and fit_dtype is None else "declared"
    record: dict[str, Any] = {
        "declared_by_registry": declared,
        "capture_dtype": capture_dtype,
        "capture_batch": capture_batch,
        "layer": None if layer is None else int(layer),
        "status": status,
        **fit,
    }
    if status == "undeclared":
        record["note"] = (
            "neither the registry nor the lens declares a fit precision; this reading does not "
            "say what arithmetic its map was taken in, and upstream's hosted lens is by "
            "measurement a schedule-specific object (WS-D, the width rows)"
        )
    crossings = []
    if capture_dtype is not None and fit_dtype is not None and capture_dtype != fit_dtype:
        crossings.append(f"the lens was fitted in {fit_dtype} and the capture is {capture_dtype}")
    if (
        capture_batch is not None
        and fit["forward_batch"] is not None
        and int(capture_batch) != int(fit["forward_batch"])
    ):
        crossings.append(
            f"the lens was fitted at forward width {fit['forward_batch']} and the capture is at "
            f"width {capture_batch}"
        )
    if not crossings:
        return record

    record["crossing"] = crossings
    if layer is None:
        raise ValueError(
            "this is a cross-path reading ("
            + "; ".join(crossings)
            + ") and no layer was given, so whether its pairing has been measured cannot be "
            "decided. A lens is a statement about the neighbourhood it was fitted in, and that "
            "neighbourhood is a per-layer quantity (WS-D, the displacement control)"
        )
    identity = reading_identity(
        nu, capture_dtype=capture_dtype, capture_batch=capture_batch, reading=reading
    )
    record["proposed_pair"] = identity
    term = path_term_for_layer(layer, base=base, identity=identity)
    record["path_term"] = term
    if term["measured"]:
        return record

    sensitivity = term["anchor_sensitivity"]
    if sensitivity is None:
        known = measured_layers(base)
        size = (
            f"the anchor sensitivity at layer {layer} of {base!r} was not measured"
            + (
                f"; it was measured at layers {known} of that model, and a value is not carried "
                "across from a neighbouring layer"
                if known
                else ", and no layer of that model was, so there is no figure to reason from"
            )
        )
    else:
        sampled = sensitivity["sampled_displacement"]
        f32, native = sampled["float32"], sampled["native"]
        size = (
            f"at layer {layer}, displacing the anchor moved the sampled scalar derivatives by a "
            f"median {f32['median']} of their own size in float32 (native {native['median']}), "
            f"over {f32['count']} projections in one draw, worst {f32['max']}. That is a sampled "
            "directional sensitivity, not a condition number, and a median does not settle a "
            "layer"
        )
    raise ValueError(
        f"refusing a cross-path reading at layer {layer}: "
        + "; ".join(crossings)
        + f". {size}. {term['unmeasured_because']}. Until this pair is measured the difference "
        "is not an annotation on the reading but potentially the whole of it (WS-D, the float32 "
        "displacement control). Measure it: " + term["to_measure"]
    )


def hook_alignment(
    dictionary: JumpReLUDictionary,
    lens: Any,
    *,
    base: str | None = None,
    lens_fit_dtype: str | None = None,
    nu: dict | None = None,
    capture_dtype: str | None = None,
    capture_batch: int | None = None,
    reading: dict[str, Any] | None = None,
) -> int:
    """The lens layer this dictionary reads, refusing a lens that is not of the same model.

    Three things are compared and any disagreement refuses: the dictionary's ``model_name``,
    the lens identity's base, and, when the bridge runs from a registry entry, that entry's
    ``base``. Layer and hidden size are checked too, but they are the checks a dictionary
    trained on a different checkpoint of the same architecture passes.

    ``lens_fit_dtype`` is the registry entry's ``probes.lens_fit_dtype`` and ``nu`` the lens's
    own declaration; together they add the fourth comparison, in
    :func:`fit_precision_record`, which is what the width rows made necessary: two maps of one
    model in two arithmetics are two different maps. Callers that name neither get the check
    they had before, which is why the artefacts already recorded here did not change.

    ``capture_dtype`` and ``capture_batch`` describe the capture the reading will be taken on,
    and naming them adds the fifth: a capture from a path the lens was not fitted on is refused
    at any layer whose pairing is unmeasured. They default to ``None``, meaning *no capture is
    read*, which is the truth for a dictionary-direction readout and is why A1 is untouched by
    this; a caller that reads a residual passes that residual's own dtype and width.
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
    # Last, and with the layer in hand: the fit precision and, where a capture is named, whether
    # this reading crosses paths at a layer whose pairing nobody has measured.
    fit_precision_record(
        nu,
        declared=lens_fit_dtype,
        layer=layer,
        base=dict_base,
        capture_dtype=capture_dtype,
        capture_batch=capture_batch,
        reading=reading,
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
    fit_precision: dict[str, Any] | None = None,
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
            "fit_precision": fit_precision,
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
