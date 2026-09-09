"""Fit a Jacobian lens by **calling** upstream's exact-autograd estimator, not reimplementing it.

Why this module exists at all. Until now this repository estimated ``J_l`` by finite differences:
push a unit direction through the residual stream at one position, measure the response, divide by
epsilon. Upstream (``neuronpedia/jacobian-lens``, Apache-2.0, commit 581d398) computes the same
object by reverse-mode autograd, exactly, with no epsilon. Those are two different instruments and
nobody has ever measured the difference between them on the same model and the same corpus. The
WS-D golden test is that measurement, and it only means something if our side of the comparison is
upstream's own code rather than a re-derivation of it that might agree or disagree for reasons of
its own. So this module owns the *adapter*: it feeds our corpus rows to
``jlens.fitting.jacobian_for_prompt`` and turns what comes back into our artefact. The estimator
itself is imported, never copied, and the clone is read-only reference — nothing here vendors it.

Two conventions meet in this file and they differ by one everywhere:

* **upstream** indexes decoder *blocks*, zero-based. ``J[i]`` transports the output of block ``i``
  into the target basis, and the target defaults to block ``n_layers - 1`` — the last block's
  output **before** the final norm (``unembed`` supplies the norm, so the pairing is consistent,
  but it is not HF's ``hidden_states[-1]``).
* **this repository** indexes residual *layers*, one-based. Repo layer ``L`` is the output of block
  ``L - 1`` and is stored under npz key ``J{L-1}``; repo layer ``num_layers`` is the identity by
  construction and is never stored.

An off-by-one here is the "precise measurement of the wrong artefact" failure: adjacent hosted maps
are 0.72-0.98 alike once the identity is removed, so geometry cannot detect a one-layer shift.
:func:`assert_orientation_matches_upstream` therefore checks the written artefact against upstream's
own :meth:`JacobianLens.transport` numerically, at every layer, rather than trusting the comment
above. That check is what a transposed or layer-shifted artefact fails.

Declared ν (order §6.1). A lens is not "the Jacobian of a model"; it is the Jacobian under a
particular endpoint, position weighting, pair weighting, corpus and precision, computed by a
particular estimator. Two lenses that differ on any of those are not comparable, and yesterday's
hosted-versus-fitted comparison was made without knowing they differed on position composition.
Every artefact this module writes carries a ``nu`` block naming all six, and :func:`read_declared_nu`
**refuses** an artefact that has none — because the whole point of a declaration is that its absence
must not read like a canonical default.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from local_llm_lab.pipeline.lens_fitting.artifacts import write_lens
from local_llm_lab.pipeline.live_lens.instruments import (
    LensIdentity,
    LensMaps,
    file_sha256,
)
from local_llm_lab.project import PROJECT_ROOT

__all__ = [
    "ESTIMATOR_EXACT_AUTOGRAD",
    "ESTIMATOR_FINITE_DIFFERENCE",
    "CorpusLensModel",
    "CotangentSelectionError",
    "MissingDeclaredNu",
    "UpstreamJacobianFit",
    "UpstreamUnavailable",
    "assert_orientation_matches_upstream",
    "declare_nu",
    "default_position_selector",
    "fit_upstream_jacobian",
    "load_upstream",
    "read_declared_nu",
    "repo_layer_of_upstream",
    "upstream_index_of_repo_layer",
    "write_upstream_lens",
]

#: The two estimators this repository has used, named so an artefact says which one made it.
#: They are different instruments, not two implementations of one, and the difference between
#: them is exactly what the WS-D golden test measures for the first time.
ESTIMATOR_EXACT_AUTOGRAD = "upstream-exact-autograd"
ESTIMATOR_FINITE_DIFFERENCE = "finite-difference"

#: Environment variable naming the read-only upstream clone, for boxes that keep it elsewhere.
JLENS_PATH_ENV = "JLENS_PATH"

#: Where the clone lives by convention on this box and in the migration plan. Tried only when
#: neither an argument nor ``$JLENS_PATH`` names one; never vendored, never edited.
DEFAULT_JLENS_PATHS = (
    PROJECT_ROOT / "reference" / "jacobian-lens",
    Path.home() / "reference" / "jacobian-lens",
)

#: The commit the interface reports were written against. Recorded in ν, not enforced: a different
#: commit is a fact to declare, not a reason to refuse, but a comparison across two commits that
#: does not know it crossed them is worthless.
EXPECTED_JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


class UpstreamUnavailable(RuntimeError):
    """The read-only upstream clone is not where we were told to find it."""


class CotangentSelectionError(Exception):
    """A position selector chose nothing, or chose something malformed, for some prompt.

    Deliberately **not** a :class:`ValueError`. Upstream's ``fit()`` catches ``ValueError`` from
    ``jacobian_for_prompt`` and treats it as "skip this prompt", advancing its cursor and logging a
    warning. A selector that silently empties on some rows would shrink the prompt count through
    that path and the fit would still return a lens — a missing figure reading as a passing one.
    This exception is shaped so that it cannot be swallowed by that ``except``.
    """


class MixedPrecisionModel(ValueError):
    """The fitting model has more than one dtype or device across its blocks.

    Refused rather than described, because the sidecar declares one precision and the backward
    accumulates in the target block's, so a mixed model has no single true answer to put there.
    Reachable on the CUDA leg through `device_map="auto"`, bitsandbytes quantisation and
    `_keep_in_fp32_modules`, none of which are exotic.
    """


class MissingDeclaredNu(ValueError):
    """A lens artefact carries no declared ν, or one bound to a different file."""


# ----------------------------------------------------------------------- the upstream clone


@dataclass(frozen=True)
class Upstream:
    """The imported upstream modules plus where they came from, so ν can record it."""

    fitting: ModuleType
    lens: ModuleType
    path: Path
    commit: str | None

    @property
    def valid_position_mask(self) -> Callable[..., Any]:
        return self.fitting.valid_position_mask

    @property
    def skip_first_default(self) -> int:
        return int(self.fitting.SKIP_FIRST_N_POSITIONS)

    def provenance(self) -> dict:
        return {
            "repository": "neuronpedia/jacobian-lens",
            "licence": "Apache-2.0",
            "path": str(self.path),
            "commit": self.commit,
            "expected_commit": EXPECTED_JLENS_COMMIT,
            "commit_matches_expected": None
            if self.commit is None
            else self.commit == EXPECTED_JLENS_COMMIT,
            "vendored": False,
        }


def _clone_commit(path: Path) -> str | None:
    """The clone's HEAD, or ``None`` when it cannot be read — never a guess, never a default."""
    head = path / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
    except OSError:
        return None
    if ref.startswith("ref: "):
        try:
            return (path / ".git" / ref[5:]).read_text().strip()
        except OSError:
            return None
    return ref or None


def load_upstream(path: str | Path | None = None) -> Upstream:
    """Import ``jlens`` from the read-only reference clone.

    If ``path`` or ``$JLENS_PATH`` names a clone, **that one and only that one** is used — a named
    path that is wrong raises rather than falling back, because a fit run against a clone the caller
    did not choose is a measurement of an artefact nobody chose. With neither given, the
    :data:`DEFAULT_JLENS_PATHS` are tried and then an already-importable ``jlens`` (a pinned
    install). The clone goes on ``sys.path``; it is never copied into this package, because a
    vendored estimator would make the golden test a comparison of our transcription against itself.

    Raises:
        UpstreamUnavailable: naming every path tried, so the fix is obvious from the message.
    """
    named = path if path is not None else os.environ.get(JLENS_PATH_ENV)
    # An explicit path that is wrong must fail, not be silently replaced by a default: a fit run
    # against a clone the caller did not name is a measurement of an artefact nobody chose.
    candidates = [Path(named).expanduser()] if named else list(DEFAULT_JLENS_PATHS)
    tried: list[str] = []

    for candidate in candidates:
        tried.append(str(candidate))
        if not (candidate / "jlens" / "fitting.py").is_file():
            continue
        root = str(candidate.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        import jlens.fitting as fitting
        import jlens.lens as lens

        found = Path(fitting.__file__).resolve().parents[1]
        if found != candidate.resolve():
            # A different jlens was already imported into this interpreter. Say so rather than
            # reporting the path we wanted as the path we used.
            raise UpstreamUnavailable(
                f"jlens is already imported from {found}, but {candidate} was requested. "
                "Start a fresh interpreter, or point every caller at one clone."
            )
        return Upstream(fitting, lens, candidate.resolve(), _clone_commit(candidate))

    if not named:
        # No clone on this box, but an installed or already-imported jlens is a legitimate pin.
        try:
            import jlens.fitting as fitting
            import jlens.lens as lens
        except ImportError:
            pass
        else:
            found = Path(fitting.__file__).resolve().parents[1]
            return Upstream(fitting, lens, found, _clone_commit(found))

    raise UpstreamUnavailable(
        "the read-only jacobian-lens clone was not found. Upstream is called, never vendored, so "
        "there is no in-tree copy to fall back on. Tried, in order: "
        + ", ".join(tried)
        + f". Set ${JLENS_PATH_ENV} or pass path= to load_upstream()."
    )


# ------------------------------------------------------------------- layer index conventions


def repo_layer_of_upstream(index: int) -> int:
    """Upstream block index -> this repository's one-based residual layer."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"upstream layer index must be a non-negative int, got {index!r}")
    return index + 1


def upstream_index_of_repo_layer(layer: int) -> int:
    """This repository's one-based residual layer -> upstream block index."""
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 1:
        raise ValueError(f"repo layer must be an int >= 1, got {layer!r}")
    return layer - 1


# --------------------------------------------------------------------- the cotangent selector

#: A selector maps a sequence length to a boolean mask over positions. Upstream ties the source
#: role (the positions whose gradient is averaged) and the target role (the positions the one-hot
#: cotangent is written at) to a single mask; so does this seam, deliberately. Decoupling them is
#: a real semantic fork -- on a toy decoder the two choices differ by 8e-2 -- and it belongs to the
#: Chief, not to a default. §6.2 bands land here without touching the estimator.
PositionSelector = Callable[[int], Any]


def default_position_selector(
    *, skip_first: int | None = None, upstream: Upstream | None = None
) -> PositionSelector:
    """Upstream's own ``valid_position_mask``, bound to a ``skip_first``.

    Returned as a closure over the upstream function object rather than a reimplementation, so
    "the default selector is upstream's behaviour" is true by construction and not by a test that
    happens to agree today. The function object is captured **now**, not looked up per call: while
    an explicit selector is installed, ``jlens.fitting.valid_position_mask`` is this adapter's
    trampoline, and a late lookup would make the default selector call itself forever.
    """
    up = upstream or load_upstream()
    mask_fn = up.fitting.valid_position_mask
    skip = up.skip_first_default if skip_first is None else int(skip_first)
    return lambda seq_len: mask_fn(seq_len, skip_first=skip)


def _check_mask(mask: Any, seq_len: int, torch_module: ModuleType) -> Any:
    if not torch_module.is_tensor(mask) or mask.dtype != torch_module.bool:
        raise CotangentSelectionError(
            f"position selector must return a bool tensor, got {type(mask).__name__} "
            f"{getattr(mask, 'dtype', '')}"
        )
    if tuple(mask.shape) != (seq_len,):
        raise CotangentSelectionError(
            f"position selector returned shape {tuple(mask.shape)} for seq_len={seq_len}"
        )
    if not bool(mask.any()):
        raise CotangentSelectionError(
            f"position selector chose no positions at seq_len={seq_len}. An empty selection is "
            "refused rather than producing a zero J, which would read as a fitted map."
        )
    return mask


def _runs(chosen: list[int]) -> list[list[int]]:
    """``[start, stop)`` pairs covering exactly ``chosen``, so the fingerprint is lossless.

    A contiguous band is one pair, which is why this costs almost nothing in the ordinary case and
    still distinguishes two selectors that pick the same count over the same span.
    """
    runs: list[list[int]] = []
    for position in chosen:
        if runs and position == runs[-1][1]:
            runs[-1][1] = position + 1
        else:
            runs.append([position, position + 1])
    return runs


def _is_short_prompt(error: Exception) -> bool:
    """Whether a ValueError from upstream is its short-prompt refusal rather than a real defect.

    Matched on the message because upstream raises a bare ``ValueError`` for both this and a bad
    layer index, and only this one describes a row we may legitimately skip. Brittle to upstream's
    wording by construction -- and upstream is one unmaintained commit, so the wording will not
    move. If it ever does, this stops skipping and starts raising, which is the safe direction.
    """
    return "prompt too short" in str(error)


def selector_descriptor(selector: PositionSelector, *, probe_lengths: Sequence[int]) -> dict:
    """A serialisable fingerprint of a selector, for ν and for checkpoint compatibility checks.

    A tensor-valued knob cannot go where a scalar one went: upstream's ``fit()`` resume check does
    ``if state[key] != expected``, which on a mask raises *Boolean value of Tensor with more than
    one value is ambiguous*. So a selector is described by what it selects at a few probe lengths,
    hashed, never by the mask itself.
    """
    import torch

    rows = []
    for seq_len in probe_lengths:
        try:
            mask = _check_mask(selector(int(seq_len)), int(seq_len), torch)
        # Narrowed on the Chief's reading: catching ValueError by class absorbed *any* ValueError
        # from inside upstream as a counted skip, including ones that mean the fit is wrong rather
        # than the row is short. Upstream raises it from exactly two places, and only the
        # short-prompt one is a legitimate skip; `_check_layer_indices` is pre-validated above, so
        # anything else reaching here is a defect and must not read as a skipped row.
        except (CotangentSelectionError, ValueError) as error:
            if not isinstance(error, CotangentSelectionError) and not _is_short_prompt(error):
                raise
            rows.append({"seq_len": int(seq_len), "selected": None, "refused": str(error)})
            continue
        chosen = mask.nonzero(as_tuple=True)[0].tolist()
        rows.append(
            {
                "seq_len": int(seq_len),
                "selected": len(chosen),
                "first": int(chosen[0]),
                "last": int(chosen[-1]),
                "contiguous": chosen == list(range(chosen[0], chosen[-1] + 1)),
                # The exact selection, not a summary of it. An adversarial pass built two selectors
                # differing only in which interior positions they chose -- {20..28, 38} against
                # {20..26, 29, 30, 38} -- and the four summary numbers were identical, so the whole
                # position_weighting block and its sha256 collided while J differed by 17.5% of
                # scale. A fingerprint that cannot separate two incomparable lenses is not a
                # fingerprint; it is the appearance of one. Stored as a run-length encoding because
                # the common case is contiguous and the cost is then three integers.
                "runs": _runs(chosen),
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "probe": rows,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


# ------------------------------------------------------------------------- the model adapter


class CorpusLensModel:
    """Wrap a ``jlens.protocol.LensModel`` so upstream can fit on our corpus rows.

    Our corpus is token ids, verified and frozen in a manifest; upstream's entry point takes a
    ``prompt: str`` and tokenizes it itself. Re-tokenizing text we already froze would put a second,
    unverified tokenization inside the estimator, so instead the adapter registers each row's ids
    under a key and this wrapper's ``encode`` returns them. Every other member of the protocol is
    the inner model's, untouched.

    ``encode_calls`` is not decoration. It is how the caller proves upstream actually went through
    this path for this prompt; if a future upstream stopped calling ``encode``, the fit would
    silently be on something else.
    """

    def __init__(self, inner: Any):
        self.inner = inner
        self._rows: dict[str, Any] = {}
        self.encode_calls = 0

    # -- the LensModel protocol, delegated ------------------------------------------------
    @property
    def n_layers(self) -> int:
        return self.inner.n_layers

    @property
    def d_model(self) -> int:
        return self.inner.d_model

    @property
    def layers(self) -> Sequence[Any]:
        return self.inner.layers

    @property
    def tokenizer(self) -> Any:
        return getattr(self.inner, "tokenizer", None)

    def forward(self, input_ids: Any) -> Any:
        return self.inner.forward(input_ids)

    def unembed(self, residual: Any) -> Any:
        return self.inner.unembed(residual)

    # -- the corpus seam ------------------------------------------------------------------
    def register(self, key: str, ids: Sequence[int]) -> str:
        import torch

        device = getattr(self.inner, "input_device", None)
        tensor = torch.as_tensor(list(ids), dtype=torch.int64).reshape(1, -1)
        self._rows[key] = tensor if device is None else tensor.to(device)
        return key

    def encode(self, text: str, *, max_length: int | None = None) -> Any:
        """Return the frozen ids for a registered row, truncated exactly as upstream would.

        ``jacobian_for_prompt`` does ``input_ids.expand(dim_batch, -1)``, which requires shape
        ``[1, seq_len]``. One row at a time, never a padded batch: upstream's HF forward passes no
        attention mask, so anything padded would be wrong the moment it were introduced.

        ``max_length`` has **no default here**, unlike ``HFLensModel.encode``, which defaults to 512
        while ``jacobian_for_prompt`` passes its own 128 down into it. That mismatch is the trap the
        order names: fit at 128, read out at 512, no warning from either signature. This adapter
        never relies on either default, at either end, so an omitted length is an error rather than
        a silent 512.
        """
        if max_length is None:
            raise ValueError(
                "CorpusLensModel.encode requires an explicit max_length. Upstream's HFLensModel "
                "defaults it to 512 while jacobian_for_prompt passes 128, and a lens read four "
                "times outside its fitted length raises no error anywhere."
            )
        if text not in self._rows:
            raise KeyError(
                f"no corpus row registered under {text!r}; CorpusLensModel.encode never "
                "tokenizes text, because the corpus manifest already froze the ids."
            )
        self.encode_calls += 1
        return self._rows[text][:, :max_length]


def _observed_precision(model: Any) -> dict:
    """Read the device and dtype off the model rather than believing the caller's declaration.

    A number in a sidecar must be a measurement of the artefact it sits beside. ``cotangent =
    torch.zeros_like(target_activation)`` inherits the *model's* dtype, so on a bf16 model the whole
    backward accumulates in bf16 and only the final per-position mean is cast to float32 -- and the
    returned J is float32 CPU either way. The dtype of the artefact therefore tells you nothing
    about the precision it was computed in, which is why it is measured here and declared.
    """
    # Every block, not the first parameter of the first one. An adversarial pass found the earlier
    # version accepting `dtype="float32"` for a model whose blocks 1..N were bf16 -- and the
    # docstring above says why that is the wrong sample: the cotangent inherits the *target*
    # block's dtype, so a one-sample read of block 0 measures the one block that does not set the
    # accumulation precision. Same hole on device, which is the `device_map="auto"` shape: blocks
    # spread over cpu and mps declared "cpu". A real, precise measurement of the wrong artefact,
    # inside the function written to stop exactly that.
    dtypes: dict[str, int] = {}
    devices: dict[str, int] = {}
    grads: set[bool] = set()
    for block in model.layers:
        for parameter in block.parameters():
            dtypes[str(parameter.dtype).removeprefix("torch.")] = 1
            devices[parameter.device.type] = 1
            grads.add(bool(parameter.requires_grad))
    if not dtypes:
        raise ValueError("model.layers has no parameters; cannot observe device or dtype")
    if len(dtypes) > 1 or len(devices) > 1:
        raise MixedPrecisionModel(
            "the fitting model is not uniform, so no single declared precision describes the run: "
            f"dtypes {sorted(dtypes)}, devices {sorted(devices)}. The backward accumulates in the "
            "target block's dtype, so a mixed model produces a J whose precision the sidecar cannot "
            "state. Cast the model, or fit each uniform shard and merge."
        )
    return {
        "device": next(iter(devices)),
        "dtype": next(iter(dtypes)),
        "dtypes_observed": sorted(dtypes),
        "devices_observed": sorted(devices),
        "blocks_measured": len(list(model.layers)),
        "requires_grad": bool(next(iter(grads))) if len(grads) == 1 else None,
        "attn_implementation": _observed_attention(model),
    }


def _observed_attention(model: Any) -> str | None:
    """Which attention kernel the fitting model is using, or ``None`` when it cannot be read.

    ``attn_implementation`` appears nowhere in ``jlens``, so the fitting model inherits
    transformers' default, and the order requires ``eager`` or batched rows regress to sequential.
    Recorded, not enforced: on a hand-rolled model there is no config to read, and ``None`` is the
    honest value for "not determined" — it must not be written as ``"eager"`` merely because that is
    what we hoped for.
    """
    for candidate in (model, getattr(model, "inner", None)):
        config = getattr(getattr(candidate, "hf_model", candidate), "config", None)
        chosen = getattr(config, "_attn_implementation", None) if config is not None else None
        if isinstance(chosen, str):
            return chosen
    return None


def _require_frozen(model: Any) -> None:
    """Upstream's recorder calls ``requires_grad_(True)`` on a block output to root the graph.

    That only works if the output is a leaf, which it only is if nothing upstream of it requires
    grad. ``HFLensModel`` freezes every parameter for exactly this reason; a hand-rolled LensModel
    that skips the freeze fails deep inside the recorder with a message about leaf variables. Say
    it here instead.
    """
    for index, block in enumerate(model.layers):
        for name, parameter in block.named_parameters():
            if parameter.requires_grad:
                raise ValueError(
                    f"layers[{index}].{name} requires grad. Freeze the model "
                    "(`for p in model.parameters(): p.requires_grad_(False)`) before fitting: "
                    "upstream roots the autograd graph by making a block output a leaf, which "
                    "fails if any parameter behind it requires grad."
                )


# ------------------------------------------------------------------------------- the fit


@dataclass(frozen=True)
class UpstreamJacobianFit:
    """What one adapter run produced, in **upstream** (zero-based block) indexing.

    ``jacobians`` are float32 CPU arrays keyed by upstream source index. Conversion to repo layers
    happens once, in :func:`write_upstream_lens`, so there is exactly one place to get it wrong.
    """

    jacobians: dict[int, np.ndarray]
    target_layer: int
    n_prompts: int
    d_model: int
    per_prompt: list[dict]
    skipped: list[dict]
    elapsed_s: float
    provenance: dict
    precision: dict
    selector: dict


def fit_upstream_jacobian(
    model: Any,
    rows: Iterable[dict],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    position_selector: PositionSelector | None = None,
    skip_first: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int,
    device: str = "cpu",
    dtype: str = "float32",
    split: str | None = "fit",
    max_rows: int | None = None,
    upstream: Upstream | None = None,
    progress: Callable[[dict], None] | None = None,
) -> UpstreamJacobianFit:
    """Accumulate upstream's per-prompt Jacobian over our corpus rows.

    The estimator is upstream's, called once per row: a one-hot cotangent at every selected target
    position at once, so the gradient at source position ``p`` is the **sum** over target positions
    ``p' >= p``, then the **mean** over selected source positions. It is not normalised by the
    number of targets, so ``J`` scales with the count of selected positions and hence with
    ``max_seq_len``: a fit at 128 tokens and a fit at 512 are not comparable, and the count goes in
    ν for that reason.

    Prompts are weighted **equally**, matching upstream's ``fit()``, regardless of how many valid
    positions each contributed. That makes the corpus's length distribution part of the estimator,
    which is also declared.

    ``max_seq_len`` is **required**, with no default. Upstream fits at 128 by default and reads out
    at 512 by default (``HFLensModel.encode`` and ``JacobianLens.apply`` both default to 512, while
    ``jacobian_for_prompt`` passes its own 128 down into ``encode``), and neither signature warns
    that the two differ. The order's rule is that this adapter passes lengths explicitly at both
    ends and relies on neither default; a required argument is how that rule is enforced rather than
    remembered. 128 is the reference estimator's definition and the length the hosted lenses were
    fitted at, so it is what the golden test passes — deliberately, in the caller, in writing.

    Args:
        model: anything satisfying upstream's ``LensModel``. Wrapped in :class:`CorpusLensModel`
            unless it already is one.
        rows: corpus rows (``read_corpus`` output); only ``ids``, ``split`` and ``index`` are read.
        source_layers: upstream block indices. ``None`` means every block below the target, which
            is upstream's default and the one the hosted lenses were fitted with.
        target_layer: upstream block index. ``None`` means ``n_layers - 1``, the last block's
            output **before** the final norm.
        position_selector: the cotangent selector seam. ``None`` runs upstream's own
            ``valid_position_mask`` through upstream's own code path, unpatched.
        skip_first: forwarded to upstream's mask when no selector is given.
        device, dtype: the precision this fit *claims*; both are checked against the model and a
            disagreement is refused. Defaults are CPU float32 because no CUDA device exists yet
            and local-first development must run before a GPU does.
        split: keep only rows in this corpus split; ``None`` keeps all.
        max_rows: stop after this many usable rows. Fewer prompts is the CPU budget, not a default.

    Raises:
        CotangentSelectionError: a selector chose nothing for some row.
        ValueError: no row was long enough, or the declared precision is not the observed one.
    """
    import torch

    up = upstream or load_upstream()
    started = time.monotonic()

    wrapped = model if isinstance(model, CorpusLensModel) else CorpusLensModel(model)
    _require_frozen(wrapped)
    observed = _observed_precision(wrapped)
    if observed["device"] != device or observed["dtype"] != dtype:
        raise ValueError(
            f"this fit declares {dtype} on {device} and the model is "
            f"{observed['dtype']} on {observed['device']}. The declaration must be a measurement "
            "of what actually ran, not an intention."
        )

    n_layers = int(wrapped.n_layers)
    resolved_target = n_layers - 1 if target_layer is None else int(target_layer)
    if resolved_target < 0:
        resolved_target += n_layers
    sources = (
        list(range(resolved_target))
        if source_layers is None
        else sorted({int(s) for s in source_layers})
    )
    if not sources or sources[0] < 0 or sources[-1] >= resolved_target:
        raise ValueError(
            f"source layers {sources} must be nonempty and strictly below target {resolved_target}"
        )

    skip = up.skip_first_default if skip_first is None else int(skip_first)
    selector = position_selector
    # The seam: with no selector we do not patch anything, so the default path is literally
    # upstream's function called by upstream's code. With one, we substitute it for the duration
    # of the call -- upstream reads the mask through this single module attribute, and both the
    # target scatter and the source mean take it, tied, exactly as upstream ties them.
    original_mask_fn = up.fitting.valid_position_mask

    d_model = int(wrapped.d_model)
    running = {layer: np.zeros((d_model, d_model), dtype=np.float64) for layer in sources}
    per_prompt: list[dict] = []
    skipped: list[dict] = []
    n_done = 0

    def _patched(seq_len: int, *, skip_first: int = 0) -> Any:
        # `skip_first` is upstream's keyword and is accepted so the call site is unchanged; an
        # explicit selector owns the whole position rule, so it is deliberately ignored here.
        del skip_first
        return _check_mask(selector(int(seq_len)), int(seq_len), torch)

    if selector is not None:
        # Refuse a nested fit while the module attribute is already substituted. The patch is a
        # module global with exactly one call site, so a second concurrent fit would silently take
        # the first one's selector and produce a lens whose declared position rule is another run's.
        # The restore is in a `finally`, so this only fires on genuine nesting, never on a leaked
        # patch from a failed call.
        if getattr(original_mask_fn, "_wsd_patched", False):
            raise CotangentSelectionError(
                "a selector fit is already in progress on this process: "
                "jlens.fitting.valid_position_mask is already substituted. Nested selector fits "
                "would share one module global and the inner run's declared position rule would be "
                "the outer run's. Run them in sequence."
            )
        _patched._wsd_patched = True
        up.fitting.valid_position_mask = _patched
    try:
        for row in rows:
            if max_rows is not None and n_done >= max_rows:
                break
            if split is not None and row.get("split") != split:
                continue
            key = f"row:{row['index']}"
            wrapped.register(key, row["ids"])
            before = wrapped.encode_calls
            try:
                jacobians, seq_len, n_valid = up.fitting.jacobian_for_prompt(
                    wrapped,
                    key,
                    sources,
                    target_layer=resolved_target,
                    dim_batch=dim_batch,
                    max_seq_len=max_seq_len,
                    skip_first=skip,
                )
            except ValueError as error:
                # Upstream raises this for a prompt shorter than the mask needs. Counted and
                # reported, never absorbed into a smaller n that still returns a lens.
                skipped.append({"index": int(row["index"]), "reason": str(error)})
                continue
            if wrapped.encode_calls != before + 1:
                raise ValueError(
                    "upstream did not tokenize through CorpusLensModel.encode; the fit would be "
                    "on something other than the frozen corpus ids"
                )
            for layer in sources:
                running[layer] += jacobians[layer].numpy().astype(np.float64)
            n_done += 1
            record = {
                "index": int(row["index"]),
                "seq_len": int(seq_len),
                "n_valid_positions": int(n_valid),
                "split": row.get("split"),
                "domain": row.get("domain"),
            }
            per_prompt.append(record)
            if progress:
                progress({"event": "prompt", "n_done": n_done, **record})
    finally:
        up.fitting.valid_position_mask = original_mask_fn

    if n_done == 0:
        raise ValueError(
            "no corpus row produced a Jacobian: "
            + (f"{len(skipped)} rows were refused" if skipped else "no row matched the split")
        )

    maps = {layer: (running[layer] / n_done).astype(np.float32) for layer in sources}
    for layer, matrix in maps.items():
        if not np.isfinite(matrix).all():
            raise ValueError(f"non-finite Jacobian at upstream layer {layer}")

    probe_lengths = sorted({record["seq_len"] for record in per_prompt})
    described = selector_descriptor(
        selector
        if selector is not None
        else default_position_selector(skip_first=skip, upstream=up),
        probe_lengths=probe_lengths,
    )
    return UpstreamJacobianFit(
        jacobians=maps,
        target_layer=resolved_target,
        n_prompts=n_done,
        d_model=d_model,
        per_prompt=per_prompt,
        skipped=skipped,
        elapsed_s=time.monotonic() - started,
        provenance=up.provenance(),
        precision=observed
        | {
            "declared_device": device,
            "declared_dtype": dtype,
            "backward_accumulation_dtype": observed["dtype"],
            "position_mean_dtype": "float32",
            "prompt_accumulation_dtype": "float64",
            "stored_dtype": "float32",
        },
        selector=described
        | {
            "rule": "upstream valid_position_mask" if selector is None else "explicit selector",
            "skip_first": skip if selector is None else None,
            "upstream_default_path": selector is None,
            "source_and_target_tied": True,
            "max_seq_len": int(max_seq_len),
            "dim_batch": int(dim_batch),
        },
    )


# ------------------------------------------------------------------------------ declared ν


def declare_nu(
    fit: UpstreamJacobianFit,
    *,
    num_layers: int,
    corpus: dict,
    estimator: str = ESTIMATOR_EXACT_AUTOGRAD,
) -> dict:
    """The six fields §6.1 requires, each saying what it is a statement *about*.

    ``estimator`` is the field that separates this lens from every lens this repository fitted
    before: ``upstream-exact-autograd`` against ``finite-difference``. The golden test's residual is
    only the difference between those two if everything else here matches between the two artefacts
    -- endpoint, positions, pairs, corpus and precision -- which is why they are declared beside it
    rather than left to be inferred from a filename.
    """
    if estimator not in (ESTIMATOR_EXACT_AUTOGRAD, ESTIMATOR_FINITE_DIFFERENCE):
        raise ValueError(f"unknown estimator {estimator!r}")
    valid = [record["n_valid_positions"] for record in fit.per_prompt]
    lengths = [record["seq_len"] for record in fit.per_prompt]
    return {
        "schema_version": 1,
        "estimator": estimator,
        "endpoint": {
            "target_layer_upstream": fit.target_layer,
            "target_layer_repo": repo_layer_of_upstream(fit.target_layer),
            "source_layers_upstream": sorted(fit.jacobians),
            "source_layers_repo": [repo_layer_of_upstream(i) for i in sorted(fit.jacobians)],
            "target_is_pre_final_norm": True,
            "note": (
                "upstream's target is the block output before the final norm; unembed supplies "
                "the norm, so it is not HF hidden_states[-1]"
            ),
        },
        "position_weighting": {
            **fit.selector,
            "source_reduction": "mean over selected source positions",
            "target_reduction": "sum over selected target positions, not normalised",
            "scale_warning": (
                "J scales with the number of selected target positions, so fits at different "
                "max_seq_len or different band widths are not comparable in magnitude"
            ),
            "readout_warning": (
                "this lens was fitted at the max_seq_len recorded here; upstream's readout "
                "defaults (HFLensModel.encode and JacobianLens.apply, both 512) are larger than "
                "its fit default (128) and neither warns, so a readout beyond the fitted length "
                "is extrapolation and must be passed explicitly"
            ),
            "n_valid_positions": {
                "min": min(valid) if valid else None,
                "max": max(valid) if valid else None,
                "total": int(sum(valid)) if valid else None,
            },
            "seq_len": {
                "min": min(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
            },
        },
        "pair_weighting": {
            "source_target_pairs": "causal, p' >= p, within the selected set",
            "per_prompt": "equal weight per prompt regardless of n_valid_positions",
            "n_prompts": fit.n_prompts,
            "n_skipped": len(fit.skipped),
            "skipped": fit.skipped,
            "note": (
                "equal per-prompt weighting makes the corpus length distribution part of the "
                "estimator; upstream's fit() weights the same way"
            ),
        },
        "corpus": dict(corpus),
        "precision": dict(fit.precision),
        "upstream": dict(fit.provenance),
        "decoder_depth": int(num_layers),
    }


def read_declared_nu(path: Path) -> dict:
    """Return the ν beside a lens, refusing an artefact that declares none.

    Honest about its own strength. ``LensMaps.load`` reads the in-archive ``identity`` and never
    opens the sidecar for a lens written by ``write_lens``, so nothing in the loader checks this
    block. What is checked here is the same binding the hosted identity route uses: the sidecar's
    ``npz_sha256`` must be the digest of the npz beside it, so a ν describing a different file is
    not evidence about this one. That is a real check with a real failure mode; it is not proof
    against someone rewriting both files together, and this docstring is where that limit is
    recorded rather than discovered later.
    """
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        raise MissingDeclaredNu(
            f"{path.name} has no sidecar at {sidecar.name}, so it declares no ν"
        )
    meta = json.loads(sidecar.read_text())
    if not isinstance(meta, dict) or "nu" not in meta:
        raise MissingDeclaredNu(
            f"{sidecar.name} carries no 'nu' block. A lens whose ν is undeclared is refused, not "
            "assumed to have upstream's defaults: an absent declaration must not read like a "
            "canonical one."
        )
    recorded = meta.get("npz_sha256")
    actual = file_sha256(path)
    if recorded != actual:
        raise MissingDeclaredNu(
            f"{sidecar.name} records npz_sha256 {recorded} and {path.name} hashes to {actual}; "
            "the ν describes a different file."
        )
    nu = meta["nu"]
    if not isinstance(nu, dict) or not {
        "estimator",
        "endpoint",
        "position_weighting",
        "pair_weighting",
        "corpus",
        "precision",
    } <= set(nu):
        raise MissingDeclaredNu(f"{sidecar.name} 'nu' block is missing required fields")
    return nu


# ------------------------------------------------------------------------- the artefact


def assert_orientation_matches_upstream(
    maps: LensMaps | dict[int, np.ndarray],
    fit: UpstreamJacobianFit,
    *,
    upstream: Upstream | None = None,
    seed: int = 0,
    atol: float = 1e-4,
) -> dict:
    """Check that maps keyed by **repo layer** transport like upstream's lens, at every layer.

    Both sides are ``residual @ J.T`` with ``J`` in ``[output, input]`` orientation, and the
    converter that produced the hosted lenses adds no transpose. That is a claim about two lines of
    code in two repositories, and a claim is not a check: this runs a random probe through
    upstream's own :meth:`JacobianLens.transport` at block ``L-1`` and through our transport at repo
    layer ``L``, and compares. A transposed artefact fails it because ``J`` is not symmetric; a
    layer-shifted artefact fails it because layer ``L`` would carry a different block's map.

    Note exactly what this is a check **of**, because a check whose scope is misread is worse than
    none. It checks *an artefact against the fit it claims to come from*. ``write_lens`` already
    round-trips its own archive, but that check is symmetric — the writer's ``J{L-1}`` and the
    loader's ``L = i + 1`` are inverses, so they agree even if both are wrong. This one is
    asymmetric: it goes through upstream's transport, the convention we are converting *from*, so it
    catches a transpose or a shift introduced anywhere between the fit and the loaded file. It
    cannot say anything about whether the fit itself is right — that is the golden test's job, not
    this one's. Run it on a **loaded** :class:`LensMaps`: run on the same dict the caller is about
    to write, it compares a value with itself and cannot fail.

    Args:
        maps: a loaded :class:`LensMaps` (the useful case), or a plain ``{repo_layer: array}`` dict.

    Returns:
        Per-layer max absolute deviation, so a record can say what the check measured rather than
        only that it passed.
    """
    import torch

    up = upstream or load_upstream()
    if isinstance(maps, LensMaps):
        available = set(maps.maps)

        def transport(probe: np.ndarray, layer: int) -> np.ndarray:
            return maps.apply(probe, layer)
    else:
        available = set(maps)

        def transport(probe: np.ndarray, layer: int) -> np.ndarray:
            return probe @ np.asarray(maps[layer]).T

    reference = up.lens.JacobianLens(
        {
            layer: torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
            for layer, matrix in fit.jacobians.items()
        },
        n_prompts=fit.n_prompts,
        d_model=fit.d_model,
    )
    generator = torch.Generator().manual_seed(seed)
    probe = torch.randn(4, fit.d_model, generator=generator, dtype=torch.float32)
    deviations: dict[str, float] = {}
    for index in sorted(fit.jacobians):
        layer = repo_layer_of_upstream(index)
        if layer not in available:
            raise ValueError(f"artefact has no map at repo layer {layer} (upstream block {index})")
        theirs = reference.transport(probe, index).numpy()
        ours = transport(probe.numpy(), layer)
        deviation = float(np.max(np.abs(theirs - ours)))
        scale = float(np.max(np.abs(theirs))) or 1.0
        if not deviation <= atol * scale:
            raise ValueError(
                f"repo layer {layer} does not transport like upstream block {index}: max deviation "
                f"{deviation:.3e} against tolerance {atol * scale:.3e}. The artefact is in a "
                "different orientation or a different layer convention than J{L-1}."
            )
        deviations[str(layer)] = deviation
    return deviations


def write_upstream_lens(
    path: Path,
    fit: UpstreamJacobianFit,
    *,
    identity: LensIdentity,
    hidden_size: int,
    num_layers: int,
    corpus: dict,
    metadata: dict | None = None,
    estimator: str = ESTIMATOR_EXACT_AUTOGRAD,
    kind: str | None = None,
    upstream: Upstream | None = None,
) -> dict:
    """Convert upstream indices to repo layers, write the artefact, then verify what was written.

    **The artefact is in this repository's convention**: npz key ``J{L-1}`` holds the map for repo
    layer ``L``, which is the output of decoder block ``L-1`` — i.e. upstream's ``J[L-1]`` copied
    with no transpose and no reindexing beyond the ``+1``. ``write_lens`` demands all and only
    layers ``1..num_layers-1``, which is exactly upstream's default source set ``0..n_layers-2``
    shifted by one; layer ``num_layers`` is the identity by construction and is never stored.

    The orientation check runs **after** the write, on the file reloaded through ``LensMaps.load``,
    and that is not a convenience: run on the ``maps`` dict this function is about to hand to
    ``write_lens``, it would compare the line above with itself and could never fail. Run on the
    reloaded artefact it crosses three independent pieces — ``repo_layer_of_upstream`` here,
    ``write_lens``'s ``J{layer-1}`` key, and ``LensMaps.load``'s ``layer = i + 1`` — against
    upstream's own transport, and any one of them moving breaks it.

    The consequence, stated rather than hidden: the sidecar ``write_lens`` writes cannot carry the
    check's numbers, because the check is performed on the file the sidecar describes. The sidecar
    carries the convention as a *statement*; the executed deviations are in the returned dict, under
    ``orientation_check``, for the caller's run record. Returned keys are therefore a superset of
    the sidecar's, and this paragraph is where that is recorded.

    ``kind`` is ``hosted-jacobian`` only when the fit took upstream's own position rule — that is
    what ``profiles.py`` means by the label, and it is what makes a lens comparable with the hosted
    ones. A fit through the selector seam is a **declared departure** (§6.2 bands are a different
    estimator by construction and are never compared to the hosted lens as if they were one), so it
    is not labelled automatically: the caller must name its kind. Guessing here would let a band
    fit read as a hosted-recipe fit, which is the whole failure the ν block exists to prevent.

    Returns:
        The sidecar dict, plus ``orientation_check`` (per-layer max deviation, measured post-write).
    """
    up = upstream or load_upstream()
    expected_sources = set(range(num_layers - 1))
    if set(fit.jacobians) != expected_sources:
        raise ValueError(
            f"a complete artefact needs upstream blocks 0..{num_layers - 2}; this fit has "
            f"{sorted(fit.jacobians)}. Merge shards with JacobianLens.merge before writing -- "
            "write_lens refuses a partial lens, deliberately."
        )
    if fit.target_layer != num_layers - 1:
        raise ValueError(
            f"artefact layer {num_layers} is the identity only when the target is upstream block "
            f"{num_layers - 1}; this fit targeted {fit.target_layer}"
        )

    # The one place the convention is converted. Repo layer L <- upstream block L-1.
    maps = {repo_layer_of_upstream(index): matrix for index, matrix in fit.jacobians.items()}
    if set(maps) != set(range(1, num_layers)):
        raise ValueError("layer conversion must cover exactly repo layers 1..num_layers-1")

    if kind is None:
        if not fit.selector.get("upstream_default_path"):
            raise ValueError(
                "this fit used the cotangent selector, so it is a declared departure from "
                "upstream's ν and cannot be labelled 'hosted-jacobian' by default. Pass kind= "
                "explicitly: a band or span lens read as a hosted-recipe lens is exactly the "
                "confusion the ν block exists to prevent."
            )
        kind = "hosted-jacobian"

    nu = declare_nu(fit, num_layers=num_layers, corpus=corpus, estimator=estimator)
    payload = dict(metadata or {})
    payload |= {
        "schema_version": 1,
        "kind": kind,
        "nu": nu,
        # Repeated at the top level so a shallow reader deciding comparability does not have to
        # dig, and cannot read a lens of one estimator as a lens of the other.
        "estimator": nu["estimator"],
        "orientation": "output-by-input; prediction directions @ J.T",
        "upstream_layer_convention": "npz J{i} == upstream source block i == repo layer i+1",
        "elapsed_s": fit.elapsed_s,
        "n_prompts": fit.n_prompts,
        "n_skipped": len(fit.skipped),
        "per_prompt": fit.per_prompt,
    }
    written = write_lens(
        path,
        maps,
        hidden_size=hidden_size,
        num_layers=num_layers,
        metadata=payload,
        identity=identity,
    )
    loaded = LensMaps.load(
        Path(path),
        expected_sha256=written["npz_sha256"],
        hidden_size=hidden_size,
        num_layers=num_layers,
        identity=identity,
    )
    return written | {
        "orientation_check": assert_orientation_matches_upstream(loaded, fit, upstream=up)
    }
