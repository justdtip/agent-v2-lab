"""Jacobian-lens ("J-lens") interpretability probe.

Implements the readout described in Gurnee et al. 2026, "Verbalizable Representations Form a
Global Workspace in Language Models". The standard logit lens reads a mid-network residual
stream ``h`` by pretending it is already the final-layer residual: ``softmax(W_U * norm(h))``.
That is a poor approximation for early/mid layers because the network still has many more
layers of transformation to apply to ``h`` before it reaches the unembedding.

The J-lens instead approximates the *linearised* map the rest of the network applies. For a
layer ``L``, define

    J_L = E_corpus[ d h_final / d h_L ]

the Jacobian of the final residual stream with respect to the layer-``L`` residual stream,
evaluated at a token position and averaged over token positions and a corpus of contexts (the
network is highly non-linear, so this local linearisation is corpus- and position-dependent;
averaging over many contexts gives a single direction-independent summary of "what layer L's
residual space typically maps to"). The lens readout for an activation ``h`` is then

    lens(h) = softmax( W_U . norm( J_L . h ) )

i.e. push ``h`` through the *averaged local linearisation* of the remaining layers, apply the
model's final norm, and unembed. Sorting that distribution gives the tokens the model is
"prepared to verbalise" from that activation once its remaining transformation is accounted
for. The subspace spanned by these readout vectors is called J-space.

Efficiency (the reason this is tractable at all): we never materialise the full ``d x d``
Jacobian ``J_c`` for a context ``c``, let alone the average ``J_L``. We only ever need
``J_L . h`` for one probe vector ``h``, and because expectation and the (locally) linear map
commute,

    E_c[J_c] . h = E_c[J_c . h]

each ``J_c . h`` is exactly a forward-mode Jacobian-vector product (JVP) of the tail of the
network (from layer ``L`` to the final norm) evaluated at context ``c``, with tangent ``h``
placed at the probed token position and zero elsewhere. So one J-lens readout costs one JVP per
corpus context (``N`` forward-mode passes through the tail of the network), not one pass per
dimension of ``h`` as a naive finite-difference or reverse-mode Jacobian materialisation would.
MLX's ``mx.jvp(fn, primals, tangents)`` computes exactly this, in a single forward sweep, for
both arguments and both results as lists.

Numerics: the tail-network JVP is run and accumulated in float32. On the real (4-bit quantized)
checkpoint the tangent has been observed to overflow to infinity in float16, so float32
accumulation throughout (primal, tangent, and the corpus average) is mandatory, not a nicety.

``DEFAULT_CORPUS`` below is a small (24-snippet), hand-written stand-in for the pretraining
distribution the paper averages over. It is not a sample from any real pretraining corpus, so
J-lens estimates computed from it are approximate and corpus-dependent; treat differences
between the J-lens and the logit-lens baseline as suggestive, not as a calibrated probability
of "the model can verbalise this", and prefer changes in relative ranking over absolute
probabilities.

Attention masking: layers are invoked with the same causal mask the model builds itself, via
``create_attention_mask``. Calling transformer blocks directly bypasses
``Qwen2Model.__call__``, which is where that mask is normally constructed; passing ``None``
would give every multi-token prompt bidirectional self-attention, so the residual stream would
encode something the model never computes at inference and every readout taken from it would be
meaningless. ``residual_at`` composed with ``jacobian_vector_product``'s tail is verified to
reproduce the model's own forward pass exactly.

Conformance statement (R34; EXP-001 §3.2 and the Head of Interpretability's addendum of
2026-09-05). Every artifact this module writes carries a ``conformance`` block, and the
variant it names is the following.

- **Layer-index convention.** Layer ``L`` is the residual *after* block ``L - 1``; layer
  ``0`` is the embedding output and ``L = num_layers`` is the pre-final-norm residual
  (``capture.py:11-13``). A probe layer's *kind* is the kind of the block that **wrote**
  it, block ``L - 1``, not the block about to read it. Under the reader convention the
  kind-matched pairs of EXP-001 §3.5 would not be contrasts at all.
- **Layer family.** With ``--layers`` omitted the sweep is the registry fractions *plus*
  the kind-matched partners the model configuration's ``full_attention_interval`` implies
  (:func:`kind_matched_layer_family`), so EXP-001 §2's contrast between an attention layer
  and a linear-attention layer at comparable depth is guaranteed to be in the run rather
  than left to whether a fraction happened to land on an attention block. The artifact
  records the family, the period and where it was read from, the pairs, and each layer's
  ``primary`` (in-band), ``partner`` or ``reported`` role. An explicit ``--layers`` is
  honoured verbatim and every layer is marked ``explicit``.
- **Source positions.** Where the tangent is placed in each corpus context, given as
  fractions of the context (``--source-positions``, default
  ``0.25,0.5,0.75``) or as explicit indices. The mapping function's historical default
  was the last token, where the future window is empty.
- **Output positions read, and both reduction axes.** ``self`` reads the output tangent at
  the source position (the paper's self-only limiting case, the variant every J-lens number
  in this repository was drawn under before this module changed); ``future`` sums the output
  tangent over every position strictly after the source (the paper's broadcast component);
  ``all`` is their sum (the paper's default estimator up to a scale the sign test ignores)
  and is primary. Those sums are **within** one sample; **across** samples the per-sample
  vectors are **averaged**. Naming only one axis leaves "sum" ambiguous between the two, so
  R34 requires both (Head of Interpretability, issue #61). All three readouts come from one
  JVP per (context, source) sample. Two guards keep a structural zero from being reported as
  a measurement, and R34 names them apart because they answer different questions: a readout
  whose future *window* is empty -- the source is the context's last token -- raises
  :class:`EmptyFutureWindowError`, and a ``future``/``all`` readout at ``L == num_layers``,
  where no decoder *block* remains in the tail, raises :class:`NoTailBlocksError` (issue #68).
  Neither ever reads as a zero; a CLI excludes the final layer's two cells and records the
  exclusion in the conformance block.
- **Corpus size and context length.** The number of contexts averaged over and their token
  lengths; the median future window per context and per readout is recorded beside them.
- **JVP method.** ``forward`` or ``finite_difference``, with the source it was established
  from (``cli`` or ``preflight``). It is never defaulted silently: see
  :func:`resolve_jvp_method`.

Precision boundary (R18/R18a/R18b, briefing §1.5). Stored activations, tangents and the
tail JVP are float32 throughout; ``ArchitectureView.embed``, ``run_block`` and
``final_norm`` each cast their result to float32, so the cast boundary sits at the view's
own block execution and the capture this module takes is a float32 capture regardless of
the model's native dtype. ``residual_at`` therefore takes the registry's
``probes.capture_dtype`` and records what was requested against what the view could
deliver (:func:`resolve_capture_dtype`) rather than claiming a native capture it cannot
perform; the artifact copies the preflight's ``fp32_manual_vs_native`` deviation block so
the size of that difference is on the record, per R18a.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from local_llm_lab.arch import ArchitectureView

__all__ = [
    "CAPTURE_DTYPES",
    "DEFAULT_CORPUS",
    "DEFAULT_CORPUS_LENGTH",
    "DEFAULT_MODEL",
    "DEFAULT_SOURCE_POSITIONS",
    "EmptyFutureWindowError",
    "NoTailBlocksError",
    "FINAL_LAYER_EXCLUDED_READOUTS",
    "readouts_for_layer",
    "final_layer_exclusion",
    "HYBRID_PERIOD_FIELD",
    "IN_BAND_FRACTIONS",
    "JVP_METHODS",
    "JvpMethodUnresolved",
    "LAYER_FAMILY_RULE",
    "LayerFamily",
    "READOUTS",
    "SourcePosition",
    "build_corpus",
    "comparability_block",
    "conformance_block",
    "distribution",
    "encode",
    "hybrid_period",
    "in_band_layers",
    "jacobian_vector_product",
    "jlens_map",
    "jlens_readouts",
    "kind_matched_layer_family",
    "logit_lens",
    "main",
    "probe_layer_kind",
    "probe_layers",
    "readout",
    "render_probe_prompt",
    "residual_at",
    "resolve_capture_dtype",
    "resolve_jvp_method",
    "resolve_source_positions",
    "source_index",
    "token_evidence",
]

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"

#: The three readouts EXP-001 §3.2 (B3) requires, in report order; ``all`` is primary.
READOUTS: tuple[str, ...] = ("self", "future", "all")

#: The readouts that need at least one decoder block left in the tail to be a measurement.
#: At ``L == num_layers`` there is none, so both are structural zeros (issue #68).
FINAL_LAYER_EXCLUDED_READOUTS: tuple[str, ...] = ("future", "all")
PRIMARY_READOUT = "all"
JVP_METHODS: tuple[str, ...] = ("forward", "finite_difference")
CAPTURE_DTYPES: tuple[str, ...] = ("native", "float32")
#: Interior sources, as fractions of each corpus context (Head of Interpretability, A1).
DEFAULT_SOURCE_POSITIONS = "0.25,0.5,0.75"
#: Minimum tokens per corpus context; the paper uses 128-token sequences (§A.7).
DEFAULT_CORPUS_LENGTH = 128
#: The block kinds ``ArchitectureView.layer_kind`` reports, under the names it uses.
ATTENTION_KIND = "attention"
RECURRENT_KIND = "linear_attention"
#: A hybrid backbone's attention period, under the name its configuration gives it.
HYBRID_PERIOD_FIELD = "full_attention_interval"
#: Attribute names a text configuration is ever reached through, and how deep to follow them.
_CONFIG_ATTRIBUTES = ("args", "config", "text_config", "language_model", "model")
_CONFIG_SEARCH_DEPTH = 3
#: EXP-001 §2's decisive band, as fractions of decoder depth. Fraction 1/6 and the final
#: layer sit outside it: reported, not decisive.
IN_BAND_FRACTIONS: tuple[float, float] = (1 / 3, 5 / 6)
#: Stated in every artifact so a reader can see how the kind-matched pairs were formed (R34).
LAYER_FAMILY_RULE = (
    "layer L is the residual after block L-1, and on a hybrid with "
    f"{HYBRID_PERIOD_FIELD}=p that block is an attention block exactly when L % p == 0; for "
    "each in-band layer the kind-matched partner is the nearest such layer (smaller depth "
    "difference first, lower index on a tie) and joins the sweep when it is not already "
    "there; an in-band layer that is itself an attention output takes no partner"
)


class JvpMethodUnresolved(ValueError):
    """Neither a flag nor a preflight record established the derivative method (R18a)."""


class EmptyFutureWindowError(ValueError):
    """A ``future``/``all`` readout was asked for where no later position exists (A1)."""


class NoTailBlocksError(EmptyFutureWindowError):
    """A ``future``/``all`` readout was asked for where no decoder block remains (issue #68).

    The sibling condition to :class:`EmptyFutureWindowError`, and deliberately a *distinct*
    name: that one is about **positions in the context** -- the source sits at the last token,
    so there is nothing after it -- while this one is about **blocks in the tail**. At layer
    ``L == num_layers`` the tail is the final norm and the unembedding and nothing else. Both
    are position-wise, so no perturbation at the source can reach any later position and the
    future readout is identically zero for every context of every model.

    A subclass rather than a plain sibling so a caller that already handles the empty-window
    case keeps catching this one; the name and message are what tell the two apart.
    """


@dataclass(frozen=True)
class SourcePosition:
    """One tangent placement, kept as the token the operator wrote plus its parsed value."""

    token: str
    kind: Literal["index", "fraction"]
    value: float

    def as_dict(self) -> dict[str, Any]:
        return {"token": self.token, "kind": self.kind, "value": self.value}


def resolve_source_positions(raw: str | None) -> tuple[SourcePosition, ...]:
    """Parse ``--source-positions``: comma-separated fractions in (0, 1) or integer indices.

    A fraction is resolved against each context's own length by :func:`source_index`, so one
    flag covers corpus contexts of different lengths. An integer is an absolute index and may
    be negative, counting from the end as Python does.
    """
    text = DEFAULT_SOURCE_POSITIONS if raw is None else raw
    tokens = [token.strip() for token in text.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--source-positions must be comma-separated fractions or indices")
    parsed: list[SourcePosition] = []
    for token in tokens:
        try:
            index = int(token)
        except ValueError:
            try:
                fraction = float(token)
            except ValueError as error:
                raise ValueError(
                    "--source-positions must be comma-separated fractions or indices"
                ) from error
            if not 0.0 <= fraction < 1.0:
                raise ValueError(
                    "--source-positions fractions must lie in [0, 1)"
                ) from None
            parsed.append(SourcePosition(token=token, kind="fraction", value=fraction))
        else:
            parsed.append(SourcePosition(token=token, kind="index", value=float(index)))
    return tuple(parsed)


def source_index(source: SourcePosition, length: int) -> int:
    """Resolve one source against a context of ``length`` tokens; may fall outside it."""
    if source.kind == "fraction":
        return int(round(source.value * max(length - 1, 0)))
    index = int(source.value)
    return index if index >= 0 else length + index


def resolve_jvp_method(
    requested: str | None, spec: Any, *, output_root: Path | None = None
) -> tuple[str, str]:
    """The derivative method and where it came from: ``("forward"|"finite_difference", src)``.

    The flag wins when given; otherwise the method is the one the model's own preflight
    artifact established (SPEC-001 §2 runs forward mode and falls back to central finite
    differences, recording which survived). There is no third branch: a model whose preflight
    has not run raises :class:`JvpMethodUnresolved` rather than silently taking ``forward``,
    which on the hybrid is the method SPEC-001 §2 flags as unsupported through
    ``gated_delta_update``.
    """
    if requested is not None:
        if requested not in JVP_METHODS:
            raise ValueError(f"unknown JVP method {requested!r}")
        return requested, "cli"
    from local_llm_lab.pipeline import preflight

    name = getattr(spec, "name", None)
    path = preflight.artifact_path(spec, output_root) if isinstance(name, str) and name else None
    record: Any = None
    if path is not None:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = None
    block = record.get("jvp") if isinstance(record, dict) else None
    method = block.get("method") if isinstance(block, dict) else None
    if method not in JVP_METHODS:
        raise JvpMethodUnresolved(
            f"no JVP method established for model {name!r}: pass --jvp-method explicitly, or "
            f"run `agent-pipeline preflight` so that {path} records jvp.method "
            "(SPEC-001 §2, ruling R18a)"
        )
    return str(method), "preflight"


def resolve_capture_dtype(view: Any, requested: str) -> dict[str, Any]:
    """What ``probes.capture_dtype`` asked for against what the view can actually deliver.

    ``ArchitectureView`` casts every block output to float32 (``arch.py`` ``embed``,
    ``run_block``, ``final_norm``), so a ``native`` request is honoured only by a view that
    accepts a dtype on its capture. Recording the difference is the point: R18a wants the
    float32 path's deviation visible, not asserted away.
    """
    if requested not in CAPTURE_DTYPES:
        requested = "native"
    supported = _view_accepts_capture_dtype(view)
    effective = requested if supported else "float32"
    return {
        "requested": requested,
        "effective": effective,
        "view_supports_dtype": supported,
        "cast_boundary": (
            "ArchitectureView.embed/run_block/final_norm cast every block output to float32; "
            "the tangent and the tail JVP are float32 by mandate (R18, briefing §1.5)"
        ),
    }


def _view_accepts_capture_dtype(view: Any) -> bool:
    import inspect

    residuals = getattr(view, "residuals", None)
    if residuals is None:
        return False
    try:
        parameters = inspect.signature(residuals).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        return False
    return "capture_dtype" in parameters or "dtype" in parameters


def render_probe_prompt(tokenizer: Any, messages: list[dict[str, Any]], *, spec: Any) -> str:
    """Render the J-lens probe context under the selected model declaration."""
    from local_llm_lab.pipeline.protocol import build_prompt

    return build_prompt(tokenizer, messages, spec=spec)

# A small, hand-written stand-in for a pretraining distribution: plain prose, code, structured
# key=value lines, file paths, and a little dialogue. This is NOT a sample of the model's real
# pretraining data, so J-lens averages over it are an approximation; keep it varied rather than
# representative of any one domain.
DEFAULT_CORPUS: tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog near the old stone bridge.",
    "def add(a, b):\n    return a + b",
    "status=ready\nretries=3\nowner=platform-team",
    '"Are you coming?" she asked, glancing at the clock above the door.',
    "/var/log/app/error-2024-11-03.log",
    "In 1969, astronauts first walked on the surface of the Moon.",
    "SELECT id, name FROM users WHERE active = 1 ORDER BY name;",
    "invoice-2-537.txt: total due $412.50, payment terms net 30.",
    "The committee will reconvene next Tuesday to finalise the budget.",
    "import numpy as np\narr = np.zeros((3, 3))",
    "host=10.0.0.14 port=8080 protocol=https timeout=30s",
    "He whispered, 'Not now, someone might hear us,' and stepped back.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "class Node:\n    def __init__(self, value):\n        self.value = value",
    "reports/quarterly/q3-2025-summary.csv",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "git commit -m 'fix off-by-one error in the retry loop'",
    "name=Ada Lovelace\nrole=mathematician\nborn=1815",
    "Despite the storm, the ferry departed only twenty minutes late.",
    "for i in range(10):\n    print(i * i)",
    "config/production.yaml",
    '"I disagree," he said flatly, folding his arms across his chest.',
    "Mount Everest is the tallest mountain above sea level on Earth.",
    "user_id=48213 action=login result=success timestamp=2026-01-04T09:12:00Z",
)


def encode(tokenizer: Any, text: str) -> list[int]:
    """Token ids for ``text`` with no added special tokens, where the tokenizer supports that."""
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def build_corpus(
    tokenizer: Any,
    snippets: Sequence[str] = DEFAULT_CORPUS,
    *,
    size: int,
    length: int,
) -> list[list[int]]:
    """``size`` corpus contexts of at least ``length`` tokens each.

    The paper averages over a thousand 128-token sequences; ``DEFAULT_CORPUS`` is 24 snippets
    of roughly six to twenty tokens, several of which (a bare file path, say) have no usable
    future window at all. Concatenating the snippets into longer sequences is the cheapest fix
    and leaves the distribution unchanged (Head of Interpretability, A1(2)): context ``i``
    starts at snippet ``i`` and appends the following snippets, cycling, until it reaches
    ``length`` tokens. Deterministic, so the artifact's corpus is reproducible from
    ``--corpus-size`` and ``--corpus-length`` alone.
    """
    if size < 1:
        raise ValueError("corpus size must be positive")
    if length < 1:
        raise ValueError("corpus length must be positive")
    if not snippets:
        raise ValueError("corpus snippets must not be empty")
    contexts: list[list[int]] = []
    for start in range(size):
        pieces: list[str] = []
        ids: list[int] = []
        for offset in range(len(snippets)):
            pieces.append(snippets[(start + offset) % len(snippets)])
            ids = encode(tokenizer, "\n".join(pieces))
            if len(ids) >= length:
                break
        contexts.append(ids)
    return contexts


def _view(value: Any) -> Any:
    """Return a view, temporarily adapting legacy model callers until Task 6."""
    if callable(getattr(value, "residuals", None)) and callable(getattr(value, "tail", None)):
        return value
    return ArchitectureView.from_model(value)


def residual_at(
    view: ArchitectureView,
    token_ids: Any,
    layer: int,
    *,
    capture_dtype: str = "float32",
) -> Any:
    """Residual stream entering layer ``layer`` for a single context.

    Runs the view's embedding plus the blocks before ``layer`` (an empty prefix for
    ``layer == 0`` returns the raw embedding), each layer called with the model's own causal
    mask so the result matches its true forward pass.

    ``capture_dtype`` carries the registry's ``probes.capture_dtype`` (R18b) to the view. The
    float32 cast boundary is inside the view: ``embed``, ``run_block`` and ``final_norm`` each
    cast to float32, so a view without a dtype-aware capture returns float32 whatever is asked
    for, and :func:`resolve_capture_dtype` is what records that gap. Returns an ``mx.array``
    of shape ``(1, T, d)``.
    """
    architecture = _view(view)
    if capture_dtype != "float32" and _view_accepts_capture_dtype(architecture):
        return architecture.residuals(token_ids, [layer], capture_dtype=capture_dtype)[layer]
    return architecture.residuals(token_ids, [layer])[layer]


def readouts_for_layer(
    names: Sequence[str], layer: int, num_layers: int | None
) -> tuple[str, ...]:
    """The requested readouts that are *measurements* at ``layer`` (issue #68).

    At ``layer == num_layers`` no decoder block remains, so ``future`` and ``all`` are
    structural zeros. A CLI computes what is left there rather than propagating
    :class:`NoTailBlocksError` and losing that layer's ``self`` and logit-lens rows along with
    them. Every other layer keeps every readout it was asked for.
    """
    if not isinstance(num_layers, int) or layer < num_layers:
        return tuple(names)
    kept = tuple(name for name in names if name not in FINAL_LAYER_EXCLUDED_READOUTS)
    # Asking *only* for excluded readouts there leaves nothing to compute; hand the names back
    # unchanged so the caller meets NoTailBlocksError and its explanation rather than an empty
    # request and a generic "unknown readout(s) []".
    return kept or tuple(names)


def final_layer_exclusion(
    layers: Sequence[int], names: Sequence[str], num_layers: int | None
) -> dict[str, Any] | None:
    """R34's record of issue #68's exclusion, or ``None`` when it did not apply to this run.

    Written into the conformance block so an artifact *says* why those cells are absent
    instead of leaving a reader to derive it from a gap in a table.
    """
    if not isinstance(num_layers, int) or num_layers not in tuple(layers):
        return None
    excluded = [name for name in names if name in FINAL_LAYER_EXCLUDED_READOUTS]
    if not excluded:
        return None
    return {
        "layer": num_layers,
        "excluded": excluded,
        "computed": [*(name for name in names if name not in excluded), "logit_lens"],
        "reason": (
            "no decoder block remains at the final layer: the tail is the final norm and the "
            "unembedding, both position-wise, so the output tangent is exactly zero at every "
            "position after the source. These readouts would be structural zeros for every "
            "context, identical across models and conditions, rather than measurements "
            "(issue #68)"
        ),
        "holm": (
            "the excluded readouts' Holm families are built from the layers actually scored "
            "under them, so this layer is not a member and does not enter the correction"
        ),
    }


def probe_layer_kind(view: Any, layer: int) -> str:
    """The kind of the block that **wrote** probe layer ``layer`` (R34's writer convention).

    Layer ``L`` is the residual after block ``L - 1``, so the writing block is ``L - 1``;
    layer ``0`` is the embedding output and has no writing block. A view without
    ``layer_kind`` (the probe fakes) records ``unknown`` rather than guessing.
    """
    if layer == 0:
        return "embedding"
    kind_of = getattr(view, "layer_kind", None)
    if not callable(kind_of):
        return "unknown"
    try:
        return str(kind_of(layer - 1))
    except (ValueError, IndexError, AttributeError):
        return "unknown"


def hybrid_period(view: Any) -> tuple[int | None, str]:
    """The hybrid backbone's attention period, derived from the decoder's own blocks.

    A Qwen3.5-style hybrid builds every block from one number:
    ``DecoderLayer.is_linear = (layer_idx + 1) % full_attention_interval != 0``. Inverted, the
    attention blocks sit at indices ``p-1, 2p-1, ...``, so the period is the first attention
    block's index plus one -- readable from exactly the place
    :meth:`ArchitectureView.layer_kind` already reads, which is why that method stayed correct
    through the whole of the defect this replaces.

    Structural first, deliberately. ``ArchitectureView`` exists to "discover the decoder
    structurally, without consulting a model-type string" (``arch.py``), and a walk over
    configuration attribute names is against that grain. It went stale silently: it returned
    ``(None, "unavailable")`` on a real ``qwen3_5`` model that plainly carried the field, the
    sweep fell back to the six registry fractions instead of the pre-registered nine-layer
    kind-matched family, and the artifact logged ``hybrid_period=-`` beside a perfectly
    correct list of block kinds.

    The configuration is still read, as a **cross-check** rather than as the source. When both
    routes answer and they disagree, that is a genuine inconsistency between a model's blocks
    and its own configuration -- impossible on a consistent model, since a real
    ``DecoderLayer`` derives ``is_linear`` from the field -- so this raises rather than quietly
    preferring one. A view with no blocks (the probe fakes) falls back to the walk alone.

    Returns ``(period, source)``, the source naming which route produced the number. A dense
    backbone has no linear-attention block and therefore no period at all: ``(None, ...)``
    with a source that says so, so an artifact's dash is explained rather than bare.
    """
    structural, structural_source = _structural_period(view)
    configured, configured_source = _configured_period(view)
    if structural is not None and configured is not None and structural != configured:
        raise ValueError(
            f"hybrid period disagreement: the decoder's own blocks imply {structural} "
            f"({structural_source}), but the configuration says {configured} "
            f"({configured_source}). A real DecoderLayer derives is_linear from "
            f"{HYBRID_PERIOD_FIELD}, so a consistent model cannot produce both; refusing to "
            "guess which one the sweep should pre-register its layer family from."
        )
    if structural is not None:
        if configured is None:
            return structural, structural_source
        return structural, f"{structural_source}, cross-checked against {configured_source}"
    if configured is not None:
        # No structural answer, but the configuration still carries what the blocks cannot:
        # either the view exposes no blocks at all (the probe fakes), or every block is one
        # kind -- a period of 1 makes them all attention, a period past the depth makes them
        # all linear -- and only the field distinguishes those from a genuinely dense model.
        return configured, configured_source
    return None, structural_source


def _structural_period(view: Any) -> tuple[int | None, str]:
    """Invert ``is_linear`` over the view's own blocks; the source names which case applied."""
    blocks = getattr(view, "blocks", None)
    kind_of = getattr(view, "layer_kind", None)
    if blocks is None or not callable(kind_of):
        # A view without blocks is a probe fake, not a model. The configuration walk is all
        # there is, and "unavailable" is the string those artifacts have always carried.
        return None, "unavailable"
    try:
        kinds = [str(kind_of(index)) for index in range(len(blocks))]
    except (ValueError, IndexError, AttributeError, TypeError):  # pragma: no cover - defensive
        return None, "unavailable"
    if RECURRENT_KIND not in kinds:
        return None, (
            "view.blocks: no linear-attention block, so the backbone is dense and has no "
            "hybrid period"
        )
    if ATTENTION_KIND not in kinds:
        return None, (
            "view.blocks: no attention block, so any hybrid period exceeds the decoder depth"
        )
    first = kinds.index(ATTENTION_KIND)
    # ``is_linear = (index + 1) % p != 0`` puts the attention blocks at p-1, 2p-1, ...; the
    # first of them is at p-1, so the period is that index plus one.
    return first + 1, (
        f"view.blocks[{first}].is_linear (first attention block; period = index + 1)"
    )


def _configured_period(view: Any) -> tuple[int | None, str]:
    """The configuration walk, kept as the cross-check rather than as the source of truth.

    ``full_attention_interval`` lives on the text configuration, which sits at a different
    place under every wrapper, so this walks the handful of attribute names configurations are
    ever reached through rather than assuming one path.
    """
    roots = [
        (name, getattr(view, name, None)) for name in ("model", "text_module")
    ]
    queue: list[tuple[str, Any, int]] = [
        (f"view.{name}", node, 0) for name, node in roots if node is not None
    ]
    seen: set[int] = set()
    while queue:
        path, node, depth = queue.pop(0)
        if id(node) in seen:
            continue
        seen.add(id(node))
        value = _period_field(node)
        if value is not None:
            return value, f"{path}.{HYBRID_PERIOD_FIELD}"
        if depth >= _CONFIG_SEARCH_DEPTH:
            continue
        for name in _CONFIG_ATTRIBUTES:
            child = _member(node, name)
            if child is not None and not isinstance(child, str | bytes | int | float | bool):
                queue.append((f"{path}.{name}", child, depth + 1))
    return None, "unavailable"


def _member(node: Any, name: str) -> Any:
    """One member of ``node``, reached by attribute **and** by mapping key -- both are needed.

    ``mlx.nn.Module`` is a ``dict`` subclass. A module is therefore simultaneously a Mapping
    and an ordinary object with attributes, and the dict half holds *only* its registered
    parameters and submodules -- never a plain Python attribute. Testing
    ``isinstance(node, Mapping)`` first and returning ``node.get(name)`` consequently
    **shadows attribute access on every module**: ``model.args`` is a dataclass hung off the
    module, not a registered member, so the mapping branch answered ``None`` for it while
    ``getattr`` answered correctly, and the period walk never reached the text configuration.
    That is the trap; do not re-set it by reordering these two branches.

    Attribute first, mapping as the fallback, so a plain ``dict`` configuration -- the other
    shape this walk crosses -- still resolves. The order is safe for the names actually asked
    for here (``args``, ``config``, ``text_config``, ``language_model``, ``model`` and
    ``full_attention_interval``): none of them is a ``dict`` method, so no attribute can
    shadow a real key going the other way.
    """
    try:
        value = getattr(node, name, None)
    except (AttributeError, KeyError, TypeError):  # pragma: no cover - defensive
        value = None
    if value is not None:
        return value
    if isinstance(node, Mapping):
        try:
            return node.get(name)
        except (AttributeError, KeyError, TypeError):  # pragma: no cover - defensive
            return None
    return None


def _period_field(node: Any) -> int | None:
    value = _member(node, HYBRID_PERIOD_FIELD)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def in_band_layers(indices: Sequence[int], num_layers: int) -> tuple[int, ...]:
    """The decisive band of EXP-001 §2: fractions 1/3 to 5/6 of the decoder's depth.

    Fraction 1/6 and the final layer are *reported, not decisive*, so they fall outside. The
    band edges are resolved against the actual depth with the same mapping
    ``probes.policies.resolve_layers`` uses, so the registry's rounded literals (``0.333``,
    ``0.833``) land on the edges exactly rather than needing a tolerance.
    """
    low = max(1, round(IN_BAND_FRACTIONS[0] * num_layers))
    high = max(1, round(IN_BAND_FRACTIONS[1] * num_layers))
    return tuple(
        index for index in indices if low <= index <= high and index != num_layers
    )


@dataclass(frozen=True)
class LayerFamily:
    """One sweep's layers, with each layer's kind and the role it plays in EXP-001 §2.

    ``primary`` layers are the in-band ones, ``partner`` layers are the kind-matched partners
    the derivation added for them, and ``reported`` layers are the remaining registry
    fractions -- fraction 1/6 and the final layer -- which §2 reports but does not decide on.
    An explicit ``--layers`` list is taken verbatim and every layer is marked ``explicit``.
    """

    layers: tuple[int, ...]
    kinds: dict[int, str]
    roles: dict[int, str]
    in_band: tuple[int, ...]
    partners: tuple[int, ...]
    pairs: dict[int, int]
    period: int | None
    period_source: str
    derived: bool
    reason: str

    @property
    def primary_layers(self) -> tuple[int, ...]:
        """EXP-001 §2's primary set: the in-band layers *and* their kind-matched partners."""
        return tuple(sorted(set(self.in_band) | set(self.partners)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "layers": list(self.layers),
            "kinds": {str(layer): kind for layer, kind in self.kinds.items()},
            "roles": {str(layer): role for layer, role in self.roles.items()},
            "in_band": list(self.in_band),
            "partners": list(self.partners),
            "pairs": {str(layer): partner for layer, partner in self.pairs.items()},
            "primary_layers": list(self.primary_layers),
            "hybrid_period": self.period,
            "hybrid_period_source": self.period_source,
            "derived": self.derived,
            "rule": LAYER_FAMILY_RULE,
            "reason": self.reason,
        }


def kind_matched_layer_family(
    selected: Sequence[int],
    *,
    num_layers: int,
    period: int | None,
    period_source: str = "unavailable",
    kind_of: Callable[[int], str] | None = None,
    derive: bool = True,
) -> LayerFamily:
    """EXP-001 §3.5's layer list: the selection plus its kind-matched partners, in one sweep.

    §2 compares an attention layer against a linear-attention layer *at comparable depth*, so
    the registry fractions alone are not enough: on the 4B they land on a recurrent block four
    times out of six, and a sweep of the fractions alone need contain no such pair at all.

    The rule. Probe layer ``L`` is the residual after block ``L - 1`` and a probe layer's kind
    is that block's kind (R34). On a hybrid with ``full_attention_interval = p`` the block
    that wrote ``L`` is an attention block exactly when ``L % p == 0``, so the layers written
    by attention are the multiples of ``p`` within ``[1, num_layers]``. For each in-band
    layer, the kind-matched partner is the nearest layer on that grid -- smaller depth
    difference first, lower index on a tie -- and it joins the sweep when it is not already
    there. An in-band layer that is *itself* on the grid is already an attention output and
    takes no partner, which is what EXP-001 §3.5 records for the 4B's middle and final
    fractions. Where a view supplies kinds, a grid layer it calls recurrent is not used as a
    partner, so the config and the blocks cannot disagree silently.

    Where no opposite kind exists -- a dense backbone with no period at all, a period of one
    (every layer an attention output), or a period past the decoder's depth (no attention
    layer at all) -- the selection is returned unchanged with the reason recorded, rather than
    raising: EXP-001 §5 runs this same code on the dense 3B as the R35 comparator.
    """
    order = tuple(dict.fromkeys(int(layer) for layer in selected))
    for layer in order:
        if not 1 <= layer <= num_layers:
            raise ValueError(f"layer {layer} is outside [1, {num_layers}]")

    if not derive:
        return LayerFamily(
            layers=order,
            kinds={layer: _family_kind(layer, kind_of, period) for layer in order},
            roles=dict.fromkeys(order, "explicit"),
            in_band=(),
            partners=(),
            pairs={},
            period=period,
            period_source=period_source,
            derived=False,
            reason=(
                "--layers was given: the list is honoured verbatim and no kind-matched "
                "partner is derived"
            ),
        )

    band = in_band_layers(order, num_layers)
    grid = () if period is None else tuple(range(period, num_layers + 1, period))
    if kind_of is not None:
        grid = tuple(
            layer
            for layer in grid
            if _family_kind(layer, kind_of, period) != RECURRENT_KIND
        )
    reason = _degenerate_reason(period, num_layers, len(grid))
    if reason is not None:
        return LayerFamily(
            layers=order,
            kinds={layer: _family_kind(layer, kind_of, period) for layer in order},
            roles={layer: ("primary" if layer in band else "reported") for layer in order},
            in_band=band,
            partners=(),
            pairs={},
            period=period,
            period_source=period_source,
            derived=True,
            reason=reason,
        )

    layers = list(order)
    partners: list[int] = []
    pairs: dict[int, int] = {}
    for layer in band:
        partner = min(grid, key=lambda candidate: (abs(candidate - layer), candidate))
        if partner == layer:
            continue
        pairs[layer] = partner
        if partner not in layers:
            layers.append(partner)
            partners.append(partner)
    ordered = tuple(sorted(layers))
    partner_set = set(partners)
    return LayerFamily(
        layers=ordered,
        kinds={layer: _family_kind(layer, kind_of, period) for layer in ordered},
        roles={
            layer: (
                "primary"
                if layer in band
                else "partner"
                if layer in partner_set
                else "reported"
            )
            for layer in ordered
        },
        in_band=band,
        partners=tuple(sorted(partner_set)),
        pairs=pairs,
        period=period,
        period_source=period_source,
        derived=True,
        reason=(
            f"kind-matched partners derived from {HYBRID_PERIOD_FIELD}={period} "
            f"({period_source})"
        ),
    )


def _degenerate_reason(period: int | None, num_layers: int, grid_size: int) -> str | None:
    """Why no partner exists, or ``None`` when both block kinds are present."""
    if period is None:
        return (
            f"no hybrid period in the model configuration ({HYBRID_PERIOD_FIELD} absent): a "
            "dense backbone has one block kind, so no kind-matched partner exists"
        )
    if grid_size == 0:
        return (
            f"{HYBRID_PERIOD_FIELD}={period} exceeds the decoder depth {num_layers}: no layer "
            "is written by an attention block, so no kind-matched partner exists"
        )
    if grid_size >= num_layers:
        return (
            f"{HYBRID_PERIOD_FIELD}={period}: every layer is written by an attention block, "
            "so no kind-matched partner exists"
        )
    return None


def _family_kind(layer: int, kind_of: Callable[[int], str] | None, period: int | None) -> str:
    """A probe layer's kind: the view's own answer, else the config period's, else unknown."""
    if kind_of is not None:
        kind = kind_of(layer)
        if kind is not None:
            return str(kind)
    if period is None or period < 1:
        return "unknown"
    return ATTENTION_KIND if layer % period == 0 else RECURRENT_KIND


def jacobian_vector_product(
    view: ArchitectureView,
    layer: int,
    primal: Any,
    tangent: Any,
    *,
    method: Literal["forward", "finite_difference"] = "forward",
) -> Any:
    """One forward-mode JVP of the tail of the network: layers ``[layer:]`` then the final norm.

    ``primal`` and ``tangent`` are both ``(1, T, d)``. This is ``J_c . tangent`` for the single
    context that produced ``primal`` (``J_c`` being the local Jacobian of
    ``norm(layers[layer:](.))`` at ``primal``), computed without ever materialising ``J_c``.
    Both primal and tangent are cast to float32 before the JVP (mandatory: the tangent has been
    observed to overflow to infinity in float16 on the real quantized checkpoint), and the
    result is returned in float32.
    """
    import mlx.core as mx

    primal32 = primal.astype(mx.float32)
    tangent32 = tangent.astype(mx.float32)

    fn = _view(view).tail(layer)
    if method == "forward":
        _, (tangent_out,) = mx.jvp(fn, [primal32], [tangent32])
        return tangent_out.astype(mx.float32)
    if method != "finite_difference":
        raise ValueError(f"unknown JVP method {method!r}")
    tangent_norm = mx.sqrt(mx.sum(tangent32 * tangent32))
    if float(tangent_norm.item()) == 0.0:
        return mx.zeros_like(primal32)
    primal_norm = mx.sqrt(mx.sum(primal32 * primal32))
    # The scale formula is undefined at the all-zero residual; retain a finite central step.
    eps = 1e-2 if float(primal_norm.item()) == 0.0 else 1e-2 * primal_norm / tangent_norm
    return ((fn(primal32 + eps * tangent32) - fn(primal32 - eps * tangent32)) / (2.0 * eps)).astype(mx.float32)


def jlens_readouts(
    view: ArchitectureView,
    layer: int,
    probe: Any,
    corpus_ids: list[Any],
    *,
    source_positions: Sequence[SourcePosition] | None = None,
    readouts: Sequence[str] = READOUTS,
    method: Literal["forward", "finite_difference"] = "forward",
    capture_dtype: str = "float32",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Estimate ``J_layer . probe`` under each requested readout, from one JVP per sample.

    For every corpus context and every requested source position: build
    ``primal = residual_at(view, ids, layer)``, place ``probe`` at the source position of an
    otherwise zero tangent, and take **one** forward-mode (or finite-difference) JVP of the
    tail. The three readouts are three different reductions of that same output tangent
    (EXP-001 §3.2, B3):

    * ``self`` -- the output tangent at the source position. This is the paper's self-only
      limiting case and is what this module computed exclusively before EXP-001.
    * ``future`` -- the sum of the output tangent over positions strictly after the source:
      the broadcast component, what a source position makes available to later positions,
      which is the readout that can see a recurrent channel.
    * ``all`` -- ``self + future``, the paper's default estimator up to a scale factor that a
      sign test ignores. Primary.

    The two reduction axes are different reductions and R34 names both (Head of
    Interpretability, issue #61): **within** a sample the output tangent is *summed* over the
    readout's window of output positions, and **across** samples the resulting vectors are
    *averaged* -- ``totals[name] / used`` below, so a skipped sample lowers the divisor rather
    than entering as a zero. The paper does the same: it differentiates a sum over target
    positions and then takes the mean over source positions and over prompts.

    A sample whose future window is empty raises :class:`EmptyFutureWindowError` whenever
    ``future`` or ``all`` is requested: with the source at the last token the sum is
    identically zero, and a structural zero is indistinguishable from a genuine null in
    exactly the readout that carries the recurrent-state question (Head of Interpretability,
    A1). Samples whose output tangent is not finite, or whose source does not exist in that
    context, are skipped rather than allowed to poison the mean.

    Returns ``(mapped, stats)`` where ``mapped`` maps each requested readout name to its
    ``(d,)`` float32 estimate, and ``stats`` records ``used``, ``skipped``, ``method``, the
    source positions as written, and the ``window`` block R34 requires (median future window
    per context, and median positions read per readout).
    """
    import mlx.core as mx

    names = tuple(readouts)
    unknown = [name for name in names if name not in READOUTS]
    if unknown or not names:
        raise ValueError(f"unknown readout(s) {unknown}; expected a subset of {list(READOUTS)}")
    sources = tuple(source_positions) if source_positions else resolve_source_positions("-1")
    needs_future = bool({"future", "all"} & set(names))

    architecture = _view(view)
    # Issue #68: the tail at ``L == num_layers`` is the final norm and the unembedding, both
    # position-wise, so ``future`` is a structural zero for every context and ``all`` collapses
    # onto ``self`` exactly. The run that raised this reported 0 of 42 in both columns with the
    # smallest rank in the Holm family, for a reason that has nothing to do with the model.
    # Checked before any work is done, so the caller pays nothing to be told.
    num_layers = getattr(architecture, "num_layers", None)
    if needs_future and isinstance(num_layers, int) and layer >= num_layers:
        blocked = ", ".join(name for name in names if name in {"future", "all"})
        raise NoTailBlocksError(
            f"layer {layer} of a {num_layers}-block decoder: no decoder block remains in the "
            f"tail, which is the final norm and the unembedding alone. Both are position-wise, "
            f"so the output tangent is exactly zero at every position after the source and the "
            f"{blocked!r} readout(s) would be a structural zero for every context rather than a "
            "measurement (issue #68). Probe a layer before the last, or ask only for 'self'."
        )
    probe32 = probe.astype(mx.float32)
    totals = {name: mx.zeros_like(probe32) for name in names}
    used = 0
    skipped = 0
    windows: list[int] = []
    lengths: list[int] = []
    for context_index, ids in enumerate(corpus_ids):
        primal = residual_at(architecture, ids, layer, capture_dtype=capture_dtype)
        length = int(primal.shape[1])
        lengths.append(length)
        for source in sources:
            pos = source_index(source, length)
            if not 0 <= pos < length:
                skipped += 1
                continue
            window = length - pos - 1
            if needs_future and window <= 0:
                raise EmptyFutureWindowError(
                    f"context {context_index} of {length} token(s) has no position after "
                    f"source {source.token!r} (position {pos}): the 'future' readout would be "
                    "a structural zero rather than a measurement. Lengthen the corpus "
                    "(--corpus-length) or move the source inward (--source-positions)."
                )
            tangent = mx.zeros_like(primal)
            tangent[0, pos] = probe32
            output = jacobian_vector_product(
                architecture, layer, primal, tangent, method=method
            )
            vectors = {"self": output[0, pos]}
            vectors["future"] = (
                mx.sum(output[0, pos + 1 :], axis=0) if window > 0 else mx.zeros_like(probe32)
            )
            vectors["all"] = vectors["self"] + vectors["future"]
            chosen = {name: vectors[name] for name in names}
            if not all(bool(mx.all(mx.isfinite(value)).item()) for value in chosen.values()):
                skipped += 1
                continue
            for name, value in chosen.items():
                totals[name] = totals[name] + value
            windows.append(window)
            used += 1
    if used == 0:
        raise ValueError("jlens_readouts: every corpus sample was skipped (empty or non-finite)")
    median_window = statistics.median(windows)
    stats: dict[str, Any] = {
        "used": used,
        "skipped": skipped,
        "method": method,
        "readouts": list(names),
        "source_positions": [source.as_dict() for source in sources],
        "capture_dtype": capture_dtype,
        "window": {
            "median_future_window": median_window,
            "min_future_window": min(windows),
            "median_context_length": statistics.median(lengths) if lengths else 0,
            "min_context_length": min(lengths) if lengths else 0,
            "median_positions_read": {
                "self": 1,
                "future": median_window,
                "all": median_window + 1,
            },
        },
    }
    mapped = {name: (totals[name] / used).astype(mx.float32) for name in names}
    return mapped, stats


def jlens_map(
    view: ArchitectureView,
    layer: int,
    probe: Any,
    corpus_ids: list[Any],
    *,
    position: int = -1,
    method: Literal["forward", "finite_difference"] = "forward",
) -> tuple[Any, dict[str, Any]]:
    """Estimate ``J_layer . probe`` (shape ``(d,)``) under the ``self`` readout only.

    The **self-only limiting case** of :func:`jlens_readouts` (R34): one source position, the
    output tangent read at that same position, which the source paper describes as the
    variant closest in spirit to the logit lens. Kept because it is the historical contract
    that ``adapter_delta`` and the preflight depend on; new work takes all three readouts from
    :func:`jlens_readouts`, whose ``all`` estimator is the paper's default.

    For each context: builds ``primal = residual_at(view, ids, layer)``, a tangent that is
    zero everywhere except ``probe`` placed at token ``position``, and reads off the output
    tangent at that same position via :func:`jacobian_vector_product`. Because
    ``E_c[J_c] . probe == E_c[J_c . probe]``, the mean of these per-context vectors *is*
    ``J_layer . probe`` for the corpus-averaged Jacobian; each JVP costs one tail-network
    forward-mode pass rather than one pass per dimension of ``probe``.

    Any context whose output tangent is not finite (or whose ``position`` does not exist in
    that context) is skipped rather than allowed to poison the mean. The returned statistics
    record ``used``, ``skipped``, and the requested derivative ``method``.

    Raises ``ValueError`` if every context was skipped.
    """
    view_call = callable(getattr(view, "residuals", None)) and callable(getattr(view, "tail", None))
    source = SourcePosition(token=str(position), kind="index", value=float(position))
    try:
        readouts, readout_stats = jlens_readouts(
            view,
            layer,
            probe,
            corpus_ids,
            source_positions=(source,),
            readouts=("self",),
            method=method,
        )
    except EmptyFutureWindowError:  # pragma: no cover - the self readout never needs a window
        raise
    except ValueError as error:
        if "every corpus sample was skipped" not in str(error):
            raise
        raise ValueError(
            "jlens_map: every corpus context was skipped (empty or non-finite)"
        ) from error
    stats: dict[str, Any] = {
        "used": readout_stats["used"],
        "skipped": readout_stats["skipped"],
        "method": method,
    }
    mapped = readouts["self"]
    # Task 6 removes raw-model callers.  The temporary coercion branch keeps their historical
    # mapped-array result while ArchitectureView callers receive the binding tuple contract.
    if not view_call:
        return mapped  # type: ignore[return-value]
    return mapped, stats


def distribution(view: ArchitectureView, vector: Any) -> Any:
    """``softmax(W_U . norm(vector))`` over the vocabulary, in float32; ``vector`` is ``(d,)``."""
    import mlx.core as mx

    v = vector.astype(mx.float32)
    architecture = _view(view)
    return mx.softmax(architecture.unembed(architecture.final_norm(v)).astype(mx.float32), axis=-1)


def _top_k(distribution: Any, tokenizer: Any, k: int) -> list[tuple[str, float, int]]:
    import mlx.core as mx

    order = mx.argsort(-distribution)[:k]
    ids = [int(i) for i in order.tolist()]
    probs = distribution.tolist()
    return [(tokenizer.decode([token_id]), float(probs[token_id]), token_id) for token_id in ids]


def readout(view: ArchitectureView, vector: Any, tokenizer: Any, k: int = 20) -> list[tuple[str, float, int]]:
    """The J-lens readout of ``vector``: ``top_k(softmax(W_U . norm(vector)))``.

    ``vector`` is expected to already be a Jacobian-mapped activation (e.g. the output of
    :func:`jlens_map`); this function itself applies no Jacobian, only the final norm and
    unembedding, so it doubles as the shared machinery behind :func:`logit_lens`. The
    ArchitectureView selects tied or untied unembedding. Returns up to ``k``
    ``(token_string, probability, token_id)`` triples sorted by descending probability.
    """
    return _top_k(distribution(view, vector), tokenizer, k)


def logit_lens(
    view: ArchitectureView, vector: Any, tokenizer: Any, k: int = 20
) -> list[tuple[str, float, int]]:
    """The standard logit-lens baseline: identical machinery to :func:`readout`, but intended
    to be called on a raw (non-Jacobian-mapped) activation. Comparing this against
    :func:`readout` on the same activation, run through :func:`jlens_map` first, is the control
    that shows whether the J-lens adds anything over just norming and unembedding directly.
    """
    return _top_k(distribution(view, vector), tokenizer, k)


def token_evidence(
    distribution: Any, tokenizer: Any, candidates: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """How strongly a full vocabulary ``distribution`` supports each candidate string.

    ``candidates`` maps a label (e.g. ``"target"``, ``"already_read"``) to a literal string.

    Candidates in this experiment are sibling file paths that share almost every token
    (``lab/test/0007/ledger/invoice-2-537.txt`` versus ``...invoice-1-924.txt``), so scoring
    over all tokens measures the shared prefix and returns an identical answer for every
    candidate, which is useless. The distinctive metrics below therefore restrict attention to
    the tokens unique to one candidate, which is what actually discriminates them. Both are
    reported so the difference stays visible. Returns, per label::

        {
            "tokens": [...], "max_probability": ..., "best_rank": ..., "total_probability": ...,
            "distinctive_tokens": [...],        # tokens no other candidate contains
            "distinctive_max_probability": ..., # the metric to read
            "distinctive_best_rank": ...,
            "distinctive_total_probability": ...,
        }
    """
    import mlx.core as mx

    order = mx.argsort(-distribution)
    order_list = [int(i) for i in order.tolist()]
    rank_of = {token_id: rank + 1 for rank, token_id in enumerate(order_list)}
    probs = distribution.tolist()
    encoded = {label: encode(tokenizer, text) for label, text in candidates.items()}
    evidence: dict[str, dict[str, Any]] = {}
    for label, ids in encoded.items():
        shared = {i for other, other_ids in encoded.items() if other != label for i in other_ids}
        distinctive = [token_id for token_id in ids if token_id not in shared]

        def summarize(token_ids: list[int]) -> tuple[float, int | None, float]:
            if not token_ids:
                return 0.0, None, 0.0
            piece_probs = [float(probs[token_id]) for token_id in token_ids]
            return (
                max(piece_probs),
                min(rank_of[token_id] for token_id in token_ids),
                sum(piece_probs),
            )

        all_max, all_rank, all_total = summarize(list(ids))
        dist_max, dist_rank, dist_total = summarize(distinctive)
        evidence[label] = {
            "tokens": [tokenizer.decode([token_id]) for token_id in ids],
            "max_probability": all_max,
            "best_rank": all_rank,
            "total_probability": all_total,
            "distinctive_tokens": [tokenizer.decode([token_id]) for token_id in distinctive],
            "distinctive_max_probability": dist_max,
            "distinctive_best_rank": dist_rank,
            "distinctive_total_probability": dist_total,
        }
    return evidence


def conformance_block(
    *,
    layers: Sequence[int],
    layer_kinds: dict[int, str],
    layer_selection: dict[str, Any] | None,
    layer_family: dict[str, Any] | None = None,
    source_positions: Sequence[SourcePosition],
    readouts: Sequence[str],
    corpus_size: int,
    corpus_length: dict[str, Any],
    window: dict[str, Any],
    jvp_method: str,
    jvp_method_source: str,
    capture_dtype: dict[str, Any],
    final_layer_readout_exclusion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The R34 conformance statement, in the artifact and mirrored by the module docstring.

    Names every element the ruling and its 2026-09-05 amendment require: the layer-index
    convention (and therefore what a probe layer's *kind* means), the source positions, the
    output positions each readout reads, corpus size and context length, the median future
    window per context and readout, the JVP method with its source, and which variant the
    paper's self-only limiting case corresponds to here. ``layer_family`` adds EXP-001 §3.5's
    derivation: the hybrid period it came from, the kind-matched pairs, and each layer's
    ``primary``/``partner`` role, so a reader can see which pairs the per-kind contrast rests
    on.

    ``reduction`` names **both** reduction axes rather than only the one inside a sample. The
    estimator sums over output positions within a sample and averages over samples, and the
    bare word "sum" is ambiguous between the two -- which is exactly how the question the Head
    of Interpretability answered on issue #61 arose. Only the within-sample axis was ever a
    choice: the across-sample division is a positive global scale that ``distribution``'s final
    RMS norm removes before the unembedding, so it cannot move a number either way.
    """
    return {
        "ruling": "R34",
        "paper": "Gurnee, Sofroniew et al., arXiv 2607.15495, §2.1/§4.1/§A.7",
        "layer_index_convention": (
            "layer L is the residual after block L-1 (layer 0 = embedding output, "
            "L = num_layers = pre-final-norm); a probe layer's kind is the kind of the block "
            "that wrote it, block L-1, not the block about to read it"
        ),
        "layers": list(layers),
        "layer_kinds": dict(layer_kinds),
        "layer_selection": layer_selection,
        "layer_family": layer_family,
        "source_positions": [source.as_dict() for source in source_positions],
        "output_positions_read": {
            "self": "the source position only",
            "future": "every position strictly after the source, summed within a sample",
            "all": (
                "the source position and every position after it, summed within a sample"
            ),
        },
        # Both reduction axes, named separately: the bare word "sum" does not say which of
        # the two it governs, and that ambiguity is what raised the C2 question (Head of
        # Interpretability, issue #61). The paper differentiates a sum over target positions
        # and then averages over sources and prompts; this is the same pair of reductions.
        "reduction": {
            "within_a_sample": (
                "the output tangent is summed over the readout's window of output positions, "
                "one (context, source) sample at a time"
            ),
            "across_samples": (
                "the per-sample vectors are averaged: the running totals are divided by the "
                "number of samples used, so a skipped sample lowers the divisor rather than "
                "entering as a zero"
            ),
            "sample": "one (corpus context, source position) pair, contributing one JVP",
        },
        "readouts": list(readouts),
        "primary_readout": PRIMARY_READOUT,
        "self_only_limiting_case": (
            "the 'self' readout is the source paper's self-only limiting case, the variant it "
            "calls closest in spirit to the logit lens; every J-lens number recorded in this "
            "repository before EXP-001 was drawn under it"
        ),
        # Two guards, named apart on purpose: the first is about positions in the context, the
        # second about blocks in the tail, and an artifact that named only one would leave a
        # reader unable to tell which condition a run actually hit.
        "empty_future_window_policy": (
            "a readout whose future window is empty -- the source position is the last token "
            "of the context, so no later position exists -- raises EmptyFutureWindowError; it "
            "is never reported as zero"
        ),
        "final_layer_readout_policy": (
            "a 'future' or 'all' readout requested at layer L == num_layers, where no decoder "
            "block remains in the tail (only the position-wise final norm and unembedding), "
            "raises NoTailBlocksError; a CLI excludes those cells rather than reporting the "
            "structural zero (issue #68)"
        ),
        "final_layer_readout_exclusion": final_layer_readout_exclusion,
        "corpus_size": corpus_size,
        "corpus_length": corpus_length,
        "window": window,
        "jvp_method": jvp_method,
        "jvp_method_source": jvp_method_source,
        "capture_dtype": capture_dtype,
    }


def comparability_block(
    *,
    model: str,
    hf_id: str | None,
    policy: str | None,
    adapter: str | None,
    jvp_method: str,
    template_kwargs: dict[str, Any],
    keep_last: int | None,
    row_keep_last: int | None = None,
    rewindowed: bool = True,
    estimator_variant: str,
    layer_selection: dict[str, Any] | None,
    layer_kinds: dict[int, str],
    layer_family: dict[str, Any] | None = None,
    generator_version: int | None,
    data_seed: int | None,
    fp32_manual_vs_native: dict[str, Any] | None,
) -> dict[str, Any]:
    """The R35 fields that decide whether two probe tables may be put in one table.

    R35: comparable only where policy, derivative method, layer selection (as fractions and
    kinds), generator version, prompt rendering (the template kwargs) and estimator variant
    are equal, or every difference is named in both artifacts. Base-versus-adapter is a named
    difference, not a comparison, so ``policy`` and ``adapter`` sit here rather than only in
    the identity block. ``layer_family`` carries the derived list and the hybrid period behind
    it, since two sweeps whose partners differ are not the same layer selection.

    Prompt rendering carries three windowing coordinates, not one (C3, issue #62). ``keep_last``
    alone is the number handed to ``build_prompt``, which does not say what window the context
    actually carries: a caller whose rows were already windowed by ``pipeline.data.build_rows``
    passes the row's own tool count so that render-time windowing is a no-op, and recording only
    that number hides both the real window and the reason it is a no-op. ``row_keep_last`` is
    the window already built into the rows (``None`` where the caller assembled its own
    messages), and ``rewindowed`` says whether ``build_prompt`` narrowed them further. It
    defaults to ``True`` because ``build_prompt`` windows by default; a caller that pre-windowed
    its rows must say so.
    """
    return {
        "ruling": "R35",
        "model": model,
        "hf_id": hf_id,
        "policy": policy,
        "adapter": adapter,
        "base_versus_adapter_is_a_named_difference": True,
        "derivative_method": jvp_method,
        "prompt_rendering": {
            "renderer": "local_llm_lab.pipeline.protocol.build_prompt",
            "template_kwargs": dict(template_kwargs),
            "keep_last": keep_last,
            "row_keep_last": row_keep_last,
            "rewindowed": rewindowed,
            "windowing_rule": (
                "row_keep_last is the observation window pipeline.data.build_rows had already "
                "applied when the row was built (null where the caller assembled its own "
                "messages); keep_last is the window build_prompt was asked for on top of it; "
                "rewindowed is false where keep_last was set to the row's own tool count so "
                "that render-time windowing is a no-op and the row's window is the one the "
                "context carries -- re-windowing an already-stubbed observation would rewrite "
                "its line count and corrupt the context"
            ),
        },
        "estimator_variant": estimator_variant,
        "layer_selection": layer_selection,
        "layer_kinds": dict(layer_kinds),
        "layer_family": layer_family,
        "generator_version": generator_version,
        "data_seed": data_seed,
        "fp32_manual_vs_native": fp32_manual_vs_native,
    }


def probe_layers(
    view: ArchitectureView,
    tokenizer: Any,
    token_ids: Any,
    layers: list[int],
    corpus_ids: list[Any],
    candidates: dict[str, str],
    position: int = -1,
    *,
    method: Literal["forward", "finite_difference"] = "forward",
    k: int = 20,
    source_positions: Sequence[SourcePosition] | None = None,
    readouts: Sequence[str] = ("self",),
    capture_dtype: str = "float32",
    progress: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the full J-lens vs. logit-lens comparison at each layer in ``layers``.

    For every layer: takes the layer's residual stream at ``token_ids`` (position
    ``position``) as the probe, estimates ``J_layer . probe`` with :func:`jlens_readouts` over
    ``corpus_ids`` under every requested readout, and computes both the J-lens distributions
    (one per readout) and the logit-lens distribution (from the raw probe). Returns one record
    per layer, in the order ``layers`` was given, with the top-``k`` tokens and
    :func:`token_evidence` for each.

    ``readouts`` and ``source_positions`` default to the historical self-only limiting case at
    the probe position, so a library caller that has not been migrated keeps its contract; the
    CLI passes all three readouts and explicit interior sources (EXP-001 §3.2). Asking for
    ``future`` or ``all`` without sources that leave a window raises rather than returning a
    structural zero.

    ``progress``, when given, is called once per layer -- the outer unit of work, so R26(g)'s
    "no run is silent for more than one unit" holds for this CLI.

    The primary readout's tokens and evidence are additionally mirrored onto the historical
    ``jlens_top_k``/``jlens_evidence`` keys so existing readers of this record keep working.
    """
    architecture = _view(view)
    names = tuple(readouts)
    primary = PRIMARY_READOUT if PRIMARY_READOUT in names else names[0]
    records = []
    for step, layer in enumerate(layers, 1):
        residual = residual_at(architecture, token_ids, layer, capture_dtype=capture_dtype)
        length = residual.shape[1]
        pos = position if position >= 0 else length + position
        probe = residual[0, pos]
        # Issue #68: at the final layer no decoder block remains, so 'future' and 'all' are
        # structural zeros. Compute what is still a measurement there rather than propagating
        # the raise and losing this layer's 'self' and logit-lens rows with it.
        layer_names = readouts_for_layer(names, layer, getattr(architecture, "num_layers", None))
        mapped, stats = jlens_readouts(
            architecture,
            layer,
            probe,
            corpus_ids,
            source_positions=source_positions,
            readouts=layer_names,
            method=method,
            capture_dtype=capture_dtype,
        )
        logit_distribution = distribution(architecture, probe)
        per_readout: dict[str, Any] = {}
        for name in layer_names:
            readout_distribution = distribution(architecture, mapped[name])
            per_readout[name] = {
                "jlens_top_k": _top_k(readout_distribution, tokenizer, k),
                "jlens_evidence": token_evidence(readout_distribution, tokenizer, candidates),
            }
        # The mirrored historical keys follow whichever readout is primary *at this layer*:
        # 'all' is excluded at the final layer (issue #68), so 'self' carries them there.
        layer_primary = primary if primary in layer_names else layer_names[0]
        record = {
            "layer": layer,
            "layer_kind": probe_layer_kind(architecture, layer),
            "corpus_used": stats.get("used", 0),
            "corpus_skipped": stats.get("skipped", 0),
            "jvp_method": stats.get("method", method),
            "source_positions": stats.get("source_positions", []),
            "window": stats.get("window", {}),
            "readouts": per_readout,
            "primary_readout": layer_primary,
            "jlens_top_k": per_readout[layer_primary]["jlens_top_k"],
            "jlens_evidence": per_readout[layer_primary]["jlens_evidence"],
            "logit_lens_top_k": _top_k(logit_distribution, tokenizer, k),
            "logit_lens_evidence": token_evidence(logit_distribution, tokenizer, candidates),
        }
        records.append(record)
        if progress is not None:
            progress(step, len(layers), "layer", layer=layer, kind=record["layer_kind"])
    return records


# --------------------------------------------------------------------------- CLI


def _replay_to_step(task: Any, step: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Replay ``task``'s EXPERT trajectory through a fresh :class:`Simulator` up to (not
    including) ``step``, returning the exact message list ``build_prompt`` would see just
    before the policy generates that step, and the paths read by earlier ``read_file`` steps.

    Deliberately replays the recorded expert actions rather than generating: this makes the
    probe's context, and therefore its result, independent of any policy's actual behaviour.
    """
    from local_llm_lab.pipeline.env import Simulator
    from local_llm_lab.pipeline.protocol import SYSTEM_PROMPT, assistant_message, tool_message

    if not 0 <= step < len(task.steps):
        raise ValueError(f"--step {step} out of range: task horizon is {len(task.steps)}")
    simulator = Simulator.for_task(task)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    read_paths: list[str] = []
    for index, expert_step in enumerate(task.steps):
        if index == step:
            break
        observation = simulator.execute(expert_step.action)
        if expert_step.action.name == "read_file":
            read_paths.append(str(expert_step.action.arguments["path"]))
        messages.append(assistant_message(expert_step.thought, expert_step.action))
        messages.append(tool_message(expert_step.action.name, observation))
    return messages, read_paths


def _unseen_path(split: str, seed: int, task_index: int, exclude: dict[str, str]) -> str:
    """A file path drawn from a different task, guaranteed absent from ``exclude``'s files."""
    from local_llm_lab.pipeline.tasks import make_tasks

    pool = make_tasks(split, max(task_index + 8, 8), seed)
    for index, other in enumerate(pool):
        if index == task_index:
            continue
        for path in other.files:
            if path not in exclude:
                return path
    raise RuntimeError("could not find a path from another task that is unseen by this one")


def _print_table(records: list[dict[str, Any]], candidates: dict[str, str]) -> None:
    for record in records:
        print(f"\n== layer {record['layer']} " + "=" * 40)
        print(f"corpus contexts used={record['corpus_used']} skipped={record['corpus_skipped']}")
        for lens_name, top_key, evidence_key in (
            ("logit-lens", "logit_lens_top_k", "logit_lens_evidence"),
            ("j-lens", "jlens_top_k", "jlens_evidence"),
        ):
            top = ", ".join(f"{tok!r}:{prob:.3f}" for tok, prob, _ in record[top_key][:8])
            print(f"  {lens_name} top tokens: {top}")
            for label in candidates:
                evidence = record[evidence_key][label]
                print(
                    f"    {label:12s} distinctive={evidence['distinctive_tokens']} "
                    f"rank={evidence['distinctive_best_rank']} "
                    f"max_p={evidence['distinctive_max_probability']:.6f} "
                    f"(all-token rank={evidence['best_rank']})"
                )


def main() -> None:  # noqa: C901 - pre-existing probe CLI orchestration
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import preflight as preflight_module
    from local_llm_lab.pipeline.evaluate import load_policy
    from local_llm_lab.pipeline.tasks import GENERATOR_VERSION, make_tasks
    from local_llm_lab.probes.guard import add_gpu_arguments, require_idle_gpu
    from local_llm_lab.probes.policies import (
        resolve_layers,
        resolve_policy,
        validate_layer_syntax,
    )
    from local_llm_lab.probes.state_probe import preflight_precision_block, spec_capture_dtype
    from local_llm_lab.provenance import write_provenance
    from local_llm_lab.runlog import RunLog, git_commit

    parser = argparse.ArgumentParser(
        description=(
            "J-lens interpretability probe: compare the Jacobian-lens and logit-lens readouts "
            "of a model's residual stream at a chosen step of an expert agent trajectory."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--policy", help="a policy named by the selected model; defaults to the base"
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Deprecated explicit adapter-directory alias; omit for the untouched base.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Which decision point in the expert trajectory to probe.",
    )
    parser.add_argument(
        "--force-prefix",
        help=(
            "Text appended to the prompt before probing, e.g. the note plus a partial tool "
            "call. Use it to probe the position where the model is actually due to emit the "
            "candidate, rather than the start of its note."
        ),
    )
    parser.add_argument(
        "--layers",
        help=(
            "comma-separated layer indices or fractions, honoured verbatim; omit for the "
            "registry fractions plus their kind-matched partners (EXP-001 §3.5)"
        ),
    )
    parser.add_argument("--corpus-size", type=int, default=16)
    parser.add_argument(
        "--corpus-length",
        type=int,
        default=DEFAULT_CORPUS_LENGTH,
        help=(
            "Minimum tokens per corpus context; snippets are concatenated up to it so the "
            "future readout has a window to sum over (R34)."
        ),
    )
    parser.add_argument(
        "--source-positions",
        default=DEFAULT_SOURCE_POSITIONS,
        help=(
            "Where the tangent is placed in each corpus context: comma-separated fractions "
            "of the context or explicit indices. The last token has an empty future window."
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--jvp-method",
        choices=JVP_METHODS,
        default=None,
        help=(
            "Directional-derivative implementation used for every J-lens record. Omitted, it "
            "is the method this model's preflight artifact established (R18a); with neither, "
            "the run fails closed rather than assuming forward mode."
        ),
    )
    parser.add_argument("--skip-preflight-check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for run.log/events.jsonl (R26 a); defaults to --output's directory.",
    )
    add_gpu_arguments(parser)
    args = parser.parse_args()
    spec = load_model_spec(args.model)
    if args.policy is not None and args.adapter is not None:
        parser.error("--policy and --adapter cannot be used together")
    try:
        validate_layer_syntax(args.layers)
        source_positions = resolve_source_positions(args.source_positions)
        policy_name = args.policy or "base"
        adapter = resolve_policy(
            str(args.adapter) if args.adapter is not None else policy_name,
            spec,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.corpus_size < 1:
        parser.error("--corpus-size must be positive")
    if args.corpus_length < 1:
        parser.error("--corpus-length must be positive")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        jvp_method, jvp_method_source = resolve_jvp_method(args.jvp_method, spec)
    except (JvpMethodUnresolved, ValueError) as error:
        parser.error(str(error))
    log_directory = args.log_dir or (args.output.parent if args.output is not None else None)
    if log_directory is None:
        parser.error("pass --output or --log-dir: every probe run writes run.log (R26 a)")

    seed = 20260902
    tasks = make_tasks(args.split, max(args.task_index + 1, 1), seed)
    if args.task_index >= len(tasks):
        parser.error(f"--task-index {args.task_index} out of range for split {args.split!r}")
    task = tasks[args.task_index]
    try:
        messages, read_paths = _replay_to_step(task, args.step)
    except ValueError as error:
        parser.error(str(error))
        return

    target_step = task.steps[args.step]
    if target_step.action.name != "read_file":
        parser.error(
            f"--step {args.step} is a {target_step.action.name!r} call, not read_file; "
            "choose a step whose expert action reads a file"
        )
    target_path = str(target_step.action.arguments["path"])
    already_read = next((path for path in reversed(read_paths) if path != target_path), None)
    if already_read is None:
        already_read = next(
            (path for path in sorted(task.files) if path != target_path), target_path
        )
    unseen_path = _unseen_path(args.split, seed, args.task_index, task.files)
    candidates = {"target": target_path, "already_read": already_read, "unseen": unseen_path}

    capture_dtype = spec_capture_dtype(spec)
    # R26(e): the log opens once the identity is known but before the preflight gate, the GPU
    # guard and the model load, so run.log alone says what ran and covers the whole run.
    with RunLog.open(
        log_directory,
        name="jlens",
        command=sys.argv,
        identity={
            "model": spec.name,
            "hf_id": spec.hf_id,
            "policy": policy_name if args.adapter is None else None,
            "adapter": str(adapter) if adapter is not None else None,
            "layers": args.layers,
            "split": args.split,
            "task_id": task.task_id,
            "step": args.step,
            "corpus_size": args.corpus_size,
            "corpus_length": args.corpus_length,
            "source_positions": args.source_positions,
            "jvp_method": jvp_method,
            "jvp_method_source": jvp_method_source,
            "capture_dtype": capture_dtype,
            "generator_version": GENERATOR_VERSION,
            "git_commit": git_commit(),
        },
    ) as log:
        preflight_module.require_preflight(spec, skip=args.skip_preflight_check)
        require_idle_gpu(parser, args, "loading the J-lens model")
        log.info("loading policy", model=args.model, policy=policy_name)
        model, tokenizer, view, resolved = load_policy(spec, adapter)
        del model
        try:
            selection = resolve_layers(args.layers, spec, view.num_layers)
        except ValueError as error:
            parser.error(str(error))
        period, period_source = hybrid_period(view)
        family = kind_matched_layer_family(
            selection.indices,
            num_layers=view.num_layers,
            period=period,
            period_source=period_source,
            kind_of=lambda layer: probe_layer_kind(view, layer),
            derive=selection.source != "cli",
        )
        layers = list(family.layers)
        layer_kinds = dict(family.kinds)
        log.info(
            "layers",
            selected=layers,
            kinds=[layer_kinds[layer] for layer in layers],
            roles=[family.roles[layer] for layer in layers],
            hybrid_period=family.period,
        )

        prompt = render_probe_prompt(tokenizer, messages, spec=spec)
        if args.force_prefix:
            prompt = prompt + args.force_prefix
        bos = getattr(tokenizer, "bos_token", None)
        add_special = bos is None or not prompt.startswith(bos)
        try:
            token_ids = tokenizer.encode(prompt, add_special_tokens=add_special)
        except TypeError:
            token_ids = tokenizer.encode(prompt)

        corpus_ids = build_corpus(
            tokenizer, DEFAULT_CORPUS, size=args.corpus_size, length=args.corpus_length
        )
        lengths = [len(ids) for ids in corpus_ids]
        log.info(
            "corpus",
            contexts=len(corpus_ids),
            median_tokens=statistics.median(lengths) if lengths else 0,
            min_tokens=min(lengths) if lengths else 0,
        )

        records = probe_layers(
            view,
            tokenizer,
            token_ids,
            layers,
            corpus_ids,
            candidates,
            method=jvp_method,
            k=args.top_k,
            source_positions=source_positions,
            readouts=READOUTS,
            capture_dtype=capture_dtype,
            progress=log.progress,
        )

        print(f"task={task.task_id} step={args.step} layers={layers} corpus_size={len(corpus_ids)}")
        print(f"candidates={candidates}")
        _print_table(records, candidates)

        conformance = conformance_block(
            layers=layers,
            layer_kinds=layer_kinds,
            layer_selection=selection.as_dict(),
            layer_family=family.as_dict(),
            source_positions=source_positions,
            readouts=READOUTS,
            corpus_size=len(corpus_ids),
            corpus_length={
                "requested_minimum": args.corpus_length,
                "median_tokens": statistics.median(lengths) if lengths else 0,
                "min_tokens": min(lengths) if lengths else 0,
            },
            window=records[0]["window"] if records else {},
            jvp_method=jvp_method,
            jvp_method_source=jvp_method_source,
            capture_dtype=resolve_capture_dtype(view, capture_dtype),
            # The registry fractions include 1.0, so this CLI sweeps the final layer too and
            # owes a reader the same explanation the sweep does (issue #68).
            final_layer_readout_exclusion=final_layer_exclusion(
                layers, READOUTS, view.num_layers
            ),
        )
        comparability = comparability_block(
            model=spec.name,
            hf_id=spec.hf_id,
            policy=policy_name if args.adapter is None else None,
            adapter=None if adapter is None else str(adapter),
            jvp_method=jvp_method,
            template_kwargs=dict(spec.chat.template_kwargs),
            keep_last=None,
            estimator_variant=PRIMARY_READOUT,
            layer_selection=selection.as_dict(),
            layer_kinds=layer_kinds,
            layer_family=family.as_dict(),
            generator_version=GENERATOR_VERSION,
            data_seed=seed,
            fp32_manual_vs_native=preflight_precision_block(spec),
        )

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": args.model,
                "policy": policy_name if args.adapter is None else None,
                "adapter": None if adapter is None else str(adapter),
                "split": args.split,
                "task_id": task.task_id,
                "step": args.step,
                "layers": layers,
                "layer_selection": selection.as_dict(),
                "layer_family": family.as_dict(),
                "layer_kinds": {str(layer): kind for layer, kind in layer_kinds.items()},
                "layer_roles": {str(layer): role for layer, role in family.roles.items()},
                "corpus_size": len(corpus_ids),
                "corpus_length": conformance["corpus_length"],
                "source_positions": [source.as_dict() for source in source_positions],
                "jvp_method": jvp_method,
                "jvp_method_source": jvp_method_source,
                "capture_dtype": conformance["capture_dtype"],
                "conformance": conformance,
                "comparability": comparability,
                "candidates": candidates,
                "records": records,
            }
            args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            print(f"\nWrote {args.output}")
            log.info("wrote artifact", path=str(args.output))
        write_provenance(
            Path(log_directory),
            resolved=resolved,
            spec=spec,
            extra={"stage": "jlens", "conformance": conformance, "comparability": comparability},
        )
