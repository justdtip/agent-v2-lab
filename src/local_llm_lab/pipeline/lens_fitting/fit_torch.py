"""The torch lens-fit stage: the CLI's dispatch target, and upstream's estimator behind it.

`scripts/lens_fit.py` reads `device.backend()` once and, on torch, imports this module. Until this
file existed it refused instead, and the refusal said why in the field that matters: **the two
backends are not two implementations of one fit.** The MLX side runs this repository's
finite-difference estimator; this side runs upstream's exact autograd. They are different
instruments on the same corpus, the difference between them is what the WS-D golden test measures,
and a CLI that quietly gave each box whichever estimator it could run would absorb that measurement
into a packaging decision. Every artefact written here therefore declares
``ESTIMATOR_EXACT_AUTOGRAD`` in its ν, and the stage refuses to write one that does not.

**What is exercised here and what is not.** The fit, the ν round-trip and the storage-dtype read are
fixture-tested against the tiny synthetic decoder in `tests/test_lens_fit_torch.py`, on CPU, with no
CUDA and no HF checkpoint. :func:`load_hf_lens_model` is the one part that cannot be: it needs a real
checkpoint and a GPU. It is therefore small, it does nothing conditional, and its three preconditions
— bfloat16 across every block, eager attention, every parameter frozen — are gates in
`research/records/WSD-DEVICE-CHECKLIST-2026-09-09/` rather than assumptions made here. This paragraph
is the declaration that it is unexecuted, so that a green suite is not read as covering it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from local_llm_lab.pipeline.lens_fitting.upstream import (
    ESTIMATOR_EXACT_AUTOGRAD,
    CorpusLensModel,
    declare_nu,
    fit_upstream_jacobian,
    load_upstream,
    read_declared_nu,
    write_upstream_lens,
)
from local_llm_lab.pipeline.live_lens.instruments import LensIdentity, LensMaps

#: The length upstream's own recipe fits at, and the length the hosted lenses were fitted at.
#: Passed explicitly at both ends, never inherited: upstream fits at 128 and reads out at 512 by
#: default and neither signature warns that they differ.
HOSTED_MAX_SEQ_LEN = 128


class TorchFitRefused(RuntimeError):
    """The stage will not produce this artefact, and the message says which precondition failed."""


def load_hf_lens_model(spec: Any, *, upstream: Any = None, device: str, dtype: str) -> Any:
    """An HF checkpoint wrapped in upstream's own ``HFLensModel``. **Never executed on this box.**

    Upstream discovers the block list itself, which is the point of using it rather than describing
    Gemma's structure a second time here. The model object is ours, so the checklist's gates apply
    to it: uniform ``bfloat16``, ``attn_implementation="eager"``, every parameter frozen. The first
    is enforced by the adapter (``MixedPrecisionModel`` reads every block, not a sample); the second
    changes throughput by more than a factor of two and is measured, not assumed; the third is
    load-bearing, because upstream roots its graph with ``requires_grad_(True)`` on a block output
    and that only works while the output is a leaf.
    """
    import torch
    from transformers import AutoModelForCausalLM

    up = upstream or load_upstream()
    import jlens.hf as upstream_hf  # only importable once load_upstream put the clone on sys.path

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        dtype=getattr(torch, dtype),
        attn_implementation="eager",
        device_map=None,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return CorpusLensModel(upstream_hf.HFLensModel(model, tokenizer=None)), up


def stage_lens_fit_torch(
    *,
    corpus: Path,
    out: Path,
    spec: Any,
    kind: str,
    device: str = "cpu",
    dtype: str = "float32",
    max_seq_len: int = HOSTED_MAX_SEQ_LEN,
    dim_batch: int = 8,
    max_rows: int | None = None,
    split: str | None = "fit",
    model: Any | None = None,
    rows: Iterable[dict] | None = None,
    upstream: Any = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Fit through upstream, write the artefact, then read back what was written.

    ``model`` and ``rows`` are seams, not conveniences: the fixture test drives this whole function
    against the tiny synthetic decoder, so the stage that runs on the device is the stage the suite
    exercised, rather than a shape resembling it.

    The verification after the write is deliberate and is the same discipline the orientation check
    follows. Reading ν back through :func:`read_declared_nu` and the maps back through
    ``LensMaps.load`` crosses the writer: checking the dict the writer returned would compare the
    writer with itself and could not fail for any input.
    """
    if kind != "jacobian":
        raise TorchFitRefused(
            f"the torch backend fits kind 'jacobian' and this run asked for {kind!r}. "
            "regression.py has no torch port; this is a missing implementation, not a refusal "
            "about the corpus or the model."
        )
    started = time.monotonic()
    up = upstream or load_upstream()
    if model is None:
        model, up = load_hf_lens_model(spec, upstream=up, device=device, dtype=dtype)
    if rows is None:
        from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus

        rows = read_corpus(Path(corpus))

    fit = fit_upstream_jacobian(
        model,
        rows,
        max_seq_len=max_seq_len,
        dim_batch=dim_batch,
        device=device,
        dtype=dtype,
        split=split,
        max_rows=max_rows,
        upstream=up,
        progress=progress,
    )

    corpus_block = {
        "path": str(corpus),
        "split": split,
        "max_seq_len": max_seq_len,
        "n_prompts": fit.n_prompts,
    }
    # Declared here as well as inside the writer, so the refusal below is a comparison between two
    # derivations rather than a statement checked against itself.
    declared = declare_nu(
        fit,
        num_layers=int(model.n_layers),
        corpus=corpus_block,
        estimator=ESTIMATOR_EXACT_AUTOGRAD,
    )
    if declared["estimator"] != ESTIMATOR_EXACT_AUTOGRAD:
        raise TorchFitRefused("a torch fit that does not declare the exact-autograd estimator")

    written = write_upstream_lens(
        Path(out),
        fit,
        identity=LensIdentity(
            base=spec.base,
            num_layers=int(model.n_layers),
            training=getattr(spec, "training", None),
        ),
        hidden_size=int(fit.d_model),
        num_layers=int(model.n_layers),
        corpus=corpus_block,
        estimator=ESTIMATOR_EXACT_AUTOGRAD,
        upstream=up,
    )

    # Read back through the loaders, not out of `written`.
    nu = read_declared_nu(Path(out))
    if nu["estimator"] != ESTIMATOR_EXACT_AUTOGRAD:
        raise TorchFitRefused(
            f"the artefact at {out} declares estimator {nu['estimator']!r} after being written "
            f"as {ESTIMATOR_EXACT_AUTOGRAD!r}"
        )
    reloaded = LensMaps.load(
        Path(out),
        expected_sha256=written["npz_sha256"],
        hidden_size=int(fit.d_model),
        num_layers=int(model.n_layers),
        identity=LensIdentity(
            base=spec.base,
            num_layers=int(model.n_layers),
            training=getattr(spec, "training", None),
        ),
    )
    return {
        "event": "done",
        "kind": kind,
        "backend": "torch",
        "estimator": ESTIMATOR_EXACT_AUTOGRAD,
        "path": str(out),
        "npz_sha256": written["npz_sha256"],
        "n_prompts": fit.n_prompts,
        "n_skipped": len(fit.skipped),
        "layers_written": sorted(reloaded.maps),
        # What the maps are *stored* at, measured from the archive rather than from the dtype they
        # are cast to on load. A declaration carries its measurement (R60(c)).
        "storage_dtype": list(reloaded.storage_dtype),
        "nu": nu,
        "upstream": up.provenance(),
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def run(args: Any, spec: Any, *, progress: Callable[[dict], None] | None = None) -> int:
    """The CLI's entry point: argparse namespace in, process exit code out."""
    try:
        result = stage_lens_fit_torch(
            corpus=args.corpus,
            out=args.out,
            spec=spec,
            kind=args.kind,
            progress=progress,
        )
    except TorchFitRefused as refused:
        print(json.dumps({"event": "refused", "reason": str(refused)}))
        return 3
    print(json.dumps(result, default=str))
    return 0


__all__ = [
    "HOSTED_MAX_SEQ_LEN",
    "TorchFitRefused",
    "load_hf_lens_model",
    "run",
    "stage_lens_fit_torch",
]
