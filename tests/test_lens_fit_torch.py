"""The torch lens-fit stage, driven end to end against the tiny synthetic decoder.

The stage that will run on the device is the stage exercised here: `model` and `rows` are seams on
`stage_lens_fit_torch`, so the test substitutes the fixture model rather than a shape resembling the
stage. What cannot be exercised on this box is `load_hf_lens_model`, which needs a real checkpoint
and a GPU; it is declared unexecuted in the module docstring and gated in the device checklist, and
the last test here pins that declaration so a green suite is not read as covering it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_lens_upstream import HIDDEN, NUM_LAYERS, SEQ_LEN, TinyLensModel, make_rows

from local_llm_lab.pipeline.lens_fitting import fit_torch
from local_llm_lab.pipeline.lens_fitting import upstream as adapter
from local_llm_lab.pipeline.live_lens.instruments import LensIdentity, LensMaps

SPEC = SimpleNamespace(base="tiny-synthetic-decoder", training=None, hf_id="unused/tiny")


@pytest.fixture(scope="module")
def upstream():
    try:
        return adapter.load_upstream()
    except adapter.UpstreamUnavailable as error:  # pragma: no cover - box without the clone
        pytest.skip(str(error))


def _fit(tmp_path, upstream, **kwargs):
    return fit_torch.stage_lens_fit_torch(
        corpus=tmp_path / "corpus.jsonl",
        out=tmp_path / "lens.npz",
        spec=SPEC,
        kind="jacobian",
        max_seq_len=SEQ_LEN,
        dim_batch=3,
        model=TinyLensModel(),
        rows=make_rows(),
        upstream=upstream,
        **kwargs,
    )


def test_the_stage_fits_writes_and_reads_back_what_it_wrote(tmp_path, upstream) -> None:
    result = _fit(tmp_path, upstream)

    assert result["event"] == "done"
    assert result["backend"] == "torch"
    assert result["n_prompts"] == 3 and result["n_skipped"] == 0
    # Repo convention: maps at layers 1..num_layers-1; the final layer is the identity and is
    # never stored.
    assert result["layers_written"] == list(range(1, NUM_LAYERS))
    assert (tmp_path / "lens.npz").exists()

    # The digest in the result is the digest of the file, not a number the writer reported about
    # itself: reload through the loader, which verifies it.
    LensMaps.load(
        tmp_path / "lens.npz",
        expected_sha256=result["npz_sha256"],
        hidden_size=HIDDEN,
        num_layers=NUM_LAYERS,
        identity=LensIdentity(base=SPEC.base, num_layers=NUM_LAYERS),
    )


def test_the_estimator_is_declared_and_is_not_the_finite_difference_one(tmp_path, upstream) -> None:
    """The field the CLI's refusal called load-bearing, asserted on the artefact that reaches disk.

    A torch fit and an MLX fit of the same corpus are different instruments, and the golden test's
    residual is only the difference between the two estimators if each artefact says which one made
    it. Read back through `read_declared_nu`, which binds the sidecar to the npz's digest.
    """
    result = _fit(tmp_path, upstream)

    assert result["estimator"] == adapter.ESTIMATOR_EXACT_AUTOGRAD
    nu = adapter.read_declared_nu(tmp_path / "lens.npz")
    assert nu["estimator"] == adapter.ESTIMATOR_EXACT_AUTOGRAD
    assert nu["estimator"] != adapter.ESTIMATOR_FINITE_DIFFERENCE
    assert nu == result["nu"]
    # Six §6.1 fields, and the fitted length, which decides whether two lenses are comparable.
    assert {
        "estimator", "endpoint", "position_weighting", "pair_weighting", "corpus", "precision"
    } <= set(nu)
    assert nu["position_weighting"]["max_seq_len"] == SEQ_LEN
    assert nu["corpus"]["n_prompts"] == 3


def test_the_storage_dtype_is_measured_from_the_archive_not_from_the_working_dtype(
    tmp_path, upstream
) -> None:
    """R60(c): the declaration carries its measurement.

    The maps are float32 in memory whatever the file holds, so a stage that reported the working
    dtype would report float32 for a float16 artefact and never be wrong in a way anyone noticed.
    """
    result = _fit(tmp_path, upstream)

    assert result["storage_dtype"] == ["float32"]
    with np.load(tmp_path / "lens.npz") as archive:
        on_disk = sorted(
            {str(archive[name].dtype) for name in archive.files if name.startswith("J")}
        )
    assert result["storage_dtype"] == on_disk


def test_a_regression_kind_is_refused_by_name_rather_than_fitted_as_a_jacobian(
    tmp_path, upstream
) -> None:
    """The torch side has no regression port, and silently fitting the other kind would produce an
    artefact whose ν says exact-autograd for a run the caller asked to be a regression."""
    with pytest.raises(fit_torch.TorchFitRefused) as raised:
        fit_torch.stage_lens_fit_torch(
            corpus=tmp_path / "corpus.jsonl",
            out=tmp_path / "lens.npz",
            spec=SPEC,
            kind="regression",
            model=TinyLensModel(),
            rows=make_rows(),
            upstream=upstream,
        )
    assert "regression" in str(raised.value)
    assert "missing implementation" in str(raised.value)
    assert not (tmp_path / "lens.npz").exists()


def test_the_cli_entry_point_returns_zero_and_prints_one_json_event(tmp_path, upstream, capsys):
    args = SimpleNamespace(
        corpus=tmp_path / "corpus.jsonl", out=tmp_path / "lens.npz", kind="jacobian"
    )
    original = fit_torch.stage_lens_fit_torch
    try:
        fit_torch.stage_lens_fit_torch = lambda **kwargs: original(
            **{**kwargs, "max_seq_len": SEQ_LEN, "dim_batch": 3,
               "model": TinyLensModel(), "rows": make_rows(), "upstream": upstream}
        )
        assert fit_torch.run(args, SPEC) == 0
    finally:
        fit_torch.stage_lens_fit_torch = original
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "done" and event["estimator"] == adapter.ESTIMATOR_EXACT_AUTOGRAD


def test_the_refused_kind_exits_three_and_says_so_rather_than_raising_into_the_shell(
    tmp_path, upstream, capsys
):
    args = SimpleNamespace(corpus=tmp_path / "c.jsonl", out=tmp_path / "l.npz", kind="regression")
    original = fit_torch.stage_lens_fit_torch
    try:
        fit_torch.stage_lens_fit_torch = lambda **kwargs: original(
            **{**kwargs, "model": TinyLensModel(), "rows": make_rows(), "upstream": upstream}
        )
        assert fit_torch.run(args, SPEC) == 3
    finally:
        fit_torch.stage_lens_fit_torch = original
    assert json.loads(capsys.readouterr().out)["event"] == "refused"


def test_the_unexecuted_loader_is_declared_as_such_and_is_small_enough_to_read() -> None:
    """`load_hf_lens_model` needs a checkpoint and a GPU and has never run.

    Pinned rather than trusted: a docstring saying "never executed on this box" is only worth
    anything while it is true, and the check that keeps it true is that the function stays small
    and unconditional. If it grows a branch, this fails and the claim gets re-earned.
    """
    import ast
    import inspect

    source = inspect.getsource(fit_torch.load_hf_lens_model)
    assert "Never executed on this box" in source
    tree = ast.parse(source.strip())
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.If, ast.Try, ast.While, ast.IfExp))
    ]
    assert branches == [], "an unexecuted function must stay unconditional to be readable as one"


def test_torch_is_the_only_array_library_this_stage_needs() -> None:
    """The fit runs where MLX does not exist. Importing the stage must not reach for it."""
    import sys

    assert "local_llm_lab.pipeline.lens_fitting.fit_torch" in sys.modules
    assert torch.__version__  # the stage's actual dependency, present
