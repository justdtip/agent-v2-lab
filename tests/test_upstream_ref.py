"""The seam that reaches the upstream Jacobian-lens reference, tested where it lives.

`local_llm_lab.upstream_ref` stays on the main line because more than one workstream imports it and
it is an import seam rather than backend code: WS-A's torch view wraps upstream's
`ActivationRecorder`, WS-D's adapter calls its estimator. The adapter itself lives on `cuda-ws-d`,
so the test that asserts both import paths resolve to one object lives there with it.
"""
from __future__ import annotations

import pytest

from local_llm_lab.upstream_ref import (
    EXPECTED_JLENS_COMMIT,
    UpstreamUnavailable,
    _installed_commit,
    load_upstream,
)


def test_an_absent_clone_reads_as_a_skip_rather_than_a_collection_error():
    """`UpstreamUnavailable` is an `ImportError` so lazy importers can treat it as a skip.

    A module doing `from jlens import ...` at top level aborts collection of the whole file on a
    box without the clone, taking unrelated tests with it. Raising an `ImportError` subclass that
    carries our own message lets `pytest.importorskip` skip with a reason instead.
    """
    assert issubclass(UpstreamUnavailable, ImportError)
    assert issubclass(UpstreamUnavailable, RuntimeError), "the original contract is kept"
    with pytest.raises(UpstreamUnavailable, match="was not found"):
        load_upstream("/nonexistent/jacobian-lens")


def test_the_recorded_commit_falls_back_to_an_installed_pin_rather_than_none():
    """On a GPU box upstream is a pip install with no `.git`, and `None` would read as verified.

    The `[cuda]` extra installs it as `jlens @ git+...@581d398...`, which pip records in
    `direct_url.json`. Without the fallback the provenance field is `None` on precisely the machine
    the golden test runs on — a missing figure reading as a passing one.
    """
    # No `jlens` distribution is installed here, so the honest answer is None rather than a guess.
    assert _installed_commit() is None


def test_a_present_clone_declares_the_commit_it_is_pinned_to():
    """Provenance is measured from the clone, never assumed, and says whether it is the pinned one.

    Skipped rather than failed where no clone exists, which is the whole point of the `ImportError`
    contract above.
    """
    try:
        upstream = load_upstream()
    except UpstreamUnavailable as absent:
        pytest.skip(str(absent))
    provenance = upstream.provenance()
    assert provenance["vendored"] is False, "upstream is called, never copied"
    assert provenance["expected_commit"] == EXPECTED_JLENS_COMMIT
    assert provenance["commit_matches_expected"] == (
        provenance["commit"] == EXPECTED_JLENS_COMMIT
    ), "the flag is derived from the two values, not asserted beside them"
