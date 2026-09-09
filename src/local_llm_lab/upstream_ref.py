"""The one seam through which this repository reaches the upstream Jacobian-lens reference.

Upstream is **called, never vendored**: the estimator lives in a read-only clone, or on a GPU box an
installed distribution pinned to a commit, and nothing here copies it. This module sits at the top
level rather than inside ``lens_fitting`` because more than one workstream needs that seam — WS-A's
torch view wraps upstream's ``ActivationRecorder``, WS-D's adapter calls its estimator — and two
independent import paths would defeat the point of having one. A module doing ``from jlens import
hf`` at its top level errors at collection wherever the clone is absent, and on a box holding both a
clone and an installed copy, whichever the interpreter imported first silently wins.

``UpstreamUnavailable`` subclasses ``ImportError`` as well as ``RuntimeError``, so
``pytest.importorskip`` and lazy importers treat an absent clone as a skip carrying the message
below rather than as a collection error carrying none.

``lens_fitting.upstream`` re-exports every name here, so nothing that already imports it changes.
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from local_llm_lab.project import PROJECT_ROOT

__all__ = [
    "DEFAULT_JLENS_PATHS",
    "EXPECTED_JLENS_COMMIT",
    "JLENS_PATH_ENV",
    "Upstream",
    "UpstreamUnavailable",
    "load_upstream",
]



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


class UpstreamUnavailable(RuntimeError, ImportError):
    """The read-only upstream clone is not where we were told to find it."""

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


def _installed_commit() -> str | None:
    """The commit an installed ``jlens`` distribution records, or ``None``.

    A GPU box installs upstream from the ``[cuda]`` extra as
    ``jlens @ git+https://github.com/anthropics/jacobian-lens@581d398...``, which is a package
    directory with no ``.git``. Without this the provenance field reads ``None`` on exactly the
    machine that matters, and a missing figure must never read as a passing one -- here it would
    quietly become "commit unknown" on the only box the golden test will ever run on.
    """
    try:
        from importlib.metadata import Distribution

        raw = Distribution.from_name("jlens").read_text("direct_url.json")
    except Exception:  # noqa: BLE001 - any metadata failure means "not installed that way"
        return None
    if not raw:
        return None
    try:
        return json.loads(raw).get("vcs_info", {}).get("commit_id") or None
    except ValueError:
        return None


def _clone_commit(path: Path) -> str | None:
    """The clone's HEAD, or ``None`` when it cannot be read — never a guess, never a default."""
    head = path / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
    except OSError:
        return _installed_commit()
    if ref.startswith("ref: "):
        try:
            return (path / ".git" / ref[5:]).read_text().strip()
        except OSError:
            return _installed_commit()
    return ref or _installed_commit()


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
