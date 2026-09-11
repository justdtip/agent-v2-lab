"""The refusal that stands between a capture and a reader (plan-progress §11, precondition 5).

The pre-registration says the reader refuses without the seal. A rule that lives only in the text is
one nobody can be stopped by, so it is a function that raises. `require_seal` is called before the
first read of any capture and returns the seal only when every claim in it is still true of the tree.

Four claims, checked separately because they are four different claims and none can stand in for
another (`METHOD-2026-09-08`, entry 34, tenth instance):

* the seal exists and is complete;
* its own bytes digest to what the caller expected;
* every file it fixes still has the bytes it fixed;
* those same files still match their bytes at the baseline commit the seal names.

The document passing its own checker is the fifth. It runs last because it is the slowest.

Git is reached through `local_llm_lab.spawn.run`, never `subprocess`: a read stage runs in a
process that has touched the model, and forking there aborts the interpreter.

**No argument disables a check.** A gate that can be talked past by a keyword is not a gate — the
same reason `seal.refuse_rederive` is a function and not a flag. The only thing that varies is what
the environment makes checkable: outside a git work tree the baseline cannot be re-derived, and that
is **reported in words** in the returned record rather than silently dropped, exactly as a capture
resumed without a prepared tokenizer reports that the consumed ids were not verified.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from local_llm_lab import spawn

#: Keys without which the file is not a seal. A partial seal is no seal.
REQUIRED_KEYS = ("schema_version", "commit", "seed", "folds", "capture_set", "tolerances", "files")

SEAL_NAME = "seal.json"


class NotSealed(RuntimeError):
    """A read was attempted where no usable seal exists."""


class SealBroken(RuntimeError):
    """A seal exists and the tree no longer matches what it fixed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal_digest(directory: Path) -> str:
    path = Path(directory) / SEAL_NAME
    if not path.exists():
        raise NotSealed(f"{path} does not exist; there is nothing to digest")
    return _sha256(path)


def load_seal(directory: Path) -> dict[str, Any]:
    """The seal, or a refusal naming what is missing."""
    path = Path(directory) / SEAL_NAME
    if not path.exists():
        raise NotSealed(
            f"{path} does not exist: no capture in {directory} is read before the pre-registration "
            "is sealed (§11, precondition 5)"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NotSealed(f"{path} is not readable as JSON: {exc}") from exc
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise NotSealed(f"{path} is missing {missing}; a partial seal is no seal")
    if not payload["files"]:
        raise NotSealed(f"{path} fixes no files; a seal over nothing is no seal")
    return payload


def verify_files(directory: Path, seal: dict[str, Any]) -> list[str]:
    """Every file the seal fixes, against the bytes it fixed. Names all of them, not the first."""
    directory = Path(directory)
    broken = []
    for name, expected in sorted(seal["files"].items()):
        path = directory / name
        if not path.exists():
            broken.append(f"{name} is gone; the seal fixes it at {expected[:12]}…")
        elif _sha256(path) != expected:
            broken.append(f"{name} has moved since the seal: {_sha256(path)[:12]}… not {expected[:12]}…")
    return broken


def verify_baseline(directory: Path, seal: dict[str, Any]) -> tuple[list[str], str]:
    """The same files against the commit the seal names. A second path to the same claim.

    Returns the disagreements and a word for what was actually checked, so a caller that cannot
    reach git reports that rather than reporting a verification it did not perform.
    """
    directory = Path(directory)
    baseline = seal.get("baseline_commit")
    if not baseline:
        return [], "no baseline commit named in the seal; not checked"
    inside = spawn.run(["git", "-C", str(directory), "rev-parse", "--is-inside-work-tree"],
                            capture_output=True, text=True)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return [], f"not a git work tree; the baseline {baseline[:12]}… was not re-derived here"
    found = spawn.run(["git", "-C", str(directory), "rev-parse", "--verify", f"{baseline}^{{commit}}"],
                           capture_output=True, text=True)
    if found.returncode != 0:
        return [f"the seal names baseline {baseline}, which this repository does not have"], "refused"
    broken = []
    for name, expected in sorted(seal["files"].items()):
        rel = spawn.run(["git", "-C", str(directory), "ls-files", "--full-name", name],
                             capture_output=True, text=True).stdout.strip()
        if not rel:
            broken.append(f"{name} is not tracked, so it cannot be compared against {baseline[:12]}…")
            continue
        shown = spawn.run(["git", "-C", str(directory), "show", f"{baseline}:{rel}"],
                               capture_output=True)
        if shown.returncode != 0:
            broken.append(f"{name} does not exist at {baseline[:12]}…")
        elif hashlib.sha256(shown.stdout).hexdigest() != expected:
            broken.append(f"{name} differs from its bytes at {baseline[:12]}…")
    return broken, f"re-derived against {baseline[:12]}…"


def verify_document(directory: Path, seal: dict[str, Any]) -> list[str]:
    """The document must still pass its own checker, run from the checker the seal fixes."""
    directory = Path(directory)
    checker = directory / "check_prereg.py"
    document = directory / "PREREGISTRATION.md"
    for path in (checker, document):
        if not path.exists():
            return [f"{path.name} is gone; the document cannot be re-checked"]
    spec = importlib.util.spec_from_file_location("_sealed_check_prereg", checker)
    if spec is None or spec.loader is None:
        return [f"{checker} could not be loaded as a module"]
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        failures = module.check(document.read_text(encoding="utf-8"), verbose=False)
    except Exception as exc:  # the checker refusing to run is itself a refusal
        return [f"{checker.name} did not run: {type(exc).__name__}: {exc}"]
    return [f"the sealed document no longer passes its checker: {f}" for f in failures]


def require_addendum(directory: Path, *, parent: dict[str, Any], must_list: dict[str, Path]) -> dict[str, Any]:
    """The active amendment addendum, or a refusal. Called before any estimand the amendment governs.

    The circularity is real and is resolved by direction: the reader is one of the addendum's sealed
    files, so it cannot carry the addendum's digest as a constant. Instead the addendum is written
    **after** the reader is final, and the reader asks only that the addendum **lists this reader's
    own current bytes** — and the rule module's — in its file table. A reader that has drifted from
    what was sealed therefore fails to find itself, and needs no advance knowledge of any digest.

    An addendum whose `<stem>-SUPERSEDED.md` marker sits beside it is void and is refused by name,
    which is what makes marking rather than deleting safe.
    """
    directory = Path(directory)
    found = sorted(directory.glob("addendum-*.json"))
    if not found:
        raise NotSealed(
            f"no addendum-*.json in {directory}: the amendment's estimands are not read without the "
            "addendum that seals its instrument"
        )
    active = [p for p in found if not (directory / f"{p.stem}-SUPERSEDED.md").exists()]
    superseded = [p.name for p in found if p not in active]
    if not active:
        raise NotSealed(f"every addendum in {directory} is superseded: {superseded}")
    if len(active) > 1:
        raise NotSealed(f"{len(active)} addenda are active at once: {[p.name for p in active]}; "
                        "which one governs is not for a reader to choose")
    path = active[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("amends", "baseline_commit", "files"):
        if key not in payload:
            raise NotSealed(f"{path.name} is missing {key!r}; a partial addendum is no addendum")
    if payload["amends"].get("sha256") != parent["verified"]["seal_sha256"]:
        raise SealBroken(f"{path.name} amends seal {str(payload['amends'].get('sha256'))[:12]}…, "
                         f"not the one verified here")

    problems = []
    for name, target in must_list.items():
        listed = payload["files"].get(name)
        if listed is None:
            problems.append(f"{path.name} does not list {name}, so it cannot have sealed it")
        elif listed != _sha256(target):
            problems.append(f"{name} has changed since {path.name} sealed it")
    root = directory.parents[2]
    for name, listed in sorted(payload["files"].items()):
        candidate = root / name
        if not candidate.exists():
            problems.append(f"{name} is gone; {path.name} fixes it at {listed[:12]}…")
        elif _sha256(candidate) != listed:
            problems.append(f"{name} differs from the bytes {path.name} fixes")
    if problems:
        raise SealBroken(f"{path.name} no longer describes this tree; {len(problems)} disagreement(s):"
                         "\n  - " + "\n  - ".join(problems))

    return {**payload, "verified": {
        "addendum": path.name,
        "sha256": _sha256(path),
        "baseline_commit": payload["baseline_commit"],
        "files": f"{len(payload['files'])} file(s) match the bytes the addendum fixes",
        "superseded_ignored": superseded,
    }}


def require_seal(directory: Path, *, expected_digest: str | None = None) -> dict[str, Any]:
    """Refuse to read anything under `directory` unless the seal is present and still true.

    Returns the seal with a `verified` record naming what was actually checked. Raises `NotSealed`
    when there is no usable seal and `SealBroken` when the tree has moved out from under one.
    """
    directory = Path(directory)
    seal = load_seal(directory)

    actual = seal_digest(directory)
    if expected_digest is not None and actual != expected_digest:
        raise SealBroken(
            f"{SEAL_NAME} digests to {actual}, not the expected {expected_digest}: this is a "
            "different seal, and a different seal is a different study"
        )

    problems = verify_files(directory, seal)
    baseline_problems, baseline_note = verify_baseline(directory, seal)
    problems += baseline_problems
    problems += verify_document(directory, seal)
    if problems:
        raise SealBroken(
            f"the seal at {directory / SEAL_NAME} no longer describes this tree; "
            f"{len(problems)} disagreement(s):\n  - " + "\n  - ".join(problems)
        )

    return {
        **seal,
        "verified": {
            "seal_sha256": actual,
            "files": f"{len(seal['files'])} file(s) match the bytes the seal fixes",
            "baseline": baseline_note,
            "document": "passes its own checker as sealed",
        },
    }


__all__ = [
    "NotSealed", "SealBroken", "SEAL_NAME", "load_seal", "require_addendum", "require_seal",
    "seal_digest",
    "verify_baseline", "verify_document", "verify_files",
]
