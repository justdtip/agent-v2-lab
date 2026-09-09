"""Assert the suite is checking the tree it was collected from.

The shared venv installs `local_llm_lab` from the main checkout, so a worktree that does not set
`PYTHONPATH` to its own `src/` imports **main's** package while running its own tests. New modules
then fail to import, which is loud; a *changed* module silently resolves to main's version, which
is not. A run like that reports on code nobody edited and passes.

This is a guard rather than a note in a record, because the failure it catches produces green tests
in the wrong tree and no other check in the suite would notice.
"""

from __future__ import annotations

from pathlib import Path


def test_the_package_under_test_comes_from_this_checkout() -> None:
    import local_llm_lab

    checkout = Path(__file__).resolve().parents[1]
    imported = Path(local_llm_lab.__file__).resolve()
    assert imported.is_relative_to(checkout / "src"), (
        f"tests collected from {checkout} are exercising {imported}.\n"
        f"Set PYTHONPATH={checkout / 'src'} for this run; the venv's install points elsewhere."
    )
