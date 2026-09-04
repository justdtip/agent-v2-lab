"""Thin wrapper: the J-space sweep now lives in the installed package.

The body moved to ``src/local_llm_lab/probes/jspace_sweep.py`` under EXP-001 (issue #54),
because ``research/`` is outside the wheel (``pyproject.toml`` builds
``packages = ["src/local_llm_lab"]``) and nothing under it is installable, so the sweep had no
console entry point and no way to take flags from an operator. Use the installed CLI:

    uv run agent-v2-jspace-sweep --model <name> --output outputs/probes/jspace-<model>-<date>

This wrapper keeps the historical *path* working; it does not reproduce the historical
*numbers*, and nothing here should be read as claiming it does (C5, issue #62). It now runs a
different variant of the estimator: three readouts rather than one, interior source positions
rather than the last token, a 128-token corpus, a derived kind-matched layer family rather than
a literal layer list, and rendering through the registry's template kwargs. The World A table
recorded in ``research/jspace_probe.md`` reproduces only from this script as it stood at commit
``1953493``, the parent of ``d0adfc6`` (the commit that moved the body into the package).

It loads model weights, so it runs only under a Director lift, in the single designated
execution lane.
"""

from __future__ import annotations

from local_llm_lab.probes.jspace_sweep import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
