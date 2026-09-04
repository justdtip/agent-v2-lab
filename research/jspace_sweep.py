"""Thin wrapper: the J-space sweep now lives in the installed package.

The body moved to ``src/local_llm_lab/probes/jspace_sweep.py`` under EXP-001 (issue #54),
because ``research/`` is outside the wheel (``pyproject.toml`` builds
``packages = ["src/local_llm_lab"]``) and nothing under it is installable, so the sweep had no
console entry point and no way to take flags from an operator. Use the installed CLI:

    uv run agent-v2-jspace-sweep --model <name> --output outputs/probes/jspace-<model>-<date>

This wrapper exists only so the historical path keeps working. It loads model weights, so it
runs only under a Director lift, in the single designated execution lane.
"""

from __future__ import annotations

from local_llm_lab.probes.jspace_sweep import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
