"""Calibrate or execute the bounded, frozen agentic-lens pilot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_llm_lab.pipeline.lens_fitting.agentic import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
