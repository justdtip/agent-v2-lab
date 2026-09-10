"""Measure exact-cell WS-D path pairings; all paths and layers are explicit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_llm_lab.probes.measure_pairings import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
