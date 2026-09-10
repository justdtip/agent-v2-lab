"""Run the file-only device SAE/J bridge; see --help for explicit inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_llm_lab.probes.device_bridge import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
