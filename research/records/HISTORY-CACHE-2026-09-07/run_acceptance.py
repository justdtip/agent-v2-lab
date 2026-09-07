"""Run isolated source with the primary checkout's existing process-scoped model lock.

This changes only the root used by the normal guard, not its acquisition, process checks,
refusal, or release behavior. All results stay in this worktree. Requires the queue to have
finished before invocation; never removes another holder's lock or stops any process.
"""

import os
import runpy
from pathlib import Path

from local_llm_lab import runlock

PRIMARY = Path("/Users/daniel.tipton/Desktop/An app")
runlock.PROJECT_ROOT = PRIMARY
os.environ["HF_HOME"] = str(PRIMARY / ".cache/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MODEL_RUN_SESSION"] = "codex-history-cache-acceptance"
runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "cache_equivalence.py"), run_name="__main__"
)
