"""Put THIS worktree's sources first, ahead of any installed copy of the same package.

The shared virtual environment on the rented card has `local_llm_lab` installed from a different
checkout, so `pytest` there resolved the package name to that one and could not collect a single
introspection test -- while the scripts themselves worked, because each inserts its own `src` at
import time. The difference made the card look like it had a broken test suite when it had a
shadowed one.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
for _p in (str(_root / "src"), str(_root / "scripts")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
