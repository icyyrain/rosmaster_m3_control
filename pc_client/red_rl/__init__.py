"""Red-sweet learning experiments; the stable pick pipeline is unchanged.

The existing PC scripts use sibling absolute imports (``import arm_control``).
Expose their directory to preserve one copy of each module when running these
experiments with ``python -m pc_client.red_rl...`` from the repository root.
"""

import sys
from pathlib import Path

_scripts = str(Path(__file__).resolve().parent.parent)
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)
