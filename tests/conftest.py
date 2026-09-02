"""Make ``src/`` importable without an install step.

Tests must run with a bare ``python3 -m pytest`` in a fresh checkout, so the
source directory is put on ``sys.path`` here rather than relying on an editable
install.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
