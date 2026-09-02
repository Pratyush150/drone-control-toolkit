"""Shared plumbing for the example scripts.

matplotlib is imported here and nowhere in ``src/dctk``. Importing it lazily
inside :func:`figure` keeps the library importable -- and the test suite
runnable -- on a headless machine with no plotting stack installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def figure(*args: Any, **kwargs: Any):
    """Create a matplotlib figure with a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")  # never needs a display
    import matplotlib.pyplot as plt

    return plt.subplots(*args, **kwargs)


def save(fig, name: str) -> Path:
    """Write ``fig`` to ``examples/output/<name>`` and return the path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")
    return path


def header(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def table(rows: list[tuple[str, str]]) -> None:
    """Print aligned label/value pairs."""
    width = max(len(r[0]) for r in rows) if rows else 0
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")
