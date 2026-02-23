"""Compatibility wrapper for the `fractals` package.

Prefer importing from `fractals` directly.
"""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fractals.core import *  # noqa: F401,F403
from fractals.presets import *  # noqa: F401,F403
