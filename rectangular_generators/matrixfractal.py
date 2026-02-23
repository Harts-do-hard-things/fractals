"""Compatibility wrapper for `fractals.ifs.matrix`."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fractals.ifs.matrix import *  # noqa: F401,F403
