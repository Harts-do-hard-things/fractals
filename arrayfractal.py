"""Compatibility wrapper for `fractals.array.ArrayFractal`."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fractals.array import ArrayFractal

__all__ = ["ArrayFractal"]
