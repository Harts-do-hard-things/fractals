"""Compatibility wrapper for rectangular IFS playground."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.rectangular_playground import *  # noqa: F401,F403
