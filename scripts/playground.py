"""Simple demo script for the fractals package."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fractals import Pentadendrite


def main():
    dragon = Pentadendrite()
    dragon.iterate(8)
    dragon.plot()


if __name__ == "__main__":
    main()
