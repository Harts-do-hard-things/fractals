# -*- coding: utf-8 -*-
"""Demo for rectangular generators loaded from IFS parser."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fractals.ifs.parser import ifs


if __name__ == "__main__":
    print("Available rectangular generators:")
    for fractal in ifs:
        print(fractal)
    h = ifs.IFS_Eisenstein(run_prob=False)
    h.iterate(1_000_000)
    image = h.make_image()
    image.save("Eisenstein.png")
