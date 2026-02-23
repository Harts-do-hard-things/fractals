from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from fractals import HeighwayDragon, LevyC
from fractals.array import ArrayFractal
from fractals.core import BinaryTree


MEDIA = ROOT / "media"
MEDIA.mkdir(parents=True, exist_ok=True)


def safe_show():
    # Prevent blocking in non-interactive runs.
    return None


plt.show = safe_show  # type: ignore[assignment]


def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def save_core_outputs(label: str, fractal, iterations: int = 8) -> list[Path]:
    out: list[Path] = []
    prefix = sanitize(label)

    fractal.reset()
    fractal.iterate(iterations)
    fractal.plot()
    png = MEDIA / f"{prefix}_plot_fig_core_py.png"
    fractal.plot_fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fractal.plot_fig)
    out.append(png)

    if hasattr(fractal, "save_gif"):
        old = Path.cwd()
        try:
            os.chdir(MEDIA)
            gif_name = f"{type(fractal).__name__}_{max(2, iterations // 2)}.gif"
            fractal.reset()
            fractal.iterate(2)
            fractal.save_gif(max(2, iterations // 2), duration=120)
            src = MEDIA / gif_name
            dst = MEDIA / f"{prefix}_save_gif_core_py.gif"
            if src.exists():
                src.replace(dst)
                out.append(dst)
        finally:
            os.chdir(old)
    return out


def make_array_from_core(fractal) -> ArrayFractal:
    s0 = np.asarray(fractal._S0, dtype=np.complex128).tolist()
    return ArrayFractal.from_imaginary(s0, fractal.func_list)


def save_array_outputs(label: str, array_fractal: ArrayFractal, iterations: int = 8) -> list[Path]:
    out: list[Path] = []
    prefix = sanitize(label)

    # Path 1: save_image (chaos-game only).
    array_fractal.reset()
    array_fractal.random_iterate(120_000)
    p1 = MEDIA / f"{prefix}_save_image_chaos_array_py.png"
    array_fractal.save_image(str(p1), resolution=(1400, 1400))
    out.append(p1)

    # Path 2: save_svg direct (deterministic segmented output).
    array_fractal.reset()
    array_fractal.iterate(iterations)
    p2 = MEDIA / f"{prefix}_save_svg_direct_array_py.svg"
    array_fractal.save_svg(str(p2), resolution=(1400, 1400))
    out.append(p2)

    # Path 3: make_image(...).save(...) deterministic.
    array_fractal.reset()
    array_fractal.iterate(iterations)
    p3 = MEDIA / f"{prefix}_make_image_save_array_py.png"
    array_fractal.make_image(resolution=(1400, 1400)).save(p3)
    out.append(p3)

    # Path 4: plot(...).save(...) deterministic.
    array_fractal.reset()
    array_fractal.iterate(iterations)
    p4 = MEDIA / f"{prefix}_plot_return_save_array_py.png"
    array_fractal.plot(autoscale=True, resolution=(1400, 1400)).save(p4)
    out.append(p4)

    # Path 5: save_gif(...)
    old = Path.cwd()
    try:
        os.chdir(MEDIA)
        array_fractal.reset()
        array_fractal.iterate(2)
        array_fractal.save_gif(
            5,
            duration=120,
            resolution=(700, 700),
            deterministic_png_resolution=(3840, 2160),
        )
        src = MEDIA / f"{type(array_fractal).__name__}_5.gif"
        p5 = MEDIA / f"{prefix}_save_gif_array_py.gif"
        if src.exists():
            src.replace(p5)
            out.append(p5)
    finally:
        os.chdir(old)

    return out


def main() -> None:
    generated: list[Path] = []

    heighway = HeighwayDragon()
    generated.extend(save_core_outputs("HeighwayDragon", heighway, iterations=10))
    generated.extend(save_array_outputs("HeighwayDragon", make_array_from_core(heighway), iterations=10))

    levy = LevyC()
    generated.extend(save_core_outputs("LevyCCurve", levy, iterations=10))
    generated.extend(save_array_outputs("LevyCCurve", make_array_from_core(levy), iterations=10))

    tree = BinaryTree(0.618, np.pi * 0.8)
    generated.extend(save_core_outputs("BinaryTree", tree, iterations=9))
    generated.extend(save_array_outputs("BinaryTree", make_array_from_core(tree), iterations=9))

    print("Generated files:")
    for path in generated:
        print(path.name)


if __name__ == "__main__":
    main()
