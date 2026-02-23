from __future__ import annotations
"""Generate ArrayFractal reference media artifacts.

This script exercises deterministic and chaos-game output paths in
`src/fractals/array.py` and writes outputs to `media/`.

Important mode constraints reflected here:
- deterministic outputs use `save_svg(...)`, `plot(...).save(...)`, or `make_image(...)`
- chaos-game outputs use `random_iterate(...)` before `save_image(...)`
"""

import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from fractals.array import ArrayFractal


MEDIA_DIR = ROOT / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


EQ_EVEN = np.array(
    [
        [0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.5, 0.5, 0.0],
    ],
    dtype=np.float64,
)

EQ_ODD = np.array(
    [
        [0.85, 0.04, -0.04, 0.85, 0.0, 1.6, 0.85],
        [0.2, -0.26, 0.23, 0.22, 0.0, 1.6, 0.07],
        [-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07],
        [0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.01],
    ],
    dtype=np.float64,
)


class OffsetTileFractal(ArrayFractal):
    def tile(self):
        self.translate(1 + 0j, 0.0)
        self.translate(0 + 1j, 0.0)


def save_gif_with_name(fractal: ArrayFractal, iterations: int, duration: int, resolution: tuple[int, int], filename: str):
    old_cwd = Path.cwd()
    try:
        os.chdir(MEDIA_DIR)
        fractal.save_gif(iterations=iterations, duration=duration, resolution=resolution)
        generated = MEDIA_DIR / f"{type(fractal).__name__}_{iterations}.gif"
        target = MEDIA_DIR / filename
        if generated.exists():
            generated.replace(target)
    finally:
        os.chdir(old_cwd)


def main():
    # 1) deterministic iterate using default initial polygon line for point seed
    a = ArrayFractal(np.array([[0.0, 0.0]]), EQ_EVEN, segment_plot=True)
    a.iterate(9)
    a.save_svg(str(MEDIA_DIR / "deterministic_default_initial_line_segmented.svg"), resolution=(1400, 1400))
    a.plot(autoscale=False, resolution=(1400, 1400)).save(
        MEDIA_DIR / "deterministic_default_initial_line_segmented.png"
    )

    # 2) deterministic with explicitly provided polygon seed
    b = ArrayFractal(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.7]]),
        EQ_EVEN,
        segment_plot=True,
        requires_initial_polygon=False,
    )
    b.iterate(8)
    b.save_svg(str(MEDIA_DIR / "deterministic_explicit_polygon_seed_segmented.svg"), resolution=(1400, 1400))
    b.plot(autoscale=False, resolution=(1400, 1400)).save(
        MEDIA_DIR / "deterministic_explicit_polygon_seed_segmented.png"
    )

    # 3) deterministic with custom initial polygon override
    c = ArrayFractal(np.array([[0.0, 0.0]]), EQ_EVEN, segment_plot=True)
    c.set_initial_polygon(np.array([[0.0, 0.0], [0.5, 0.9], [1.0, 0.0]], dtype=np.float64))
    c.iterate(8)
    c.save_svg(str(MEDIA_DIR / "deterministic_custom_initial_polygon_triangle.svg"), resolution=(1400, 1400))
    c.plot(autoscale=False, resolution=(1400, 1400)).save(
        MEDIA_DIR / "deterministic_custom_initial_polygon_triangle.png"
    )

    # 4) divided deterministic path
    d = ArrayFractal(np.array([[0.0, 0.0]]), EQ_EVEN, segment_plot=True)
    d.divided_iterate(8)
    d.save_svg(str(MEDIA_DIR / "deterministic_divided_iterate_blocks.svg"), resolution=(1400, 1400))
    d.plot(autoscale=False, resolution=(1400, 1400)).save(MEDIA_DIR / "deterministic_divided_iterate_blocks.png")

    # 5) random iterate path with transform-based coloring branch
    e = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), EQ_ODD, segment_plot=False, requires_initial_polygon=False)
    e.random_iterate(250_000)
    e.make_image(resolution=(1600, 1600)).save(MEDIA_DIR / "random_iterate_color_by_transform.png")

    # 6) plot path autoscale=False
    f = ArrayFractal(np.array([[0.0, 0.0]]), EQ_EVEN, segment_plot=True)
    f.iterate(8)
    f.plot(autoscale=False, resolution=(1400, 1400)).save(MEDIA_DIR / "plot_autoscale_false.png")

    # 7) plot path autoscale=True
    g = ArrayFractal(np.array([[0.0, 0.0]]), EQ_EVEN, segment_plot=True)
    g.iterate(8)
    g.plot(autoscale=True, resolution=(1400, 1400)).save(MEDIA_DIR / "plot_autoscale_true.png")

    # 8) transform methods path (translate, translate_in_place, scale)
    h = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), EQ_EVEN, segment_plot=True, requires_initial_polygon=False)
    h.iterate(7)
    h.translate(1 + 0j, 0.0)
    h.translate_in_place(0 + 0.5j, 0.0)
    h.scale(0.8)
    h.make_image(resolution=(1400, 1400)).save(MEDIA_DIR / "transforms_translate_translateinplace_scale.png")

    # 9) make_image background branch
    i = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), EQ_ODD, segment_plot=False, requires_initial_polygon=False)
    i.random_iterate(180_000)
    i.save_image(str(MEDIA_DIR / "save_image_chaos_game_only.png"), resolution=(1400, 1400))
    i.make_image(resolution=(1400, 1400), background=(10, 10, 20, 255)).save(
        MEDIA_DIR / "make_image_custom_background.png"
    )

    # 10) tile() hook via subclass during save_image/plot
    j = OffsetTileFractal(np.array([[0.0, 0.0]]), EQ_EVEN, segment_plot=True)
    j.iterate(7)
    j.save_svg(str(MEDIA_DIR / "tile_hook_multiple_translations.svg"), resolution=(1400, 1400))
    j.plot(autoscale=False, resolution=(1400, 1400)).save(MEDIA_DIR / "tile_hook_multiple_translations.png")

    # 11) save_gif path with deterministic segmented iteration
    k = ArrayFractal(np.array([[0.0, 0.0]]), EQ_EVEN, segment_plot=True)
    k.iterate(3)
    save_gif_with_name(
        k,
        iterations=5,
        duration=120,
        resolution=(700, 700),
        filename="save_gif_deterministic_segmented.gif",
    )

    # 12) save_gif path with random seed + non-segment rendering
    l = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), EQ_ODD, segment_plot=False, requires_initial_polygon=False)
    l.random_iterate(60_000)
    save_gif_with_name(
        l,
        iterations=4,
        duration=120,
        resolution=(700, 700),
        filename="save_gif_random_colored_points.gif",
    )

    print(f"Generated media artifacts in: {MEDIA_DIR}")


if __name__ == "__main__":
    main()
