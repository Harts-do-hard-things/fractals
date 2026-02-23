# Fractals

Constructs IFS fractals in python and plots them in the complex plane.

## Installation
1. Clone the repo
2. Install as a package:

```
python -m pip install -e .
```

Optionally install gif package for gif creation

## Usage
```python
from fractals import HeighwayDragon

fractal = HeighwayDragon() # See full list of fractals in the documentation
fractal.iterate(15)
fractal.plot()
```

## Array Output Modes
`ArrayFractal` now has explicit output-mode rules:

- Deterministic mode (`iterate`, `divided_iterate`)
  - Use `save_svg(...)` for vector output.
  - `save_gif(...)` uses per-frame SVG generation for segmented deterministic fractals.
  - `make_image(...)` / `plot(...)` are available for raster previews.

- Chaos-game mode (`random_iterate`)
  - `save_image(...)` is allowed only after `random_iterate(...)`.
  - `save_image(...)` raises if called after deterministic iteration.

This separation avoids mixing line-segment deterministic geometry with point-cloud chaos output workflows.

## Media Scripts
Project media generation scripts live in `scripts/`:

- `scripts/generate_arrayfractal_media.py`
  - Generates ArrayFractal-specific reference outputs.
- `scripts/generate_core_array_paths_media.py`
  - Generates output-path coverage artifacts for both `core.py` and `array.py`.

Both scripts write to `media/` (ignored by git).

## Documentation
Documentation can be found [here](https://harts-do-hard-things.github.io/fractals/)

## Contrubuting

Feel Free to contribute

## TODO
- [ ] Finish Docstrings for last few Classes
- [ ] Add pictures in the documentation
- [ ] Finalize animated fractals
- [ ] Debug matrix ifs funtions
- [ ] Update and finalize documentation
- [ ] Work on the collage problem
