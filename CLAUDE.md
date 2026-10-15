# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A personal research/hobby repo for generating and rendering IFS (Iterated Function System) fractals, largely based on [Larry Riddle's IFS pages](https://larryriddle.agnesscott.org/ifs/ifs.htm). There are several parallel implementations of the same idea, at different stages of maturity — a Python complex-plane version, Python matrix/affine versions, and a Julia package (the most actively developed and the only one with tests).

## Setup and commands

Python (no packaging; scripts are run directly from the directory they live in):

```bash
python -m pip install -r requirements.txt   # or: conda env create -f environment.yml
python playground.py                         # quick demo of fractal.py
```

`fractal.py` also imports `datashader`, `pandas` and `colorcet`, which are not listed in `requirements.txt`/`environment.yml`; `ifsread.py` needs `dotmap` (only in `environment.yml`). `gif` is optional.

Manim animations (`scene.py`, `rectangular_generators/scene.py`, `fern.py`):

```bash
manim -pql scene.py HeighwayDragon
```

Docs (mkdocs + mkdocstrings, built from `fractal.py` docstrings; published to GitHub Pages):

```bash
mkdocs serve
```

Julia package (`julia/Fractals`):

```bash
cd julia/Fractals && julia --project -e "using Pkg; Pkg.test()"
cd julia/Fractals && julia --project test/runtests.jl     # run tests directly
cd julia/Fractals && julia --project src/matrixfractal.jl # renders EISENSTEIN to output.png via main()
```

There is no single-test runner; to run one `@testset`, comment out the others or `include` the file in a REPL. Note `test/runtests.jl` `include`s `src/matrixfractal.jl` directly rather than `using Fractals`.

There is no linter or Python test suite configured.

## Architecture

### Two ways to represent an IFS map

1. **Complex-plane lambdas** (`fractal.py`): each map is a Python `Callable[[complex], complex]`. Maps for named fractals live in the module-level `IFS_function` dict, built from module-level constants (angles/scale factors computed with `cmath`). Each named fractal is a subclass of `Fractal` that passes an initial point list `S0` and a `func_list` to `super().__init__`, sets hard-coded `limits`, and optionally overrides `tile()` to arrange copies (e.g. KochFlake, FudgeFlake). `DragonFractal` adds `segment()` and line-based plotting; per the module comment, any fractal needing segmentation should subclass `DragonFractal`. Iteration is deterministic: every map is applied to every point (`len(func_list)**i` growth), so ~15+ iterations can exhaust memory.

2. **Affine coefficient rows** (everything else): each map is a row `a b c d e f [p]` — a 2×2 linear part, a translation, and an optional probability. If `p` is missing, weights are derived from `|det(A)|`. This is the Fractint `.ifs` format used in `rectangular_generators/ifs_data/*.ifs` (`Name { rows... }`, `;` comments, `(3D)` entries skipped).
   - `arrayfractal.py` — `ArrayFractal`: numpy version supporting both deterministic iteration and chaos-game (`random_iterate`), plus `make_image` (PIL). `ArrayFractal.from_imaginary` converts a `fractal.py`-style complex lambda list into coefficient rows, bridging the two representations.
   - `rectangular_generators/matrixfractal.py` — older `FunctionSystem` / `FunctionSystemRandom` classes.
   - `rectangular_generators/ifsread.py` — PLY lexer/parser that reads `.ifs` files and dynamically creates `FunctionSystemRandom` subclasses into a global `ifs` DotMap keyed `IFS_<Name>`. It parses `ifs_data\Edgar.ifs` at import time with a relative Windows path, so scripts importing it must be run from `rectangular_generators/`. `parsetab.py`/`parser.out` are PLY-generated.

### Julia package (`julia/Fractals`)

- `src/matrixfractal.jl` — core: `AffineMap` (StaticArrays-based, invertible), `IFS` struct, `build_maps_and_weights` (same row format as above), chaos game (`iterate!`, `iterate_parallel!`), `deterministic_iterate`, rasterization (`make_image`, pixel maps), inverse iteration / inverse rasterization, and SVG/PNG rendering of transformations. Constants `RESOLUTION`, `DEFAULT_WARMUP`, `DEFAULT_SAMPLES`, and sample systems `HEIGHWAY_DRAGON`, `EISENSTEIN`.
- `src/ifsparser.jl` — a Julia port of the `.ifs` lexer/parser (`lex_ifs` → `parse_ifs_tokens` → `IFSDefinition`) plus `prompt_ifs_and_render`, an interactive (stdin) chooser.
- `src/Fractals.jl` — module wrapper; new public functions must be added to its `export` list.

### Other directories

- `experimental/` — unfinished explorations (Cython port, L-systems); not imported elsewhere.
- `GIFs/`, `assets/images/`, `julia/Fractals/images/` — rendered outputs; `media/` (manim output) is gitignored.
