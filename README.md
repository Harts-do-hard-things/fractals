# Fractals (Julia)
[![CI](https://github.com/Harts-do-hard-things/Fractals.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/Harts-do-hard-things/Fractals.jl/actions/workflows/ci.yml)

`Fractals` is a Julia package for generating and rasterizing 2D iterated function system (IFS) fractals.

Supported Julia versions: `1.11` and `1.12`.

The package includes:
- Chaos-game iteration (`chaos_game`)
- Deterministic expansion (`deterministic_iterate`)
- Forward and inverse rasterization helpers
- An IFS parser for text-based fractal definitions

## Repository Layout

- `Project.toml` (repo root): [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) scientific project `fractals` that uses the local `Fractals` package (via `[sources]`); run scripts and renders from here
- `Fractals.jl/`: Julia package (`src/` and `test/`)
- `FractalsGUI.jl/`: separate Julia package for the desktop GUI (depends on `Fractals`)
- `scripts/`: DrWatson project scripts (`@quickactivate "fractals"`): example renders, ImageIterate animation builder, GIF/MP4 export, rotation examples
- `ifs/`: curated `.ifs` input files (tracked; `ifs_data_dir()` returns this path)
- `data/sims/`: all generated data (render records, cached point clouds, fractal records, and the prebuilt IFS library `data/sims/ifs/` from `build_ifs_library()`; git-ignored; `data_output_dir()`)
- `plots/`: rendered images and animations (git-ignored, created on demand; `output_dir()`)
- `docs/`: Documenter.jl documentation site (`docs/make.jl`, pages in `docs/src/`)

The layout follows DrWatson's default project template; `notebooks/`, `papers/`, `videos/` and `_research/` are
created on demand (all but `papers/` are git-ignored, per DrWatson's `.gitignore` template).

## Quick Start

### 1. Activate and install dependencies

```powershell
julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate()"            # DrWatson project (scripts, renders)
julia --startup-file=no --project=Fractals.jl -e "using Pkg; Pkg.instantiate()"  # package development and tests
```

For GIF/MP4 export, install `ffmpeg` separately and ensure it is on `PATH`.
Common installs:

```bash
sudo apt-get install ffmpeg
brew install ffmpeg
winget install Gyan.FFmpeg
```

### 2. Generate a fractal image

```powershell
julia --startup-file=no --project=. -e "using Fractals; render(IFS(HEIGHWAY_DRAGON; name=\"Heighway Dragon\"), Chaos(npoints=200_000); resolution=(1024, 1024))"
```

This writes `plots/Heighway_Dragon_chaos_npoints=200000_1024x1024.png` (and a `.jld2` render record under `data/sims/renders/`)
(see [Output Files](#output-files-drwatson)).

### 3. Run tests

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/test/runtests.jl
```

Optional GPU parity tests can be enabled with `FRACTALS_RUN_GPU_TESTS=1` (requires CUDA support).
Committed PNG snapshot fixtures for deterministic render paths live under `Fractals.jl/test/snapshots/`; regenerate them with `julia --startup-file=no --project=Fractals.jl Fractals.jl/test/generate_snapshots.jl` when render output intentionally changes.

### GUI package

Launch GUI:

```powershell
julia --startup-file=no --project=FractalsGUI.jl FractalsGUI.jl/bin/gui.jl
```

Run GUI package tests:

```powershell
julia --startup-file=no --project=FractalsGUI.jl -e "using Pkg; Pkg.test()"
```

Repository CI now runs the non-interactive `FractalsGUI.jl` test suite on Ubuntu. The GUI launcher itself still remains a local/manual workflow.

### 4. Run formatter check

```powershell
julia --startup-file=no -e "using Pkg; Pkg.activate(temp=true); Pkg.add(name=\"JuliaFormatter\", version=\"1.0.62\"); using JuliaFormatter; ok = format([\"Fractals.jl/src\", \"Fractals.jl/test\"]; overwrite=false, verbose=true); ok || error(\"Formatting check failed\")"
```

To apply formatting locally:

```powershell
julia --startup-file=no -e "using Pkg; Pkg.activate(temp=true); Pkg.add(name=\"JuliaFormatter\", version=\"1.0.62\"); using JuliaFormatter; format([\"Fractals.jl/src\", \"Fractals.jl/test\"]; overwrite=true, verbose=true)"
```

### 5. Generate Rotation Transformation Examples

This writes 30deg/45deg clockwise and counterclockwise SVG/PNG transformation examples to `plots/rotation_examples/`:

```powershell
julia --startup-file=no --project=. scripts/generate_rotation_examples.jl
```

Rotation convention for transformation examples/tests is canonical math-space:
positive angles are counterclockwise in model coordinates (`+x` right, `+y` up).
Transformation renders also use a shared anchored model origin, so `(0,0)` maps to
the same pixel across all `initial_polygon` presets.

## Minimal Julia Example

```julia
using Fractals

ifs = IFS(EISENSTEIN; name = "Eisenstein")
result = render(ifs, Chaos(npoints = 250_000);
                resolution = (1200, 1200),
                outpath = "eisenstein.png")

@show result.outpath
```

A render is a value: `Render(ifs, method; resolution, color, alpha, overlays)`, run by `render`. The methods are `Chaos`, `PointDeterministic`, `Inverse`, `Transformations` and `ImageIterate` (whose source is a `PolygonSource`, a `FileSource`, or any other method), each validating its own parameters. `overlays` takes `BasePolygon(polygon)` and `Axes()`.
`render` returns `(render, image, limits, outpath, record)`; `load_render(path)` reads a saved render back.
`make_image`, `iterate_image`, and `rasterize_image_inversely` still take `backend=:cpu|:gpu|:auto`, but the GPU code is not wired into `render`.
`interpolate_eq_matrix(...)` and `interpolate_ifs(...)` provide transform interpolation primitives for animation-oriented workflows.
`render_interpolation_frames(left, right, method; frames, ...)` renders deterministic numbered PNG frame sequences into a chosen directory for animation workflows, with any render method.
`export_animation(:gif|:mp4; ...)` converts those numbered PNG frames into GIF or MP4 artifacts via `ffmpeg`.
Interpolation defaults to per-map rotation+uniform-scale blending for the 2x2 linear part, with automatic fallback to coefficient-wise linear blending when a map pair is not well represented by that model. Use `interpolation_mode=:linear` to force the previous behavior.
For `.ifs` files with several definitions, pick one with `get_ifs(name; file=...)` (or index `parse_ifs_file(path)`).

Reproducible chaos iteration example:

```julia
using Fractals

ifs = IFS(HEIGHWAY_DRAGON)
points = chaos_game(ifs, 100_000; warmup=20, seed=1234)
```

## IFS Parser Example

```julia
using Fractals

text = """
Heighway Dragon {
  0.5 -0.5  0.5  0.5  0.0 0.0
 -0.5 -0.5  0.5 -0.5  1.0 0.0
}
"""

defs = parse_ifs_string(text)
ifs = defs[1]
points = chaos_game(ifs, 100_000)
```
`.ifs` files live in `ifs/`; `build_ifs_library()` prebuilds one JLD2 record per function system under `data/sims/ifs/<file>/`, and `list_ifs()` / `get_ifs(name)` prefer those records over parsing the files (see the IFS text format page).

## Documentation

The documentation is a [Documenter.jl](https://documenter.juliadocs.org/stable/) site in `docs/`:

- `docs/src/index.md`: home page.
- `docs/src/manual/`: render methods, quick recipes, animation, output files, IFS text format, troubleshooting, and development notes.
- `docs/src/api.md`: API reference generated from the package docstrings.

Build it locally and open `docs/build/index.html`:

```powershell
julia --startup-file=no --project=docs -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=docs docs/make.jl
```

## Output Files (DrWatson)

All files the package writes (`render`, `render_transformations_svg/png`, `render_interpolation_frames`,
`export_animation`) are managed with [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/):

- **Location:** relative paths resolve under `plotsdir()` of the active project, so with `--project=.` output
  lands in `<repo>/plots/` regardless of the working directory. Absolute paths are used as given. Override with
  `with_output_root(dir) do ... end` or `FRACTALS_OUTPUT_ROOT=<dir>` (output then goes to `<dir>/plots/`);
  `output_dir()` shows the current target.
- **Naming:** omit `outpath` and the file is named with `savename` from the render parameters, e.g.
  `Heighway_Dragon_chaos_npoints=250000_1200x1200_color.png`, plus a same-stem `.jld2` render record
  (`load_render(path)` reads it back, so `render(load_render(path))` repeats the render). Interpolation frames go to
  `plots/frames/<savename>/` and animations to `plots/animations/`.
- **Render records:** `render` saves the PNG and a `.jld2` with the `Render` description and its limits, tagged
  with the git commit by DrWatson's `tagsave`. SVGs carry a provenance comment (`read_provenance` reads PNG
  provenance written by the animation script).

New analysis scripts belong in `scripts/` and start with `using DrWatson; @quickactivate "fractals"`
(see `scripts/render_examples.jl`).
