# Fractals (Julia)
[![CI](https://github.com/Harts-do-hard-things/Fractals.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/Harts-do-hard-things/Fractals.jl/actions/workflows/ci.yml)

`Fractals` is a Julia package for generating and rasterizing 2D iterated function system (IFS) fractals.

Supported Julia versions: `1.11` and `1.12`.

The package includes:
- Chaos-game iteration (`iterate!`, `iterate_parallel!`)
- Deterministic expansion (`deterministic_iterate`)
- Forward and inverse rasterization helpers
- An IFS parser for text-based fractal definitions

## Repository Layout

- `Project.toml` (repo root): [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) scientific project `fractals` that uses the local `Fractals` package (via `[sources]`); run scripts and renders from here
- `Fractals.jl/`: Julia package (`src/`, `test/`, `Project.toml`)
- `FractalsGUI.jl/`: separate Julia package for the desktop GUI (depends on `Fractals`)
- `scripts/`: DrWatson project scripts (`@quickactivate "fractals"`)
- `plots/`: rendered output (git-ignored, created on demand)
- `site/`: generated static docs site artifacts

## Quick Start

### 1. Activate and install dependencies

```powershell
julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate()"            # DrWatson project (scripts, CLI, renders)
julia --startup-file=no --project=Fractals.jl -e "using Pkg; Pkg.instantiate()"  # package development and tests
```

### 2. Generate a fractal image

```powershell
julia --startup-file=no --project=. -e "using Fractals; render(HEIGHWAY_DRAGON; npoints=200_000, resolution=(1024, 1024))"
```

This writes `plots/ifs_method=chaos_npoints=200000_resolution=1024x1024.png`
(see [Output Files](#output-files-drwatson)).

### 3. Run tests

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/test/runtests.jl
```

Optional GPU parity tests can be enabled with `FRACTALS_RUN_GPU_TESTS=1` (requires CUDA support).
Optional CLI/benchmark integration tests can be enabled with `FRACTALS_RUN_CLI_BENCH_TESTS=1`.
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
julia --startup-file=no --project=. Fractals.jl/bin/generate_rotation_examples.jl
```

Rotation convention for transformation examples/tests is canonical math-space:
positive angles are counterclockwise in model coordinates (`+x` right, `+y` up).
Transformation renders also use a shared anchored model origin, so `(0,0)` maps to
the same pixel across all `initial_polygon` presets.

## Minimal Julia Example

```julia
using Fractals
using FileIO

result = render(EISENSTEIN;
                method=Chaos,
                npoints=250_000,
                resolution=(1200, 1200),
                outpath="eisenstein.png")

@show result.outpath
```

`render` accepts method as enum (`Chaos`, `Parallel`, `PointDeterministic`, `ImageIterate`, `Inverse`, `RenderTransformations`), symbol, or lowercase string.
`render`, `make_image`, `iterate_image`, and `rasterize_image_inversely` support `backend=:cpu|:gpu|:auto` (`:gpu` requires CUDA support, `:auto` falls back to CPU when unavailable).
`rasterize_image_inversely` also supports `mode=:exact|:preview` (`:exact` is default and parity-oriented).
`interpolate_eq_matrix(...)` and `interpolate_ifs(...)` provide transform interpolation primitives for animation-oriented workflows.
`render_interpolation_frames(...)` renders deterministic numbered PNG frame sequences into a chosen directory for transformation-preview animation workflows.
`export_animation(:gif|:mp4; ...)` converts those numbered PNG frames into GIF or MP4 artifacts via `ffmpeg`.
For `method=:point_deterministic` and `method=:inverse`, use `iterations=...` to control iteration depth.
For `.ifs` files with multiple definitions, select one with `ifs_index=...` or `ifs_name=...`.

Reproducible chaos iteration example:

```julia
using Fractals

ifs = IFS(HEIGHWAY_DRAGON; npoints=100_000)
iterate!(ifs; warmup=20, seed=1234)
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

defs = parse_ifs_string(text; npoints=100_000)
ifs = defs[1]
iterate!(ifs)
```

## Documentation

Docs-site source is in `docs/` with MkDocs config in `mkdocs.yml`.
Quick recipes page: `docs/quick-recipes.md`.
CLI usage page: `docs/cli.md`.
Benchmark usage page: `docs/benchmarks.md`.
Troubleshooting notes: `docs/troubleshooting.md`.
Detailed package reference is in `Fractals.jl/DOCUMENTATION.md`.
Benchmark target envelopes are versioned in `Fractals.jl/bench/perf_targets.toml` (latency + allocation thresholds).

## Output Files (DrWatson)

All files the package writes (`render*`, `render_transformations_svg/png`, `render_interpolation_frames`,
`export_animation`, the CLI) are managed with [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/):

- **Location:** relative paths resolve under `plotsdir()` of the active project, so with `--project=.` output
  lands in `<repo>/plots/` regardless of the working directory. Absolute paths are used as given. Override with
  `with_output_root(dir) do ... end` or `FRACTALS_OUTPUT_ROOT=<dir>` (output then goes to `<dir>/plots/`);
  `output_dir()` shows the current target.
- **Naming:** omit `outpath` and the file is named with `savename` from the render parameters, e.g.
  `Heighway_Dragon_color=true_method=chaos_npoints=250000_resolution=1200x1200.png`. Interpolation frames go to
  `plots/frames/<savename>/` and animations to `plots/animations/`.
- **No overwrites:** saves use `safesave`; an existing file is kept as `name_#1.png` (`#2`, ... for older ones).
- **Provenance:** PNGs embed a `fractals.provenance` text chunk (git commit via `gitdescribe`, Fractals version,
  timestamp, full render parameters) next to the source-limits chunk; SVGs carry it as a leading XML comment.
  Read it back with `read_provenance(path)`.

New analysis scripts belong in `scripts/` and start with `using DrWatson; @quickactivate "fractals"`
(see `scripts/render_examples.jl`).
