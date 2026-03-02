# Fractals (Julia)
[![CI](https://github.com/Harts-do-hard-things/fractals/actions/workflows/ci.yml/badge.svg)](https://github.com/Harts-do-hard-things/fractals/actions/workflows/ci.yml)

`Fractals` is a Julia package for generating and rasterizing 2D iterated function system (IFS) fractals.

The package includes:
- Chaos-game iteration (`iterate!`, `iterate_parallel!`)
- Deterministic expansion (`deterministic_iterate`)
- Forward and inverse rasterization helpers
- An IFS parser for text-based fractal definitions

## Repository Layout

- `Fractals/`: Julia package (`src/`, `test/`, `Project.toml`)
- `site/`: generated static docs site artifacts

## Quick Start

### 1. Activate and install dependencies

```powershell
julia --project=Fractals -e "using Pkg; Pkg.instantiate()"
```

### 2. Generate a fractal image

```powershell
julia --project=Fractals -e "using Fractals, FileIO; ifs = IFS(HEIGHWAY_DRAGON; npoints=200_000); iterate!(ifs); img = make_image(ifs; resolution=(1024, 1024)); save(\"media/heighway.png\", img)"
```

### 3. Run tests

```powershell
julia --startup-file=no --project=Fractals Fractals/test/runtests.jl
```

### 4. Run formatter check

```powershell
julia --startup-file=no -e "using Pkg; Pkg.activate(temp=true); Pkg.add(name=\"JuliaFormatter\", version=\"1\"); using JuliaFormatter; ok = format([\"Fractals/src\", \"Fractals/test\"]; overwrite=false, verbose=true); ok || error(\"Formatting check failed\")"
```

To apply formatting locally:

```powershell
julia --startup-file=no -e "using Pkg; Pkg.activate(temp=true); Pkg.add(name=\"JuliaFormatter\", version=\"1\"); using JuliaFormatter; format([\"Fractals/src\", \"Fractals/test\"]; overwrite=true, verbose=true)"
```

## Minimal Julia Example

```julia
using Fractals
using FileIO

result = render(EISENSTEIN;
                method=Chaos,
                npoints=250_000,
                resolution=(1200, 1200),
                outpath="media/eisenstein.png")

@show result.outpath
```

`render` accepts method as enum (`Chaos`, `Parallel`, `PointDeterministic`, `ImageIterate`, `Inverse`), symbol, or lowercase string.
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
Detailed package reference is in `Fractals/DOCUMENTATION.md`.
Benchmark target envelopes are versioned in `Fractals/bench/perf_targets.toml` (latency + allocation thresholds).

Generated images should be saved under `media/`.
