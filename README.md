# Fractals (Julia)

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
julia --project=Fractals Fractals/test/runtests.jl
```

## Minimal Julia Example

```julia
using Fractals
using FileIO

result = render(EISENSTEIN;
                method=:chaos,
                npoints=250_000,
                resolution=(1200, 1200),
                outpath="media/eisenstein.png")

@show result.outpath
```

For `method=:deterministic` and `method=:inverse`, use `iterations=...` to control iteration depth.
For `.ifs` files with multiple definitions, select one with `ifs_index=...` or `ifs_name=...`.

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

Detailed package documentation is in `Fractals/DOCUMENTATION.md`.

Generated images should be saved under `media/`.
