# Fractals.jl Documentation

## Overview

`Fractals` provides tools to define, iterate, and rasterize 2D affine IFS fractals.

Core source files:
- `Fractals/src/matrixfractal.jl`: main types and rendering functions
- `Fractals/src/ifsparser.jl`: parser for text-based IFS definitions

## Installation

From repository root:

```powershell
julia --project=Fractals -e "using Pkg; Pkg.instantiate()"
```

## Core Types

### `AffineMap`

Represents an affine map `x -> A*x + b` in 2D.

Constructors:
- `AffineMap(A::AbstractMatrix, b::AbstractVector)`
- `AffineMap(a11, a12, a21, a22, b1, b2)`

Operations:
- call overload: `m(x)`
- inverse map: `inv(m)`

### `IFS`

Represents one fractal system:
- `name`, `docs`
- `points`: sampled point cloud
- `maps`: affine maps
- `weights`: map probabilities
- `limits`: plotting/raster bounds

Constructors:
- `IFS(eq::AbstractMatrix; npoints, name, docs)`
- `IFS(maps, weights; npoints, name, docs, limits)`

`eq` rows are `[a11 a12 a21 a22 b1 b2]` and optional probability `p` as 7th value.

## Iteration Methods

### `iterate!(ifs; warmup=DEFAULT_WARMUP)`

Chaos-game iteration (single-threaded). Updates `ifs.points` in place.

### `iterate_parallel!(ifs; warmup=DEFAULT_WARMUP)`

Threaded chaos-game iteration. Updates `ifs.points` in place.

### `deterministic_iterate(ifs, n)`

Applies every map to every point for `n` rounds. Returns a new `IFS`.

Warning:
- Point count grows as `length(points) * length(maps)^n`
- Large `n` can allocate huge arrays

## Rasterization

### `make_image(ifs; resolution=(rows, cols))`

Rasterizes `ifs.points` into a normalized `Float32` image in `[0, 1]`.

### `iterate_image(ifs, img)`

Applies all IFS maps to an image and returns a grayscale image.

### `rasterize_image_inversely(ifs, n, limits; resolution=RESOLUTION)`

Inverse method that samples coverage via inverse map recursion.

## Parser API

Parser entry points:
- `parse_ifs_string(input; npoints=DEFAULT_SAMPLES)`
- `parse_ifs_file(path; npoints=DEFAULT_SAMPLES)`
- `lex_ifs(input)`

Interactive helper:
- `prompt_ifs_and_render(path; npoints, resolution, outpath)`

### IFS Text Format

Example:

```text
Example Fractal {
; Optional docs line
  0.5  0.0  0.0  0.5  0.0  0.0  0.6
 -0.5  0.0  0.0 -0.5  1.0  0.0  0.4
}
```

Rules:
- A block starts with `<name> {` and ends with `}`
- Lines starting with `;` are docs/comments
- Numeric rows must all have matching width
- Accepted row widths are 6 (no explicit probability) or 7 (with probability)
- Lines containing `(3D)` are ignored

## Exported Constants

- `RESOLUTION`
- `DEFAULT_WARMUP`
- `DEFAULT_SAMPLES`
- `HEIGHWAY_DRAGON`
- `EISENSTEIN`

## Common Workflows

### Basic render

```julia
using Fractals, FileIO

ifs = IFS(HEIGHWAY_DRAGON; npoints=200_000)
iterate!(ifs)
img = make_image(ifs; resolution=(1024, 1024))
save("media/dragon.png", img)
```

### Deterministic preview

```julia
using Fractals, FileIO

ifs = IFS(EISENSTEIN; npoints=200)
expanded = deterministic_iterate(ifs, 2)
img = make_image(expanded; resolution=(800, 800))
save("media/expanded.png", img)
```

### Parse from file and render interactively

```julia
using Fractals
prompt_ifs_and_render("my_fractals.ifs")
```

## Testing

Run from repository root:

```powershell
julia --project=Fractals Fractals/test/runtests.jl
```
