# Fractals.jl Documentation

## Overview

`Fractals` provides tools to define, iterate, and rasterize 2D affine IFS fractals.

Core source files:
- `Fractals/src/matrixfractal.jl`: main types and rendering functions
- `Fractals/src/ifsparser.jl`: parser for text-based IFS definitions

Docs-site sources:
- `mkdocs.yml`
- `docs/index.md`
- `docs/quick-recipes.md`
- `docs/cli.md`
- `docs/troubleshooting.md`

## Installation

From repository root:

```powershell
julia --project=Fractals -e "using Pkg; Pkg.instantiate()"
```

## Common Workflows

### High-level render API

```julia
using Fractals

result = render(HEIGHWAY_DRAGON;
                method=Chaos,
                npoints=200_000,
                resolution=(1024, 1024),
                outpath="media/render.png")

println(result.outpath)
```

Method options (all accepted by `render`):
- enum values (preferred): `Chaos`, `Parallel`, `Deterministic`, `Inverse`
- symbols: `:chaos`, `:parallel`, `:deterministic`, `:inverse`
- strings: `"chaos"`, `"parallel"`, `"deterministic"`, `"inverse"`

Dispatch behavior:
- `Parallel`/`:parallel` is treated as `Chaos` because `iterate!` is auto-threaded.

Iteration control:
- Use `iterations=...` for both `:deterministic` and `:inverse`.
- Backward-compatible aliases still work:
  - `deterministic_depth=...`
  - `inverse_depth=...`
- If `iterations` is provided, it takes precedence over depth aliases.
- `deterministic_depth` and `inverse_depth` are compatibility aliases; prefer `iterations`.
- `iterations` must be `>= 0`.
- `npoints` behavior:
  - For matrix/string/file inputs, omitted `npoints` defaults to `DEFAULT_SAMPLES`.
  - For `IFS` input, omitted `npoints` keeps the existing point count.

Selecting an IFS from `.ifs` files with multiple definitions:
- Use `ifs_index=...` or `ifs_name=...` (provide only one).
- If the provided index or name does not exist, `render(...)` throws `ArgumentError` and includes the available index/name list.
- If no selector is provided and the input has multiple definitions, `render(...)` prints available definitions and asks for confirmation to render index `1`.
- If confirmation is declined (or unavailable), `render(...)` throws `ArgumentError` with the available list.

Examples:

```julia
using Fractals

# Deterministic: apply maps for 2 rounds
out1 = render(EISENSTEIN; method=:deterministic, iterations=2, npoints=500)

# Inverse: run inverse rasterization for 6 rounds
out2 = render(EISENSTEIN; method=:inverse, iterations=6, resolution=(800, 800))
```

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

### `iterate!(ifs; warmup=DEFAULT_WARMUP, seed=nothing)`

Chaos-game iteration. Automatically uses threads when available (`Threads.nthreads() > 1`), otherwise runs single-threaded. Updates `ifs.points` in place.

Reproducibility:
- Set `seed` to an integer for deterministic sampling.
- Deterministic output is guaranteed for a fixed thread count and fixed inputs.

### `iterate_parallel!(ifs; warmup=DEFAULT_WARMUP, seed=nothing)`

Compatibility alias for `iterate!`. Kept for older call sites.

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

## Validation And Errors

Input validation is enforced for matrix-based IFS construction (`IFS(eq; ...)`):
- `eq` must have at least one row.
- `eq` must have exactly 6 or 7 columns.
- All entries must be finite (`NaN`/`Inf` are rejected).
- If a 7th probability column is present:
  - all probabilities must be nonnegative
  - total probability weight must be positive

Map-indexed operations validate bounds explicitly:
- Invalid map indices throw `ArgumentError` with the valid range.

Render selection errors for multi-definition `.ifs` files include available indexes and names.

## Exported Constants

- `RESOLUTION`
- `DEFAULT_WARMUP`
- `DEFAULT_SAMPLES`
- `HEIGHWAY_DRAGON`
- `EISENSTEIN`

## Testing

Run from repository root:

```powershell
julia --startup-file=no --project=Fractals Fractals/test/runtests.jl
```
