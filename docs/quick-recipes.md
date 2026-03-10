# Quick Recipes

## Prerequisites

```powershell
julia --project=Fractals.jl -e "using Pkg; Pkg.instantiate()"
```

## Input Selection

Use index selection from `.ifs`:

```julia
using Fractals

result = render("my_fractals.ifs";
                ifs_index=2,
                method=Chaos,
                npoints=150_000,
                outpath="media/recipe_file_index.png")
```

Use name selection from `.ifs`:

```julia
using Fractals

result = render("my_fractals.ifs";
                ifs_name="Heighway Dragon",
                method=Chaos,
                npoints=150_000,
                outpath="media/recipe_file_name.png")
```

If no `ifs_index`/`ifs_name` is provided and the file contains multiple definitions, `render(...)` prints available options and asks for confirmation before selecting index `1`.

## Render Methods

Methods are listed in canonical order: `Chaos`, `PointDeterministic`, `ImageIterate`, `Inverse`.
`Parallel` is a compatibility alias for `Chaos` and may be deprecated in a future release.

## Recipe: Chaos

Arguments used by this method:
- `method=Chaos`
- `npoints=...`
- `backend=:cpu|:gpu|:auto`
- `resolution=(rows, cols)`
- `outpath=...`

```julia
using Fractals

eq = [
    0.5  -0.5   0.5   0.5   0.0   0.0;
   -0.5  -0.5   0.5  -0.5   1.0   0.0
]

result = render(eq;
                method=Chaos,
                backend=:cpu,
                npoints=200_000,
                resolution=(1024, 1024),
                outpath="media/recipe_chaos.png")

println(result.outpath)
```

## Recipe: PointDeterministic

Arguments used by this method:
- `method=PointDeterministic`
- `iterations=...`
- `npoints=...`
- `backend=:cpu|:gpu|:auto`
- `resolution=(rows, cols)`
- `outpath=...`

```julia
using Fractals

result = render(EISENSTEIN;
                method=PointDeterministic,
                npoints=300,
                iterations=2,
                backend=:cpu,
                resolution=(900, 900),
                outpath="media/recipe_point_deterministic.png")
```

## Recipe: ImageIterate

Arguments used by this method:
- `method=ImageIterate`
- `image_source=:polygon|:chaos|:point_deterministic|:inverse|:file`
- `image_path=...` (required when `image_source=:file`)
- `image_iterations=...`
- `polygon_limits_mode=:ifs|:default`
- `initial_polygon=:default|:equilateral_triangle|:line_arrow|:line`
- `backend=:cpu|:gpu|:auto`
- `resolution=(rows, cols)`
- `outpath=...`

```julia
using Fractals

result = render(EISENSTEIN;
                method=ImageIterate,
                image_source=:polygon,
                image_iterations=3,
                polygon_limits_mode=:ifs,
                initial_polygon=:equilateral_triangle,
                backend=:cpu,
                resolution=(900, 900),
                outpath="media/recipe_image_iterate.png")
```

The `:polygon` seed is produced via `render_transformations_png(...; show_base=false, initial_polygon=...)` and then read as grayscale.
For `image_source=:polygon`, `polygon_limits_mode=:default` is treated as `:ifs` (compatibility alias).

## Recipe: Inverse

Arguments used by this method:
- `method=Inverse`
- `iterations=...`
- `show_divergence_scale=true|false` (optional)
- `backend=:cpu|:gpu|:auto`
- `resolution=(rows, cols)`
- `outpath=...`

```julia
using Fractals

result = render(EISENSTEIN;
                method=Inverse,
                iterations=6,
                backend=:cpu,
                resolution=(900, 900),
                outpath="media/recipe_inverse.png")
```
