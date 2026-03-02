# Quick Recipes

## Prerequisites

```powershell
julia --project=Fractals -e "using Pkg; Pkg.instantiate()"
```

## Recipe 1: Render From Matrix

```julia
using Fractals

eq = [
    0.5  -0.5   0.5   0.5   0.0   0.0;
   -0.5  -0.5   0.5  -0.5   1.0   0.0
]

result = render(eq;
                method=Chaos,
                npoints=200_000,
                resolution=(1024, 1024),
                outpath="media/recipe_matrix.png")

println(result.outpath)
```

## Recipe 2: Render From `.ifs` File

Use index selection:

```julia
using Fractals

result = render("my_fractals.ifs";
                ifs_index=2,
                method=Chaos,
                npoints=150_000,
                outpath="media/recipe_file_index.png")
```

Use name selection:

```julia
using Fractals

result = render("my_fractals.ifs";
                ifs_name="Heighway Dragon",
                method=Chaos,
                npoints=150_000,
                outpath="media/recipe_file_name.png")
```

If no `ifs_index`/`ifs_name` is provided and the file contains multiple definitions, `render(...)` prints available options and asks for confirmation before selecting index `1`.

## Recipe 3: Point Deterministic Preview

```julia
using Fractals

result = render(EISENSTEIN;
                method=PointDeterministic,
                npoints=300,
                iterations=2,
                resolution=(900, 900),
                outpath="media/recipe_deterministic.png")
```

## Recipe 4: Image Iterate From Polygon

```julia
using Fractals

result = render(EISENSTEIN;
                method=ImageIterate,
                image_source=:polygon,
                image_iterations=3,
                initial_polygon=:equilateral_triangle,
                polygon_limits_mode=:ifs,
                resolution=(900, 900),
                outpath="media/recipe_image_iterate.png")
```

The `:polygon` seed is produced via `render_transformations_png(...; show_base=false, limits_mode=:ifs, initial_polygon=...)` and then read as grayscale.
Built-in presets: `:default`, `:equilateral_triangle`, `:line_arrow`, `:line`.
For `image_source=:polygon`, `polygon_limits_mode=:default` is treated as `:ifs` (compatibility alias).

## Optional: Inverse Render

```julia
using Fractals

result = render(EISENSTEIN;
                method=Inverse,
                iterations=6,
                resolution=(900, 900),
                outpath="media/recipe_inverse.png")
```
