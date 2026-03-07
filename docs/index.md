# Fractals.jl Docs

Fractals.jl provides fast tools for rendering 2D iterated function system fractals in Julia.

## Start Here

- Use the high-level `render(...)` API for most workflows.
- Use the render methods in this order: `Chaos`, `PointDeterministic`, `ImageIterate`, `Inverse`.
- `Parallel` is a compatibility alias for `Chaos` and may be deprecated in a future release.
- Save outputs to `media/` (default behavior).

## Render Methods At A Glance

### `Chaos`
- Arguments used: `method`, `npoints`, `backend`, `resolution`, `outpath`
- Best for: baseline stochastic rendering from sampled points.

### `PointDeterministic`
- Arguments used: `method`, `iterations`, `npoints`, `backend`, `resolution`, `outpath`
- Best for: deterministic expansion depth previews.

### `ImageIterate`
- Arguments used: `method`, `image_source`, `image_path` (file source), `image_iterations`, `polygon_limits_mode`, `initial_polygon`, `backend`, `resolution`, `outpath`
- Best for: iterating an image/polygon seed through map-space transforms.

### `Inverse`
- Arguments used: `method`, `iterations`, `show_divergence_scale`, `backend`, `resolution`, `outpath`
- Best for: inverse branch coverage style rendering.

## Key Links

- [Quick Recipes](quick-recipes.md)
- [CLI](cli.md)
- [Benchmarks](benchmarks.md)
- [Troubleshooting](troubleshooting.md)
- Package reference: `fractals/DOCUMENTATION.md`

## Local Build

```powershell
julia --project=fractals -e "using Pkg; Pkg.instantiate()"
mkdocs serve
```
