# Fractals.jl

Fractals.jl generates and rasterizes 2D iterated function system (IFS) fractals: define a
system as a matrix of affine maps (or load one from a `.ifs` file), then render it as a
chaos-game point cloud, a deterministic expansion, an iterated image, an inverse-iteration
coverage map, or a diagram of the maps themselves.

Supported Julia versions: `1.11` and `1.12`.

## Installation

From the repository root, instantiate the DrWatson project (for renders, scripts, and the
CLI) and the package itself (for development and tests):

```bash
julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=Fractals.jl -e "using Pkg; Pkg.instantiate()"
```

## First render

```julia
using Fractals

out = render(HEIGHWAY_DRAGON; method = Chaos, npoints = 200_000, resolution = (1024, 1024))
out.outpath   # <repo>/plots/ifs_method=chaos_npoints=200000_resolution=1024x1024.png
```

[`render`](@ref) saves under DrWatson's `plotsdir()`, names the file from its parameters,
never overwrites an existing file, and embeds provenance in the PNG. See
[Output files](@ref).

## Render methods at a glance

| Method | Best for | Method-specific keywords |
|:-------|:---------|:-------------------------|
| [`Chaos`](@ref) | baseline stochastic rendering from sampled points | `warmup` |
| [`PointDeterministic`](@ref) | exact expansion previews at a fixed depth | `iterations` |
| [`ImageIterate`](@ref) | iterating a seed image or polygon through the maps | `image_source`, `image_iterations`, `initial_polygon`, ... |
| [`Inverse`](@ref) | inverse-branch coverage and divergence maps | `iterations`, `show_divergence_scale` |
| [`RenderTransformations`](@ref) | previewing each affine map on a seed polygon | `initial_polygon`, `show_base`, `axis` |

[`Parallel`](@ref) is a compatibility alias for `Chaos` and may be deprecated.

## Where to go next

- [Render methods](@ref): the `render` API and each method's options.
- [Quick recipes](@ref): copy-paste examples for each workflow.
- [Animation](@ref): interpolation, frame sequences, and GIF/MP4 export.
- [IFS text format](@ref): writing `.ifs` files and validation rules.
- [CLI](@ref): rendering and benchmarking from the command line.
- [API reference](@ref): every exported function and type.
