# Fractals.jl

Fractals.jl generates and rasterizes 2D iterated function system (IFS) fractals: define a
system as a matrix of affine maps (or load one from a `.ifs` file), then render it as a
chaos-game point cloud, a deterministic expansion, an iterated image, an inverse-iteration
coverage map, or a diagram of the maps themselves.

Supported Julia versions: `1.11` and `1.12`.

## Installation

From the repository root, instantiate the DrWatson project (for renders and scripts) and the package itself (for development and tests):

```bash
julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=Fractals.jl -e "using Pkg; Pkg.instantiate()"
```

## First render

```julia
using Fractals

ifs = IFS(HEIGHWAY_DRAGON; name = "Heighway Dragon")
out = render(ifs, Chaos(npoints = 200_000); resolution = (1024, 1024))
out.outpath   # <repo>/plots/Heighway_Dragon_chaos_npoints=200000_1024x1024.png
```

[`render`](@ref) saves under DrWatson's `plotsdir()`, names the file from its parameters,
and saves a render record (`.jld2`) under `data/sims/renders/` so the render can be reloaded with
[`load_render`](@ref). Rendering the same description again reuses the saved image
(`out.cached`), and point clouds are cached across resolutions. See [Output files](@ref) and
[Data layout](@ref).

## Render methods at a glance

| Method | Best for | Parameters |
|:-------|:---------|:-----------|
| [`Chaos`](@ref) | baseline stochastic rendering from sampled points | `npoints`, `warmup`, `seed` |
| [`PointDeterministic`](@ref) | exact expansion previews at a fixed depth | `npoints`, `iterations`, `warmup`, `seed` |
| [`ImageIterate`](@ref) | iterating a seed image, polygon or other render through the maps | source, `iterations` |
| [`Inverse`](@ref) | inverse-branch coverage and divergence maps | `iterations`, `divergence_scale`, `preview` |
| [`Transformations`](@ref) | previewing each affine map on a seed polygon | `polygon`, `limits_iterations` |

## Where to go next

- [Render methods](@ref): the `render` API and each method's options.
- [Quick recipes](@ref): copy-paste examples for each workflow.
- [Animation](@ref): interpolation, frame sequences, and GIF/MP4 export.
- [Output files](@ref) and [Data layout](@ref): where everything is written and how it is named.
- [IFS text format](@ref): writing `.ifs` files and validation rules.
- [Benchmarks](manual/benchmarks.md): the opt-in benchmark script.
- [API reference](@ref): every exported function and type.
