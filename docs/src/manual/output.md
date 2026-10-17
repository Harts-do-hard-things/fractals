# Output files

The repository root is a [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
project (`fractals`), and every file the package writes is managed through it.

## Where files go

`outpath`, `outdir`, and `frames_dir` arguments accept:

- `nothing`: auto-named (see below) under [`output_dir`](@ref);
- a relative path, resolved under `output_dir()` (a leading `plots/` is not duplicated);
- an absolute path, used as given.

`output_dir(args...)` is DrWatson's `plotsdir(args...)` for the active project. With
`--project=.` that is `<repo>/plots/` regardless of the working directory. Override it
with [`with_output_root`](@ref) (scoped to a block) or the `FRACTALS_OUTPUT_ROOT`
environment variable; either way output goes to `<root>/plots/`.

```julia
with_output_root(mktempdir()) do
    render(IFS(EISENSTEIN), Chaos(npoints = 1_000); resolution = (64, 64))   # written to <tmp>/plots/
end
```

## Data directories

The project follows DrWatson's default layout for data as well as plots:

- [`data_output_dir`](@ref)`(args...)` is DrWatson's `datadir("exp_pro", args...)`, or
  `<root>/data/exp_pro/` under the same overrides. It receives generated data such as
  the prebuilt IFS library (`data/exp_pro/ifs/`). Generated data under `data/sims/` and
  `data/exp_pro/` is git-ignored.
- [`ifs_data_dir`](@ref)`(args...)` points at the curated, tracked `.ifs` input library in
  `ifs/` at the DrWatson project root, independent of the active project. For
  example, `parse_ifs_file(ifs_data_dir("Default.ifs"))`.

## File names

When the path is omitted, the PNG name comes from DrWatson's `savename` over the render: the
sanitized IFS name (`ifs` when unnamed) plus the method tag and the method's own fields
(`DrWatson.allaccess`), then the resolution and `_color`/`_alpha` flags:

```julia
out = render(IFS(EISENSTEIN; name = "Eisenstein"), Chaos(npoints = 100_000); resolution = (800, 800))
basename(out.outpath)   # "Eisenstein_chaos_npoints=100000_800x800.png"
```

## Render records

[`render`](@ref) (through [`save_render`](@ref)) writes two files: the PNG under `plots/` and a
`.jld2` record holding the [`Render`](@ref) description and the world-space `limits` of the
image. Records always live in the data directory, at
[`render_record_path`](@ref)`(r)` = `data/exp_pro/renders/<render name>.jld2`. `outpath` only
places the PNG; there is no path override for records. Animation frames mirror their layout:
`plots/frames/<sequence>/frame_0001.png` has its record at
`data/exp_pro/renders/frames/<sequence>/frame_0001.jld2`.

The record is written with DrWatson's `tagsave`, so it also stores the Git commit
(`gitdescribe`, e.g. `v1.0.1-41-g1922796-dirty`). [`load_render`](@ref) reads it back (pass the
`.jld2`, or a PNG at its default location or an animation frame), and an `ImageIterate` with a
[`FileSource`](@ref) uses the record's limits to frame such a PNG as a seed:

```julia
r = load_render(out.record)
render(r; outpath = "again.png")                      # same description, same seed, same image
render(Render(r; resolution = (4096, 4096)))          # same points, rasterized larger
```

The point clouds of [`Chaos`](@ref) and [`PointDeterministic`](@ref) renders are cached
separately by [`cached_points`](@ref) under `data/exp_pro/points/<ifs_key>/`, keyed only on the
IFS's maps and weights ([`ifs_key`](@ref), not its name) and the method's fields. Resolution,
color, alpha and overlays are not part of the key, so re-rendering at another size or style
only redoes the rasterization.

Existing files at the same path are overwritten. SVGs start with an
`<!-- fractals.provenance: ... -->` comment, and [`read_provenance`](@ref) reads the
provenance chunk of PNGs written by the animation script.

## Saving fractal data (JLD2)

Images are lossy views of a fractal. [`save_fractal`](@ref) keeps the data itself as a
[JLD2](https://github.com/JuliaIO/JLD2.jl) file under [`data_output_dir`](@ref), written
with DrWatson's `tagsave` (so it records the Git commit) and `safesave` (so nothing is
overwritten). A record contains only plain arrays, strings and numbers: the equation matrix
`"eq"`, `"weights"`, `"limits"`, the point cloud `"points"` as a `2×N` `Matrix{Float64}`, an
optional `"image"`, the render `"method"`, `"params"`, and the `"seed"`. It therefore loads
without this package:

```julia
using DrWatson
ifs = IFS(EISENSTEIN); points = chaos_game(ifs, 100_000; seed = 1)
path = save_fractal(ifs, points; seed = 1)
rec = wload(path)
rec["points"]   # 2×100000 Matrix{Float64}
```

[`load_fractal`](@ref) goes the other way and rebuilds an [`IFS`](@ref) with its points and
limits, together with the metadata. `load_fractal` returns the points as `points` and the system as `ifs`.

### Caching with `fractal_data`

[`fractal_data`](@ref) computes a point cloud once with DrWatson's `produce_or_load`. The
file name is a `savename` of the method, parameters and a hash of the equation matrix, so an
identical later call reloads the file (`cached = true`) instead of iterating again.
Rasterize the saved points at any resolution with [`make_image`](@ref):

```julia
data = fractal_data(EISENSTEIN; npoints = 1_000_000, seed = 1)   # computed and saved
data = fractal_data(EISENSTEIN; npoints = 1_000_000, seed = 1)   # loaded: data.cached == true
make_image(data.points, data.ifs.limits; resolution = (2048, 2048))
```

### Resuming and cataloging

[`extend_fractal`](@ref) continues a saved chaos game from its last point and saves the
accumulated cloud as a new record, so a fractal can be refined over several sessions.
[`fractal_catalog`](@ref) lists every saved record as a vector of named tuples (reading only
a small summary per file).

```julia
more = extend_fractal(data, 500_000; seed = 2)
filter(r -> r.method == :chaos && r.npoints > 1_000_000, fractal_catalog())
```