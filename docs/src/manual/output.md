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
    render(EISENSTEIN; npoints = 1_000, resolution = (64, 64))   # written to <tmp>/plots/
end
```

## Data directories

The project follows DrWatson's default layout for data as well as plots:

- [`data_output_dir`](@ref)`(args...)` is DrWatson's `datadir("exp_pro", args...)`, or
  `<root>/data/exp_pro/` under the same overrides. It receives generated data such as
  benchmark JSON (`fractals.jl benchmark --json <path>`). Generated data under `data/sims/` and
  `data/exp_pro/` is git-ignored.
- [`ifs_data_dir`](@ref)`(args...)` points at the curated, tracked `.ifs` input library in
  `ifs/` at the DrWatson project root, independent of the active project. For
  example, `parse_ifs_file(ifs_data_dir("Default.ifs"))`.

## File names

When the path is omitted, the name comes from DrWatson's `savename`. The prefix is the
sanitized IFS name (`ifs` when unnamed), followed by the content-defining render settings,
with `false`/`nothing` values left out:

```julia
out = render(EISENSTEIN; method = Chaos, npoints = 100_000, resolution = (800, 800))
basename(out.outpath)   # "ifs_method=chaos_npoints=100000_resolution=800x800.png"
```

## No overwrites

Every save goes through DrWatson's `safesave`: if the file exists, the old one is moved to
`name_#1.ext` (and older backups to `#2`, `#3`, ...) before the new file is written.

## Provenance metadata

PNGs embed two `tEXt` chunks:

- `fractals.source_limits`: the world-space framing, so `image_source = :file` can reuse it;
- `fractals.provenance`: JSON with `gitcommit` (DrWatson's `gitdescribe`, e.g.
  `v1.0.1-41-g1922796-dirty`), `fractals_version`, `created`, and all render `params`.

SVGs start with an equivalent `<!-- fractals.provenance: ... -->` comment. Read PNG
provenance back with [`read_provenance`](@ref):

```julia
prov = read_provenance(out.outpath)
prov.params.method, prov.gitcommit   # ("chaos", "v1.0.1-41-g...")
```

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
limits, together with the metadata.

### Caching with `fractal_data`

[`fractal_data`](@ref) computes a point cloud once with DrWatson's `produce_or_load`. The
file name is a `savename` of the method, parameters and a hash of the equation matrix, so an
identical later call reloads the file (`cached = true`) instead of iterating again.
[`render_fractal`](@ref) then rasterizes the saved points at any resolution or coloring:

```julia
data = fractal_data(EISENSTEIN; npoints = 1_000_000, seed = 1)   # computed and saved
data = fractal_data(EISENSTEIN; npoints = 1_000_000, seed = 1)   # loaded: data.cached == true
render_fractal(data; resolution = (2048, 2048), color = true)
```

### Resuming and cataloging

[`extend_fractal`](@ref) continues a saved chaos game from its last point and saves the
accumulated cloud as a new record, so a fractal can be refined over several sessions.
[`fractal_catalog`](@ref) lists every saved record as a vector of named tuples (reading only
a small summary per file). `render(...; save_data = true)` writes a record alongside the
image it produces.

```julia
more = extend_fractal(data, 500_000; seed = 2)
filter(r -> r.method == :chaos && r.npoints > 1_000_000, fractal_catalog())
```