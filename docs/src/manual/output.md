# Output files

The repository root is a [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
project (`fractals`), and every file the package writes is managed through it. Images go to
`plots/`; everything else the package generates (render records, cached point clouds, the
prebuilt IFS library, benchmark results) goes to `data/sims/`. [Data layout](@ref) describes
each directory, how its files are named, what is git-tagged and what gets overwritten.

## Where files go

`outpath`, `outdir`, and `frames_dir` arguments accept:

- `nothing`: auto-named (see below) under [`output_dir`](@ref);
- a relative path, resolved under `output_dir()` (a leading `plots/` is not duplicated);
- an absolute path, used as given.

`output_dir(args...)` is DrWatson's `plotsdir(args...)` for the active project. With
`--project=.` that is `<repo>/plots/` regardless of the working directory.
[`data_output_dir`](@ref)`(args...)` is the matching `datadir("sims", args...)` for generated
data. Override both with [`with_output_root`](@ref) (scoped to a block) or the
`FRACTALS_OUTPUT_ROOT` environment variable; either way output goes to `<root>/plots/` and
`<root>/data/sims/`.

```julia
with_output_root(mktempdir()) do
    render(IFS(EISENSTEIN), Chaos(npoints = 1_000); resolution = (64, 64))   # written under <tmp>
end
```

[`ifs_data_dir`](@ref)`(args...)` points at the curated, tracked `.ifs` input files in `ifs/` at
the DrWatson project root, independent of the active project. For example,
`parse_ifs_file(ifs_data_dir("Default.ifs"))`.

Only `outpath` places a file: the JLD2 files (render records, point clouds) always go to their
fixed locations under `data/sims/` and cannot be redirected one by one. Use
`with_output_root` to move everything at once.

## File names

When the path is omitted, the PNG name comes from DrWatson's `savename` over the render: the
sanitized IFS name (`ifs` when unnamed) plus the method tag and the method's own fields
(`DrWatson.allaccess`), then the resolution and `_color`/`_alpha` flags:

```julia
out = render(IFS(EISENSTEIN; name = "Eisenstein"), Chaos(npoints = 100_000); resolution = (800, 800))
basename(out.outpath)   # "Eisenstein_chaos_npoints=100000_800x800.png"
```

`Inverse(preview = true)` adds `preview=true` to the name; exact renders keep their names. SVG renders (`format = SVG()`) end in `_svg.svg` instead of `.png`, so a PNG and an SVG render of one description have different file names and different render records.

## Render records and reuse

[`render`](@ref) (through [`save_render`](@ref)) writes two files: the PNG under `plots/` and a
`.jld2` record holding the [`Render`](@ref) description and the world-space `limits` of the
image. The record lives at [`render_record_path`](@ref)`(r)` =
`data/sims/renders/<render name>.jld2`, whatever `outpath` the PNG used. Animation frames mirror
their layout: `plots/frames/<sequence>/frame_0001.png` has its record at
`data/sims/renders/frames/<sequence>/frame_0001.jld2`, and `render_record_path(png)` finds the
record of a PNG at its default location or of a frame.

The record is written with DrWatson's `tagsave`, so it also stores the Git commit
(`gitdescribe`, e.g. `v1.0.1-41-g1922796-dirty`). [`load_render`](@ref) reads it back (pass the
`.jld2`, or such a PNG), and an `ImageIterate` with a [`FileSource`](@ref) uses the record's
limits to frame a PNG as a seed:

```julia
r = load_render(out.record)
render(r; outpath = "again.png")                      # same description, same seed, same image
render(Render(r; resolution = (4096, 4096)))          # same points, rasterized larger
```

`Render(r; resolution = ..., color = ..., ...)` copies a render with some fields changed.

`render` works like DrWatson's `produce_or_load`: when the PNG and a record describing the same
render (same IFS content, method, resolution, color, alpha and overlays) already exist, they
are loaded instead of redrawn and the result has `cached = true` (the image is then the PNG as
read from disk). `render(r; force = true)` always redraws and overwrites both.

```julia
out = render(ifs, Chaos(npoints = 1_000_000, seed = 1))   # draws; out.cached == false
out = render(ifs, Chaos(npoints = 1_000_000, seed = 1))   # reuses; out.cached == true
out = render(ifs, Chaos(npoints = 1_000_000, seed = 1); force = true)
```

## Point clouds

The point clouds of [`Chaos`](@ref) and [`PointDeterministic`](@ref) renders are cached
separately by [`cached_points`](@ref), one flat file per cloud in `data/sims/points/` named
like `chaos_ifs=<key>_npoints=1000_seed=1_warmup=50.jld2`. The key is [`ifs_key`](@ref), a hash of
the IFS's maps and weights only (not its name), plus the method's fields. Resolution, color,
alpha and overlays are not part of it, so re-rendering at another size or style only redoes the
rasterization. An unseeded method is cached too, so repeated renders reuse one cloud.

```julia
points, path, cached = cached_points(Chaos(npoints = 100_000, seed = 1), ifs)
```

[`extend_fractal`](@ref) continues a cached chaos game from its last point and stores the
accumulated cloud as the cache entry for the larger `npoints`, so a later render with that count
reuses it. This refines a fractal over several sessions without redoing earlier work:

```julia
points, path = extend_fractal(Chaos(npoints = 1_000_000, seed = 1), ifs, 500_000; seed = 2)
render(ifs, Chaos(npoints = 1_500_000, seed = 1))   # uses the extended cloud
```

[`fractal_catalog`](@ref) lists every saved render and point cloud as rows of named tuples
(`kind`, `name`, `method`, `npoints`, `resolution`, `ifs_key`, `gitcommit`, `path`), reading only
small keys from each file:

```julia
filter(r -> r.kind == :points && r.npoints > 1_000_000, fractal_catalog())
```

## SVG and provenance

SVG renders (`format = SVG()`, or [`render_transformations_svg`](@ref)) overwrite an existing file,
like PNG renders, and start with an
`<!-- fractals.provenance: ... -->` comment (git commit, version, timestamp, parameters). Their render
record holds the `Render` (with its `SVG` format) like a PNG's, and a reused SVG render returns the
file read back as text.
[`read_provenance`](@ref) reads the equivalent PNG text chunk from PNGs written by the
`_save_image` helper, which the ImageIterate animation script uses; PNGs written by `render`
carry no text chunks, because their render record holds that information.
