# Data layout

Everything the package writes lands in one of two git-ignored trees of the DrWatson project:
`plots/` for images and `data/sims/` for generated data. Nothing generated is tracked; the
curated inputs in `ifs/` are. This page lists what lives where, how files are named, what is
tagged with the Git commit, and what is overwritten.

```text
fractals/
├── ifs/                      curated .ifs inputs (tracked)
├── plots/                    images and animations          output_dir()
│   ├── <name>.png            render output (render, outpath = nothing)
│   ├── frames/<sequence>/    render_interpolation_frames output
│   └── animations/           export_animation output (.gif, .mp4)
└── data/sims/                all generated data             data_output_dir()
    ├── renders/              one render record per PNG (and frames/<sequence>/ mirrors)
    ├── points/               cached Chaos / PointDeterministic point clouds
    ├── ifs/<file stem>/      prebuilt IFS library, one record per function system
    └── benchmarks/           results of scripts/benchmark.jl
```

Both roots follow the active DrWatson project (`projectdir()`): with `--project=.` they are
`<repo>/plots` and `<repo>/data/sims`. [`with_output_root`](@ref) and the `FRACTALS_OUTPUT_ROOT`
environment variable move them together to `<root>/plots` and `<root>/data/sims`; the tests use
this to keep everything in a temporary directory. `data/exp_pro/` is also git-ignored, for
files left from before generated data moved to `data/sims/`.

## What each directory holds

| Directory | Contents | Written by |
|:----------|:---------|:-----------|
| `plots/` | PNG renders; `plots/frames/<sequence>/frame_NNNN.png`; `plots/animations/<basename>.gif` or `.mp4`; SVG diagrams | [`render`](@ref), [`render_interpolation_frames`](@ref), [`export_animation`](@ref), [`render_transformations_svg`](@ref) |
| `data/sims/renders/` | a `.jld2` per render: the [`Render`](@ref) description and its `limits`; frames in `renders/frames/<sequence>/` | [`save_render`](@ref) (via `render`) |
| `data/sims/points/` | a `.jld2` per point cloud: `points` (2×N), the `method`, the IFS key | [`cached_points`](@ref), [`extend_fractal`](@ref) |
| `data/sims/ifs/<file stem>/` | `ifs_<key>.jld2` per function system: equation matrix, weights, name, docs, source file | [`build_ifs_library`](@ref) |
| `data/sims/benchmarks/` | one `.jld2` per benchmark run | `scripts/benchmark.jl` (see [Benchmarks](@ref)) |

## File names

- **PNGs** are named by DrWatson's `savename`: `<ifs name>_<method tag>_<method fields>_<rows>x<cols>`
  plus `_color` and/or `_alpha`, for example
  `Heighway_Dragon_chaos_npoints=200000_seed=1_1024x1024_color.png`. Only fields that matter to
  the method appear (`npoints` and `seed` for `Chaos`; `iterations` for `Inverse`, plus `preview`
  when set; `polygon` and `limits_iterations` for `Transformations`). An explicit `outpath`
  replaces the name (and may be relative to `plots/` or absolute).
- **Render records** reuse the PNG's name with a `.jld2` extension, under `data/sims/renders/`
  regardless of `outpath` ([`render_record_path`](@ref)). Frames keep their
  `frames/<sequence>/frame_NNNN` layout.
- **Point clouds** are `<method tag>_ifs=<key>_<every method field>.jld2`, for example
  `chaos_ifs=<key>_npoints=1000_seed=1_warmup=50.jld2`. `<key>` is [`ifs_key`](@ref), a stable
  hash of the maps and weights (the IFS name, resolution and styling are not part of it).
  Unset fields such as a missing seed are left out.
- **IFS library records** are `ifs_<16 hex digits>.jld2` in a folder named after the source file;
  the digits hash the equation matrix and weights only, so renaming a fractal does not create a
  new record (see [IFS text format](@ref)).
- **Benchmark records** are `benchmarks_profile=<profile>_time=<yyyymmdd-HHMMSS>.jld2`.
- **Animation frames** are `<basename>_0001.png`, `<basename>_0002.png`, ...; the animation is
  `plots/animations/<basename>.<format>`.

## What is tagged with the Git commit

- Render records are written with DrWatson's `tagsave`, so they carry a `gitcommit` entry
  (`gitdescribe`, with `-dirty` for uncommitted changes).
- Cached point clouds carry the same `gitcommit` entry.
- Benchmark records are `tagsave`d too, and `compare_benchmarks` prints the two commits.
- IFS library records and PNGs are not tagged (a PNG's record is).

## What is overwritten

| File | Behavior |
|:-----|:---------|
| render PNG and record | Reused when both exist and describe the same render (`cached = true`); redrawn and overwritten when the description differs or with `force = true`. |
| point cloud | Reused (loaded) when the file exists; `cached_points(...; force = true)` recomputes it. [`extend_fractal`](@ref) overwrites the entry for the larger `npoints`. |
| IFS library record | Reused when the key matches; `build_ifs_library(force = true)` rewrites. |
| benchmark record | Never overwritten: names include the date and time. |
| animation (GIF/MP4) | Overwritten by [`export_animation`](@ref). |
| SVG diagrams, `_save_image` PNGs | Written with DrWatson's `safesave`: an existing file is kept as `name_#1.ext` (older ones `#2`, ...). |

Because generated files are git-ignored, delete `data/sims/points/` to free space (every cloud is
recomputed on demand) and `data/sims/renders/` to forget which renders exist (the PNGs are just
redrawn).
