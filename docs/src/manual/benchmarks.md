# Benchmarks

`scripts/benchmark.jl` times the core kernels and every render method with
[BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) and saves the results so runs can
be compared. It is opt-in: CI and the test suite never run it, and BenchmarkTools is a
dependency of the root DrWatson project only (not of `Fractals.jl`).

## Running

From the repository root:

```bash
julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=. scripts/benchmark.jl small     # or: medium
```

The profile defaults to `small`. The script prints one line per case (median time, allocations,
memory) and saves a single record when it finishes. From a REPL you can call it directly:

```julia
include("scripts/benchmark.jl")
record = run_benchmarks(profile = :medium, only = r"^render/")   # only cases matching the regex
run_benchmarks(profile = :small, save = false)                    # print, do not save
```

## Profiles

| Profile | Chaos points | Render resolution | Inverse depth | Deterministic points | Budget per case |
|:--------|-------------:|:------------------|--------------:|---------------------:|:----------------|
| `small` | 100 000 | 256×256 | 5 | 200 | 2 s, up to 20 samples |
| `medium` | 1 000 000 | 512×512 | 8 | 1 000 | 5 s, up to 30 samples |

The standalone `make_image` cases use three resolutions per profile (128, 256, 512 for `small`;
256, 512, 1024 for `medium`). All cases use the Heighway Dragon.

## Cases

- `chaos_game`: generating the points.
- `cached_points/cold` and `cached_points/warm`: a cache miss (recompute and write) versus a hit
  (load only).
- `make_image/<rows>x<cols>`: rasterizing a fixed point cloud.
- `draw/<method>`: the drawing step of each render method (`chaos`, `point_deterministic`,
  `inverse`, `transformations`, `image_iterate-polygon`, `image_iterate-chaos`).
- `render/<method>`: the whole [`render`](@ref) call with `force = true`, so it redraws and saves
  instead of loading a cached PNG.
- `build_ifs_library`: building the prebuilt library from `ifs/` into a temporary directory.

Each case runs once before timing to compile it and warm caches. All files the benchmarks write
go to a temporary output root, so they never touch `plots/` or `data/`.

## Where records go

Each run saves one record to `data/sims/benchmarks/`, named from the profile and the start time
(`benchmarks_profile=small_time=20261017-120500.jld2`), so runs never overwrite each other. The
record is written with DrWatson's `tagsave`, so it carries the Git commit, and holds the profile,
timestamp, Julia version, thread count, CPU model, and per case the median and minimum time,
allocation count, memory, and sample count. It is a plain `Dict` of numbers and strings, so
`JLD2.load(path)` reads it without Fractals.

## Comparing runs

```julia
include("scripts/benchmark.jl")
latest_benchmarks()                     # lists the two newest records, newest first
a, b = latest_benchmarks(2)[[2, 1]]     # older, newer
compare_benchmarks(a, b)
```

`compare_benchmarks(a, b)` prints both runs' headers (profile, time, Julia version, threads, CPU,
commit) and a table of median times with the time and memory ratios `b / a` per case (below 1
means `b` is faster or smaller). Use the same profile, machine and thread count for both runs;
the script warns if the profiles differ. Cases present in only one record are skipped.
