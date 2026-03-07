# Benchmarks

The benchmark suite provides standard workloads for performance tracking.

## Render Method Context

Render methods are documented in canonical order: `Chaos`, `PointDeterministic`, `ImageIterate`, `Inverse`.
`Parallel` is a compatibility alias for `Chaos` and may be deprecated in a future release.

Method-specific render arguments:
- `Chaos`: `npoints`, `backend`, `resolution`, `outpath`
- `PointDeterministic`: `iterations`, `npoints`, `backend`, `resolution`, `outpath`
- `ImageIterate`: `image_source`, `image_path`, `image_iterations`, `polygon_limits_mode`, `initial_polygon`, `backend`, `resolution`, `outpath`
- `Inverse`: `iterations`, `show_divergence_scale`, `backend`, `resolution`, `outpath`

These render arguments are separate from benchmark CLI flags on this page.

## Run via CLI

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl benchmark --profile all --repeats 3
```

Benchmark GPU-capable image operations with explicit backend:

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl benchmark --profile small --repeats 3 --backend cpu
```

Include optional GPU benchmark lines (when CUDA is available):

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl benchmark --profile small --repeats 3 --backend auto --include-gpu-bench
```

Write JSON output:

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl benchmark --profile small --repeats 1 --json benchmarks/bench_small.json
```

Compare against target envelopes:

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl benchmark --profile all --repeats 3 --targets Fractals.jl/bench/perf_targets.toml
```

Strict mode (fails command on target status `fail`):

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl benchmark --profile all --repeats 3 --targets Fractals.jl/bench/perf_targets.toml --strict
```

## Profiles

- `small`: fast smoke benchmark
- `medium`: balanced local performance check
- `large`: heavy run for throughput baselining
- `all`: run all three profiles

## Metrics

Each profile reports min/mean/max for:
- `iterate!`
- `iterate_parallel!`
- `make_image`
- `iterate_image`
- `rasterize_image_inversely`

Both latency and memory allocation metrics are recorded:
- `min_s`, `mean_s`, `max_s`
- `min_alloc_bytes`, `mean_alloc_bytes`, `max_alloc_bytes`

When `--include-gpu-bench` is enabled, report may include:
- `make_image_gpu` (or a skipped reason if backend mode is incompatible)
- `iterate_image_gpu` (or a skipped reason if backend mode is incompatible)
- `rasterize_image_inversely_gpu_exact` (or a skipped reason if backend mode is incompatible)
- `rasterize_image_inversely_gpu_preview` (or a skipped reason if backend mode is incompatible)

`iterate_parallel!` is tracked separately for historical/performance monitoring, even though it currently aliases auto-threaded `iterate!`.

## Notes

- Compare results only under similar machine/load/thread conditions.
- Use a fixed profile and repeats for trend comparisons.
- Thread count is included in benchmark output metadata.
- CPU/OS metadata is included in benchmark output metadata.

## CI Regression Gate

CI runs benchmark comparison using:

```powershell
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl benchmark --profile small --repeats 3 --targets Fractals.jl/bench/perf_targets.toml --json benchmarks/bench_ci.json
```

Behavior:
- `warn`: CI job stays green and emits workflow warnings.
- `fail`: CI job fails.
