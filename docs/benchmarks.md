# Benchmarks

The benchmark suite provides standard workloads for performance tracking.

## Run via CLI

```powershell
julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl benchmark --profile all --repeats 3
```

Write JSON output:

```powershell
julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl benchmark --profile small --repeats 1 --json benchmarks/bench_small.json
```

Compare against target envelopes:

```powershell
julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl benchmark --profile all --repeats 3 --targets Fractals/bench/perf_targets.toml
```

Strict mode (fails command on target status `fail`):

```powershell
julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl benchmark --profile all --repeats 3 --targets Fractals/bench/perf_targets.toml --strict
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
- `rasterize_image_inversely`

Both latency and memory allocation metrics are recorded:
- `min_s`, `mean_s`, `max_s`
- `min_alloc_bytes`, `mean_alloc_bytes`, `max_alloc_bytes`

`iterate_parallel!` is tracked separately for historical/performance monitoring, even though it currently aliases auto-threaded `iterate!`.

## Notes

- Compare results only under similar machine/load/thread conditions.
- Use a fixed profile and repeats for trend comparisons.
- Thread count is included in benchmark output metadata.
- CPU/OS metadata is included in benchmark output metadata.

## CI Regression Gate

CI runs benchmark comparison using:

```powershell
julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl benchmark --profile small --repeats 3 --targets Fractals/bench/perf_targets.toml --json benchmarks/bench_ci.json
```

Behavior:
- `warn`: CI job stays green and emits workflow warnings.
- `fail`: CI job fails.
