# Benchmarks

The benchmark suite provides standard workloads for performance tracking.

## Run via CLI

```powershell
julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl benchmark --profile all --repeats 3
```

Write JSON output:

```powershell
julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl benchmark --profile small --repeats 1 --json media/bench_small.json
```

## Profiles

- `small`: fast smoke benchmark
- `medium`: balanced local performance check
- `large`: heavy run for throughput baselining
- `all`: run all three profiles

## Metrics

Each profile reports min/mean/max seconds for:
- `iterate!`
- `iterate_parallel!`
- `make_image`
- `rasterize_image_inversely`

`iterate_parallel!` is tracked separately for historical/performance monitoring, even though it currently aliases auto-threaded `iterate!`.

## Notes

- Compare results only under similar machine/load/thread conditions.
- Use a fixed profile and repeats for trend comparisons.
- Thread count is included in benchmark output metadata.
