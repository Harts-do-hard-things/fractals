# Benchmarks

The benchmark suite provides standard workloads for performance tracking.

Benchmark flags are separate from render arguments; see [Render methods](@ref) for those.

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

Write JSON output (relative `--json` paths resolve under DrWatson's `datadir("exp_pro")`, so this writes `data/exp_pro/benchmarks/bench_small.json`; an existing file is kept as a `_#1` backup):

```powershell
julia --startup-file=no --project=. Fractals.jl/bin/fractals.jl benchmark --profile small --repeats 1 --json benchmarks/bench_small.json
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

| Profile | Purpose | `npoints` | `resolution` | `inverse_iterations` |
|:--------|:--------|----------:|:-------------|---------------------:|
| `small` | fast smoke benchmark | 20,000 | `(128, 128)` | 2 |
| `medium` | balanced local performance check | 100,000 | `(256, 256)` | 3 |
| `large` | heavy run for throughput baselining | 300,000 | `(512, 512)` | 4 |

`--profile all` runs all three.

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
julia --startup-file=no --project=. Fractals.jl/bin/fractals.jl benchmark --profile small --repeats 3 --targets Fractals.jl/bench/perf_targets.toml --json benchmarks/bench_ci.json
```

The JSON lands in `data/exp_pro/benchmarks/bench_ci.json` and is uploaded as the `bench-ci-json` artifact.

Behavior:
- `warn`: CI job stays green and emits workflow warnings.
- `fail`: CI job fails.

## Target envelopes

Latency and allocation targets are versioned in `Fractals.jl/bench/perf_targets.toml`.
Each operation's measured/target ratio is classified with the thresholds in that file
(`warn_ratio = 1.15`, `fail_ratio = 1.30`):

- `pass`: ratio `<= warn_ratio`
- `warn`: `warn_ratio <` ratio `<= fail_ratio`
- `fail`: ratio `> fail_ratio`

Run with target comparison, optionally failing on `fail` with `--strict`:

```bash
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl benchmark --profile all --repeats 3 --targets Fractals.jl/bench/perf_targets.toml
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl benchmark --profile all --repeats 3 --targets Fractals.jl/bench/perf_targets.toml --strict
```

Interpreting the envelopes:

- Chaos throughput is roughly `npoints / mean_s` for `iterate!`.
- Raster throughput is roughly `(height * width) / mean_s` for `make_image` and inverse
  rasterization.
- Memory is bounded by `mean_alloc_bytes` within the same warn/fail thresholds.

Only compare runs with the same Julia version, thread count, OS/CPU class, profile, and
repeat count.
