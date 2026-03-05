# Troubleshooting

This page covers common local issues and the fastest recovery paths.

## Use CI-Equivalent Commands

Run local commands the same way CI runs them:

```powershell
julia --startup-file=no --project=Fractals -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=Fractals Fractals/test/runtests.jl
```

## Startup File Issues (`startup.jl`)

Symptoms:
- Errors during startup from packages loaded in `~/.julia/config/startup.jl`
- Unexpected precompile/lock behavior before tests even begin

Fix:
- Disable startup file for project commands:

```powershell
julia --startup-file=no --project=Fractals Fractals/test/runtests.jl
```

## Cache/Permission Lock Issues

Symptoms:
- `permission denied (EACCES)` for `.pidfile`, `.ji`, or `.julia/logs/*`
- lock file errors while precompiling or instantiating

Likely causes:
- Another Julia process is still running
- Restricted permissions/sandbox environment
- Stale lock files in Julia cache/log directories

Recommended recovery flow:
1. Stop any stray Julia processes.
2. Retry with `--startup-file=no`.
3. Ensure your environment can write to `.julia` cache/log directories.
4. If still blocked, clean stale lock files in your `.julia` cache/log folders and retry.

## Environment Sanity Checks

```powershell
julia --startup-file=no --project=Fractals -e "using InteractiveUtils; versioninfo()"
julia --startup-file=no --project=Fractals -e "println(Base.active_project())"
julia --startup-file=no --project=Fractals -e "using Base.Threads; println(nthreads())"
julia --startup-file=no --project=Fractals -e "using Pkg; Pkg.instantiate()"
```

## Quick Render Smoke Test

```powershell
julia --startup-file=no --project=Fractals -e "using Fractals; render(HEIGHWAY_DRAGON; method=Chaos, npoints=50_000, outpath=\"media/smoke.png\")"
```

## Method-Specific Troubleshooting

Methods are listed in canonical order: `Chaos`, `PointDeterministic`, `ImageIterate`, `Inverse`.
`Parallel` is a compatibility alias for `Chaos` and may be deprecated in a future release.

### `Chaos`

Arguments commonly involved:
- `npoints`
- `backend`
- `resolution`
- `outpath`

Common issues:
- Very high `npoints` can increase runtime/memory pressure.
- `backend=:gpu` fails when CUDA is unavailable; use `:cpu` or `:auto`.

### `PointDeterministic`

Arguments commonly involved:
- `iterations`
- `npoints`
- `backend`
- `resolution`
- `outpath`

Common issues:
- Large `iterations` causes point growth and heavy allocation.
- Keep `iterations` low for preview use.

### `ImageIterate`

Arguments commonly involved:
- `image_source`
- `image_path` (when `image_source=:file`)
- `image_iterations`
- `polygon_limits_mode`
- `initial_polygon`
- `backend`
- `resolution`
- `outpath`

Common issues:
- `image_source=:file` requires a valid `image_path`.
- `polygon_limits_mode=:default` is treated as `:ifs` for polygon source.
- `color=true` is not supported for `method=ImageIterate`.

### `Inverse`

Arguments commonly involved:
- `iterations`
- `show_divergence_scale`
- `backend`
- `resolution`
- `outpath`

Common issues:
- Increasing `iterations` can significantly increase runtime.
- If GPU path is unavailable or capacity-limited, use `backend=:cpu` or `:auto`.
