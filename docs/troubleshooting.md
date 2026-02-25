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
