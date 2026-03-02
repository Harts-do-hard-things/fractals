# Fractals.jl Docs

Fractals.jl provides fast tools for rendering 2D iterated function system fractals in Julia.

## Start Here

- Use the high-level `render(...)` API for most workflows.
- Use `Chaos`, `Parallel`, `PointDeterministic`, `ImageIterate`, and `Inverse` methods depending on your use case.
- Save outputs to `media/` (default behavior).

## Key Links

- [Quick Recipes](quick-recipes.md)
- [CLI](cli.md)
- [Benchmarks](benchmarks.md)
- [Troubleshooting](troubleshooting.md)
- Package reference: `Fractals/DOCUMENTATION.md`

## Local Build

```powershell
julia --project=Fractals -e "using Pkg; Pkg.instantiate()"
mkdocs serve
```
