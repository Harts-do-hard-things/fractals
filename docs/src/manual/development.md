# Development

Run every command from the repository root, and always pass `--startup-file=no`. CI does,
and a user `startup.jl` can break precompilation and tests (see [Troubleshooting](@ref)).

## Tests

```bash
julia --startup-file=no --project=Fractals.jl Fractals.jl/test/runtests.jl
julia --startup-file=no --project=IFSParse.jl IFSParse.jl/test/runtests.jl
julia --startup-file=no --project=FractalsGUI.jl FractalsGUI.jl/test/runtests.jl
```

Opt-in groups, off in CI:

- `FRACTALS_RUN_GPU_TESTS=1`: CUDA parity tests.

The suite points `FRACTALS_OUTPUT_ROOT` at a temporary directory, so tests never write into
`plots/`. It includes seeded property tests for affine-map inverse round trips.

## Snapshot images

Deterministic CPU render paths are compared against PNG fixtures in
`Fractals.jl/test/snapshots/`: a transformation render, an inverse mask
(`divergence_scale = false`), and a polygon-seeded image-iterate render. On a
mismatch, the actual and diff images are written to a temp directory. When render output
changes intentionally, regenerate the fixtures:

```bash
julia --startup-file=no --project=Fractals.jl Fractals.jl/test/generate_snapshots.jl
```

## Benchmarks

The opt-in benchmark script is not part of the tests or CI; see [Benchmarks](@ref).

## Formatting

`Fractals.jl/src`, `Fractals.jl/test`, `IFSParse.jl/src` and `IFSParse.jl/test` are formatted with JuliaFormatter 1.0.62
(`.JuliaFormatter.toml`: default style, margin 100), and CI fails on unformatted code:

```bash
julia --startup-file=no -e "using Pkg; Pkg.activate(temp=true); Pkg.add(name=\"JuliaFormatter\", version=\"1.0.62\"); using JuliaFormatter; format([\"Fractals.jl/src\", \"Fractals.jl/test\", \"IFSParse.jl/src\", \"IFSParse.jl/test\"]; overwrite=true, verbose=true)"
```

## Documentation

This site is built with [Documenter.jl](https://documenter.juliadocs.org/stable/) from
`docs/src/`. The [API reference](@ref) is generated from docstrings, and the build fails if
an exported name has no docstring or a `jldoctest` block fails:

```bash
julia --startup-file=no --project=docs -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=docs docs/make.jl
```

Open `docs/build/index.html` to browse the result. New exports need a docstring and an
entry in `docs/src/api.md`.
