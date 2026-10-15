# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Two Julia packages for generating and rasterizing 2D IFS (iterated function system) fractals, supporting Julia 1.11 and 1.12 (CI uses 1.12):

- `Fractals.jl/` — the core package (with an optional CUDA extension).
- `FractalsGUI.jl/` — a Gtk desktop GUI for editing IFS matrices and previewing transformation SVGs. Its `Manifest.toml` depends on `Fractals` by relative path (`../Fractals.jl`), so it always uses the working-tree version.

User-facing docs: `README.md`, `Fractals.jl/DOCUMENTATION.md` (detailed API reference), `docs/` (MkDocs site: quick recipes, CLI, benchmarks, troubleshooting). `CHANGELOG.md` follows Keep a Changelog — add user-visible changes under `[Unreleased]`.

## Commands

Always pass `--startup-file=no` (CI does; a user `startup.jl` can break precompilation/tests — see `docs/troubleshooting.md`). Run from the repo root.

```bash
julia --startup-file=no --project=Fractals.jl -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=Fractals.jl Fractals.jl/test/runtests.jl
julia --startup-file=no --project=FractalsGUI.jl FractalsGUI.jl/test/runtests.jl
julia --startup-file=no --project=FractalsGUI.jl FractalsGUI.jl/bin/gui.jl
```

`Fractals.jl/test/runtests.jl` is a single large file of top-level `@testset`s with no filter mechanism; to iterate on one area, run the shared header (imports, `SMALL_EQ`, snapshot helpers at the top of the file) plus the relevant `@testset` in a REPL with `--project=Fractals.jl`.

Opt-in test groups (off in CI): `FRACTALS_RUN_GPU_TESTS=1` (CUDA parity tests) and `FRACTALS_RUN_CLI_BENCH_TESTS=1` (CLI/benchmark integration tests).

Snapshot tests compare against PNGs in `Fractals.jl/test/snapshots/`; on mismatch, actual/diff images are written to a temp dir. When render output changes intentionally, regenerate:

```bash
julia --startup-file=no --project=Fractals.jl Fractals.jl/test/generate_snapshots.jl
```

Formatting (JuliaFormatter v1, `.JuliaFormatter.toml`: default style, margin 100) covers `Fractals.jl/src` and `Fractals.jl/test`; CI runs it as a check. Apply locally with:

```bash
julia --startup-file=no -e "using Pkg; Pkg.activate(temp=true); Pkg.add(name=\"JuliaFormatter\", version=\"1\"); using JuliaFormatter; format([\"Fractals.jl/src\", \"Fractals.jl/test\"]; overwrite=true, verbose=true)"
```

CLI (`render`, `batch-render`, `validate-ifs`, `benchmark`; see `docs/cli.md`):

```bash
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl help
julia --startup-file=no --project=Fractals.jl Fractals.jl/bin/fractals.jl benchmark --profile small --repeats 3 --targets Fractals.jl/bench/perf_targets.toml
```

CI also runs that benchmark and fails on regressions against the latency/allocation envelopes in `Fractals.jl/bench/perf_targets.toml`, so performance-sensitive changes may require updating those targets.

## Architecture (`Fractals.jl/src`)

**Include order in `Fractals.jl` matters** — later files call into earlier ones, and the comments there document the constraints (e.g. `image_iterate.jl` must follow all render-method files; `interactive.jl` must precede `render.jl`). Keep that order when adding files, and add new public functions to the `export` list.

Core data model (`types.jl`):
- An IFS is defined by an equation matrix with one row per map: `a11 a12 a21 a22 b1 b2 [p]` (Fractint `.ifs` order). With no 7th column, weights are derived from `|det(A)|`.
- `AffineMap` wraps a StaticArrays 2×2 matrix + offset, and is callable and invertible. `IFS` holds maps, weights, the point cloud, and `limits` (world-space bounds used to frame every raster).

Render methods — each lives in its own file and is wrapped by a `render_*` function in `render.jl`:
- `chaos.jl` — chaos game (`iterate!`, threaded `iterate_parallel!`), with an optional `seed` for reproducibility.
- `point_deterministic.jl` — applies every map to every point, `iterations` deep.
- `rasterize.jl` — point cloud → normalized Float32 grayscale heatmap (`make_image`).
- `inverse.jl` — per-pixel inverse-map coverage (`rasterize_image_inversely`, `mode=:exact|:preview`).
- `image_iterate.jl` — applies the IFS to an image rather than to points. The source image is chosen by `image_source` (`:polygon`, `:chaos`, `:point_deterministic`, `:inverse`, `:file`) through the `_IMAGE_SOURCE_DISPATCH` table.
- `render_transformations.jl` — SVG/PNG diagrams of the affine maps applied to an initial polygon (presets are in `config.jl`). Rotation convention is math-space: positive angles are counterclockwise, `+y` is up, and the model origin is anchored to the same pixel across presets.
- `interpolation.jl` — interpolates between equation matrices to produce animation frames. `bin/export_gif.jl` and `bin/export_mp4.jl` turn the frames into GIF/MP4 via `ffmpeg`.

`render(input; method=...)` in `render.jl` is the top-level entrypoint:
- `input` can be an equation matrix, an `IFS`, or an `.ifs` path (choose a definition with `ifs_index`/`ifs_name`; otherwise you are prompted interactively).
- `method` accepts the `RenderMethod` enum, a symbol, or a lowercase string.
- Outputs are routed under `media/` (`_normalize_media_outpath`).

PNG outputs embed source-limits metadata in a custom PNG text chunk (`utils.jl`, keyword `fractals.source_limits`). This lets `image_source=:file` reuse the original pixel-to-world framing, so save renders through `_save_image_with_source_limits`, not a bare `save`.

GPU backends:
- `make_image`, `iterate_image`, `rasterize_image_inversely`, and `render` accept `backend=:cpu|:gpu|:auto`, routed through `_dispatch_backend` in `utils.jl`.
- The core package only stubs `_gpu_backend_available` (returns false) and the `_*_gpu` functions. The real kernels live in `ext/FractalsCUDAExt.jl`, which loads only when CUDA.jl is present.
- GPU paths are expected to match CPU output; parity tests enforce this.

Parsing and interactivity:
- `ifsparser.jl` is pure (no I/O): `lex_ifs` → `parse_ifs_tokens` → `IFSDefinition`. Non-contractive definitions are skipped with a warning.
- All stdin prompting lives in `interactive.jl`. Interactive functions take an `input_fn` (default `readline`) so tests can drive them.
- Curated `.ifs` libraries live in `Fractals.jl/data/`.
