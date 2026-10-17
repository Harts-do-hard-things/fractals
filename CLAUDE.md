# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Two Julia packages for generating and rasterizing 2D IFS (iterated function system) fractals, supporting Julia 1.11 and 1.12 (CI uses 1.12). They sit inside a DrWatson scientific project:

- Repo root (`Project.toml`, name `fractals`) — the DrWatson project. It depends on DrWatson, the local `Fractals` package (via `[sources]` path), and the packages the scripts and `bin/fractals.jl` load. It follows the DrWatson template: `scripts/` (project scripts, `@quickactivate "fractals"`), `data/exp_raw/` (curated `.ifs` inputs, tracked), `data/exp_pro/` and `data/sims/` (generated, git-ignored), `plots/` (images/animations, git-ignored). Run renders, CLI commands, and scripts with `--project=.`. `.gitignore` is DrWatson's template with `/data` narrowed so `data/exp_raw` stays tracked.
- `Fractals.jl/` — the core package (with an optional CUDA extension). Tests and package development use `--project=Fractals.jl`.
- `FractalsGUI.jl/` — a Gtk desktop GUI for editing IFS matrices and previewing transformation SVGs. Its `Manifest.toml` depends on `Fractals` by relative path (`../Fractals.jl`), so it always uses the working-tree version.

Adding a dependency to `Fractals.jl/Project.toml` also requires re-resolving the root and `FractalsGUI.jl` manifests (`Pkg.resolve()` with each project active).

User-facing docs: `README.md` and a Documenter.jl site in `docs/` (`docs/make.jl`; manual pages in `docs/src/manual/`, API reference in `docs/src/api.md` built from docstrings).

## Commands

Always pass `--startup-file=no` (CI does; a user `startup.jl` can break precompilation/tests — see `docs/src/manual/troubleshooting.md`). Run from the repo root.

```bash
julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=Fractals.jl -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=Fractals.jl Fractals.jl/test/runtests.jl
julia --startup-file=no --project=FractalsGUI.jl FractalsGUI.jl/test/runtests.jl
julia --startup-file=no --project=FractalsGUI.jl FractalsGUI.jl/bin/gui.jl
```

`Fractals.jl/test/runtests.jl` is a single large file of top-level `@testset`s with no filter mechanism; to iterate on one area, run the shared header (imports, `SMALL_EQ`, `FRACTALS_OUTPUT_ROOT` setup, snapshot helpers at the top of the file) plus the relevant `@testset` in a REPL with `--project=Fractals.jl`. The header points `FRACTALS_OUTPUT_ROOT` at a temp dir so tests never write into `plots/`; assert on paths with `output_dir(...)`, not literal directories.

Opt-in test groups (off in CI): `FRACTALS_RUN_GPU_TESTS=1` (CUDA parity tests) and `FRACTALS_RUN_CLI_BENCH_TESTS=1` (CLI command and benchmark tests). CLI tests call `bin/fractals.jl` in-process via `_run_cli` (the script is loaded into a `FractalsCLI` module and only calls `exit` when run as a script); do not reintroduce per-command `julia` subprocesses, which made this group take many minutes.

Snapshot tests compare against PNGs in `Fractals.jl/test/snapshots/`; on mismatch, actual/diff images are written to a temp dir. When render output changes intentionally, regenerate:

```bash
julia --startup-file=no --project=Fractals.jl Fractals.jl/test/generate_snapshots.jl
```

Formatting (JuliaFormatter pinned to 1.0.62, `.JuliaFormatter.toml`: default style, margin 100) covers `Fractals.jl/src` and `Fractals.jl/test`. CI fails on any unformatted code and prints the diff, so format before committing changes there (note this style uses spaces around keyword `=`, e.g. `f(x; kw = 1)`). Apply locally with:

```bash
julia --startup-file=no -e "using Pkg; Pkg.activate(temp=true); Pkg.add(name=\"JuliaFormatter\", version=\"1.0.62\"); using JuliaFormatter; format([\"Fractals.jl/src\", \"Fractals.jl/test\"]; overwrite=true, verbose=true)"
```

Docs (Documenter.jl; `docs/Project.toml` gets `Fractals` via a `[sources]` path, and `docs/Manifest.toml`/`docs/build/` are gitignored):

```bash
julia --startup-file=no --project=docs -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=docs docs/make.jl
```

The build (also run in CI) uses `checkdocs = :exports` and runs `jldoctest` blocks, so every exported name needs a docstring and a `@docs` entry in `docs/src/api.md`, and broken `@ref` links fail the build. Doctests run with `FRACTALS_OUTPUT_ROOT` set to a temp dir.

CLI (`render`, `batch-render`, `validate-ifs`, `benchmark`; see `docs/src/manual/cli.md`):

```bash
julia --startup-file=no --project=. Fractals.jl/bin/fractals.jl help
julia --startup-file=no --project=. Fractals.jl/bin/fractals.jl render --input data/exp_raw/Default.ifs --ifs-index 1 --resolution 512x512
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
- `interpolation.jl` — interpolates between equation matrices to produce animation frames. `scripts/export_gif.jl` and `scripts/export_mp4.jl` turn the frames into GIF/MP4 via `ffmpeg`; `scripts/animate_image_iterate.jl` builds per-map ImageIterate animations (the test suite `include`s it, so its `@quickactivate` only runs when executed as a script).

`render(input; method=...)` in `render.jl` is the top-level entrypoint:
- `input` can be an equation matrix, an `IFS`, or an `.ifs` path (choose a definition with `ifs_index`/`ifs_name`; otherwise you are prompted interactively).
- `method` accepts the `RenderMethod` enum, a symbol, or a lowercase string.
- Each `render_*` saves through `_save_render_output`, passing its full `params` (recorded in provenance) and the `name_keys` subset used for the filename.

Saving is DrWatson-managed in `output.jl`. Every file the package writes goes through it, so never call `save`/`write` directly in `src/`:
- `_resolve_output_path(outpath, default_name)` handles all `outpath`/`outdir` arguments. `nothing` means auto-named, relative paths go under `output_dir()`, and absolute paths are used as-is.
- `output_dir()` is DrWatson's `plotsdir()` and `data_output_dir()` is `datadir("exp_pro")` for the active project; `with_output_root` (a ScopedValue) or `FRACTALS_OUTPUT_ROOT` override both (`<root>/plots`, `<root>/data/exp_pro`). Generated data (benchmark JSON) goes through `_save_data_text`.
- `_output_name` builds `savename` filenames, `_save_image`/`_save_svg`/`_safesave_moved_file` write via `safesave` (existing files become `_#N` backups), and the wrapper types plug into DrWatson's `_wsave` extension point.
- PNGs get two tEXt chunks (helpers in `utils.jl`):
  - `fractals.source_limits` lets `image_source=:file` reuse the original pixel-to-world framing.
  - `fractals.provenance` is JSON with the `gitdescribe` commit (cached for 2 s, since it scans the work tree), the version, a timestamp and the params; read it back with `read_provenance`.

GPU backends:
- `make_image`, `iterate_image`, `rasterize_image_inversely`, and `render` accept `backend=:cpu|:gpu|:auto`, routed through `_dispatch_backend` in `utils.jl`.
- The core package only stubs `_gpu_backend_available` (returns false) and the `_*_gpu` functions. The real kernels live in `ext/FractalsCUDAExt.jl`, which loads only when CUDA.jl is present.
- GPU paths are expected to match CPU output; parity tests enforce this.

Parsing and interactivity:
- `ifsparser.jl` is pure (no I/O): `lex_ifs` → `parse_ifs_tokens` → `IFSDefinition`. Non-contractive definitions are skipped with a warning.
- All stdin prompting lives in `interactive.jl`. Interactive functions take an `input_fn` (default `readline`) so tests can drive them.
- Curated `.ifs` libraries live in `data/exp_raw/` at the repo root. Locate them with `ifs_data_dir(...)`, not `datadir()`, which follows the active project (`Fractals.jl/data` under `--project=Fractals.jl`).
