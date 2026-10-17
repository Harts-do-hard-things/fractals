# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Three Julia packages for generating and rasterizing 2D IFS (iterated function system) fractals, supporting Julia 1.11 and 1.12 (CI uses 1.12). They sit inside a DrWatson scientific project:

- Repo root (`Project.toml`, name `fractals`) — the DrWatson project. It depends on DrWatson, the local `Fractals` package (via `[sources]` path), and the packages the scripts load. It follows the DrWatson template: `scripts/` (project scripts, `@quickactivate "fractals"`), `ifs/` (curated `.ifs` inputs, tracked), `data/sims/` (generated, git-ignored), `plots/` (images/animations, git-ignored). Run renders and scripts with `--project=.`. `.gitignore` is DrWatson's template; generated data (including the prebuilt IFS library in `data/sims/ifs/`) is ignored.
- `Fractals.jl/` — the core package (with an optional CUDA extension). Tests and package development use `--project=Fractals.jl`.
- `IFSParse.jl/` — the standalone `.ifs` parser and prebuilt-library package (depends only on DrWatson and JLD2, not on `Fractals`). `Fractals.jl` and `docs/` depend on it through `[sources]` path entries (and the root and `FractalsGUI.jl` projects list it too, since `[sources]` of dependencies are not inherited); it has its own tests in `IFSParse.jl/test/runtests.jl`.
- `FractalsGUI.jl/` — a Gtk desktop GUI for editing IFS matrices and previewing transformation SVGs. Its `Manifest.toml` depends on `Fractals` by relative path (`../Fractals.jl`), so it always uses the working-tree version.

Adding a dependency to `Fractals.jl/Project.toml` or `IFSParse.jl/Project.toml` also requires re-resolving the root, `Fractals.jl` (for IFSParse changes) and `FractalsGUI.jl` manifests (`Pkg.resolve()` with each project active).

User-facing docs: `README.md` and a Documenter.jl site in `docs/` (`docs/make.jl`; manual pages in `docs/src/manual/`, API reference in `docs/src/api.md` built from docstrings).

## Commands

Always pass `--startup-file=no` (CI does; a user `startup.jl` can break precompilation/tests — see `docs/src/manual/troubleshooting.md`). Run from the repo root.

```bash
julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=Fractals.jl -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=Fractals.jl Fractals.jl/test/runtests.jl
julia --startup-file=no --project=IFSParse.jl IFSParse.jl/test/runtests.jl
julia --startup-file=no --project=FractalsGUI.jl FractalsGUI.jl/test/runtests.jl
julia --startup-file=no --project=FractalsGUI.jl FractalsGUI.jl/bin/gui.jl
```

`Fractals.jl/test/runtests.jl` is a single large file of top-level `@testset`s with no filter mechanism; to iterate on one area, run the shared header (imports, `SMALL_EQ`, `FRACTALS_OUTPUT_ROOT` setup, snapshot helpers at the top of the file) plus the relevant `@testset` in a REPL with `--project=Fractals.jl`. The header points `FRACTALS_OUTPUT_ROOT` at a temp dir so tests never write into `plots/`; assert on paths with `output_dir(...)`, not literal directories.

Opt-in test group (off in CI): `FRACTALS_RUN_GPU_TESTS=1` (CUDA parity tests).

Snapshot tests compare against PNGs in `Fractals.jl/test/snapshots/`; on mismatch, actual/diff images are written to a temp dir. When render output changes intentionally, regenerate:

```bash
julia --startup-file=no --project=Fractals.jl Fractals.jl/test/generate_snapshots.jl
```

Formatting (JuliaFormatter pinned to 1.0.62, `.JuliaFormatter.toml`: default style, margin 100) covers `Fractals.jl/src`, `Fractals.jl/test`, `IFSParse.jl/src` and `IFSParse.jl/test`. CI fails on any unformatted code and prints the diff, so format before committing changes there (note this style uses spaces around keyword `=`, e.g. `f(x; kw = 1)`). Apply locally with:

```bash
julia --startup-file=no -e "using Pkg; Pkg.activate(temp=true); Pkg.add(name=\"JuliaFormatter\", version=\"1.0.62\"); using JuliaFormatter; format([\"Fractals.jl/src\", \"Fractals.jl/test\", \"IFSParse.jl/src\", \"IFSParse.jl/test\"]; overwrite=true, verbose=true)"
```

Docs (Documenter.jl; `docs/Project.toml` gets `Fractals` via a `[sources]` path, and `docs/Manifest.toml`/`docs/build/` are gitignored):

```bash
julia --startup-file=no --project=docs -e "using Pkg; Pkg.instantiate()"
julia --startup-file=no --project=docs docs/make.jl
```

The build (also run in CI) uses `checkdocs = :exports` and runs `jldoctest` blocks, so every exported name needs a docstring and a `@docs` entry in `docs/src/api.md`, and broken `@ref` links fail the build. Doctests run with `FRACTALS_OUTPUT_ROOT` set to a temp dir.

There is no CLI: renders are plain Julia calls, e.g.

```bash
julia --startup-file=no --project=. -e "using Fractals; render(IFS(HEIGHWAY_DRAGON), Chaos(npoints = 100_000); resolution = (512, 512))"
```

Benchmarks are opt-in and never run in CI or the tests (BenchmarkTools is a dependency of the root project only):

```bash
julia --startup-file=no --project=. scripts/benchmark.jl [small|medium]
```

Each run saves one DrWatson-tagged record to `data/sims/benchmarks/`; from a REPL, `include("scripts/benchmark.jl")` then `latest_benchmarks()` and `compare_benchmarks(a, b)` compare runs. All render output goes to a temp root.

## Architecture (`Fractals.jl/src`)

**Include order in `Fractals.jl` matters** — later files call into earlier ones, and the comments there document the constraints (e.g. `render_types.jl` follows `types.jl` and `render_pipeline.jl` follows every file whose kernels `draw` calls). Keep that order when adding files, and add new public functions to the `export` list.

Core data model (`types.jl`):
- An IFS is defined by an equation matrix with one row per map: `a11 a12 a21 a22 b1 b2 [p]` (Fractint `.ifs` order). With no 7th column, weights are derived from `|det(A)|`.
- `AffineMap` wraps a StaticArrays 2×2 matrix + offset, and is callable and invertible. `IFS` holds `name`, maps, weights, and `limits` (world-space bounds used to frame every raster); it stores no points, and `chaos_game`/`deterministic_iterate` return them.

Render kernels — each lives in its own file and is called by the render pipeline:
- `chaos.jl` — chaos game (`chaos_game(ifs, n)` returns the points; threaded), with an optional `seed` for reproducibility.
- `point_deterministic.jl` — applies every map to every point, `iterations` deep (`deterministic_iterate(ifs, points, n)`).
- `rasterize.jl` — points + limits → normalized Float32 grayscale heatmap (`make_image(points, limits)`).
- `inverse.jl` — per-pixel inverse-map coverage (`rasterize_image_inversely`, `mode=:exact|:preview`).
- `image_iterate.jl` — `iterate_image`, which applies the IFS to an image rather than to points.
- `drawing/` — the drawing helpers behind `Transformations` and the overlays: `projection.jl` (world to pixel/SVG mapping), `polygon.jl` (preset polygon framing, seed image, transformed segments), `post_processing.jl` (line/axes drawing and the `BasePolygon`/`Axes` overlays used by `finish`), `svg.jl` (the SVG document writer behind `finish` for `SVG()` renders, plus the `render_transformations_svg` wrapper), `transformations.jl` (the raster drawing behind `draw(::Transformations)`). Presets are in `config.jl`. Rotation convention is math-space: positive angles are counterclockwise, `+y` is up, and the model origin is anchored to the same pixel across presets.
- `interpolation.jl` — interpolates between equation matrices and renders animation frames (`render_interpolation_frames` takes any render method). `scripts/export_animation.jl --format gif|mp4` turns the frames into a GIF/MP4 via `ffmpeg`; `scripts/animate_image_iterate.jl` builds per-map ImageIterate animations (the test suite `include`s it, so its `@quickactivate` only runs when executed as a script).

The render pipeline (`render_types.jl`, `render_pipeline.jl`):
- `Render(ifs, method; resolution, color, alpha, overlays)` is a value. `method` is a `RenderMethod` type — `Chaos`, `PointDeterministic`, `Inverse`, `Transformations`, or `ImageIterate{S}` — carrying only its own parameters and validating them in its constructor (so invalid renders cannot be built). `ImageIterate`'s source is a `PolygonSource`, a `FileSource`, or any `RenderMethod`. Overlays are `BasePolygon(polygon)` and `Axes()`. `format = PNG()` (default) or `SVG()` is a type parameter of `Render` (`Render{M,F}`); the constructor checks `supports(method, format)`, and only `Transformations` and `ImageIterate{PolygonSource}` support `SVG()`. Their SVG `draw` returns a `SegmentImage` (segments grouped by map, capped at 1_000_000 segments), `finish` turns it into the SVG document text (`drawing/svg.jl`; `color` = per-map strokes, `alpha` = no black background, overlays = base polygon lines and axes lines), and `save_render` writes it via `_save_svg`. SVG names end `_svg.svg` so PNG and SVG renders of one description have different file and record names; `render` reuse reads the SVG back as text. `render_transformations_svg` is a thin wrapper over this (always `force = true`).
- `render(r)` (or `render(ifs, method; kwargs...)`) runs `draw(method, r)` → `(image, limits)`, then `finish` (`colorize`, `overlay!` for each overlay, alpha mask), then `save_render`, and returns `(render, image, limits, outpath, record, cached)`. Per-method behavior is a method of `draw`, `colorize`, `seed_image` or `overlay!`; adding a method means one struct in `render_types.jl` plus its `draw` (and `DrWatson.allaccess`/`method_tag` for file naming) in `render_pipeline.jl`.
- `save_render` writes the PNG (placed by `outpath`) and a `.jld2` record (the `Render` and its `limits`, written with DrWatson's `tagsave`) that always lives in the data directory at `render_record_path(r)` = `data_output_dir("renders", <render name>.jld2)`; there is no `outpath` override for any JLD2 file. Animation frames mirror their PNG layout (`renders/frames/<sequence>/frame_NNNN.jld2`, via `render_record_path(png)`). `load_render(path)` reads a record (or the record of a default-located PNG or frame), so `render(load_render(path))` repeats a render and `render(Render(r; resolution = ...))` re-renders at another size. PNG names come from DrWatson `savename` over the method's `allaccess` fields. Point clouds live in one flat `points/` cache (`cached_points`; records hold `points`, `method`, `ifs`, `gitcommit`); `extend_fractal(Chaos, ifs, n)` continues a cached chaos game and stores it as the entry for the larger `npoints`, and `fractal_catalog()` lists the render and point records (`fractal_store.jl`, `save_fractal`, `load_fractal` and `fractal_data` are gone).
- `Chaos`/`PointDeterministic` point clouds are cached by `cached_points` as one flat file per cloud in `data_output_dir("points")`, named by `savename` over a config of `ifs_key(ifs)` plus every method field (e.g. `chaos_ifs=<key>_npoints=1000_seed=1_warmup=50.jld2`), keyed only on the IFS's maps and weights and every method field (not the name, resolution, color, alpha or overlays), so re-renders only redo rasterization.
- There is no `backend` argument in the render pipeline; the GPU code is kept but unwired.

Saving is DrWatson-managed in `output.jl`. Every file the package writes goes through it, so never call `save`/`write` directly in `src/`:
- `_resolve_output_path(outpath, default_name)` handles all `outpath`/`outdir` arguments. `nothing` means auto-named, relative paths go under `output_dir()`, and absolute paths are used as-is.
- `output_dir()` is DrWatson's `plotsdir()` and `data_output_dir()` is `datadir("sims")` for the active project; `with_output_root` (a ScopedValue) or `FRACTALS_OUTPUT_ROOT` override both (`<root>/plots`, `<root>/data/sims`). Generated data goes through `_save_data_text`.
- `_output_name` builds `savename` filenames, `_save_image`/`_save_svg` write via `safesave` (existing files become `_#N` backups), and the wrapper types plug into DrWatson's `_wsave` extension point. `save_render` writes through `_save_image_file` and `tagsave` instead (it overwrites).
- `_save_image` PNGs (used by the animation script) get two tEXt chunks (helpers in `utils.jl`): `fractals.source_limits` (the pixel-to-world framing) and `fractals.provenance` (JSON with the `gitdescribe` commit, cached for 2 s, the version, a timestamp and the params; read it back with `read_provenance`). SVGs carry the provenance as a leading comment.

GPU backends:
- `make_image`, `iterate_image`, and `rasterize_image_inversely` accept `backend=:cpu|:gpu|:auto`, routed through `_dispatch_backend` in `utils.jl`; the render pipeline does not pass it.
- The core package only stubs `_gpu_backend_available` (returns false) and the `_*_gpu` functions. The real kernels live in `ext/FractalsCUDAExt.jl`, which loads only when CUDA.jl is present; it is unwired and predates the slim `IFS` (it still references removed fields).
- GPU paths are expected to match CPU output; parity tests enforce this.

Parsing:
- `IFSParse.jl/` (`src/types.jl` IFSToken/IFSDefinition; `src/parse.jl` lexer → `_parse_ifs_tokens` → `IFSDefinition`, contractivity checks, weights, content key; `src/library.jl` the prebuilt library) works on plain equation matrices and has no dependency on `Fractals`; `Fractals` re-exports its public names, and the IFS-returning `parse_ifs_string`/`parse_ifs_file`/`get_ifs` wrappers are in `Fractals.jl/src/ifs_load.jl`. Non-contractive definitions are skipped with a warning. `.ifs` sources live in `ifs/` (`ifs_data_dir()`, the default `dir` regardless of the active project); `build_ifs_library()` saves each function system as its own JLD2 record in `data/sims/ifs/<file stem>/ifs_<key>.jld2` via `produce_or_load`, keyed by a hash of the equation matrix and weights only (name stored, not keyed). `list_ifs_files`, `list_ifs` and `get_ifs(name; file)` (`get_ifs_definition` in IFSParse) prefer prebuilt records and fall back to the `.ifs` files. The library dir (`datadir("sims", "ifs")`) follows the active project, so pass `libdir=` under `--project=Fractals.jl` if you want the repo-root library.
- Curated `.ifs` files live in `ifs/` at the repo root. Locate them with `ifs_data_dir(...)` (always the repo-root `ifs/`), not `projectdir("ifs")`, which follows the active project (`Fractals.jl/ifs` under `--project=Fractals.jl`).

## Load/compile performance notes

- `Fractals` deliberately avoids `Images` (huge load cost); PNG I/O uses `PNGFiles` via `_load_image`/`_save_image_file` in `utils.jl`, and pixel types come from `Colors`. Do not reintroduce `Images`/`ImageIO` as a dependency of `Fractals.jl`.
- There is no `PrecompileTools` workload: `Pkg.precompile()` of `Fractals` takes about 7 s, and the first render in a session (and the test suite) pays the compilation instead. A workload was tried and moved roughly 25–60 s of compilation from the tests into precompilation without removing it; see git history (`Fractals.jl/src/precompile.jl`) if you want it back.
- `test/runtests.jl` sets `Base.Experimental.@compiler_options optimize = 0 compile = min infer = false` for the test bodies; keep it, and do not add `@inferred`/allocation assertions to that file without scoping them elsewhere.
- In test bodies use `_with_tmpdir() do ... end` (defined in the header) instead of `mktempdir() do ... end`: the latter compiles a new `mktempdir` method per closure (~0.1 s each). Remaining test compile time is mostly Base generics (`==`, `isapprox`, `collect`) at test-only types plus `Test` machinery; to profile, wrap testsets with `Base.cumulative_compile_time_ns()` deltas or run with `--trace-compile=file --trace-compile-timing`.
