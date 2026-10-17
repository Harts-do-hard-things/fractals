# Changelog

All notable changes to this project will be documented in this file.

This changelog follows the intent of Keep a Changelog and uses Semantic Versioning.

## [Unreleased]

### Changed
- `save_fractal`, `load_fractal`, `fractal_record` and `fractal_data` are removed: point clouds are cached by `cached_points` and renders by render records. `extend_fractal(Chaos, ifs, n)` extends a cached chaos game into the entry for the larger `npoints`, and `fractal_catalog(; dir)` lists render and point records.
- Rendering is redesigned around values: `Render(ifs, method; resolution, color, alpha, overlays)` is run by `render(r)` or `render(ifs, method; kwargs...)`, which returns `(render, image, limits, outpath, record)`. Methods are the types `Chaos`, `PointDeterministic`, `Inverse`, `Transformations` and `ImageIterate` (seeded by `PolygonSource`, `FileSource` or any method); overlays are `BasePolygon` and `Axes`. `save_render` writes the PNG plus a same-stem `.jld2` record and `load_render` reads it back. File names come from DrWatson `savename` over each method's fields.
- `render_interpolation_frames(left, right, method; frames, ...)` takes any render method and returns `(outdir, paths, ts)`.
- Select a definition from a `.ifs` file with `get_ifs(name; file)` or `parse_ifs_file(path)[i]` instead of `ifs_index`/`ifs_name`.
- `scripts/export_gif.jl` and `scripts/export_mp4.jl` are merged into `scripts/export_animation.jl --format gif|mp4`; `export_animation` overwrites an existing output instead of keeping a `_#N` backup.

### Removed
- `render(input; method = ...)`, the `render_*` wrappers, the `RenderMethod` enum (with `Parallel` and `RenderTransformations`), `render_fractal`, and interactive prompting (`prompt_ifs_and_render`).
- The CLI (`Fractals.jl/bin/fractals.jl`) and the benchmark suite with its CI job and manual pages.
- PNG provenance and source-limits chunks from `render` output (the render record carries them; `_save_image` and the animation script still write them).

### Added
- `chaos_game(ifs, n)` returns the points, and `IFS` no longer stores `points` or `docs` (breaking). `iterate!`, `iterate_parallel!`, the `npoints`/`docs` constructor keywords and `make_image(ifs)` are gone; `make_image(points, limits)`, `deterministic_iterate(ifs, points, n)` and `save_fractal(ifs, points)` take the points explicitly.
- `IFSParse` submodule (`Fractals.jl/src/ifs_parse.jl`) with a DrWatson-managed JLD2 library of prebuilt IFS definitions: `build_ifs_library`, `list_ifs_files`, `list_ifs`, `get_ifs`. `.ifs` files moved from `data/exp_raw/` to `ifs/`.
- `interpolate_eq_matrix(...)` and `interpolate_ifs(...)` helpers for animation-oriented transform interpolation.
- `render_interpolation_frames(...)` for deterministic transformation-frame rendering into numbered PNG sequences; it also supports `render_method=Chaos`.
- `interpolation_mode=:rotation_scale` (default) for `interpolate_eq_matrix`, `interpolate_ifs`, and `render_interpolation_frames`: interpolates each map's 2x2 part as rotation plus uniform scale, falling back to linear; `:linear` keeps the previous behavior.
- `scripts/animate_image_iterate.jl`: builds a per-map ImageIterate animation (frames + GIF) from a GrayA seed image.
- `ifs/imported_from_python.ifs`: IFS definitions imported from the original Python fractal classes.
- PNG outputs now embed source-limits metadata so `image_source=:file` can reuse the original pixel-to-world framing when that metadata is present.
- DrWatson-managed output: the repo root is now a DrWatson project (`fractals`, with `scripts/`), and `Fractals` depends on DrWatson for saving.
  - `output_dir(...)`, `with_output_root(f, root)`, and the `FRACTALS_OUTPUT_ROOT` environment variable control where files go.
  - Omitting `outpath` names files with `savename` from the render parameters.
  - All saves use `safesave`, so existing files are kept as `name_#N` backups.
  - PNGs embed a `fractals.provenance` chunk (git commit, version, timestamp, parameters), readable with `read_provenance(path)`; SVGs carry it as a comment.
- JLD2 data records: `save_fractal`, `load_fractal`, `fractal_record`, `fractal_data` (cached with DrWatson `produce_or_load`), `render_fractal` (re-rasterize saved points at any resolution without recomputing), `extend_fractal` (resume and accumulate a chaos game), and `fractal_catalog` (list saved records).
  - Records hold plain arrays only, so they load with `JLD2.load`/`wload` without Fractals, and carry DrWatson's `gitcommit`.
  - `render(...; save_data = true)` also writes a record named like the image under `data_output_dir()`.
  - New dependency: JLD2.

### Changed
- Load and precompile times: `Fractals` no longer depends on `Images` (PNG I/O goes through `PNGFiles` directly; only `Colors`/`ColorTypes` are needed for pixel types), roughly halving `using Fractals`. A `PrecompileTools` workload now precompiles the common render, parser, and PNG paths, so package precompilation takes longer once but first calls and the test suite are much faster. Non-PNG image paths still go through `FileIO` and need a backend such as `ImageIO` installed by the caller.
- Test suite runs its (run-once) test bodies with `optimize = 0, compile = min, infer = false` to avoid spending most of its time in compilation.
- Repository follows DrWatson's default project layout.
  - The curated `.ifs` library moved from `Fractals.jl/data/` to `data/exp_raw/`; use `ifs_data_dir(...)` to locate it (the GUI's default `data_dir` now does).
  - Project scripts moved from `Fractals.jl/bin/` to `scripts/` with `@quickactivate "fractals"`: `animate_image_iterate.jl`, `export_gif.jl`, `export_mp4.jl`, `generate_rotation_examples.jl`. `Fractals.jl/bin/fractals.jl` remains the package CLI.
  - Relative `fractals.jl benchmark --json` paths resolve under DrWatson's `datadir("exp_pro")` via the new `data_output_dir(...)`, and existing files get `_#N` backups.
  - `.gitignore` is DrWatson's template, with `/data` narrowed to `data/sims` and `data/exp_pro` so `data/exp_raw` stays tracked.
- Documentation moved from MkDocs to a Documenter.jl site in `docs/` (`docs/make.jl`, pages in `docs/src/`), built in CI without deployment.
  - The former `Fractals.jl/DOCUMENTATION.md` is split into manual pages (render methods, animation, output files, IFS text format, development) plus an API reference generated from docstrings; `mkdocs.yml` is removed.
  - Every exported name now has a docstring, and the docs build fails on undocumented exports, broken cross-references, or failing doctests.
- Breaking: output moved from a working-directory-relative `media/` to DrWatson's `plotsdir()` (`<repo>/plots/` with `--project=.`).
  - Relative paths keep their subdirectories under `plots/` instead of being flattened into `media/`, and absolute paths are now honored as given.
  - `outpath` defaults are `nothing` (auto-named) instead of fixed names like `media/render.png`.
  - `render_interpolation_frames` writes to `plots/frames/<savename>/` by default.
  - `export_animation`'s `outpath` is optional (default `plots/animations/<basename>.<format>`).
  - CLI `--out` defaults to an auto-generated name, and `batch-render --out-dir` defaults to `plots/batch`.
- Breaking API change: transformation renders no longer accept `limits_mode`; PNG now frames from derived polygon limits by default, and SVG now uses the same derived polygon limits behavior via `polygon_limits_iterations`.
- Transformation axis overlays now render within the existing `ifs.limits` projection instead of affecting the visible framing.
- Breaking (FractalsGUI): removed the unused `load_ifs_definition` and `load_ifs_definition!`; use `parse_ifs_definition_for_gui` and `load_ifs_definition_gui!`.

### Fixed
- FractalsGUI: applying an edited matrix no longer overwrites the loaded baseline, so "Reset to Loaded" restores the file's definition.
- FractalsGUI: the previous temporary transformation SVG is deleted after each re-render, and `Rsvg` is imported explicitly.
- Removed transformation SVGs leaked into `FractalsGUI.jl/test/media/` and ignored `FractalsGUI.jl/media` and `FractalsGUI.jl/test/media`.
- `prompt_ifs_and_render` rejects non-positive point counts and image sizes (falling back to the default) instead of crashing after iterating, no longer swallows unrelated errors as "invalid input", and renders through `render_chaos`/`render_point_deterministic`.

## [1.0.0] - 2026-03-02

### Added
- High-level `render(...)` API covering chaos, point-deterministic, image-iterate, and inverse workflows.
- CLI entrypoint (`Fractals.jl/bin/fractals.jl`) with `render`, `batch-render`, `validate-ifs`, and `benchmark`.
- Benchmark suite and regression target checking support.
- Curated `Fractals.jl/data/*.ifs` dataset.
- Polygon seed preset selection (`initial_polygon`) for transformation renders and image-iterate polygon source.
- Expanded parser negative tests and affine inverse property tests.

### Changed
- `ImageIterate` polygon seed generation now consistently uses IFS limits (`polygon_limits_mode=:default` is treated as `:ifs` for polygon source).
- Documentation and recipes aligned with current API/CLI behavior and media output defaults.

### Fixed
- Multiple validation and error-path consistency issues across API/CLI entrypoints.
- Roadmap status normalization to remove stale completion contradictions.
