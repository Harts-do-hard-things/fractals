# Changelog

All notable changes to this project will be documented in this file.

This changelog follows the intent of Keep a Changelog and uses Semantic Versioning.

## [Unreleased]

### Added
- Rendering as values: `Render(ifs, method; resolution, color, alpha, overlays)` is run by `render(r)` or `render(ifs, method; kwargs...)`, which returns `(render, image, limits, outpath, record, cached)`. `Render(r; resolution = ...)` copies a render with fields changed.
  - Methods are the types `Chaos`, `PointDeterministic`, `Inverse` (with `divergence_scale` and `preview`), `Transformations` and `ImageIterate`, each validating its own parameters. `ImageIterate` is seeded by a `PolygonSource`, a `FileSource` or any other method; overlays are `BasePolygon` and `Axes`.
  - `render` reuses a matching saved PNG and record (`cached = true`); `force = true` redraws.
  - `save_render`, `load_render` and `render_record_path`: each render saves its PNG (under `plots/`) and a `.jld2` record (the `Render` and its limits, git-tagged) under `data/sims/renders/`.
  - File names come from DrWatson `savename` over each method's fields.
- Point clouds are cached independently of resolution and styling: `cached_points`, `ifs_key`, `extend_fractal(::Chaos, ifs, n)` (grow a cached chaos game) and `fractal_catalog()`, in `data/sims/points/`.
- `chaos_game(ifs, n)` returns the points; `deterministic_iterate(ifs, points, n)` and `make_image(points, limits)` take them explicitly.
- `IFSParse.jl`, a standalone package for the `.ifs` format with a DrWatson-managed JLD2 library of prebuilt definitions: `build_ifs_library`, `list_ifs_files`, `list_ifs`, `get_ifs`, `get_ifs_definition`. The curated `.ifs` files moved from `data/exp_raw/` to `ifs/`.
- `scripts/benchmark.jl`: opt-in BenchmarkTools benchmarks (`run_benchmarks`, `latest_benchmarks`, `compare_benchmarks`) saving to `data/sims/benchmarks/`; never run in CI or the tests.
- `scripts/export_animation.jl --format gif|mp4`.
- `interpolate_eq_matrix(...)` and `interpolate_ifs(...)` for animation-oriented transform interpolation (default `interpolation_mode = :rotation_scale`, `:linear` for coefficient-wise blending), and `render_interpolation_frames(left, right, method; frames, ...)` for numbered PNG sequences, now with any render method and returning `(outdir, paths, ts)`.
- `scripts/animate_image_iterate.jl` for per-map ImageIterate animations, and `ifs/imported_from_python.ifs`.
- DrWatson-managed output: the repo root is a DrWatson project (`fractals`, with `scripts/`); `output_dir(...)`, `data_output_dir(...)`, `with_output_root(f, root)` and `FRACTALS_OUTPUT_ROOT` control where files go (`plots/` for images, `data/sims/` for generated data). New dependencies: DrWatson and JLD2.

### Changed
- Breaking: `IFS` is now `(name, maps, weights, limits)`; it no longer stores `points` or `docs`, and the constructors lose `npoints` and `docs`.
- Breaking: select a definition from a `.ifs` file with `get_ifs(name; file)` or `parse_ifs_file(path)[i]` instead of `ifs_index`/`ifs_name`; `parse_ifs_string`/`parse_ifs_file` lose `npoints`.
- Breaking: all generated data lives under `data/sims/` (renders, points, the IFS library, benchmarks); `data_output_dir()` is `datadir("sims")`. `data/exp_pro` remains git-ignored.
- Breaking: output moved from a working-directory-relative `media/` to DrWatson's `plotsdir()` (`<repo>/plots/` with `--project=.`); relative paths keep their subdirectories, absolute paths are honored, and `outpath` defaults to `nothing` (auto-named). `export_animation` overwrites its output and defaults to `plots/animations/<basename>.<format>`.
- `render_transformations.jl` was split into `drawing/` helper files; `render_transformations_svg` remains the SVG writer.
- Breaking: transformation renders no longer accept `limits_mode`; they frame from derived polygon limits. Axis overlays no longer affect framing.
- Breaking (FractalsGUI): removed the unused `load_ifs_definition` and `load_ifs_definition!`; use `parse_ifs_definition_for_gui` and `load_ifs_definition_gui!`.
- Load and precompile times: `Fractals` no longer depends on `Images` (PNG I/O goes through `PNGFiles`), and the test suite runs its bodies with `optimize = 0, compile = min, infer = false`.
- Documentation moved from MkDocs to a Documenter.jl site in `docs/` (manual pages plus an API reference generated from docstrings); the build fails on undocumented exports, broken cross-references and failing doctests. New pages: Data layout and Benchmarks.
- The repository follows DrWatson's default project layout (`scripts/` replaces `Fractals.jl/bin/` for project scripts).

### Removed
- The CLI (`Fractals.jl/bin/fractals.jl`), the old `benchmark` command and suite with its CI job, and the CLI and benchmark manual pages.
- `render(input; method = ...)`, the `render_*` wrappers, the `RenderMethod` enum (with `Parallel` and `RenderTransformations`), `render_transformations_png` (use `render(ifs, Transformations(...); overlays = [...])`), and interactive prompting (`prompt_ifs_and_render`).
- `iterate!`, `iterate_parallel!`, `make_image(ifs)`, `render_fractal`, and the JLD2 record API (`save_fractal`, `load_fractal`, `fractal_record`, `fractal_data`): use render records and `cached_points`.
- `scripts/export_gif.jl` and `scripts/export_mp4.jl` (merged into `export_animation.jl`) and the `_#N` backups of exported animations.
- PNG provenance and source-limits chunks from `render` output (the render record carries them; `_save_image` and the animation script still write them).

### Fixed
- FractalsGUI: applying an edited matrix no longer overwrites the loaded baseline, so "Reset to Loaded" restores the file's definition; the previous temporary transformation SVG is deleted after each re-render, `Rsvg` is imported explicitly, and transformation SVGs no longer leak into `FractalsGUI.jl/test/media/`.

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
