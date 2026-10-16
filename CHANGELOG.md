# Changelog

All notable changes to this project will be documented in this file.

This changelog follows the intent of Keep a Changelog and uses Semantic Versioning.

## [Unreleased]

### Added
- `interpolate_eq_matrix(...)` and `interpolate_ifs(...)` helpers for animation-oriented transform interpolation.
- `render_interpolation_frames(...)` for deterministic transformation-frame rendering into numbered PNG sequences; it also supports `render_method=Chaos`.
- `interpolation_mode=:rotation_scale` (default) for `interpolate_eq_matrix`, `interpolate_ifs`, and `render_interpolation_frames`: interpolates each map's 2x2 part as rotation plus uniform scale, falling back to linear; `:linear` keeps the previous behavior.
- `Fractals.jl/bin/animate_image_iterate.jl`: builds a per-map ImageIterate animation (frames + GIF) from a GrayA seed image.
- `Fractals.jl/data/imported_from_python.ifs`: IFS definitions imported from the original Python fractal classes.
- PNG outputs now embed source-limits metadata so `image_source=:file` can reuse the original pixel-to-world framing when that metadata is present.
- DrWatson-managed output: the repo root is now a DrWatson project (`fractals`, with `scripts/`), and `Fractals` depends on DrWatson for saving.
  - `output_dir(...)`, `with_output_root(f, root)`, and the `FRACTALS_OUTPUT_ROOT` environment variable control where files go.
  - Omitting `outpath` names files with `savename` from the render parameters.
  - All saves use `safesave`, so existing files are kept as `name_#N` backups.
  - PNGs embed a `fractals.provenance` chunk (git commit, version, timestamp, parameters), readable with `read_provenance(path)`; SVGs carry it as a comment.

### Changed
- Breaking: output moved from a working-directory-relative `media/` to DrWatson's `plotsdir()` (`<repo>/plots/` with `--project=.`).
  - Relative paths keep their subdirectories under `plots/` instead of being flattened into `media/`, and absolute paths are now honored as given.
  - `outpath` defaults are `nothing` (auto-named) instead of fixed names like `media/render.png`.
  - `render_interpolation_frames` writes to `plots/frames/<savename>/` by default.
  - `export_animation`'s `outpath` is optional (default `plots/animations/<basename>.<format>`).
  - CLI `--out` defaults to an auto-generated name, and `batch-render --out-dir` defaults to `plots/batch`.
- Breaking API change: transformation renders no longer accept `limits_mode`; PNG now frames from derived polygon limits by default, and SVG now uses the same derived polygon limits behavior via `polygon_limits_iterations`.
- Transformation axis overlays now render within the existing `ifs.limits` projection instead of affecting the visible framing.

### Fixed
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
