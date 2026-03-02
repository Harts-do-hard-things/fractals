# Changelog

All notable changes to this project will be documented in this file.

This changelog follows the intent of Keep a Changelog and uses Semantic Versioning.

## [Unreleased]

### Added
- Placeholder section for upcoming changes after `1.0.0`.

## [1.0.0] - 2026-03-02

### Added
- High-level `render(...)` API covering chaos, point-deterministic, image-iterate, and inverse workflows.
- CLI entrypoint (`Fractals/bin/fractals.jl`) with `render`, `batch-render`, `validate-ifs`, and `benchmark`.
- Benchmark suite and regression target checking support.
- Curated `Fractals/data/*.ifs` dataset.
- Polygon seed preset selection (`initial_polygon`) for transformation renders and image-iterate polygon source.
- Expanded parser negative tests and affine inverse property tests.

### Changed
- `ImageIterate` polygon seed generation now consistently uses IFS limits (`polygon_limits_mode=:default` is treated as `:ifs` for polygon source).
- Documentation and recipes aligned with current API/CLI behavior and media output defaults.

### Fixed
- Multiple validation and error-path consistency issues across API/CLI entrypoints.
- Roadmap status normalization to remove stale completion contradictions.
