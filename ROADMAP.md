# Fractals Project Roadmap (30/60/90)

## North Star
- Deliver a fast, reproducible, and easy-to-use Julia fractal engine with a stable API, CLI workflows, and production-quality output.

## Current Status
- [x] README updated for Julia workflows
- [x] Core documentation added (`Fractals/DOCUMENTATION.md`)
- [x] Tests fixed and expanded
- [x] Image output defaults routed to `media/`

---

## 0-30 Days: Stabilize and Standardize

### API and UX
- [x] Add high-level `render(...)` entrypoint that wraps parser + iteration + rasterization.
- [x] Define method enum/options (`chaos`, `parallel`, `deterministic`, `inverse`) with consistent argument names.
- [x] Add argument validation and clear error messages for bad `eq` shapes and invalid map indices.

### Reproducibility
- [x] Add optional RNG seed support to `iterate!`.
- [x] Add optional RNG seed support to `iterate_parallel!`.
- [x] Add tests proving same-seed reproducibility for single-threaded iteration.

### Quality and CI
- [x] Add CI workflow for Windows + Linux with Julia matrix.
- [x] Run tests with `--startup-file=no` in CI.
- [x] Add lint/format check (`JuliaFormatter`) gate.

### Docs
- [x] Add "Quick Recipes" page: render from matrix, render from `.ifs`, deterministic preview.
- [x] Add troubleshooting notes for startup-file and cache/permission issues.

---

## 31-60 Days: Tooling and Performance

### CLI
- [ ] Add `Fractals/bin/fractals.jl` CLI with commands:
- [ ] `render` (single render from matrix/IFS file)
- [ ] `batch-render` (all definitions in a file)
- [ ] `validate-ifs` (syntax/shape checks without rendering)
- [ ] `benchmark` (run standard perf suite)

### Performance Baselines
- [ ] Create benchmark suite with standard workloads.
- [ ] Track `iterate!`, `iterate_parallel!`, `make_image`, `rasterize_image_inversely`.
- [ ] Document target throughput and memory envelopes.
- [ ] Add regression threshold checks in CI (report warnings/failures).

### Test Expansion
- [ ] Add parser negative tests (malformed braces, mixed row widths, bad floats).
- [ ] Add property tests for affine inverse round-trip.
- [ ] Add low-res golden-image snapshot tests for known fractals.

---

## 61-90 Days: Visual Quality and Distribution

### Rendering Features
- [ ] Add colormap support (grayscale + perceptual maps).
- [ ] Add gamma/exposure controls.
- [ ] Add supersampling / anti-aliasing path.
- [ ] Add optional 16-bit export pipeline.

### Animation
- [ ] Add transform interpolation for animated sequences.
- [ ] Add frame renderer (`media/frames/...`) with deterministic naming.
- [ ] Add GIF/MP4 export helper scripts.

### Ecosystem/Release
- [ ] Add curated `Fractals/data/*.ifs` library with metadata.
- [ ] Add release checklist (version bump, changelog, benchmarks, docs updates).
- [ ] Tag stable `v1.0.0` when API/CLI contracts are locked.

---

## Milestones and Exit Criteria

### M1 (Day 30)
- [ ] `render(...)` API merged and documented
- [x] Seeded reproducibility in place
- [ ] CI passing on supported platforms

### M2 (Day 60)
- [ ] CLI commands usable end-to-end
- [ ] Benchmarks and regression checks live
- [ ] Parser and snapshot test coverage increased

### M3 (Day 90)
- [ ] Enhanced rendering controls shipped
- [ ] Animation workflow documented and tested
- [ ] Release process established for regular tagged versions

---

## Suggested Priority Order (Backlog Top 10)
- [ ] `render(...)` high-level API
- [x] Seed support for iteration methods
- [ ] CI matrix with startup-file-safe test invocation
- [ ] CLI `render`
- [ ] Parser validation + negative tests
- [ ] Benchmark harness
- [ ] Snapshot image tests
- [ ] Colormaps + gamma
- [ ] Batch rendering CLI
- [ ] Curated `.ifs` dataset
