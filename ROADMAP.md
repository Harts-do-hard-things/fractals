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
- [x] Add `Fractals/bin/fractals.jl` CLI with commands:
- [x] `render` (single render from matrix/IFS file)
- [x] `batch-render` (all definitions in a file)
- [x] `validate-ifs` (syntax/shape checks without rendering)
- [x] `benchmark` (run standard perf suite)

### Performance Baselines
- [x] Create benchmark suite with standard workloads.
- [x] Track `iterate!`, `iterate_parallel!`, `make_image`, `rasterize_image_inversely`.
- [x] Document target throughput and memory envelopes.
- [x] Add regression threshold checks in CI (report warnings/failures).

### Test Expansion
- [x] Add parser negative tests (malformed braces, mixed row widths, bad floats).
- [x] Add property tests for affine inverse round-trip.

---

## 61-90 Days: Visual Quality and Distribution

### Ecosystem/Release
- [x] Add curated `Fractals/data/*.ifs` library with metadata.
- [x] Add release checklist (version bump, changelog, benchmarks, docs updates).
- [ ] Tag stable `v1.0.0` when API/CLI contracts are locked.

### Release Checklist
- [x] Confirm working tree is clean and branch is up to date with target base branch.
- [x] Bump project version in package metadata as needed for the release.
- [x] Update changelog/release notes text to summarize user-visible API/CLI/docs changes.
- [x] Run full test suite:
  - `julia --startup-file=no --project=Fractals Fractals/test/runtests.jl`
- [x] Run benchmark suite and review target envelope status:
  - `julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl benchmark --profile medium --repeats 3`
- [x] If target envelopes are configured, run strict benchmark validation:
  - `julia --startup-file=no --project=Fractals Fractals/bin/fractals.jl benchmark --profile medium --repeats 3 --targets <path-to-targets.toml> --strict`
- [x] Verify docs impacted by the release are updated (`Fractals/DOCUMENTATION.md`, `docs/quick-recipes.md`, `docs/cli.md`, troubleshooting as needed).
- [x] Verify CLI help output matches docs for newly added or changed flags/options.
- [ ] Commit release-prep changes with clear message.
- [ ] Create and push release tag:
  - `git tag vX.Y.Z`
  - `git push origin vX.Y.Z`
- [ ] Draft and publish release notes (GitHub release entry or equivalent).

### Animation
- [ ] Add transform interpolation for animated sequences.
- [ ] Add frame renderer (`media/frames/...`) with deterministic naming.
- [ ] Add GIF/MP4 export helper scripts.

---

## Milestones and Exit Criteria

Note: milestone checks below are based on repository state; live CI pass/fail depends on the latest GitHub Actions runs.

### M1 (Day 30)
- [x] `render(...)` API merged and documented
- [x] Seeded reproducibility in place
- [ ] CI passing on supported platforms

### M2 (Day 60)
- [x] CLI commands usable end-to-end
- [x] Benchmarks and regression checks live
- [x] Parser negative/validation test coverage increased
- [ ] Snapshot image tests added

### M3 (Day 90)
- [ ] Enhanced rendering controls shipped
- [ ] Animation workflow documented and tested
- [ ] Release process established for regular tagged versions

---

## Suggested Priority Order (Backlog Top 10)
- [x] `render(...)` high-level API
- [x] Seed support for iteration methods
- [x] CI matrix with startup-file-safe test invocation
- [x] CLI `render`
- [x] Parser validation + negative tests
- [x] Benchmark harness
- [ ] Snapshot image tests
- [x] Batch rendering CLI
- [x] Curated `.ifs` dataset

---

## Optional Plan: GPU Acceleration for `make_image`

### Goal
- Add a GPU-accelerated `make_image` path while preserving current API behavior and CPU fallback.

### Step 1: Baseline CPU Performance
1. Measure current `make_image` runtimes at representative resolutions and point counts.
2. Capture where time is spent (mapping, accumulation, normalization).

### Step 2: Add Backend Abstraction
1. Add backend selection to `make_image` (for example: `backend=:cpu|:gpu|:auto`).
2. Keep existing CPU behavior unchanged as a reference implementation.

### Step 3: Select GPU Stack and Scope
1. Target `CUDA.jl` first for an MVP.
2. Restrict initial support to NVIDIA CUDA devices.
3. Keep clear CPU fallback or clear errors when CUDA is unavailable.

### Step 4: GPU-Friendly Data Layout
1. Convert points into dense arrays suitable for device kernels (e.g., `x` and `y` arrays).
2. Avoid scalar indexing patterns that degrade GPU performance.

### Step 5: GPU Rasterization Kernel
1. Implement kernel to map points to pixel coordinates.
2. Accumulate into an image buffer using atomic writes.
3. Start with `Float32` accumulation; evaluate alternatives after correctness.

### Step 6: GPU Normalization
1. Compute max on device.
2. Apply log normalization on device.
3. Copy to CPU only at API boundary (or offer optional GPU output mode).

### Step 7: Preserve Output Parity
1. Match CPU behavior for coordinate transform, clamping, and `[0, 1]` normalization.
2. Add deterministic parity tests (`isapprox`) against CPU output.

### Step 8: Fallback and Feature Gating
1. Implement `:auto` backend behavior (prefer GPU if available).
2. Ensure predictable behavior when GPU support is missing.

### Step 9: Benchmarks and Test Strategy
1. Add parity tests on small deterministic workloads.
2. Add performance benchmarks for larger workloads.
3. Keep CI CPU-only by default; run GPU tests behind an environment flag.

### Step 10: Optimization Pass
1. Tune launch configuration (threads/blocks).
2. Reduce atomic contention if needed.
3. Evaluate shared-memory or tiled accumulation strategies.

---

## Potential Improvement: GPU Inverse Iteration Preview

### Goal
- Add a GPU-accelerated inverse iteration path for fast preview rendering (targeting ~5 iterations at interactive rates on moderate resolutions).

### Mode Split
- `mode=:exact`:
- Track all corner branches with growth `K = 4 * (n_affine_maps^n_iterations)`.
- Use this mode when exact inverse-branch coverage is required.
- `mode=:preview`:
- Use a fixed-budget approximation for interactive rendering.
- Keep bounded state size for real-time responsiveness.

### Action Items
1. Add `inverse_iterate_gpu(ifs, resolution; n=5, mode=:preview)` entrypoint with CPU fallback.
2. Implement shared per-pixel validity mask buffers (`valid_a` / `valid_b` as `UInt8`).
3. Precompute and upload inverse map coefficients once per run.
4. Implement initialization kernel for per-pixel corner state.
5. Implement `mode=:exact` buffers and iteration:
6. Dynamic corner-count state with `K = 4 * (n_affine_maps^iter)` (or final-capacity allocation).
7. Ping-pong corner buffers `(2, K, H, W)` and bounds masking per iteration.
8. Add strict guardrails (`K_max`, memory checks) with clear errors when exact state is too large.
9. Implement `mode=:preview` buffers and iteration:
10. Fixed-budget corner state for interactive rendering.
11. Deterministic approximation policy for branch selection/retention.
12. Implement classification kernel:
13. black for 0 valid corners, gray for 1-3, white for all tracked corners valid.
14. Convert classification output to grayscale image.
15. Reuse GPU allocations across calls for preview performance.
16. Add profile-based preview settings (`256x256`, `384x384`, `512x512`).
17. Add tests for shape, class mapping, and deterministic behavior in both modes.
18. Add parity tests:
19. `mode=:exact` vs CPU reference on small resolutions/iterations.
20. `mode=:preview` stability tests under fixed seeds/settings.
21. Add benchmark command/profile to report per-stage timings for both modes.
22. Add docs for mode selection, complexity, limits, and recommended settings.

---

## Potential Improvement: GPU Acceleration for `iterate_image`

### Option A: Scatter/Atomic Kernel (Parity-first)

Goal:
- Preserve current `iterate_image` semantics with minimal algorithm changes.

Approach:
1. Launch GPU kernel over source pixels (or source-pixel/map pairs).
2. For each nonzero source pixel, apply each affine map and pixel transform.
3. Atomically accumulate into destination buffer.
4. For `colors=true`, atomically accumulate into 3 channel buffers (`R`, `G`, `B`).

Pros:
- Closest behavior to existing CPU implementation.
- Lower migration risk and simpler parity validation.

Cons:
- High atomic contention in dense image regions.
- Performance may flatten at higher resolutions/densities.

### Option B: Gather/Backward-Warp Kernel (Throughput-first)

Goal:
- Reduce contention by writing each destination pixel once.

Approach:
1. Launch GPU kernel over destination pixels.
2. Use inverse mapping logic to estimate source contributions per map.
3. Read source values, sum locally in registers/shared memory.
4. Write one final value per destination pixel (or RGB tuple for `colors=true`).

Pros:
- Avoids global atomic hotspots.
- Better scaling potential for dense images.

Cons:
- Higher implementation complexity.
- Requires careful parity checks and possibly interpolation/approximation choices.

---

## Planned GUI Milestone: Iteractable 3-Panel Desktop GUI

### Summary
- Build a new Gtk.jl desktop GUI (`Fractals/bin/gui.jl`) with three synchronized panels:
1. Editable transformation matrix panel (always 7 columns).
2. True SVG preview panel of current transformations.
3. Placeholder fractal panel with wiring hooks for future rendering.

- Add context-menu-driven loading that supports:
  - Quick-load from `./Fractals/data/*.ifs`
  - External `.ifs` file picker
  - Two-step definition selection for multi-definition files

- This milestone delivers GUI infrastructure and interaction model; fractal rendering panel remains placeholder but fully integrated in event flow.

### Architecture and File Layout
1. New launcher script:
  - `Fractals/bin/gui.jl`
  - Responsibilities:
    - Start Gtk app/window
    - Construct all panels and menu actions
    - Wire state updates and refresh pipeline

2. New GUI module source:
  - `Fractals/src/gui.jl`
  - Responsibilities:
    - App state model
    - Widget construction helpers
    - Menu/context actions
    - Matrix-table <-> model sync
    - SVG refresh/render pipeline
    - Placeholder fractal panel update hook

3. Package export wiring:
  - `Fractals/src/Fractals.jl`
  - Add `include("gui.jl")` and export GUI entrypoint (e.g. `launch_gui`).

4. Documentation updates:
  - `README.md` and `Fractals/DOCUMENTATION.md`:
    - How to launch GUI
    - `.ifs` load behavior
    - Panel behavior and current limitations (fractal pane placeholder)

### Public API / Interface Additions
1. New API entrypoint:
  - `launch_gui(; data_dir::AbstractString="Fractals/data")`
  - Opens main GUI window.
  - Optional `data_dir` for testing/custom datasets.

2. New CLI-style launcher script:
  - `julia --startup-file=no --project=Fractals Fractals/bin/gui.jl`

- No breaking changes to existing render/CLI APIs.

### State Model (Decision-Complete)
- Create a central mutable GUI state struct containing:

1. Current source metadata:
  - `source_file::Union{Nothing,String}`
  - `definition_index::Int`
  - `definition_name::String`

2. Current IFS definition data:
  - `eq_matrix::Matrix{Float64}` always normalized to 7 columns in UI state
  - `docs::String`

3. Derived render state:
  - `ifs::IFS`
  - `svg_string::String` (current transformation SVG markup)
  - `svg_temp_path::String` (for Gtk SVG widget source)
  - `fractal_placeholder_text::String` (status-only for v1)

4. Dirty/validation state:
  - `last_error::Union{Nothing,String}`
  - `is_valid::Bool`

- Normalization rule:
  - UI always presents 7 columns: `a11 a12 a21 a22 b1 b2 p`.
  - If loaded source had 6 cols, populate `p` using existing weight derivation logic so edits remain explicit and stable.

### GUI Layout and Behavior

#### Window
- Single top-level window with horizontal 3-panel split.

#### Panel 1: Matrix Panel (editable)
- Gtk table/grid with one row per transform.
- 7 editable numeric columns.
- Row add/remove controls.
- “Apply” + “Reset to loaded” actions.
- Validation feedback inline/status bar:
  - non-numeric
  - invalid dimensions
  - negative probability column (if present)
  - zero-row matrix

- On successful apply:
  - Rebuild current `IFS`
  - Refresh SVG panel
  - Trigger placeholder fractal panel update hook text

#### Panel 2: SVG Panel (true SVG)
- Use Gtk SVG-capable widget path (e.g., Rsvg-backed image/viewer in Gtk) to display real vector SVG.
- SVG generated via existing transformation pipeline (reuse `render_transformations_svg` logic).
- Refresh on:
  - file/definition load
  - matrix edits apply
  - row add/remove

#### Panel 3: Fractal Panel (placeholder with wiring)
- Placeholder widget with:
  - title
  - current source + definition
  - “Fractal preview not yet implemented” state message
  - hook status (“ready to render” if matrix valid)

- This panel subscribes to same state updates so future render integration only plugs into existing event path.

### Context Menu and Load Flow
- Right-click context menu on main window (and mirrored in menu bar if desired):

1. `Load from data/`
  - Dynamically lists `./Fractals/data/*.ifs`.
  - Selecting a file triggers definition picker dialog if multiple definitions found.
  - Two-step selection:
    - file selection
    - definition index/name selection dialog

2. `Open .ifs...`
  - Native file chooser for external path.
  - Same two-step definition selection flow.

3. `Reload current source`
  - Re-parse and refresh state.

4. `Clear to empty template`
  - Initializes one-row editable default matrix.

- Definition selection behavior:
  - Always explicit picker for multi-definition files.
  - Auto-select definition 1 only when file contains exactly one definition.

### Data and Parser Reuse
- Reuse existing parser APIs:
  - `parse_ifs_file`
  - `parse_ifs_definitions_file` (for names/docs and selection dialog metadata)

- Reuse existing transformation rendering path:
  - `render_transformations_svg` (or extracted in-memory variant to avoid unnecessary disk churn where feasible)

### Validation and Error Handling
1. Input validation before apply:
  - finite numeric values only
  - at least one row
  - exactly 7 columns in UI model
  - `p >= 0` for all rows

2. Error UX:
  - Non-blocking status bar + modal dialog for critical load/parse errors.
  - Keep last valid state when apply fails.

3. File handling:
  - Gracefully handle missing/unreadable files.
  - Defensive parsing with clear error propagation.

### Tests and Scenarios
- Add `Fractals/test/gui_smoke.jl` (or integrated guarded tests) with non-interactive coverage:

1. State/model tests:
  - 6-col source normalization to 7-col UI matrix.
  - Apply path rebuilds `IFS` and updates SVG string marker.
  - Validation rejects invalid numeric/probability inputs.

2. Load workflow tests:
  - Enumerate `Fractals/data/*.ifs` list.
  - Parse selected file and choose definition by index.
  - External path load behavior.

3. SVG generation tests:
  - Updated matrix changes SVG output deterministically (basic string/line count assertions).

4. GUI construction smoke test (headless-safe where possible):
  - Window/panel initialization without crash.
  - Event handlers registered.

- Manual QA checklist:
  - Right-click menu appears.
  - Data file load + definition selection works.
  - Matrix edits update SVG panel.
  - Placeholder panel reflects state changes.

### Dependencies and Build Notes
1. Add GUI dependencies to `Fractals/Project.toml`:
  - `Gtk` (or `Gtk4`) and SVG rendering dependency (Rsvg-capable path).

2. Keep startup recommendations:
  - Run GUI with `--startup-file=no` in docs examples to avoid local startup interference.

3. Platform notes:
  - Document native library requirements if Gtk/Rsvg backends require them.

### Acceptance Criteria
1. Launching `Fractals/bin/gui.jl` opens a 3-panel window.
2. Matrix panel is editable and always 7 columns.
3. SVG panel shows true SVG and refreshes after edits/loads.
4. Fractal panel exists as placeholder with active state wiring.
5. Context menu supports:
  - `.ifs` selection from `./Fractals/data`
  - external `.ifs` via file picker
  - multi-definition selection flow
6. Existing package APIs and CLI remain backward-compatible.
7. Basic automated smoke/model tests pass.

### Assumptions and Defaults
1. First milestone is desktop Gtk.jl (not web UI).
2. Fractal panel intentionally remains non-rendering placeholder in v1.
3. UI normalizes to 7-column matrix editing model.
4. SVG must be true vector display (not PNG fallback) for this milestone.
