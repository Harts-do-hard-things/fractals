# Simplification plan

A prioritized list of ways to shrink this repo, with an estimated effect for each one. This is a
plan only; none of it has been applied. Baseline is branch `claude/long-running-test-jfd`, which
is `julia-fractals-dev` plus the JLD2 store and the load/compile work.

## How the effects were estimated

Measured on this machine (Windows, Julia 1.12.5, quiet, warm caches):

| Quantity | Value | Breakdown |
|---|---|---|
| `using Fractals` | 2.3 s | Heaviest deps: SparseArrays 0.40 s (only via StatsBase), Pkg 0.36 s (via DrWatson and FileIO), ImageCore 0.15 s (via PNGFiles), JLD2 0.15 s, StaticArrays 0.15 s, JSON3 + StructTypes + Parsers ≈ 0.1 s |
| Fractals precompile after a src edit | ≈ 33 s | ≈ 7 s is the package itself, ≈ 26 s is the `PrecompileTools` workload; see the table below |
| Test run | ≈ 28 s wall | ≈ 2.3 s load, ≈ 2.6 s header (includes the CLI script and `scripts/animate_image_iterate.jl`), ≈ 22 s inside testsets, ≈ 65% of it compilation |

Each workload block was timed in a fresh process with the workload switched off. The precompile
seconds below are those timings scaled to the measured 26 s total:

| Workload block | ≈ precompile s | Main contents |
|---|---|---|
| `render` over matrix, IFS and file inputs | 8.6 | PNG save for 4 pixel types, provenance JSON, PNG tEXt chunks, `savename`, `safesave` |
| JLD2 records | 6.9 | `save_fractal`, `load_fractal`, `fractal_data` (`produce_or_load`), catalog |
| Parse + chaos game | 4.7 | Parser, `iterate!`, `make_image`, `iterate_image` |
| Interpolation + `export_animation` | 2.5 | |
| ImageIterate (`:polygon` and `:file`) | 1.6 | |
| Transformations (SVG + PNG) | 1.1 | |
| PointDeterministic + Inverse | 0.7 | |

Test seconds per area, measured: transformations ≈ 5.0, interpolation and animation ≈ 3.9,
JLD2 ≈ 2.6, ImageIterate API ≈ 2.3, DrWatson output ≈ 1.8, interactive ≈ 1.1, everything else
(core math, parser, rasterize, inverse, backends) ≈ 5.

**Line counts are counts. Every time effect is an estimate,** reasoned from the tables above.
Treat ±50% as normal. "Lines" means tracked source, tests, docs and config, excluding manifests.

## Summary

| # | Change | Lines removed (src / test / other) | Load | Precompile | Tests | Breaks API? |
|---|---|---|---|---|---|---|
| **Tier 1: safe quick wins (landed on `julia-fractals-dev` as 0746414)** | | | | | | |
| 1 | Delete `Julia.vim`, unused DelimitedFiles, junk `[extras]` | 0 / 0 / ~145 | – | – | – | no |
| 2 | One validation helper for the `render_*` wrappers; inline single-caller `_impl`s | ~100 / 0 / 0 | – | −0.3 s | – | no |
| 3 | `chaos.jl`: one RNG code path instead of 4 copies | ~30 / 0 / 0 | – | −0.2 s | – | no |
| 4 | Delete `CHANGELOG.md` (git log suffices) | 0 / 0 / ~70 | – | – | – | no |
| | **Tier 1 total** | **≈ 350 lines** | ~0 | ≈ −0.5 s | ~0 | none |
| **Tier 2: moderate refactors** | | | | | | |
| 5 | Drop PNG provenance, tEXt chunk I/O, `gitdescribe` cache, JSON3, Dates | ~330 / ~200 / ~60 docs | −0.1 s | −2 to −3 s | −0.5 s | `read_provenance` gone |
| 6 | Drop StatsBase: a small weighted sampler in `chaos.jl` | ~10 net / 0 / 0 | **−0.4 s (≈17%)** | −0.5 s | – | `IFS.weights` becomes `Vector{Float64}`; seeded output changes, so regenerate the 3 snapshots |
| 7 | Plain overwrite instead of `safesave` backups and `_wsave` wrapper types | ~80 / ~40 / ~20 | – | −0.5 to −1 s | −0.3 s | `_#N` backups stop |
| 8 | Output root via DrWatson: `with_output_root` switches the active project; drop the ScopedValue and `FRACTALS_OUTPUT_ROOT` | ~45 / ~25 / ~25 | – | – | – | `FRACTALS_OUTPUT_ROOT` gone; `with_output_root` kept but no longer task-scoped |
| 9 | Remove interactive prompting (`interactive.jl`, `input_fn` everywhere) | ~250 / ~120 / ~30 | – | −0.3 s | −1.1 s | `prompt_ifs_and_render` and `input_fn` gone |
| 10 | `image_source`: keep `:polygon` and `:file`, drop the dispatch table | ~150 / ~300 / ~30 | – | −0.5 s | −1 s | 3 `image_source` values gone |
| 11 | Inverse: keep `:exact`, drop `:preview` | ~100 / ~120 / ~20 | – | −0.1 s | −0.1 s | `mode=` gone |
| 12 | Transformations: SVG only (the GUI uses only SVG) | ~200 / ~150 / ~20 | – | −0.5 s | −1.5 s | `render_transformations_png` gone |
| 13 | Delete `scripts/export_gif.jl` and `export_mp4.jl` (they duplicate `export_animation`) | 0 / 0 / ~110 | – | – | – | no |
| 14 | Simplify the JLD2 record shape (concrete NamedTuples, not nested `Dict{String,Any}`) | ~40 / ~10 / 0 | – | **−3 s** | −1 s | file format version bump |
| 15 | Drop the formatter CI gate (keep the formatter as a local tool) | 0 / 0 / ~40 | – | – | – | no |
| | **Tier 2 total** | **≈ 1,350 src, ≈ 970 test, ≈ 350 other** | ≈ −0.5 s | ≈ −8 to −10 s | ≈ −6 s | yes, see column |
| **Tier 3: drastic, needs your yes** | | | | | | |
| 16 | Delete the GPU path: CUDA extension, `backend=` everywhere, parity tests | ~1,100 / ~350 / ~80 | – | −0.5 s | −0.3 s | `backend=` gone |
| 17 | Delete benchmark infra and the regression CI job | ~460 / ~50 / ~190 | – | – | – | CLI `benchmark` gone |
| 18 | Delete the CLI (`bin/fractals.jl`) | ~340 / ~200 / ~150 | – | – | −1 s (header) | CLI gone |
| 19 | Delete or split out `FractalsGUI.jl` | ~720 / ~200 / ~60 | – | – | – | GUI gone |
| 20 | Cut tests to ≈ 800 lines (one test per method plus math and parser) | 0 / ~2,000 / 0 | – | – | −8 to −10 s | no, but much less coverage |
| 21 | Docs: README plus docstrings only, drop Documenter | 0 / 0 / ~1,100 | – | – | – | no |
| 22 | Merge `Fractals.jl/` into the root project (one Project, one Manifest) | ~0 / 0 / ~30 | – | – | – | layout changes |
| | **Tier 3 total** | **≈ 2,600 src, ≈ 2,800 test, ≈ 1,600 other** | – | ≈ −0.5 s | ≈ −10 s | yes |

**If you take everything:** about 14k tracked lines become about 4.5k.

| | Now | All tiers (estimate) |
|---|---|---|
| `using Fractals` | 2.3 s | ≈ 1.8 s |
| Precompile | 33 s | ≈ 20–23 s |
| Test run | 28 s | ≈ 10 s |
| CI jobs | 5 | 1 |
| Manifests to keep resolved | 3 | 1 |

The workload would shrink further as features go, because fewer keyword combinations need
precompiling.

**If you take only Tiers 1 and 2:** ≈ 2,700 fewer lines, `using` ≈ 1.8 s, precompile ≈ 23–25 s,
tests ≈ 22 s, and no loss of GPU, the GUI, the CLI or the docs.

---

## Details

### Tier 1: safe quick wins

1. **Dead files and dependencies.** `Fractals.jl/Julia.vim` (134 lines) is an editor syntax file.
   `using DelimitedFiles` in `bin/fractals.jl` is never called. The `[extras]` entries Debugger,
   Revise, StatProfilerHTML and BenchmarkTools are listed in no target. Each needs one manifest
   re-resolve.
2. **Shared render validation.** `render.jl` (728 lines) repeats the npoints, warmup, resolution
   and backend checks at the top of five wrappers. One `_check_common(...)` helper replaces them.
   `_render_image_iterate_impl` and `_render_transformations_impl` each have exactly one caller,
   so inline them.
3. **`chaos.jl`.** `iterate!` and `iterate_parallel!` hold four near-identical `maps[sample(...)]`
   branches: seeded or unseeded, with or without warmup. Use one RNG,
   `seed === nothing ? default_rng() : Xoshiro(seed)`, and one loop. Seeded-reproducibility tests
   already exist.
4. **CHANGELOG.** For a solo hobby repo, commit messages carry this information. Also drop the
   "add under `[Unreleased]`" rule from CLAUDE.md.

### Tier 2: moderate refactors

5. **Provenance machinery** (`utils.jl` lines 44–243, plus provenance in `output.jl`). It
   hand-rolls PNG chunk parsing and writing with a CRC32 table, embeds a JSON payload, and caches
   `gitdescribe` for 2 s.
   - **Two consumers:** `read_provenance`, and `image_source = :file`, which reads the world-space
     framing back out of the PNG.
   - **Replacement:** now that JLD2 records exist, `render(...; save_data = true)` already stores
     limits, params and the git commit next to the PNG. Let `:file` accept a `.jld2` record, or
     an explicit `limits =`, instead of parsing PNG chunks.
   - **Effect:** JSON3 and Dates leave the core deps, and the output part of the 8.6 s render
     workload block shrinks.
6. **StatsBase** is used only for `sample(indices, Weights)` and the `Weights` type, and it alone
   pulls in SparseArrays, which is 0.40 s of the 2.3 s load. A cumulative-sum plus
   `searchsortedfirst` sampler, or an alias table for many maps, is about 15 lines and at least as
   fast in the hot loop.
   - **Cost:** seeded output changes, so regenerate the 3 snapshots with
     `test/generate_snapshots.jl`. `fractal_store.jl`'s `extend_fractal` also calls `sample`.
7. **`safesave` backups.** The `_PNGOutput`, `_TextOutput` and `_MovedFileOutput` wrapper types
   exist only to route writes through DrWatson's `_wsave` so `safesave` can make `_#N` backups.
   Overwrite instead, or keep `safesave` only for JLD2 data files where it comes free. Keep
   `savename`.
8. **Output root via DrWatson.** Today two custom mechanisms do one job: the `_OUTPUT_ROOT`
   ScopedValue behind `with_output_root`, and the `FRACTALS_OUTPUT_ROOT` env var, both read by
   `_output_root_override()` in `output.jl`. DrWatson already answers "where does output go":
   `plotsdir()` and `datadir()` resolve against `projectdir() == dirname(Base.active_project())`.
   So let DrWatson own it:
   - **`output.jl`:** `output_dir(args...) = plotsdir(args...)` and
     `data_output_dir(args...) = datadir("exp_pro", args...)`, with no override logic. Delete
     `_OUTPUT_ROOT`, `_output_root_override` and the env-var parsing.
   - **`with_output_root(f, root)`:** becomes a ~6-line helper. It makes `root` the active
     project for the duration of `f`, which is the same thing `@quickactivate` does in scripts.
     DrWatson then resolves `plots/` and `data/exp_pro/` under `root`, and the previous project is
     restored in a `finally`. It can call `Pkg.activate(root; io = devnull)`, since Pkg is
     already loaded through DrWatson, or set `Base.ACTIVE_PROJECT[]` directly. A `root` with no
     `Project.toml` works, because DrWatson only takes the directory name.
   - **Tests and docs:** the `runtests.jl` header, `generate_snapshots.jl` and the `docs/make.jl`
     doctest setup activate a `mktempdir()` project once, after `using Fractals`, instead of
     setting `ENV["FRACTALS_OUTPUT_ROOT"]`. The "output root precedence" testset shrinks to one
     check that `output_dir() == plotsdir()` under a temp project.
   - **CI:** the smoke-test render drops the env var. It either writes into the repo's
     git-ignored `plots/`, or passes an absolute `--output` path.
   - **Side effect to fix with it:** provenance and JLD2 records take the git commit from
     `projectdir()` (`_gitcommit` in `output.jl`). Under a temp project that finds no repo and
     records `nothing`. Point it at the package's repo (`dirname(pkgdir(Fractals))`), which is
     the right commit anyway: it describes the code that produced the file.
   - **Caveats:**
     - The active project is process-global, not task-scoped like the ScopedValue, so two tasks
       can't render to different roots at once. Nothing here does that.
     - Don't `using` new packages inside `f`, because code loading follows the active project.
     - The precompile workload on `claude/long-running-test-jfd` uses `with_output_root(scratch)`.
       Switching the active project inside a precompile process works, but it should set
       `Base.ACTIVE_PROJECT[]` directly rather than go through `Pkg.activate`.
   - **Docs:** update the mentions in `README.md`, `docs/src/manual/output.md`,
     `docs/src/manual/development.md`, `docs/src/api.md` and CLAUDE.md.
9. **Interactive prompting.** `interactive.jl` (206 lines), the `input_fn = readline` keyword
   threaded through the `render_*` signatures (24 uses in `render.jl`), and the stdin branch of
   `_resolve_render_input`. `ifs_index` and `ifs_name` already cover the use case; an `.ifs` path
   with neither could simply list the names in the error.
10. **`image_source` dispatch** (`image_iterate.jl`: the `ImageSourceRequest` struct, choices
    tuple, dispatch Dict and five resolvers). `:chaos`, `:point_deterministic` and `:inverse` only
    call other render paths; a user can pass that image directly. Keep `:polygon` and `:file`.
11. **Inverse `:preview` mode.** A speed hack next to `:exact`. Removing it drops the `mode`
    keyword plumbing and the GPU capacity helper.
12. **Transformations PNG raster path.** `render_transformations.jl` (810 lines) has its own line
    rasterizer for PNG output. The GUI calls only `render_transformations_svg`. Its tests are the
    slowest area (≈ 5 s), and about a third of that is the PNG path.
13. **Export scripts.** `scripts/export_gif.jl` and `export_mp4.jl` wrap `export_animation` with
    near-identical bodies.
14. **JLD2 record shape** (new, from this branch's work). Its workload block costs ≈ 6.9 s, the
    second largest, because JLD2 compiles per field type and the record nests
    `Dict{String,Any}`s. Store one concrete NamedTuple, or a small struct with plain-array fields
    plus a `meta` NamedTuple. That should roughly halve its compile cost. Bump
    `_FRACTAL_RECORD_FORMAT`.
15. **Formatter CI gate.** A pinned formatter version plus a failing CI job is heavy for one
    author. Keep `.JuliaFormatter.toml` for local use.

### Tier 3: drastic, needs your yes

16. **GPU.** `ext/FractalsCUDAExt.jl` (734 lines), `_dispatch_backend` and its stubs, the
    `backend` keyword (≈ 85 mentions across 8 src files), and the parity and backend tests.
    - **Timing effect:** small, because the extension only loads with CUDA.
    - **Maintenance effect:** the biggest of any item. Every render signature loses a keyword, and
      every new feature stops needing a GPU twin.
    - Keep it if you render at very large resolutions.
17. **Benchmarks.** `bench/` (457 lines), the `benchmark` CLI command, `docs/manual/benchmarks.md`,
    and the regression CI job. Latency thresholds on shared CI runners are noisy by nature.
18. **CLI.** `render`, `batch-render` and `validate-ifs` are thin wrappers over `render(...)`, and
    `scripts/` plus the REPL cover the same ground. Deleting it also removes the CLI module from
    the test header.
19. **GUI.** Gtk and Rsvg are the heaviest install in the repo, and it is the third manifest to
    re-resolve on every dependency change. You could move it to its own repo instead.
20. **Test suite.** 3,576 + 142 lines. After Tiers 2 and 3 much of it targets removed options.
    Keep:
    - AffineMap and inverse properties, and the IFS constructor
    - Parser, positive and negative cases
    - Chaos and deterministic correctness, and seeded reproducibility
    - One render per method, and the 3 snapshots
    - The JLD2 round-trip
21. **Docs site.** 10 manual pages plus `api.md`. Dropping Documenter also removes the rule that
    every export needs an `api.md` entry, and a CI job.
22. **One package.** Only worth doing after 16, 17 and 19. Then the repo is one package plus
    `data/`, `scripts/` and `test/`.

## Dependencies

| Dep | Used for | Load cost | Verdict |
|---|---|---|---|
| StaticArrays | `AffineMap` | 0.15 s | keep |
| DrWatson | `savename`, `datadir`/`plotsdir`, `produce_or_load`, `wload`, `safesave` | 0.36 s (Pkg) + JLD2 | **keep**: it is the data layer for the JLD2 store |
| JLD2 | fractal records | 0.15 s | keep (arrives with DrWatson anyway) |
| PNGFiles (+ ImageCore) | PNG read/write | 0.15 s | keep |
| Colors | pixel types, `distinguishable_colors` | 0.04 s | keep |
| FileIO | non-PNG image fallback, DrWatson's save path | small | keep |
| StatsBase | `sample`, `Weights` | **0.40 s** (SparseArrays) | **drop** (#6) |
| JSON3 | provenance payload, benchmark JSON | ≈ 0.1 s | drop with #5 and #17 |
| Dates | provenance timestamp | – | drop with #5 |
| Statistics, TOML | benchmarks only | – | drop with #17 |
| DelimitedFiles | nothing | – | drop now (#1) |
| PrecompileTools | the workload | – | keep |
| CUDA (weak) | extension | – | drop with #16 |

## Suggested order

1. Tier 1 as one commit.
2. #6 StatsBase, then regenerate snapshots. This gives the biggest load win for little code.
3. #14 JLD2 record shape, which gives the biggest precompile win.
4. #5, #7 and #8 together, to collapse the output layer onto DrWatson and JLD2.
5. #9 to #13.
6. Tier 3, item by item, as you decide.

## Open questions

1. Do you use the GPU backend (#16)?
2. Do you use the GUI (#19)? Keep it here, move it out, or delete it?
3. Is "params, limits and commit live in the JLD2 sidecar" an acceptable replacement for PNG
   provenance (#5)?
4. Which `image_source` values do you actually use (#10)?
5. Keep the published docs site (#21)? Keep the CLI (#18)?
6. How much test coverage do you want to keep (#20)?
