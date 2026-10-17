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
| 5a | Drop PNG provenance, tEXt chunk I/O, `gitdescribe` cache, JSON3, Dates | ~330 / ~200 / ~60 docs | −0.1 s | −2 to −3 s | −0.5 s | `read_provenance` gone |
| 5b | Rebuild `_save_render_output` on DrWatson: one `Dict{String,Any}` config, `savename(...; accesses)`, `tagsave` of a JLD2 sidecar; `render` loads `.jld2` fractals (builds on 5a) | ~150 more / ~80 / ~30 | – | −1 to −2 s more | −0.5 s | `render_fractal` folds into `render`; every PNG gets a `.jld2` next to it |
| 6 | Drop StatsBase: a small weighted sampler in `chaos.jl` | ~10 net / 0 / 0 | **−0.4 s (≈17%)** | −0.5 s | – | `IFS.weights` becomes `Vector{Float64}`; seeded output changes, so regenerate the 3 snapshots |
| 7 | Plain overwrite instead of `safesave` backups and `_wsave` wrapper types | ~80 / ~40 / ~20 | – | −0.5 to −1 s | −0.3 s | `_#N` backups stop |
| 8 | Delete `with_output_root`, `output_dir`, `data_output_dir` and `FRACTALS_OUTPUT_ROOT`; call DrWatson's `plotsdir()` / `datadir()` directly | ~60 / ~40 / ~30 | – | – | – | 3 exports and the env var gone |
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

5. **Output and provenance layer.** Two steps; b) builds on a).

   **a) Drop the provenance machinery** (`utils.jl` lines 44–243, plus provenance in `output.jl`). It
   hand-rolls PNG chunk parsing and writing with a CRC32 table, embeds a JSON payload, and caches
   `gitdescribe` for 2 s.
   - **Two consumers:** `read_provenance`, and `image_source = :file`, which reads the world-space
     framing back out of the PNG.
   - **Replacement:** now that JLD2 records exist, `render(...; save_data = true)` already stores
     limits, params and the git commit next to the PNG. Let `:file` accept a `.jld2` record, or
     an explicit `limits =`, instead of parsing PNG chunks.
   - **Effect:** JSON3 and Dates leave the core deps, and the output part of the 8.6 s render
     workload block shrinks.

   **b) Rebuild `_save_render_output` on DrWatson, and load fractals back from JLD2.**
   - **Why it specializes today:** the function body is short (`render.jl` lines 7–20). The cost
     is its callers.
     - Six call sites (five `render_*` wrappers plus `render_fractal`) each pass a
       differently-shaped `params` NamedTuple and a different `name_keys` tuple.
     - Every distinct NamedTuple type gets its own `merge`, `NamedTuple{name_keys}(params)`,
       `_output_name` and `_provenance_payload` instances. The `@nospecialize` annotations limit
       inference but not the per-type `merge`/constructor/kwcall code.
     - Downstream sits `_save_image` → `_PNGOutput` → `DrWatson._wsave` overload → PNG write →
       re-read the file → splice tEXt chunks → rewrite.
   - **New shape:** one concrete config type, DrWatson's native `Dict{String,Any}`, built with
     `@strdict`. Then exactly one instance of everything after the call site.

     ```julia
     # render_chaos (each render_* does the same, with its own variables):
     method = "chaos"
     config = @strdict method npoints warmup resolution color alpha backend
     final_outpath = _save_render_output(outpath, img, ifs, ifs.limits, config,
                                         ["npoints", "resolution", "color", "alpha"])

     # render.jl: the whole save path
     function _save_render_output(outpath, img, ifs::IFS, limits,
                                  config::Dict{String,Any}, name_keys::Vector{String})
         name = savename(_savename_prefix(ifs.name), _name_values(config), "png";
                         accesses = name_keys)                       # DrWatson picks the keys
         png = _resolve_output_path(outpath, name)
         _save_image_file(png, img)                                  # plain PNGFiles write
         record = fractal_record(ifs; limits, method = config["method"], params = config)
         tagsave(_sidecar_path(png), record; safe = true, warn = false)  # gitcommit via DrWatson
         return png
     end

     _sidecar_path(png) = first(splitext(png)) * ".jld2"            # foo.png ↔ foo.jld2
     _name_values(c) = Dict(k => (v isa Tuple ? join(v, "x") : v) for (k, v) in c)  # 512x512
     ```
   - **Loading serialized fractals:** add a `.jld2` branch to
     `_resolve_render_input(input::AbstractString)`. It returns `load_fractal(path).ifs`, which
     is the full IFS with maps, weights, limits and points.
     - `render("foo.jld2"; method = :chaos, resolution = (2048, 2048))` then re-renders a saved
       fractal. With saved points and no `npoints` given, it rasterizes them instead of re-running
       the chaos game. `render_fractal` becomes that one branch and its export goes away.
     - `image_source = :file` reads its framing from the PNG's sidecar
       (`wload(_sidecar_path(png))["limits"]`) instead of a PNG tEXt chunk.
     - `read_provenance(png)` becomes `wload(_sidecar_path(png))`, so the function can go too.
     - The same applies to SVG transformations and interpolation frames: they write a sidecar
       instead of an XML comment or tEXt chunk.
   - **What goes away, on top of a):**
     - The `_PNGOutput` and `_TextOutput` wrapper types and their `DrWatson._wsave` overloads
     - The NamedTuple `merge` and `name_keys` juggling
     - `_output_name`, because `savename(...; accesses)` does the key selection
     - `_tag_record!` and the 2-second `_gitcommit` cache, because `tagsave` adds `"gitcommit"`
     - `render_fractal`
     - The PNG re-read-and-rewrite on every save
   - **Effect:**
     - About 150 src and 80 test lines on top of a).
     - The render block of the precompile workload (≈ 8.6 s) drops to one instance per pixel type
       for the PNG write, plus one each for `savename`, `tagsave` and JLD2 save of
       `Dict{String,Any}`. JLD2's share is already paid by the fractal store.
     - Estimated −1 to −2 s precompile beyond a), and fewer new compiles in tests, since a new
       keyword combination no longer creates a new NamedTuple type downstream.
   - **Costs and choices:**
     - **Disk:** every render writes a `.jld2` next to its PNG. With points included, a
       1-million-point chaos render is about 16 MB. Default to saving only maps, weights, limits
       and config (a few KB), and store points only with `save_points = true` (today's
       `save_data = true`). Re-rendering a sidecar without points re-runs the chaos game from the
       saved seed.
     - **Location:** the sidecar goes next to the PNG in `plots/`, not in `data/exp_pro/`. That
       keeps `foo.png` ↔ `foo.jld2` a pure path rule even for absolute `outpath`s, at the price of
       data files in `plots/`.
     - **`tagsave` cost:** it runs `gitdescribe` on every save (≈ 60 ms, it scans the work tree).
       Fine for single renders, noticeable in batch renders and interpolation frames. If that
       matters, keep a cached commit and pass it as a pre-tagged `"gitcommit"` key, which
       `tagsave` won't overwrite.
     - **Seed:** the config should carry `seed` so a points-free sidecar can reproduce the image.
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
8. **Output root: delete it and call DrWatson directly.** `with_output_root` (the `_OUTPUT_ROOT`
   ScopedValue), the `FRACTALS_OUTPUT_ROOT` env var, `_output_root_override()`, and the
   `output_dir` / `data_output_dir` wrappers exist only to redirect output somewhere other than
   the active project. Remove all of them, with no replacement helper.
   - **`src/`:** every `output_dir(...)` becomes `plotsdir(...)`, and every
     `data_output_dir(...)` becomes `datadir("exp_pro", ...)`. That covers `_resolve_output_path`,
     `_save_data_text`, `interpolation.jl`, `render.jl`, `interactive.jl` and (on
     `claude/long-running-test-jfd`) `fractal_store.jl`. Drop the three names from the `export`
     list and `docs/src/api.md`. `ifs_data_dir` stays, since it is about inputs, not output.
   - **Tests and docs:** delete the `ENV["FRACTALS_OUTPUT_ROOT"] = mktempdir()` lines in the
     `runtests.jl` header, `generate_snapshots.jl` and `docs/make.jl`.
     - Under `--project=Fractals.jl`, output lands in `Fractals.jl/plots` and
       `Fractals.jl/data/exp_pro`, which `.gitignore` already ignores.
     - Assertions use `plotsdir(...)` instead of `output_dir(...)`.
     - The "output root precedence" testset is deleted, and the `with_output_root(d) do` wrappers
       around other testsets are dropped.
     - **Trade-off:** test runs leave files in those ignored folders, and the `safesave` tests
       must not assume an empty directory. Use unique names, or `rm` the file first. If #7 lands
       first, the backup tests go away.
   - **CI:** the smoke-test render drops `FRACTALS_OUTPUT_ROOT` and writes into the runner
     checkout's ignored `plots/`.
   - **Precompile workload** (on `claude/long-running-test-jfd`): pass absolute `outpath`s inside
     its `mktempdir()` scratch dir instead of `with_output_root(scratch)`. Absolute paths already
     bypass `plotsdir()`.
   - **Docs:** remove the mentions in `README.md`, `docs/src/manual/output.md`,
     `docs/src/manual/render_methods.md` and CLAUDE.md.
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
| JSON3 | provenance payload, benchmark JSON | ≈ 0.1 s | drop with #5a and #17 |
| Dates | provenance timestamp | – | drop with #5a |
| Statistics, TOML | benchmarks only | – | drop with #17 |
| DelimitedFiles | nothing | – | drop now (#1) |
| PrecompileTools | the workload | – | keep |
| CUDA (weak) | extension | – | drop with #16 |

## Suggested order

1. Tier 1 as one commit.
2. #6 StatsBase, then regenerate snapshots. This gives the biggest load win for little code.
3. #14 JLD2 record shape, which gives the biggest precompile win.
4. #5a, #5b, #7 and #8 together, to collapse the output layer onto DrWatson and JLD2.
5. #9 to #13.
6. Tier 3, item by item, as you decide.

## Open questions

1. Do you use the GPU backend (#16)?
2. Do you use the GUI (#19)? Keep it here, move it out, or delete it?
3. Is "params, limits and commit live in the JLD2 sidecar" an acceptable replacement for PNG
   provenance (#5a), and are `.jld2` sidecars next to every PNG acceptable (#5b)?
4. Which `image_source` values do you actually use (#10)?
5. Keep the published docs site (#21)? Keep the CLI (#18)?
6. How much test coverage do you want to keep (#20)?
