# Simplification plan

Scope: reading-only audit of branch `claude/long-running-test-jfd`. Nothing was changed except this file. Line counts are from `wc -l`; "lines removed" are estimates.

## Current size (tracked, excluding manifests/assets/LICENSE)

| Area | Lines |
|---|---|
| `Fractals.jl/src` (17 files) | ~4,900 (largest: render_transformations 810, render 706, interpolation 509, types 452, image_iterate 433, utils 401, inverse 358, output 251, ifsparser 249, interactive 206) |
| `Fractals.jl/test/runtests.jl` (73 top-level testsets, + snapshots + generate_snapshots.jl 58) | 3,555 |
| `Fractals.jl/ext/FractalsCUDAExt.jl` | 734 |
| `Fractals.jl/bin/fractals.jl` (CLI) | 343 |
| `Fractals.jl/bench/` (suite 389 + perf_targets.toml 68) | 457 |
| `FractalsGUI.jl` (src 721, test 201) | ~940 |
| `docs/` (10 manual pages ~1,000, api.md 101, make.jl 47) | ~1,200 |
| `scripts/` (5 files) | ~456 (animate_image_iterate.jl alone is 282) |
| CI `ci.yml` (5 jobs), CHANGELOG 63, CLAUDE.md 102, README 192, Julia.vim 134 | ~700 |

The code is not tangled, but it is sized like a library with users. The bulk is (a) the GPU/CUDA path across 4 render methods, (b) output/provenance machinery, (c) a test file about as big as all of `src`, (d) docs/CI/benchmarks.

---

## Tier 1 — safe quick wins (no behavior users would miss)

1. **Delete `Fractals.jl/Julia.vim`** (134 lines, unrelated editor syntax file). Risk none; no API.
2. **Remove unused deps/imports.** `DelimitedFiles` is imported (`bin/fractals.jl:4`) but never called (no `readdlm/writedlm` anywhere) and is in `[deps]`/`[compat]`. Drop it. `Statistics` is only used in `bench/` (not in `[deps]`; fine). Drop unused `using Dates: now` in `Fractals.jl:10` if only provenance uses it (see Tier 2). ~5 lines + one manifest re-resolve x3 (root, GUI, docs). Risk low.
3. **Drop the `[extras]` junk** in `Fractals.jl/Project.toml:30-34` (Debugger, Revise, StatProfilerHTML, BenchmarkTools listed as extras but not in a target). Risk none.
4. **Collapse duplicated validation in `render_*`.** `render.jl` repeats the same npoints/warmup/resolution/backend checks at the top of each wrapper (e.g. `render.jl:36-60`, and again in 93, 159, 217, 389). Extract one `_check_common(; resolution, backend, ...)` helper. ~60 lines, risk low, no API change.
5. **Merge `_resolve_render_input` and `_render_*_impl` split** where the `_impl` function has exactly one caller (`_render_image_iterate_impl` 217 -> 389, `_render_transformations_impl` 437 -> 514). Inline. ~40 lines, no API change.
6. **Remove `Parallel`/`Chaos` duplication in `chaos.jl`.** `iterate!` (lines 15-23) and `iterate_parallel!` (56-67) each have 4 copies of `maps[sample(...)](x)` for seeded/unseeded and in/out of loop. Use one RNG (`seed === nothing ? Random.default_rng() : Xoshiro(seed)`), one code path. ~30 lines. Risk low (seeded reproducibility tests exist at runtests.jl:722).
7. **Delete `CHANGELOG.md`** (63 lines; git history + the commit log are enough for a solo project) and the "add under [Unreleased]" rule in CLAUDE.md. No API effect.
8. **Delete `Fractals.jl/test/snapshots` redundancy**: keep at most 2-3 snapshots (one per render method is plenty); `generate_snapshots.jl` shrinks accordingly. Only 3 files appear tracked; if more exist locally, prune. Low risk.
9. **Remove `.git-blame-ignore-revs`, `.gitattributes` tweaks** if the formatter pin goes away (Tier 2 #14). Trivial.

Tier 1 total: roughly 350-400 lines, plus 134 for the vim file.

---

## Tier 2 — moderate refactors

10. **Remove provenance + gitdescribe + PNG tEXt chunk code.** `utils.jl:44-232` (CRC32 table, chunk parser/writer, `_embed_png_text_chunks`, `_read_png_text_chunk`, source-limits payload) plus `output.jl:13-15,133-187` (`_in_git_repo`, `_gitcommit` + TTL cache, `_provenance_payload`, `read_provenance`) plus `interpolation.jl:312`. This is ~330 lines hand-rolling PNG chunk IO, a JSON payload, and a 2-second git cache. Two uses exist: provenance (nice-to-have) and *source-limits* (functional: `ImageIterate` with `:file` source reads framing limits back out of a PNG).
   - Replace provenance with a sidecar-free approach: nothing (filename already encodes params via `savename`), or a one-line `.json` sidecar with `JSON3`/`TOML`.
   - Replace source-limits by requiring `limits` to be passed explicitly or defaulting to the unit box; or write limits in the filename.
   - Removes tests `runtests.jl:2809-2893` (PNG Source Limits Metadata, Render Outputs Embed..., ~85 lines) and the provenance part of the 150-line output testset (1390-1543).
   - Also removes `JSON3` from the core deps and `gitdescribe` use. **Lines removed: ~330 src + ~200 test. Risk medium** (the :file source path loses automatic framing). **Public API: removes `read_provenance`.**
11. **Replace `ScopedValue` output root + `with_output_root` + `FRACTALS_OUTPUT_ROOT`** (`output.jl:13-60`, exported `with_output_root`, `output_dir`, `data_output_dir`, `ifs_data_dir`). Keep a plain `OUTPUT_DIR = Ref(plotsdir())` or just `plotsdir()` and let tests `cd(mktempdir())`. ~40 lines src, simplifies every test header. Risk low-medium. API: drop `with_output_root`.
12. **Drop `safesave` backups (`_#N` files).** `output.jl` `_PNGOutput/_TextOutput/_MovedFileOutput` structs + `DrWatson._wsave` method overloading (lines 189-251) exist only to route saves through DrWatson. For a hobby tool, overwrite (or timestamp) with plain `save`/`write`. ~80 lines, drops `DrWatson.safesave`. Risk low. Keep `savename` if you like the naming (or inline a 5-line version and drop the DrWatson dep from the *core* package entirely; the root project can keep it for `@quickactivate`).
13. **Collapse the image-source dispatch table** (`image_iterate.jl:83-275`: `ImageSourceRequest` struct, `_IMAGE_SOURCE_CHOICES`, `_IMAGE_SOURCE_DISPATCH` Dict, 5 `_resolve_image_source_from_*`, two context resolvers, `_resolve_image_source`). Five sources is too many for one feature. Keep `:polygon` and `:file` (the two that produce interesting results from outside the pipeline); `:chaos`, `:point_deterministic`, `:inverse` just call `render_*` internals and can be done by the user passing the image they got. ~150 lines src, ~300 lines tests (`runtests.jl:2628-2808` and parts of 2182-2627, which is a 450-line testset for one function). Risk medium. API: `image_source` choices shrink.
14. **Inverse `:exact` vs `:preview`** (`inverse.jl:164-358`): pick one mode (keep `:exact`; `:preview` is a speed hack). Removes `mode` kwarg plumbing through `render_inverse`, GPU capacity function (`_inverse_gpu_capacity`, 258), and tests 1027-1160. ~100 lines src, ~120 test. Risk low-medium; API: drops `mode=`.
15. **Remove interactive prompting.** `interactive.jl` (206 lines: `_prompt_int/_string/_choice`, `_select_ifs_definition`, `prompt_ifs_and_render`), the `input_fn = readline` kwarg threaded through every `render_*` (7 signatures in `render.jl`), `_resolve_render_input` stdin branch, and "Interactive And Main Entrypoints" testset (3080-3143) + "Render IFS Selection" (2939-3006, keep the by-index/name part). Replace with `ifs_index`/`ifs_name` only (already supported). ~250 src + ~120 tests. Risk low. API: removes `prompt_ifs_and_render`, `input_fn`.
16. **Trim `examples.jl`/`main()`**: exported `HEIGHWAY_DRAGON`, `EISENSTEIN`, `main` (45 lines) overlap with `data/exp_raw/*.ifs`. Keep constants only if tests need them; move to tests. ~30 lines.
17. **Cut `render_transformations.jl` (810 lines).** It renders SVG and PNG diagrams of maps (own line rasterizer `_draw_line!`, axes overlays, preset polygons, color-by-map). It's the largest src file and is tangled with the GUI (GUI previews via SVG). Options: keep SVG only (drop the PNG raster path ~lines 570-700 + `render_transformations_png` + the "Transformation Raster Renders Follow Iterative Geometry" tests ~50 lines) and the `RGBA/Gray` drawing. ~200 src + ~150 tests. Risk medium. API: removes `render_transformations_png`.
18. **Interpolation/animation (509 lines + scripts).** `interpolation.jl` (interpolate, frame renderer, `export_animation` shelling out to ffmpeg) + `scripts/export_gif.jl`, `export_mp4.jl` duplicate each other and `export_animation`. Delete the two scripts (112 lines) and keep `export_animation`; test blocks 485-710 (~225 lines, 3 testsets incl. the script test that `include`s `animate_image_iterate.jl`) shrink by half. Risk low.
19. **Drop the formatting CI gate + pinned JuliaFormatter** (`ci.yml` `format` job, `.JuliaFormatter.toml`, CLAUDE.md formatting section). Keep the formatter locally if you like. Removes one CI job.

Tier 2 total: roughly 1,100 src lines and 1,000 test lines.

---

## Tier 3 — drastic options (need owner sign-off)

20. **Delete the GPU path entirely**: `ext/FractalsCUDAExt.jl` (734), `_gpu_backend_available/_dispatch_backend/_validate_backend` in `utils.jl:~290-338`, `backend=` kwarg in every render function (79 `backend` mentions in src), `_iterate_image_gpu`, `_rasterize_image_inversely_gpu*`, `_inverse_gpu_capacity`, GPU parity tests (887-901, 970-995, 1162-1209), "Shared Backend Dispatch" + "Backend Dispatch Consistency" (854-886, 1122-1161), `[weakdeps]`/`[extensions]` and the CUDA CI/test opt-in. **~1,100 src/ext lines + ~350 test lines.** Risk: loses GPU rendering, which is the only performance headroom for huge resolutions. If CPU threaded rendering is fast enough for your use, this is the single biggest bloat cut. API: removes `backend` everywhere.
21. **Delete benchmark infra**: `bench/` (457), `benchmark` CLI command (part of `bin/fractals.jl`), `benchmark-regression` CI job (ci.yml 148-211), `docs/manual/benchmarks.md` (120), and `FRACTALS_RUN_CLI_BENCH_TESTS` tests. Hardware-dependent latency thresholds in CI (`perf_targets.toml`, fail ratio 1.30) are flaky-by-design on shared runners. If you want numbers, keep a 30-line `bench.jl` with BenchmarkTools, uncommitted CI. **~650 lines + a CI job.** Risk low practically; API: CLI `benchmark` removed.
22. **Delete the CLI (`bin/fractals.jl`, 343 lines) + `cli.md` (151) + CLI tests.** `render`, `batch-render`, `validate-ifs`, `benchmark` are all thin over `render(...)`. Scripts in `scripts/` and the REPL cover the use. ~500 lines + tests.
23. **Delete `FractalsGUI.jl`** (940 lines; Gtk + Rsvg are heavy, slow to precompile, and the third manifest to keep resolved), plus the `gui-test` CI job. Or move it to a separate repo. Impact: lose the visual IFS editor. Alternative: keep but accept it isn't tested in CI.
24. **Cut the test suite to ~800 lines.** Today it re-tests the same behavior across CPU/GPU/backends/colors/alpha/snapshots. Keep: AffineMap inverse properties, IFS constructor, parser (positive+negative), chaos/deterministic correctness, one render test per method, 2-3 snapshots. Delete everything else (~2,500 lines). Risk: less safety net when changing things, but most of the deleted tests cover options Tier 2/3 remove anyway.
25. **Collapse docs to README + 2 pages** (`render_methods.md` + `ifs_format.md`), drop Documenter entirely (`docs/`, `docs` CI job, `checkdocs=:exports` requirement for docstrings/`api.md`, jldoctests, docs manifest). Remaining docs are the docstrings + README. **~1,100 lines**; also removes the rule that every export must be in `api.md`. Risk none functionally; you lose a published site.
26. **Merge `Fractals.jl` into the root project** (one `Project.toml`, no `[sources]`, no 3-manifest re-resolve dance). Only worth it if 20/22/23 are all done; then the repo is one package + data + scripts.

---

## Dependencies

| Dep | Where used | Verdict |
|---|---|---|
| Images | Umbrella: provides `Gray`, `RGB`, `RGBA`, `Colorant`, `color_type` (`image_iterate.jl:51-58`, `render_transformations.jl:584-675`, `utils.jl:375-398`). No actual image algorithms (`imresize` etc.) used | **Replace** with `ImageCore` (or `ColorTypes` + `FixedPointNumbers`); saves most of Images' precompile/load time. |
| Colors | `distinguishable_colors` in `_map_colors` (`utils.jl:343`), `RGB` types | **Replace/keep minimal**: either keep `Colors` (small) or use a fixed 12-color palette / HSV hue cycle in 5 lines and drop the call. |
| FileIO | `load` (`image_iterate.jl:117`), `save` (`output.jl:203`) | **Replace** with `PNGFiles.save/load` directly (FileIO pulls in format detection/registry). Read side for `:file` source only needs PNG. |
| StatsBase | `sample(...)` with `Weights` (`chaos.jl`, `types.jl:104,184,331,340,444`) | **Drop**: a 10-line cumulative-weights sampler (`searchsortedfirst(cumsum, rand())`) is also faster in the hot loop. Removes `Weights` from the `IFS` struct type. |
| JSON3 | provenance (`output.jl:168,185`), source-limits payload (`utils.jl:71,79`), bench JSON | **Drop** with Tier 2 #10 (or use stdlib `TOML`/hand-written JSON line). |
| Statistics | bench only (`bench/benchmark_suite.jl:6`) | **Drop** with benchmarks; otherwise it is stdlib/cheap. |
| DelimitedFiles | imported in `bin/fractals.jl:4`, no calls | **Drop now.** |
| Dates | `now()` for provenance timestamp | **Drop** with #10. |
| TOML | bench targets only | **Drop** with #21. |
| DrWatson | `plotsdir`, `datadir`, `projectdir`, `savename`, `safesave`, `gitdescribe`, `_wsave` overload | **Keep** (reviewer override). The audit originally said drop it from core, but the JLD2 work on this branch uses DrWatson's `produce_or_load`/`wload`/`tagsave` as the data layer, so the package is now leaning *into* it. Keep the core DrWatson calls and trim the custom wrappers instead: the `_wsave` overloads, `safesave` for images, and the provenance machinery that `tagsave` now covers for data. |
| StaticArrays | `AffineMap` SMatrix/SVector | **Keep** (core to performance). |
| LinearAlgebra, Random, Printf, Base.Threads | stdlib | **Keep.** |
| CUDA (weak) | extension | **Drop** if Tier 3 #20 approved. |
| Gtk, Rsvg (GUI) | `FractalsGUI.jl` | **Drop** with #23. |

After Tier 1+2+3 the core deps would be: StaticArrays, ImageCore, PNGFiles, (optionally ColorTypes) plus stdlibs.

---

## Suggested target shape

```
fractals/
  Project.toml            # single package (or keep Fractals.jl/ if you prefer)
  README.md               # usage + IFS format + 3 examples (absorbs 2 manual pages)
  src/
    Fractals.jl           # module, include list, ~25 exports
    types.jl              # AffineMap, IFS (~400)
    chaos.jl              # one RNG path (~70)
    deterministic.jl      # ~50
    rasterize.jl          # ~110
    inverse.jl            # :exact only (~150)
    image_iterate.jl      # :polygon/:file sources (~250)
    transformations.jl    # SVG only (~500)
    interpolation.jl      # ~400 incl. ffmpeg export
    ifsparser.jl          # ~250
    render.jl             # ~300 (shared validation helper)
    output.jl             # ~60 (plain save, savename inline)
  test/runtests.jl        # ~800 lines, 2-3 snapshots
  data/exp_raw/*.ifs
  scripts/                # render_examples.jl, animate_image_iterate.jl
  .github/workflows/ci.yml # one `test` job
```

Estimated size: ~3,000 src lines (from ~4,900 + 734 ext), ~800 test lines (from 3,555), no GUI/CUDA/bench/docs-site. Roughly 11k -> 4.5k tracked source lines.

Suggested order: Tier 1 -> #15, #14, #12, #11 -> #10 -> dependency swaps (StatsBase, Images->ImageCore, FileIO->PNGFiles) -> Tier 3 per owner answers.

---

## Open questions for the owner

1. Do you actually use the GPU backend? If not, #20 is the biggest win (~1,450 lines). If yes, keep it but consider keeping only `iterate`/`make_image` kernels.
2. Do you use the Gtk GUI regularly? If not, delete or move to its own repo (#23).
3. Is the PNG-embedded provenance/limits worth keeping, or is "params in the filename" enough? Does `:file` image-iterate need auto-framing from the PNG?
4. Is `image_iterate` the feature you care most about (it takes ~430 src + ~450 test lines + an animation script)? Which of the 5 sources do you use?
5. Keep the published docs site, or are README + docstrings enough?
6. Do you want a CLI at all, or only Julia scripts?
7. Is anything outside this repo (notebooks, other repos, blog posts) calling `read_provenance`, `with_output_root`, `prompt_ifs_and_render`, or `backend=`? (You said nothing depends on it; confirm.)
8. Keep CI at all? A single `test` job is the practical minimum; benchmark/format/docs/gui jobs are optional.
9. Willing to lose the 3-snapshot visual regression tests in favor of property checks (non-empty, coverage ratio, determinism with seed)?
10. Single package at root vs. current `Fractals.jl/` + DrWatson root layout?
