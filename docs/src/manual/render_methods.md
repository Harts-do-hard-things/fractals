# Render methods

A render is a value. [`Render`](@ref)`(ifs, method; resolution, color, alpha, overlays)` bundles
an [`IFS`](@ref), a [`RenderMethod`](@ref) (each method is its own type carrying only its own
parameters), the image size, whether to color by map, whether zero pixels become transparent,
and any overlays. [`render`](@ref) runs it: draw with the method, color, overlay, apply
alpha, then save the PNG and a render record (in `data/sims/renders/`). It returns
`(render, image, limits, outpath, record, cached)`.

```julia
using Fractals

ifs = IFS(HEIGHWAY_DRAGON; name = "Heighway Dragon")

out = render(ifs, Chaos(npoints = 200_000); resolution = (1024, 1024), outpath = "render.png")
out.outpath   # <repo>/plots/render.png
out.record    # <repo>/data/sims/renders/Heighway_Dragon_chaos_npoints=200000_1024x1024.jld2

# The same thing with the description built first:
r = Render(ifs, Chaos(npoints = 200_000); resolution = (1024, 1024))
out = render(r; outpath = "render.png")
```

Method constructors validate their own fields, so an invalid description throws
`ArgumentError` before anything is drawn:

```julia
Chaos(npoints = 0)                  # ArgumentError: npoints must be > 0, got 0
Render(ifs, Chaos(); resolution = (0, 10))   # ArgumentError
```

## Shared options

These are keywords of [`Render`](@ref) (and so of `render(ifs, method; ...)`):

- `resolution = (rows, cols)`.
- `color = true` tints the output by map.
- `alpha = true` makes zero-valued pixels transparent.
- `overlays = [BasePolygon(:line), Axes()]` draws on top of the finished image:
  [`BasePolygon`](@ref) the untransformed polygon preset, [`Axes`](@ref) the coordinate axes.

Where the file goes is `outpath`, a keyword of `render` itself; see [Output files](@ref).

## Selecting a definition from a `.ifs` file

Build the [`IFS`](@ref) first, by name, with [`get_ifs`](@ref) (which prefers a prebuilt
record and falls back to parsing the file):

```julia
ifs = get_ifs("Heighway Dragon"; file = "Default.ifs")
render(ifs, Chaos(npoints = 100_000))
```

A name that doesn't exist throws `ArgumentError`. [`parse_ifs_file`](@ref) returns every
definition in a file as a `Vector{IFS}`.

## [`Chaos`](@ref)

`Chaos(; npoints, warmup, seed)`: chaos-game sampling with [`chaos_game`](@ref), rasterized by
[`make_image`](@ref). A `seed` makes the result reproducible for a fixed thread count.

```julia
out = render(IFS(EISENSTEIN), Chaos(npoints = 250_000, seed = 1); resolution = (1024, 1024))
```

## [`PointDeterministic`](@ref)

`PointDeterministic(; npoints, iterations, warmup, seed)`: applies every map to every point
`iterations` times ([`deterministic_iterate`](@ref)), starting from `npoints` chaos-game points.
The point count grows as `npoints * nmaps^iterations`, so keep `npoints` small.

```julia
out = render(IFS(EISENSTEIN), PointDeterministic(npoints = 500, iterations = 2);
             resolution = (800, 800))
```

## [`Inverse`](@ref)

`Inverse(; iterations, divergence_scale, preview)`: scores each pixel by pushing it back through the
inverse maps ([`rasterize_image_inversely`](@ref)). `divergence_scale = false` gives a binary
mask where non-diverged and origin-containing regions are white. `preview = true` caps the tracked
branch frontier for a faster, approximate image.

```julia
out = render(IFS(EISENSTEIN), Inverse(iterations = 6); resolution = (800, 800))
```

## [`Transformations`](@ref)

`Transformations(; polygon, limits_iterations)`: draws each map applied to the
[`initial_polygon`](@ref) preset (`:default`, `:equilateral_triangle`, `:line_arrow`, `:line`),
without chaos or image iteration, framed by the polygon after `limits_iterations` iterations.
`color = true` tints by map. For SVG output, see [`render_transformations_svg`](@ref).

Rotation follows the math convention: positive angles are counterclockwise in model
coordinates (`+x` right, `+y` up). Transformation renders share an anchored origin, so
`(0, 0)` maps to the same pixel for every polygon preset.

```julia
out = render(IFS(HEIGHWAY_DRAGON), Transformations(polygon = :line_arrow);
             color = true, overlays = [Axes()], resolution = (1024, 1024))
```

## [`ImageIterate`](@ref)

`ImageIterate(source = PolygonSource(); iterations = 1)`: starts from a seed image and applies
the IFS to it `iterations` times with [`iterate_image`](@ref). The source is one of:

- [`PolygonSource`](@ref)`(; polygon, frame, limits_iterations)`: the polygon preset drawn in
  pixel space. `frame = :geometry` (the default) frames by the polygon's iterated geometry,
  `limits_iterations` deep (default: the `ImageIterate` iteration count); `frame = :ifs` uses
  `ifs.limits`.
- [`FileSource`](@ref)`(path)`: the grayscale image at `path`. When it has a render record
  ([`render_record_path`](@ref)`(path)`: any PNG `render` wrote at its default location, or an
  animation frame), its saved limits frame the image; otherwise the IFS's limits are used.
- any other [`RenderMethod`](@ref): its image (and frame) is the seed, so
  `ImageIterate(Chaos(npoints = 100_000))` iterates a chaos-game heatmap.

`color = true` colors the final pass.

```julia
out = render(IFS(EISENSTEIN),
             ImageIterate(PolygonSource(polygon = :equilateral_triangle, frame = :ifs);
                          iterations = 3);
             resolution = (800, 800))

out = render(IFS(EISENSTEIN), ImageIterate(Chaos(npoints = 50_000); iterations = 2))
```

## Saving, loading, reuse and file names

[`save_render`](@ref) (called by `render`) writes the PNG (placed by `outpath`) and a `.jld2` record
holding the [`Render`](@ref) and its `limits` at [`render_record_path`](@ref)`(r)` under
`data/sims/renders/`, tagged with the Git commit by DrWatson's `tagsave`.
[`load_render`](@ref) reads a record back (pass the `.jld2`, or a PNG at its default location), so
a render can be repeated exactly or changed:

```julia
r = load_render(out.record)
render(r; outpath = "again.png")
render(Render(r; resolution = (4096, 4096)))   # copy with a field changed
```

`render` reuses work like DrWatson's `produce_or_load`: when the PNG and a record for the same
render already exist they are loaded (`cached = true` in the result) instead of redrawn;
`render(r; force = true)` redraws and overwrites. The point clouds of `Chaos` and
`PointDeterministic` are cached on their own ([`cached_points`](@ref)), independent of resolution,
color and overlays, so a larger or recolored re-render only rasterizes again. See
[Output files](@ref) and [Data layout](@ref).

With `outpath = nothing` the PNG is named with DrWatson's `savename`: the IFS name, the method
tag and parameters, the resolution, and `_color`/`_alpha` flags, for example
`Heighway_Dragon_chaos_npoints=200000_1024x1024.png`.

## Lower-level building blocks

The pieces can be used directly, without a [`Render`](@ref):

```julia
using Fractals, DrWatson

ifs = IFS(HEIGHWAY_DRAGON)
points = chaos_game(ifs, 200_000; seed = 1234)   # reproducible chaos game
img = make_image(points, ifs.limits; resolution = (1024, 1024))
safesave(output_dir("dragon.png"), img)     # DrWatson: creates dirs, keeps an existing file as _#1

eis = IFS(EISENSTEIN)
expanded = deterministic_iterate(eis, chaos_game(eis, 200), 2)
img2 = make_image(expanded, eis.limits; resolution = (800, 800))
```

Saving through `safesave` directly skips the render record that [`render`](@ref) writes. See
[Output files](@ref).

The SVG diagram of the maps is produced separately by [`render_transformations_svg`](@ref).
