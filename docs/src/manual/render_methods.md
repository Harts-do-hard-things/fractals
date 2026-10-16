# Render methods

[`render`](@ref) is the high-level entry point: it resolves the input, runs one render
method, saves the image, and returns `(ifs, image, outpath, method)`.

```julia
using Fractals

result = render(HEIGHWAY_DRAGON;
                method = Chaos,
                backend = :cpu,
                npoints = 200_000,
                resolution = (1024, 1024),
                outpath = "render.png")

result.outpath   # <repo>/plots/render.png
```

## Choosing a method

`method` accepts any of these forms:

- enum values (preferred): `Chaos`, `Parallel`, `PointDeterministic`, `ImageIterate`,
  `Inverse`, `RenderTransformations`
- symbols: `:chaos`, `:parallel`, `:point_deterministic`, `:image_iterate`, `:inverse`,
  `:render_transformations`
- the same names as strings: `"chaos"`, `"point_deterministic"`, ...

`Parallel` renders as `Chaos`, because [`iterate!`](@ref) is already threaded. The removed
`:deterministic` method throws `ArgumentError` with a pointer to `:point_deterministic`.

## Shared options

- `iterations` sets the depth for `PointDeterministic`, `Inverse`, and the
  `:point_deterministic`/`:inverse` image sources. It must be `>= 0`.
- `warmup` applies to the chaos, point-deterministic, and image-source generation paths.
- `backend = :cpu | :gpu | :auto` controls the GPU-capable steps (`make_image`,
  `iterate_image`, and inverse rasterization):
  - `:cpu` always uses the CPU implementation;
  - `:gpu` requires the CUDA extension and errors if it's unavailable;
  - `:auto` falls back to the CPU when no GPU is available.
- `npoints`: for matrix, string, and file inputs, omitting it uses `DEFAULT_SAMPLES`. For an
  `IFS` input, omitting it keeps the existing point count.
- `color = true` tints the output by map (the default for `RenderTransformations`), and
  `alpha = true` makes zero-valued pixels transparent.

## Selecting a definition from a `.ifs` file

- Pass `ifs_index = ...` or `ifs_name = ...`, not both.
- An index or name that doesn't exist throws `ArgumentError` listing the available
  definitions.
- With no selector and several definitions, `render` lists them and asks (through
  `input_fn`) to confirm rendering definition `1`. Declining throws `ArgumentError`.

## `Chaos`

Chaos-game sampling with [`iterate!`](@ref), rasterized by [`make_image`](@ref). See
[`render_chaos`](@ref).

```julia
out = render(EISENSTEIN; method = Chaos, npoints = 250_000, resolution = (1024, 1024))
```

## `PointDeterministic`

Applies every map to every point `iterations` times ([`deterministic_iterate`](@ref)). The
point count grows as `npoints * nmaps^iterations`, so keep `npoints` small. See
[`render_point_deterministic`](@ref).

```julia
out = render(EISENSTEIN; method = PointDeterministic, iterations = 2, npoints = 500,
             resolution = (800, 800))
```

## `ImageIterate`

Builds a seed image, then applies the IFS to it `image_iterations` times with
[`iterate_image`](@ref). See [`render_image_iterate`](@ref).

- `image_source = :polygon | :chaos | :point_deterministic | :inverse | :file`:
  - `:polygon` renders the [`initial_polygon`](@ref) preset (`:default`,
    `:equilateral_triangle`, `:line_arrow`, `:line`).
  - `:file` loads `image_path` and reuses the framing embedded in PNGs written by this
    package.
- `polygon_limits_mode = :geometry | :ifs` (polygon source only) frames by the polygon's
  iterated geometry (the default) or by `ifs.limits`. `:default` is accepted and treated as
  `:geometry` with a warning.
- `show_base` and `axis` overlay the seed polygon and coordinate axes (polygon source only).
- `color = true` colors the final pass.

```julia
out = render(EISENSTEIN; method = ImageIterate, image_source = :polygon, image_iterations = 3,
             polygon_limits_mode = :ifs, initial_polygon = :equilateral_triangle,
             resolution = (800, 800))
```

## `Inverse`

Scores each pixel by pushing it back through the inverse maps
([`rasterize_image_inversely`](@ref)). `show_divergence_scale = false` switches to a binary
mask where non-diverged and origin-containing regions are white. See
[`render_inverse`](@ref).

```julia
out = render(EISENSTEIN; method = Inverse, iterations = 6, resolution = (800, 800))
```

## `RenderTransformations`

Draws each map applied to the seed polygon directly, without chaos or image iteration.
Options: `initial_polygon`, `show_base`, `axis`, `color`, `polygon_limits_iterations`. See
[`render_transformations`](@ref) and, for SVG output,
[`render_transformations_svg`](@ref).

Rotation follows the math convention: positive angles are counterclockwise in model
coordinates (`+x` right, `+y` up). Transformation renders share an anchored origin, so
`(0, 0)` maps to the same pixel for every `initial_polygon` preset.

```julia
out = render(HEIGHWAY_DRAGON; method = RenderTransformations, initial_polygon = :line_arrow,
             axis = true, resolution = (1024, 1024))
```

## Lower-level building blocks

The render methods are thin wrappers, and the pieces can be used directly:

```julia
using Fractals, DrWatson

ifs = IFS(HEIGHWAY_DRAGON; npoints = 200_000)
iterate!(ifs; seed = 1234)                  # reproducible chaos game
img = make_image(ifs; resolution = (1024, 1024))
safesave(output_dir("dragon.png"), img)     # DrWatson: creates dirs, keeps an existing file as _#1

expanded = deterministic_iterate(IFS(EISENSTEIN; npoints = 200), 2)
img2 = make_image(expanded; resolution = (800, 800))
```

Saving through `safesave` directly skips the source-limits and provenance metadata that
[`render`](@ref) embeds. See [Output files](@ref).
