# Quick recipes

## Prerequisites

```powershell
julia --project=Fractals.jl -e "using Pkg; Pkg.instantiate()"
```

## Input Selection

Pick a definition from a `.ifs` file by name with [`get_ifs`](@ref), then render it:

```julia
using Fractals

ifs = get_ifs("Heighway Dragon"; file = "my_fractals.ifs", dir = ".")
result = render(ifs, Chaos(npoints = 150_000); outpath = "recipe_file_name.png")
```

[`parse_ifs_file`](@ref) returns every definition in a file, so index selection is ordinary
indexing:

```julia
ifs = parse_ifs_file("my_fractals.ifs")[2]
result = render(ifs, Chaos(npoints = 150_000); outpath = "recipe_file_index.png")
```

An equation matrix works too: `IFS(eq)`.

## Render Methods

The methods are [`Chaos`](@ref), [`PointDeterministic`](@ref), [`Inverse`](@ref),
[`Transformations`](@ref) and [`ImageIterate`](@ref); each takes only its own parameters, and
`render(ifs, method; resolution, color, alpha, overlays, outpath)` runs it. See
[Render methods](@ref).

## Recipe: Chaos

```julia
using Fractals

eq = [
    0.5  -0.5   0.5   0.5   0.0   0.0;
   -0.5  -0.5   0.5  -0.5   1.0   0.0
]

result = render(IFS(eq), Chaos(npoints = 200_000, seed = 1);
                resolution = (1024, 1024),
                outpath = "recipe_chaos.png")

println(result.outpath)
```

## Recipe: PointDeterministic

```julia
using Fractals

result = render(IFS(EISENSTEIN), PointDeterministic(npoints = 300, iterations = 2);
                resolution = (900, 900),
                outpath = "recipe_point_deterministic.png")
```

## Recipe: ImageIterate

```julia
using Fractals

result = render(IFS(EISENSTEIN),
                ImageIterate(PolygonSource(polygon = :equilateral_triangle, frame = :ifs);
                             iterations = 3);
                resolution = (900, 900),
                outpath = "recipe_image_iterate.png")
```

The seed can also be an image file (`FileSource("seed.png")`, grayscale) or another render
method (`ImageIterate(Chaos(npoints = 50_000))`). The polygon seed is the same image
[`Transformations`](@ref) draws, read as grayscale.

## Recipe: Inverse

```julia
using Fractals

result = render(IFS(EISENSTEIN), Inverse(iterations = 6);
                resolution = (900, 900),
                outpath = "recipe_inverse.png")
```

`Inverse(iterations = 6, divergence_scale = false)` gives a binary mask, and `Inverse(preview = true)` a faster approximate image.

## Recipe: Transformations with overlays

```julia
using Fractals

result = render(IFS(HEIGHWAY_DRAGON), Transformations(polygon = :line_arrow);
                color = true,
                overlays = [BasePolygon(:line_arrow), Axes()],
                resolution = (900, 900),
                outpath = "recipe_transformations.png")
```

## Recipe: Reload and repeat a render

```julia
using Fractals

result = render(IFS(EISENSTEIN), Chaos(npoints = 100_000, seed = 7); outpath = "recipe_repeat.png")
again = render(load_render("recipe_repeat.png"); outpath = "recipe_repeat_2.png")
again.image == result.image   # true: same description, same seed
```

## Recipe: Reuse, force, and re-render larger

```julia
using Fractals

ifs = IFS(EISENSTEIN; name = "Eisenstein")
first_run = render(ifs, Chaos(npoints = 1_000_000, seed = 1))                  # draws
second_run = render(ifs, Chaos(npoints = 1_000_000, seed = 1))                 # second_run.cached == true
render(ifs, Chaos(npoints = 1_000_000, seed = 1); force = true)                # redraw and overwrite

# Another resolution or style reuses the cached points; only rasterizing is redone.
render(Render(first_run.render; resolution = (4096, 4096), color = true))

# Grow the cached chaos game instead of starting over, then render the larger count.
extend_fractal(Chaos(npoints = 1_000_000, seed = 1), ifs, 500_000; seed = 2)
render(ifs, Chaos(npoints = 1_500_000, seed = 1))
```

## Recipe: Interpolate Between Two IFS States

```julia
using Fractals

start_eq = [
    0.5  -0.5   0.5   0.5   0.0   0.0
   -0.5  -0.5   0.5  -0.5   1.0   0.0
]

finish_eq = [
    0.5   0.0   0.0   0.5   0.0   0.0   0.3
    0.0   0.5  -0.5   0.0   1.0   0.0   0.7
]

# For animation work, keep the same transform count/order on both sides.
mid = interpolate_ifs(IFS(start_eq; name="start"),
                      IFS(finish_eq; name="finish"),
                      0.5;
                      limits_mode=:interpolate)

render(mid, Transformations(); resolution=(900, 900), outpath="recipe_interpolated_transforms.png")
```

Use `interpolate_eq_matrix(...)` when you want the blended 7-column equation matrix directly.
By default, interpolation uses a rotation-plus-uniform-scale path for each map's 2x2 linear part, with automatic fallback to plain linear blending when needed. Pass `interpolation_mode=:linear` to force the previous coefficient-wise behavior.

## Recipe: Render Deterministic Animation Frames

```julia
using Fractals

start = IFS(start_eq; name="start")
finish = IFS(finish_eq; name="finish")

frames = render_interpolation_frames(start, finish, Chaos(npoints = 100_000, warmup = 20, seed = 1);
                                     frames=4,
                                     outdir="frames",
                                     basename="recipe_anim",
                                     interpolation_mode=:rotation_scale,
                                     resolution=(512, 512))

println.(frames.paths)
```

Interpolated frame rendering accepts any render method (`Transformations()` is the default),
with deterministic file names like `recipe_anim_0001.png`; each frame also gets a render record.

Convert those frames into an animation artifact with `ffmpeg`:

```julia
gif = export_animation(:gif;
                       frames_dir="frames",
                       basename="recipe_anim",
                       outpath="recipe_anim.gif",
                       fps=12)

mp4 = export_animation(:mp4;
                       frames_dir="frames",
                       basename="recipe_anim",
                       outpath="recipe_anim.mp4",
                       fps=12)
```

An equivalent helper script is available for shell workflows (`--format gif|mp4`, default `gif`):

```bash
julia --startup-file=no --project=. scripts/export_animation.jl \
  --format gif \
  --frames-dir frames \
  --basename recipe_anim \
  --out recipe_anim.gif \
  --fps 12

julia --startup-file=no --project=. scripts/export_animation.jl \
  --format mp4 \
  --frames-dir frames \
  --basename recipe_anim \
  --out recipe_anim.mp4 \
  --fps 12
```

Install `ffmpeg` separately before using the export step:

```bash
sudo apt-get install ffmpeg
brew install ffmpeg
winget install Gyan.FFmpeg
```
