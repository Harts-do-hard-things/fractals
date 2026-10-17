# Animation

Animations are built in three steps: interpolate between two IFS states, render a numbered
frame sequence, and encode the frames with `ffmpeg`.

## Interpolating between systems

```julia
using Fractals

start_eq = [
    0.5 -0.5 0.5 0.5 0.0 0.0
    -0.5 -0.5 0.5 -0.5 1.0 0.0
]
finish_eq = [
    0.5 0.0 0.0 0.5 0.0 0.0 0.3
    0.0 0.5 -0.5 0.0 1.0 0.0 0.7
]

mid_eq = interpolate_eq_matrix(start_eq, finish_eq, 0.5)
mid_ifs = interpolate_ifs(IFS(start_eq), IFS(finish_eq), 0.5)
```

- [`interpolate_eq_matrix`](@ref) accepts compatible 6- or 7-column matrices and returns a
  normalized 7-column matrix. 6-column inputs derive weights from their determinants first.
- `interpolation_mode = :rotation_scale` (the default) interpolates each map's 2×2 part as a
  rotation plus a uniform scale, falling back to coefficient-wise blending for map pairs
  that don't fit that model. `interpolation_mode = :linear` always blends coefficients.
- [`interpolate_ifs`](@ref) also blends `ifs.limits` by default (`limits_mode =
  :interpolate`), which avoids frame-to-frame jitter. `limits_mode = :left | :right |
  :recompute` gives a fixed or freshly sampled frame instead.
- Both systems need the same number of maps, in the same order.

## Rendering frames

[`render_interpolation_frames`](@ref) writes `basename_0001.png`, `basename_0002.png`, ...,
rendering each interpolated system with any [`RenderMethod`](@ref) (`Transformations()` by
default). `resolution`, `color`, `alpha` and `overlays` are the same as for [`Render`](@ref):

```julia
start = IFS(start_eq; name = "start")
finish = IFS(finish_eq; name = "finish")

frames = render_interpolation_frames(start, finish, Chaos(npoints = 100_000, warmup = 20, seed = 1);
                                     frames = 5,
                                     outdir = "frames",
                                     basename = "dragon_morph",
                                     resolution = (512, 512))
```

The result is `(outdir, paths, ts)`. Omit `outdir` to write to an auto-named directory under
`plots/frames/`. Each frame also gets a render record (see [`load_render`](@ref)) holding its
interpolated system, mirrored under `data/sims/renders/frames/<sequence>/`; frames are reused
like any other [`render`](@ref) when nothing changed.

## Exporting GIF and MP4

[`export_animation`](@ref) encodes a frame sequence with `ffmpeg`:

```julia
gif = export_animation(:gif; frames_dir = "frames", basename = "dragon_morph", outpath = "dragon_morph.gif", fps = 12)
mp4 = export_animation(:mp4; frames_dir = "frames", basename = "dragon_morph", outpath = "dragon_morph.mp4", fps = 12)
```

The sequence must start at `basename_0001.png`. Omitting `outpath` writes
`plots/animations/<basename>.<format>`, and an existing animation is overwritten.
The same export is available from the shell:

```bash
julia --startup-file=no --project=. scripts/export_animation.jl --format gif --frames-dir frames --basename dragon_morph --out dragon_morph.gif --fps 12
julia --startup-file=no --project=. scripts/export_animation.jl --format mp4 --frames-dir frames --basename dragon_morph --out dragon_morph.mp4 --fps 12
```

`ffmpeg` must be installed separately and on `PATH`:

```bash
sudo apt-get install ffmpeg   # Debian/Ubuntu
brew install ffmpeg           # macOS
winget install Gyan.FFmpeg    # Windows
```

## ImageIterate build animations

`scripts/animate_image_iterate.jl` animates an `ImageIterate` build-up from a GrayA
seed image, adding one map at a time with interpolated frames, and exports a GIF:

```bash
julia --startup-file=no --project=. scripts/animate_image_iterate.jl --input ifs/Default.ifs --image-path seed.png --frames-per-map 8 --fps 12
```

Frames default to `plots/frames/<basename>/` and the GIF to
`plots/animations/<basename>.gif`. Run the script with `--help` for all options.
