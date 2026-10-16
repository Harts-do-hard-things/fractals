# API reference

```@meta
CurrentModule = Fractals
```

## Rendering

```@docs
render
RenderMethod
Chaos
Parallel
PointDeterministic
ImageIterate
Inverse
RenderTransformations
render_chaos
render_point_deterministic
render_image_iterate
render_inverse
render_transformations
render_transformations_svg
render_transformations_png
```

## Core types

```@docs
AffineMap
IFS
refresh_limits
random_eq
random_ifs
```

## Iteration

```@docs
iterate!
iterate_parallel!
deterministic_iterate
```

## Rasterization

```@docs
make_image
make_pixelate_map
iterate_image
rasterize_image_inversely
inverse_iterate
```

## Seed polygons

```@docs
InitialPolygonPreset
initial_polygon
supported_initial_polygons
```

## Interpolation and animation

```@docs
interpolate_eq_matrix
interpolate_ifs
render_interpolation_frames
export_animation
```

## Parsing

```@docs
parse_ifs_string
parse_ifs_file
lex_ifs
IFSDefinition
IFSToken
prompt_ifs_and_render
```

## Output

```@docs
output_dir
data_output_dir
ifs_data_dir
with_output_root
read_provenance
```

## Constants

```@docs
RESOLUTION
DEFAULT_WARMUP
DEFAULT_SAMPLES
HEIGHWAY_DRAGON
EISENSTEIN
```
