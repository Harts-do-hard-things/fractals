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
chaos_game
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
parse_ifs_definitions_string
parse_ifs_definitions_file
build_ifs_library
list_ifs_files
list_ifs
get_ifs
get_ifs_definition
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

## Saved data (JLD2)

```@docs
save_fractal
load_fractal
fractal_record
fractal_data
render_fractal
extend_fractal
fractal_catalog
```

## Constants

```@docs
RESOLUTION
DEFAULT_WARMUP
DEFAULT_SAMPLES
HEIGHWAY_DRAGON
EISENSTEIN
```
