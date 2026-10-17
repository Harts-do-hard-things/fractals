# API reference

```@meta
CurrentModule = Fractals
```

## Rendering

```@docs
render
Render
save_render
load_render
render_record_path
cached_points
ifs_key
RenderMethod
Chaos
PointDeterministic
Inverse
Transformations
ImageIterate
ImageSource
PolygonSource
FileSource
Overlay
BasePolygon
Axes
OutputFormat
PNG
SVG
render_transformations_svg
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
```

## Output

```@docs
output_dir
data_output_dir
ifs_data_dir
with_output_root
read_provenance
```

## Point clouds and the catalog

```@docs
cached_points
ifs_key
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
