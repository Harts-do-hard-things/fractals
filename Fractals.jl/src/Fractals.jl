module Fractals

using StaticArrays
using LinearAlgebra
using StatsBase
import PNGFiles
import FileIO
using Colors
using Random
using Base.Threads
using Printf
using JSON3
using Dates: now
using Base.ScopedValues: ScopedValue, with
import DrWatson
import JLD2
using DrWatson: plotsdir, datadir, projectdir, savename, safesave, gitdescribe

include("config.jl")                  # constants, polygon presets
include("utils.jl")                   # shared utilities: PNG metadata, color maps, GPU stub
include("output.jl")                  # DrWatson-managed output paths, naming, safesave, provenance
# IFSParse is a standalone package (no dependency on Fractals); types.jl and ifs_load.jl import from it
using IFSParse
using IFSParse:
    _validated_ifs_definitions,
    _split_valid_ifs_definitions,
    _contractive_row,
    _noncontractive_rows,
    _format_contractivity_failure
include("types.jl")                   # AffineMap, IFS structs and constructors
include("render_types.jl")            # RenderMethod types, ImageSource, Overlay, Render
include("drawing/projection.jl")      # world <-> pixel/SVG coordinate mapping
include("drawing/polygon.jl")         # preset polygon: framing, seed image, transformed segments
include("drawing/post_processing.jl") # line/axes drawing and the overlays used by finish
include("drawing/svg.jl")             # render_transformations_svg
include("drawing/transformations.jl") # raster drawing behind draw(::Transformations)
include("chaos.jl")                   # chaos_game
include("point_deterministic.jl")     # deterministic_iterate
include("rasterize.jl")               # make_image, make_pixelate_map
include("inverse.jl")                 # rasterize_image_inversely
include("image_iterate.jl")           # iterate_image
include("interpolation.jl")           # interpolate_eq_matrix, interpolate_ifs, render_interpolation_frames
include("ifs_load.jl")                # parse_ifs_string/file, get_ifs: IFS-returning wrappers over IFSParse
# render_pipeline.jl follows every file whose kernels draw() calls
include("render_pipeline.jl")         # render(::Render): draw, finish, save_render, load_render
include("examples.jl")                # HEIGHWAY_DRAGON, EISENSTEIN, main()

export AffineMap,
    IFS,
    IFSToken,
    IFSDefinition,
    InitialPolygonPreset,
    RenderMethod,
    Chaos,
    PointDeterministic,
    ImageIterate,
    Transformations,
    ImageSource,
    PolygonSource,
    FileSource,
    Overlay,
    BasePolygon,
    Axes,
    OutputFormat,
    PNG,
    SVG,
    Render,
    save_render,
    load_render,
    render_record_path,
    cached_points,
    ifs_key,
    Inverse,
    lex_ifs,
    parse_ifs_string,
    parse_ifs_file,
    build_ifs_library,
    list_ifs_files,
    list_ifs,
    get_ifs,
    get_ifs_definition,
    random_eq,
    random_ifs,
    render,
    chaos_game,
    deterministic_iterate,
    interpolate_eq_matrix,
    interpolate_ifs,
    render_interpolation_frames,
    export_animation,
    refresh_limits,
    make_pixelate_map,
    make_image,
    iterate_image,
    render_transformations_svg,
    inverse_iterate,
    rasterize_image_inversely,
    initial_polygon,
    supported_initial_polygons,
    output_dir,
    data_output_dir,
    ifs_data_dir,
    with_output_root,
    read_provenance,
    extend_fractal,
    fractal_catalog,
    RESOLUTION,
    DEFAULT_WARMUP,
    DEFAULT_SAMPLES,
    HEIGHWAY_DRAGON,
    EISENSTEIN
end
