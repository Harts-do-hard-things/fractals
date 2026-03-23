module Fractals

using StaticArrays
using LinearAlgebra
using StatsBase
using Images
using FileIO
using Colors
using Random
using Base.Threads
using Printf

include("config.jl")                 # constants, RenderMethod enum, polygon presets
include("utils.jl")                  # shared utilities: path helpers, color maps, GPU stub
include("types.jl")                  # AffineMap, IFS structs and constructors
include("render_transformations.jl") # SVG/PNG visualization of affine maps
include("chaos.jl")                  # iterate!, iterate_parallel!
include("point_deterministic.jl")    # deterministic_iterate
include("rasterize.jl")              # make_image, make_pixelate_map
include("inverse.jl")                # rasterize_image_inversely
# image_iterate.jl must follow chaos, point_deterministic, rasterize, inverse, and
# render_transformations — _resolve_image_source calls all of them
include("image_iterate.jl")          # iterate_image, _resolve_image_source
include("interpolation.jl")          # interpolate_eq_matrix, interpolate_ifs, render_interpolation_frames
include("ifsparser.jl")              # IFS file/string parser (pure; no I/O)
# interactive.jl must follow ifsparser.jl (calls parse_ifs_file) and all render-method files,
# but must come BEFORE render.jl — render.jl's _resolve_render_input calls _select_ifs_definition
include("interactive.jl")            # all stdin logic: prompt helpers, _select_ifs_definition, prompt_ifs_and_render
include("render.jl")                 # render (top-level API, must come after all methods)
include("examples.jl")               # HEIGHWAY_DRAGON, EISENSTEIN, main()

export AffineMap,
       IFS,
       IFSToken,
       IFSDefinition,
       RenderMethod,
       Chaos,
       Parallel,
       PointDeterministic,
       ImageIterate,
       Inverse,
       RenderTransformations,
       lex_ifs,
       parse_ifs_string,
       parse_ifs_file,
       prompt_ifs_and_render,
       render,
       render_chaos,
       render_point_deterministic,
       render_inverse,
       render_image_iterate,
       iterate!,
       iterate_parallel!,
       deterministic_iterate,
       interpolate_eq_matrix,
       interpolate_ifs,
       render_interpolation_frames,
       make_pixelate_map,
       make_image,
       iterate_image,
       render_transformations_svg,
       render_transformations_png,
       inverse_iterate,
       rasterize_image_inversely,
       RESOLUTION,
       DEFAULT_WARMUP,
       DEFAULT_SAMPLES,
       HEIGHWAY_DRAGON,
       EISENSTEIN
end
