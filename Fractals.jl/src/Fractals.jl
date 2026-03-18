module Fractals

include("matrixfractal.jl")
include("ifsparser.jl")

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
       iterate!,
       iterate_parallel!,
       deterministic_iterate,
       interpolate_eq_matrix,
       interpolate_ifs,
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
