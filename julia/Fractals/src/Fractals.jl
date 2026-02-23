module Fractals

include("matrixfractal.jl")
include("ifsparser.jl")

export AffineMap,
       IFS,
       IFSToken,
       IFSDefinition,
       lex_ifs,
       parse_ifs_tokens,
       parse_ifs_string,
       parse_ifs_file,
       prompt_ifs_and_render,
       build_maps_and_weights,
       get_limits,
       iterate!,
       iterate_parallel!,
       deterministic_iterate,
       make_pixelate_map,
       make_pixeliterate_map,
       make_image,
       iterate_image,
       iterate_image_single_map,
       base_limits_image,
       image_to_svg,
       base_l_image,
       base_l_image_svg,
       render_transformations_svg,
       render_transformations_png_from_base_l_svg,
       inverse_iterate,
       rasterize_image_inversely,
       RESOLUTION,
       DEFAULT_WARMUP,
       DEFAULT_SAMPLES,
       HEIGHWAY_DRAGON,
       EISENSTEIN

end
