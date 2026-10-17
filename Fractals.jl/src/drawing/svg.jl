function _render_transformations_svg_impl(
    ifs;
    outpath::Union{Nothing,AbstractString} = nothing,
    width::Int = 1200,
    height::Int = 1200,
    show_base::Bool = false,
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol} = initial_polygon(),
    polygon_limits_iterations::Integer = 1,
    stroke_width::Real = 2.0,
    axis::Bool = false,
)
    svg_ifs = _build_polygon_raster_ifs(
        ifs;
        initial_polygon_spec = initial_polygon_spec,
        polygon_limits_iterations = polygon_limits_iterations,
    )
    _, base_segments, transformed_by_map, _ = _collect_transformed_base_segments(
        ifs;
        show_base = show_base,
        initial_polygon_spec = initial_polygon_spec,
    )
    colors = _map_colors(length(transformed_by_map))

    sx, sy, ox, oy = _projection_from_limits(svg_ifs.limits, width, height)
    lines = String[]

    if show_base
        for (p1, p2) in base_segments
            x1, y1 = _to_svg_xy(p1, sx, sy, ox, oy)
            x2, y2 = _to_svg_xy(p2, sx, sy, ox, oy)
            push!(
                lines,
                @sprintf(
                    """  <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="#222222" stroke-width="%.2f" opacity="0.65" />""",
                    x1,
                    y1,
                    x2,
                    y2,
                    stroke_width
                )
            )
        end
    end

    for (i, transformed) in enumerate(transformed_by_map)
        color = _rgb_to_hex(_map_color_rgb(i, colors))
        for (p1, p2) in transformed
            x1, y1 = _to_svg_xy(p1, sx, sy, ox, oy)
            x2, y2 = _to_svg_xy(p2, sx, sy, ox, oy)
            push!(
                lines,
                @sprintf(
                    """  <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" stroke-width="%.2f" />""",
                    x1,
                    y1,
                    x2,
                    y2,
                    color,
                    stroke_width
                )
            )
        end
    end

    if axis
        axis_color = "#6B7280"
        tick_half = 3.0
        xmin, xmax, ymin, ymax = _visible_world_bounds(sx, sy, ox, oy, width, height)

        if ymin <= 0.0 <= ymax
            x1, y1 = _to_svg_xy(SVector{2,Float64}(xmin, 0.0), sx, sy, ox, oy)
            x2, y2 = _to_svg_xy(SVector{2,Float64}(xmax, 0.0), sx, sy, ox, oy)
            push!(
                lines,
                @sprintf(
                    """  <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" stroke-width="1.00" opacity="0.8" data-role="axis-x" />""",
                    x1,
                    y1,
                    x2,
                    y2,
                    axis_color
                )
            )
            for xtick = ceil(Int, xmin):floor(Int, xmax)
                x, y = _to_svg_xy(SVector{2,Float64}(Float64(xtick), 0.0), sx, sy, ox, oy)
                push!(
                    lines,
                    @sprintf(
                        """  <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" stroke-width="1.00" opacity="0.8" data-role="axis-x-tick" />""",
                        x,
                        y - tick_half,
                        x,
                        y + tick_half,
                        axis_color
                    )
                )
            end
        end

        if xmin <= 0.0 <= xmax
            x1, y1 = _to_svg_xy(SVector{2,Float64}(0.0, ymin), sx, sy, ox, oy)
            x2, y2 = _to_svg_xy(SVector{2,Float64}(0.0, ymax), sx, sy, ox, oy)
            push!(
                lines,
                @sprintf(
                    """  <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" stroke-width="1.00" opacity="0.8" data-role="axis-y" />""",
                    x1,
                    y1,
                    x2,
                    y2,
                    axis_color
                )
            )
            for ytick = ceil(Int, ymin):floor(Int, ymax)
                x, y = _to_svg_xy(SVector{2,Float64}(0.0, Float64(ytick)), sx, sy, ox, oy)
                push!(
                    lines,
                    @sprintf(
                        """  <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" stroke-width="1.00" opacity="0.8" data-role="axis-y-tick" />""",
                        x - tick_half,
                        y,
                        x + tick_half,
                        y,
                        axis_color
                    )
                )
            end
        end
    end

    body = join(lines, "\n")
    svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $width $height" style="background: transparent;" fill="none">
$body
</svg>
"""

    params = (;
        method = :transformations_svg,
        ifs_name = ifs.name,
        initial_polygon = _resolve_initial_polygon(initial_polygon_spec).name,
        polygon_limits_iterations,
        show_base,
        axis,
        resolution = (height, width),
    )
    final_outpath = _resolve_output_path(
        outpath,
        _output_name(ifs.name, params[(:method, :initial_polygon, :resolution)], "svg"),
    )
    return _save_svg(final_outpath, svg; params = params)
end

"""
    render_transformations_svg(ifs; outpath=nothing, width=1200, height=1200, show_base=false,
                               initial_polygon=initial_polygon(), polygon_limits_iterations=1,
                               stroke_width=2.0, axis=false) -> String

Write an SVG of each affine map applied to the seed polygon (one stroke color per map) and
return its path. Framing, `show_base`, and `axis` work as in
[`render_transformations`](@ref). `outpath` is resolved like `render`'s; the file starts
with a `<!-- fractals.provenance: ... -->` comment.
"""
function render_transformations_svg(
    ifs;
    outpath::Union{Nothing,AbstractString} = nothing,
    width::Int = 1200,
    height::Int = 1200,
    show_base::Bool = false,
    initial_polygon::Union{InitialPolygonPreset,Symbol} = initial_polygon(),
    polygon_limits_iterations::Integer = 1,
    stroke_width::Real = 2.0,
    axis::Bool = false,
)
    return _render_transformations_svg_impl(
        ifs;
        outpath = outpath,
        width = width,
        height = height,
        show_base = show_base,
        initial_polygon_spec = initial_polygon,
        polygon_limits_iterations = polygon_limits_iterations,
        stroke_width = stroke_width,
        axis = axis,
    )
end
