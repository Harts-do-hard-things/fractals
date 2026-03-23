function _anchored_projection_from_bbox(
    xmin::Float64,
    xmax::Float64,
    ymin::Float64,
    ymax::Float64,
    width::Int,
    height::Int;
    margin=0.06,
)
    # Include the world origin so (0, 0) lands on the same pixel anchor
    # for every initial polygon preset.
    xmin = min(xmin, 0.0)
    xmax = max(xmax, 0.0)
    ymin = min(ymin, 0.0)
    ymax = max(ymax, 0.0)

    ox = 1.0 + margin * (width - 1)
    oy = height - margin * (height - 1)

    scales = Float64[]
    if xmax > 0
        push!(scales, (width - ox) / xmax)
    end
    if xmin < 0
        push!(scales, (ox - 1.0) / (-xmin))
    end
    if ymax > 0
        push!(scales, (oy - 1.0) / ymax)
    end
    if ymin < 0
        push!(scales, (height - oy) / (-ymin))
    end

    s = isempty(scales) ? 1.0 : minimum(scales)
    s = isfinite(s) && s > 0 ? s : 1.0
    return s, s, ox, oy
end

function _fit_bounds(segments, width::Int, height::Int; margin=0.06)
    xmin = Inf; xmax = -Inf; ymin = Inf; ymax = -Inf
    for (p1, p2) in segments
        x1, y1 = p1; x2, y2 = p2
        xmin = min(xmin, x1, x2); xmax = max(xmax, x1, x2)
        ymin = min(ymin, y1, y2); ymax = max(ymax, y1, y2)
    end
    if !isfinite(xmin) || !isfinite(xmax) || !isfinite(ymin) || !isfinite(ymax)
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    end
    return _anchored_projection_from_bbox(xmin, xmax, ymin, ymax, width, height; margin=margin)
end

function _projection_from_limits(limits, width::Int, height::Int; margin=0.06)
    (xmin, xmax), (ymin, ymax) = limits
    return _anchored_projection_from_bbox(Float64(xmin), Float64(xmax), Float64(ymin), Float64(ymax), width, height; margin=margin)
end

@inline function _to_svg_xy(p::SVector{2,Float64}, sx, sy, ox, oy)
    x = ox + sx * p[1]
    y = oy - sy * p[2]
    return x, y
end

function _visible_world_bounds(sx, sy, ox, oy, width::Int, height::Int)
    xmin = (0.0 - ox) / sx
    xmax = (width - ox) / sx
    ymin = (oy - height) / sy
    ymax = (oy - 0.0) / sy
    return xmin, xmax, ymin, ymax
end

function _collect_transformed_base_segments(
    ifs;
    show_base::Bool=false,
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol}=initial_polygon(),
)
    preset = _resolve_initial_polygon(initial_polygon_spec)
    base_segments = preset.segments
    transformed_by_map = Vector{Vector{Tuple{SVector{2,Float64},SVector{2,Float64}}}}(undef, length(ifs.maps))
    all_segments = Vector{Tuple{SVector{2,Float64},SVector{2,Float64}}}()

    if show_base
        append!(all_segments, base_segments)
    end

    for (i, m) in enumerate(ifs.maps)
        transformed = [(m(p1), m(p2)) for (p1, p2) in base_segments]
        transformed_by_map[i] = transformed
        append!(all_segments, transformed)
    end

    return preset, base_segments, transformed_by_map, all_segments
end

function _render_transformations_svg_impl(
    ifs;
    outpath::AbstractString="media/affine_maps.svg",
    width::Int=1200,
    height::Int=1200,
    show_base::Bool=false,
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol}=initial_polygon(),
    stroke_width::Real=2.0,
    axis::Bool=false,
)
    _, base_segments, transformed_by_map, _ =
        _collect_transformed_base_segments(ifs; show_base=show_base, initial_polygon_spec=initial_polygon_spec)
    colors = _map_colors(length(transformed_by_map))

    sx, sy, ox, oy = _projection_from_limits(ifs.limits, width, height)
    lines = String[]

    if show_base
        for (p1, p2) in base_segments
            x1, y1 = _to_svg_xy(p1, sx, sy, ox, oy)
            x2, y2 = _to_svg_xy(p2, sx, sy, ox, oy)
            push!(lines,
                  @sprintf("""  <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="#222222" stroke-width="%.2f" opacity="0.65" />""",
                           x1, y1, x2, y2, stroke_width))
        end
    end

    for (i, transformed) in enumerate(transformed_by_map)
        color = _rgb_to_hex(_map_color_rgb(i, colors))
        for (p1, p2) in transformed
            x1, y1 = _to_svg_xy(p1, sx, sy, ox, oy)
            x2, y2 = _to_svg_xy(p2, sx, sy, ox, oy)
            push!(lines,
                  @sprintf("""  <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" stroke-width="%.2f" />""",
                           x1, y1, x2, y2, color, stroke_width))
        end
    end

    if axis
        axis_color = "#6B7280"
        tick_half = 3.0
        xmin, xmax, ymin, ymax = _visible_world_bounds(sx, sy, ox, oy, width, height)

        if ymin <= 0.0 <= ymax
            x1, y1 = _to_svg_xy(SVector{2,Float64}(xmin, 0.0), sx, sy, ox, oy)
            x2, y2 = _to_svg_xy(SVector{2,Float64}(xmax, 0.0), sx, sy, ox, oy)
            push!(lines,
                  @sprintf("""  <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" stroke-width="1.00" opacity="0.8" data-role="axis-x" />""",
                           x1, y1, x2, y2, axis_color))
            for xtick in ceil(Int, xmin):floor(Int, xmax)
                x, y = _to_svg_xy(SVector{2,Float64}(Float64(xtick), 0.0), sx, sy, ox, oy)
                push!(lines,
                      @sprintf("""  <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" stroke-width="1.00" opacity="0.8" data-role="axis-x-tick" />""",
                               x, y - tick_half, x, y + tick_half, axis_color))
            end
        end

        if xmin <= 0.0 <= xmax
            x1, y1 = _to_svg_xy(SVector{2,Float64}(0.0, ymin), sx, sy, ox, oy)
            x2, y2 = _to_svg_xy(SVector{2,Float64}(0.0, ymax), sx, sy, ox, oy)
            push!(lines,
                  @sprintf("""  <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" stroke-width="1.00" opacity="0.8" data-role="axis-y" />""",
                           x1, y1, x2, y2, axis_color))
            for ytick in ceil(Int, ymin):floor(Int, ymax)
                x, y = _to_svg_xy(SVector{2,Float64}(0.0, Float64(ytick)), sx, sy, ox, oy)
                push!(lines,
                      @sprintf("""  <line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" stroke-width="1.00" opacity="0.8" data-role="axis-y-tick" />""",
                               x - tick_half, y, x + tick_half, y, axis_color))
            end
        end
    end

    body = join(lines, "\n")
    svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $width $height" style="background: transparent;" fill="none">
$body
</svg>
"""

    final_outpath = _normalize_media_outpath(outpath)
    write(final_outpath, svg)
    return final_outpath
end

function render_transformations_svg(
    ifs;
    outpath::AbstractString="media/affine_maps.svg",
    width::Int=1200,
    height::Int=1200,
    show_base::Bool=false,
    initial_polygon::Union{InitialPolygonPreset,Symbol}=initial_polygon(),
    stroke_width::Real=2.0,
    axis::Bool=false,
)
    return _render_transformations_svg_impl(ifs;
                                            outpath=outpath,
                                            width=width,
                                            height=height,
                                            show_base=show_base,
                                            initial_polygon_spec=initial_polygon,
                                            stroke_width=stroke_width,
                                            axis=axis)
end

function _draw_line!(
    img::AbstractMatrix{RGBA{Float32}},
    x1::Real, y1::Real, x2::Real, y2::Real,
    color::RGBA{Float32}
)
    height, width = size(img)
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))
    n = max(1, ceil(Int, steps))
    for k in 0:n
        t = k / n
        x = round(Int, x1 + t * dx)
        y = round(Int, y1 + t * dy)
        if 1 <= x <= width && 1 <= y <= height
            @inbounds img[y, x] = color
        end
    end
end

function _limits_xy(p::SVector{2,Float64}, limits, width::Int, height::Int)
    sx, sy, ox, oy = _projection_from_limits(limits, width, height)
    return _to_svg_xy(p, sx, sy, ox, oy)
end

function _draw_axes_on_image!(
    img::AbstractMatrix{RGBA{Float32}},
    sx,
    sy,
    ox,
    oy,
    width::Int,
    height::Int,
    axis_color::RGBA{Float32},
)
    xmin, xmax, ymin, ymax = _visible_world_bounds(sx, sy, ox, oy, width, height)

    if ymin <= 0.0 <= ymax
        x1, y1 = _to_svg_xy(SVector{2,Float64}(xmin, 0.0), sx, sy, ox, oy)
        x2, y2 = _to_svg_xy(SVector{2,Float64}(xmax, 0.0), sx, sy, ox, oy)
        _draw_line!(img, x1, y1, x2, y2, axis_color)
        for xtick in ceil(Int, xmin):floor(Int, xmax)
            x, y = _to_svg_xy(SVector{2,Float64}(Float64(xtick), 0.0), sx, sy, ox, oy)
            _draw_line!(img, x, y - 3.0, x, y + 3.0, axis_color)
        end
    end

    if xmin <= 0.0 <= xmax
        x1, y1 = _to_svg_xy(SVector{2,Float64}(0.0, ymin), sx, sy, ox, oy)
        x2, y2 = _to_svg_xy(SVector{2,Float64}(0.0, ymax), sx, sy, ox, oy)
        _draw_line!(img, x1, y1, x2, y2, axis_color)
        for ytick in ceil(Int, ymin):floor(Int, ymax)
            x, y = _to_svg_xy(SVector{2,Float64}(0.0, Float64(ytick)), sx, sy, ox, oy)
            _draw_line!(img, x - 3.0, y, x + 3.0, y, axis_color)
        end
    end
end

function _render_transformations_image(
    ifs;
    width::Int=1200,
    height::Int=1200,
    show_base::Bool=false,
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol}=initial_polygon(),
    color::Bool=true,
    axis::Bool=false,
)
    _, base_segments, transformed_by_map, _ =
        _collect_transformed_base_segments(ifs; show_base=show_base, initial_polygon_spec=initial_polygon_spec)
    map_colors = _map_colors(length(transformed_by_map))
    img = fill(RGBA{Float32}(0.0f0, 0.0f0, 0.0f0, 0.0f0), height, width)
    limits = ifs.limits

    if show_base
        base_color = color ? RGBA{Float32}(0.2f0, 0.2f0, 0.2f0, 1.0f0) : RGBA{Float32}(1.0f0, 1.0f0, 1.0f0, 1.0f0)
        for (p1, p2) in base_segments
            x1, y1 = _limits_xy(p1, limits, width, height)
            x2, y2 = _limits_xy(p2, limits, width, height)
            _draw_line!(img, x1, y1, x2, y2, base_color)
        end
    end

    for (i, transformed) in enumerate(transformed_by_map)
        map_color = color ? begin
            c = _map_color_rgb(i, map_colors)
            RGBA{Float32}(c.r, c.g, c.b, 1.0f0)
        end : RGBA{Float32}(1.0f0, 1.0f0, 1.0f0, 1.0f0)
        for (p1, p2) in transformed
            x1, y1 = _limits_xy(p1, limits, width, height)
            x2, y2 = _limits_xy(p2, limits, width, height)
            _draw_line!(img, x1, y1, x2, y2, map_color)
        end
    end

    if axis
        sx, sy, ox, oy = _projection_from_limits(limits, width, height)
        axis_color = color ? RGBA{Float32}(0.42f0, 0.45f0, 0.50f0, 1.0f0) : RGBA{Float32}(1.0f0, 1.0f0, 1.0f0, 1.0f0)
        _draw_axes_on_image!(img, sx, sy, ox, oy, width, height, axis_color)
    end

    return img
end

function _render_transformations_png_impl(
    ifs;
    outpath::AbstractString="media/affine_maps.png",
    width::Int=1200,
    height::Int=1200,
    show_base::Bool=false,
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol}=initial_polygon(),
    color::Bool=true,
    axis::Bool=false,
)
    img = _render_transformations_image(ifs;
                                        width=width,
                                        height=height,
                                        show_base=show_base,
                                        initial_polygon_spec=initial_polygon_spec,
                                        color=color,
                                        axis=axis)

    final_outpath = _normalize_media_outpath(outpath)
    save(final_outpath, img)
    return final_outpath
end

function render_transformations_png(
    ifs;
    outpath::AbstractString="media/affine_maps.png",
    width::Int=1200,
    height::Int=1200,
    show_base::Bool=false,
    initial_polygon::Union{InitialPolygonPreset,Symbol}=initial_polygon(),
    color::Bool=true,
    axis::Bool=false,
)
    return _render_transformations_png_impl(ifs;
                                            outpath=outpath,
                                            width=width,
                                            height=height,
                                            show_base=show_base,
                                            initial_polygon_spec=initial_polygon,
                                            color=color,
                                            axis=axis)
end

const _render_transformations_png_from_base_l_svg = render_transformations_png
