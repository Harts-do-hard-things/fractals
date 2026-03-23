function _fit_bounds(segments, width::Int, height::Int; margin=0.06)
    xmin = Inf
    xmax = -Inf
    ymin = Inf
    ymax = -Inf

    for (p1, p2) in segments
        x1, y1 = p1
        x2, y2 = p2
        xmin = min(xmin, x1, x2)
        xmax = max(xmax, x1, x2)
        ymin = min(ymin, y1, y2)
        ymax = max(ymax, y1, y2)
    end

    if !isfinite(xmin) || !isfinite(xmax) || !isfinite(ymin) || !isfinite(ymax)
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    end

    dx = xmax - xmin
    dy = ymax - ymin
    dx = dx == 0 ? 1.0 : dx
    dy = dy == 0 ? 1.0 : dy

    draw_w = (1 - 2margin) * width
    draw_h = (1 - 2margin) * height
    s = min(draw_w / dx, draw_h / dy)
    sx = sy = s

    offset_x = (width - sx * dx) / 2 - sx * xmin
    # SVG y-axis grows down, so invert y when mapping
    offset_y = (height - sy * dy) / 2 + sy * ymax

    return sx, sy, offset_x, offset_y
end

@inline function _to_svg_xy(p::SVector{2,Float64}, sx, sy, ox, oy)
    x = ox + sx * p[1]
    y = oy - sy * p[2]
    return x, y
end

function _collect_transformed_base_segments(ifs; show_base::Bool=false, initial_polygon::Symbol=:default)
    base_segments = _base_l_image(initial_polygon)
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

    return base_segments, transformed_by_map, all_segments
end

function render_transformations_svg(
    ifs;
    outpath::AbstractString="media/affine_maps.svg",
    width::Int=1200,
    height::Int=1200,
    show_base::Bool=false,
    initial_polygon::Symbol=:default,
    stroke_width::Real=2.0
)
    base_segments, transformed_by_map, all_segments =
        _collect_transformed_base_segments(ifs; show_base=show_base, initial_polygon=initial_polygon)
    colors = _map_colors(length(transformed_by_map))

    sx, sy, ox, oy = _fit_bounds(all_segments, width, height)
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
    (xmin, xmax), (ymin, ymax) = limits
    dx = xmax - xmin
    dy = ymax - ymin
    dx = dx == 0 ? 1.0 : dx
    dy = dy == 0 ? 1.0 : dy

    nx = (p[1] - xmin) / dx
    ny = (p[2] - ymin) / dy
    x = 1 + nx * (width - 1)
    y = 1 + (1 - ny) * (height - 1)
    return x, y
end

function _render_transformations_image(
    ifs;
    width::Int=1200,
    height::Int=1200,
    show_base::Bool=false,
    initial_polygon::Symbol=:default,
    limits_mode::Symbol=:ifs,
    color::Bool=true,
)
    base_segments, transformed_by_map, all_segments =
        _collect_transformed_base_segments(ifs; show_base=show_base, initial_polygon=initial_polygon)
    map_colors = _map_colors(length(transformed_by_map))
    img = fill(RGBA{Float32}(0.0f0, 0.0f0, 0.0f0, 0.0f0), height, width)

    if limits_mode == :ifs || limits_mode == :default
        limits = limits_mode == :ifs ? ifs.limits : _base_limits_image(initial_polygon)
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
        return img
    elseif limits_mode == :fit
        sx, sy, ox, oy = _fit_bounds(all_segments, width, height)

        if show_base
            base_color = color ? RGBA{Float32}(0.2f0, 0.2f0, 0.2f0, 1.0f0) : RGBA{Float32}(1.0f0, 1.0f0, 1.0f0, 1.0f0)
            for (p1, p2) in base_segments
                x1, y1 = _to_svg_xy(p1, sx, sy, ox, oy)
                x2, y2 = _to_svg_xy(p2, sx, sy, ox, oy)
                _draw_line!(img, x1, y1, x2, y2, base_color)
            end
        end

        for (i, transformed) in enumerate(transformed_by_map)
            map_color = color ? begin
                c = _map_color_rgb(i, map_colors)
                RGBA{Float32}(c.r, c.g, c.b, 1.0f0)
            end : RGBA{Float32}(1.0f0, 1.0f0, 1.0f0, 1.0f0)
            for (p1, p2) in transformed
                x1, y1 = _to_svg_xy(p1, sx, sy, ox, oy)
                x2, y2 = _to_svg_xy(p2, sx, sy, ox, oy)
                _draw_line!(img, x1, y1, x2, y2, map_color)
            end
        end
        return img
    end

    throw(ArgumentError("Invalid limits_mode '$limits_mode'. Supported: :ifs, :default"))
end

function render_transformations_png(
    ifs;
    outpath::AbstractString="media/affine_maps.png",
    width::Int=1200,
    height::Int=1200,
    show_base::Bool=false,
    initial_polygon::Symbol=:default,
    limits_mode::Symbol=:ifs,
    color::Bool=true,
)
    img = _render_transformations_image(ifs;
                                        width=width,
                                        height=height,
                                        show_base=show_base,
                                        initial_polygon=initial_polygon,
                                        limits_mode=limits_mode,
                                        color=color)

    final_outpath = _normalize_media_outpath(outpath)
    save(final_outpath, img)
    return final_outpath
end

const _render_transformations_png_from_base_l_svg = render_transformations_png
