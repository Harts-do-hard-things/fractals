function _draw_line!(
    img::AbstractMatrix{T},
    x1::Real,
    y1::Real,
    x2::Real,
    y2::Real,
    color::T,
) where {T}
    height, width = size(img)
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))
    n = max(1, ceil(Int, steps))
    for k = 0:n
        t = k / n
        x = round(Int, x1 + t * dx)
        y = round(Int, y1 + t * dy)
        if 1 <= x <= width && 1 <= y <= height
            @inbounds img[y, x] = color
        end
    end
end

function _draw_axes_on_image!(
    img::AbstractMatrix{T},
    sx,
    sy,
    ox,
    oy,
    width::Int,
    height::Int,
    axis_color::T,
) where {T}
    xmin, xmax, ymin, ymax = _visible_world_bounds(sx, sy, ox, oy, width, height)

    if ymin <= 0.0 <= ymax
        x1, y1 = _to_svg_xy(SVector{2,Float64}(xmin, 0.0), sx, sy, ox, oy)
        x2, y2 = _to_svg_xy(SVector{2,Float64}(xmax, 0.0), sx, sy, ox, oy)
        _draw_line!(img, x1, y1, x2, y2, axis_color)
        for xtick = ceil(Int, xmin):floor(Int, xmax)
            x, y = _to_svg_xy(SVector{2,Float64}(Float64(xtick), 0.0), sx, sy, ox, oy)
            _draw_line!(img, x, y - 3.0, x, y + 3.0, axis_color)
        end
    end

    if xmin <= 0.0 <= xmax
        x1, y1 = _to_svg_xy(SVector{2,Float64}(0.0, ymin), sx, sy, ox, oy)
        x2, y2 = _to_svg_xy(SVector{2,Float64}(0.0, ymax), sx, sy, ox, oy)
        _draw_line!(img, x1, y1, x2, y2, axis_color)
        for ytick = ceil(Int, ymin):floor(Int, ymax)
            x, y = _to_svg_xy(SVector{2,Float64}(0.0, Float64(ytick)), sx, sy, ox, oy)
            _draw_line!(img, x - 3.0, y, x + 3.0, y, axis_color)
        end
    end
end

function _draw_axes_on_raster!(
    img::AbstractMatrix{T},
    limits,
    width::Int,
    height::Int,
    axis_color::T,
) where {T}
    pmap = make_pixelate_map(limits; resolution = (height, width))
    inv_pmap = inv(pmap)
    corners = (
        inv_pmap(SVector{2,Float64}(1.0, 1.0)),
        inv_pmap(SVector{2,Float64}(width * 1.0, 1.0)),
        inv_pmap(SVector{2,Float64}(1.0, height * 1.0)),
        inv_pmap(SVector{2,Float64}(width * 1.0, height * 1.0)),
    )
    xs = map(p -> p[1], corners)
    ys = map(p -> p[2], corners)
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)

    if ymin <= 0.0 <= ymax
        x1, y1 = pmap(SVector{2,Float64}(xmin, 0.0))
        x2, y2 = pmap(SVector{2,Float64}(xmax, 0.0))
        _draw_line!(img, x1, y1, x2, y2, axis_color)
        for xtick = ceil(Int, xmin):floor(Int, xmax)
            x, y = pmap(SVector{2,Float64}(Float64(xtick), 0.0))
            _draw_line!(img, x, y - 3.0, x, y + 3.0, axis_color)
        end
    end

    if xmin <= 0.0 <= xmax
        x1, y1 = pmap(SVector{2,Float64}(0.0, ymin))
        x2, y2 = pmap(SVector{2,Float64}(0.0, ymax))
        _draw_line!(img, x1, y1, x2, y2, axis_color)
        for ytick = ceil(Int, ymin):floor(Int, ymax)
            x, y = pmap(SVector{2,Float64}(0.0, Float64(ytick)))
            _draw_line!(img, x - 3.0, y, x + 3.0, y, axis_color)
        end
    end
end

function _overlay_initial_polygon!(
    img::AbstractMatrix{Gray{Float32}},
    ifs;
    width::Int,
    height::Int,
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol} = initial_polygon(),
    color::Bool = false,
)
    preset = _resolve_initial_polygon(initial_polygon_spec)
    line_color = Gray{Float32}(1.0f0)
    for (p1, p2) in preset.segments
        x1, y1 = _to_raster_xy(p1, ifs.limits, width, height)
        x2, y2 = _to_raster_xy(p2, ifs.limits, width, height)
        _draw_line!(img, x1, y1, x2, y2, line_color)
    end
    return img
end

function _overlay_initial_polygon!(
    img::AbstractMatrix{RGB{Float32}},
    ifs;
    width::Int,
    height::Int,
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol} = initial_polygon(),
    color::Bool = true,
)
    preset = _resolve_initial_polygon(initial_polygon_spec)
    line_color = color ? RGB{Float32}(0.2f0, 0.2f0, 0.2f0) : RGB{Float32}(1.0f0, 1.0f0, 1.0f0)
    for (p1, p2) in preset.segments
        x1, y1 = _to_raster_xy(p1, ifs.limits, width, height)
        x2, y2 = _to_raster_xy(p2, ifs.limits, width, height)
        _draw_line!(img, x1, y1, x2, y2, line_color)
    end
    return img
end

function _overlay_axes!(img::AbstractMatrix{Gray{Float32}}, ifs; width::Int, height::Int)
    _draw_axes_on_raster!(img, ifs.limits, width, height, Gray{Float32}(1.0f0))
    return img
end

function _overlay_axes!(img::AbstractMatrix{RGB{Float32}}, ifs; width::Int, height::Int)
    _draw_axes_on_raster!(img, ifs.limits, width, height, RGB{Float32}(0.42f0, 0.45f0, 0.50f0))
    return img
end
