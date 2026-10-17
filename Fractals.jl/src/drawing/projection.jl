function _anchored_projection_from_bbox(
    xmin::Float64,
    xmax::Float64,
    ymin::Float64,
    ymax::Float64,
    width::Int,
    height::Int;
    margin = 0.06,
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

function _projection_from_limits(limits, width::Int, height::Int; margin = 0.06)
    (xmin, xmax), (ymin, ymax) = limits
    return _anchored_projection_from_bbox(
        Float64(xmin),
        Float64(xmax),
        Float64(ymin),
        Float64(ymax),
        width,
        height;
        margin = margin,
    )
end

@inline function _to_svg_xy(p::SVector{2,Float64}, sx, sy, ox, oy)
    x = ox + sx * p[1]
    y = oy - sy * p[2]
    return x, y
end

@inline function _to_raster_xy(p::SVector{2,Float64}, limits, width::Int, height::Int)
    pmap = make_pixelate_map(limits; resolution = (height, width))
    return pmap(p)
end

function _visible_world_bounds(sx, sy, ox, oy, width::Int, height::Int)
    xmin = (0.0 - ox) / sx
    xmax = (width - ox) / sx
    ymin = (oy - height) / sy
    ymax = (oy - 0.0) / sy
    return xmin, xmax, ymin, ymax
end

function _limits_xy(p::SVector{2,Float64}, limits, width::Int, height::Int)
    sx, sy, ox, oy = _projection_from_limits(limits, width, height)
    return _to_svg_xy(p, sx, sy, ox, oy)
end
