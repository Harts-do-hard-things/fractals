# SVG output for the render pipeline: `finish(::Render{<:Any,SVG}, ::SegmentImage, limits)` turns
# vector geometry plus the render's color, alpha and overlays into the document text.

function _svg_line(x1, y1, x2, y2, stroke, stroke_width; opacity = nothing, role = nothing)
    attrs = @sprintf(
        "x1=\"%.3f\" y1=\"%.3f\" x2=\"%.3f\" y2=\"%.3f\" stroke=\"%s\" stroke-width=\"%.2f\"",
        x1,
        y1,
        x2,
        y2,
        stroke,
        stroke_width
    )
    isnothing(opacity) || (attrs *= @sprintf(" opacity=\"%.2f\"", opacity))
    isnothing(role) || (attrs *= " data-role=\"$role\"")
    return "  <line $attrs />"
end

function _svg_base_polygon_lines(o::BasePolygon, r::Render, sx, sy, ox, oy)
    stroke = r.color ? "#222222" : "#FFFFFF"
    lines = String[]
    for (p1, p2) in _resolve_initial_polygon(o.polygon).segments
        x1, y1 = _to_svg_xy(p1, sx, sy, ox, oy)
        x2, y2 = _to_svg_xy(p2, sx, sy, ox, oy)
        push!(
            lines,
            _svg_line(x1, y1, x2, y2, stroke, r.format.stroke_width; opacity = 0.65),
        )
    end
    return lines
end

function _svg_axes_lines(r::Render, sx, sy, ox, oy, width::Int, height::Int)
    stroke = r.color ? "#6B7280" : "#FFFFFF"
    tick_half = 3.0
    lines = String[]
    xmin, xmax, ymin, ymax = _visible_world_bounds(sx, sy, ox, oy, width, height)

    function add(x1, y1, x2, y2, role)
        push!(lines, _svg_line(x1, y1, x2, y2, stroke, 1.0; opacity = 0.8, role))
    end

    if ymin <= 0.0 <= ymax
        x1, y1 = _to_svg_xy(SVector{2,Float64}(xmin, 0.0), sx, sy, ox, oy)
        x2, y2 = _to_svg_xy(SVector{2,Float64}(xmax, 0.0), sx, sy, ox, oy)
        add(x1, y1, x2, y2, "axis-x")
        for xtick = ceil(Int, xmin):floor(Int, xmax)
            x, y = _to_svg_xy(SVector{2,Float64}(Float64(xtick), 0.0), sx, sy, ox, oy)
            add(x, y - tick_half, x, y + tick_half, "axis-x-tick")
        end
    end

    if xmin <= 0.0 <= xmax
        x1, y1 = _to_svg_xy(SVector{2,Float64}(0.0, ymin), sx, sy, ox, oy)
        x2, y2 = _to_svg_xy(SVector{2,Float64}(0.0, ymax), sx, sy, ox, oy)
        add(x1, y1, x2, y2, "axis-y")
        for ytick = ceil(Int, ymin):floor(Int, ymax)
            x, y = _to_svg_xy(SVector{2,Float64}(0.0, Float64(ytick)), sx, sy, ox, oy)
            add(x - tick_half, y, x + tick_half, y, "axis-y-tick")
        end
    end
    return lines
end

# The SVG document of a render: an opaque black background unless `alpha`, any `BasePolygon`
# overlays, the segment groups (one stroke color per group when `color`, otherwise white), then
# any `Axes` overlays.
function _svg_document(r::Render, img::SegmentImage, limits)
    rows, cols = r.resolution
    sx, sy, ox, oy = _projection_from_limits(limits, cols, rows)
    lines = String[]
    r.alpha || push!(lines, """  <rect width="100%" height="100%" fill="#000000" />""")

    for o in r.overlays
        o isa BasePolygon && append!(lines, _svg_base_polygon_lines(o, r, sx, sy, ox, oy))
    end

    colors = _map_colors(length(img.groups))
    for (i, group) in enumerate(img.groups)
        stroke = r.color ? _rgb_to_hex(_map_color_rgb(i, colors)) : "#FFFFFF"
        for (p1, p2) in group
            x1, y1 = _to_svg_xy(p1, sx, sy, ox, oy)
            x2, y2 = _to_svg_xy(p2, sx, sy, ox, oy)
            push!(lines, _svg_line(x1, y1, x2, y2, stroke, r.format.stroke_width))
        end
    end

    for o in r.overlays
        o isa Axes && append!(lines, _svg_axes_lines(r, sx, sy, ox, oy, cols, rows))
    end

    body = join(lines, "\n")
    return """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $cols $rows" style="background: transparent;" fill="none">
$body
</svg>
"""
end

"""
    render_transformations_svg(ifs; outpath=nothing, width=1200, height=1200, show_base=false,
                               initial_polygon=initial_polygon(), polygon_limits_iterations=1,
                               stroke_width=2.0, axis=false) -> String

Write an SVG of each affine map applied to the seed polygon (one stroke color per map, transparent
background) and return its path. A thin wrapper over
`render(Render(ifs, Transformations(...); format = SVG(), ...))`: `show_base` and `axis` become
the [`BasePolygon`](@ref) and [`Axes`](@ref) overlays, and the render is always redrawn and
written (`force = true`), with its render record. `outpath` is resolved like `render`'s; the
file starts with a `<!-- fractals.provenance: ... -->` comment.
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
    polygon = initial_polygon isa InitialPolygonPreset ? initial_polygon.name : initial_polygon
    overlays = Overlay[]
    show_base && push!(overlays, BasePolygon(polygon))
    axis && push!(overlays, Axes())
    r = Render(
        ifs,
        Transformations(; polygon, limits_iterations = polygon_limits_iterations);
        resolution = (height, width),
        color = true,
        alpha = true,
        overlays,
        format = SVG(; stroke_width),
    )
    return render(r; outpath, force = true).outpath
end
