function _limits_from_polygon_points(points::AbstractVector{SVector{2,Float64}})
    xmin = Inf
    xmax = -Inf
    ymin = Inf
    ymax = -Inf
    for p in points
        xmin = min(xmin, p[1])
        xmax = max(xmax, p[1])
        ymin = min(ymin, p[2])
        ymax = max(ymax, p[2])
    end
    if !isfinite(xmin) || !isfinite(xmax) || !isfinite(ymin) || !isfinite(ymax)
        return initial_polygon().limits
    end
    if xmin == xmax
        xmin -= 0.5
        xmax += 0.5
    end
    if ymin == ymax
        ymin -= 0.5
        ymax += 0.5
    end

    dx = xmax - xmin
    dy = ymax - ymin
    m = max(dx, dy)
    pad = 0.06 * m

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    half = (m + 2pad) / 2

    return ((cx - half, cx + half), (cy - half, cy + half))
end

function _unique_polygon_points(segments)
    seen = Set{Tuple{Float64,Float64}}()
    points = SVector{2,Float64}[]
    for (p1, p2) in segments
        for p in (p1, p2)
            key = (p[1], p[2])
            key in seen && continue
            push!(seen, key)
            push!(points, p)
        end
    end
    return points
end

function _select_extreme_polygon_points(points::AbstractVector{SVector{2,Float64}})
    isempty(points) && return SVector{2,Float64}[]

    minx = minimum(p -> p[1], points)
    maxx = maximum(p -> p[1], points)
    miny = minimum(p -> p[2], points)
    maxy = maximum(p -> p[2], points)

    kept = SVector{2,Float64}[]
    seen = Set{Tuple{Float64,Float64}}()

    function keep_matches(pred)
        saved = 0
        for p in points
            pred(p) || continue
            key = (p[1], p[2])
            key in seen && continue
            push!(seen, key)
            push!(kept, p)
            saved += 1
            saved == 2 && break
        end
    end

    keep_matches(p -> p[1] == minx)
    keep_matches(p -> p[1] == maxx)
    keep_matches(p -> p[2] == miny)
    keep_matches(p -> p[2] == maxy)
    return kept
end

function _limits_from_polygon_point_iterations(
    ifs;
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol} = initial_polygon(),
    polygon_limits_iterations::Integer = 1,
)
    polygon_limits_iterations >= 0 || throw(
        ArgumentError("polygon_limits_iterations must be >= 0, got $polygon_limits_iterations"),
    )

    preset = _resolve_initial_polygon(initial_polygon_spec)
    frontier = _select_extreme_polygon_points(_unique_polygon_points(preset.segments))
    retained = copy(frontier)

    for _ = 1:polygon_limits_iterations
        next_points = SVector{2,Float64}[]
        for p in frontier, m in ifs.maps
            push!(next_points, m(p))
        end
        isempty(next_points) && break
        frontier = _select_extreme_polygon_points(next_points)
        append!(retained, frontier)
    end

    return _limits_from_polygon_points(retained)
end

function _build_polygon_raster_ifs(
    ifs;
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol} = initial_polygon(),
    polygon_limits_iterations::Integer = 1,
)
    limits = _limits_from_polygon_point_iterations(
        ifs;
        initial_polygon_spec = initial_polygon_spec,
        polygon_limits_iterations = polygon_limits_iterations,
    )
    return IFS(ifs.name, ifs.maps, ifs.weights, limits)
end

@inline function _render_transformations_effective_ifs(
    ifs;
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol} = initial_polygon(),
    polygon_limits_iterations::Integer = 1,
)
    return _build_polygon_raster_ifs(
        ifs;
        initial_polygon_spec = initial_polygon_spec,
        polygon_limits_iterations = polygon_limits_iterations,
    )
end

function _collect_transformed_base_segments(
    ifs;
    show_base::Bool = false,
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol} = initial_polygon(),
)
    preset = _resolve_initial_polygon(initial_polygon_spec)
    base_segments = preset.segments
    transformed_by_map =
        Vector{Vector{Tuple{SVector{2,Float64},SVector{2,Float64}}}}(undef, length(ifs.maps))
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

function _render_initial_polygon_seed_image(
    ifs;
    width::Int = 1200,
    height::Int = 1200,
    initial_polygon_spec::Union{InitialPolygonPreset,Symbol} = initial_polygon(),
)
    preset = _resolve_initial_polygon(initial_polygon_spec)
    img = fill(Gray{Float32}(0.0f0), height, width)
    for (p1, p2) in preset.segments
        x1, y1 = _to_raster_xy(p1, ifs.limits, width, height)
        x2, y2 = _to_raster_xy(p2, ifs.limits, width, height)
        _draw_line!(img, x1, y1, x2, y2, Gray{Float32}(1.0f0))
    end
    return img
end

# --------------------------------
# Vector geometry for SVG renders
# --------------------------------

const _Segment = Tuple{SVector{2,Float64},SVector{2,Float64}}

# The "image" an SVG render draws: line segments grouped by the map that produced them (one
# stroke color per group when the render is colored).
struct SegmentImage
    groups::Vector{Vector{_Segment}}
end

# Upper bound on the segments of one SVG render, checked before iterating.
const _MAX_SVG_SEGMENTS = 1_000_000

"""
    _iterate_segments(ifs, segments, n) -> Vector{Vector{Segment}}

Apply every map of `ifs` to every segment, `n` times over. The result is grouped by the last map
applied (`length(ifs.maps)` groups holding `length(segments) * nmaps^n` segments in all); with
`n = 0` it is the single group `segments`. Throws `ArgumentError`, before iterating, when the
total would exceed 1_000_000 segments.
"""
function _iterate_segments(ifs, segments, n::Integer)
    n >= 0 || throw(ArgumentError("iterations must be >= 0, got $n"))
    nmaps = length(ifs.maps)
    total = float(length(segments)) * float(nmaps)^n
    if total > _MAX_SVG_SEGMENTS
        shown = total < 1e18 ? string(round(Int, total)) : "more than 10^18"
        throw(
            ArgumentError(
                "SVG render would draw $shown segments ($(length(segments)) x $nmaps^$n), " *
                "more than the limit of $_MAX_SVG_SEGMENTS; lower the iterations or render a PNG",
            ),
        )
    end
    current = Vector{_Segment}(segments)
    groups = Vector{Vector{_Segment}}([current])
    for _ = 1:n
        groups = Vector{_Segment}[[(m(p1), m(p2)) for (p1, p2) in current] for m in ifs.maps]
        current = reduce(vcat, groups)
    end
    return groups
end
