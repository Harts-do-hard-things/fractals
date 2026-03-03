#!/usr/bin/env julia
# ===============================
# matrixfractal.jl (cleaned)
# ===============================

using StaticArrays
using LinearAlgebra
using StatsBase
using Images
using FileIO
using Colors
using Random
using Base.Threads
using Printf

# --------------------------------
# Configuration
# --------------------------------

const RESOLUTION = (1504, 2256)
const DEFAULT_WARMUP = 50
const DEFAULT_SAMPLES = 1_000_000
const DEFAULT_MEDIA_DIR = "media"
const _DEFAULT_INITIAL_POLYGON_LIMITS = ((0.0, 1.0), (0.0, 1.0))
const _DEFAULT_INITIAL_POLYGON_SEGMENTS = [
    (SVector{2,Float64}(0.0, 1.0), SVector{2,Float64}(1.0, 1.0)),
    (SVector{2,Float64}(1.0, 1.0), SVector{2,Float64}(1.0, 0.0)),
    (SVector{2,Float64}(1.0, 0.0), SVector{2,Float64}(0.0, 0.0)),
    (SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(0.0, 1.0)),
    (SVector{2,Float64}(1/6, 5/6), SVector{2,Float64}(1/6, 1/6)),
    (SVector{2,Float64}(1/6, 1/6), SVector{2,Float64}(5/9, 1/6)),
]
const _EQUILATERAL_TRIANGLE_LIMITS = ((0.0, 1.0), (0.0, sqrt(3.0) / 2.0))
const _EQUILATERAL_TRIANGLE_SEGMENTS = [
    (SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)),
    (SVector{2,Float64}(1.0, 0.0), SVector{2,Float64}(0.5, sqrt(3.0) / 2.0)),
    (SVector{2,Float64}(0.5, sqrt(3.0) / 2.0), SVector{2,Float64}(0.0, 0.0)),
]
const _LINE_BASE_LIMITS = ((0.0, 1.0), (-0.1, 0.1))
const _LINE_ARROW_SEGMENTS = [
    (SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)),
    (SVector{2,Float64}(0.5, 0.0), SVector{2,Float64}(0.45, 0.05)),
    (SVector{2,Float64}(0.5, 0.0), SVector{2,Float64}(0.45, -0.05)),
]
const _LINE_SEGMENTS = [
    (SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)),
]
const _INITIAL_POLYGON_NAMES = (:default, :equilateral_triangle, :line_arrow, :line)

@enum RenderMethod begin
    Chaos
    Parallel
    PointDeterministic
    ImageIterate
    Inverse
end

const _RENDER_METHOD_CHOICES = (
    Chaos,
    Parallel,
    PointDeterministic,
    ImageIterate,
    Inverse,
)

function _render_method_symbol(m::RenderMethod)
    if m == PointDeterministic
        return :point_deterministic
    elseif m == ImageIterate
        return :image_iterate
    end
    return Symbol(lowercase(string(m)))
end

function _parse_render_method(method::RenderMethod)
    return method
end

function _parse_render_method(method::Symbol)
    m = Symbol(lowercase(String(method)))
    if m == :chaos
        return Chaos
    elseif m == :parallel
        return Parallel
    elseif m == :point_deterministic || m == :pointdeterministic
        return PointDeterministic
    elseif m == :image_iterate || m == :imageiterate
        return ImageIterate
    elseif m == :inverse
        return Inverse
    elseif m == :deterministic
        throw(ArgumentError("Method '$method' was removed. Use point_deterministic."))
    end
    throw(ArgumentError("Invalid method '$method'. Supported methods: chaos, parallel, point_deterministic, image_iterate, inverse"))
end

function _parse_render_method(method::AbstractString)
    return _parse_render_method(Symbol(lowercase(strip(method))))
end

function _normalize_media_outpath(outpath::AbstractString)
    raw = String(outpath)
    media_prefix = string(DEFAULT_MEDIA_DIR, Base.Filesystem.path_separator)
    normalized = normpath(raw)

    if normalized == DEFAULT_MEDIA_DIR || startswith(normalized, media_prefix)
        mkpath(dirname(normalized))
        return normalized
    end

    final = joinpath(DEFAULT_MEDIA_DIR, basename(normalized))
    mkpath(dirname(final))
    return final
end

@inline function _initial_polygon_names_text()
    return join(string.(collect(_INITIAL_POLYGON_NAMES)), ", ")
end

function _resolve_initial_polygon(initial_polygon::Symbol)
    normalized = Symbol(lowercase(String(initial_polygon)))
    if normalized == :default
        return _DEFAULT_INITIAL_POLYGON_SEGMENTS, _DEFAULT_INITIAL_POLYGON_LIMITS
    elseif normalized == :equilateral_triangle
        return _EQUILATERAL_TRIANGLE_SEGMENTS, _EQUILATERAL_TRIANGLE_LIMITS
    elseif normalized == :line_arrow
        return _LINE_ARROW_SEGMENTS, _LINE_BASE_LIMITS
    elseif normalized == :line
        return _LINE_SEGMENTS, _LINE_BASE_LIMITS
    end
    throw(ArgumentError("Invalid initial_polygon '$initial_polygon'. Supported: $(_initial_polygon_names_text())"))
end

function _base_limits_image(initial_polygon::Symbol=:default)
    _, limits = _resolve_initial_polygon(initial_polygon)
    return limits
end

function _base_l_image(initial_polygon::Symbol=:default)
    segments, _ = _resolve_initial_polygon(initial_polygon)
    return segments
end

@inline function _thread_buffer_slots()
    return max(nthreads(), Base.Threads.maxthreadid())
end

@inline function _gpu_backend_available(::Val{:cuda})
    return false
end

function _map_colors(n::Integer)
    n <= 0 && return RGB{Float32}[]
    # Generate visually distinct, deterministic colors and exclude near-white/near-black colors.
    candidates = distinguishable_colors(max(3n, n + 8), [RGB(1, 1, 1), RGB(0, 0, 0)])
    palette = RGB{Float32}[]
    sizehint!(palette, n)
    for c in candidates
        # Relative luminance in sRGB space; keep mid-range tones only.
        lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
        if 0.15 < lum < 0.85
            push!(palette, RGB{Float32}(Float32(c.r), Float32(c.g), Float32(c.b)))
            length(palette) == n && break
        end
    end
    # Fallback to initial candidates if filtering was too aggressive for small n.
    if length(palette) < n
        for c in candidates
            push!(palette, RGB{Float32}(Float32(c.r), Float32(c.g), Float32(c.b)))
            length(palette) == n && break
        end
    end
    return palette
end

@inline function _map_color_rgb(i::Integer, colors::Vector{RGB{Float32}})
    return colors[i]
end

@inline function _rgb_to_hex(c::RGB{Float32})
    r = round(Int, clamp(c.r, 0.0f0, 1.0f0) * 255)
    g = round(Int, clamp(c.g, 0.0f0, 1.0f0) * 255)
    b = round(Int, clamp(c.b, 0.0f0, 1.0f0) * 255)
    return @sprintf("#%02X%02X%02X", r, g, b)
end

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


# --------------------------------
# Affine Map
# --------------------------------

struct AffineMap{T<:AbstractFloat}
    A::SMatrix{2,2,T,4}
    b::SVector{2,T}
end

(m::AffineMap)(x::SVector{2,T}) where T = m.A * x + m.b

function Base.show(io::IO, m::AffineMap)
    println(io, "AffineMap:")
    println(io, "  [", @sprintf("% .5f", m.A[1,1]), "  ", @sprintf("% .5f", m.A[1,2]), "] [x]   [", @sprintf("% .5f", m.b[1]), "]")
    println(io, "  [", @sprintf("% .5f", m.A[2,1]), "  ", @sprintf("% .5f", m.A[2,2]), "] [y] + [", @sprintf("% .5f", m.b[2]), "]")
end

Base.inv(m::AffineMap{T}) where T = begin
    Ainv = inv(m.A)
    AffineMap(Ainv, -Ainv * m.b)
end

AffineMap(A::AbstractMatrix{T},
          b::AbstractVector{T}) where {T<:AbstractFloat} =
    AffineMap(
        SMatrix{2,2,T}(A),
        SVector{2,T}(b)
    )
AffineMap(a11::T,a12::T,
          a21::T,a22::T,
          b1::T,b2::T) where {T<:AbstractFloat} =
    AffineMap(
        SMatrix{2,2,Float64,4}((a11,a12,a21,a22)),
        SVector{2,Float64}(b1,b2)
    )

# --------------------------------
# IFS Type
# --------------------------------

struct IFS
    name::String
    docs::String
    points::Vector{SVector{2,Float64}} # TODO make this parameterizable
    maps::Vector{AffineMap{Float64}}
    weights::Weights{Float64,Float64,Vector{Float64}}
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}}
end

function Base.show(io::IO, f::IFS)
    println(io, "IFS: $(f.name)")
    if !isempty(f.docs)
        println(io, "Docs:")
        for line in split(f.docs, '\n')
            println(io, "  ", line)
        end
    end
    println(io, "Maps ($(length(f.maps))):")
    for (i, m) in enumerate(f.maps)
        println(io, "  [$i]")
        show(io, m)
    end
    println(io, "Points: $(length(f.points))")
end

# --------------------------------
# Build maps from equation matrix
# --------------------------------

function _build_maps_and_weights(eq::AbstractMatrix{<:Real})
    n = size(eq, 1)
    maps = Vector{AffineMap{Float64}}(undef, n)
    probs = Vector{Float64}(undef, n)

    has_probs = size(eq, 2) == 7

    for i in 1:n
        a11,a12,a21,a22,b1,b2 = Float64.(eq[i,1:6])
        maps[i] = AffineMap(a11,a12,a21,a22,b1,b2)

        if has_probs
            probs[i] = Float64(eq[i,7])
        else
            A = SMatrix{2,2,Float64,4}((a11,a12,a21,a22))
            probs[i] = abs(det(A))
        end
    end

    return maps, Weights(probs)
end

function _validate_eq_matrix(eq::AbstractMatrix{<:Real})
    nrows, ncols = size(eq)
    nrows > 0 || throw(ArgumentError("IFS equation matrix must have at least one row, got size $(size(eq))"))
    (ncols == 6 || ncols == 7) || throw(ArgumentError("IFS equation matrix must have exactly 6 or 7 columns, got $ncols"))

    all(isfinite, eq) || throw(ArgumentError("IFS equation matrix contains non-finite values (NaN or Inf)"))

    if ncols == 7
        probs = eq[:, 7]
        all(p -> p >= 0, probs) || throw(ArgumentError("IFS probability column (7th column) must be nonnegative"))
        sum(probs) > 0 || throw(ArgumentError("IFS probability column (7th column) must have positive total weight"))
    end

    return nothing
end

# --------------------------------
# Compute limits via sampling
# --------------------------------

function _get_limits(maps, weights;
                    warmup=DEFAULT_WARMUP,
                    n=10_000)

    map_indices = Base.OneTo(length(maps))
    x = SVector{2,Float64}(0.0, 0.0)

    for _ in 1:warmup
        x = maps[sample(map_indices, weights)](x)
    end

    xmin = Inf; xmax = -Inf
    ymin = Inf; ymax = -Inf

    for _ in 1:n
        x = maps[sample(map_indices, weights)](x)
        xx, yy = x
        xmin = min(xmin, xx)
        xmax = max(xmax, xx)
        ymin = min(ymin, yy)
        ymax = max(ymax, yy)
    end

    dx = xmax - xmin
    dy = ymax - ymin
    m = max(dx, dy)
    pad = 0.05m

    cx = (xmin + xmax)/2
    cy = (ymin + ymax)/2
    half = (m + 2pad)/2

    return ((cx-half, cx+half),
            (cy-half, cy+half))
end

# --------------------------------
# Constructors
# --------------------------------

function IFS(eq::AbstractMatrix{<:Real};
             npoints=DEFAULT_SAMPLES,
             name::AbstractString="",
             docs::AbstractString="")

    _validate_eq_matrix(eq)
    maps, weights = _build_maps_and_weights(eq)
    limits = _get_limits(maps, weights)

    points = [SVector{2,Float64}(0.0,0.0)
              for _ in 1:npoints]

    return IFS(String(name), String(docs), points, maps, weights, limits)
end

function IFS(maps::Vector{AffineMap{Float64}},
             weights::Weights;
             npoints::Integer=DEFAULT_SAMPLES,
             name::AbstractString="",
             docs::AbstractString="",
             limits=_get_limits(maps, weights))
    points = [SVector{2,Float64}(0.0,0.0)
              for _ in 1:npoints]
    return IFS(String(name), String(docs), points, maps, weights, limits)
end

# --------------------------------
# Chaos Game Iteration
# --------------------------------

function _iterate_serial!(ifs::IFS;
                          warmup=DEFAULT_WARMUP,
                          seed::Union{Nothing,Integer}=nothing)

    maps = ifs.maps
    weights = ifs.weights
    map_indices = Base.OneTo(length(maps))

    x = SVector{2,Float64}(0.0, 0.0)

    if isnothing(seed)
        for _ in 1:warmup
            x = maps[sample(map_indices, weights)](x)
        end
        idxs = sample(map_indices, weights, length(ifs.points))
    else
        rng = MersenneTwister(seed)
        for _ in 1:warmup
            x = maps[sample(rng, map_indices, weights)](x)
        end
        idxs = sample(rng, map_indices, weights, length(ifs.points))
    end

    for i in eachindex(ifs.points)
        x = maps[idxs[i]](x)
        ifs.points[i] = x
    end

    return ifs
end

function _iterate_parallel!(ifs::IFS;
                            warmup=DEFAULT_WARMUP,
                            seed::Union{Nothing,Integer}=nothing)
    maps = ifs.maps
    weights = ifs.weights
    map_indices = Base.OneTo(length(maps))
    npts = length(ifs.points)

    @threads for tid in 1:nthreads()
        # chunk for this thread
        lo = fld((tid-1)*npts, nthreads()) + 1
        hi = fld(tid*npts, nthreads())
        lo > hi && continue

        rng = isnothing(seed) ? nothing : MersenneTwister(seed + tid - 1)

        # independent chain per thread
        x = SVector{2,Float64}(0.0, 0.0)
        if isnothing(rng)
            for _ in 1:warmup
                x = maps[sample(map_indices, weights)](x)
            end
            @inbounds for i in lo:hi
                x = maps[sample(map_indices, weights)](x)
                ifs.points[i] = x
            end
        else
            for _ in 1:warmup
                x = maps[sample(rng, map_indices, weights)](x)
            end
            @inbounds for i in lo:hi
                x = maps[sample(rng, map_indices, weights)](x)
                ifs.points[i] = x
            end
        end
    end

    return ifs
end

function iterate!(ifs::IFS;
                  warmup=DEFAULT_WARMUP,
                  seed::Union{Nothing,Integer}=nothing)
    if nthreads() > 1
        return _iterate_parallel!(ifs; warmup=warmup, seed=seed)
    end
    return _iterate_serial!(ifs; warmup=warmup, seed=seed)
end

# Backward-compatible entrypoint. Iteration is now automatically threaded via iterate!.
function iterate_parallel!(ifs::IFS;
                           warmup=DEFAULT_WARMUP,
                           seed::Union{Nothing,Integer}=nothing)
    return iterate!(ifs; warmup=warmup, seed=seed)
end

# --------------------------------
# Deterministic Iteration
# --------------------------------

function _deterministic_expand_points(
    points::Vector{SVector{2,Float64}},
    maps::Vector{AffineMap{Float64}},
    n::Integer
)
    nmaps = length(maps)

    newsize = length(points) * nmaps^n
    @assert newsize ≥ 0 "Integer overflow"
    @assert newsize < 10^8 "Too many points allocated"
    current = points
    for _ in 1:n
        current_len = length(current)
        next = Vector{SVector{2,Float64}}(undef, current_len * nmaps)

        @threads for i in 1:nmaps
            start = (i-1) * current_len + 1
            stop = i * current_len
            @inbounds next[start:stop] = maps[i].(current)
        end

        current = next
    end

    return current
end

function deterministic_iterate(ifs::IFS,
                               n::Integer;
                               warmup::Integer=DEFAULT_WARMUP,
                               seed::Union{Nothing,Integer}=nothing)
    base = IFS(ifs.name, ifs.docs, copy(ifs.points), ifs.maps, ifs.weights, ifs.limits)
    iterate!(base; warmup=warmup, seed=seed)

    result = _deterministic_expand_points(base.points, ifs.maps, n)
    return IFS(ifs.name, ifs.docs, result, ifs.maps, ifs.weights, ifs.limits)
end

# --------------------------------
# Rasterization
# --------------------------------

function make_pixelate_map(limits;
                           resolution=RESOLUTION)

    rows, cols = resolution
    (xlim, ylim) = limits
    xmin, xmax = xlim
    ymin, ymax = ylim

    r = min(rows, cols)
    sx = r / (xmax - xmin)
    sy = r / (ymax - ymin)

    A = SMatrix{2,2,Float64,4}((sx,0.0,
                                0.0,sy))

    b = SVector(
        -xmin*sx + (cols-r)/2 + 0.5,
        -ymin*sy + (rows-r)/2 + 0.5
    )

    return AffineMap(A, b)
end

function _make_image_cpu(ifs::IFS; resolution::Tuple{Int,Int}=RESOLUTION)
    map = make_pixelate_map(ifs.limits; resolution=resolution)
    rows, cols = resolution
    nthreads_local = _thread_buffer_slots()
    buffers = [zeros(Float32, rows, cols) for _ in 1:nthreads_local]

    @threads for idx in eachindex(ifs.points)
        tid = threadid()
        pt = ifs.points[idx]
        pixels = map(pt)
        pixelx = clamp(round(Int, pixels[1]), 1, cols)
        pixely = clamp(round(Int, pixels[2]), 1, rows)

        @inbounds buffers[tid][pixely, pixelx] += 1.0f0
    end

    img = buffers[1]
    for t in 2:nthreads_local
        img .+= buffers[t]
    end
    maxv = maximum(img)
    if maxv > 0f0
        logmax = log1p(maxv)
        @inbounds for i in eachindex(img)
            img[i] = log1p(img[i]) / logmax
        end
    end
    return img
end

function _make_image_gpu(ifs::IFS, ::Val; resolution::Tuple{Int,Int}=RESOLUTION)
    throw(ArgumentError("GPU backend is not available. Install CUDA.jl and ensure a functional CUDA runtime, or use backend=:cpu/:auto."))
end

function _rasterize_image_inversely_gpu(
    ifs::IFS,
    n::Integer,
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}},
    ::Val;
    resolution::Tuple{Int,Int}=RESOLUTION,
    show_divergence_scale::Bool=true,
    mode::Symbol=:exact
)
    throw(ArgumentError("GPU backend is not available. Install CUDA.jl and ensure a functional CUDA runtime, or use backend=:cpu/:auto."))
end

function _iterate_image_gpu(
    ifs::IFS,
    src::AbstractMatrix{Float32};
    colors::Bool=false,
    seed::Union{Nothing,Integer}=nothing
)
    throw(ArgumentError("GPU backend is not available. Install CUDA.jl and ensure a functional CUDA runtime, or use backend=:cpu/:auto."))
end

function make_image(ifs::IFS; resolution::Tuple{Int,Int}=RESOLUTION, backend::Symbol=:cpu)
    if backend == :cpu
        return _make_image_cpu(ifs; resolution=resolution)
    elseif backend == :gpu
        return _make_image_gpu(ifs, Val(:cuda); resolution=resolution)
    elseif backend == :auto
        if _gpu_backend_available(Val(:cuda))
            return _make_image_gpu(ifs, Val(:cuda); resolution=resolution)
        end
        return _make_image_cpu(ifs; resolution=resolution)
    end
    throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))
end

# -------------------------------------------------
# Check if triangle contains origin
# -------------------------------------------------

function isinspace(v::AbstractVector, limits::Tuple{Tuple,Tuple})
    ( xlim, ylim ) = limits
    xmin, xmax = xlim
    ymin, ymax = ylim
    return xmin ≤ v[1] ≤ xmax && ymin ≤ v[2] ≤ ymax 
end

function isinspace(point::SVector{2, Float64}, point1::SVector{2, Float64}, point2::SVector{2, Float64}, limits::Tuple{Tuple,Tuple})
    return isinspace(point, limits) || isinspace(point1, limits) || isinspace(point2, limits)
end

@inline function points_contain_zero(
    points::NTuple{3,SVector{2,Float64}}
)::Bool
    p0, p1, p2 = points

    v1 = p1 - p0
    v2 = p2 - p0
    vO = -p0  

    return (0 ≤ dot(v1, vO) ≤ dot(v1, v1)) &&
           (0 ≤ dot(v2, vO) ≤ dot(v2, v2))
end


# -------------------------------------------------
# Inverse iteration (no comprehension allocations)
# -------------------------------------------------

function inverse_iterate(
    inverse_maps::Vector{AffineMap{Float64}},
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}},
    n::Integer,
    p0::SVector{2,Float64},
    p1::SVector{2,Float64},
    p2::SVector{2,Float64}
)::Float32

    current = Vector{NTuple{3,SVector{2,Float64}}}()
    push!(current, (p0, p1, p2))

    next = Vector{NTuple{3,SVector{2,Float64}}}()

    for j in 1:n

        empty!(next)

        for tri in current
            if !isinspace(tri[1], limits) &&
               !isinspace(tri[2], limits) &&
               !isinspace(tri[3], limits)
                continue
            end

            for imap in inverse_maps
                newtri = (imap(tri[1]),
                          imap(tri[2]),
                          imap(tri[3]))

                push!(next, newtri)
            end
        end

        if isempty(next)
            return Float32(j / n * 0.5)
        end

        if any(points_contain_zero, next)
            return 1.0f0
        end

        current, next = next, current
    end

    return 0.0f0
end

@inline function _inverse_hide_scale_value(v::Float32)::Float32
    # 0.0 means did not diverge, 1.0 means contains-zero; both are white in hide mode.
    return (v == 0.0f0 || v == 1.0f0) ? 1.0f0 : 0.0f0
end

function _inverse_iterate!(
    current::Vector{NTuple{3,SVector{2,Float64}}},
    next::Vector{NTuple{3,SVector{2,Float64}}},
    inverse_maps::Vector{AffineMap{Float64}},
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}},
    n::Integer,
    p0::SVector{2,Float64},
    p1::SVector{2,Float64},
    p2::SVector{2,Float64}
)::Float32
    empty!(current)
    empty!(next)
    push!(current, (p0, p1, p2))

    for j in 1:n
        empty!(next)

        for tri in current
            if !isinspace(tri[1], limits) &&
               !isinspace(tri[2], limits) &&
               !isinspace(tri[3], limits)
                continue
            end

            for imap in inverse_maps
                push!(next, (imap(tri[1]), imap(tri[2]), imap(tri[3])))
            end
        end

        if isempty(next)
            return Float32(j / n * 0.5)
        end

        if any(points_contain_zero, next)
            return 1.0f0
        end

        current, next = next, current
    end

    return 0.0f0
end

function inverse_iterate(
    ifs::IFS,
    n::Integer,
    p0::SVector{2,Float64},
    p1::SVector{2,Float64},
    p2::SVector{2,Float64}
)::Float32

    inverse_maps = inv.(ifs.maps)
    return inverse_iterate(inverse_maps, ifs.limits, n, p0, p1, p2)
end


# -------------------------------------------------
# Inverse rasterization
# -------------------------------------------------

function _rasterize_image_inversely_cpu(
    ifs::IFS,
    n::Integer,
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}};
    resolution::Tuple{Int,Int}=RESOLUTION,
    show_divergence_scale::Bool=true
    )

    pixel_map = make_pixelate_map(limits;
                                  resolution=resolution)
    inverse_maps = inv.(ifs.maps)

    inv_pixel_map = inv(pixel_map)

    rows, cols = resolution
    img = zeros(Float32, rows, cols)
    nthreads_local = _thread_buffer_slots()
    current_buffers = [Vector{NTuple{3,SVector{2,Float64}}}() for _ in 1:nthreads_local]
    next_buffers = [Vector{NTuple{3,SVector{2,Float64}}}() for _ in 1:nthreads_local]

    @threads for x in 1:cols
        tid = threadid()
        current = current_buffers[tid]
        next = next_buffers[tid]
        @inbounds for y in 1:rows
            p0 = inv_pixel_map(SVector{2,Float64}(x,     y))
            p1 = inv_pixel_map(SVector{2,Float64}(x + 1, y))
            p2 = inv_pixel_map(SVector{2,Float64}(x,     y + 1))

            v = _inverse_iterate!(current, next, inverse_maps, ifs.limits, n, p0, p1, p2)
            img[y, x] = show_divergence_scale ? v : _inverse_hide_scale_value(v)
        end
    end

    return img
end

@inline function _inverse_gpu_exact_capacity(nmaps::Int, n::Int)::Int
    if n == 0
        return 1
    end
    cap = 1
    for _ in 1:n
        cap = cap * nmaps
    end
    return max(1, cap)
end

@inline function _inverse_gpu_preview_capacity(nmaps::Int, n::Int)::Int
    base = min(_inverse_gpu_exact_capacity(nmaps, n), 256)
    return max(16, base)
end

@inline function _inverse_gpu_buffer_bytes(cap::Int, rows::Int, cols::Int)::Int
    npix = rows * cols
    # 2 ping-pong buffers * 6 Float64 lanes per triangle.
    return 2 * 6 * sizeof(Float64) * cap * npix
end

function _inverse_gpu_capacity(
    ifs::IFS,
    n::Integer,
    resolution::Tuple{Int,Int},
    mode::Symbol
)::Int
    nmaps = length(ifs.maps)
    nmaps > 0 || throw(ArgumentError("IFS must have at least one map"))
    n >= 0 || throw(ArgumentError("n must be >= 0, got $n"))
    n_int = Int(n)
    if mode == :exact
        return _inverse_gpu_exact_capacity(nmaps, n_int)
    elseif mode == :preview
        return _inverse_gpu_preview_capacity(nmaps, n_int)
    end
    throw(ArgumentError("Invalid mode '$mode'. Supported: :exact, :preview"))
end

function _rasterize_image_inversely_gpu_or_throw(
    ifs::IFS,
    n::Integer,
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}};
    resolution::Tuple{Int,Int}=RESOLUTION,
    show_divergence_scale::Bool=true,
    mode::Symbol=:exact
)
    cap = _inverse_gpu_capacity(ifs, n, resolution, mode)
    rows, cols = resolution
    bytes = _inverse_gpu_buffer_bytes(cap, rows, cols)
    max_bytes = 768 * 1024 * 1024
    if bytes > max_bytes
        throw(ArgumentError("Inverse GPU '$mode' buffers are too large for resolution=$resolution and n=$n (estimated $(round(bytes / 1024^2; digits=1)) MiB > $(round(max_bytes / 1024^2; digits=1)) MiB). Reduce resolution/iterations or use backend=:cpu."))
    end

    return _rasterize_image_inversely_gpu(ifs, n, limits, Val(:cuda);
                                          resolution=resolution,
                                          show_divergence_scale=show_divergence_scale,
                                          mode=mode)
end

function rasterize_image_inversely(
    ifs::IFS,
    n::Integer,
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}};
    resolution::Tuple{Int,Int}=RESOLUTION,
    show_divergence_scale::Bool=true,
    backend::Symbol=:cpu,
    mode::Symbol=:exact
    )
    backend in (:cpu, :gpu, :auto) ||
        throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))
    mode in (:exact, :preview) ||
        throw(ArgumentError("Invalid mode '$mode'. Supported: :exact, :preview"))

    if backend == :cpu
        return _rasterize_image_inversely_cpu(ifs, n, limits;
                                              resolution=resolution,
                                              show_divergence_scale=show_divergence_scale)
    elseif backend == :gpu
        return _rasterize_image_inversely_gpu_or_throw(ifs, n, limits;
                                                       resolution=resolution,
                                                       show_divergence_scale=show_divergence_scale,
                                                       mode=mode)
    elseif _gpu_backend_available(Val(:cuda))
        try
            return _rasterize_image_inversely_gpu_or_throw(ifs, n, limits;
                                                           resolution=resolution,
                                                           show_divergence_scale=show_divergence_scale,
                                                           mode=mode)
        catch err
            if err isa ArgumentError
                return _rasterize_image_inversely_cpu(ifs, n, limits;
                                                      resolution=resolution,
                                                      show_divergence_scale=show_divergence_scale)
            end
            rethrow(err)
        end
    end

    return _rasterize_image_inversely_cpu(ifs, n, limits;
                                          resolution=resolution,
                                          show_divergence_scale=show_divergence_scale)
end

# -------------------------------------------------
# Compose iterate map with pixel map
# -------------------------------------------------

function _make_pixeliterate_map(
    iterate_map::AffineMap{Float64},
    pixel_map::AffineMap{Float64}
)

    inv_pixel = inv(pixel_map)

    A = pixel_map.A * iterate_map.A * inv_pixel.A
    b = pixel_map.A * (iterate_map.A * inv_pixel.b + iterate_map.b) +
        pixel_map.b

    return AffineMap(A, b)
end

function _normalize_rgb_buffers!(
    rimg::AbstractMatrix{Float32},
    gimg::AbstractMatrix{Float32},
    bimg::AbstractMatrix{Float32}
)
    maxv = max(maximum(rimg), maximum(gimg), maximum(bimg))
    if maxv > 0f0
        logmax = log(1f0 + maxv)
        @inbounds for i in eachindex(rimg)
            rimg[i] = log(1f0 + rimg[i]) / logmax
            gimg[i] = log(1f0 + gimg[i]) / logmax
            bimg[i] = log(1f0 + bimg[i]) / logmax
        end
    end
    return nothing
end

function _rgb_image_from_buffers(
    rimg::AbstractMatrix{Float32},
    gimg::AbstractMatrix{Float32},
    bimg::AbstractMatrix{Float32}
)
    rows, cols = size(rimg)
    out = Matrix{RGB{Float32}}(undef, rows, cols)
    @inbounds for y in 1:rows
        for x in 1:cols
            out[y, x] = RGB{Float32}(rimg[y, x], gimg[y, x], bimg[y, x])
        end
    end
    return out
end

function _to_grayscale_matrix(img::AbstractMatrix)
    return Float32.(Gray.(img))
end

function _limits_from_points(points::Vector{SVector{2,Float64}})
    isempty(points) && return _DEFAULT_INITIAL_POLYGON_LIMITS

    xmin = Inf; xmax = -Inf
    ymin = Inf; ymax = -Inf
    @inbounds for p in points
        x, y = p
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
    end

    dx = xmax - xmin
    dy = ymax - ymin
    m = max(dx, dy)
    m = m == 0 ? 1e-9 : m
    pad = 0.05m

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    half = (m + 2pad) / 2

    return ((cx - half, cx + half),
            (cy - half, cy + half))
end

function _resolve_image_source(
    ifs::IFS,
    image_source::Symbol,
    image_path::Union{Nothing,AbstractString},
    resolution::Tuple{Int,Int},
    warmup::Integer,
    deterministic_iters::Integer,
    inverse_iters::Integer,
    polygon_limits_mode::Symbol,
    show_divergence_scale::Bool,
    initial_polygon::Symbol=:default,
    backend::Symbol=:cpu
)
    if image_source == :file
        isnothing(image_path) && throw(ArgumentError("image_path is required when image_source=:file"))
        isfile(image_path) || throw(ArgumentError("image_path '$image_path' does not exist"))
        return _to_grayscale_matrix(load(image_path))
    elseif image_source == :chaos
        tifs = IFS(ifs.name, ifs.docs, copy(ifs.points), ifs.maps, ifs.weights, ifs.limits)
        iterate!(tifs; warmup=warmup)
        return make_image(tifs; resolution=resolution, backend=backend)
    elseif image_source == :point_deterministic
        tifs = deterministic_iterate(ifs, deterministic_iters; warmup=warmup)
        return make_image(tifs; resolution=resolution, backend=backend)
    elseif image_source == :inverse
        return rasterize_image_inversely(ifs, inverse_iters, ifs.limits;
                                         resolution=resolution,
                                         show_divergence_scale=show_divergence_scale,
                                         backend=backend)
    elseif image_source == :polygon
        limits_mode = if polygon_limits_mode == :ifs
            :ifs
        elseif polygon_limits_mode == :default
            @warn "polygon_limits_mode=:default is treated as :ifs for image_source=:polygon."
            :ifs
        else
            throw(ArgumentError("Invalid polygon_limits_mode '$polygon_limits_mode'. Supported: :ifs, :default"))
        end
        p = joinpath(DEFAULT_MEDIA_DIR, "polygon_seed_$(randstring(12)).png")
        try
            render_transformations_png(ifs;
                                       outpath=p,
                                       width=resolution[2],
                                       height=resolution[1],
                                       show_base=false,
                                       initial_polygon=initial_polygon,
                                       limits_mode=limits_mode,
                                       color=false)
            return _to_grayscale_matrix(load(p))
        finally
            rm(p; force=true)
        end
    end

    throw(ArgumentError("Invalid image_source '$image_source'. Supported: :polygon, :chaos, :point_deterministic, :inverse, :file"))
end

function _iterate_image_cpu(
    ifs::IFS,
    src::AbstractMatrix{Float32};
    colors::Bool=false,
    seed::Union{Nothing,Integer}=nothing
)
    rows, cols = size(src)

    nthreads_local = _thread_buffer_slots()
    buffers = [zeros(Float32, rows, cols) for _ in 1:nthreads_local]
    rbuffers = colors ? [zeros(Float32, rows, cols) for _ in 1:nthreads_local] : nothing
    gbuffers = colors ? [zeros(Float32, rows, cols) for _ in 1:nthreads_local] : nothing
    bbuffers = colors ? [zeros(Float32, rows, cols) for _ in 1:nthreads_local] : nothing
    map_colors = colors ? _map_colors(length(ifs.maps)) : RGB{Float32}[]

    pmap = make_pixelate_map(ifs.limits; resolution=(rows, cols))

    for (map_idx, nmap) in enumerate(ifs.maps)
        cmap = _make_pixeliterate_map(nmap, pmap)  # use it!
        map_color = colors ? _map_color_rgb(map_idx, map_colors) : RGB{Float32}(0.0f0, 0.0f0, 0.0f0)
        cr = map_color.r
        cg = map_color.g
        cb = map_color.b

        @threads for x in 1:cols
            tid = threadid()
            graybuf = buffers[tid]
            rbuf = colors ? rbuffers[tid] : graybuf
            gbuf = colors ? gbuffers[tid] : graybuf
            bbuf = colors ? bbuffers[tid] : graybuf
            @inbounds for y in 1:rows
                val = src[y, x]
                if val != 0
                    fx, fy = cmap(SVector(x, y))

                    newx = round(Int, fx)
                    newy = round(Int, fy)
                    if !(1 <= newx <= cols && 1 <= newy <= rows)
                        continue
                    end

                    if colors
                        fval = Float32(val)
                        rbuf[newy, newx] += fval * cr
                        gbuf[newy, newx] += fval * cg
                        bbuf[newy, newx] += fval * cb
                    else
                        graybuf[newy, newx] += val
                    end
                end
            end
        end
    end

    if !colors
        newimg = buffers[1]
        for t in 2:nthreads_local
            newimg .+= buffers[t]
        end

        maxv = maximum(newimg)
        if maxv > 0f0
            logmax = log(1f0 + maxv)
            @inbounds for i in eachindex(newimg)
                newimg[i] = log(1f0 + newimg[i]) / logmax
            end
        end

        return Gray.(newimg)
    end

    rimg = rbuffers[1]
    gimg = gbuffers[1]
    bimg = bbuffers[1]
    for t in 2:nthreads_local
        rimg .+= rbuffers[t]
        gimg .+= gbuffers[t]
        bimg .+= bbuffers[t]
    end

    _normalize_rgb_buffers!(rimg, gimg, bimg)
    return _rgb_image_from_buffers(rimg, gimg, bimg)
end

function iterate_image(
    ifs::IFS,
    img::AbstractMatrix;
    colors::Bool=false,
    seed::Union{Nothing,Integer}=nothing,
    backend::Symbol=:cpu
)
    src = _to_grayscale_matrix(img)
    if backend == :cpu
        return _iterate_image_cpu(ifs, src; colors=colors, seed=seed)
    elseif backend == :gpu
        return _iterate_image_gpu(ifs, src; colors=colors, seed=seed)
    elseif backend == :auto
        if _gpu_backend_available(Val(:cuda))
            try
                return _iterate_image_gpu(ifs, src; colors=colors, seed=seed)
            catch err
                if err isa ArgumentError
                    return _iterate_image_cpu(ifs, src; colors=colors, seed=seed)
                end
                rethrow(err)
            end
        end
        return _iterate_image_cpu(ifs, src; colors=colors, seed=seed)
    end

    throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))
end

function _iterate_image_single_map(ifs::IFS, img::AbstractMatrix{<:Real}, map_index::Integer)
    1 <= map_index <= length(ifs.maps) || throw(ArgumentError("map_index $map_index is out of range. Valid range: 1-$(length(ifs.maps))"))
    rows, cols = size(img)
    newimg = zeros(Float32, rows, cols)

    pmap = make_pixelate_map(ifs.limits; resolution=(rows, cols))
    nmap = ifs.maps[map_index]
    cmap = _make_pixeliterate_map(nmap, pmap)

    @inbounds for x in 1:cols
        for y in 1:rows
            val = img[y, x]
            if val != 0
                fx, fy = cmap(SVector(x, y))
                newx = round(Int, fx)
                newy = round(Int, fy)
                if !(1 <= newx <= cols && 1 <= newy <= rows)
                    continue
                end
                newimg[newy, newx] += val
            end
        end
    end

    maxv = maximum(newimg)
    if maxv > 0f0
        logmax = log(1f0 + maxv)
        @inbounds for i in eachindex(newimg)
            newimg[i] = log(1f0 + newimg[i]) / logmax
        end
    end

    return Gray.(newimg)
end

# --------------------------------
# High-level render entrypoint
# --------------------------------

function _validate_render_options(
    method::RenderMethod,
    npoints::Union{Nothing,Integer},
    warmup::Integer,
    color::Bool,
    resolution::Tuple{Int,Int},
    ifs_index::Union{Nothing,Integer},
    ifs_name::Union{Nothing,AbstractString},
    image_source::Symbol,
    image_path::Union{Nothing,AbstractString},
    image_iterations::Integer,
    polygon_limits_mode::Symbol,
    backend::Symbol,
    initial_polygon::Symbol,
    deterministic_depth::Integer,
    inverse_depth::Integer
)
    isnothing(npoints) || npoints > 0 || throw(ArgumentError("npoints must be > 0, got $npoints"))
    warmup >= 0 || throw(ArgumentError("warmup must be >= 0, got $warmup"))
    resolution[1] > 0 && resolution[2] > 0 || throw(ArgumentError("resolution must be positive, got $resolution"))
    isnothing(ifs_index) || ifs_index > 0 || throw(ArgumentError("ifs_index must be >= 1, got $ifs_index"))
    isnothing(ifs_name) || !isempty(strip(ifs_name)) || throw(ArgumentError("ifs_name must be non-empty when provided"))
    image_iterations > 0 || throw(ArgumentError("image_iterations must be > 0, got $image_iterations"))
    deterministic_depth >= 0 || throw(ArgumentError("deterministic_depth must be >= 0, got $deterministic_depth"))
    inverse_depth >= 0 || throw(ArgumentError("inverse_depth must be >= 0, got $inverse_depth"))
    image_source in (:polygon, :chaos, :point_deterministic, :inverse, :file) ||
        throw(ArgumentError("Invalid image_source '$image_source'. Supported: :polygon, :chaos, :point_deterministic, :inverse, :file"))
    polygon_limits_mode in (:ifs, :default) ||
        throw(ArgumentError("Invalid polygon_limits_mode '$polygon_limits_mode'. Supported: :ifs, :default"))
    backend in (:cpu, :gpu, :auto) ||
        throw(ArgumentError("Invalid backend '$backend'. Supported: :cpu, :gpu, :auto"))
    _resolve_initial_polygon(initial_polygon)
    image_source == :file && isnothing(image_path) &&
        throw(ArgumentError("image_path is required when image_source=:file"))
    method == ImageIterate && color &&
        throw(ArgumentError("color=true is not supported for method=ImageIterate. ImageIterate is grayscale-only."))
    return nothing
end

function _resolve_render_input(
    input::IFS;
    npoints::Union{Nothing,Integer},
    ifs_index::Union{Nothing,Integer},
    ifs_name::Union{Nothing,AbstractString}
)
    if !isnothing(ifs_index) || !isnothing(ifs_name)
        throw(ArgumentError("ifs_index/ifs_name are only valid for parsed string/file inputs"))
    end

    if isnothing(npoints) || length(input.points) == npoints
        return input
    end

    return IFS(input.maps, input.weights;
               npoints=npoints,
               name=input.name,
               docs=input.docs,
               limits=input.limits)
end

function _resolve_render_input(
    input::AbstractMatrix{<:Real};
    npoints::Union{Nothing,Integer},
    ifs_index::Union{Nothing,Integer},
    ifs_name::Union{Nothing,AbstractString}
)
    if !isnothing(ifs_index) || !isnothing(ifs_name)
        throw(ArgumentError("ifs_index/ifs_name are only valid for parsed string/file inputs"))
    end
    resolved_npoints = isnothing(npoints) ? DEFAULT_SAMPLES : npoints
    return IFS(input; npoints=resolved_npoints)
end

function _ifs_choices_text(defs::AbstractVector)
    lines = ["Available IFS definitions:"]
    for (i, d) in enumerate(defs)
        push!(lines, "  [$i] $(d.name)")
    end
    return join(lines, "\n")
end

function _select_ifs_definition(
    defs::AbstractVector;
    ifs_index::Union{Nothing,Integer},
    ifs_name::Union{Nothing,AbstractString}
)
    choices = _ifs_choices_text(defs)

    if !isnothing(ifs_index) && !isnothing(ifs_name)
        throw(ArgumentError("Provide only one of ifs_index or ifs_name.\n$choices"))
    end

    if !isnothing(ifs_name)
        for d in defs
            if d.name == ifs_name
                return d
            end
        end
        throw(ArgumentError("IFS name '$ifs_name' not found.\n$choices"))
    end

    if !isnothing(ifs_index)
        if ifs_index <= length(defs)
            return defs[ifs_index]
        end
        throw(ArgumentError("ifs_index=$ifs_index is out of range (1-$(length(defs))).\n$choices"))
    end

    if length(defs) == 1
        return defs[1]
    end

    println(choices)
    print("No ifs_index/ifs_name provided. Render [1] $(defs[1].name)? [y/N]: ")
    answer = try
        lowercase(strip(readline()))
    catch
        ""
    end

    if answer in ("y", "yes")
        return defs[1]
    end

    throw(ArgumentError("No IFS selection confirmed.\n$choices"))
end

function _resolve_render_input(
    input::AbstractString;
    npoints::Union{Nothing,Integer},
    ifs_index::Union{Nothing,Integer},
    ifs_name::Union{Nothing,AbstractString}
)
    resolved_npoints = isnothing(npoints) ? DEFAULT_SAMPLES : npoints
    defs = isfile(input) ? parse_ifs_definitions_file(input) :
                           parse_ifs_definitions_string(input)

    isempty(defs) && throw(ArgumentError("No IFS definitions found in input"))
    selected = _select_ifs_definition(defs; ifs_index=ifs_index, ifs_name=ifs_name)
    return IFS(selected.eq; npoints=resolved_npoints, name=selected.name, docs=selected.docs)
end

function _resolve_render_iterations(
    iterations::Union{Nothing,Integer},
    deterministic_depth::Integer,
    inverse_depth::Integer
)
    if isnothing(iterations)
        if deterministic_depth != 1 || inverse_depth != 8
            @warn "deterministic_depth/inverse_depth are compatibility aliases; prefer iterations=..."
        end
        deterministic_iters = deterministic_depth
        inverse_iters = inverse_depth
    else
        if deterministic_depth != 1 || inverse_depth != 8
            @warn "iterations takes precedence over deterministic_depth/inverse_depth"
        end
        deterministic_iters = iterations
        inverse_iters = iterations
    end

    deterministic_iters >= 0 || throw(ArgumentError("iterations must be >= 0, got $deterministic_iters"))
    inverse_iters >= 0 || throw(ArgumentError("iterations must be >= 0, got $inverse_iters"))
    return deterministic_iters, inverse_iters
end

function render(
    input;
    method::Union{RenderMethod,Symbol,AbstractString}=Chaos,
    npoints::Union{Nothing,Integer}=nothing,
    warmup::Integer=DEFAULT_WARMUP,
    color::Bool=false,
    show_divergence_scale::Bool=true,
    image_source::Symbol=:polygon,
    image_path::Union{Nothing,AbstractString}=nothing,
    image_iterations::Integer=1,
    polygon_limits_mode::Symbol=:ifs,
    backend::Symbol=:cpu,
    initial_polygon::Symbol=:default,
    resolution::Tuple{Int,Int}=RESOLUTION,
    outpath::AbstractString="media/render.png",
    ifs_index::Union{Nothing,Integer}=nothing,
    ifs_name::Union{Nothing,AbstractString}=nothing,
    iterations::Union{Nothing,Integer}=nothing,
    deterministic_depth::Integer=1,
    inverse_depth::Integer=8,
)
    parsed_method = _parse_render_method(method)
    _validate_render_options(parsed_method, npoints, warmup, color, resolution, ifs_index, ifs_name, image_source, image_path, image_iterations, polygon_limits_mode, backend, initial_polygon, deterministic_depth, inverse_depth)

    ifs = _resolve_render_input(input; npoints=npoints, ifs_index=ifs_index, ifs_name=ifs_name)

    render_method = parsed_method == Parallel ? Chaos : parsed_method
    rendered_ifs = ifs
    deterministic_iters, inverse_iters =
        _resolve_render_iterations(iterations, deterministic_depth, inverse_depth)

    if render_method == Chaos
        iterate!(rendered_ifs; warmup=warmup)
        img = make_image(rendered_ifs; resolution=resolution, backend=backend)
    elseif render_method == PointDeterministic
        rendered_ifs = deterministic_iterate(rendered_ifs, deterministic_iters; warmup=warmup)
        img = make_image(rendered_ifs; resolution=resolution, backend=backend)
    elseif render_method == Inverse
        img = rasterize_image_inversely(rendered_ifs, inverse_iters, rendered_ifs.limits;
                                        resolution=resolution,
                                        show_divergence_scale=show_divergence_scale,
                                        backend=backend)
    else
        img = _resolve_image_source(rendered_ifs,
                                    image_source,
                                    image_path,
                                    resolution,
                                    warmup,
                                    deterministic_iters,
                                    inverse_iters,
                                    polygon_limits_mode,
                                    show_divergence_scale,
                                    initial_polygon,
                                    backend)
        for _ in 1:image_iterations
            img = iterate_image(rendered_ifs, img; colors=false, backend=backend)
        end
    end

    if color && render_method != ImageIterate
        img = iterate_image(rendered_ifs, img; colors=true, backend=backend)
    end

    final_outpath = _normalize_media_outpath(outpath)
    save(final_outpath, img)

    return (ifs=rendered_ifs,
            image=img,
            outpath=final_outpath,
            method=_render_method_symbol(render_method))
end

# --------------------------------
# Example Systems
# --------------------------------

const HEIGHWAY_DRAGON = [
     0.5  -0.5   0.5   0.5   0.0   0.0;
    -0.5  -0.5   0.5  -0.5   1.0   0.0
]

const EISENSTEIN = [
-0.5 0.0 0.0 -0.5  0.0   0.0  0.25;
-0.5 0.0 0.0 -0.5 -0.5   0.0  0.25;
-0.5 0.0 0.0 -0.5  0.25 -0.433 0.25;
-0.5 0.0 0.0 -0.5  0.25  0.433 0.25;
]

# --------------------------------
# Main
# --------------------------------

function main(; eq=EISENSTEIN,
               npoints=1_000_000,
               outpath="media/output.png")

    println("Building IFS...")
    ifs = IFS(eq; npoints=npoints)

    println("Running chaos game...")
    iterate!(ifs)

    println("Rasterizing...")
    img = make_image(ifs)

    final_outpath = _normalize_media_outpath(outpath)
    println("Saving to $final_outpath")
    save(final_outpath, img)

    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
