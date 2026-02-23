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
using Base.Threads
using Printf

# --------------------------------
# Configuration
# --------------------------------

const RESOLUTION = (1504, 2256)
const DEFAULT_WARMUP = 50
const DEFAULT_SAMPLES = 1_000_000
const DEFAULT_MEDIA_DIR = "media"
const DEFAULT_BASE_LIMITS = ((0.0, 1.0), (0.0, 1.0))

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

_base_limits_image() = DEFAULT_BASE_LIMITS

function _base_l_image()
    # Normalized to viewBox [0, 1] x [0, 1]
    return [
        (SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)),
        (SVector{2,Float64}(1.0, 0.0), SVector{2,Float64}(1.0, 1.0)),
        (SVector{2,Float64}(1.0, 1.0), SVector{2,Float64}(0.0, 1.0)),
        (SVector{2,Float64}(0.0, 1.0), SVector{2,Float64}(0.0, 0.0)),
        (SVector{2,Float64}(1/6, 1/6), SVector{2,Float64}(1/6, 5/6)),
        (SVector{2,Float64}(1/6, 5/6), SVector{2,Float64}(5/9, 5/6)),
    ]
end

function _map_colors(n::Integer)
    n <= 0 && return RGB{Float32}[]
    # Generate visually distinct, deterministic colors and avoid white/black background tones.
    palette = distinguishable_colors(n, [RGB(1, 1, 1), RGB(0, 0, 0)])
    return [RGB{Float32}(Float32(c.r), Float32(c.g), Float32(c.b)) for c in palette]
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

function _collect_transformed_base_segments(ifs; show_base::Bool=false)
    base_segments = _base_l_image()
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
    stroke_width::Real=2.0
)
    base_segments, transformed_by_map, all_segments =
        _collect_transformed_base_segments(ifs; show_base=show_base)
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

function render_transformations_png(
    ifs;
    outpath::AbstractString="media/affine_maps.png",
    width::Int=1200,
    height::Int=1200,
    show_base::Bool=false,
)
    base_segments, transformed_by_map, all_segments =
        _collect_transformed_base_segments(ifs; show_base=show_base)
    colors = _map_colors(length(transformed_by_map))

    sx, sy, ox, oy = _fit_bounds(all_segments, width, height)
    img = fill(RGBA{Float32}(0.0f0, 0.0f0, 0.0f0, 0.0f0), height, width)

    if show_base
        base_color = RGBA{Float32}(0.2f0, 0.2f0, 0.2f0, 1.0f0)
        for (p1, p2) in base_segments
            x1, y1 = _to_svg_xy(p1, sx, sy, ox, oy)
            x2, y2 = _to_svg_xy(p2, sx, sy, ox, oy)
            _draw_line!(img, x1, y1, x2, y2, base_color)
        end
    end

    for (i, transformed) in enumerate(transformed_by_map)
        c = _map_color_rgb(i, colors)
        color = RGBA{Float32}(c.r, c.g, c.b, 1.0f0)
        for (p1, p2) in transformed
            x1, y1 = _to_svg_xy(p1, sx, sy, ox, oy)
            x2, y2 = _to_svg_xy(p2, sx, sy, ox, oy)
            _draw_line!(img, x1, y1, x2, y2, color)
        end
    end

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

function _iterate_serial!(ifs::IFS; warmup=DEFAULT_WARMUP)

    maps = ifs.maps
    weights = ifs.weights
    map_indices = Base.OneTo(length(maps))

    x = SVector{2,Float64}(0.0, 0.0)

    for _ in 1:warmup
        x = maps[sample(map_indices, weights)](x)
    end

    idxs = sample(map_indices, weights, length(ifs.points))

    for i in eachindex(ifs.points)
        x = maps[idxs[i]](x)
        ifs.points[i] = x
    end

    return ifs
end

function _iterate_parallel!(ifs::IFS; warmup=DEFAULT_WARMUP)
    maps = ifs.maps
    alias = StatsBase.AliasTable(ifs.weights)  # fast discrete sampling
    npts = length(ifs.points)

    @threads for tid in 1:nthreads()
        # thread-local RNG (deterministic per thread if you want reproducibility)
        # rng = MersenneTwister(seed + UInt(tid))

        # chunk for this thread
        lo = fld((tid-1)*npts, nthreads()) + 1
        hi = fld(tid*npts, nthreads())
        lo > hi && continue

        # independent chain per thread
        x = SVector{2,Float64}(0.0, 0.0)
        for _ in 1:warmup
            x = maps[rand(alias)](x)
        end

        @inbounds for i in lo:hi
            x = maps[rand(alias)](x)
            ifs.points[i] = x
        end
    end

    return ifs
end

function iterate!(ifs::IFS; warmup=DEFAULT_WARMUP)
    if nthreads() > 1
        return _iterate_parallel!(ifs; warmup=warmup)
    end
    return _iterate_serial!(ifs; warmup=warmup)
end

# Backward-compatible entrypoint. Iteration is now automatically threaded via iterate!.
function iterate_parallel!(ifs::IFS; warmup=DEFAULT_WARMUP)
    return iterate!(ifs; warmup=warmup)
end

# --------------------------------
# Deterministic Iteration
# --------------------------------

function deterministic_iterate(ifs::IFS, n::Integer)

    maps = ifs.maps
    points = ifs.points
    nmaps = length(maps)

    newsize = length(points) * nmaps^n
    @assert newsize ≥ 0 "Integer overflow"
    @assert newsize < 10^8 "Too many points allocated"

    result = Vector{SVector{2,Float64}}(undef, newsize)
    result[1:length(points)] = points

    current_len = length(points)

    for _ in 1:n
        base = result[1:current_len]

        @threads for i in 1:nmaps
            start = (i-1)*current_len + 1
            stop  = i*current_len
            @inbounds result[start:stop] =
                maps[i].(base)
        end

        current_len *= nmaps
    end

    return IFS(ifs.name, ifs.docs, result, maps, ifs.weights, ifs.limits)
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

function make_image(ifs::IFS; resolution::Tuple{Int,Int}=RESOLUTION)
    map = make_pixelate_map(ifs.limits; resolution=resolution)
    rows, cols = resolution
    nthreads_local = nthreads()
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
        img .= log.(1 .+ img) ./ log(1 .+ maxv)
    end
    return img
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
    ifs::IFS,
    n::Integer,
    p0::SVector{2,Float64},
    p1::SVector{2,Float64},
    p2::SVector{2,Float64}
)::Float32

    inverse_maps = inv.(ifs.maps)

    current = Vector{NTuple{3,SVector{2,Float64}}}()
    push!(current, (p0, p1, p2))

    next = Vector{NTuple{3,SVector{2,Float64}}}()

    for j in 1:n

        empty!(next)

        for tri in current
            if !isinspace(tri[1], ifs.limits) &&
               !isinspace(tri[2], ifs.limits) &&
               !isinspace(tri[3], ifs.limits)
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


# -------------------------------------------------
# Inverse rasterization
# -------------------------------------------------

function rasterize_image_inversely(
    ifs::IFS,
    n::Integer,
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}};
    resolution::Tuple{Int,Int}=RESOLUTION
    )

    pixel_map = make_pixelate_map(limits;
                                  resolution=resolution)

    inv_pixel_map = inv(pixel_map)

    rows, cols = resolution
    img = zeros(Float32, rows, cols)

    @threads for x in 1:cols
        @inbounds for y in 1:rows
            p0 = inv_pixel_map(SVector{2,Float64}(x,     y))
            p1 = inv_pixel_map(SVector{2,Float64}(x + 1, y))
            p2 = inv_pixel_map(SVector{2,Float64}(x,     y + 1))

            img[y, x] = inverse_iterate(ifs, n, p0, p1, p2)
        end
    end

    return img
end


# -------------------------------------------------
# Pixelate map (fully concrete matrix construction)
# -------------------------------------------------

function make_pixelate_map(
    limits::Tuple{Tuple{Float64,Float64},
                  Tuple{Float64,Float64}};
    resolution::Tuple{Int,Int}=RESOLUTION
    )

    rows, cols = resolution

    (xlim, ylim) = limits
    xmin, xmax = xlim
    ymin, ymax = ylim

    r = min(rows, cols)
    sx = r / (xmax - xmin)
    sy = r / (ymax - ymin)

    A = SMatrix{2,2,Float64,4}((sx, 0.0,
                                0.0, sy))

    b = SVector{2,Float64}(
        -xmin*sx + (cols-r)/2 + 0.5,
        -ymin*sy + (rows-r)/2 + 0.5
    )

    return AffineMap(A, b)
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

function iterate_image(ifs::IFS, img::AbstractMatrix{<:Real})
    rows, cols = size(img)

    nthreads_local = nthreads()
    buffers = [zeros(Float32, rows, cols) for _ in 1:nthreads_local]

    pmap = make_pixelate_map(ifs.limits; resolution=(rows, cols))
    invpmap = inv(pmap)  # compute once

    for nmap in ifs.maps
        cmap = _make_pixeliterate_map(nmap, pmap)  # use it!

        @threads for x in 1:cols
            tid = threadid()
            @inbounds for y in 1:rows
                val = img[y, x]
                if val != 0
                    fx, fy = cmap(SVector(x, y))

                    newx = clamp(round(Int, fx), 1, cols)
                    newy = clamp(round(Int, fy), 1, rows)

                    buffers[tid][newy, newx] += val
                end
            end
        end
    end
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

function _iterate_image_single_map(ifs::IFS, img::AbstractMatrix{<:Real}, map_index::Integer)
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
                newx = clamp(round(Int, fx), 1, cols)
                newy = clamp(round(Int, fy), 1, rows)
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
    method::Symbol,
    npoints::Integer,
    warmup::Integer,
    resolution::Tuple{Int,Int},
    ifs_index::Integer,
    deterministic_depth::Integer,
    inverse_depth::Integer
)
    supported = (:chaos, :parallel, :deterministic, :inverse)
    method in supported || throw(ArgumentError("Invalid method '$method'. Supported methods: $(collect(supported))"))
    npoints > 0 || throw(ArgumentError("npoints must be > 0, got $npoints"))
    warmup >= 0 || throw(ArgumentError("warmup must be >= 0, got $warmup"))
    resolution[1] > 0 && resolution[2] > 0 || throw(ArgumentError("resolution must be positive, got $resolution"))
    ifs_index > 0 || throw(ArgumentError("ifs_index must be >= 1, got $ifs_index"))
    deterministic_depth >= 0 || throw(ArgumentError("deterministic_depth must be >= 0, got $deterministic_depth"))
    inverse_depth >= 0 || throw(ArgumentError("inverse_depth must be >= 0, got $inverse_depth"))
    return nothing
end

function _resolve_render_input(
    input::IFS;
    npoints::Integer,
    ifs_index::Integer
)
    if ifs_index != 1
        throw(ArgumentError("ifs_index is only valid for parsed string/file inputs"))
    end

    if length(input.points) == npoints
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
    npoints::Integer,
    ifs_index::Integer
)
    if ifs_index != 1
        throw(ArgumentError("ifs_index is only valid for parsed string/file inputs"))
    end
    return IFS(input; npoints=npoints)
end

function _resolve_render_input(
    input::AbstractString;
    npoints::Integer,
    ifs_index::Integer
)
    defs = isfile(input) ? parse_ifs_file(input; npoints=npoints) :
                           parse_ifs_string(input; npoints=npoints)

    isempty(defs) && throw(ArgumentError("No IFS definitions found in input"))
    ifs_index <= length(defs) || throw(ArgumentError("ifs_index=$ifs_index is out of range (1-$(length(defs)))"))
    return defs[ifs_index]
end

function render(
    input;
    method::Symbol=:chaos,
    npoints::Integer=DEFAULT_SAMPLES,
    warmup::Integer=DEFAULT_WARMUP,
    resolution::Tuple{Int,Int}=RESOLUTION,
    outpath::AbstractString="media/render.png",
    ifs_index::Integer=1,
    deterministic_depth::Integer=1,
    inverse_depth::Integer=8,
)
    _validate_render_options(method, npoints, warmup, resolution, ifs_index, deterministic_depth, inverse_depth)

    ifs = _resolve_render_input(input; npoints=npoints, ifs_index=ifs_index)

    render_method = method == :parallel ? :chaos : method
    rendered_ifs = ifs

    if render_method == :chaos
        iterate!(rendered_ifs; warmup=warmup)
        img = make_image(rendered_ifs; resolution=resolution)
    elseif render_method == :deterministic
        rendered_ifs = deterministic_iterate(rendered_ifs, deterministic_depth)
        img = make_image(rendered_ifs; resolution=resolution)
    else
        img = rasterize_image_inversely(rendered_ifs, inverse_depth, rendered_ifs.limits; resolution=resolution)
    end

    final_outpath = _normalize_media_outpath(outpath)
    save(final_outpath, img)

    return (ifs=rendered_ifs, image=img, outpath=final_outpath, method=render_method)
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
