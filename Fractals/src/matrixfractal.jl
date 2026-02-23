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

function build_maps_and_weights(eq::AbstractMatrix{<:Real})
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

function get_limits(maps, weights;
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

    maps, weights = build_maps_and_weights(eq)
    limits = get_limits(maps, weights)

    points = [SVector{2,Float64}(0.0,0.0)
              for _ in 1:npoints]

    return IFS(String(name), String(docs), points, maps, weights, limits)
end

function IFS(maps::Vector{AffineMap{Float64}},
             weights::Weights;
             npoints::Integer=DEFAULT_SAMPLES,
             name::AbstractString="",
             docs::AbstractString="",
             limits=get_limits(maps, weights))
    points = [SVector{2,Float64}(0.0,0.0)
              for _ in 1:npoints]
    return IFS(String(name), String(docs), points, maps, weights, limits)
end

# --------------------------------
# Chaos Game Iteration
# --------------------------------

function iterate!(ifs::IFS;
                  warmup=DEFAULT_WARMUP)

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

function iterate_parallel!(ifs::IFS; warmup=DEFAULT_WARMUP)
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

function make_pixeliterate_map(
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
        cmap = make_pixeliterate_map(nmap, pmap)  # use it!

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

function iterate_image_single_map(ifs::IFS, img::AbstractMatrix{<:Real}, map_index::Integer)
    rows, cols = size(img)
    newimg = zeros(Float32, rows, cols)

    pmap = make_pixelate_map(ifs.limits; resolution=(rows, cols))
    nmap = ifs.maps[map_index]
    cmap = make_pixeliterate_map(nmap, pmap)

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
               outpath="output.png")

    println("Building IFS...")
    ifs = IFS(eq; npoints=npoints)

    println("Running chaos game...")
    iterate!(ifs)

    println("Rasterizing...")
    img = make_image(ifs)

    println("Saving to $outpath")
    save(outpath, img)

    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
