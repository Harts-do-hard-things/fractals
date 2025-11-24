# matrixfractal.jl
using StaticArrays
using LinearAlgebra
using StatsBase
using Images
using FileIO
using Colors
using BenchmarkTools

const RESOLUTION = (1504, 2256)
const DEFAULT_WARMUP = 50
const DEFAULT_SAMPLES = 1_000_000


struct AffineMap{T<:AbstractFloat}
    A::SMatrix{2,2,T}
    b::SVector{2,T}
end

AffineMap(A::AbstractMatrix, b::AbstractVector) = AffineMap(SMatrix{2,2,Float64}(A), SVector{2,Float64}(b))
AffineMap(a11::Float64,a12::Float64,a21::Float64,a22::Float64,b1::Float64,b2::Float64) =
    AffineMap(SMatrix{2,2,Float64}((a11,a12,a21,a22)), SVector{2,Float64}(b1,b2))

# make it callable 
(m::AffineMap)(x::SVector{2,Float64}) = m.A * x + m.b

struct Fractal{P}
    points::Vector{P}
    maps::Vector{AffineMap{Float64}}
    weights::Weights
    limits::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}
end

function build_maps_and_weights(eq::AbstractMatrix{<:Real})
    n = size(eq,1)
    maps = Vector{AffineMap{Float64}}(undef, n)
    probs_present = size(eq,2) == 7
    p = Vector{Float64}(undef, n)

    for i in 1:n
        maps[i] = AffineMap((Float64.(eq[i,1:6])...))

        if probs_present
            p[i] = float(eq[i,7])
        else
            # fallback: use absolute determinant as a proxy for area contraction
            p[i] = abs(det(SMatrix{2,2,Float64}((Float64.(eq[i,1:4])...))))
        end
    end
    return maps, Weights(p)
end

function Fractal(eq::AbstractMatrix{<:Real}; npoints::Integer=DEFAULT_SAMPLES)
    maps, weights = build_maps_and_weights(eq)
    # default initial point: origin as SVector{2,Float64}
    init_points = [SVector{2,Float64}(0.0, 0.0) for _ in 1:npoints]
    limits = get_limits(maps, weights)  # compute limits before using
    return Fractal{SVector{2,Float64}}(init_points, maps, weights, limits)
end

function Fractal(f::Fractal{T}; npoints::Integer=DEFAULT_SAMPLES) where T
    init_points = [SVector{2,Float64}(0.0, 0.0) for _ in 1:npoints]
    return Fractal{T}(init_points, f.maps, f.weights, f.limits)
end

function Fractal(maps::Vector{AffineMap{Float64}}, weights::Weights; npoints::Integer=DEFAULT_SAMPLES)
    limits = get_limits(maps, weights)
    pts = [SVector{2,Float64}(0.0, 0.0) for _ in 1:npoints]
    return Fractal{SVector{2,Float64}}(pts, maps, weights, limits)
end

function get_limits(maps::Vector{AffineMap{Float64}}, weights::Weights; warmup::Int=DEFAULT_WARMUP, n::Int=10_000)::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}
    x = SVector{2,Float64}(0.0, 0.0)

    for _ in 1:warmup
        idx = sample(1:length(maps), weights)
        x = maps[idx](x)
    end

    xmin = Inf; xmax = -Inf; ymin = Inf; ymax = -Inf
    for _ in 1:n
        idx = sample(1:length(maps), weights)
        x = maps[idx](x)
        xx, yy = x[1], x[2]
        if xx < xmin; xmin = xx; end
        if xx > xmax; xmax = xx; end
        if yy < ymin; ymin = yy; end
        if yy > ymax; ymax = yy; end
    end

    dx = xmax - xmin
    dy = ymax - ymin
    md = max(dx, dy)
    # add small margin
    pad = md * 0.05
    cx = 0.5*(xmin + xmax)
    cy = 0.5*(ymin + ymax)
    xhalf = 0.5*(md + 2*pad)
    yhalf = xhalf
    return ((cx - xhalf, cx + xhalf), (cy - yhalf, cy + yhalf))
end

function iterate!(ifs::Fractal{SVector{2,Float64}}; warmup::Int=DEFAULT_WARMUP)
    maps = ifs.maps
    weights = ifs.weights
    N = length(ifs.points)

    # warmup
    x = SVector{2,Float64}(0.0, 0.0)
    for _ in 1:warmup
        idx = sample(1:length(maps), weights)
        x = maps[idx](x)
    end

    # sample indices in bulk (fast)
    idxs = sample(1:length(maps), weights, N)
    # apply maps
    for i in 1:N
        x = maps[idxs[i]](x)
        ifs.points[i] = x
    end
    return ifs
end

function iterate(ifs::Fractal{SVector{2,Float64}}, npoints::Integer; warmup::Int=DEFAULT_WARMUP)
    maps = ifs.maps
    weights = ifs.weights
    points = Vector{SVector{2,Float64}}(undef, npoints)
    # warmup
    x = SVector{2,Float64}(0.0, 0.0)
    for _ in 1:warmup
        idx = sample(1:length(maps), weights)
        x = maps[idx](x)
    end

    # sample indices in bulk (fast)
    idxs = sample(1:length(maps), weights, npoints)
    # apply maps
    for i in 1:npoints
        x = maps[idxs[i]](x)
        points[i] = x
    end
    return Fractal{SVector{2,Float64}}(points, maps, weights, ifs.limits)
end

function make_image(ifs::Fractal{SVector{2,Float64}}; resolution::Tuple{Int,Int}=RESOLUTION)
    rows, cols = resolution
    img = zeros(Float32, rows, cols)

    ( xlim, ylim ) = ifs.limits
    xmin, xmax = xlim
    ymin, ymax = ylim

    r = minimum(RESOLUTION)
    xrange = 1 / (xmax - xmin) * r
    yrange = 1 / (ymax - ymin) * r

    for pt in ifs.points
        x = pt[1]; y = pt[2]

        pixelx = clamp(floor(Int, (x - xmin) * xrange) + 1, 1, cols)
        pixely = rows - clamp(floor(Int, (ymax - y) * yrange) + 1, 1, rows)

        img[pixely, pixelx] += 1.0f0
    end
    maxv = maximum(img)
    if maxv > 0f0
        img .= log.(1 .+ img) ./ log(1 .+ maxv)
    end
    return Gray.(img)
end

HEIGHWAY_EQ = [
   0.5 -0.5  0.5  0.5  0.0  0.0 0.5;
   0.5  0.5 -0.5  0.5  0.5  0.5 0.5
]

GOLDEN_EQ =  [ 
    0.62367 -0.40337 0.40337 0.62367 0.0 0.0;
    −0.37633 −0.40337 0.40337 −0.37633 1.0 0
   ]

function main(; eq = GOLDEN_EQ, npoints::Int = 1_000_000, outpath::AbstractString = "output.png")
    println("Building IFS with $npoints points...")
    maps, weights = build_maps_and_weights(eq)
    ifs = Fractal(maps, weights; npoints = npoints)

    println("Iterating (chaos game) ...")
    iterate!(ifs; warmup=DEFAULT_WARMUP)

    println("Rasterizing to image ...")
    img = make_image(ifs; resolution = RESOLUTION)

    println("Saving image to '$outpath' ...")
    save(outpath, img)
    println("Done.")
    # return ifs, img
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(npoints=1_000_000, outpath="fractal_output.png")
end
