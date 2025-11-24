# matrixfractal_refactored.jl
using StaticArrays
using LinearAlgebra
using StatsBase
using Images
using FileIO
using Colors
using BenchmarkTools

const RESOLUTION = (1504, 2256)  # (rows, cols) = (height, width)
const DEFAULT_WARMUP = 50        # chaos game warmup iterations
const DEFAULT_SAMPLES = 1_000_000

# ---------------------------------------------------------------------
# Affine map: z -> A*z + b
# ---------------------------------------------------------------------
struct AffineMap{T<:AbstractFloat}
    A::SMatrix{2,2,T}
    b::SVector{2,T}
end

AffineMap(A::AbstractMatrix, b::AbstractVector) = AffineMap(SMatrix{2,2,Float64}(A), SVector{2,Float64}(b))
AffineMap(a11::Float64,a12::Float64,a21::Float64,a22::Float64,b1::Float64,b2::Float64) =
    AffineMap(SMatrix{2,2,Float64}((a11,a12,a21,a22)), SVector{2,Float64}(b1,b2))

# call
(m::AffineMap)(x::SVector{2,Float64}) = m.A * x + m.b

# ---------------------------------------------------------------------
# Parametric IFS type (P is point type; we use SVector{2,Float64})
# ---------------------------------------------------------------------
struct IFS{P}
    points::Vector{P}           # preallocated container for points
    maps::Vector{AffineMap{Float64}}
    weights::Weights            # sampling weights for maps
    limits::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}  # ((xmin,xmax),(ymin,ymax))
end

# ---------------------------------------------------------------------
# Helpers to build map list & weights from matrix `eq`:
# eq is n x 6 or n x 7 where last column optionally holds probabilities.
# Each row: [a11 a12 a21 a22 b1 b2 (p?)] using row-major order for 2x2
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Constructor that accepts eq matrix and number of points to allocate
# ---------------------------------------------------------------------
function IFS(eq::AbstractMatrix{<:Real}; npoints::Integer=DEFAULT_SAMPLES)
    maps, weights = build_maps_and_weights(eq)
    # default initial point: origin as SVector{2,Float64}
    init_points = [SVector{2,Float64}(0.0, 0.0) for _ in 1:npoints]
    limits = get_limits(maps, weights)  # compute limits before using
    return IFS{SVector{2,Float64}}(init_points, maps, weights, limits)
end

# ---------------------------------------------------------------------
# Compute limits (bounding box) by sampling transient + many points
# ---------------------------------------------------------------------
function get_limits(maps::Vector{AffineMap{Float64}}, weights::Weights; warmup::Int=DEFAULT_WARMUP, n::Int=100_000)
    # start at origin
    x = SVector{2,Float64}(0.0, 0.0)

    # warmup
    for _ in 1:warmup
        idx = sample(1:length(maps), weights)
        x = maps[idx](x)
    end

    # sample n points and compute min/max
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

    # pad to a square region (preserve aspect)
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

# ---------------------------------------------------------------------
# iterate! - perform chaos game into ifs.points (fills the vector)
# Uses vectorized sampling of map indices for speed, then applies in a loop.
# ---------------------------------------------------------------------
function iterate!(ifs::IFS{SVector{2,Float64}}; warmup::Int=DEFAULT_WARMUP)
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

# ---------------------------------------------------------------------
# make_image - rasterize points into a grayscale image using log scaling
# ---------------------------------------------------------------------
function make_image(ifs::IFS{SVector{2,Float64}}; resolution::Tuple{Int,Int}=RESOLUTION)
    rows, cols = resolution
    img = zeros(Float32, rows, cols)  # row-major (y,x)

    (xlim, ylim) = ifs.limits
    xmin, xmax = xlim
    ymin, ymax = ylim

    # scale factors: column per x-unit, row per y-unit
    sx = (cols - 1) / (xmax - xmin)
    sy = (rows - 1) / (ymax - ymin)

    for pt in ifs.points
        x = pt[1]; y = pt[2]
        # convert to pixel coords (clamp to image)
        col = clamp(Int(floor((x - xmin) * sx)) + 1, 1, cols)
        # flip y so larger y is up (image row 1 at top)
        row = clamp(Int(floor((ymax - y) * sy)) + 1, 1, rows)
        # accumulate with soft log normalization later
        img[row, col] += 1.0f0
    end

    # apply logarithmic scaling and clamp to [0,1]
    # log(1 + count) / log(1 + maxcount) creates a normalized image
    maxv = maximum(img)
    if maxv > 0f0
        img .= log.(1 .+ img) ./ log(1 .+ maxv)
    end

    # convert to Gray image
    return Gray.(img)
end

# ---------------------------------------------------------------------
# Small utility constructor from list of AffineMaps & weights
# ---------------------------------------------------------------------
function IFS(maps::Vector{AffineMap{Float64}}, weights::Weights; npoints::Integer=DEFAULT_SAMPLES)
    limits = get_limits(maps, weights)
    pts = [SVector{2,Float64}(0.0, 0.0) for _ in 1:npoints]
    return IFS{SVector{2,Float64}}(pts, maps, weights, limits)
end

# ---------------------------------------------------------------------
# Example fractal matrices (Koch-like example, Heighway dragon)
# Note: ensure plain ASCII minus signs
# ---------------------------------------------------------------------
HEIGHWAY_EQ = [
    0.5 -0.5  0.5  0.5  0.0  0.0;
    0.5  0.5 -0.5  0.5  0.5  0.5
]

# A simple Koch-ish example (scaled to 0..1 area)
KOCH_EQ = [
    0.333333 0.0      0.0     0.333333 0.0     0.0  1/3;
    0.333333 0.0      0.0     0.333333 0.333333 0.0 1/3;
    0.333333 0.0      0.0     0.333333 0.666666 0.0 1/3
]

# ---------------------------------------------------------------------
# main demonstration
# ---------------------------------------------------------------------
function main(; eq = HEIGHWAY_EQ, npoints::Int = 1_000_000, outpath::AbstractString = "output.png")
    println("Building IFS with $npoints points...")
    maps, weights = build_maps_and_weights(eq)
    ifs = IFS(maps, weights; npoints = npoints)

    println("Iterating (chaos game) ...")
    @btime iterate!($ifs; warmup=DEFAULT_WARMUP) setup=(GC.gc())

    println("Rasterizing to image ...")
    img = make_image(ifs; resolution = RESOLUTION)

    println("Saving image to '$outpath' ...")
    save(outpath, img)
    println("Done.")
    # return ifs, img
end

# Only run main when executed directly (not when included)
if abspath(PROGRAM_FILE) == @__FILE__
    main(npoints=1_000_000, outpath="fractal_output.png")
end
