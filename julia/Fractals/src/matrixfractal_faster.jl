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


# struct AffineMap{T<:AbstractFloat, AType<:AbstractMatrix{T}, BType<:AbstractVector{T}}
struct AffineMap{T<:AbstractFloat, AType<:StaticMatrix{2,2,T}, BType<:StaticVector{2,T}}
    A::AType
    b::BType
end

function Base.inv(m::AffineMap)
    A = m.A
    Ainv = inv(A)
    binv = -Ainv * m.b
    return AffineMap(Ainv, binv)
end

# TODO fix the column/row major issue
AffineMap(A::AbstractMatrix, b::AbstractVector) = AffineMap(SMatrix{2,2,Float64}(A), SVector{2,Float64}(b))
AffineMap(a11::Float64,a12::Float64,a21::Float64,a22::Float64,b1::Float64,b2::Float64) =
    AffineMap(SMatrix{2,2,Float64}((a11,a12,a21,a22)), SVector{2,Float64}(b1,b2))

# make it callable 
(m::AffineMap)(x::SVector{2, T}) where T = m.A * x + m.b

struct Fractal{P}
    points::Vector{P}
    maps::Vector{AffineMap{Float64}}
    weights::Weights
    limits::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}
end

# struct Fractal{P, T}
#     points::Vector{P{T}}
#     maps::Vector{AffineMap{T}}
#     weights::Weights
#     limits::Tuple{Tuple,Tuple}
# end

struct ImageFractal
    maps::Vector{AffineMap{Float64}}
    weights::Weights
    limits::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}
    img
end

# function Base.show(io::IO, m::AffineMap)
#     println(io, "AffineMap(")
#     print(io, "  A = ")
#     show(io, "text/plain", m.A)
#     println(io, ",")
#     print(io, "  b = ")
#     show(io, "text/plain", m.b)
#     print(io, "\n)")
# end

function Base.show(io::IO, f::Fractal)
    println(io, "Fractal with $( length(f.maps) ) equations and $( length(f.points) ) points")
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

function isinspace(v::AbstractVector, limits::Tuple{Tuple,Tuple})
    ( xlim, ylim ) = limits
    xmin, xmax = xlim
    ymin, ymax = ylim
    return xmin ≤ v[1] ≤ xmax && ymin ≤ v[2] ≤ ymax 
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


# function InitialPolyFractal(eq::AbstractMatrix{<:Real}; npoints::Integer=DEFAULT_SAMPLES)
#     maps, weights = build_maps_and_weights(eq)
#     rows, cols = resolution
#     # default initial point: origin as SVector{2,Float64}
#     limits = get_limits(maps, weights)  # compute limits before using
#     ( xlim, ylim ) = ifs.limits
#     xmin, xmax = xlim
#     ymin, ymax = ylim
#     r = minimum(RESOLUTION)
#     xrange = r / (xmax - xmin)
#     yrange = r / (ymax - ymin)
#     initial_poly = [
#                     SVector{2, Float64}(0.0, 0.0),
#                     SVector{2, Float64}(0.0, 1.0),
#                    ]
#     for pt in initial_poly
#         x, y = pt
#         pixelx = clamp(round(
#                              Int, (x - xmin) * xrange
#                             ) + 1 + (cols - r) ÷ 2, 1, cols)
#         # pixelx = clamp(round(Int, (x - xmin) * xrange) + 1, 1, cols)
#         pixely = rows - clamp(round(
#                                     Int, (ymax - y) * yrange
#                                    ) + 1 + (rows - r) ÷ 2, 1, rows)
#     end
#     init_points = [SVector{2,Float64}(0.0, 0.0) for _ in 1:npoints]

#     return Fractal{SVector{2,Float64}}(init_points, maps, weights, limits)
# end

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

# TODO get this function to work without an ifs, just with a set of 
# affine maps and weights and a number of points
# Same with the deterministic iterate
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

function deterministic_apply(maps::Vector, n::Integer, points::Vector)
    nmaps = length(maps)
    l = length(points)
    new_points_size = l * nmaps ^ n
    new_points = Vector{SVector{2,Float64}}(undef, new_points_size)
    new_points[1:l] = points

    for j in 1:n
        new_points[end-l+1:end] = new_points[1:l]  # copy previous points
        for i in 1:length(maps)
            # if j < n
            new_points[1 + (i-1)*l : i*l] = maps[i].(new_points[end-l+1:end])
            # else
                # on the last iteration, use the points from the end of the array
            # points[1 + (i-1)*l : i*l] = maps[i].(points[end - l + 1:end])

            if i == length(maps)
                l = l * nmaps  # update l for next iteration
            end
        end
    end
    return new_points
end

function deterministic_iterate(ifs::Fractal{SVector{2,Float64}}, n::Integer)
    points = deterministic_apply(ifs.maps, n, ifs.points)
    return Fractal{SVector{2,Float64}}(points, ifs.maps, ifs.weights, ifs.limits)
end

function inverse_iterate(ifs::Fractal, n::Integer, point::SVector{2, Float64}, atol::AbstractFloat)
    inverse_maps = inv.(ifs.maps)

    ( xlim, ylim ) = ifs.limits
    xmin, xmax = xlim
    ymin, ymax = ylim
    max_atol = min(xmax - xmin, ymax - ymin) * 0.5

    points = [ point ]
    for j in 1:n
        points = [imap(p) for p in points for imap in inverse_maps if isinspace(p, f.limits)]
        # if any(i -> isapprox(i, SVector(0., 0.), atol=atol), points)
        #     return 1.0f0
        #     println("bounced by zero, ending early")
        # end
        if length(points) == 0
            return j / n * 0.5f0
        end
    end
    # return points
    # println("Never approached zero")
    return 1.0f0
end

function test_image(ifs::Fractal, n::Integer, limits::Tuple{Tuple, Tuple}; resolution::Tuple{Int,Int}=RESOLUTION)
    pmap = make_pixelate_map(limits, resolution=resolution)
    xmap = inv(pmap)
    rows, cols = resolution
    img = zeros(Float32, rows, cols)
    for y in 1:rows
        for x in 1:cols
            img[y, x] = inverse_iterate(ifs, n, xmap(SVector(x, y)), xmap.A[1,1])
        end
    end
    return img
end

function make_pixelate_map(limits::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}; resolution::Tuple{Int,Int}=RESOLUTION)
    rows, cols = resolution

    ( xlim, ylim ) = limits
    xmin, xmax = xlim
    ymin, ymax = ylim

    r = minimum(resolution)
    xrange = 1 / (xmax - xmin) * r
    yrange = 1 / (ymax - ymin) * r
    A = SMatrix{2, 2}([xrange 0;
                       0 yrange])
    b = SVector(-xmin * xrange + (cols - r) / 2 + 0.5,
                -ymin * yrange + (rows - r) / 2 + 0.5)
    return AffineMap(A, b)
end

function make_pixeliterate_map(iterate_map::AffineMap, pixelate_map::AffineMap)
    numberize_map = inv(pixelate_map)
    A = pixelate_map.A * iterate_map.A * numberize_map.A
    b =  pixelate_map.A * ( iterate_map.A * numberize_map.b + iterate_map.b) + pixelate_map.b
    return AffineMap(A, b)
end

function make_image(ifs::Fractal; resolution::Tuple{Int,Int}=RESOLUTION)
    map = make_pixelate_map(ifs.limits; resolution=resolution)
    rows, cols = resolution
    img = zeros(Float32, rows, cols)

    for pt in ifs.points
        pixels = map(pt)
        pixelx = clamp(round(Int, pixels[1]), 1, cols)
        pixely = clamp(round(Int, pixels[2]), 1, rows)

        img[pixely, pixelx] += 1.0f0
    end
    maxv = maximum(img)
    if maxv > 0f0
        img .= log.(1 .+ img) ./ log(1 .+ maxv)
    end
    return img
end

function iterate_image(ifs::Fractal, img)
    newimg = zeros(Float32, size(img))
    pmap = make_pixelate_map(ifs.limits; resolution=size(img))
    for nmap in ifs.maps
        cmap = make_pixeliterate_map(nmap, pmap)
        for y in 1:size(img)[1]
            for x in 1:size(img)[2]
                if img[y, x] != 0
                    fx, fy = pmap(nmap(inv(pmap)(SVector(x, y))))
                    newx = clamp(round(Int, fx), 1, size(img)[2])
                    newy = clamp(round(Int, fy), 1, size(img)[1])
                    newimg[newy, newx] += img[y, x]
                end
            end
        end
    end
    maxv = maximum(newimg)
    if maxv > 0f0
        newimg .= log.(1 .+ newimg) ./ log(1 .+ maxv)
    end
    return Gray.(newimg)
end

GOLDEN_EQ =  [ 
    0.62367 -0.40337 0.40337 0.62367 0.0 0.0;
    −0.37633 −0.40337 0.40337 −0.37633 1.0 0.0
   ]

HEIGHWAY_DRAGON = [ 
                 0.5 -0.5  0.5  0.5  0  0;
                 -0.5 -0.5  0.5 -0.5  1  0
                ]

Levy_Dragon = [ 
             0.5 -0.5  0.5  0.5  0  0 0.5;
             0.5  0.5 -0.5  0.5  0.5  0.5 0.5 ]

const EISENSTEIN = [
-0.500000  0.000000  0.000000 -0.500000  0.000000  0.000000  0.250000; 
-0.500000  0.000000  0.000000 -0.500000 -0.500000  0.000000  0.250000;
-0.500000  0.000000  0.000000 -0.500000  0.250000 -0.433000  0.250000;
-0.500000  0.000000  0.000000 -0.500000  0.250000  0.433000  0.250000;
]

function main(; eq = EISENSTEIN, npoints::Int = 1_000_000, outpath::AbstractString = "output.png")
    println("Building IFS with $npoints points...")
    maps, weights = build_maps_and_weights(eq)
    ifs = Fractal(maps, weights; npoints = npoints)

    println("Iterating (chaos game) ...")
    iterate!(ifs; warmup=DEFAULT_WARMUP)
    println("Performing deterministic iteration ...")
    ifs = deterministic_iterate(ifs, 1)

    println("Rasterizing to image ...")
    img = make_image(ifs; resolution = RESOLUTION)
    # newimg = iterate_image(ifs, img)

    println("Saving image to '$outpath' ...")
    save(outpath, img)
    println("Done.")
    # return ifs, img
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(npoints=1_000_000, outpath="fractal_output.png")
end
