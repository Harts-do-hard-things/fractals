using Images
using Colors
using FileIO
using StatsBase

RESOLUTION = (1504, 2256)

struct Fractal
    S::Matrix
    func_list::Vector{Function}
    limits::Vector
    probabilities::Weights
end


function Fractal(S::Matrix, func_list::Vector, weights::Weights)
    return Fractal(S, func_list, get_limits(func_list), weights)
end

H = [
   0.5 -0.5  0.5  0.5  0  0 0.5
   0.5  0.5 -0.5  0.5  0.5  0.5 0.5
]

Koch_Curve =  [ 
   0.333  0  0  0.333  0  0  0.25 
   0.167 -0.289  0.289  0.167  0.333  0  0.25 
   0.167  0.289 -0.289  0.167  0.5  0.289  0.25 
   0.333  0  0  0.333  0.667  0  0.25 
 ]

function Fractal(S::Matrix, eq::Matrix)
    func_list = []
    for i in 1:size(eq, 1)
        f(x::Vector)::Vector = permutedims(reshape(eq[i, 1:4], (2, 2)))*x .+ eq[i, 5:6]
        push!(func_list, f)
    end
    probabilities = Weights(eq[:, end])
    return Fractal(S, func_list, get_limits(func_list, probabilities), probabilities)
end

function apply(A::Vector, func_list::Vector)
    rand(func_list)(A)
end

function apply(A::Vector, func_list::Vector, weights::Weights)
    return sample(func_list, weights)(A)
end

function apply(A::Vector, f::Fractal) 
    return sample(f.func_list, f.probabilities)(A)
end

function iterate!(f::Fractal)
    A = f.S[:, 1]
    for _ in 1:50
        A = apply(A, f)
    end
    for i in 1:size(f.S, 2)
        A = apply(A, f)
        f.S[:, i] = A
    end
end

function get_limits(func_list)
    n = 10_000
    X = zeros(2, n)
    A = zeros(2)
    for i in 1:50
        A = apply(A, func_list)
    end
    for i in 1:n
        A = apply(A, func_list)
        X[:, i] = A
    end
    limits = extrema(X, dims=2)
    diff = (limits[1][2] - limits[1][1], limits[2][2] - limits[2][1])
    mdiff = maximum(diff)
    limits = [(limits[1][1] - (mdiff*1.05 - diff[1]) * 0.5,
               limits[1][2] + (mdiff*1.05 - diff[1]) * 0.5), 
              (limits[2][1] - (mdiff*1.05 - diff[2]) * 0.5,
               limits[2][2] + (mdiff*1.05 - diff[2]) * 0.5)]
    # print(limits)
    return limits
end


function get_limits(func_list, probabilities)
    n = 10_000
    X = zeros(2, n)
    A = zeros(2)
    for i in 1:50
        A = apply(A, func_list, probabilities)
    end
    for i in 1:n
        A = apply(A, func_list, probabilities)
        X[:, i] = A
    end
    limits = extrema(X, dims=2)
    diff = (limits[1][2] - limits[1][1], limits[2][2] - limits[2][1])
    mdiff = maximum(diff)
    limits = [(limits[1][1] - (mdiff*1.05 - diff[1]) * 0.5,
               limits[1][2] + (mdiff*1.05 - diff[1]) * 0.5), 
              (limits[2][1] - (mdiff*1.05 - diff[2]) * 0.5,
               limits[2][2] + (mdiff*1.05 - diff[2]) * 0.5)]
    return limits
end

function make_image(f::Fractal)
    img = zeros(Float32, RESOLUTION...)
    xlim, ylim = f.limits
    r = minimum(RESOLUTION)
    xrange = 1 / (xlim[2] - xlim[1])*r
    yrange = 1 / (ylim[2] - ylim[1])*r
    for col in eachcol(f.S)
        x, y = col
        pixelx = floor(Int, (x - xlim[1]) * xrange) + 1
        pixely = RESOLUTION[1] + floor(Int, (ylim[1] - y) * yrange) + 1
        v = img[pixely, pixelx]
        img[pixely, pixelx] = log(v + 2) > 1 ? 1 : log(v + 2)
    end
    return img
end

function main()
    f = Fractal(zeros(2, 1_000_000), H)
    iterate!(f)
    img = make_image(f)
    imgg = Gray.(img)
    save("output.png", imgg)
end
