using Test
using LinearAlgebra
using StaticArrays

include("../src/matrixfractal.jl")

const SMALL_EQ = [
    0.5  0.0  0.0  0.5  0.0  0.0  0.6;
   -0.5  0.0  0.0 -0.5  1.0  0.0  0.4
]

@testset "AffineMap" begin
    A = [0.5 0.0; 0.0 0.5]
    b = [1.0, -2.0]
    m = AffineMap(A, b)
    x = SVector{2,Float64}(2.0, 4.0)
    y = m(x)
    @test y ≈ SVector{2,Float64}(2.0, 0.0)

    mi = inv(m)
    @test mi(y) ≈ x
end

@testset "Build Maps And Weights" begin
    maps, weights = build_maps_and_weights(SMALL_EQ)
    @test length(maps) == size(SMALL_EQ, 1)
    @test length(weights) == size(SMALL_EQ, 1)
    @test sum(weights) > 0
end

@testset "Get Limits" begin
    maps, weights = build_maps_and_weights(SMALL_EQ)
    limits = get_limits(maps, weights; warmup=10, n=200)
    (xlim, ylim) = limits
    @test xlim[1] < xlim[2]
    @test ylim[1] < ylim[2]
end

@testset "IFS Constructors" begin
    ifs = IFS(SMALL_EQ; npoints=1000)
    @test length(ifs.points) == 1000
    @test length(ifs.maps) == size(SMALL_EQ, 1)
end

@testset "Iterate" begin
    ifs = IFS(SMALL_EQ; npoints=5000)
    iterate!(ifs; warmup=5)
    @test length(ifs.points) == 5000

    ifs2 = IFS(SMALL_EQ; npoints=5000)
    iterate_parallel!(ifs2; warmup=5)
    @test length(ifs2.points) == 5000
end

@testset "Deterministic Iterate" begin
    ifs = IFS(SMALL_EQ; npoints=50)
    n = 2
    out = deterministic_iterate(ifs, n)
    @test length(out.points) == length(ifs.points) * length(ifs.maps)^n
end

@testset "Pixel Maps" begin
    ifs = IFS(SMALL_EQ; npoints=100)
    pmap = make_pixelate_map(ifs.limits; resolution=(32, 32))
    nmap = ifs.maps[1]
    cmap = make_pixeliterate_map(nmap, pmap)

    p = SVector{2,Float64}(16.0, 16.0)
    direct = pmap(nmap(inv(pmap)(p)))
    composed = cmap(p)
    @test direct ≈ composed
end

@testset "Make Image" begin
    ifs = IFS(SMALL_EQ; npoints=2000)
    iterate!(ifs; warmup=5)
    img = make_image(ifs; resolution=(32, 32))
    @test size(img) == (32, 32)
    @test eltype(img) == Float32
    @test all(img .>= 0f0)
    @test maximum(img) <= 1f0
end

@testset "Iterate Image" begin
    ifs = IFS(SMALL_EQ; npoints=2000)
    iterate!(ifs; warmup=5)
    img = make_image(ifs; resolution=(32, 32))
    out = iterate_image(ifs, img)
    @test size(out) == (32, 32)
end

@testset "Inverse Rasterize" begin
    ifs = IFS(SMALL_EQ; npoints=100)
    lims = ifs.limits
    img = rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8))
    @test size(img) == (8, 8)
    @test eltype(img) == Float32
    @test all(img .>= 0f0)
    @test maximum(img) <= 1f0
end

@testset "IFS Parser" begin
    sample = """
    Test IFS {; Doc line 1
    ; Doc line 2
      0.5 0.0 0.0 0.5 0.0 0.0 0.6
     -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
    }
    """
    defs = parse_ifs_string(sample; npoints=10)
    @test length(defs) == 1
    @test defs[1].name == "Test IFS"
    @test occursin("Doc line 1", defs[1].docs)
    @test length(defs[1].points) == 10
end
