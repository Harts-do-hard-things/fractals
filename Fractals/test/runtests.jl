using Test
using LinearAlgebra
using StaticArrays
using Fractals
using Random
using FileIO
using Colors

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
    maps, weights = Fractals._build_maps_and_weights(SMALL_EQ)
    @test length(maps) == size(SMALL_EQ, 1)
    @test length(weights) == size(SMALL_EQ, 1)
    @test sum(weights) > 0
end

@testset "Get Limits" begin
    maps, weights = Fractals._build_maps_and_weights(SMALL_EQ)
    limits = Fractals._get_limits(maps, weights; warmup=10, n=200)
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
    cmap = Fractals._make_pixeliterate_map(nmap, pmap)

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

@testset "IFS Parser File" begin
    sample = """
    File IFS {
      0.5 0.0 0.0 0.5 0.0 0.0 0.5
      0.5 0.0 0.0 0.5 0.5 0.0 0.5
    }
    """

    mktemp() do path, io
        write(io, sample)
        close(io)
        defs = parse_ifs_file(path; npoints=12)
        @test length(defs) == 1
        @test defs[1].name == "File IFS"
        @test length(defs[1].points) == 12
    end
end

@testset "Media Output Path" begin
    p1 = Fractals._normalize_media_outpath("output.png")
    @test p1 == joinpath("media", "output.png")

    p2 = Fractals._normalize_media_outpath(joinpath("media", "nested", "x.png"))
    @test p2 == joinpath("media", "nested", "x.png")

    p3 = Fractals._normalize_media_outpath(joinpath("other", "path", "image.png"))
    @test p3 == joinpath("media", "image.png")

    @test isdir("media")
end

@testset "Affine Map SVG Rendering" begin
    ifs = IFS(SMALL_EQ; npoints=50)

    suffix = randstring(8)
    svg_path = render_transformations_svg(ifs; outpath=joinpath("media", "maps_$suffix.svg"), width=300, height=300)
    @test isfile(svg_path)

    svg_text = read(svg_path, String)
    @test occursin("<svg", svg_text)
    @test occursin("<line", svg_text)
    @test occursin("stroke=\"#", svg_text)
    @test !occursin("stroke=\"#222222\"", svg_text)

    colors = Set{String}()
    for m in eachmatch(r"stroke=\"(#[0-9A-Fa-f]{6})\"", svg_text)
        push!(colors, m.captures[1])
    end
    @test length(colors) >= length(ifs.maps)

    png_path = render_transformations_png(ifs; outpath=joinpath("media", "maps_$suffix.png"), width=256, height=256)
    @test isfile(png_path)
    @test filesize(png_path) > 0
    png_img = load(png_path)
    @test alpha(png_img[1, 1]) == 0

    rm(svg_path; force=true)
    rm(png_path; force=true)
end

@testset "Affine Map Color Assignment By Map" begin
    ifs = IFS(EISENSTEIN; npoints=50)
    suffix = randstring(8)
    svg_path = render_transformations_svg(ifs; outpath=joinpath("media", "maps_colors_$suffix.svg"), width=320, height=320)
    @test isfile(svg_path)

    svg_text = read(svg_path, String)
    map_line_count = length(Fractals._base_l_image())

    counts = Dict{String,Int}()
    for m in eachmatch(r"stroke=\"(#[0-9A-Fa-f]{6})\"", svg_text)
        c = m.captures[1]
        counts[c] = get(counts, c, 0) + 1
    end

    @test length(counts) == length(ifs.maps)
    @test all(v == map_line_count for v in values(counts))

    rm(svg_path; force=true)
end
