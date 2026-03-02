using Test
using LinearAlgebra
using StaticArrays
using Fractals
using Random
using FileIO
using Colors
using JSON3

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

@testset "Affine Inverse Property Tests" begin
    rng = MersenneTwister(20260228)
    n_maps = 180
    n_points_per_map = 4
    det_eps = 1e-3

    for _ in 1:n_maps
        A = zeros(Float64, 2, 2)
        # Resample until we get a comfortably invertible matrix.
        while true
            A .= randn(rng, 2, 2)
            abs(det(A)) > det_eps && break
        end
        b = randn(rng, 2)
        m = AffineMap(A, b)
        mi = inv(m)

        for _ in 1:n_points_per_map
            x = SVector{2,Float64}(randn(rng), randn(rng))
            y = m(x)
            z = mi(x)
            @test mi(y) ≈ x atol=1e-10 rtol=1e-10
            @test m(z) ≈ x atol=1e-10 rtol=1e-10
        end
    end
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

@testset "Seeded Iterate Reproducibility" begin
    ifs1 = IFS(SMALL_EQ; npoints=1500)
    ifs2 = IFS(SMALL_EQ; npoints=1500)
    ifs3 = IFS(SMALL_EQ; npoints=1500)

    Fractals._iterate_serial!(ifs1; warmup=10, seed=12345)
    Fractals._iterate_serial!(ifs2; warmup=10, seed=12345)
    Fractals._iterate_serial!(ifs3; warmup=10, seed=54321)

    @test ifs1.points == ifs2.points
    @test ifs1.points != ifs3.points

    # Alias should preserve seeded behavior.
    ifs4 = IFS(SMALL_EQ; npoints=1200)
    ifs5 = IFS(SMALL_EQ; npoints=1200)
    iterate!(ifs4; warmup=8, seed=99)
    iterate_parallel!(ifs5; warmup=8, seed=99)
    @test ifs4.points == ifs5.points
end

@testset "Deterministic Iterate" begin
    ifs = IFS(SMALL_EQ; npoints=20)
    n = 2
    ifs_orig_points = copy(ifs.points)

    out = deterministic_iterate(ifs, n; warmup=5, seed=123)
    @test length(out.points) == length(ifs.points) * length(ifs.maps)^n
    @test ifs.points == ifs_orig_points

    # Deterministic iterate should chaos-initialize internally before expansion.
    ifs2 = IFS(SMALL_EQ; npoints=20)
    base = IFS(ifs2.name, ifs2.docs, copy(ifs2.points), ifs2.maps, ifs2.weights, ifs2.limits)
    iterate!(base; warmup=5, seed=123)
    expected_points = Fractals._deterministic_expand_points(base.points, ifs2.maps, n)
    out2 = deterministic_iterate(ifs2, n; warmup=5, seed=123)
    @test out2.points == expected_points
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

@testset "Iterate Image Colors" begin
    ifs = IFS(SMALL_EQ; npoints=1500)
    iterate!(ifs; warmup=5)
    img = make_image(ifs; resolution=(32, 32))

    out1 = iterate_image(ifs, img; colors=true, seed=123)
    out2 = iterate_image(ifs, img; colors=true, seed=123)

    @test size(out1) == (32, 32)
    @test eltype(out1) == RGB{Float32}
    @test out1 == out2
end

@testset "Inverse Rasterize" begin
    ifs = IFS(SMALL_EQ; npoints=100)
    lims = ifs.limits
    img = rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8))
    @test size(img) == (8, 8)
    @test eltype(img) == Float32
    @test all(img .>= 0f0)
    @test maximum(img) <= 1f0

    img_hide = rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8), show_divergence_scale=false)
    @test size(img_hide) == (8, 8)
    @test eltype(img_hide) == Float32
    @test all(v -> v == 0.0f0 || v == 1.0f0, img_hide)
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

@testset "IFS Parser Negative Cases" begin
    # Missing closing brace.
    missing_close = """
    BrokenIFS {
      0.5 0.0 0.0 0.5 0.0 0.0
    """
    @test_throws EOFError parse_ifs_string(missing_close; npoints=1)

    # Unexpected closing brace.
    stray_close = """
    }
    ValidName {
      0.5 0.0 0.0 0.5 0.0 0.0
    }
    """
    err = try
        parse_ifs_string(stray_close; npoints=1)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Unexpected '}' outside IFS block", sprint(showerror, err))

    # Mixed row widths in same block.
    mixed_width = """
    MixedWidth {
      0.5 0.0 0.0 0.5 0.0 0.0
      0.5 0.0 0.0 0.5 0.5 0.0 0.5
    }
    """
    err = try
        parse_ifs_string(mixed_width; npoints=1)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("has length", sprint(showerror, err))

    # Bad float token.
    bad_float = """
    BadFloat {
      0.5 0.0 abc 0.5 0.0 0.0
    }
    """
    err = try
        parse_ifs_string(bad_float; npoints=1)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Invalid numeric token 'abc'", sprint(showerror, err))

    # File parse should surface the same failures.
    mktemp() do path, io
        write(io, mixed_width)
        close(io)
        @test_throws ArgumentError parse_ifs_file(path; npoints=1)
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

@testset "Render Entrypoint" begin
    suffix = randstring(8)

    out1 = render(SMALL_EQ; npoints=5000, method=:chaos, resolution=(64, 64), outpath=joinpath("media", "render_matrix_$suffix.png"))
    @test isfile(out1.outpath)
    @test size(out1.image) == (64, 64)

    ifs = IFS(SMALL_EQ; npoints=2000)
    out2 = render(ifs; method=:deterministic, iterations=1, resolution=(48, 48), outpath=joinpath("media", "render_ifs_$suffix.png"))
    @test isfile(out2.outpath)
    @test size(out2.image) == (48, 48)
    @test length(out2.ifs.points) == length(ifs.points) * length(ifs.maps)

    text = """
    RenderTest {
      0.5 0.0 0.0 0.5 0.0 0.0 0.6
     -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
    }
    """
    out3 = render(text; npoints=4000, method=:parallel, resolution=(40, 40), outpath=joinpath("media", "render_text_$suffix.png"))
    @test isfile(out3.outpath)
    @test size(out3.image) == (40, 40)
    @test out3.method == :chaos

    out3b = render(text; npoints=2000, method="deterministic", iterations=1, resolution=(24, 24), outpath=joinpath("media", "render_text_string_method_$suffix.png"))
    @test isfile(out3b.outpath)
    @test size(out3b.image) == (24, 24)
    @test out3b.method == :deterministic

    out3c = render(text; npoints=1500, method=Parallel, resolution=(20, 20), outpath=joinpath("media", "render_text_enum_method_$suffix.png"))
    @test isfile(out3c.outpath)
    @test size(out3c.image) == (20, 20)
    @test out3c.method == :chaos

    mktemp() do path, io
        write(io, text)
        close(io)
        out4 = render(path; npoints=3000, method=:inverse, ifs_index=1, iterations=2, resolution=(32, 32), outpath=joinpath("media", "render_file_$suffix.png"))
        @test isfile(out4.outpath)
        @test size(out4.image) == (32, 32)

        out4_hide = render(path; npoints=3000, method=:inverse, ifs_index=1, iterations=2, show_divergence_scale=false, resolution=(32, 32), outpath=joinpath("media", "render_file_hide_$suffix.png"))
        @test isfile(out4_hide.outpath)
        @test size(out4_hide.image) == (32, 32)
        @test all(v -> v == 0.0f0 || v == 1.0f0, out4_hide.image)
    end

    @test_throws ArgumentError render(SMALL_EQ; method=:badmethod)
    @test_throws ArgumentError render(SMALL_EQ; method="not_a_method")

    rm(out1.outpath; force=true)
    rm(out2.outpath; force=true)
    rm(out3.outpath; force=true)
    rm(out3b.outpath; force=true)
    rm(out3c.outpath; force=true)
    rm(joinpath("media", "render_file_$suffix.png"); force=true)
    rm(joinpath("media", "render_file_hide_$suffix.png"); force=true)
end

@testset "Render IFS Selection By Index Or Name" begin
    suffix = randstring(8)
    text = """
    FirstIFS {
      0.5 0.0 0.0 0.5 0.0 0.0 0.5
      0.5 0.0 0.0 0.5 0.5 0.0 0.5
    }
    SecondIFS {
      0.5 -0.5 0.5 0.5 0.0 0.0
     -0.5 -0.5 0.5 -0.5 1.0 0.0
    }
    """

    mktemp() do path, io
        write(io, text)
        close(io)

        out_idx = render(path; ifs_index=2, npoints=1500, method=:chaos, resolution=(24, 24), outpath=joinpath("media", "render_multi_idx_$suffix.png"))
        @test out_idx.ifs.name == "SecondIFS"
        @test isfile(out_idx.outpath)

        out_name = render(path; ifs_name="FirstIFS", npoints=1500, method=:chaos, resolution=(24, 24), outpath=joinpath("media", "render_multi_name_$suffix.png"))
        @test out_name.ifs.name == "FirstIFS"
        @test isfile(out_name.outpath)

        @test_throws ArgumentError render(path; ifs_index=99, npoints=100, method=:chaos, resolution=(16, 16), outpath=joinpath("media", "never_written_$suffix.png"))
        @test_throws ArgumentError render(path; ifs_name="MissingIFS", npoints=100, method=:chaos, resolution=(16, 16), outpath=joinpath("media", "never_written2_$suffix.png"))
        @test_throws ArgumentError render(path; ifs_name="FirstIFS", ifs_index=1, npoints=100, method=:chaos, resolution=(16, 16), outpath=joinpath("media", "never_written3_$suffix.png"))

        rm(out_idx.outpath; force=true)
        rm(out_name.outpath; force=true)
    end
end

@testset "Validation And Errors" begin
    bad5 = ones(2, 5)
    err = try
        IFS(bad5; npoints=10)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("6 or 7 columns", sprint(showerror, err))

    bad8 = ones(2, 8)
    err = try
        IFS(bad8; npoints=10)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("6 or 7 columns", sprint(showerror, err))

    empty_eq = Matrix{Float64}(undef, 0, 6)
    err = try
        IFS(empty_eq; npoints=10)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("at least one row", sprint(showerror, err))

    bad_nan = copy(SMALL_EQ)
    bad_nan[1, 1] = NaN
    err = try
        IFS(bad_nan; npoints=10)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("non-finite values", sprint(showerror, err))

    bad_prob_neg = copy(SMALL_EQ)
    bad_prob_neg[1, 7] = -0.1
    err = try
        IFS(bad_prob_neg; npoints=10)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("nonnegative", sprint(showerror, err))

    bad_prob_zero = copy(SMALL_EQ)
    bad_prob_zero[:, 7] .= 0.0
    err = try
        IFS(bad_prob_zero; npoints=10)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("positive total weight", sprint(showerror, err))

    ifs = IFS(SMALL_EQ; npoints=100)
    img = make_image(ifs; resolution=(16, 16))
    err = try
        Fractals._iterate_image_single_map(ifs, img, 99)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("out of range", sprint(showerror, err))
end

@testset "Interactive And Main Entrypoints" begin
    sample = """
    PromptIFS {
      0.5 0.0 0.0 0.5 0.0 0.0 0.6
     -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
    }
    """

    mktempdir() do d
        old = pwd()
        cd(d)
        try
            ifs_path = joinpath(d, "prompt.ifs")
            write(ifs_path, sample)

            input_lines = "\n\n\n\n\n\n"  # accept defaults for selection/method/size/path prompts
            stdin_path = joinpath(d, "stdin.txt")
            write(stdin_path, input_lines)
            selected = open(stdin_path, "r") do io
                redirect_stdin(io) do
                    prompt_ifs_and_render(ifs_path; npoints=100, resolution=(32, 32), outpath="media/prompt_output.png")
                end
            end
            @test selected isa IFS
            @test isfile(joinpath("media", "prompt_output.png"))

            Fractals.main(eq=SMALL_EQ, npoints=1000, outpath="media/main_output.png")
            @test isfile(joinpath("media", "main_output.png"))
        finally
            cd(old)
        end
    end
end

@testset "Render Color API" begin
    mktempdir() do d
        old = pwd()
        cd(d)
        try
            out_chaos = render(SMALL_EQ;
                               method=Chaos,
                               npoints=800,
                               warmup=5,
                               color=true,
                               resolution=(40, 40),
                               outpath="media/render_chaos_color.png")
            @test size(out_chaos.image) == (40, 40)
            @test eltype(out_chaos.image) == RGB{Float32}
            @test isfile(joinpath("media", "render_chaos_color.png"))

            out_det = render(SMALL_EQ;
                             method=Deterministic,
                             npoints=40,
                             iterations=1,
                             color=true,
                             resolution=(40, 40),
                             outpath="media/render_det_color.png")
            @test size(out_det.image) == (40, 40)
            @test eltype(out_det.image) == RGB{Float32}
            @test isfile(joinpath("media", "render_det_color.png"))

            out_inv = render(SMALL_EQ;
                             method=Inverse,
                             iterations=2,
                             color=true,
                             resolution=(40, 40),
                             outpath="media/render_inv_color.png")
            @test size(out_inv.image) == (40, 40)
            @test eltype(out_inv.image) == RGB{Float32}
            @test isfile(joinpath("media", "render_inv_color.png"))
        finally
            cd(old)
        end
    end
end

@testset "CLI Commands" begin
    script = abspath(joinpath(@__DIR__, "..", "bin", "fractals.jl"))
    project = abspath(joinpath(@__DIR__, ".."))
    jcmd = Base.julia_cmd()

    sample = """
    CLI_First {
      0.5 0.0 0.0 0.5 0.0 0.0 0.6
     -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
    }
    CLI_Second {
      0.5 -0.5 0.5 0.5 0.0 0.0
     -0.5 -0.5 0.5 -0.5 1.0 0.0
    }
    """

    mktempdir() do d
        old = pwd()
        cd(d)
        try
            ifs_path = joinpath(d, "cli.ifs")
            write(ifs_path, sample)

            validate_out = read(`$jcmd --startup-file=no --project=$project $script validate-ifs --input $ifs_path`, String)
            @test occursin("Definitions: 2", validate_out)
            @test occursin("CLI_First", validate_out)

            render_out = read(`$jcmd --startup-file=no --project=$project $script render --input $ifs_path --ifs-index 2 --npoints 2000 --resolution 32x32 --out media/cli_single.png`, String)
            @test occursin("Rendered", render_out)
            @test isfile(joinpath("media", "cli_single.png"))

            batch_out = read(`$jcmd --startup-file=no --project=$project $script batch-render --input $ifs_path --npoints 1000 --resolution 24x24 --out-dir media/batch`, String)
            @test occursin("Batch rendering 2 definitions", batch_out)
            @test isfile(joinpath("media", "batch", "01_cli_first.png"))
            @test isfile(joinpath("media", "batch", "02_cli_second.png"))

            bench_out = read(`$jcmd --startup-file=no --project=$project $script benchmark --profile small --repeats 1 --npoints 500 --resolution 12x12 --inverse-iterations 1`, String)
            @test occursin("Benchmark suite", bench_out)
            @test occursin("iterate!", bench_out)
            @test occursin("alloc_mean=", bench_out)

            bench_json = joinpath("benchmarks", "bench_small.json")
            bench_out_json = read(`$jcmd --startup-file=no --project=$project $script benchmark --profile small --repeats 1 --npoints 500 --resolution 12x12 --inverse-iterations 1 --json $bench_json`, String)
            @test occursin("Wrote benchmark JSON", bench_out_json)
            @test isfile(bench_json)
            json_txt = read(bench_json, String)
            @test occursin("\"results\"", json_txt)
            @test occursin("\"small\"", json_txt)

            payload = JSON3.read(json_txt)
            @test haskey(payload.results, :small)
            small = payload.results.small
            for k in (:iterate!, Symbol("iterate_parallel!"), :make_image, :rasterize_image_inversely)
                @test haskey(small, k)
                @test haskey(small[k], :min_s)
                @test haskey(small[k], :mean_s)
                @test haskey(small[k], :max_s)
                @test haskey(small[k], :mean_alloc_bytes)
                @test small[k].min_s >= 0
                @test small[k].mean_s >= 0
                @test small[k].max_s >= 0
                @test small[k].mean_alloc_bytes >= 0
            end

            targets_path = abspath(joinpath(project, "bench", "perf_targets.toml"))
            bench_targets_out = read(`$jcmd --startup-file=no --project=$project $script benchmark --profile small --repeats 1 --npoints 500 --resolution 12x12 --inverse-iterations 1 --targets $targets_path`, String)
            @test occursin("Target comparison", bench_targets_out)
            @test occursin("targets:", bench_targets_out)

            bad = run(`$jcmd --startup-file=no --project=$project $script validate-ifs --input missing_file.ifs`; wait=false)
            wait(bad)
            @test !success(bad)
        finally
            cd(old)
        end
    end
end
