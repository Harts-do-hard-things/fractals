using Test
using LinearAlgebra
using StaticArrays
using Fractals
using Random
using PNGFiles: load, save
using Colors
using JSON3
using Printf

# The test bodies are huge one-shot closures (`_with_tmpdir() do ... end` full of `@test`s) that run
# exactly once, so optimizing/type-inferring them is pure compile-time overhead: most of the suite
# used to be spent in LLVM. The package itself is precompiled and unaffected by this setting.
Base.Experimental.@compiler_options optimize = 0 compile = min infer = false

# `mktempdir() do ... end` compiles a fresh `mktempdir` method instance for every test-body
# closure (~100 ms each, 35+ times). This helper takes the closure untyped, so it compiles once.
function _with_tmpdir(@nospecialize(f))
    dir = mktempdir()
    try
        return f(dir)
    finally
        try
            rm(dir; recursive = true, force = true)
        catch
        end
    end
end

# Route all package output into a scratch directory so tests never write into the DrWatson
# project's plots/.
const TEST_OUTPUT_ROOT = mktempdir(; prefix = "fractals_test_output_")
ENV["FRACTALS_OUTPUT_ROOT"] = TEST_OUTPUT_ROOT

# The CLI script, loaded once so CLI tests call it in-process instead of paying Julia startup
# and package loading for a subprocess per command.
module FractalsCLI
Base.Experimental.@compiler_options optimize = 0 compile = min infer = false
include(joinpath(@__DIR__, "..", "bin", "fractals.jl"))
end

if !isdefined(Main, :AnimateImageIterateScript)
    include(joinpath(@__DIR__, "..", "..", "scripts", "animate_image_iterate.jl"))
end

# Run `bin/fractals.jl` with `args` in-process, capturing output; `code` is the exit code the
# script would return (exceptions → 2).
function _run_cli(args::AbstractString...)
    _with_tmpdir() do tmp
        out_path = joinpath(tmp, "stdout")
        err_path = joinpath(tmp, "stderr")
        code = open(out_path, "w") do out
            open(err_path, "w") do err
                redirect_stdio(stdout = out, stderr = err) do
                    FractalsCLI.run_cli(String[args...])
                end
            end
        end
        return (code = code, stdout = read(out_path, String), stderr = read(err_path, String))
    end
end

function _env_flag(name::AbstractString)::Bool
    value = get(ENV, name, "")
    lowercase(strip(value)) in ("1", "true", "yes", "on")
end

const SMALL_EQ = [
    0.5 0.0 0.0 0.5 0.0 0.0 0.6
    -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
]
const SNAPSHOT_DIR = joinpath(@__DIR__, "snapshots")

function _probe_gpu_available()::Bool
    ifs = IFS(SMALL_EQ)
    try
        make_image([SVector{2,Float64}(0.0, 0.0)], ifs.limits; resolution = (4, 4), backend = :gpu)
        return true
    catch e
        e isa ArgumentError && return false
        rethrow()
    end
end

function _skip_optional_gpu_parity!(name::AbstractString)
    @warn "$name skipped: FRACTALS_RUN_GPU_TESTS=1 but no CUDA GPU is available."
    @test_skip "GPU available"
end

function _capture_exception(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

_snapshot_path(name::AbstractString) = joinpath(SNAPSHOT_DIR, name * ".png")

function _snapshot_diff_pixel(expected::Gray, actual::Gray)
    return Gray{Float32}(abs(Float32(expected) - Float32(actual)))
end

function _snapshot_diff_pixel(expected, actual)
    return RGBA{Float32}(
        abs(Float32(red(expected)) - Float32(red(actual))),
        abs(Float32(green(expected)) - Float32(green(actual))),
        abs(Float32(blue(expected)) - Float32(blue(actual))),
        1.0f0,
    )
end

function _write_snapshot_debug(name::AbstractString, expected, actual)
    debug_dir = mktempdir(prefix = "fractals_snapshot_")
    actual_path = joinpath(debug_dir, name * "_actual.png")
    diff_path = joinpath(debug_dir, name * "_diff.png")
    save(actual_path, actual)
    diff = [_snapshot_diff_pixel(expected[i], actual[i]) for i in eachindex(expected)]
    save(diff_path, reshape(diff, size(expected)))
    return debug_dir, actual_path, diff_path
end

@inline function _is_drawn_pixel(px)
    return alpha(px) > 0 && Float32(Gray(px)) > 0.0f0
end

function _assert_matches_snapshot(name::AbstractString, actual)
    path = _snapshot_path(name)
    isfile(path) || error("Missing snapshot fixture: $path")
    expected = load(path)
    encoded_actual = mktempdir(prefix = "fractals_snapshot_actual_") do tmp
        actual_path = joinpath(tmp, name * ".png")
        save(actual_path, actual)
        load(actual_path)
    end
    size(expected) == size(encoded_actual) || error(
        "Snapshot '$name' size mismatch: expected $(size(expected)), got $(size(encoded_actual))",
    )
    if expected != encoded_actual
        debug_dir, actual_path, diff_path = _write_snapshot_debug(name, expected, encoded_actual)
        error(
            "Snapshot '$name' mismatch. Debug artifacts written to $debug_dir (actual: $actual_path, diff: $diff_path)",
        )
    end
    return nothing
end

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

    for _ = 1:n_maps
        A = zeros(Float64, 2, 2)
        # Resample until we get a comfortably invertible matrix.
        while true
            A .= randn(rng, 2, 2)
            abs(det(A)) > det_eps && break
        end
        b = randn(rng, 2)
        m = AffineMap(A, b)
        mi = inv(m)

        for _ = 1:n_points_per_map
            x = SVector{2,Float64}(randn(rng), randn(rng))
            y = m(x)
            z = mi(x)
            @test mi(y) ≈ x atol = 1e-10 rtol = 1e-10
            @test m(z) ≈ x atol = 1e-10 rtol = 1e-10
        end
    end
end

@testset "IFS constructor initializes maps, weights, and bounds from equation matrices" begin
    ifs = IFS(SMALL_EQ)
    @test length(ifs.maps) == size(SMALL_EQ, 1)
    @test length(ifs.weights) == size(SMALL_EQ, 1)
    @test collect(ifs.weights) ≈ [0.6, 0.4]
    @test ifs.limits[1][1] < ifs.limits[1][2]
    @test ifs.limits[2][1] < ifs.limits[2][2]

    eq6 = [
        1.0 0.0 0.0 1.0 0.0 0.0
        0.5 0.0 0.0 0.5 1.0 0.0
    ]
    ifs6 = IFS(eq6; name = "Derived")
    @test ifs6.name == "Derived"
    @test fieldnames(IFS) == (:name, :maps, :weights, :limits)
    @test collect(ifs6.weights) ≈ [1.0, 0.25]
    @test ifs6.maps[1](SVector{2,Float64}(2.0, 3.0)) ≈ SVector{2,Float64}(2.0, 3.0)
    @test ifs6.maps[2](SVector{2,Float64}(2.0, 3.0)) ≈ SVector{2,Float64}(2.0, 1.5)
end

@testset "Random IFS Generation And Contractivity" begin
    eq1 = random_eq(; rng = MersenneTwister(20260324))
    eq2 = random_eq(; rng = MersenneTwister(20260324))
    @test eq1 == eq2
    @test size(eq1, 1) in 2:4
    @test size(eq1, 2) == 7
    @test isempty(Fractals._noncontractive_rows(eq1))
    @test isapprox(sum(eq1[:, 7]), 1.0; atol = 1e-6)
    @test all(eq1[:, 7] .> 0)

    eq6 = random_eq(; rng = MersenneTwister(7), nmaps = 3, with_probs = false)
    @test size(eq6) == (3, 6)
    @test isempty(Fractals._noncontractive_rows(eq6))

    ifs1 = random_ifs(; rng = MersenneTwister(99), name = "Random")
    ifs2 = random_ifs(; rng = MersenneTwister(99), name = "Random")
    @test ifs1.name == "Random"
    @test length(ifs1.maps) in 2:4
    @test collect(ifs1.weights) ≈ collect(ifs2.weights)
    @test all(
        Fractals._contractive_row(m.A[1, 1], m.A[1, 2], m.A[2, 1], m.A[2, 2]) for m in ifs1.maps
    )
end

@testset "Interpolation Helpers" begin
    left6 = [
        1.0 0.0 0.0 1.0 0.0 0.0
        0.5 0.0 0.0 0.5 1.0 0.0
    ]
    right7 = [
        0.0 1.0 -1.0 0.0 0.0 1.0 0.2
        1.0 0.0 0.0 1.0 2.0 0.0 0.8
    ]

    midpoint = interpolate_eq_matrix(left6, right7, 0.5)
    @test size(midpoint) == (2, 7)
    @test midpoint[:, 1:6] ≈ [
        sqrt(0.5) sqrt(0.5) -sqrt(0.5) sqrt(0.5) 0.0 0.5
        0.75 0.0 0.0 0.75 1.5 0.0
    ]
    @test midpoint[:, 7] ≈ [0.5, 0.5]
    @test isapprox(sum(midpoint[:, 7]), 1.0; atol = 1e-12)

    midpoint_linear = interpolate_eq_matrix(left6, right7, 0.5; interpolation_mode = :linear)
    @test midpoint_linear[:, 1:6] ≈ [
        0.5 0.5 -0.5 0.5 0.0 0.5
        0.75 0.0 0.0 0.75 1.5 0.0
    ]
    @test midpoint_linear[:, 7] ≈ [0.5, 0.5]

    @test interpolate_eq_matrix(left6, right7, 0.0) ≈ [
        1.0 0.0 0.0 1.0 0.0 0.0 0.8
        0.5 0.0 0.0 0.5 1.0 0.0 0.2
    ]
    @test interpolate_eq_matrix(left6, right7, 1.0) ≈ right7

    left_ifs = IFS(left6; name = "Left")
    right_ifs = IFS(right7; name = "Right")
    blended = interpolate_ifs(left_ifs, right_ifs, 0.25)

    @test blended.name == "Interpolated(Left -> Right)"
    @test blended.maps[1](SVector{2,Float64}(1.0, 0.0)) ≈
          SVector{2,Float64}(cospi(0.125), -sinpi(0.125) + 0.25)
    @test blended.maps[1](SVector{2,Float64}(0.0, 1.0)) ≈
          SVector{2,Float64}(sinpi(0.125), cospi(0.125) + 0.25)
    @test collect(blended.weights) ≈ [0.65, 0.35]
    @test blended.limits[1][1] ≈ 0.75 * left_ifs.limits[1][1] + 0.25 * right_ifs.limits[1][1]
    @test blended.limits[1][2] ≈ 0.75 * left_ifs.limits[1][2] + 0.25 * right_ifs.limits[1][2]
    @test blended.limits[2][1] ≈ 0.75 * left_ifs.limits[2][1] + 0.25 * right_ifs.limits[2][1]
    @test blended.limits[2][2] ≈ 0.75 * left_ifs.limits[2][2] + 0.25 * right_ifs.limits[2][2]

    blended_linear = interpolate_ifs(left_ifs, right_ifs, 0.25; interpolation_mode = :linear)
    @test blended_linear.maps[1](SVector{2,Float64}(1.0, 0.0)) ≈ SVector{2,Float64}(0.75, 0.0)
    @test blended_linear.maps[1](SVector{2,Float64}(0.0, 1.0)) ≈ SVector{2,Float64}(0.25, 1.0)

    right_locked = interpolate_ifs(left_ifs, right_ifs, 0.5; limits_mode = :right, name = "Blend")
    @test right_locked.name == "Blend"
    @test right_locked.limits == right_ifs.limits

    @test_throws ArgumentError interpolate_eq_matrix(left6, right7[:, 1:6], -0.1)
    @test_throws ArgumentError interpolate_eq_matrix(left6, ones(3, 6), 0.5)
    @test_throws ArgumentError interpolate_eq_matrix(left6, right7, 0.5; interpolation_mode = :bad)
    @test_throws ArgumentError interpolate_ifs(left_ifs, right_ifs, 0.5; limits_mode = :bad)
    @test_throws ArgumentError interpolate_ifs(left_ifs, right_ifs, 0.5; interpolation_mode = :bad)

    recomputed = interpolate_ifs(left_ifs, right_ifs, 0.5; limits_mode = :recompute)
    @test recomputed.limits == refresh_limits(recomputed).limits

    shear_left = [1.0 0.2 0.0 1.0 0.0 0.0]
    shear_right = [0.9 0.4 0.1 0.8 1.0 2.0]
    @test interpolate_eq_matrix(shear_left, shear_right, 0.5) ≈
          interpolate_eq_matrix(shear_left, shear_right, 0.5; interpolation_mode = :linear)
end

@testset "Limits Refresh And Staleness" begin
    ifs = IFS(SMALL_EQ; name = "Limits")
    original_limits = ifs.limits
    pts = chaos_game(ifs, 128; warmup = 5, seed = 4242)
    @test length(pts) == 128
    @test ifs.limits == original_limits

    # Limits are computed once at construction, and refresh_limits keeps everything else.
    refreshed = refresh_limits(ifs)
    @test refreshed.name == ifs.name
    @test refreshed.maps == ifs.maps
    @test collect(refreshed.weights) == collect(ifs.weights)
    @test refreshed.limits == original_limits

    # Limits that do not match the points change the raster; the maps' own limits fit.
    shifted_points = [p + SVector{2,Float64}(25.0, -17.0) for p in pts]
    stale_wide =
        IFS(ifs.maps, ifs.weights; name = ifs.name, limits = ((-100.0, 100.0), (-100.0, 100.0)))
    stale_img = make_image(shifted_points, stale_wide.limits; resolution = (32, 32), backend = :cpu)
    fitted_img =
        make_image(pts, refresh_limits(stale_wide).limits; resolution = (32, 32), backend = :cpu)
    @test stale_img != fitted_img
end

@testset "Interpolation Frame Renderer" begin
    left = IFS(
        [
            0.5 -0.5 0.5 0.5 0.0 0.0
            -0.5 -0.5 0.5 -0.5 1.0 0.0
        ];
        name = "left",
    )
    right = IFS(
        [
            0.5 0.0 0.0 0.5 0.0 0.0 0.3
            0.0 0.5 -0.5 0.0 1.0 0.0 0.7
        ];
        name = "right",
    )

    _with_tmpdir() do tmp
        result = render_interpolation_frames(
            left,
            right;
            frames = 3,
            outdir = tmp,
            basename = "anim",
            render_method = RenderTransformations,
            resolution = (24, 24),
            color = true,
            initial_polygon = :line_arrow,
            axis = true,
        )

        @test result.outdir == tmp
        @test result.render_method == :render_transformations
        @test result.paths == [
            joinpath(tmp, "anim_0001.png"),
            joinpath(tmp, "anim_0002.png"),
            joinpath(tmp, "anim_0003.png"),
        ]
        @test result.ts ≈ [0.0, 0.5, 1.0]
        @test all(isfile, result.paths)

        imgs = load.(result.paths)
        @test all(size(img) == (24, 24) for img in imgs)
    end

    _with_tmpdir() do tmp
        result = render_interpolation_frames(
            left,
            right;
            frames = 3,
            outdir = tmp,
            basename = "chaos_anim",
            render_method = Chaos,
            resolution = (24, 24),
            npoints = 250,
            warmup = 5,
        )

        @test result.outdir == tmp
        @test result.render_method == :chaos
        @test result.paths == [
            joinpath(tmp, "chaos_anim_0001.png"),
            joinpath(tmp, "chaos_anim_0002.png"),
            joinpath(tmp, "chaos_anim_0003.png"),
        ]
        @test result.ts ≈ [0.0, 0.5, 1.0]
        @test all(isfile, result.paths)

        imgs = load.(result.paths)
        @test all(size(img) == (24, 24) for img in imgs)
        @test any(any(_is_drawn_pixel, img) for img in imgs)

        prov = read_provenance(result.paths[2])
        @test prov.params.method == "interpolation_frame"
        @test prov.params.render_method == "chaos"
        @test prov.params.interpolation_mode == "rotation_scale"
        @test prov.params.t ≈ 0.5
        @test prov.params.npoints == 250
    end

    # Default output: an auto-named sequence directory under output_dir("frames").
    default_result = render_interpolation_frames(
        left,
        right;
        frames = 2,
        render_method = Chaos,
        resolution = (16, 16),
        npoints = 100,
        warmup = 2,
    )
    @test dirname(default_result.outdir) == output_dir("frames")
    @test startswith(basename(default_result.outdir), "left_to_right_")
    @test occursin("method=chaos", basename(default_result.outdir))
    @test all(isfile, default_result.paths)

    mismatch = IFS(
        [
            0.5 0.0 0.0 0.5 0.0 0.0
            0.5 0.0 0.0 0.5 0.5 0.0
            0.0 0.5 -0.5 0.0 1.0 0.0
        ];
        name = "mismatch",
    )

    _with_tmpdir() do tmp
        @test_throws ArgumentError render_interpolation_frames(
            left,
            right;
            frames = 0,
            outdir = tmp,
        )
        @test_throws ArgumentError render_interpolation_frames(
            left,
            right;
            frames = 2,
            outdir = tmp,
            basename = " ",
        )
        @test_throws ArgumentError render_interpolation_frames(
            left,
            mismatch;
            frames = 2,
            outdir = tmp,
        )

        for method in (PointDeterministic, ImageIterate, Inverse)
            err = _capture_exception(
                () -> render_interpolation_frames(
                    left,
                    right;
                    frames = 2,
                    outdir = tmp,
                    render_method = method,
                ),
            )
            @test err isa ArgumentError
            @test occursin(
                "supports method=Chaos or method=RenderTransformations",
                sprint(showerror, err),
            )
        end
    end
end

@testset "Animation Export Helpers" begin
    _with_tmpdir() do tmp
        missing_dir = joinpath(tmp, "missing_frames")
        err = _capture_exception(
            () -> export_animation(
                :gif;
                frames_dir = missing_dir,
                basename = "anim",
                outpath = joinpath(tmp, "out.gif"),
                ffmpeg_cmd = "definitely_missing_ffmpeg_binary",
            ),
        )
        @test err isa ArgumentError
        @test occursin("frames_dir", sprint(showerror, err))

        frames_dir = joinpath(tmp, "frames")
        mkpath(frames_dir)
        no_frames_err = _capture_exception(
            () -> export_animation(
                :gif;
                frames_dir = frames_dir,
                basename = "anim",
                outpath = joinpath(tmp, "out.gif"),
                ffmpeg_cmd = "definitely_missing_ffmpeg_binary",
            ),
        )
        @test no_frames_err isa ArgumentError
        @test occursin("No PNG frames matching", sprint(showerror, no_frames_err))

        for i = 1:3
            img = fill(Gray{Float32}(0.0f0), 12, 12)
            img[2+i:5+i, 3:6] .= Gray{Float32}(1.0f0)
            save(joinpath(frames_dir, @sprintf("anim_%04d.png", i)), img)
        end

        missing_ffmpeg = _capture_exception(
            () -> export_animation(
                :gif;
                frames_dir = frames_dir,
                basename = "anim",
                outpath = joinpath(tmp, "nested", "anim.gif"),
                ffmpeg_cmd = "definitely_missing_ffmpeg_binary",
            ),
        )
        @test missing_ffmpeg isa ArgumentError
        @test occursin("ffmpeg executable", sprint(showerror, missing_ffmpeg))

        start_gap_dir = joinpath(tmp, "gap_frames")
        mkpath(start_gap_dir)
        save(joinpath(start_gap_dir, "gap_0002.png"), fill(Gray{Float32}(1.0f0), 8, 8))
        gap_err = _capture_exception(
            () -> export_animation(
                :mp4;
                frames_dir = start_gap_dir,
                basename = "gap",
                outpath = joinpath(tmp, "gap.mp4"),
                ffmpeg_cmd = "definitely_missing_ffmpeg_binary",
            ),
        )
        @test gap_err isa ArgumentError
        @test occursin("must start at", sprint(showerror, gap_err))

        if !isnothing(Sys.which("ffmpeg"))
            gif_out = joinpath(tmp, "nested", "anim.gif")
            gif_result = export_animation(
                :gif;
                frames_dir = frames_dir,
                basename = "anim",
                outpath = gif_out,
                fps = 6,
            )
            @test gif_result.format == :gif
            @test isfile(gif_result.outpath)
            @test filesize(gif_result.outpath) > 0

            mp4_out = joinpath(tmp, "nested", "anim.mp4")
            mp4_result = export_animation(
                :mp4;
                frames_dir = frames_dir,
                basename = "anim",
                outpath = mp4_out,
                fps = 6,
            )
            @test mp4_result.format == :mp4
            @test isfile(mp4_result.outpath)
            @test filesize(mp4_result.outpath) > 0

            left = IFS(
                [
                    0.5 -0.5 0.5 0.5 0.0 0.0
                    -0.5 -0.5 0.5 -0.5 1.0 0.0
                ];
                name = "left",
            )
            right = IFS(
                [
                    0.5 0.0 0.0 0.5 0.0 0.0 0.3
                    0.0 0.5 -0.5 0.0 1.0 0.0 0.7
                ];
                name = "right",
            )
            generated_frames = render_interpolation_frames(
                left,
                right;
                frames = 3,
                outdir = joinpath(tmp, "generated_frames"),
                basename = "chaos_anim",
                render_method = Chaos,
                resolution = (24, 24),
                npoints = 250,
                warmup = 5,
            )
            generated_gif = export_animation(
                :gif;
                frames_dir = generated_frames.outdir,
                basename = "chaos_anim",
                outpath = joinpath(tmp, "nested", "chaos_anim.gif"),
                fps = 6,
            )
            @test generated_gif.format == :gif
            @test isfile(generated_gif.outpath)
            @test filesize(generated_gif.outpath) > 0
        else
            @test_skip "ffmpeg available"
        end
    end
end

@testset "ImageIterate Animation Script" begin
    script = Main.AnimateImageIterateScript

    _nontinted_pixels(img) = count(
        px ->
            abs(Float32(red(px)) - Float32(green(px))) < 1.0f-4 &&
                abs(Float32(green(px)) - Float32(blue(px))) < 1.0f-4,
        img,
    )
    _tinted_pixels(img) = count(
        px -> !(
            abs(Float32(red(px)) - Float32(green(px))) < 1.0f-4 &&
            abs(Float32(green(px)) - Float32(blue(px))) < 1.0f-4
        ),
        img,
    )

    _with_tmpdir() do tmp
        seed = fill(GrayA{Float32}(0.15f0, 0.0f0), 20, 20)
        seed[7:14, 7:14] .= GrayA{Float32}(0.85f0, 1.0f0)
        seed_path = joinpath(tmp, "seed_graya.png")
        save(seed_path, seed)

        ifs = IFS(SMALL_EQ)
        frames = script.render_image_iterate_build_frames(
            ifs,
            seed_path;
            frames_per_map = 2,
            outdir = joinpath(tmp, "frames"),
            basename = "image_iter_anim",
        )

        @test frames.basename == "image_iter_anim"
        @test frames.stage_ranges == [1:2, 3:4]
        @test length(frames.paths) == 4
        @test all(isfile, frames.paths)
        @test size(frames.final_image) == (20, 20)
        @test _tinted_pixels(frames.final_image) > 0
        @test read_provenance(first(frames.paths)).params.method == "image_iterate_build_frame"

        first_frame = load(first(frames.paths))
        last_frame = load(last(frames.paths))
        @test size(first_frame) == (20, 20)
        @test size(last_frame) == (20, 20)
        @test first_frame != last_frame
        @test _nontinted_pixels(first_frame) > 0

        # Default frames directory: output_dir("frames", basename).
        default_frames = script.render_image_iterate_build_frames(
            ifs,
            seed_path;
            frames_per_map = 1,
            basename = "default_anim",
        )
        @test default_frames.outdir == output_dir("frames", "default_anim")
        @test all(isfile, default_frames.paths)

        bad_seed = fill(Gray{Float32}(0.5f0), 8, 8)
        bad_seed_path = joinpath(tmp, "seed_gray.png")
        save(bad_seed_path, bad_seed)
        err = _capture_exception(
            () -> script.render_image_iterate_build_frames(
                ifs,
                bad_seed_path;
                frames_per_map = 2,
                outdir = joinpath(tmp, "bad_frames"),
                basename = "bad_anim",
            ),
        )
        @test err isa ArgumentError
        @test occursin("GrayA", sprint(showerror, err))

        if !isnothing(Sys.which("ffmpeg"))
            result = script.create_animation([
                "--input",
                ifs_data_dir("Default.ifs"),
                "--image-path",
                seed_path,
                "--frames-per-map",
                "1",
                "--out-dir",
                joinpath(tmp, "cli_frames"),
                "--basename",
                "cli_anim",
                "--out",
                joinpath(tmp, "cli_anim.gif"),
                "--fps",
                "6",
            ])
            @test isfile(result.gif.outpath)
            @test filesize(result.gif.outpath) > 0
            @test length(result.frames.paths) == 3
        else
            @test_skip "ffmpeg available"
        end
    end
end

@testset "iterate! and iterate_parallel! populate the IFS point cloud" begin
    ifs = IFS(SMALL_EQ)
    ifs_pts = chaos_game(ifs, 5000; warmup = 5)
    @test length(ifs_pts) == 5000

    ifs2 = IFS(SMALL_EQ)
    ifs2_pts = chaos_game(ifs2, 5000; warmup = 5)
    @test length(ifs2_pts) == 5000
end

@testset "Seeded Iterate Reproducibility" begin
    ifs1 = IFS(SMALL_EQ)
    ifs2 = IFS(SMALL_EQ)
    ifs3 = IFS(SMALL_EQ)

    ifs1_pts = chaos_game(ifs1, 1500; warmup = 10, seed = 12345)
    ifs2_pts = chaos_game(ifs2, 1500; warmup = 10, seed = 12345)
    ifs3_pts = chaos_game(ifs3, 1500; warmup = 10, seed = 54321)

    @test ifs1_pts == ifs2_pts
    @test ifs1_pts != ifs3_pts

    # Alias should preserve seeded behavior.
    ifs4 = IFS(SMALL_EQ)
    ifs5 = IFS(SMALL_EQ)
    ifs4_pts = chaos_game(ifs4, 1200; warmup = 8, seed = 99)
    ifs5_pts = chaos_game(ifs5, 1200; warmup = 8, seed = 99)
    @test ifs4_pts == ifs5_pts
end

@testset "Deterministic Iterate" begin
    function _safe_expand(points, maps, n)
        cur = copy(points)
        for _ = 1:n
            next = Vector{SVector{2,Float64}}(undef, length(cur) * length(maps))
            clen = length(cur)
            for i = 1:length(maps)
                start = (i - 1) * clen + 1
                stop = i * clen
                next[start:stop] = maps[i].(cur)
            end
            cur = next
        end
        return cur
    end

    ifs = IFS(SMALL_EQ)
    n = 2
    seeds = chaos_game(ifs, 20; warmup = 5, seed = 123)
    seeds_orig = copy(seeds)

    out = deterministic_iterate(ifs, seeds, n)
    @test length(out) == length(seeds) * length(ifs.maps)^n
    @test seeds == seeds_orig
    @test out == _safe_expand(seeds, ifs.maps, n)

    # For one round, each map block must be exactly map(points).
    one_round = deterministic_iterate(ifs, seeds, 1)
    base_len = length(seeds)
    for i = 1:length(ifs.maps)
        start = (i - 1) * base_len + 1
        stop = i * base_len
        @test one_round[start:stop] == ifs.maps[i].(seeds)
    end
end

@testset "make_pixelate_map produces an invertible coordinate transform into pixel space" begin
    limits = ((0.0, 2.0), (-1.0, 1.0))
    m = make_pixelate_map(limits; resolution = (100, 200))
    @test m isa AffineMap
    # Round-trip: mapping a coordinate to pixels and back recovers the original
    p = SVector{2,Float64}(1.0, 0.0)
    @test inv(m)(m(p)) ≈ p
    # The center of the limits maps into the interior of the image
    center = SVector{2,Float64}(1.0, 0.0)  # center of limits ((0.0,2.0), (-1.0,1.0))
    pixel = m(center)
    @test 1.0 <= pixel[1] <= 200.0
    @test 1.0 <= pixel[2] <= 100.0
end

@testset "Make Image Resolution Behavior" begin
    ifs = IFS(SMALL_EQ)
    ifs_pts = chaos_game(ifs, 4_000; warmup = 5, seed = 123)

    square = make_image(ifs_pts, ifs.limits; resolution = (32, 32))
    wide = make_image(ifs_pts, ifs.limits; resolution = (24, 40))
    tall = make_image(ifs_pts, ifs.limits; resolution = (40, 24))

    @test size(square) == (32, 32)
    @test size(wide) == (24, 40)
    @test size(tall) == (40, 24)
    @test maximum(square) > 0.0f0
    @test maximum(wide) > 0.0f0
    @test maximum(tall) > 0.0f0
    @test all(square .>= 0.0f0) && maximum(square) <= 1.0f0
    @test all(wide .>= 0.0f0) && maximum(wide) <= 1.0f0
    @test all(tall .>= 0.0f0) && maximum(tall) <= 1.0f0
end

@testset "make_image converts point cloud to a normalized Float32 grayscale heatmap" begin
    ifs = IFS(SMALL_EQ)
    ifs_pts = chaos_game(ifs, 2000; warmup = 5)
    img = make_image(ifs_pts, ifs.limits; resolution = (32, 32))
    @test size(img) == (32, 32)
    @test eltype(img) == Float32
    @test all(img .>= 0.0f0)
    @test maximum(img) <= 1.0f0
end

@testset "Make Image Backends" begin
    ifs = IFS(SMALL_EQ)
    ifs_pts = chaos_game(ifs, 2000; warmup = 5, seed = 123)

    img_default = make_image(ifs_pts, ifs.limits; resolution = (32, 32))
    img_cpu = make_image(ifs_pts, ifs.limits; resolution = (32, 32), backend = :cpu)
    @test img_default == img_cpu

    img_auto = make_image(ifs_pts, ifs.limits; resolution = (32, 32), backend = :auto)
    @test size(img_auto) == (32, 32)
    @test eltype(img_auto) == Float32
    @test all(img_auto .>= 0.0f0)
    @test maximum(img_auto) <= 1.0f0

    if _probe_gpu_available()
        img_gpu = make_image(ifs_pts, ifs.limits; resolution = (32, 32), backend = :gpu)
        @test size(img_gpu) == (32, 32)
        @test eltype(img_gpu) == Float32
        @test all(img_gpu .>= 0.0f0)
        @test maximum(img_gpu) <= 1.0f0
    else
        @test_throws ArgumentError make_image(
            ifs_pts,
            ifs.limits;
            resolution = (32, 32),
            backend = :gpu,
        )
    end

    @test_throws ArgumentError make_image(
        ifs_pts,
        ifs.limits;
        resolution = (32, 32),
        backend = :bad,
    )
end

@testset "Shared Backend Dispatch" begin
    @test Fractals._dispatch_backend(:cpu, () -> :cpu, () -> :gpu; gpu_available = false) == :cpu
    @test Fractals._dispatch_backend(:auto, () -> :cpu, () -> :gpu; gpu_available = false) == :cpu
    @test Fractals._dispatch_backend(:auto, () -> :cpu, () -> :gpu; gpu_available = true) == :gpu
    @test Fractals._dispatch_backend(
        :auto,
        () -> :cpu,
        () -> throw(ArgumentError("gpu failed"));
        gpu_available = true,
    ) == :cpu
    @test_throws DomainError Fractals._dispatch_backend(
        :auto,
        () -> :cpu,
        () -> throw(DomainError(:gpu, "boom"));
        gpu_available = true,
    )

    gpu_err = _capture_exception(
        () -> Fractals._dispatch_backend(:gpu, () -> :cpu, () -> :gpu; gpu_available = false),
    )
    @test gpu_err isa ArgumentError
    @test sprint(showerror, gpu_err) == sprint(showerror, Fractals._gpu_backend_unavailable_error())

    bad_err = _capture_exception(
        () -> Fractals._dispatch_backend(:bad, () -> :cpu, () -> :gpu; gpu_available = false),
    )
    @test bad_err isa ArgumentError
    @test occursin(
        "Invalid backend 'bad'. Supported: :cpu, :gpu, :auto",
        sprint(showerror, bad_err),
    )
end

@testset "GPU Parity (Optional)" begin
    if get(ENV, "FRACTALS_RUN_GPU_TESTS", "0") == "1"
        if _probe_gpu_available()
            ifs = IFS(SMALL_EQ)
            ifs_pts = chaos_game(ifs, 5000; warmup = 5, seed = 987)
            cpu_img = make_image(ifs_pts, ifs.limits; resolution = (48, 48), backend = :cpu)
            gpu_img = make_image(ifs_pts, ifs.limits; resolution = (48, 48), backend = :gpu)
            @test isapprox(cpu_img, gpu_img; atol = 1e-5, rtol = 1e-5)
        else
            _skip_optional_gpu_parity!("GPU Parity (Optional)")
        end
    else
        @test true
    end
end

@testset "Iterate Image" begin
    ifs = IFS(SMALL_EQ)
    ifs_pts = chaos_game(ifs, 2000; warmup = 5)
    img = make_image(ifs_pts, ifs.limits; resolution = (32, 32))
    out = iterate_image(ifs, img)
    out2 = iterate_image(ifs, out)
    @test size(out) == (32, 32)
    @test size(out2) == (32, 32)

    rgb_src = fill(RGB{Float32}(0.1f0, 0.2f0, 0.3f0), 8, 8)
    bad_src = _capture_exception(() -> iterate_image(ifs, rgb_src))
    @test bad_src isa ArgumentError
    @test occursin("grayscale source image", sprint(showerror, bad_src))
end

@testset "Iterate Image Colors" begin
    ifs = IFS(SMALL_EQ)
    ifs_pts = chaos_game(ifs, 1500; warmup = 5)
    img = make_image(ifs_pts, ifs.limits; resolution = (32, 32))

    out1 = iterate_image(ifs, img; colors = true, seed = 123)
    out2 = iterate_image(ifs, img; colors = true, seed = 123)

    @test size(out1) == (32, 32)
    @test eltype(out1) == RGB{Float32}
    @test out1 == out2
end

@testset "Iterate Image Backends" begin
    ifs = IFS(SMALL_EQ)
    ifs_pts = chaos_game(ifs, 2000; warmup = 5, seed = 321)
    src = make_image(ifs_pts, ifs.limits; resolution = (24, 24), backend = :cpu)

    out_default = iterate_image(ifs, src)
    out_cpu = iterate_image(ifs, src; backend = :cpu)
    @test out_default == out_cpu

    out_auto = iterate_image(ifs, src; backend = :auto)
    @test size(out_auto) == (24, 24)
    @test eltype(out_auto) == Gray{Float32}

    out_auto_color = iterate_image(ifs, src; colors = true, backend = :auto, seed = 777)
    @test size(out_auto_color) == (24, 24)
    @test eltype(out_auto_color) == RGB{Float32}

    if _probe_gpu_available()
        out_gpu = iterate_image(ifs, src; backend = :gpu)
        @test size(out_gpu) == (24, 24)
        @test eltype(out_gpu) == Gray{Float32}

        out_gpu_color = iterate_image(ifs, src; colors = true, backend = :gpu, seed = 777)
        @test size(out_gpu_color) == (24, 24)
        @test eltype(out_gpu_color) == RGB{Float32}
    else
        @test_throws ArgumentError iterate_image(ifs, src; backend = :gpu)
        @test_throws ArgumentError iterate_image(
            ifs,
            src;
            colors = true,
            backend = :gpu,
            seed = 777,
        )
    end

    @test_throws ArgumentError iterate_image(ifs, src; backend = :bad)
end

@testset "Iterate Image GPU Parity (Optional)" begin
    if get(ENV, "FRACTALS_RUN_GPU_TESTS", "0") == "1"
        if _probe_gpu_available()
            ifs = IFS(SMALL_EQ)
            ifs_pts = chaos_game(ifs, 2000; warmup = 5, seed = 4321)
            src = make_image(ifs_pts, ifs.limits; resolution = (28, 28), backend = :cpu)

            cpu_gray = iterate_image(ifs, src; backend = :cpu)
            gpu_gray = iterate_image(ifs, src; backend = :gpu)
            @test isapprox(Float32.(cpu_gray), Float32.(gpu_gray); atol = 1e-5, rtol = 1e-5)

            cpu_rgb = iterate_image(ifs, src; colors = true, backend = :cpu, seed = 99)
            gpu_rgb = iterate_image(ifs, src; colors = true, backend = :gpu, seed = 99)
            @test isapprox(
                Float32.(Gray.(cpu_rgb)),
                Float32.(Gray.(gpu_rgb));
                atol = 1e-5,
                rtol = 1e-5,
            )
        else
            _skip_optional_gpu_parity!("Iterate Image GPU Parity (Optional)")
        end
    else
        @test true
    end
end

@testset "Inverse Rasterize" begin
    ifs = IFS(SMALL_EQ)
    lims = ifs.limits
    img = rasterize_image_inversely(ifs, 2, lims; resolution = (8, 8))
    @test size(img) == (8, 8)
    @test eltype(img) == Float32
    @test all(img .>= 0.0f0)
    @test maximum(img) <= 1.0f0

    img_hide =
        rasterize_image_inversely(ifs, 2, lims; resolution = (8, 8), show_divergence_scale = false)
    @test size(img_hide) == (8, 8)
    @test eltype(img_hide) == Float32
    @test all(v -> v == 0.0f0 || v == 1.0f0, img_hide)
end

@testset "inverse_iterate returns a Float32 coverage value in [0, 1] for pixel corners" begin
    ifs = IFS(SMALL_EQ)
    p0 = SVector{2,Float64}(0.0, 0.0)
    p1 = SVector{2,Float64}(0.1, 0.0)
    p2 = SVector{2,Float64}(0.0, 0.1)
    result = inverse_iterate(ifs, 2, p0, p1, p2)
    @test result isa Float32
    @test 0.0f0 <= result <= 1.0f0
    # Deeper iteration returns a value in the same range
    result_deep = inverse_iterate(ifs, 4, p0, p1, p2)
    @test result_deep isa Float32
    @test 0.0f0 <= result_deep <= 1.0f0
end

@testset "Inverse Rasterize Backends" begin
    ifs = IFS(SMALL_EQ)
    lims = ifs.limits

    img_default = rasterize_image_inversely(ifs, 2, lims; resolution = (8, 8))
    img_cpu =
        rasterize_image_inversely(ifs, 2, lims; resolution = (8, 8), backend = :cpu, mode = :exact)
    @test img_default == img_cpu

    img_auto =
        rasterize_image_inversely(ifs, 2, lims; resolution = (8, 8), backend = :auto, mode = :exact)
    @test size(img_auto) == (8, 8)
    @test eltype(img_auto) == Float32
    @test all(img_auto .>= 0.0f0)
    @test maximum(img_auto) <= 1.0f0

    img_cpu_preview_a = rasterize_image_inversely(
        ifs,
        2,
        lims;
        resolution = (8, 8),
        backend = :cpu,
        mode = :preview,
    )
    img_cpu_preview_b = rasterize_image_inversely(
        ifs,
        2,
        lims;
        resolution = (8, 8),
        backend = :cpu,
        mode = :preview,
    )
    @test img_cpu_preview_a == img_cpu_preview_b

    if _probe_gpu_available()
        img_gpu_exact = rasterize_image_inversely(
            ifs,
            2,
            lims;
            resolution = (8, 8),
            backend = :gpu,
            mode = :exact,
        )
        @test size(img_gpu_exact) == (8, 8)
        @test eltype(img_gpu_exact) == Float32
        @test all(img_gpu_exact .>= 0.0f0)
        @test maximum(img_gpu_exact) <= 1.0f0

        img_gpu_preview = rasterize_image_inversely(
            ifs,
            2,
            lims;
            resolution = (8, 8),
            backend = :gpu,
            mode = :preview,
        )
        @test size(img_gpu_preview) == (8, 8)
        @test eltype(img_gpu_preview) == Float32
        @test all(img_gpu_preview .>= 0.0f0)
        @test maximum(img_gpu_preview) <= 1.0f0
    else
        @test_throws ArgumentError rasterize_image_inversely(
            ifs,
            2,
            lims;
            resolution = (8, 8),
            backend = :gpu,
            mode = :exact,
        )
        @test_throws ArgumentError rasterize_image_inversely(
            ifs,
            2,
            lims;
            resolution = (8, 8),
            backend = :gpu,
            mode = :preview,
        )
    end

    @test_throws ArgumentError rasterize_image_inversely(
        ifs,
        2,
        lims;
        resolution = (8, 8),
        backend = :bad,
    )
    @test_throws ArgumentError rasterize_image_inversely(
        ifs,
        2,
        lims;
        resolution = (8, 8),
        mode = :bad,
    )
end

@testset "Backend Dispatch Consistency" begin
    ifs = IFS(SMALL_EQ)
    ifs_pts = chaos_game(ifs, 2000; warmup = 5, seed = 2468)
    lims = ifs.limits
    src = make_image(ifs_pts, ifs.limits; resolution = (24, 24), backend = :cpu)

    bad_make = _capture_exception(
        () -> make_image(ifs_pts, ifs.limits; resolution = (24, 24), backend = :bad),
    )
    bad_iterate = _capture_exception(() -> iterate_image(ifs, src; backend = :bad))
    bad_inverse = _capture_exception(
        () -> rasterize_image_inversely(ifs, 2, lims; resolution = (8, 8), backend = :bad),
    )
    @test bad_make isa ArgumentError
    @test sprint(showerror, bad_make) == sprint(showerror, bad_iterate)
    @test sprint(showerror, bad_make) == sprint(showerror, bad_inverse)

    if _probe_gpu_available()
        make_auto = make_image(ifs_pts, ifs.limits; resolution = (24, 24), backend = :auto)
        iterate_auto = iterate_image(ifs, src; backend = :auto)
        inverse_auto = rasterize_image_inversely(ifs, 2, lims; resolution = (8, 8), backend = :auto)
        @test size(make_auto) == (24, 24)
        @test size(iterate_auto) == (24, 24)
        @test size(inverse_auto) == (8, 8)
    else
        gpu_make = _capture_exception(
            () -> make_image(ifs_pts, ifs.limits; resolution = (24, 24), backend = :gpu),
        )
        gpu_iterate = _capture_exception(() -> iterate_image(ifs, src; backend = :gpu))
        gpu_inverse = _capture_exception(
            () -> rasterize_image_inversely(ifs, 2, lims; resolution = (8, 8), backend = :gpu),
        )
        @test gpu_make isa ArgumentError
        @test sprint(showerror, gpu_make) == sprint(showerror, gpu_iterate)
        @test sprint(showerror, gpu_make) == sprint(showerror, gpu_inverse)

        @test make_image(ifs_pts, ifs.limits; resolution = (24, 24), backend = :auto) ==
              make_image(ifs_pts, ifs.limits; resolution = (24, 24), backend = :cpu)
        @test iterate_image(ifs, src; backend = :auto) == iterate_image(ifs, src; backend = :cpu)
        @test rasterize_image_inversely(ifs, 2, lims; resolution = (8, 8), backend = :auto) ==
              rasterize_image_inversely(ifs, 2, lims; resolution = (8, 8), backend = :cpu)
    end
end

@testset "Inverse Rasterize GPU Parity (Optional)" begin
    if get(ENV, "FRACTALS_RUN_GPU_TESTS", "0") == "1"
        if _probe_gpu_available()
            ifs = IFS(SMALL_EQ)
            lims = ifs.limits

            cpu_img = rasterize_image_inversely(
                ifs,
                2,
                lims;
                resolution = (16, 16),
                backend = :cpu,
                mode = :exact,
            )
            gpu_img = rasterize_image_inversely(
                ifs,
                2,
                lims;
                resolution = (16, 16),
                backend = :gpu,
                mode = :exact,
            )
            @test isapprox(cpu_img, gpu_img; atol = 1e-5, rtol = 1e-5)

            gpu_preview_a = rasterize_image_inversely(
                ifs,
                2,
                lims;
                resolution = (16, 16),
                backend = :gpu,
                mode = :preview,
            )
            gpu_preview_b = rasterize_image_inversely(
                ifs,
                2,
                lims;
                resolution = (16, 16),
                backend = :gpu,
                mode = :preview,
            )
            @test gpu_preview_a == gpu_preview_b
        else
            _skip_optional_gpu_parity!("Inverse Rasterize GPU Parity (Optional)")
        end
    else
        @test true
    end
end

# Lexing, parsing, contractivity checks and the IFS library live in IFSParse.jl (see its own
# tests); these cover only what Fractals adds on top: building `IFS` values.
@testset "IFS wrappers over IFSParse build IFS values" begin
    sample = """
    Test IFS {; Doc line 1
    ; Doc line 2
      0.5 0.0 0.0 0.5 0.0 0.0 0.6
     -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
    }
    """
    defs = parse_ifs_string(sample)
    @test length(defs) == 1
    @test defs[1] isa IFS
    @test defs[1].name == "Test IFS"
    @test length(defs[1].maps) == 2

    mixed = """
    ValidOne {
      0.5 0.0 0.0 0.5 0.0 0.0 1.0
    }
    InvalidOne {
      1.0 0.0 0.0 1.0 0.0 0.0 1.0
    }
    """
    kept =
        @test_logs (:warn, r"Skipping non-contractive IFS definition 'InvalidOne'") parse_ifs_string(
            mixed,
        )
    @test [x.name for x in kept] == ["ValidOne"]

    bad_eq = [1.0 0.0 0.0 1.0 0.0 0.0 1.0]
    err = try
        Fractals._validate_contractivity(bad_eq)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("non-contractive affine maps", sprint(showerror, err))
    @test occursin("row 1", sprint(showerror, err))

    _with_tmpdir() do tmp
        path = joinpath(tmp, "file.ifs")
        write(path, sample)
        from_file = parse_ifs_file(path)
        @test length(from_file) == 1
        @test from_file[1] isa IFS

        # get_ifs resolves by name, preferring prebuilt records and falling back to files.
        lib = joinpath(tmp, "lib")
        got = get_ifs("Test IFS"; dir = tmp, libdir = lib)
        @test got isa IFS
        @test got.name == "Test IFS"
        build_ifs_library(; dir = tmp, outdir = lib)
        got_prebuilt = get_ifs("Test IFS"; file = "file.ifs", dir = tmp, libdir = lib)
        @test got_prebuilt.maps == got.maps
        @test_throws ArgumentError get_ifs("Missing"; dir = tmp, libdir = lib)
    end
end

@testset "DrWatson Output Management" begin
    _with_tmpdir() do d
        plots = joinpath(abspath(d), "plots")

        @testset "output root precedence" begin
            withenv("FRACTALS_OUTPUT_ROOT" => nothing) do
                @test output_dir() == Fractals.plotsdir()
                @test output_dir("a", "b.png") == Fractals.plotsdir("a", "b.png")
            end
            withenv("FRACTALS_OUTPUT_ROOT" => d) do
                @test output_dir("a.png") == joinpath(plots, "a.png")
            end
            other = mktempdir()
            withenv("FRACTALS_OUTPUT_ROOT" => d) do
                with_output_root(other) do
                    @test output_dir() == joinpath(abspath(other), "plots")
                    @test data_output_dir() == joinpath(abspath(other), "data", "exp_pro")
                end
            end
            withenv("FRACTALS_OUTPUT_ROOT" => nothing) do
                @test data_output_dir("b.json") == Fractals.datadir("exp_pro", "b.json")
            end
        end

        @testset "DrWatson data layout" begin
            @test !isempty(parse_ifs_file(ifs_data_dir("Default.ifs")))

            with_output_root(d) do
                json1 = Fractals._save_data_text(joinpath("benchmarks", "x.json"), "{\"a\":1}")
                @test json1 == joinpath(abspath(d), "data", "exp_pro", "benchmarks", "x.json")
                @test read(json1, String) == "{\"a\":1}"
                @test Fractals._save_data_text(joinpath("data", "benchmarks", "x.json"), "{}") ==
                      json1
                @test read(joinpath(dirname(json1), "x_#1.json"), String) == "{\"a\":1}"
                absolute = joinpath(d, "elsewhere", "y.json")
                @test Fractals._save_data_text(absolute, "{}") == absolute
            end
        end

        with_output_root(d) do
            @testset "path resolution" begin
                out1 = render(
                    SMALL_EQ;
                    method = Chaos,
                    npoints = 500,
                    resolution = (16, 16),
                    outpath = "output.png",
                )
                @test out1.outpath == joinpath(plots, "output.png")
                @test isfile(out1.outpath)

                out2 = render(
                    SMALL_EQ;
                    method = Chaos,
                    npoints = 500,
                    resolution = (16, 16),
                    outpath = joinpath("nested", "x.png"),
                )
                @test out2.outpath == joinpath(plots, "nested", "x.png")
                @test isfile(out2.outpath)

                out3 = render(
                    SMALL_EQ;
                    method = Chaos,
                    npoints = 500,
                    resolution = (16, 16),
                    outpath = joinpath("plots", "y.png"),
                )
                @test out3.outpath == joinpath(plots, "y.png")

                absolute = joinpath(d, "elsewhere", "z.png")
                out4 = render(
                    SMALL_EQ;
                    method = Chaos,
                    npoints = 500,
                    resolution = (16, 16),
                    outpath = absolute,
                )
                @test out4.outpath == absolute
                @test isfile(absolute)
            end

            @testset "savename auto-naming" begin
                out = render(SMALL_EQ; method = Chaos, npoints = 500, resolution = (16, 24))
                @test dirname(out.outpath) == plots
                @test basename(out.outpath) == "ifs_method=chaos_npoints=500_resolution=16x24.png"

                named = IFS(SMALL_EQ, name = "My Fractal!")
                det = render(
                    named;
                    method = PointDeterministic,
                    iterations = 2,
                    color = true,
                    resolution = (16, 16),
                )
                @test basename(det.outpath) ==
                      "My_Fractal_color=true_iterations=2_method=point_deterministic_npoints=200_resolution=16x16.png"

                svg = render_transformations_svg(named; width = 32, height = 16)
                @test basename(svg) ==
                      "My_Fractal_initial_polygon=default_method=transformations_svg_resolution=16x32.svg"
                @test startswith(read(svg, String), "<!-- fractals.provenance: ")
            end

            @testset "safesave keeps backups" begin
                first_out = render(
                    SMALL_EQ;
                    method = Chaos,
                    npoints = 500,
                    resolution = (16, 16),
                    outpath = "kept.png",
                )
                first_bytes = read(first_out.outpath)
                second_out = render(
                    SMALL_EQ;
                    method = Chaos,
                    npoints = 600,
                    resolution = (16, 16),
                    outpath = "kept.png",
                )
                @test second_out.outpath == first_out.outpath
                backup = joinpath(plots, "kept_#1.png")
                @test isfile(backup)
                @test read(backup) == first_bytes
                @test read_provenance(second_out.outpath).params.npoints == 600
                @test read_provenance(backup).params.npoints == 500
            end

            @testset "provenance metadata" begin
                out = render(
                    SMALL_EQ;
                    method = Inverse,
                    iterations = 2,
                    resolution = (16, 16),
                    outpath = "prov.png",
                )
                prov = read_provenance(out.outpath)
                @test prov.version == 1
                @test prov.params.method == "inverse"
                @test prov.params.iterations == 2
                @test collect(prov.params.resolution) == [16, 16]
                @test prov.params.backend == "cpu"
                @test haskey(prov, :gitcommit)
                @test prov.fractals_version == string(pkgversion(Fractals))
                @test Fractals._read_png_source_limits(out.outpath) == out.ifs.limits
                @test isnothing(read_provenance(joinpath(d, "missing.png")))
            end
        end
    end
end

@testset "Affine Map SVG Rendering" begin
    ifs = IFS(SMALL_EQ)
    _with_tmpdir() do tmp
        suffix = randstring(8)
        svg_path = render_transformations_svg(
            ifs;
            outpath = joinpath(tmp, "maps_$suffix.svg"),
            width = 300,
            height = 300,
        )
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

        svg_axis_path = render_transformations_svg(
            ifs;
            outpath = joinpath(tmp, "maps_axis_$suffix.svg"),
            width = 300,
            height = 300,
            initial_polygon = :line,
            axis = true,
        )
        @test isfile(svg_axis_path)
        svg_axis_text = read(svg_axis_path, String)
        @test occursin("data-role=\"axis-x\"", svg_axis_text)
        @test occursin("data-role=\"axis-y\"", svg_axis_text)
        @test length(collect(eachmatch(r"data-role=\"axis-x-tick\"", svg_axis_text))) >= 2
        @test length(collect(eachmatch(r"data-role=\"axis-y-tick\"", svg_axis_text))) >= 2

        png_path = render_transformations_png(
            ifs;
            outpath = joinpath(tmp, "maps_$suffix.png"),
            width = 256,
            height = 256,
            alpha = true,
        )
        @test isfile(png_path)
        @test filesize(png_path) > 0
        png_img = load(png_path)
        @test alpha(png_img[1, 1]) == 0
        colored_nonzero = count(
            px -> _is_drawn_pixel(px) && (red(px) != green(px) || green(px) != blue(px)),
            png_img,
        )
        @test colored_nonzero > 0

        png_path_white = render_transformations_png(
            ifs;
            outpath = joinpath(tmp, "maps_white_$suffix.png"),
            width = 256,
            height = 256,
            color = false,
            show_base = true,
        )
        @test isfile(png_path_white)
        png_img_white = load(png_path_white)
        white_nonzero =
            count(px -> _is_drawn_pixel(px) && Float32(Gray(px)) == 1.0f0, png_img_white)
        @test white_nonzero > 0

        eq_identity = reshape([1.0 0.0 0.0 1.0 0.0 0.0 1.0], 1, 7)
        ifs_identity = IFS(eq_identity)
        png_no_axis = render_transformations_png(
            ifs_identity;
            outpath = joinpath(tmp, "maps_no_axis_$suffix.png"),
            width = 256,
            height = 256,
            initial_polygon = :line,
            color = false,
            axis = false,
        )
        png_with_axis = render_transformations_png(
            ifs_identity;
            outpath = joinpath(tmp, "maps_with_axis_$suffix.png"),
            width = 256,
            height = 256,
            initial_polygon = :line,
            color = false,
            axis = true,
        )
        @test isfile(png_no_axis)
        @test isfile(png_with_axis)
        img_no_axis = load(png_no_axis)
        img_with_axis = load(png_with_axis)
        no_axis_nonzero = count(_is_drawn_pixel, img_no_axis)
        with_axis_nonzero = count(_is_drawn_pixel, img_with_axis)
        @test with_axis_nonzero > no_axis_nonzero + 20
    end
end

@testset "Affine Map Color Assignment By Map" begin
    ifs = IFS(EISENSTEIN)
    _with_tmpdir() do tmp
        suffix = randstring(8)
        svg_path = render_transformations_svg(
            ifs;
            outpath = joinpath(tmp, "maps_colors_$suffix.svg"),
            width = 320,
            height = 320,
        )
        @test isfile(svg_path)

        svg_text = read(svg_path, String)
        counts = Dict{String,Int}()
        for m in eachmatch(r"stroke=\"(#[0-9A-Fa-f]{6})\"", svg_text)
            c = m.captures[1]
            counts[c] = get(counts, c, 0) + 1
        end

        @test length(counts) == length(ifs.maps)
        @test all(v > 0 for v in values(counts))
        @test length(unique(values(counts))) == 1
    end
end

@testset "Affine Map Rotation Orientation (30/45 CW/CCW)" begin
    function rotation_eq(degrees::Real; clockwise::Bool)::Matrix{Float64}
        θ = Float64(degrees) * (pi / 180.0)
        c = cos(θ)
        s = sin(θ)
        # Fractals eq row is [a11, a12, a21, a22, b1, b2, p] with
        # x' = a11*x + a21*y + b1, y' = a12*x + a22*y + b2.
        # Canonical convention: positive angle is counterclockwise in model space.
        row = clockwise ? [c -s s c 0.0 0.0 1.0] : [c s -s c 0.0 0.0 1.0]
        return reshape(row, 1, 7)
    end

    function line_endpoints_from_svg(svg_text::AbstractString)
        m = match(
            r"<line x1=\"([^\"]+)\" y1=\"([^\"]+)\" x2=\"([^\"]+)\" y2=\"([^\"]+)\"",
            svg_text,
        )
        m === nothing && error("No SVG line found")
        x1 = parse(Float64, m.captures[1])
        y1 = parse(Float64, m.captures[2])
        x2 = parse(Float64, m.captures[3])
        y2 = parse(Float64, m.captures[4])
        return x1, y1, x2, y2
    end

    cases = [
        (30.0, false, "30_ccw"),
        (30.0, true, "30_cw"),
        (45.0, false, "45_ccw"),
        (45.0, true, "45_cw"),
    ]

    _with_tmpdir() do tmp
        for (deg, clockwise, label) in cases
            eq = rotation_eq(deg; clockwise = clockwise)
            ifs = IFS(eq)

            p0 = SVector{2,Float64}(0.0, 0.0)
            p1 = SVector{2,Float64}(1.0, 0.0)
            q0 = ifs.maps[1](p0)
            q1 = ifs.maps[1](p1)
            dx_math = q1[1] - q0[1]
            dy_math = q1[2] - q0[2]
            @test dx_math > 0
            if clockwise
                @test dy_math > 0
            else
                @test dy_math < 0
            end

            svg_path = render_transformations_svg(
                ifs;
                outpath = joinpath(tmp, "rot_$label.svg"),
                width = 320,
                height = 320,
                initial_polygon = :line,
            )
            @test isfile(svg_path)
            svg_text = read(svg_path, String)
            x1, y1, x2, y2 = line_endpoints_from_svg(svg_text)
            dx_svg = x2 - x1
            dy_svg = y2 - y1
            @test dx_svg > 0
            @test signbit(dy_svg) != signbit(dy_math)

            png_path = render_transformations_png(
                ifs;
                outpath = joinpath(tmp, "rot_$label.png"),
                width = 320,
                height = 320,
                initial_polygon = :line,
                color = false,
            )
            @test isfile(png_path)
            img = load(png_path)
            @test count(_is_drawn_pixel, img) > 0
        end
    end
end

@testset "Transformation Origin Anchor Across Initial Polygons" begin
    presets = (:default, :equilateral_triangle, :line_arrow, :line)
    width, height = 320, 320
    ifs = IFS(reshape([1.0 0.0 0.0 1.0 0.0 0.0 1.0], 1, 7))
    _with_tmpdir() do tmp
        centers = Tuple{Float64,Float64}[]
        for preset in presets
            out = render_transformations_png(
                ifs;
                outpath = joinpath(tmp, "origin_$(preset).png"),
                width = width,
                height = height,
                initial_polygon = preset,
                color = false,
            )
            img = load(out)
            xs = Int[]
            ys = Int[]
            for y in axes(img, 1), x in axes(img, 2)
                if _is_drawn_pixel(img[y, x])
                    push!(xs, x)
                    push!(ys, y)
                end
            end
            isempty(xs) && error("Rendered PNG for preset $(preset) has no non-transparent pixels")
            push!(centers, ((minimum(xs) + maximum(xs)) / 2, (minimum(ys) + maximum(ys)) / 2))
        end
        ref_x, ref_y = centers[1]
        @test all(c -> abs(c[1] - ref_x) <= 1.0, centers)
        @test all(c -> abs(c[2] - ref_y) <= 1.0, centers)
    end
end

@testset "Transformation Raster Renders Follow Iterative Geometry" begin
    base_eq = reshape([1.0 0.0 0.0 1.0 0.0 0.0 1.0], 1, 7)
    shifted_eq = reshape([1.0 0.0 0.0 1.0 0.0 10.0 1.0], 1, 7)
    ifs = IFS(base_eq)
    shifted_ifs = IFS(shifted_eq)
    width, height = 320, 320

    _with_tmpdir() do tmp
        function rendered_bbox_center_and_span(out)
            img = load(out)
            xs = Int[]
            ys = Int[]
            for y in axes(img, 1), x in axes(img, 2)
                if _is_drawn_pixel(img[y, x])
                    push!(xs, x)
                    push!(ys, y)
                end
            end
            isempty(xs) && error("Rendered PNG has no non-transparent pixels")
            xcenter = (minimum(xs) + maximum(xs)) / 2
            ycenter = (minimum(ys) + maximum(ys)) / 2
            yspan = maximum(ys) - minimum(ys)
            return xcenter, ycenter, yspan
        end

        base_out = render_transformations_png(
            ifs;
            outpath = joinpath(tmp, "line_arrow_base.png"),
            width = width,
            height = height,
            initial_polygon = :line_arrow,
            color = false,
        )
        shifted_out = render_transformations_png(
            shifted_ifs;
            outpath = joinpath(tmp, "line_arrow_shifted.png"),
            width = width,
            height = height,
            initial_polygon = :line_arrow,
            color = false,
        )
        @test isfile(base_out)
        @test isfile(shifted_out)
        bx, by, bspan = rendered_bbox_center_and_span(base_out)
        sx, sy, sspan = rendered_bbox_center_and_span(shifted_out)
        @test bspan > 0
        @test sspan > 0
        @test (bx, by, bspan) != (sx, sy, sspan)
    end
end

@testset "Transformation SVG Uses PNG-Derived Limits" begin
    function line_endpoints_from_svg(svg_text::AbstractString)
        m = match(
            r"<line x1=\"([^\"]+)\" y1=\"([^\"]+)\" x2=\"([^\"]+)\" y2=\"([^\"]+)\"",
            svg_text,
        )
        m === nothing && error("No SVG line found")
        x1 = parse(Float64, m.captures[1])
        y1 = parse(Float64, m.captures[2])
        x2 = parse(Float64, m.captures[3])
        y2 = parse(Float64, m.captures[4])
        return (x1, y1), (x2, y2)
    end

    eq = reshape([cos(pi / 6) sin(pi / 6) -sin(pi / 6) cos(pi / 6) 0.0 0.0 1.0], 1, 7)
    ifs = IFS(eq)
    width, height = 320, 320

    _with_tmpdir() do tmp
        svg_path = render_transformations_svg(
            ifs;
            outpath = joinpath(tmp, "rot.svg"),
            width = width,
            height = height,
            initial_polygon = :line,
        )
        @test isfile(svg_path)
        svg_text = read(svg_path, String)
        got1, got2 = line_endpoints_from_svg(svg_text)
        @test got1 != got2

        png_path = render_transformations_png(
            ifs;
            outpath = joinpath(tmp, "rot.png"),
            width = width,
            height = height,
            initial_polygon = :line,
            color = false,
        )
        @test isfile(png_path)
        img = load(png_path)
        @test count(_is_drawn_pixel, img) > 0

        svg_legacy_path = render_transformations_svg(
            ifs;
            outpath = joinpath(tmp, "rot_legacy.svg"),
            width = width,
            height = height,
            initial_polygon = :line,
            polygon_limits_iterations = 0,
        )
        @test isfile(svg_legacy_path)
        svg_legacy_text = read(svg_legacy_path, String)
        legacy1, legacy2 = line_endpoints_from_svg(svg_legacy_text)

        @test (got1, got2) != (legacy1, legacy2)
    end
end

@testset "Transformation SVG Polygon Limits Iterations Affect Framing" begin
    eq = reshape([1.0 0.0 0.0 1.0 0.0 10.0 1.0], 1, 7)
    ifs = IFS(eq)

    _with_tmpdir() do tmp
        zero_iter_path = render_transformations_svg(
            ifs;
            outpath = joinpath(tmp, "limits0.svg"),
            width = 320,
            height = 320,
            initial_polygon = :line,
            polygon_limits_iterations = 0,
        )
        one_iter_path = render_transformations_svg(
            ifs;
            outpath = joinpath(tmp, "limits1.svg"),
            width = 320,
            height = 320,
            initial_polygon = :line,
            polygon_limits_iterations = 1,
        )
        @test isfile(zero_iter_path)
        @test isfile(one_iter_path)
        @test read(zero_iter_path, String) != read(one_iter_path, String)

        @test_throws ArgumentError render_transformations_svg(
            ifs;
            outpath = joinpath(tmp, "never_write_bad_limits.svg"),
            polygon_limits_iterations = -1,
        )
    end
end

@testset "Initial Polygon Presets" begin
    @test supported_initial_polygons() == [:default, :equilateral_triangle, :line_arrow, :line]

    for name in supported_initial_polygons()
        preset = initial_polygon(name)
        @test preset isa InitialPolygonPreset
        @test preset.name == name
        @test !isempty(preset.segments)
        @test length(preset.limits) == 2
    end

    err = try
        initial_polygon(:not_a_polygon)
        nothing
    catch ex
        ex
    end
    @test err isa ArgumentError
    @test occursin(
        "Supported: default, equilateral_triangle, line_arrow, line",
        sprint(showerror, err),
    )

    ifs = IFS(SMALL_EQ)
    _with_tmpdir() do tmp
        suffix = randstring(8)

        png_triangle = render_transformations_png(
            ifs;
            outpath = joinpath(tmp, "maps_triangle_$suffix.png"),
            width = 192,
            height = 192,
            initial_polygon = :equilateral_triangle,
            color = false,
        )
        @test isfile(png_triangle)
        tri_img = load(png_triangle)
        @test count(_is_drawn_pixel, tri_img) > 0

        png_line = render_transformations_png(
            ifs;
            outpath = joinpath(tmp, "maps_line_$suffix.png"),
            width = 192,
            height = 192,
            initial_polygon = :line,
            color = false,
        )
        @test isfile(png_line)
        line_img = load(png_line)
        @test count(_is_drawn_pixel, line_img) > 0

        default_preset = initial_polygon()
        png_default_preset = render_transformations_png(
            ifs;
            outpath = joinpath(tmp, "maps_default_preset_$suffix.png"),
            width = 192,
            height = 192,
            initial_polygon = default_preset,
            color = false,
        )
        @test isfile(png_default_preset)
        default_img = load(png_default_preset)
        @test count(_is_drawn_pixel, default_img) > 0

        @test_throws ArgumentError render_transformations_png(
            ifs;
            outpath = joinpath(tmp, "never_write_badpoly_$suffix.png"),
            initial_polygon = :not_a_polygon,
        )
        @test_throws MethodError render_transformations_png(
            ifs;
            outpath = joinpath(tmp, "never_write_badkw_$suffix.png"),
            limits_mode = :fit,
        )

        seed_line = render(
            SMALL_EQ;
            method = :image_iterate,
            image_source = :polygon,
            image_iterations = 1,
            initial_polygon = :line,
            resolution = (64, 64),
            outpath = joinpath(tmp, "seed_line_$suffix.png"),
        )
        seed_triangle = render(
            SMALL_EQ;
            method = :image_iterate,
            image_source = :polygon,
            image_iterations = 1,
            initial_polygon = :equilateral_triangle,
            resolution = (64, 64),
            outpath = joinpath(tmp, "seed_triangle_$suffix.png"),
        )
        @test count(>(Gray{Float32}(0)), seed_line.image) > 0
        @test count(>(Gray{Float32}(0)), seed_triangle.image) > 0
        @test seed_line.image != seed_triangle.image
    end
end

@testset "Render Entrypoint" begin
    _with_tmpdir() do tmp
        suffix = randstring(8)

        out1 = render(
            SMALL_EQ;
            npoints = 5000,
            method = :chaos,
            resolution = (64, 64),
            outpath = joinpath(tmp, "render_matrix_$suffix.png"),
        )
        @test isfile(out1.outpath)
        @test size(out1.image) == (64, 64)

        ifs = IFS(SMALL_EQ)
        out2 = render(
            ifs;
            method = :point_deterministic,
            iterations = 1,
            npoints = 2000,
            resolution = (48, 48),
            outpath = joinpath(tmp, "render_ifs_$suffix.png"),
        )
        @test isfile(out2.outpath)
        @test size(out2.image) == (48, 48)
        @test length(out2.points) == 2000 * length(ifs.maps)

        text = """
        RenderTest {
          0.5 0.0 0.0 0.5 0.0 0.0 0.6
         -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
        }
        """
        out3 = render(
            text;
            npoints = 4000,
            method = :parallel,
            resolution = (40, 40),
            outpath = joinpath(tmp, "render_text_$suffix.png"),
        )
        @test isfile(out3.outpath)
        @test size(out3.image) == (40, 40)
        @test out3.method == :chaos

        out3b = render(
            text;
            npoints = 2000,
            method = "point_deterministic",
            iterations = 1,
            resolution = (24, 24),
            outpath = joinpath(tmp, "render_text_string_method_$suffix.png"),
        )
        @test isfile(out3b.outpath)
        @test size(out3b.image) == (24, 24)
        @test out3b.method == :point_deterministic

        out3c = render(
            text;
            npoints = 1500,
            method = Parallel,
            resolution = (20, 20),
            outpath = joinpath(tmp, "render_text_enum_method_$suffix.png"),
        )
        @test isfile(out3c.outpath)
        @test size(out3c.image) == (20, 20)
        @test out3c.method == :chaos

        out_tf = render(
            text;
            npoints = 10,
            method = :render_transformations,
            resolution = (48, 48),
            initial_polygon = :line_arrow,
            color = true,
            outpath = joinpath(tmp, "render_transformations_$suffix.png"),
        )
        @test isfile(out_tf.outpath)
        @test size(out_tf.image) == (48, 48)
        @test out_tf.method == :render_transformations
        tf_nonzero = count(
            px -> _is_drawn_pixel(px) && (red(px) != green(px) || green(px) != blue(px)),
            out_tf.image,
        )
        @test tf_nonzero > 0

        mktemp() do path, io
            write(io, text)
            close(io)
            out4 = render(
                path;
                npoints = 3000,
                method = :inverse,
                ifs_index = 1,
                iterations = 2,
                resolution = (32, 32),
                outpath = joinpath(tmp, "render_file_$suffix.png"),
            )
            @test isfile(out4.outpath)
            @test size(out4.image) == (32, 32)

            out4_hide = render(
                path;
                npoints = 3000,
                method = :inverse,
                ifs_index = 1,
                iterations = 2,
                show_divergence_scale = false,
                resolution = (32, 32),
                outpath = joinpath(tmp, "render_file_hide_$suffix.png"),
            )
            @test isfile(out4_hide.outpath)
            @test size(out4_hide.image) == (32, 32)
            @test all(v -> v == 0.0f0 || v == 1.0f0, out4_hide.image)
        end
    end

    @test_throws ArgumentError render(SMALL_EQ; method = :badmethod)
    @test_throws ArgumentError render(SMALL_EQ; method = "not_a_method")
    @test_throws ArgumentError render(SMALL_EQ; method = :deterministic)

    # render() must delegate to render_*() — verify output parity for deterministic methods
    _with_tmpdir() do tmp
        res = (32, 32)
        ifs = IFS(SMALL_EQ)

        # Inverse is deterministic given the same IFS limits
        c = render(
            ifs;
            method = :inverse,
            iterations = 2,
            resolution = res,
            outpath = joinpath(tmp, "parity_inv_render.png"),
        )
        d = render_inverse(
            ifs;
            iterations = 2,
            resolution = res,
            outpath = joinpath(tmp, "parity_inv_direct.png"),
        )
        @test c.image == d.image
        @test c.method == d.method

        # RenderTransformations is fully deterministic
        e = render(
            ifs;
            method = :render_transformations,
            resolution = res,
            outpath = joinpath(tmp, "parity_tf_render.png"),
        )
        f = render_transformations(
            ifs;
            resolution = res,
            outpath = joinpath(tmp, "parity_tf_direct.png"),
        )
        @test e.image == f.image
        @test e.method == f.method
    end
end

@testset "Render Image Iterate API" begin
    _with_tmpdir() do tmp
        @info "Render Image Iterate API: basic source-path coverage"
        suffix = randstring(8)

        out_poly = render(
            SMALL_EQ;
            method = :image_iterate,
            image_source = :polygon,
            image_iterations = 2,
            resolution = (48, 48),
            outpath = joinpath(tmp, "render_image_poly_$suffix.png"),
        )
        @test isfile(out_poly.outpath)
        @test size(out_poly.image) == (48, 48)
        @test out_poly.method == :image_iterate
        @test eltype(out_poly.image) <: Gray

        out_chaos = render(
            SMALL_EQ;
            method = ImageIterate,
            image_source = :chaos,
            image_iterations = 2,
            npoints = 400,
            warmup = 5,
            resolution = (40, 40),
            outpath = joinpath(tmp, "render_image_chaos_$suffix.png"),
        )
        @test isfile(out_chaos.outpath)
        @test size(out_chaos.image) == (40, 40)

        p = joinpath(tmp, "seed.png")
        src = fill(Gray{Float32}(0.0f0), 20, 20)
        src[6:15, 6:15] .= Gray{Float32}(1.0f0)
        save(p, src)
        out_file = render(
            SMALL_EQ;
            method = :image_iterate,
            image_source = :file,
            image_path = p,
            image_iterations = 2,
            resolution = (20, 20),
            outpath = joinpath(tmp, "render_image_file_$suffix.png"),
        )
        @test isfile(out_file.outpath)
        @test size(out_file.image) == (20, 20)

        out_color = render(
            SMALL_EQ;
            method = :image_iterate,
            image_source = :polygon,
            image_iterations = 2,
            color = true,
            resolution = (24, 24),
            outpath = joinpath(tmp, "render_image_color_$suffix.png"),
        )
        @test isfile(out_color.outpath)
        @test size(out_color.image) == (24, 24)
        @test eltype(out_color.image) == RGB{Float32}

        rgb_path = joinpath(tmp, "seed_rgb.png")
        rgb_seed = fill(RGB{Float32}(0.0f0, 0.5f0, 1.0f0), 20, 20)
        save(rgb_path, rgb_seed)
        rgb_err = _capture_exception(
            () -> render(
                SMALL_EQ;
                method = :image_iterate,
                image_source = :file,
                image_path = rgb_path,
                image_iterations = 1,
                resolution = (20, 20),
                outpath = joinpath(tmp, "never_write_rgb_$suffix.png"),
            ),
        )
        @test rgb_err isa ArgumentError
        @test occursin("grayscale source image", sprint(showerror, rgb_err))

        graya_path = joinpath(tmp, "seed_graya.png")
        graya_seed = fill(GrayA{Float32}(0.0f0, 0.0f0), 20, 20)
        graya_seed[6:15, 6:15] .= GrayA{Float32}(1.0f0, 0.5f0)
        save(graya_path, graya_seed)
        out_graya = render(
            SMALL_EQ;
            method = :image_iterate,
            image_source = :file,
            image_path = graya_path,
            image_iterations = 1,
            resolution = (20, 20),
            outpath = joinpath(tmp, "render_image_graya_$suffix.png"),
        )
        @test isfile(out_graya.outpath)
        @test size(out_graya.image) == (20, 20)
        @test count(px -> Float32(gray(px)) > 0, out_graya.image) > 0

        direct_graya = iterate_image(IFS(SMALL_EQ), graya_seed)
        @test size(direct_graya) == (20, 20)
        @test eltype(direct_graya) <: Gray
        @test count(px -> Float32(gray(px)) > 0, direct_graya) > 0

        @test_throws ArgumentError render(
            SMALL_EQ;
            method = :image_iterate,
            image_source = :file,
            image_path = nothing,
            resolution = (24, 24),
            outpath = joinpath(tmp, "never_write2_$suffix.png"),
        )
        @test_throws ArgumentError render(
            SMALL_EQ;
            method = :image_iterate,
            polygon_limits_mode = :bad_mode,
            resolution = (24, 24),
            outpath = joinpath(tmp, "never_write3_$suffix.png"),
        )
        @test_throws ArgumentError render(
            SMALL_EQ;
            method = :image_iterate,
            polygon_limits_iterations = -1,
            resolution = (24, 24),
            outpath = joinpath(tmp, "never_write3b_$suffix.png"),
        )
        @test_throws ArgumentError render(
            SMALL_EQ;
            method = :image_iterate,
            image_source = :chaos,
            polygon_limits_iterations = 2,
            resolution = (24, 24),
            outpath = joinpath(tmp, "never_write3c_$suffix.png"),
        )
        @test_throws ArgumentError render(
            SMALL_EQ;
            method = :image_iterate,
            image_source = :file,
            image_path = p,
            show_base = true,
            resolution = (24, 24),
            outpath = joinpath(tmp, "never_write_show_base_$suffix.png"),
        )
        @test_throws ArgumentError render(
            SMALL_EQ;
            method = :image_iterate,
            image_source = :chaos,
            axis = true,
            resolution = (24, 24),
            outpath = joinpath(tmp, "never_write_axis_$suffix.png"),
        )
    end

    @testset "Polygon Limits Geometry Helpers" begin
        @info "Polygon Limits Geometry Helpers"

        function limits_spans(limits)
            (xmin, xmax), (ymin, ymax) = limits
            return xmax - xmin, ymax - ymin
        end

        rot_eq = reshape([cos(pi / 6) sin(pi / 6) -sin(pi / 6) cos(pi / 6) 0.0 0.0 1.0], 1, 7)
        rot_ifs = IFS(rot_eq)
        rot_limits_0 = Fractals._limits_from_polygon_point_iterations(
            rot_ifs;
            initial_polygon_spec = initial_polygon(:line_arrow),
            polygon_limits_iterations = 0,
        )
        rot_limits_1 = Fractals._limits_from_polygon_point_iterations(
            rot_ifs;
            initial_polygon_spec = initial_polygon(:line_arrow),
            polygon_limits_iterations = 1,
        )
        @test rot_limits_0 != rot_limits_1
        rot_dx, rot_dy = limits_spans(rot_limits_1)
        @test isapprox(rot_dx, rot_dy; atol = 1e-9)

        offset_eq = reshape([0.5 0.0 0.0 0.5 10.0 10.0 1.0], 1, 7)
        offset_ifs = IFS(offset_eq)
        offset_limits_1 = Fractals._limits_from_polygon_point_iterations(
            offset_ifs;
            initial_polygon_spec = initial_polygon(:line),
            polygon_limits_iterations = 1,
        )
        offset_limits_3 = Fractals._limits_from_polygon_point_iterations(
            offset_ifs;
            initial_polygon_spec = initial_polygon(:line),
            polygon_limits_iterations = 3,
        )
        @test offset_limits_1 != offset_limits_3
        offset_dx, offset_dy = limits_spans(offset_limits_3)
        @test isapprox(offset_dx, offset_dy; atol = 1e-9)

        arrow_limits = Fractals._limits_from_polygon_point_iterations(
            offset_ifs;
            initial_polygon_spec = initial_polygon(:line_arrow),
            polygon_limits_iterations = 2,
        )
        @test arrow_limits != offset_limits_3
        @test_throws ArgumentError Fractals._limits_from_polygon_point_iterations(
            offset_ifs;
            initial_polygon_spec = initial_polygon(:line),
            polygon_limits_iterations = -1,
        )
    end

    @testset "Polygon Raster IFS Builder" begin
        @info "Polygon Raster IFS Builder"

        function limits_spans(limits)
            (xmin, xmax), (ymin, ymax) = limits
            return xmax - xmin, ymax - ymin
        end

        edgar_path = ifs_data_dir("Edgar.ifs")
        edgar_defs = Fractals.parse_ifs_definitions_file(edgar_path)
        wreath_idx = findfirst(d -> String(d.name) == "Barnsleys Wreath", edgar_defs)
        wreath_idx === nothing && error("Barnsleys Wreath not found in Edgar.ifs")
        wreath_ifs = IFS(edgar_defs[wreath_idx].eq)

        wreath_raster_1 = Fractals._build_polygon_raster_ifs(
            wreath_ifs;
            initial_polygon_spec = initial_polygon(:line_arrow),
            polygon_limits_iterations = 1,
        )
        wreath_raster_2 = Fractals._build_polygon_raster_ifs(
            wreath_ifs;
            initial_polygon_spec = initial_polygon(:line_arrow),
            polygon_limits_iterations = 2,
        )

        @test wreath_raster_1.name == wreath_ifs.name
        @test wreath_raster_1.maps == wreath_ifs.maps
        @test wreath_raster_1.weights == wreath_ifs.weights
        @test wreath_raster_1.limits != wreath_ifs.limits
        @test wreath_raster_1.limits != wreath_raster_2.limits

        wreath_dx, wreath_dy = limits_spans(wreath_raster_2.limits)
        @test isapprox(wreath_dx, wreath_dy; atol = 1e-9)
    end

    @testset "Polygon Source Resolver Branching" begin
        @info "Polygon Source Resolver Branching"

        offset_eq = reshape([0.5 0.0 0.0 0.5 10.0 10.0 1.0], 1, 7)
        offset_ifs = IFS(offset_eq)

        @test Fractals._normalize_polygon_limits_mode(:geometry) == :geometry
        @test Fractals._normalize_polygon_limits_mode(:ifs) == :ifs
        @test_logs (:warn, r"polygon_limits_mode=:default is treated as :geometry") @test Fractals._normalize_polygon_limits_mode(
            :default,
        ) == :geometry

        geometry_raster_ifs = Fractals._build_polygon_raster_ifs(
            offset_ifs;
            initial_polygon_spec = initial_polygon(:line),
            polygon_limits_iterations = 1,
        )
        @test geometry_raster_ifs !== offset_ifs
        @test geometry_raster_ifs.name == offset_ifs.name
        @test geometry_raster_ifs.maps == offset_ifs.maps
        @test geometry_raster_ifs.weights == offset_ifs.weights
        @test geometry_raster_ifs.limits != offset_ifs.limits
    end

    @testset "Polygon Source Resolver Seed Smoke" begin
        @info "Polygon Source Resolver Seed Smoke"

        offset_eq = reshape([0.5 0.0 0.0 0.5 10.0 10.0 1.0], 1, 7)
        offset_ifs = IFS(offset_eq)

        geometry_source = Fractals._resolve_image_source_context(
            offset_ifs,
            :polygon,
            nothing,
            (32, 32),
            5,
            1,
            1,
            :geometry,
            1,
            true,
            initial_polygon(:line),
            :cpu,
        )
        @test geometry_source.ifs !== offset_ifs
        @test geometry_source.ifs.limits != offset_ifs.limits
        @test size(geometry_source.image) == (32, 32)
        @test eltype(geometry_source.image) == Float32
        @test count(>(0.0f0), geometry_source.image) > 0
    end

    @testset "Render Image Iterate API: polygon geometry public guard" begin
        _with_tmpdir() do tmp
            @info "Render Image Iterate API: polygon geometry public guard"
            rot_eq = reshape([cos(pi / 6) sin(pi / 6) -sin(pi / 6) cos(pi / 6) 0.0 0.0 1.0], 1, 7)
            rot_ifs = IFS(rot_eq)
            poly_seed = render(
                rot_ifs;
                method = :image_iterate,
                image_source = :polygon,
                image_iterations = 1,
                initial_polygon = :line_arrow,
                resolution = (96, 96),
                outpath = joinpath(tmp, "poly_seed.png"),
            )
            tf_reference = render(
                rot_ifs;
                method = :render_transformations,
                initial_polygon = :line_arrow,
                color = false,
                resolution = (96, 96),
                outpath = joinpath(tmp, "poly_reference.png"),
            )
            @test poly_seed.image == tf_reference.image
        end
    end

    @testset "Imported Python Data File" begin
        @info "Imported Python Data File"

        imported_path = ifs_data_dir("imported_from_python.ifs")
        imported_defs = Fractals.parse_ifs_definitions_file(imported_path)
        imported_names = [String(def.name) for def in imported_defs]

        @test imported_names == [
            "Flowsnake",
            "Heighway",
            "TwinDragon",
            "GoldenDragon",
            "Z2 Dragon",
            "LevyC",
            "Z2 Levy",
            "Terdragon",
            "Z2 Golden Dragon",
            "Pentigree",
            "Pentadentrite",
            "KochFlake",
            "Kochawave",
            "DurerPentagon",
        ]

        defs_by_name = Dict(String(def.name) => def for def in imported_defs)

        @test size(defs_by_name["Flowsnake"].eq) == (7, 6)
        @test size(defs_by_name["Heighway"].eq) == (2, 6)
        @test size(defs_by_name["Terdragon"].eq) == (3, 6)
        @test size(defs_by_name["Pentigree"].eq) == (6, 6)
        @test size(defs_by_name["Pentadentrite"].eq) == (6, 6)
        @test size(defs_by_name["DurerPentagon"].eq) == (4, 6)

        @test defs_by_name["Heighway"].eq ≈ [
            0.5 -0.5 0.5 0.5 0.0 0.0
            -0.5 -0.5 0.5 -0.5 1.0 0.0
        ]

        @test defs_by_name["Pentigree"].eq[1:2, :] ≈ [
            0.309016994375 -0.224513988290 0.224513988290 0.309016994375 0.0 0.0
            -0.118033988750 -0.363271264003 0.363271264003 -0.118033988750 0.309016994375 0.224513988290
        ]

        @test defs_by_name["Pentadentrite"].eq[4:5, :] ≈ [
            -0.233977061461 0.258209017515 -0.258209017515 -0.233977061461 0.719645709553 0.489158183710
            0.173268480388 0.302316383479 -0.302316383479 0.173268480388 0.485668648091 0.230949166195
        ]

        @test defs_by_name["Z2 Golden Dragon"].eq ≈ [
            0.623665256330 -0.403371701830 0.403371701830 0.623665256330 0.0 0.0
            -0.623665256330 0.403371701830 -0.403371701830 -0.623665256330 0.0 -0.0
            -0.376334743670 -0.403371701830 0.403371701830 -0.376334743670 1.0 0.0
            0.376334743670 0.403371701830 -0.403371701830 0.376334743670 -1.0 0.0
        ]
    end

    @testset "Render Image Iterate API: polygon overlays and alpha" begin
        _with_tmpdir() do tmp
            rot_eq = reshape([cos(pi / 6) sin(pi / 6) -sin(pi / 6) cos(pi / 6) 0.0 0.0 1.0], 1, 7)
            rot_ifs = IFS(rot_eq)
            base_off = render(
                rot_ifs;
                method = :image_iterate,
                image_source = :polygon,
                image_iterations = 1,
                initial_polygon = :line_arrow,
                show_base = false,
                resolution = (96, 96),
                outpath = joinpath(tmp, "poly_seed_base_off.png"),
            )
            base_on = render(
                rot_ifs;
                method = :image_iterate,
                image_source = :polygon,
                image_iterations = 1,
                initial_polygon = :line_arrow,
                show_base = true,
                resolution = (96, 96),
                outpath = joinpath(tmp, "poly_seed_base_on.png"),
            )
            @test base_on.image != base_off.image
            @test count(>(Gray{Float32}(0)), base_on.image) >
                  count(>(Gray{Float32}(0)), base_off.image)

            axis_off = render(
                rot_ifs;
                method = :image_iterate,
                image_source = :polygon,
                image_iterations = 1,
                initial_polygon = :line_arrow,
                axis = false,
                resolution = (96, 96),
                outpath = joinpath(tmp, "poly_seed_axis_off.png"),
            )
            axis_on = render(
                rot_ifs;
                method = :image_iterate,
                image_source = :polygon,
                image_iterations = 1,
                initial_polygon = :line_arrow,
                axis = true,
                resolution = (96, 96),
                outpath = joinpath(tmp, "poly_seed_axis_on.png"),
            )
            @test axis_on.image != axis_off.image
            @test count(>(Gray{Float32}(0)), axis_on.image) >
                  count(>(Gray{Float32}(0)), axis_off.image)

            alpha_overlay = render(
                rot_ifs;
                method = :image_iterate,
                image_source = :polygon,
                image_iterations = 1,
                initial_polygon = :line_arrow,
                show_base = true,
                axis = true,
                alpha = true,
                resolution = (96, 96),
                outpath = joinpath(tmp, "poly_seed_alpha_overlay.png"),
            )
            @test eltype(alpha_overlay.image) == GrayA{Float32}
            @test any(px -> alpha(px) == 0.0f0, alpha_overlay.image)
            @test count(_is_drawn_pixel, alpha_overlay.image) >
                  count(>(Gray{Float32}(0)), axis_off.image)
        end
    end

end

@testset "Image Source Resolution" begin
    _with_tmpdir() do tmp
        ifs = IFS(SMALL_EQ)
        res = (32, 32)

        polygon_img = Fractals._resolve_image_source(
            ifs,
            :polygon,
            nothing,
            res,
            5,
            1,
            2,
            :geometry,
            1,
            true,
            initial_polygon(:line),
            :cpu,
        )
        @test size(polygon_img) == res
        @test eltype(polygon_img) <: AbstractFloat
        @test count(>(0.0f0), polygon_img) > 0

        polygon_default =
            @test_logs (:warn, r"polygon_limits_mode=:default is treated as :geometry") Fractals._resolve_image_source(
                ifs,
                :polygon,
                nothing,
                res,
                5,
                1,
                2,
                :default,
                1,
                true,
                initial_polygon(:line),
                :cpu,
            )
        @test polygon_default == polygon_img

        chaos_img = Fractals._resolve_image_source(
            ifs,
            :chaos,
            nothing,
            res,
            5,
            1,
            2,
            :ifs,
            1,
            true,
            initial_polygon(),
            :cpu,
        )
        @test size(chaos_img) == res
        @test eltype(chaos_img) <: AbstractFloat

        deterministic_img = Fractals._resolve_image_source(
            ifs,
            :point_deterministic,
            nothing,
            res,
            5,
            1,
            2,
            :ifs,
            1,
            true,
            initial_polygon(),
            :cpu,
        )
        @test size(deterministic_img) == res
        @test eltype(deterministic_img) <: AbstractFloat

        inverse_img = Fractals._resolve_image_source(
            ifs,
            :inverse,
            nothing,
            res,
            5,
            1,
            2,
            :ifs,
            1,
            false,
            initial_polygon(),
            :cpu,
        )
        @test size(inverse_img) == res
        @test eltype(inverse_img) <: AbstractFloat

        src = fill(Gray{Float32}(0.0f0), res...)
        src[8:24, 8:24] .= Gray{Float32}(1.0f0)
        p = joinpath(tmp, "resolver_seed.png")
        save(p, src)
        file_img = Fractals._resolve_image_source(
            ifs,
            :file,
            p,
            res,
            5,
            1,
            2,
            :ifs,
            1,
            true,
            initial_polygon(),
            :cpu,
        )
        @test size(file_img) == res
        @test file_img == Float32.(Gray.(src))

        err = try
            Fractals._resolve_image_source(
                ifs,
                :bad_source,
                nothing,
                res,
                5,
                1,
                2,
                :ifs,
                1,
                true,
                initial_polygon(),
                :cpu,
            )
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin(
            "Supported: :polygon, :chaos, :point_deterministic, :inverse, :file",
            sprint(showerror, err),
        )
        @test_throws ArgumentError Fractals._resolve_image_source(
            ifs,
            :file,
            nothing,
            res,
            5,
            1,
            2,
            :ifs,
            1,
            true,
            initial_polygon(),
            :cpu,
        )
        @test_throws ArgumentError Fractals._resolve_image_source(
            ifs,
            :file,
            joinpath(tmp, "missing.png"),
            res,
            5,
            1,
            2,
            :ifs,
            1,
            true,
            initial_polygon(),
            :cpu,
        )
        @test_throws ArgumentError Fractals._resolve_image_source(
            ifs,
            :polygon,
            nothing,
            res,
            5,
            1,
            2,
            :bad_mode,
            1,
            true,
            initial_polygon(),
            :cpu,
        )
    end
end

@testset "PNG Source Limits Metadata" begin
    _with_tmpdir() do tmp
        res = (32, 32)
        src = fill(Gray{Float32}(0.0f0), res...)
        src[8:24, 8:24] .= Gray{Float32}(1.0f0)
        tagged_path = joinpath(tmp, "tagged.png")
        plain_path = joinpath(tmp, "plain.png")
        tagged_limits = ((-1.0, 1.0), (-1.0, 1.0))

        Fractals._save_image(tagged_path, src, tagged_limits)
        save(plain_path, src)

        @test isfile(tagged_path)
        @test Fractals._read_png_source_limits(tagged_path) == tagged_limits
        @test Fractals._read_png_source_limits(plain_path) === nothing
        @test load(tagged_path) == src

        ifs = IFS(reshape([0.5 0.0 0.0 0.5 0.0 10.0 1.0], 1, 7))
        tagged_ctx = Fractals._resolve_image_source_context(
            ifs,
            :file,
            tagged_path,
            res,
            5,
            1,
            2,
            :ifs,
            1,
            true,
            initial_polygon(),
            :cpu,
        )
        plain_ctx = Fractals._resolve_image_source_context(
            ifs,
            :file,
            plain_path,
            res,
            5,
            1,
            2,
            :ifs,
            1,
            true,
            initial_polygon(),
            :cpu,
        )

        @test tagged_ctx.ifs.limits == tagged_limits
        @test plain_ctx.ifs.limits == ifs.limits

        tagged_iter = iterate_image(tagged_ctx.ifs, tagged_ctx.image; backend = :cpu)
        plain_iter = iterate_image(plain_ctx.ifs, plain_ctx.image; backend = :cpu)
        @test tagged_iter != plain_iter
    end
end

@testset "Render Outputs Embed Source Limits Metadata" begin
    _with_tmpdir() do tmp
        chaos_out = render(
            SMALL_EQ;
            method = :chaos,
            npoints = 200,
            resolution = (32, 32),
            outpath = joinpath(tmp, "chaos_meta.png"),
        )
        @test Fractals._read_png_source_limits(chaos_out.outpath) == chaos_out.ifs.limits

        base_ifs = IFS(SMALL_EQ)
        transform_out = render(
            base_ifs;
            method = :render_transformations,
            initial_polygon = :line,
            polygon_limits_iterations = 2,
            resolution = (64, 64),
            outpath = joinpath(tmp, "transform_meta.png"),
        )
        expected_limits =
            Fractals._render_transformations_effective_ifs(
                transform_out.ifs;
                initial_polygon_spec = initial_polygon(:line),
                polygon_limits_iterations = 2,
            ).limits
        @test Fractals._read_png_source_limits(transform_out.outpath) == expected_limits
    end
end

@testset "Snapshot Image Tests" begin
    _with_tmpdir() do d
        old = pwd()
        cd(d)
        try
            transform = render(
                SMALL_EQ;
                npoints = 50,
                method = RenderTransformations,
                resolution = (96, 96),
                initial_polygon = :line_arrow,
                color = true,
                axis = true,
                outpath = "snapshot_transformations.png",
            )
            _assert_matches_snapshot("transformations-line-arrow-color-axis", transform.image)

            inverse = render(
                SMALL_EQ;
                method = Inverse,
                iterations = 2,
                show_divergence_scale = false,
                backend = :cpu,
                resolution = (32, 32),
                outpath = "snapshot_inverse.png",
            )
            _assert_matches_snapshot("inverse-mask-small", inverse.image)

            image_iter = render(
                SMALL_EQ;
                method = ImageIterate,
                image_source = :polygon,
                image_iterations = 2,
                backend = :cpu,
                resolution = (48, 48),
                outpath = "snapshot_image_iterate.png",
            )
            _assert_matches_snapshot("image-iterate-polygon-small", image_iter.image)
        finally
            cd(old)
        end
    end
end

@testset "Render IFS Selection By Index Or Name" begin
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
        _with_tmpdir() do tmp
            suffix = randstring(8)
            out_idx = render(
                path;
                ifs_index = 2,
                npoints = 1500,
                method = :chaos,
                resolution = (24, 24),
                outpath = joinpath(tmp, "render_multi_idx_$suffix.png"),
            )
            @test out_idx.ifs.name == "SecondIFS"
            @test isfile(out_idx.outpath)

            out_name = render(
                path;
                ifs_name = "FirstIFS",
                npoints = 1500,
                method = :chaos,
                resolution = (24, 24),
                outpath = joinpath(tmp, "render_multi_name_$suffix.png"),
            )
            @test out_name.ifs.name == "FirstIFS"
            @test isfile(out_name.outpath)

            @test_throws ArgumentError render(
                path;
                ifs_index = 99,
                npoints = 100,
                method = :chaos,
                resolution = (16, 16),
                outpath = joinpath(tmp, "never_written_$suffix.png"),
            )
            @test_throws ArgumentError render(
                path;
                ifs_name = "MissingIFS",
                npoints = 100,
                method = :chaos,
                resolution = (16, 16),
                outpath = joinpath(tmp, "never_written2_$suffix.png"),
            )
            @test_throws ArgumentError render(
                path;
                ifs_name = "FirstIFS",
                ifs_index = 1,
                npoints = 100,
                method = :chaos,
                resolution = (16, 16),
                outpath = joinpath(tmp, "never_written3_$suffix.png"),
            )
        end
    end
end

@testset "Validation And Errors" begin
    bad5 = ones(2, 5)
    err = try
        IFS(bad5)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("6 or 7 columns", sprint(showerror, err))

    bad8 = ones(2, 8)
    err = try
        IFS(bad8)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("6 or 7 columns", sprint(showerror, err))

    empty_eq = Matrix{Float64}(undef, 0, 6)
    err = try
        IFS(empty_eq)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("at least one row", sprint(showerror, err))

    bad_nan = copy(SMALL_EQ)
    bad_nan[1, 1] = NaN
    err = try
        IFS(bad_nan)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("non-finite values", sprint(showerror, err))

    bad_prob_neg = copy(SMALL_EQ)
    bad_prob_neg[1, 7] = -0.1
    err = try
        IFS(bad_prob_neg)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("nonnegative", sprint(showerror, err))

    bad_prob_zero = copy(SMALL_EQ)
    bad_prob_zero[:, 7] .= 0.0
    err = try
        IFS(bad_prob_zero)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("positive total weight", sprint(showerror, err))

    @test_throws ArgumentError render(
        SMALL_EQ;
        method = :chaos,
        backend = :bad,
        resolution = (24, 24),
        outpath = "never_write_backend_bad.png",
    )
end

@testset "Interactive And Main Entrypoints" begin
    sample = """
    PromptIFS {
      0.5 0.0 0.0 0.5 0.0 0.0 0.6
     -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
    }
    """

    _with_tmpdir() do d
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
                    prompt_ifs_and_render(
                        ifs_path;
                        npoints = 100,
                        resolution = (32, 32),
                        outpath = "prompt_output.png",
                    )
                end
            end
            @test selected isa IFS
            @test isfile(output_dir("prompt_output.png"))

            Fractals.main(eq = SMALL_EQ, npoints = 1000, outpath = "main_output.png")
            @test isfile(output_dir("main_output.png"))

            @testset "_prompt_int enforces min and narrows catch" begin
                @test Fractals._prompt_int("p", 5; input_fn = () -> "0", min = 1) == 5
                @test Fractals._prompt_int("p", 5; input_fn = () -> "-3", min = 1) == 5
                @test Fractals._prompt_int("p", 5; input_fn = () -> "abc") == 5
                @test Fractals._prompt_int("p", 5; input_fn = () -> "7", min = 1) == 7
                @test Fractals._prompt_choice("p", 3, 1; input_fn = () -> "xyz") == 1
            end

            @testset "Zero width falls back to default, no crash" begin
                queued = ["", "", "", "0", "", ""]
                idx = Ref(0)
                fake_input = () -> (idx[] += 1; idx[] <= length(queued) ? queued[idx[]] : "")
                zero_selected = prompt_ifs_and_render(
                    ifs_path;
                    npoints = 100,
                    resolution = (32, 32),
                    outpath = "prompt_zero_width.png",
                    input_fn = fake_input,
                )
                @test zero_selected isa IFS
                out_path = output_dir("prompt_zero_width.png")
                @test isfile(out_path)
                @test size(load(out_path)) == (32, 32)
            end
        finally
            cd(old)
        end
    end
end

@testset "Render Color API" begin
    _with_tmpdir() do d
        old = pwd()
        cd(d)
        try
            out_chaos = render(
                SMALL_EQ;
                method = Chaos,
                npoints = 800,
                warmup = 5,
                color = true,
                resolution = (40, 40),
                outpath = "render_chaos_color.png",
            )
            @test size(out_chaos.image) == (40, 40)
            @test eltype(out_chaos.image) == RGB{Float32}
            @test isfile(output_dir("render_chaos_color.png"))

            out_det = render(
                SMALL_EQ;
                method = PointDeterministic,
                npoints = 40,
                iterations = 1,
                color = true,
                resolution = (40, 40),
                outpath = "render_det_color.png",
            )
            @test size(out_det.image) == (40, 40)
            @test eltype(out_det.image) == RGB{Float32}
            @test isfile(output_dir("render_det_color.png"))

            out_inv = render(
                SMALL_EQ;
                method = Inverse,
                iterations = 2,
                color = true,
                resolution = (40, 40),
                outpath = "render_inv_color.png",
            )
            @test size(out_inv.image) == (40, 40)
            @test eltype(out_inv.image) == RGB{Float32}
            @test isfile(output_dir("render_inv_color.png"))

            out_tf = render(
                SMALL_EQ;
                method = RenderTransformations,
                npoints = 40,
                color = true,
                resolution = (40, 40),
                outpath = "render_tf_color.png",
            )
            @test size(out_tf.image) == (40, 40)
            @test eltype(out_tf.image) == RGB{Float32}
            @test isfile(output_dir("render_tf_color.png"))
        finally
            cd(old)
        end
    end
end

@testset "Render Alpha API" begin
    _with_tmpdir() do d
        old = pwd()
        cd(d)
        try
            methods = (
                (Chaos, (; npoints = 800, warmup = 5)),
                (PointDeterministic, (; npoints = 40, iterations = 1)),
                (Inverse, (; iterations = 2)),
                (ImageIterate, (; image_source = :polygon, image_iterations = 2)),
                (RenderTransformations, (; npoints = 40)),
            )

            for (method, extra) in methods
                gray_out = render(
                    SMALL_EQ;
                    method = method,
                    color = false,
                    alpha = true,
                    resolution = (40, 40),
                    outpath = "alpha_gray_$(method).png",
                    extra...,
                )
                @test size(gray_out.image) == (40, 40)
                @test eltype(gray_out.image) == GrayA{Float32}
                @test any(px -> alpha(px) == 0.0f0, gray_out.image)
                @test any(_is_drawn_pixel, gray_out.image)

                color_out = render(
                    SMALL_EQ;
                    method = method,
                    color = true,
                    alpha = true,
                    resolution = (40, 40),
                    outpath = "alpha_color_$(method).png",
                    extra...,
                )
                @test size(color_out.image) == (40, 40)
                @test eltype(color_out.image) == RGBA{Float32}
                @test any(px -> alpha(px) == 0.0f0, color_out.image)
                @test any(_is_drawn_pixel, color_out.image)
            end
        finally
            cd(old)
        end
    end
end

# Opt-in: CLI tests run in-process (see _run_cli), so their cost is first-call compilation of
# the CLI code paths rather than a Julia subprocess per command.
if _env_flag("FRACTALS_RUN_CLI_BENCH_TESTS")
    @testset "CLI Commands" begin
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
        invalid_sample = """
        CLI_Good {
          0.5 0.0 0.0 0.5 0.0 0.0 1.0
        }
        CLI_Bad {
          1.0 0.0 0.0 1.0 0.0 0.0 1.0
        }
        """

        _with_tmpdir() do d
            ifs_path = joinpath(d, "cli.ifs")
            write(ifs_path, sample)
            invalid_ifs_path = joinpath(d, "cli_invalid.ifs")
            write(invalid_ifs_path, invalid_sample)

            @testset "help and unknown commands" begin
                help = _run_cli("help")
                @test help.code == 0
                @test occursin("batch-render", help.stdout)
                @test _run_cli().code == 1
                unknown = _run_cli("no-such-command")
                @test unknown.code == 2
                @test occursin("Unknown command", unknown.stderr)
            end

            @testset "validate-ifs" begin
                valid = _run_cli("validate-ifs", "--input", ifs_path)
                @test valid.code == 0
                @test occursin("Definitions: 2", valid.stdout)
                @test occursin("CLI_First", valid.stdout)

                invalid = _run_cli("validate-ifs", "--input", invalid_ifs_path)
                @test invalid.code == 2
                @test occursin("Invalid definitions: 1", invalid.stderr)
                @test occursin("CLI_Bad", invalid.stderr)

                @test _run_cli("validate-ifs", "--input", joinpath(d, "missing_file.ifs")).code == 2
            end

            @testset "render" begin
                single = _run_cli(
                    "render",
                    "--input",
                    ifs_path,
                    "--ifs-index",
                    "2",
                    "--npoints",
                    "2000",
                    "--resolution",
                    "32x32",
                    "--out",
                    "cli_single.png",
                )
                @test single.code == 0
                @test occursin("Rendered", single.stdout)
                @test isfile(output_dir("cli_single.png"))

                auto = _run_cli(
                    "render",
                    "--input",
                    ifs_path,
                    "--ifs-index",
                    "1",
                    "--npoints",
                    "500",
                    "--resolution",
                    "16x16",
                )
                @test auto.code == 0
                @test isfile(output_dir("CLI_First_method=chaos_npoints=500_resolution=16x16.png"))

                tf = _run_cli(
                    "render",
                    "--input",
                    ifs_path,
                    "--ifs-index",
                    "1",
                    "--method",
                    "RenderTransformations",
                    "--color",
                    "true",
                    "--initial-polygon",
                    "line_arrow",
                    "--resolution",
                    "24x24",
                    "--out",
                    "cli_transformations.png",
                )
                @test tf.code == 0
                tf_img = load(output_dir("cli_transformations.png"))
                @test count(
                    px -> _is_drawn_pixel(px) && (red(px) != green(px) || green(px) != blue(px)),
                    tf_img,
                ) > 0

                inverse = _run_cli(
                    "render",
                    "--input",
                    ifs_path,
                    "--ifs-index",
                    "1",
                    "--method",
                    "Inverse",
                    "--iterations",
                    "2",
                    "--show-divergence-scale",
                    "false",
                    "--resolution",
                    "24x24",
                    "--out",
                    "cli_inverse_mask.png",
                )
                @test inverse.code == 0
                inverse_img = load(output_dir("cli_inverse_mask.png"))
                @test eltype(inverse_img) <: Gray
                @test all(px -> Float32(px.val) == 0 || Float32(px.val) == 1, inverse_img)
            end

            @testset "batch-render" begin
                batch = _run_cli(
                    "batch-render",
                    "--input",
                    ifs_path,
                    "--method",
                    "Inverse",
                    "--iterations",
                    "2",
                    "--show-divergence-scale",
                    "false",
                    "--resolution",
                    "24x24",
                    "--out-dir",
                    "batch",
                )
                @test batch.code == 0
                @test occursin("Batch rendering 2 definitions", batch.stdout)
                @test isfile(output_dir("batch", "01_cli_first.png"))
                @test isfile(output_dir("batch", "02_cli_second.png"))
                batch_img = load(output_dir("batch", "01_cli_first.png"))
                @test eltype(batch_img) <: Gray
                @test all(px -> Float32(px.val) == 0 || Float32(px.val) == 1, batch_img)

                batch_tf = _run_cli(
                    "batch-render",
                    "--input",
                    ifs_path,
                    "--method",
                    "RenderTransformations",
                    "--color",
                    "true",
                    "--initial-polygon",
                    "line_arrow",
                    "--resolution",
                    "24x24",
                    "--out-dir",
                    "batch_transformations",
                )
                @test batch_tf.code == 0
                batch_tf_img = load(output_dir("batch_transformations", "01_cli_first.png"))
                @test count(
                    px -> _is_drawn_pixel(px) && (red(px) != green(px) || green(px) != blue(px)),
                    batch_tf_img,
                ) > 0

                batch_ii = _run_cli(
                    "batch-render",
                    "--input",
                    ifs_path,
                    "--method",
                    "ImageIterate",
                    "--image-source",
                    "polygon",
                    "--image-iterations",
                    "3",
                    "--color",
                    "true",
                    "--resolution",
                    "24x24",
                    "--out-dir",
                    "batch_image_iterate",
                )
                @test batch_ii.code == 0
                @test isfile(output_dir("batch_image_iterate", "01_cli_first.png"))
                @test isfile(output_dir("batch_image_iterate", "02_cli_second.png"))
                batch_ii_img = load(output_dir("batch_image_iterate", "01_cli_first.png"))
                @test eltype(batch_ii_img) <: RGB
                @test count(px -> gray(Gray(px)) > 0, batch_ii_img) > 0
            end
        end
    end

    @testset "CLI Benchmarks" begin
        _with_tmpdir() do d
            bench_args = (
                "benchmark",
                "--profile",
                "small",
                "--repeats",
                "1",
                "--npoints",
                "500",
                "--resolution",
                "12x12",
                "--inverse-iterations",
                "1",
            )

            bench = _run_cli(bench_args...)
            @test bench.code == 0
            @test occursin("Benchmark suite", bench.stdout)
            @test occursin("iterate!", bench.stdout)
            @test occursin("alloc_mean=", bench.stdout)

            bench_json = joinpath(d, "benchmarks", "bench_small.json")
            bench_with_json = _run_cli(bench_args..., "--json", bench_json)
            @test bench_with_json.code == 0
            @test occursin("Wrote benchmark JSON", bench_with_json.stdout)
            @test isfile(bench_json)

            # Relative --json paths land in DrWatson's data/exp_pro.
            bench_rel = _run_cli(bench_args..., "--json", joinpath("benchmarks", "rel.json"))
            @test bench_rel.code == 0
            rel_json = data_output_dir("benchmarks", "rel.json")
            @test isfile(rel_json)
            @test occursin(rel_json, bench_rel.stdout)

            json_txt = read(bench_json, String)
            @test occursin("\"results\"", json_txt)
            @test occursin("\"small\"", json_txt)

            payload = JSON3.read(json_txt)
            @test haskey(payload.results, :small)
            small = payload.results.small
            for k in (
                :iterate!,
                Symbol("iterate_parallel!"),
                :make_image,
                :iterate_image,
                :rasterize_image_inversely,
            )
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

            bench_gpu_json = joinpath(d, "benchmarks", "bench_small_gpu_lines.json")
            bench_gpu = _run_cli(
                bench_args...,
                "--backend",
                "auto",
                "--include-gpu-bench",
                "--json",
                bench_gpu_json,
            )
            @test bench_gpu.code == 0
            @test occursin("Wrote benchmark JSON", bench_gpu.stdout)
            gpu_small = JSON3.read(read(bench_gpu_json, String)).results.small
            for k in (
                :make_image_gpu,
                :iterate_image_gpu,
                :rasterize_image_inversely_gpu_exact,
                :rasterize_image_inversely_gpu_preview,
            )
                @test haskey(gpu_small, k)
                item = gpu_small[k]
                if haskey(item, :status)
                    @test item.status == "skipped"
                    @test haskey(item, :reason)
                else
                    @test haskey(item, :mean_s)
                    @test haskey(item, :mean_alloc_bytes)
                end
            end

            targets_path = abspath(joinpath(@__DIR__, "..", "bench", "perf_targets.toml"))
            bench_targets = _run_cli(bench_args..., "--targets", targets_path)
            @test bench_targets.code == 0
            @test occursin("Target comparison", bench_targets.stdout)
            @test occursin("targets:", bench_targets.stdout)
        end
    end
else
    @info "Skipping CLI Commands and CLI Benchmarks testsets; set FRACTALS_RUN_CLI_BENCH_TESTS=1 to enable them."
end
include("test_jld2.jl")
