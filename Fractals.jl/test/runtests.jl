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

if !isdefined(Main, :AnimateImageIterateScript)
    include(joinpath(@__DIR__, "..", "..", "scripts", "animate_image_iterate.jl"))
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
            right,
            Transformations(polygon = :line_arrow);
            frames = 3,
            outdir = tmp,
            basename = "anim",
            resolution = (24, 24),
            color = true,
            overlays = [Axes()],
        )

        @test result.outdir == tmp
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

    # The method is optional (Transformations by default).
    _with_tmpdir() do tmp
        result = render_interpolation_frames(
            left,
            right;
            frames = 1,
            outdir = tmp,
            resolution = (16, 16),
        )
        @test result.ts == [0.0]
        @test all(isfile, result.paths)
        @test load_render(only(result.paths)).method isa Transformations
    end

    _with_tmpdir() do tmp
        result = render_interpolation_frames(
            left,
            right,
            Chaos(npoints = 250, warmup = 5);
            frames = 3,
            outdir = tmp,
            basename = "chaos_anim",
            resolution = (24, 24),
        )

        @test result.outdir == tmp
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

        # Each frame's render record mirrors it under the data directory (renders/frames/...).
        @test all(p -> isfile(render_record_path(p)), result.paths)
        @test all(
            p -> startswith(render_record_path(p), data_output_dir("renders", "frames")),
            result.paths,
        )
        mid = load_render(result.paths[2])
        @test mid isa Render{Chaos}
        @test mid.method.npoints == 250
        @test mid.resolution == (24, 24)
        @test mid.ifs.name == "Interpolated(left -> right)"
        expected_mid = interpolate_ifs(left, right, 0.5)
        @test mid.ifs.maps == expected_mid.maps
        @test mid.ifs.limits == expected_mid.limits
    end

    # Default output: an auto-named sequence directory under output_dir("frames").
    default_result = render_interpolation_frames(
        left,
        right,
        Chaos(npoints = 100, warmup = 2);
        frames = 2,
        resolution = (16, 16),
    )
    @test dirname(default_result.outdir) == output_dir("frames")
    @test startswith(basename(default_result.outdir), "left_to_right")
    @test occursin("chaos", basename(default_result.outdir))
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
        @test_throws ArgumentError render_interpolation_frames(
            left,
            right;
            frames = 2,
            outdir = tmp,
            t_end = 1.5,
        )
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
                right,
                Chaos(npoints = 250, warmup = 5);
                frames = 3,
                outdir = joinpath(tmp, "generated_frames"),
                basename = "chaos_anim",
                resolution = (24, 24),
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
            @testset "render records" begin
                ifs = IFS(SMALL_EQ)
                out =
                    render(ifs, Inverse(iterations = 2); resolution = (16, 16), outpath = "rec.png")
                @test out.outpath == joinpath(plots, "rec.png")
                # The record is named from the render, in the data directory, not from `outpath`.
                @test out.record == render_record_path(out.render)
                @test startswith(out.record, data_output_dir("renders"))
                @test basename(out.record) == "ifs_inverse_iterations=2_16x16.jld2"
                raw = Fractals.JLD2.load(out.record)
                @test raw["limits"] == out.limits
                @test raw["render"] isa Render{Inverse}
                @test load_render(out.record).method.iterations == 2

                # SVG output keeps its savename-based name and embedded provenance comment.
                named = IFS(SMALL_EQ; name = "My Fractal!")
                svg = render_transformations_svg(named; width = 32, height = 16)
                @test basename(svg) ==
                      "My_Fractal_initial_polygon=default_method=transformations_svg_resolution=16x32.svg"
                @test startswith(read(svg, String), "<!-- fractals.provenance: ")
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

    end
end

@testset "Render Entrypoint" begin
    _with_tmpdir() do tmp
        suffix = randstring(8)
        ifs = IFS(SMALL_EQ)

        out1 = render(
            ifs,
            Chaos(npoints = 5000);
            resolution = (64, 64),
            outpath = joinpath(tmp, "render_chaos_$suffix.png"),
        )
        @test isfile(out1.outpath)
        @test isfile(out1.record)
        @test size(out1.image) == (64, 64)
        @test out1.render isa Render{Chaos}
        @test out1.limits == ifs.limits
        @test keys(out1) == (:render, :image, :limits, :outpath, :record)

        # render(::Render) is the same pipeline as render(ifs, method; kwargs...).
        r = Render(ifs, Chaos(npoints = 3000, seed = 7); resolution = (32, 32))
        via_struct = render(r; outpath = joinpath(tmp, "via_struct_$suffix.png"))
        via_kwargs = render(
            ifs,
            Chaos(npoints = 3000, seed = 7);
            resolution = (32, 32),
            outpath = joinpath(tmp, "via_kwargs_$suffix.png"),
        )
        @test via_struct.image == via_kwargs.image

        out2 = render(
            ifs,
            PointDeterministic(npoints = 2000, iterations = 1);
            resolution = (48, 48),
            outpath = joinpath(tmp, "render_det_$suffix.png"),
        )
        @test isfile(out2.outpath)
        @test size(out2.image) == (48, 48)

        out_tf = render(
            ifs,
            Transformations(polygon = :line_arrow);
            resolution = (48, 48),
            color = true,
            outpath = joinpath(tmp, "render_transformations_$suffix.png"),
        )
        @test isfile(out_tf.outpath)
        @test size(out_tf.image) == (48, 48)
        tf_nonzero = count(
            px -> _is_drawn_pixel(px) && (red(px) != green(px) || green(px) != blue(px)),
            out_tf.image,
        )
        @test tf_nonzero > 0

        out_inv = render(
            ifs,
            Inverse(iterations = 2);
            resolution = (32, 32),
            outpath = joinpath(tmp, "render_inverse_$suffix.png"),
        )
        @test size(out_inv.image) == (32, 32)

        out_hide = render(
            ifs,
            Inverse(iterations = 2, divergence_scale = false);
            resolution = (32, 32),
            outpath = joinpath(tmp, "render_inverse_hide_$suffix.png"),
        )
        @test all(v -> v == 0.0f0 || v == 1.0f0, out_hide.image)

        # Seeded chaos renders are reproducible; different seeds differ.
        a = render(
            ifs,
            Chaos(npoints = 1500, seed = 1);
            resolution = (24, 24),
            outpath = joinpath(tmp, "seed_a_$suffix.png"),
        )
        b = render(
            ifs,
            Chaos(npoints = 1500, seed = 1);
            resolution = (24, 24),
            outpath = joinpath(tmp, "seed_b_$suffix.png"),
        )
        c = render(
            ifs,
            Chaos(npoints = 1500, seed = 2);
            resolution = (24, 24),
            outpath = joinpath(tmp, "seed_c_$suffix.png"),
        )
        @test a.image == b.image
        @test a.image != c.image

        # Inverse and Transformations are deterministic, so two renders agree exactly.
        inv_again = render(
            ifs,
            Inverse(iterations = 2);
            resolution = (32, 32),
            outpath = joinpath(tmp, "render_inverse_again_$suffix.png"),
        )
        @test inv_again.image == out_inv.image
    end

    # Bad arguments are rejected before anything is drawn or written.
    @test_throws ArgumentError render(IFS(SMALL_EQ), Chaos(npoints = 0))
    @test_throws ArgumentError render(IFS(SMALL_EQ), Chaos(); resolution = (0, 10))
    @test_throws ArgumentError render(IFS(SMALL_EQ), Inverse(iterations = -1))
end

@testset "Render Types" begin
    ifs = IFS(SMALL_EQ; name = "Types")

    @testset "constructor validation" begin
        @test_throws ArgumentError Chaos(npoints = 0)
        @test_throws ArgumentError Chaos(npoints = -5)
        @test_throws ArgumentError Chaos(warmup = -1)
        @test_throws ArgumentError PointDeterministic(npoints = 0)
        @test_throws ArgumentError PointDeterministic(iterations = -1)
        @test_throws ArgumentError PointDeterministic(warmup = -1)
        @test_throws ArgumentError Inverse(iterations = -1)
        @test_throws ArgumentError Transformations(polygon = :not_a_polygon)
        @test_throws ArgumentError Transformations(limits_iterations = -1)
        @test_throws ArgumentError PolygonSource(polygon = :not_a_polygon)
        @test_throws ArgumentError PolygonSource(frame = :bad_frame)
        @test_throws ArgumentError PolygonSource(limits_iterations = -1)
        @test_throws ArgumentError FileSource(joinpath(tempdir(), "no_such_seed_image.png"))
        @test_throws ArgumentError ImageIterate(iterations = 0)
        @test_throws ArgumentError BasePolygon(:not_a_polygon)
        @test_throws ArgumentError Render(ifs, Chaos(); resolution = (0, 10))
        @test_throws ArgumentError Render(ifs, Chaos(); resolution = (10, -1))

        err = _capture_exception(() -> Chaos(npoints = 0))
        @test occursin("npoints must be > 0", sprint(showerror, err))
        err = _capture_exception(() -> PolygonSource(frame = :bad_frame))
        @test occursin("frame must be :geometry or :ifs", sprint(showerror, err))
    end

    @testset "defaults and fields" begin
        @test Chaos().npoints == DEFAULT_SAMPLES
        @test Chaos().warmup == DEFAULT_WARMUP
        @test Chaos().seed === nothing
        @test Chaos(npoints = 10, seed = 3).seed == 3
        @test PointDeterministic().iterations == 1
        @test Inverse().iterations == 8
        @test Inverse().divergence_scale
        @test Transformations().polygon == :default
        @test PolygonSource().frame == :geometry
        @test ImageIterate().source isa PolygonSource
        @test ImageIterate(Chaos(npoints = 5)).source isa Chaos
        @test ImageIterate(Chaos(npoints = 5)) isa RenderMethod

        r = Render(ifs, Chaos(npoints = 10))
        @test r.resolution == RESOLUTION
        @test !r.color && !r.alpha
        @test isempty(r.overlays)
        r2 = Render(ifs, Chaos(npoints = 10); overlays = (Axes(), BasePolygon(:line)), color = true)
        @test r2.color
        @test length(r2.overlays) == 2
        @test r2.overlays[1] isa Axes
        @test r2.overlays[2] isa BasePolygon
    end

    @testset "one render per method type" begin
        _with_tmpdir() do tmp
            methods = (
                Chaos(npoints = 800, warmup = 5, seed = 1),
                PointDeterministic(npoints = 40, iterations = 1),
                Inverse(iterations = 2),
                Transformations(polygon = :line_arrow),
                ImageIterate(PolygonSource(); iterations = 2),
            )
            for (i, m) in enumerate(methods)
                out =
                    render(ifs, m; resolution = (32, 32), outpath = joinpath(tmp, "method_$i.png"))
                @test out.render.method === m
                @test size(out.image) == (32, 32)
                @test isfile(out.outpath) && isfile(out.record)
                @test count(px -> Float32(Gray(px)) > 0, out.image) > 0
            end
        end
    end

    @testset "ImageIterate composition and sources" begin
        _with_tmpdir() do tmp
            composed = render(
                ifs,
                ImageIterate(Chaos(npoints = 600, warmup = 5, seed = 1); iterations = 2);
                resolution = (32, 32),
                outpath = joinpath(tmp, "composed.png"),
            )
            @test size(composed.image) == (32, 32)
            @test eltype(composed.image) <: Gray
            @test composed.limits == ifs.limits
            @test count(px -> Float32(gray(px)) > 0, composed.image) > 0

            # Any method can be the source, including Transformations (framed by its polygon).
            from_tf = render(
                ifs,
                ImageIterate(Transformations(polygon = :line); iterations = 1);
                resolution = (32, 32),
                outpath = joinpath(tmp, "from_transformations.png"),
            )
            tf_alone = render(
                ifs,
                Transformations(polygon = :line);
                resolution = (32, 32),
                outpath = joinpath(tmp, "tf_alone.png"),
            )
            @test from_tf.limits == tf_alone.limits

            offset_ifs = IFS(reshape([0.5 0.0 0.0 0.5 10.0 10.0 1.0], 1, 7))
            geometry = render(
                offset_ifs,
                ImageIterate(PolygonSource(polygon = :line, frame = :geometry); iterations = 1);
                resolution = (32, 32),
                outpath = joinpath(tmp, "frame_geometry.png"),
            )
            own = render(
                offset_ifs,
                ImageIterate(PolygonSource(polygon = :line, frame = :ifs); iterations = 1);
                resolution = (32, 32),
                outpath = joinpath(tmp, "frame_ifs.png"),
            )
            @test geometry.limits != offset_ifs.limits
            @test own.limits == offset_ifs.limits
            @test count(>(Gray{Float32}(0)), geometry.image) > 0
        end
    end

    @testset "FileSource reads limits from a render record" begin
        _with_tmpdir() do tmp
            seed_out =
                render(ifs, Chaos(npoints = 800, warmup = 5, seed = 1); resolution = (24, 24))
            # At its default location the PNG maps to its record (plots/<rel> -> renders/<rel>).
            @test render_record_path(seed_out.outpath) == seed_out.record
            @test isfile(seed_out.record)
            other = IFS(ifs.maps, ifs.weights; name = "Other", limits = ((-5.0, 5.0), (-5.0, 5.0)))

            with_record = render(
                other,
                ImageIterate(FileSource(seed_out.outpath));
                resolution = (24, 24),
                outpath = joinpath(tmp, "with_record.png"),
            )
            @test with_record.limits == ifs.limits

            # The same image without a record falls back to the IFS's own limits.
            plain_path = joinpath(tmp, "plain.png")
            save(plain_path, Gray{Float32}.(seed_out.image))
            plain = render(
                other,
                ImageIterate(FileSource(plain_path));
                resolution = (24, 24),
                outpath = joinpath(tmp, "plain_out.png"),
            )
            @test plain.limits == other.limits
            @test with_record.image != plain.image
        end
    end

    @testset "overlays, color and alpha" begin
        _with_tmpdir() do tmp
            rot_eq = reshape([cos(pi / 6) sin(pi / 6) -sin(pi / 6) cos(pi / 6) 0.0 0.0 1.0], 1, 7)
            rot = IFS(rot_eq)
            method = ImageIterate(PolygonSource(polygon = :line_arrow); iterations = 1)
            plain = render(
                rot,
                method;
                resolution = (96, 96),
                outpath = joinpath(tmp, "overlay_none.png"),
            )
            base = render(
                rot,
                method;
                resolution = (96, 96),
                overlays = [BasePolygon(:line_arrow)],
                outpath = joinpath(tmp, "overlay_base.png"),
            )
            axes_on = render(
                rot,
                method;
                resolution = (96, 96),
                overlays = [Axes()],
                outpath = joinpath(tmp, "overlay_axes.png"),
            )
            @test base.image != plain.image
            @test count(>(Gray{Float32}(0)), base.image) > count(>(Gray{Float32}(0)), plain.image)
            @test axes_on.image != plain.image
            @test count(>(Gray{Float32}(0)), axes_on.image) >
                  count(>(Gray{Float32}(0)), plain.image)

            alpha_overlay = render(
                rot,
                method;
                resolution = (96, 96),
                overlays = [BasePolygon(:line_arrow), Axes()],
                alpha = true,
                outpath = joinpath(tmp, "overlay_alpha.png"),
            )
            @test eltype(alpha_overlay.image) == GrayA{Float32}
            @test any(px -> alpha(px) == 0.0f0, alpha_overlay.image)
            @test count(_is_drawn_pixel, alpha_overlay.image) >
                  count(>(Gray{Float32}(0)), plain.image)

            # Overlays work on a raw heatmap too (it is converted to Gray first).
            heat = render(
                ifs,
                Chaos(npoints = 500, seed = 1);
                resolution = (32, 32),
                overlays = [Axes()],
                outpath = joinpath(tmp, "heat_axes.png"),
            )
            @test eltype(heat.image) <: Gray
            heat_ca = render(
                ifs,
                Chaos(npoints = 500, seed = 1);
                resolution = (32, 32),
                color = true,
                alpha = true,
                outpath = joinpath(tmp, "heat_color_alpha.png"),
            )
            @test eltype(heat_ca.image) == RGBA{Float32}
        end
    end

    @testset "load_render round trip" begin
        _with_tmpdir() do tmp
            named = IFS(SMALL_EQ; name = "Round Trip")
            out = render(
                named,
                Chaos(npoints = 700, warmup = 7, seed = 11);
                resolution = (20, 28),
                color = true,
                overlays = [Axes(), BasePolygon(:line)],
                outpath = joinpath(tmp, "trip.png"),
            )
            @test out.record == render_record_path(out.render)

            # The record loads from its own path (the PNG has a custom outpath, so it has no mirror).
            for path in (out.record,)
                loaded = load_render(path)
                @test loaded isa Render{Chaos}
                @test loaded.method.npoints == 700
                @test loaded.method.warmup == 7
                @test loaded.method.seed == 11
                @test loaded.resolution == (20, 28)
                @test loaded.color && !loaded.alpha
                @test [typeof(o) for o in loaded.overlays] == [Axes, BasePolygon]
                @test loaded.overlays[2].polygon == :line
                @test loaded.ifs.name == "Round Trip"
                @test loaded.ifs.maps == named.maps
                @test collect(loaded.ifs.weights) == collect(named.weights)
                @test loaded.ifs.limits == named.limits
            end

            # Re-rendering the loaded description reproduces a seeded render exactly.
            again = render(load_render(out.record); outpath = joinpath(tmp, "trip_again.png"))
            @test again.image == out.image
            @test again.limits == out.limits

            # Other method types round-trip as well, including a composed ImageIterate.
            composed = render(
                named,
                ImageIterate(Chaos(npoints = 300, seed = 2); iterations = 2);
                resolution = (16, 16),
                outpath = joinpath(tmp, "composed_trip.png"),
            )
            loaded = load_render(composed.record)
            @test loaded isa Render{<:ImageIterate}
            @test loaded.method.source isa Chaos
            @test loaded.method.iterations == 2

            raw = Fractals.JLD2.load(out.record)
            @test raw["limits"] == out.limits
            @test raw["render"] isa Render
        end
    end

    @testset "savename-based file names" begin
        _with_tmpdir() do tmp
            with_output_root(tmp) do
                plots = joinpath(abspath(tmp), "plots")
                named = IFS(SMALL_EQ; name = "My Fractal!")
                res = (16, 24)

                chaos = render(named, Chaos(npoints = 500); resolution = res)
                @test dirname(chaos.outpath) == plots
                @test basename(chaos.outpath) == "My_Fractal_chaos_npoints=500_16x24.png"
                @test basename(chaos.record) == "My_Fractal_chaos_npoints=500_16x24.jld2"

                seeded = render(named, Chaos(npoints = 500, seed = 3); resolution = res)
                @test basename(seeded.outpath) == "My_Fractal_chaos_npoints=500_seed=3_16x24.png"

                det = render(
                    named,
                    PointDeterministic(npoints = 20, iterations = 2);
                    resolution = res,
                    color = true,
                )
                @test basename(det.outpath) ==
                      "My_Fractal_point_deterministic_iterations=2_npoints=20_16x24_color.png"

                inv = render(named, Inverse(iterations = 2); resolution = res)
                @test basename(inv.outpath) == "My_Fractal_inverse_iterations=2_16x24.png"

                tf = render(named, Transformations(polygon = :line); resolution = res, alpha = true)
                @test basename(tf.outpath) ==
                      "My_Fractal_transformations_limits_iterations=1_polygon=line_16x24_alpha.png"

                ii = render(
                    named,
                    ImageIterate(PolygonSource(polygon = :line); iterations = 2);
                    resolution = res,
                )
                @test basename(ii.outpath) ==
                      "My_Fractal_image_iterate-polygon_iterations=2_16x24.png"

                composed = render(
                    named,
                    ImageIterate(Chaos(npoints = 100); iterations = 1);
                    resolution = res,
                )
                @test basename(composed.outpath) ==
                      "My_Fractal_image_iterate-chaos_iterations=1_16x24.png"

                # Relative paths resolve under output_dir(); absolute paths are used as given.
                rel = render(named, Chaos(npoints = 50); resolution = res, outpath = "rel.png")
                @test rel.outpath == joinpath(plots, "rel.png")
                nested = render(
                    named,
                    Chaos(npoints = 50);
                    resolution = res,
                    outpath = joinpath("nested", "x.png"),
                )
                @test nested.outpath == joinpath(plots, "nested", "x.png")
                @test isfile(nested.record)
                absolute = joinpath(tmp, "elsewhere", "z.png")
                abs_out = render(named, Chaos(npoints = 50); resolution = res, outpath = absolute)
                @test abs_out.outpath == absolute
                @test isfile(absolute)
            end
        end
    end
end

@testset "Render Image Iterate API" begin
    _with_tmpdir() do tmp
        suffix = randstring(8)
        ifs = IFS(SMALL_EQ)

        out_poly = render(
            ifs,
            ImageIterate(PolygonSource(); iterations = 2);
            resolution = (48, 48),
            outpath = joinpath(tmp, "render_image_poly_$suffix.png"),
        )
        @test isfile(out_poly.outpath)
        @test size(out_poly.image) == (48, 48)
        @test eltype(out_poly.image) <: Gray

        out_chaos = render(
            ifs,
            ImageIterate(Chaos(npoints = 400, warmup = 5); iterations = 2);
            resolution = (40, 40),
            outpath = joinpath(tmp, "render_image_chaos_$suffix.png"),
        )
        @test size(out_chaos.image) == (40, 40)

        p = joinpath(tmp, "seed.png")
        src = fill(Gray{Float32}(0.0f0), 20, 20)
        src[6:15, 6:15] .= Gray{Float32}(1.0f0)
        save(p, src)
        out_file = render(
            ifs,
            ImageIterate(FileSource(p); iterations = 2);
            resolution = (20, 20),
            outpath = joinpath(tmp, "render_image_file_$suffix.png"),
        )
        @test size(out_file.image) == (20, 20)

        out_color = render(
            ifs,
            ImageIterate(PolygonSource(); iterations = 2);
            color = true,
            resolution = (24, 24),
            outpath = joinpath(tmp, "render_image_color_$suffix.png"),
        )
        @test size(out_color.image) == (24, 24)
        @test eltype(out_color.image) == RGB{Float32}

        rgb_path = joinpath(tmp, "seed_rgb.png")
        rgb_seed = fill(RGB{Float32}(0.0f0, 0.5f0, 1.0f0), 20, 20)
        save(rgb_path, rgb_seed)
        rgb_err = _capture_exception(
            () -> render(
                ifs,
                ImageIterate(FileSource(rgb_path));
                resolution = (20, 20),
                outpath = joinpath(tmp, "never_write_rgb_$suffix.png"),
            ),
        )
        @test rgb_err isa ArgumentError
        @test occursin("grayscale source image", sprint(showerror, rgb_err))
        @test !isfile(joinpath(tmp, "never_write_rgb_$suffix.png"))

        graya_path = joinpath(tmp, "seed_graya.png")
        graya_seed = fill(GrayA{Float32}(0.0f0, 0.0f0), 20, 20)
        graya_seed[6:15, 6:15] .= GrayA{Float32}(1.0f0, 0.5f0)
        save(graya_path, graya_seed)
        out_graya = render(
            ifs,
            ImageIterate(FileSource(graya_path); iterations = 1);
            resolution = (20, 20),
            outpath = joinpath(tmp, "render_image_graya_$suffix.png"),
        )
        @test size(out_graya.image) == (20, 20)
        @test count(px -> Float32(gray(px)) > 0, out_graya.image) > 0

        direct_graya = iterate_image(IFS(SMALL_EQ), graya_seed)
        @test size(direct_graya) == (20, 20)
        @test eltype(direct_graya) <: Gray
        @test count(px -> Float32(gray(px)) > 0, direct_graya) > 0

        # Seeds: the polygon preset drawn under different presets gives different images.
        seed_line = render(
            ifs,
            ImageIterate(PolygonSource(polygon = :line); iterations = 1);
            resolution = (64, 64),
            outpath = joinpath(tmp, "seed_line_$suffix.png"),
        )
        seed_triangle = render(
            ifs,
            ImageIterate(PolygonSource(polygon = :equilateral_triangle); iterations = 1);
            resolution = (64, 64),
            outpath = joinpath(tmp, "seed_triangle_$suffix.png"),
        )
        @test count(>(Gray{Float32}(0)), seed_line.image) > 0
        @test count(>(Gray{Float32}(0)), seed_triangle.image) > 0
        @test seed_line.image != seed_triangle.image
    end

    @testset "Polygon Limits Geometry Helpers" begin
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
        function limits_spans(limits)
            (xmin, xmax), (ymin, ymax) = limits
            return xmax - xmin, ymax - ymin
        end

        edgar_defs = Fractals.parse_ifs_definitions_file(ifs_data_dir("Edgar.ifs"))
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

    @testset "polygon geometry framing matches Transformations" begin
        _with_tmpdir() do tmp
            rot_eq = reshape([cos(pi / 6) sin(pi / 6) -sin(pi / 6) cos(pi / 6) 0.0 0.0 1.0], 1, 7)
            rot_ifs = IFS(rot_eq)
            poly_seed = render(
                rot_ifs,
                ImageIterate(PolygonSource(polygon = :line_arrow); iterations = 1);
                resolution = (96, 96),
                outpath = joinpath(tmp, "poly_seed.png"),
            )
            tf_reference = render(
                rot_ifs,
                Transformations(polygon = :line_arrow);
                resolution = (96, 96),
                outpath = joinpath(tmp, "poly_reference.png"),
            )
            @test poly_seed.image == tf_reference.image
            @test poly_seed.limits == tf_reference.limits
        end
    end

    @testset "Imported Python Data File" begin
        imported_defs =
            Fractals.parse_ifs_definitions_file(ifs_data_dir("imported_from_python.ifs"))
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
    end
end

@testset "Snapshot Image Tests" begin
    _with_tmpdir() do d
        ifs = IFS(SMALL_EQ)
        transform = render(
            ifs,
            Transformations(polygon = :line_arrow);
            resolution = (96, 96),
            color = true,
            overlays = [Axes()],
            outpath = joinpath(d, "snapshot_transformations.png"),
        )
        _assert_matches_snapshot("transformations-line-arrow-color-axis", transform.image)

        inverse = render(
            ifs,
            Inverse(iterations = 2, divergence_scale = false);
            resolution = (32, 32),
            outpath = joinpath(d, "snapshot_inverse.png"),
        )
        _assert_matches_snapshot("inverse-mask-small", inverse.image)

        image_iter = render(
            ifs,
            ImageIterate(PolygonSource(); iterations = 2);
            resolution = (48, 48),
            outpath = joinpath(d, "snapshot_image_iterate.png"),
        )
        _assert_matches_snapshot("image-iterate-polygon-small", image_iter.image)
    end
end

@testset "get_ifs selects a definition by name" begin
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

    _with_tmpdir() do tmp
        write(joinpath(tmp, "multi.ifs"), text)
        write(joinpath(tmp, "other.ifs"), "OtherIFS {\n  0.5 0.0 0.0 0.5 0.0 0.0\n}\n")
        lib = joinpath(tmp, "lib")
        suffix = randstring(8)

        second = get_ifs("SecondIFS"; file = "multi.ifs", dir = tmp, libdir = lib)
        first_ifs = get_ifs("FirstIFS"; dir = tmp, libdir = lib)
        @test second.name == "SecondIFS"
        @test length(second.maps) == 2
        @test first_ifs.name == "FirstIFS"

        out = render(
            second,
            Chaos(npoints = 1500);
            resolution = (24, 24),
            outpath = joinpath(tmp, "render_multi_$suffix.png"),
        )
        @test out.render.ifs.name == "SecondIFS"
        @test isfile(out.outpath)

        # The same lookup through a prebuilt library.
        build_ifs_library(; dir = tmp, outdir = lib)
        @test get_ifs("SecondIFS"; file = "multi.ifs", dir = tmp, libdir = lib).maps == second.maps

        @test_throws ArgumentError get_ifs("MissingIFS"; dir = tmp, libdir = lib)
        @test_throws ArgumentError get_ifs("FirstIFS"; file = "other.ifs", dir = tmp, libdir = lib)
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
end

@testset "Render Color API" begin
    _with_tmpdir() do d
        ifs = IFS(SMALL_EQ)
        methods = (
            Chaos(npoints = 800, warmup = 5),
            PointDeterministic(npoints = 40, iterations = 1),
            Inverse(iterations = 2),
            Transformations(),
            ImageIterate(PolygonSource(); iterations = 2),
        )
        for (i, m) in enumerate(methods)
            out = render(
                ifs,
                m;
                color = true,
                resolution = (40, 40),
                outpath = joinpath(d, "color_$i.png"),
            )
            @test size(out.image) == (40, 40)
            @test eltype(out.image) == RGB{Float32}
            @test isfile(out.outpath)
        end
    end
end

@testset "Render Alpha API" begin
    _with_tmpdir() do d
        ifs = IFS(SMALL_EQ)
        methods = (
            Chaos(npoints = 800, warmup = 5),
            PointDeterministic(npoints = 40, iterations = 1),
            Inverse(iterations = 2),
            ImageIterate(PolygonSource(); iterations = 2),
            Transformations(),
        )
        for (i, m) in enumerate(methods)
            gray_out = render(
                ifs,
                m;
                alpha = true,
                resolution = (40, 40),
                outpath = joinpath(d, "alpha_gray_$i.png"),
            )
            @test size(gray_out.image) == (40, 40)
            @test eltype(gray_out.image) == GrayA{Float32}
            @test any(px -> alpha(px) == 0.0f0, gray_out.image)
            @test any(_is_drawn_pixel, gray_out.image)

            color_out = render(
                ifs,
                m;
                color = true,
                alpha = true,
                resolution = (40, 40),
                outpath = joinpath(d, "alpha_color_$i.png"),
            )
            @test size(color_out.image) == (40, 40)
            @test eltype(color_out.image) == RGBA{Float32}
            @test any(px -> alpha(px) == 0.0f0, color_out.image)
            @test any(_is_drawn_pixel, color_out.image)
        end
    end
end

@testset "Main Example Entrypoint" begin
    _with_tmpdir() do d
        old = pwd()
        cd(d)
        try
            Fractals.main(eq = SMALL_EQ, npoints = 1000, outpath = "main_output.png")
            @test isfile(output_dir("main_output.png"))
        finally
            cd(old)
        end
    end
end

@testset "Point cloud cache ignores name, resolution and style" begin
    ifs = IFS(SMALL_EQ; name = "cache")
    m = Chaos(npoints = 500, warmup = 3, seed = 4)
    out = render(ifs, m; resolution = (16, 16))
    points, path, cached = Fractals.cached_points(m, ifs)
    @test cached
    @test startswith(path, data_output_dir("points", ifs_key(ifs)))
    stamp = mtime(path)
    big = render(Render(out.render; resolution = (48, 48), color = true, alpha = true))
    @test size(big.image) == (48, 48)
    renamed = IFS(SMALL_EQ; name = "renamed")
    @test ifs_key(renamed) == ifs_key(ifs)
    @test Fractals.cached_points(m, renamed)[1] == points
    @test mtime(path) == stamp
    @test length(readdir(dirname(path))) == 1
    @test !Fractals.cached_points(m, ifs; force = true)[3]
    @test Fractals.cached_points(Chaos(npoints = 501, warmup = 3, seed = 4), ifs)[2] != path
end

include("test_jld2.jl")
