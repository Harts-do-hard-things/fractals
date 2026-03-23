using Test
using LinearAlgebra
using StaticArrays
using Fractals
using Random
using FileIO
using Colors
using JSON3

function _env_flag(name::AbstractString)::Bool
    value = get(ENV, name, "")
    lowercase(strip(value)) in ("1", "true", "yes", "on")
end

const SMALL_EQ = [
    0.5  0.0  0.0  0.5  0.0  0.0  0.6;
   -0.5  0.0  0.0 -0.5  1.0  0.0  0.4
]
const SNAPSHOT_DIR = joinpath(@__DIR__, "snapshots")

function _probe_gpu_available()::Bool
    ifs = IFS(SMALL_EQ; npoints=4)
    try
        make_image(ifs; resolution=(4, 4), backend=:gpu)
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
    return RGBA{Float32}(abs(Float32(red(expected)) - Float32(red(actual))),
                         abs(Float32(green(expected)) - Float32(green(actual))),
                         abs(Float32(blue(expected)) - Float32(blue(actual))),
                         1.0f0)
end

function _write_snapshot_debug(name::AbstractString, expected, actual)
    debug_dir = mktempdir(prefix="fractals_snapshot_")
    actual_path = joinpath(debug_dir, name * "_actual.png")
    diff_path = joinpath(debug_dir, name * "_diff.png")
    save(actual_path, actual)
    diff = [_snapshot_diff_pixel(expected[i], actual[i]) for i in eachindex(expected)]
    save(diff_path, reshape(diff, size(expected)))
    return debug_dir, actual_path, diff_path
end

function _assert_matches_snapshot(name::AbstractString, actual)
    path = _snapshot_path(name)
    isfile(path) || error("Missing snapshot fixture: $path")
    expected = load(path)
    encoded_actual = mktempdir(prefix="fractals_snapshot_actual_") do tmp
        actual_path = joinpath(tmp, name * ".png")
        save(actual_path, actual)
        load(actual_path)
    end
    size(expected) == size(encoded_actual) || error("Snapshot '$name' size mismatch: expected $(size(expected)), got $(size(encoded_actual))")
    if expected != encoded_actual
        debug_dir, actual_path, diff_path = _write_snapshot_debug(name, expected, encoded_actual)
        error("Snapshot '$name' mismatch. Debug artifacts written to $debug_dir (actual: $actual_path, diff: $diff_path)")
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

@testset "IFS constructor initializes maps, weights, and bounds from equation matrices" begin
    ifs = IFS(SMALL_EQ; npoints=1000)
    @test length(ifs.points) == 1000
    @test length(ifs.maps) == size(SMALL_EQ, 1)
    @test length(ifs.weights) == size(SMALL_EQ, 1)
    @test collect(ifs.weights) ≈ [0.6, 0.4]
    @test ifs.limits[1][1] < ifs.limits[1][2]
    @test ifs.limits[2][1] < ifs.limits[2][2]

    eq6 = [
        1.0 0.0 0.0 1.0 0.0 0.0
        0.5 0.0 0.0 0.5 1.0 0.0
    ]
    ifs6 = IFS(eq6; npoints=25, name="Derived", docs="from 6-column matrix")
    @test length(ifs6.points) == 25
    @test ifs6.name == "Derived"
    @test ifs6.docs == "from 6-column matrix"
    @test collect(ifs6.weights) ≈ [1.0, 0.25]
    @test ifs6.maps[1](SVector{2,Float64}(2.0, 3.0)) ≈ SVector{2,Float64}(2.0, 3.0)
    @test ifs6.maps[2](SVector{2,Float64}(2.0, 3.0)) ≈ SVector{2,Float64}(2.0, 1.5)
end

@testset "Interpolation Helpers" begin
    left6 = [
        1.0  0.0  0.0  1.0  0.0  0.0
        0.5  0.0  0.0  0.5  1.0  0.0
    ]
    right7 = [
        0.0  1.0  -1.0  0.0  0.0  1.0  0.2
        1.0  0.0   0.0  1.0  2.0  0.0  0.8
    ]

    midpoint = interpolate_eq_matrix(left6, right7, 0.5)
    @test size(midpoint) == (2, 7)
    @test midpoint[:, 1:6] ≈ [
        0.5  0.5  -0.5  0.5  0.0  0.5
        0.75 0.0   0.0  0.75 1.5  0.0
    ]
    @test midpoint[:, 7] ≈ [0.5, 0.5]
    @test isapprox(sum(midpoint[:, 7]), 1.0; atol=1e-12)

    @test interpolate_eq_matrix(left6, right7, 0.0) ≈ [
        1.0  0.0  0.0  1.0  0.0  0.0  0.8
        0.5  0.0  0.0  0.5  1.0  0.0  0.2
    ]
    @test interpolate_eq_matrix(left6, right7, 1.0) ≈ right7

    left_ifs = IFS(left6; npoints=12, name="Left", docs="left docs")
    right_ifs = IFS(right7; npoints=20, name="Right", docs="right docs")
    blended = interpolate_ifs(left_ifs, right_ifs, 0.25)

    @test length(blended.points) == 12
    @test blended.name == "Interpolated(Left -> Right)"
    @test occursin("Interpolated IFS at t=0.25", blended.docs)
    # Verify the blended map transforms points correctly (t=0.25 linear blend):
    # left map 1 is identity; right map 1 maps (x,y)→(-y, x+1). Blended: A*x+b.
    @test blended.maps[1](SVector{2,Float64}(1.0, 0.0)) ≈ SVector{2,Float64}(0.75, 0.0)
    @test blended.maps[1](SVector{2,Float64}(0.0, 1.0)) ≈ SVector{2,Float64}(0.25, 1.0)
    @test collect(blended.weights) ≈ [0.65, 0.35]
    @test blended.limits[1][1] ≈ 0.75 * left_ifs.limits[1][1] + 0.25 * right_ifs.limits[1][1]
    @test blended.limits[1][2] ≈ 0.75 * left_ifs.limits[1][2] + 0.25 * right_ifs.limits[1][2]
    @test blended.limits[2][1] ≈ 0.75 * left_ifs.limits[2][1] + 0.25 * right_ifs.limits[2][1]
    @test blended.limits[2][2] ≈ 0.75 * left_ifs.limits[2][2] + 0.25 * right_ifs.limits[2][2]

    right_locked = interpolate_ifs(left_ifs, right_ifs, 0.5; limits_mode=:right, npoints=3, name="Blend", docs="manual docs")
    @test length(right_locked.points) == 3
    @test right_locked.name == "Blend"
    @test right_locked.docs == "manual docs"
    @test right_locked.limits == right_ifs.limits

    @test_throws ArgumentError interpolate_eq_matrix(left6, right7[:, 1:6], -0.1)
    @test_throws ArgumentError interpolate_eq_matrix(left6, ones(3, 6), 0.5)
    @test_throws ArgumentError interpolate_ifs(left_ifs, right_ifs, 0.5; limits_mode=:bad)
    @test_throws ArgumentError interpolate_ifs(left_ifs, right_ifs, 0.5; npoints=-1)

    recomputed = interpolate_ifs(left_ifs, right_ifs, 0.5; limits_mode=:recompute)
    @test recomputed.limits == refresh_limits(recomputed; source=:maps).limits
end

@testset "Limits Refresh And Staleness" begin
    ifs = IFS(SMALL_EQ; npoints=128, name="Limits", docs="stale limits")
    original_limits = ifs.limits
    iterate!(ifs; warmup=5, seed=4242)
    @test ifs.limits == original_limits

    shifted_points = [p + SVector{2,Float64}(25.0, -17.0) for p in ifs.points]
    stale = IFS(ifs.name, ifs.docs, shifted_points, ifs.maps, ifs.weights, ifs.limits)
    point_refreshed = refresh_limits(stale; source=:points)
    @test point_refreshed.name == stale.name
    @test point_refreshed.docs == stale.docs
    @test point_refreshed.maps == stale.maps
    @test collect(point_refreshed.weights) == collect(stale.weights)
    @test point_refreshed.points == stale.points
    @test point_refreshed.limits != stale.limits

    maps_refreshed = refresh_limits(ifs; source=:maps)
    @test maps_refreshed.limits == original_limits
    @test maps_refreshed.points == ifs.points

    bad_source = _capture_exception(() -> refresh_limits(ifs; source=:bad))
    @test bad_source isa ArgumentError
    @test occursin("Supported: :maps, :points", sprint(showerror, bad_source))

    stale_wide = IFS(stale.name, stale.docs, stale.points, stale.maps, stale.weights, ((-100.0, 100.0), (-100.0, 100.0)))
    stale_img = make_image(stale_wide; resolution=(32, 32), backend=:cpu)
    refreshed_img = make_image(refresh_limits(stale_wide; source=:points); resolution=(32, 32), backend=:cpu)
    @test stale_img != refreshed_img
end

@testset "Interpolation Frame Renderer" begin
    left = IFS([
        0.5  -0.5   0.5   0.5   0.0   0.0
       -0.5  -0.5   0.5  -0.5   1.0   0.0
    ]; npoints=40, name="left")
    right = IFS([
        0.5   0.0   0.0   0.5   0.0   0.0   0.3
        0.0   0.5  -0.5   0.0   1.0   0.0   0.7
    ]; npoints=60, name="right")

    mktempdir() do tmp
        result = render_interpolation_frames(left, right;
                                             frames=3,
                                             outdir=tmp,
                                             basename="anim",
                                             render_method=RenderTransformations,
                                             resolution=(24, 24),
                                             color=true,
                                             initial_polygon=:line_arrow,
                                             axis=true)

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

    mismatch = IFS([
        0.5  0.0  0.0  0.5  0.0  0.0
        0.5  0.0  0.0  0.5  0.5  0.0
        0.0  0.5 -0.5  0.0  1.0  0.0
    ]; npoints=10, name="mismatch")

    mktempdir() do tmp
        @test_throws ArgumentError render_interpolation_frames(left, right; frames=0, outdir=tmp)
        @test_throws ArgumentError render_interpolation_frames(left, right; frames=2, outdir=tmp, basename=" ")
        @test_throws ArgumentError render_interpolation_frames(left, right; frames=2, outdir=tmp, render_method=Chaos)
        @test_throws ArgumentError render_interpolation_frames(left, mismatch; frames=2, outdir=tmp)
    end
end

@testset "iterate! and iterate_parallel! populate the IFS point cloud" begin
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

    iterate_parallel!(ifs1; warmup=10, seed=12345)
    iterate_parallel!(ifs2; warmup=10, seed=12345)
    iterate_parallel!(ifs3; warmup=10, seed=54321)

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
    function _safe_expand(points, maps, n)
        cur = copy(points)
        for _ in 1:n
            next = Vector{SVector{2,Float64}}(undef, length(cur) * length(maps))
            clen = length(cur)
            for i in 1:length(maps)
                start = (i - 1) * clen + 1
                stop = i * clen
                next[start:stop] = maps[i].(cur)
            end
            cur = next
        end
        return cur
    end

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
    expected_points = _safe_expand(base.points, ifs2.maps, n)
    out2 = deterministic_iterate(ifs2, n; warmup=5, seed=123)
    @test out2.points == expected_points

    # For one round, each map block must be exactly map(points).
    one_round = deterministic_iterate(ifs2, 1; warmup=5, seed=123).points
    base_len = length(base.points)
    for i in 1:length(ifs2.maps)
        start = (i - 1) * base_len + 1
        stop = i * base_len
        @test one_round[start:stop] == ifs2.maps[i].(base.points)
    end
end

@testset "make_pixelate_map produces an invertible coordinate transform into pixel space" begin
    limits = ((0.0, 2.0), (-1.0, 1.0))
    m = make_pixelate_map(limits; resolution=(100, 200))
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
    ifs = IFS(SMALL_EQ; npoints=4_000)
    iterate!(ifs; warmup=5, seed=123)

    square = make_image(ifs; resolution=(32, 32))
    wide = make_image(ifs; resolution=(24, 40))
    tall = make_image(ifs; resolution=(40, 24))

    @test size(square) == (32, 32)
    @test size(wide) == (24, 40)
    @test size(tall) == (40, 24)
    @test maximum(square) > 0f0
    @test maximum(wide) > 0f0
    @test maximum(tall) > 0f0
    @test all(square .>= 0f0) && maximum(square) <= 1f0
    @test all(wide .>= 0f0) && maximum(wide) <= 1f0
    @test all(tall .>= 0f0) && maximum(tall) <= 1f0
end

@testset "make_image converts point cloud to a normalized Float32 grayscale heatmap" begin
    ifs = IFS(SMALL_EQ; npoints=2000)
    iterate!(ifs; warmup=5)
    img = make_image(ifs; resolution=(32, 32))
    @test size(img) == (32, 32)
    @test eltype(img) == Float32
    @test all(img .>= 0f0)
    @test maximum(img) <= 1f0
end

@testset "Make Image Backends" begin
    ifs = IFS(SMALL_EQ; npoints=2000)
    iterate!(ifs; warmup=5, seed=123)

    img_default = make_image(ifs; resolution=(32, 32))
    img_cpu = make_image(ifs; resolution=(32, 32), backend=:cpu)
    @test img_default == img_cpu

    img_auto = make_image(ifs; resolution=(32, 32), backend=:auto)
    @test size(img_auto) == (32, 32)
    @test eltype(img_auto) == Float32
    @test all(img_auto .>= 0f0)
    @test maximum(img_auto) <= 1f0

    if _probe_gpu_available()
        img_gpu = make_image(ifs; resolution=(32, 32), backend=:gpu)
        @test size(img_gpu) == (32, 32)
        @test eltype(img_gpu) == Float32
        @test all(img_gpu .>= 0f0)
        @test maximum(img_gpu) <= 1f0
    else
        @test_throws ArgumentError make_image(ifs; resolution=(32, 32), backend=:gpu)
    end

    @test_throws ArgumentError make_image(ifs; resolution=(32, 32), backend=:bad)
end

@testset "GPU Parity (Optional)" begin
    if get(ENV, "FRACTALS_RUN_GPU_TESTS", "0") == "1"
        if _probe_gpu_available()
            ifs = IFS(SMALL_EQ; npoints=5000)
            iterate!(ifs; warmup=5, seed=987)
            cpu_img = make_image(ifs; resolution=(48, 48), backend=:cpu)
            gpu_img = make_image(ifs; resolution=(48, 48), backend=:gpu)
            @test isapprox(cpu_img, gpu_img; atol=1e-5, rtol=1e-5)
        else
            _skip_optional_gpu_parity!("GPU Parity (Optional)")
        end
    else
        @test true
    end
end

@testset "Iterate Image" begin
    ifs = IFS(SMALL_EQ; npoints=2000)
    iterate!(ifs; warmup=5)
    img = make_image(ifs; resolution=(32, 32))
    out = iterate_image(ifs, img)
    out2 = iterate_image(ifs, out)
    @test size(out) == (32, 32)
    @test size(out2) == (32, 32)
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

@testset "Iterate Image Backends" begin
    ifs = IFS(SMALL_EQ; npoints=2000)
    iterate!(ifs; warmup=5, seed=321)
    src = make_image(ifs; resolution=(24, 24), backend=:cpu)

    out_default = iterate_image(ifs, src)
    out_cpu = iterate_image(ifs, src; backend=:cpu)
    @test out_default == out_cpu

    out_auto = iterate_image(ifs, src; backend=:auto)
    @test size(out_auto) == (24, 24)
    @test eltype(out_auto) == Gray{Float32}

    out_auto_color = iterate_image(ifs, src; colors=true, backend=:auto, seed=777)
    @test size(out_auto_color) == (24, 24)
    @test eltype(out_auto_color) == RGB{Float32}

    if _probe_gpu_available()
        out_gpu = iterate_image(ifs, src; backend=:gpu)
        @test size(out_gpu) == (24, 24)
        @test eltype(out_gpu) == Gray{Float32}

        out_gpu_color = iterate_image(ifs, src; colors=true, backend=:gpu, seed=777)
        @test size(out_gpu_color) == (24, 24)
        @test eltype(out_gpu_color) == RGB{Float32}
    else
        @test_throws ArgumentError iterate_image(ifs, src; backend=:gpu)
        @test_throws ArgumentError iterate_image(ifs, src; colors=true, backend=:gpu, seed=777)
    end

    @test_throws ArgumentError iterate_image(ifs, src; backend=:bad)
end

@testset "Iterate Image GPU Parity (Optional)" begin
    if get(ENV, "FRACTALS_RUN_GPU_TESTS", "0") == "1"
        if _probe_gpu_available()
            ifs = IFS(SMALL_EQ; npoints=2000)
            iterate!(ifs; warmup=5, seed=4321)
            src = make_image(ifs; resolution=(28, 28), backend=:cpu)

            cpu_gray = iterate_image(ifs, src; backend=:cpu)
            gpu_gray = iterate_image(ifs, src; backend=:gpu)
            @test isapprox(Float32.(cpu_gray), Float32.(gpu_gray); atol=1e-5, rtol=1e-5)

            cpu_rgb = iterate_image(ifs, src; colors=true, backend=:cpu, seed=99)
            gpu_rgb = iterate_image(ifs, src; colors=true, backend=:gpu, seed=99)
            @test isapprox(Float32.(Gray.(cpu_rgb)), Float32.(Gray.(gpu_rgb)); atol=1e-5, rtol=1e-5)
        else
            _skip_optional_gpu_parity!("Iterate Image GPU Parity (Optional)")
        end
    else
        @test true
    end
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

@testset "inverse_iterate returns a Float32 coverage value in [0, 1] for pixel corners" begin
    ifs = IFS(SMALL_EQ; npoints=10)
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
    ifs = IFS(SMALL_EQ; npoints=100)
    lims = ifs.limits

    img_default = rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8))
    img_cpu = rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8), backend=:cpu, mode=:exact)
    @test img_default == img_cpu

    img_auto = rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8), backend=:auto, mode=:exact)
    @test size(img_auto) == (8, 8)
    @test eltype(img_auto) == Float32
    @test all(img_auto .>= 0f0)
    @test maximum(img_auto) <= 1f0

    img_cpu_preview_a = rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8), backend=:cpu, mode=:preview)
    img_cpu_preview_b = rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8), backend=:cpu, mode=:preview)
    @test img_cpu_preview_a == img_cpu_preview_b

    if _probe_gpu_available()
        img_gpu_exact = rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8), backend=:gpu, mode=:exact)
        @test size(img_gpu_exact) == (8, 8)
        @test eltype(img_gpu_exact) == Float32
        @test all(img_gpu_exact .>= 0f0)
        @test maximum(img_gpu_exact) <= 1f0

        img_gpu_preview = rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8), backend=:gpu, mode=:preview)
        @test size(img_gpu_preview) == (8, 8)
        @test eltype(img_gpu_preview) == Float32
        @test all(img_gpu_preview .>= 0f0)
        @test maximum(img_gpu_preview) <= 1f0
    else
        @test_throws ArgumentError rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8), backend=:gpu, mode=:exact)
        @test_throws ArgumentError rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8), backend=:gpu, mode=:preview)
    end

    @test_throws ArgumentError rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8), backend=:bad)
    @test_throws ArgumentError rasterize_image_inversely(ifs, 2, lims; resolution=(8, 8), mode=:bad)
end

@testset "Inverse Rasterize GPU Parity (Optional)" begin
    if get(ENV, "FRACTALS_RUN_GPU_TESTS", "0") == "1"
        if _probe_gpu_available()
            ifs = IFS(SMALL_EQ; npoints=100)
            lims = ifs.limits

            cpu_img = rasterize_image_inversely(ifs, 2, lims; resolution=(16, 16), backend=:cpu, mode=:exact)
            gpu_img = rasterize_image_inversely(ifs, 2, lims; resolution=(16, 16), backend=:gpu, mode=:exact)
            @test isapprox(cpu_img, gpu_img; atol=1e-5, rtol=1e-5)

            gpu_preview_a = rasterize_image_inversely(ifs, 2, lims; resolution=(16, 16), backend=:gpu, mode=:preview)
            gpu_preview_b = rasterize_image_inversely(ifs, 2, lims; resolution=(16, 16), backend=:gpu, mode=:preview)
            @test gpu_preview_a == gpu_preview_b
        else
            _skip_optional_gpu_parity!("Inverse Rasterize GPU Parity (Optional)")
        end
    else
        @test true
    end
end

@testset "parse_ifs_string extracts name, documentation, and maps from IFS text blocks" begin
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

@testset "lex_ifs tokenizes IFS text into a structured token stream" begin
    input = """
    MyFractal {; Some documentation
    ; More docs
      0.5 0.0 0.0 0.5 0.0 0.0 0.6
     -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
    }
    """
    tokens = lex_ifs(input)
    kinds = [t.kind for t in tokens]
    @test :NAME in kinds
    @test :LBRACE in kinds
    @test :DOCS in kinds
    @test :ARRAY in kinds
    @test :RBRACE in kinds

    name_tok = first(t for t in tokens if t.kind == :NAME)
    @test name_tok.value == "MyFractal"

    array_toks = filter(t -> t.kind == :ARRAY, tokens)
    @test length(array_toks) == 2
    @test length(array_toks[1].value) == 7
    @test array_toks[1].value[1] ≈ 0.5
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
    mktempdir() do d
        old = pwd()
        cd(d)
        try
            out1 = render(SMALL_EQ; method=Chaos, npoints=500, resolution=(16, 16), outpath="output.png")
            @test out1.outpath == joinpath("media", "output.png")
            @test isfile(out1.outpath)

            out2 = render(SMALL_EQ; method=Chaos, npoints=500, resolution=(16, 16), outpath=joinpath("media", "nested", "x.png"))
            @test out2.outpath == joinpath("media", "nested", "x.png")
            @test isfile(out2.outpath)

            out3 = render(SMALL_EQ; method=Chaos, npoints=500, resolution=(16, 16), outpath=joinpath("other", "path", "image.png"))
            @test out3.outpath == joinpath("media", "image.png")
            @test isfile(out3.outpath)

            @test isdir("media")
        finally
            cd(old)
        end
    end
end

@testset "Affine Map SVG Rendering" begin
    ifs = IFS(SMALL_EQ; npoints=50)
    mktempdir() do tmp
        suffix = randstring(8)
        svg_path = render_transformations_svg(ifs; outpath=joinpath(tmp, "maps_$suffix.svg"), width=300, height=300)
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

        svg_axis_path = render_transformations_svg(ifs;
                                                   outpath=joinpath(tmp, "maps_axis_$suffix.svg"),
                                                   width=300,
                                                   height=300,
                                                   initial_polygon=:line,
                                                   axis=true)
        @test isfile(svg_axis_path)
        svg_axis_text = read(svg_axis_path, String)
        @test occursin("data-role=\"axis-x\"", svg_axis_text)
        @test occursin("data-role=\"axis-y\"", svg_axis_text)
        @test length(collect(eachmatch(r"data-role=\"axis-x-tick\"", svg_axis_text))) >= 2
        @test length(collect(eachmatch(r"data-role=\"axis-y-tick\"", svg_axis_text))) >= 2

        png_path = render_transformations_png(ifs; outpath=joinpath(tmp, "maps_$suffix.png"), width=256, height=256)
        @test isfile(png_path)
        @test filesize(png_path) > 0
        png_img = load(png_path)
        @test alpha(png_img[1, 1]) == 0
        colored_nonzero = count(px -> alpha(px) > 0 && (red(px) != green(px) || green(px) != blue(px)), png_img)
        @test colored_nonzero > 0

        png_path_white = render_transformations_png(ifs;
                                                    outpath=joinpath(tmp, "maps_white_$suffix.png"),
                                                    width=256,
                                                    height=256,
                                                    color=false,
                                                    show_base=true)
        @test isfile(png_path_white)
        png_img_white = load(png_path_white)
        white_nonzero = count(px -> alpha(px) > 0 && red(px) == 1 && green(px) == 1 && blue(px) == 1, png_img_white)
        @test white_nonzero > 0

        eq_identity = reshape([1.0 0.0 0.0 1.0 0.0 0.0 1.0], 1, 7)
        ifs_identity = IFS(eq_identity; npoints=10)
        png_no_axis = render_transformations_png(ifs_identity;
                                                 outpath=joinpath(tmp, "maps_no_axis_$suffix.png"),
                                                 width=256,
                                                 height=256,
                                                 initial_polygon=:line,
                                                 color=false,
                                                 axis=false)
        png_with_axis = render_transformations_png(ifs_identity;
                                                   outpath=joinpath(tmp, "maps_with_axis_$suffix.png"),
                                                   width=256,
                                                   height=256,
                                                   initial_polygon=:line,
                                                   color=false,
                                                   axis=true)
        @test isfile(png_no_axis)
        @test isfile(png_with_axis)
        img_no_axis = load(png_no_axis)
        img_with_axis = load(png_with_axis)
        no_axis_nonzero = count(px -> alpha(px) > 0, img_no_axis)
        with_axis_nonzero = count(px -> alpha(px) > 0, img_with_axis)
        @test with_axis_nonzero > no_axis_nonzero + 20
    end
end

@testset "Affine Map Color Assignment By Map" begin
    ifs = IFS(EISENSTEIN; npoints=50)
    mktempdir() do tmp
        suffix = randstring(8)
        svg_path = render_transformations_svg(ifs; outpath=joinpath(tmp, "maps_colors_$suffix.svg"), width=320, height=320)
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
        m = match(r"<line x1=\"([^\"]+)\" y1=\"([^\"]+)\" x2=\"([^\"]+)\" y2=\"([^\"]+)\"", svg_text)
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

    mktempdir() do tmp
        for (deg, clockwise, label) in cases
            eq = rotation_eq(deg; clockwise=clockwise)
            ifs = IFS(eq; npoints=10)

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

            svg_path = render_transformations_svg(ifs;
                                                  outpath=joinpath(tmp, "rot_$label.svg"),
                                                  width=320,
                                                  height=320,
                                                  initial_polygon=:line)
            @test isfile(svg_path)
            svg_text = read(svg_path, String)
            x1, y1, x2, y2 = line_endpoints_from_svg(svg_text)
            dx_svg = x2 - x1
            dy_svg = y2 - y1
            @test dx_svg > 0
            @test signbit(dy_svg) != signbit(dy_math)

            png_path = render_transformations_png(ifs;
                                                  outpath=joinpath(tmp, "rot_$label.png"),
                                                  width=320,
                                                  height=320,
                                                  initial_polygon=:line,
                                                  color=false)
            @test isfile(png_path)
            img = load(png_path)
            @test count(px -> alpha(px) > 0, img) > 0
        end
    end
end

@testset "Transformation Origin Anchor Across Initial Polygons" begin
    presets = (:default, :equilateral_triangle, :line_arrow, :line)
    width, height = 320, 320
    ifs = IFS(reshape([1.0 0.0 0.0 1.0 0.0 0.0 1.0], 1, 7); npoints=10)
    mktempdir() do tmp
        centers = Tuple{Float64,Float64}[]
        for preset in presets
            out = render_transformations_png(ifs;
                                             outpath=joinpath(tmp, "origin_$(preset).png"),
                                             width=width,
                                             height=height,
                                             initial_polygon=preset,
                                             color=false)
            img = load(out)
            xs = Int[]
            ys = Int[]
            for y in axes(img, 1), x in axes(img, 2)
                if alpha(img[y, x]) > 0
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

@testset "Transformation Renders Use IFS Bounds" begin
    base_eq = reshape([1.0 0.0 0.0 1.0 0.0 0.0 1.0], 1, 7)
    shifted_eq = reshape([1.0 0.0 0.0 1.0 0.0 10.0 1.0], 1, 7)
    ifs = IFS(base_eq; npoints=10)
    shifted_ifs = IFS(shifted_eq; npoints=10)
    width, height = 320, 320

    mktempdir() do tmp
        function rendered_bbox_center_and_span(out)
            img = load(out)
            xs = Int[]
            ys = Int[]
            for y in axes(img, 1), x in axes(img, 2)
                if alpha(img[y, x]) > 0
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

        base_out = render_transformations_png(ifs;
                                              outpath=joinpath(tmp, "line_arrow_base.png"),
                                              width=width,
                                              height=height,
                                              initial_polygon=:line_arrow,
                                              color=false)
        shifted_out = render_transformations_png(shifted_ifs;
                                                 outpath=joinpath(tmp, "line_arrow_shifted.png"),
                                                 width=width,
                                                 height=height,
                                                 initial_polygon=:line_arrow,
                                                 color=false)
        @test isfile(base_out)
        @test isfile(shifted_out)
        bx, by, bspan = rendered_bbox_center_and_span(base_out)
        sx, sy, sspan = rendered_bbox_center_and_span(shifted_out)
        @test abs(bx - sx) <= 3.0
        @test abs(by - sy) <= 3.0
        @test abs(bspan - sspan) <= 3.0
    end
end

@testset "Transformation SVG-PNG Endpoint Consistency" begin
    function line_endpoints_from_svg(svg_text::AbstractString)
        m = match(r"<line x1=\"([^\"]+)\" y1=\"([^\"]+)\" x2=\"([^\"]+)\" y2=\"([^\"]+)\"", svg_text)
        m === nothing && error("No SVG line found")
        x1 = parse(Float64, m.captures[1])
        y1 = parse(Float64, m.captures[2])
        x2 = parse(Float64, m.captures[3])
        y2 = parse(Float64, m.captures[4])
        return (x1, y1), (x2, y2)
    end

    eq = reshape([cos(pi / 6) sin(pi / 6) -sin(pi / 6) cos(pi / 6) 0.0 0.0 1.0], 1, 7)
    ifs = IFS(eq; npoints=10)
    width, height = 320, 320

    mktempdir() do tmp
        svg_path = render_transformations_svg(ifs;
                                              outpath=joinpath(tmp, "rot.svg"),
                                              width=width,
                                              height=height,
                                              initial_polygon=:line)
        @test isfile(svg_path)
        svg_text = read(svg_path, String)
        got1, got2 = line_endpoints_from_svg(svg_text)

        png_path = render_transformations_png(ifs;
                                              outpath=joinpath(tmp, "rot.png"),
                                              width=width,
                                              height=height,
                                              initial_polygon=:line,
                                              color=false)
        @test isfile(png_path)
        img = load(png_path)
        pixels = Tuple{Float64,Float64}[]
        for y in axes(img, 1), x in axes(img, 2)
            if alpha(img[y, x]) > 0
                push!(pixels, (Float64(x), Float64(y)))
            end
        end
        isempty(pixels) && error("Rendered PNG has no non-transparent pixels")

        function min_dist_to_pixels(p::Tuple{Float64,Float64}, pts::Vector{Tuple{Float64,Float64}})
            best = Inf
            for q in pts
                d = hypot(p[1] - q[1], p[2] - q[2])
                if d < best
                    best = d
                end
            end
            return best
        end

        @test min_dist_to_pixels(got1, pixels) <= 1.5
        @test min_dist_to_pixels(got2, pixels) <= 1.5
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
    @test occursin("Supported: default, equilateral_triangle, line_arrow, line", sprint(showerror, err))

    ifs = IFS(SMALL_EQ; npoints=20)
    mktempdir() do tmp
        suffix = randstring(8)

        png_triangle = render_transformations_png(ifs;
                                                  outpath=joinpath(tmp, "maps_triangle_$suffix.png"),
                                                  width=192,
                                                  height=192,
                                                  initial_polygon=:equilateral_triangle,
                                                  color=false)
        @test isfile(png_triangle)
        tri_img = load(png_triangle)
        @test count(px -> alpha(px) > 0, tri_img) > 0

        png_line = render_transformations_png(ifs;
                                              outpath=joinpath(tmp, "maps_line_$suffix.png"),
                                              width=192,
                                              height=192,
                                              initial_polygon=:line,
                                              color=false)
        @test isfile(png_line)
        line_img = load(png_line)
        @test count(px -> alpha(px) > 0, line_img) > 0

        @test_throws ArgumentError render_transformations_png(ifs;
                                                              outpath=joinpath(tmp, "never_write_badpoly_$suffix.png"),
                                                              initial_polygon=:not_a_polygon)
        @test_throws MethodError render_transformations_png(ifs;
                                                            outpath=joinpath(tmp, "never_write_badkw_$suffix.png"),
                                                            limits_mode=:fit)

        seed_line = render(SMALL_EQ;
                           method=:image_iterate,
                           image_source=:polygon,
                           image_iterations=1,
                           initial_polygon=:line,
                           resolution=(64, 64),
                           outpath=joinpath(tmp, "seed_line_$suffix.png"))
        seed_triangle = render(SMALL_EQ;
                               method=:image_iterate,
                               image_source=:polygon,
                               image_iterations=1,
                               initial_polygon=:equilateral_triangle,
                               resolution=(64, 64),
                               outpath=joinpath(tmp, "seed_triangle_$suffix.png"))
        @test count(>(Gray{Float32}(0)), seed_line.image) > 0
        @test count(>(Gray{Float32}(0)), seed_triangle.image) > 0
        @test seed_line.image != seed_triangle.image
    end
end

@testset "Render Entrypoint" begin
    mktempdir() do tmp
        suffix = randstring(8)

        out1 = render(SMALL_EQ; npoints=5000, method=:chaos, resolution=(64, 64), outpath=joinpath(tmp, "render_matrix_$suffix.png"))
        @test isfile(out1.outpath)
        @test size(out1.image) == (64, 64)

        ifs = IFS(SMALL_EQ; npoints=2000)
        out2 = render(ifs; method=:point_deterministic, iterations=1, resolution=(48, 48), outpath=joinpath(tmp, "render_ifs_$suffix.png"))
        @test isfile(out2.outpath)
        @test size(out2.image) == (48, 48)
        @test length(out2.ifs.points) == length(ifs.points) * length(ifs.maps)

        text = """
        RenderTest {
          0.5 0.0 0.0 0.5 0.0 0.0 0.6
         -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
        }
        """
        out3 = render(text; npoints=4000, method=:parallel, resolution=(40, 40), outpath=joinpath(tmp, "render_text_$suffix.png"))
        @test isfile(out3.outpath)
        @test size(out3.image) == (40, 40)
        @test out3.method == :chaos

        out3b = render(text; npoints=2000, method="point_deterministic", iterations=1, resolution=(24, 24), outpath=joinpath(tmp, "render_text_string_method_$suffix.png"))
        @test isfile(out3b.outpath)
        @test size(out3b.image) == (24, 24)
        @test out3b.method == :point_deterministic

        out3c = render(text; npoints=1500, method=Parallel, resolution=(20, 20), outpath=joinpath(tmp, "render_text_enum_method_$suffix.png"))
        @test isfile(out3c.outpath)
        @test size(out3c.image) == (20, 20)
        @test out3c.method == :chaos

        out_tf = render(text;
                        npoints=10,
                        method=:render_transformations,
                        resolution=(48, 48),
                        initial_polygon=:line_arrow,
                        color=true,
                        outpath=joinpath(tmp, "render_transformations_$suffix.png"))
        @test isfile(out_tf.outpath)
        @test size(out_tf.image) == (48, 48)
        @test out_tf.method == :render_transformations
        tf_nonzero = count(px -> alpha(px) > 0 && (red(px) != green(px) || green(px) != blue(px)), out_tf.image)
        @test tf_nonzero > 0

        mktemp() do path, io
            write(io, text)
            close(io)
            out4 = render(path; npoints=3000, method=:inverse, ifs_index=1, iterations=2, resolution=(32, 32), outpath=joinpath(tmp, "render_file_$suffix.png"))
            @test isfile(out4.outpath)
            @test size(out4.image) == (32, 32)

            out4_hide = render(path; npoints=3000, method=:inverse, ifs_index=1, iterations=2, show_divergence_scale=false, resolution=(32, 32), outpath=joinpath(tmp, "render_file_hide_$suffix.png"))
            @test isfile(out4_hide.outpath)
            @test size(out4_hide.image) == (32, 32)
            @test all(v -> v == 0.0f0 || v == 1.0f0, out4_hide.image)
        end
    end

    @test_throws ArgumentError render(SMALL_EQ; method=:badmethod)
    @test_throws ArgumentError render(SMALL_EQ; method="not_a_method")
    @test_throws ArgumentError render(SMALL_EQ; method=:deterministic)

    # render() must delegate to render_*() — verify output parity for deterministic methods
    mktempdir() do tmp
        res = (32, 32)
        ifs = IFS(SMALL_EQ; npoints=50)

        # Inverse is deterministic given the same IFS limits
        c = render(ifs; method=:inverse, iterations=2, resolution=res, outpath=joinpath(tmp, "parity_inv_render.png"))
        d = render_inverse(ifs; iterations=2, resolution=res, outpath=joinpath(tmp, "parity_inv_direct.png"))
        @test c.image == d.image
        @test c.method == d.method

        # RenderTransformations is fully deterministic
        e = render(ifs; method=:render_transformations, resolution=res, outpath=joinpath(tmp, "parity_tf_render.png"))
        f = render_transformations(ifs; resolution=res, outpath=joinpath(tmp, "parity_tf_direct.png"))
        @test e.image == f.image
        @test e.method == f.method
    end
end

@testset "Render Image Iterate API" begin
    mktempdir() do tmp
        suffix = randstring(8)

        out_poly = render(SMALL_EQ;
                          method=:image_iterate,
                          image_source=:polygon,
                          image_iterations=2,
                          resolution=(48, 48),
                          outpath=joinpath(tmp, "render_image_poly_$suffix.png"))
        @test isfile(out_poly.outpath)
        @test size(out_poly.image) == (48, 48)
        @test out_poly.method == :image_iterate
        @test eltype(out_poly.image) <: Gray

        out_chaos = render(SMALL_EQ;
                           method=ImageIterate,
                           image_source=:chaos,
                           image_iterations=2,
                           npoints=400,
                           warmup=5,
                           resolution=(40, 40),
                           outpath=joinpath(tmp, "render_image_chaos_$suffix.png"))
        @test isfile(out_chaos.outpath)
        @test size(out_chaos.image) == (40, 40)

        p = joinpath(tmp, "seed.png")
        src = fill(Gray{Float32}(0.0f0), 20, 20)
        src[6:15, 6:15] .= Gray{Float32}(1.0f0)
        save(p, src)
        out_file = render(SMALL_EQ;
                          method=:image_iterate,
                          image_source=:file,
                          image_path=p,
                          image_iterations=2,
                          resolution=(20, 20),
                          outpath=joinpath(tmp, "render_image_file_$suffix.png"))
        @test isfile(out_file.outpath)
        @test size(out_file.image) == (20, 20)

        @test_throws ArgumentError render(SMALL_EQ; method=:image_iterate, color=true, resolution=(24, 24), outpath=joinpath(tmp, "never_write_$suffix.png"))
        @test_throws ArgumentError render(SMALL_EQ; method=:image_iterate, image_source=:file, image_path=nothing, resolution=(24, 24), outpath=joinpath(tmp, "never_write2_$suffix.png"))
        @test_throws ArgumentError render(SMALL_EQ; method=:image_iterate, polygon_limits_mode=:bad_mode, resolution=(24, 24), outpath=joinpath(tmp, "never_write3_$suffix.png"))
    end

    # Polygon-backed image iteration should remain sparse line art and preserve the
    # documented :default -> :ifs compatibility behavior through the public API.
    mktempdir() do tmp
        poly_seed = render(SMALL_EQ;
                           method=:image_iterate,
                           image_source=:polygon,
                           image_iterations=1,
                           polygon_limits_mode=:ifs,
                           resolution=(64, 64),
                           outpath=joinpath(tmp, "poly_seed.png"))
        density = count(>(Gray{Float32}(0)), poly_seed.image) / length(poly_seed.image)
        @test density > 0.005
        @test density < 0.25

        poly_seed_default = @test_logs (:warn, r"polygon_limits_mode=:default is treated as :ifs") render(
            SMALL_EQ;
            method=:image_iterate,
            image_source=:polygon,
            image_iterations=1,
            polygon_limits_mode=:default,
            resolution=(64, 64),
            outpath=joinpath(tmp, "poly_seed_default.png")
        )
        @test count(>(Gray{Float32}(0)), poly_seed_default.image) > 0
        @test poly_seed_default.image == poly_seed.image
    end

end

@testset "Snapshot Image Tests" begin
    transform = render(SMALL_EQ;
                       npoints=50,
                       method=RenderTransformations,
                       resolution=(96, 96),
                       initial_polygon=:line_arrow,
                       color=true,
                       axis=true,
                       outpath=joinpath("media", "snapshot_transformations.png"))
    _assert_matches_snapshot("transformations-line-arrow-color-axis", transform.image)

    inverse = render(SMALL_EQ;
                     method=Inverse,
                     iterations=2,
                     show_divergence_scale=false,
                     backend=:cpu,
                     resolution=(32, 32),
                     outpath=joinpath("media", "snapshot_inverse.png"))
    _assert_matches_snapshot("inverse-mask-small", inverse.image)

    image_iter = render(SMALL_EQ;
                        method=ImageIterate,
                        image_source=:polygon,
                        image_iterations=2,
                        backend=:cpu,
                        resolution=(48, 48),
                        outpath=joinpath("media", "snapshot_image_iterate.png"))
    _assert_matches_snapshot("image-iterate-polygon-small", image_iter.image)
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
        mktempdir() do tmp
            suffix = randstring(8)
            out_idx = render(path; ifs_index=2, npoints=1500, method=:chaos, resolution=(24, 24), outpath=joinpath(tmp, "render_multi_idx_$suffix.png"))
            @test out_idx.ifs.name == "SecondIFS"
            @test isfile(out_idx.outpath)

            out_name = render(path; ifs_name="FirstIFS", npoints=1500, method=:chaos, resolution=(24, 24), outpath=joinpath(tmp, "render_multi_name_$suffix.png"))
            @test out_name.ifs.name == "FirstIFS"
            @test isfile(out_name.outpath)

            @test_throws ArgumentError render(path; ifs_index=99, npoints=100, method=:chaos, resolution=(16, 16), outpath=joinpath(tmp, "never_written_$suffix.png"))
            @test_throws ArgumentError render(path; ifs_name="MissingIFS", npoints=100, method=:chaos, resolution=(16, 16), outpath=joinpath(tmp, "never_written2_$suffix.png"))
            @test_throws ArgumentError render(path; ifs_name="FirstIFS", ifs_index=1, npoints=100, method=:chaos, resolution=(16, 16), outpath=joinpath(tmp, "never_written3_$suffix.png"))
        end
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

    @test_throws ArgumentError render(SMALL_EQ;
                                      method=:chaos,
                                      backend=:bad,
                                      resolution=(24, 24),
                                      outpath=joinpath("media", "never_write_backend_bad.png"))
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
                             method=PointDeterministic,
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

if _env_flag("FRACTALS_RUN_CLI_BENCH_TESTS")
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

            render_tf_out = read(`$jcmd --startup-file=no --project=$project $script render --input $ifs_path --ifs-index 1 --method RenderTransformations --color true --initial-polygon line_arrow --resolution 24x24 --out media/cli_transformations.png`, String)
            @test occursin("Rendered", render_tf_out)
            tf_img = load(joinpath("media", "cli_transformations.png"))
            @test count(px -> alpha(px) > 0 && (red(px) != green(px) || green(px) != blue(px)), tf_img) > 0

            render_inverse_out = read(`$jcmd --startup-file=no --project=$project $script render --input $ifs_path --ifs-index 1 --method Inverse --iterations 2 --show-divergence-scale false --resolution 24x24 --out media/cli_inverse_mask.png`, String)
            @test occursin("Rendered", render_inverse_out)
            inverse_img = load(joinpath("media", "cli_inverse_mask.png"))
            @test eltype(inverse_img) <: Gray
            @test all(px -> Float32(px.val) == 0 || Float32(px.val) == 1, inverse_img)

            batch_out = read(`$jcmd --startup-file=no --project=$project $script batch-render --input $ifs_path --method Inverse --iterations 2 --show-divergence-scale false --resolution 24x24 --out-dir media/batch`, String)
            @test occursin("Batch rendering 2 definitions", batch_out)
            @test isfile(joinpath("media", "batch", "01_cli_first.png"))
            @test isfile(joinpath("media", "batch", "02_cli_second.png"))
            batch_img = load(joinpath("media", "batch", "01_cli_first.png"))
            @test eltype(batch_img) <: Gray
            @test all(px -> Float32(px.val) == 0 || Float32(px.val) == 1, batch_img)

            batch_tf_out = read(`$jcmd --startup-file=no --project=$project $script batch-render --input $ifs_path --method RenderTransformations --color true --initial-polygon line_arrow --resolution 24x24 --out-dir media/batch_transformations`, String)
            @test occursin("Batch rendering 2 definitions", batch_tf_out)
            batch_tf_img = load(joinpath("media", "batch_transformations", "01_cli_first.png"))
            @test count(px -> alpha(px) > 0 && (red(px) != green(px) || green(px) != blue(px)), batch_tf_img) > 0

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
            for k in (:iterate!, Symbol("iterate_parallel!"), :make_image, :iterate_image, :rasterize_image_inversely)
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

            bench_gpu_json = joinpath("benchmarks", "bench_small_gpu_lines.json")
            bench_gpu_out = read(`$jcmd --startup-file=no --project=$project $script benchmark --profile small --repeats 1 --npoints 500 --resolution 12x12 --inverse-iterations 1 --backend auto --include-gpu-bench --json $bench_gpu_json`, String)
            @test occursin("Wrote benchmark JSON", bench_gpu_out)
            gpu_payload = JSON3.read(read(bench_gpu_json, String))
            gpu_small = gpu_payload.results.small
            for k in (:make_image_gpu, :iterate_image_gpu, :rasterize_image_inversely_gpu_exact, :rasterize_image_inversely_gpu_preview)
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
else
    @info "Skipping CLI Commands testset; set FRACTALS_RUN_CLI_BENCH_TESTS=1 to enable it."
end
