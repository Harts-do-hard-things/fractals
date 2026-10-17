# JLD2 data records: save/load, produce_or_load caching, resuming, catalog.
# Included from runtests.jl, which points FRACTALS_OUTPUT_ROOT at a temp directory.
using JLD2
using DrWatson: wload

@testset "JLD2 fractal records" begin
    root = mktempdir(; prefix = "fractals_jld2_")
    with_output_root(root) do
        @testset "save_fractal / load_fractal round trip" begin
            ifs = IFS(SMALL_EQ; name = "Small Eq")
            ifs_pts = chaos_game(ifs, 500; warmup = 10, seed = 3)
            img = make_image(ifs_pts, ifs.limits; resolution = (16, 16))
            path = save_fractal(
                ifs,
                ifs_pts;
                image = img,
                method = :chaos,
                seed = 3,
                params = (; warmup = 10),
            )
            @test isfile(path)
            @test startswith(path, data_output_dir())
            @test endswith(path, ".jld2")

            # Loadable without Fractals types: plain arrays only.
            raw = wload(path)
            @test raw["points"] isa Matrix{Float64}
            @test size(raw["points"]) == (2, 500)
            @test size(raw["eq"]) == (2, 7)
            @test raw["image"] == img
            @test raw["seed"] == 3
            @test haskey(raw, "gitcommit")

            loaded = load_fractal(path)
            @test loaded.method == :chaos
            @test loaded.seed == 3
            @test loaded.image == img
            @test loaded.params["warmup"] == 10
            @test loaded.ifs.name == "Small Eq"
            @test loaded.points == ifs_pts
            @test loaded.ifs.limits == ifs.limits
            @test collect(loaded.ifs.weights) == collect(ifs.weights)
            @test [m.A for m in loaded.ifs.maps] == [m.A for m in ifs.maps]
            @test [m.b for m in loaded.ifs.maps] == [m.b for m in ifs.maps]
            # Relative paths resolve under data_output_dir().
            @test load_fractal(basename(path)).path == path

            # Saving again keeps the old file as a backup.
            path2 = save_fractal(ifs, ifs_pts; method = :chaos, seed = 3)
            @test path2 == path
            @test isfile(replace(path, ".jld2" => "_#1.jld2"))

            # Files that are not fractal records are rejected.
            foreign = joinpath(root, "other.jld2")
            JLD2.save(foreign, "x", 1)
            @test_throws ArgumentError load_fractal(foreign)
        end

        @testset "fractal_data caches with produce_or_load" begin
            a = fractal_data(SMALL_EQ; npoints = 300, seed = 7, name = "cache")
            @test !a.cached
            @test isfile(a.path)
            @test length(a.points) == 300
            mtime_a = mtime(a.path)

            b = fractal_data(SMALL_EQ; npoints = 300, seed = 7, name = "cache")
            @test b.cached
            @test b.path == a.path
            @test b.points == a.points
            @test mtime(b.path) == mtime_a
            @test !isfile(replace(a.path, ".jld2" => "_#1.jld2"))

            # Different parameters or a different system get different files.
            c = fractal_data(SMALL_EQ; npoints = 300, seed = 8, name = "cache")
            @test !c.cached && c.path != a.path
            other_eq = copy(SMALL_EQ)
            other_eq[1, 5] = 0.25
            d = fractal_data(other_eq; npoints = 300, seed = 7, name = "cache")
            @test !d.cached && d.path != a.path

            # force=true recomputes; same seed gives the same points.
            e = fractal_data(SMALL_EQ; npoints = 300, seed = 7, name = "cache", force = true)
            @test !e.cached
            @test e.points == a.points

            det =
                fractal_data(SMALL_EQ; method = :point_deterministic, npoints = 20, iterations = 2)
            @test length(det.points) == 20 * 2^2
            @test fractal_data(
                SMALL_EQ;
                method = :point_deterministic,
                npoints = 20,
                iterations = 2,
            ).cached
            @test_throws ArgumentError fractal_data(SMALL_EQ; method = :inverse)
        end

        @testset "saved points rasterize without recomputing" begin
            data = fractal_data(SMALL_EQ; npoints = 400, seed = 1, name = "render")
            before = mtime(data.path)
            img1 = make_image(data.points, data.ifs.limits; resolution = (16, 16))
            loaded = load_fractal(data.path)
            img2 = make_image(loaded.points, loaded.ifs.limits; resolution = (24, 32))
            @test size(img1) == (16, 16)
            @test size(img2) == (24, 32)
            @test loaded.points == data.points
            @test maximum(img1) > 0.0f0
            @test mtime(data.path) == before
        end

        @testset "extend_fractal accumulates the chaos game" begin
            data = fractal_data(SMALL_EQ; npoints = 200, seed = 5, name = "grow")
            ext = extend_fractal(data, 100; seed = 11)
            @test length(ext.points) == 300
            @test ext.points[1:200] == data.points
            @test ext.path != data.path
            @test isfile(data.path) # original kept
            # The chain continues from the last saved point.
            @test any(m -> m(data.points[end]) ≈ ext.points[201], data.ifs.maps)

            again = extend_fractal(ext.path, 50; seed = 12)
            @test length(again.points) == 350
            @test load_fractal(again.path).params["npoints"] == 350

            det = fractal_data(SMALL_EQ; method = :point_deterministic, npoints = 10)
            @test_throws ArgumentError extend_fractal(det, 5)
        end

        @testset "save_fractal with an image, and fractal_catalog" begin
            ifs = IFS(SMALL_EQ; name = "catalog")
            pts = chaos_game(ifs, 300; seed = 1)
            img = make_image(pts, ifs.limits; resolution = (16, 16))
            path = save_fractal(ifs, pts; image = img, method = :chaos, seed = 1)
            @test isfile(path)
            @test load_fractal(path).image == img
            rows = fractal_catalog()
            @test rows isa Vector{<:NamedTuple}
            @test length(rows) >= 6
            @test all(r -> isfile(r.path), rows)
            @test any(r -> r.name == "grow" && r.npoints == 300, rows)
            @test any(r -> r.has_image && r.method == :chaos, rows)
            @test all(r -> r.gitcommit isa Union{Nothing,String}, rows)
            @test isempty(fractal_catalog(joinpath(root, "nope")))
            @test issorted([r.created for r in rows])
        end
    end
end
