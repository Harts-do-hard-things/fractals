# JLD2 data records: save/load, produce_or_load caching, resuming, catalog.
# Included from runtests.jl, which points FRACTALS_OUTPUT_ROOT at a temp directory.
using JLD2
using DrWatson: wload

@testset "JLD2 fractal records" begin
    root = mktempdir(; prefix = "fractals_jld2_")
    with_output_root(root) do
        @testset "save_fractal / load_fractal round trip" begin
            ifs = IFS(SMALL_EQ; npoints = 500, name = "Small Eq")
            iterate!(ifs; warmup = 10, seed = 3)
            img = make_image(ifs; resolution = (16, 16))
            path =
                save_fractal(ifs; image = img, method = :chaos, seed = 3, params = (; warmup = 10))
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
            @test loaded.ifs.points == ifs.points
            @test loaded.ifs.limits == ifs.limits
            @test collect(loaded.ifs.weights) == collect(ifs.weights)
            @test [m.A for m in loaded.ifs.maps] == [m.A for m in ifs.maps]
            @test [m.b for m in loaded.ifs.maps] == [m.b for m in ifs.maps]
            # Relative paths resolve under data_output_dir().
            @test load_fractal(basename(path)).path == path

            # Saving again keeps the old file as a backup.
            path2 = save_fractal(ifs; method = :chaos, seed = 3)
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
            @test length(a.ifs.points) == 300
            mtime_a = mtime(a.path)

            b = fractal_data(SMALL_EQ; npoints = 300, seed = 7, name = "cache")
            @test b.cached
            @test b.path == a.path
            @test b.ifs.points == a.ifs.points
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
            @test e.ifs.points == a.ifs.points

            det =
                fractal_data(SMALL_EQ; method = :point_deterministic, npoints = 20, iterations = 2)
            @test length(det.ifs.points) == 20 * 2^2
            @test fractal_data(
                SMALL_EQ;
                method = :point_deterministic,
                npoints = 20,
                iterations = 2,
            ).cached
            @test_throws ArgumentError fractal_data(SMALL_EQ; method = :inverse)
        end

        @testset "render_fractal reuses saved points" begin
            data = fractal_data(SMALL_EQ; npoints = 400, seed = 1, name = "render")
            before = mtime(data.path)
            out1 = render_fractal(data; resolution = (16, 16))
            out2 = render_fractal(data.path; resolution = (24, 32), color = true)
            @test size(out1.image) == (16, 16)
            @test size(out2.image) == (24, 32)
            @test isfile(out1.outpath) && isfile(out2.outpath)
            @test out1.outpath != out2.outpath
            @test out1.ifs.points == data.ifs.points
            @test mtime(data.path) == before
            @test read_provenance(out1.outpath).params.method == "chaos"
        end

        @testset "extend_fractal accumulates the chaos game" begin
            data = fractal_data(SMALL_EQ; npoints = 200, seed = 5, name = "grow")
            ext = extend_fractal(data, 100; seed = 11)
            @test length(ext.ifs.points) == 300
            @test ext.ifs.points[1:200] == data.ifs.points
            @test ext.path != data.path
            @test isfile(data.path) # original kept
            # The chain continues from the last saved point.
            @test any(m -> m(data.ifs.points[end]) ≈ ext.ifs.points[201], data.ifs.maps)

            again = extend_fractal(ext.path, 50; seed = 12)
            @test length(again.ifs.points) == 350
            @test load_fractal(again.path).params["npoints"] == 350

            det = fractal_data(SMALL_EQ; method = :point_deterministic, npoints = 10)
            @test_throws ArgumentError extend_fractal(det, 5)
        end

        @testset "render(save_data=true) and fractal_catalog" begin
            out = render(SMALL_EQ; npoints = 300, resolution = (16, 16), save_data = true)
            @test isfile(out.data_path)
            @test splitext(basename(out.data_path))[1] == splitext(basename(out.outpath))[1]
            @test load_fractal(out.data_path).image == out.image
            @test !haskey(render(SMALL_EQ; npoints = 300, resolution = (8, 8)), :data_path)

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
