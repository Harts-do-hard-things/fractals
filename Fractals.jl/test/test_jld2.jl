# JLD2 data records: cached point clouds, resuming a chaos game, and the catalog.
# Included from runtests.jl, which points FRACTALS_OUTPUT_ROOT at a temp directory.
using JLD2

@testset "JLD2 point clouds and catalog" begin
    _with_tmpdir() do root
        with_output_root(root) do
            ifs = IFS(SMALL_EQ; name = "Small Eq")

            @testset "cached_points records" begin
                m = Chaos(; npoints = 300, warmup = 10, seed = 7)
                pts, path, cached = cached_points(m, ifs)
                @test !cached
                @test length(pts) == 300
                @test dirname(path) == data_output_dir("points")
                @test startswith(basename(path), "chaos_")
                @test occursin(ifs_key(ifs), basename(path))
                rec = JLD2.load(path)
                @test size(rec["points"]) == (2, 300)
                @test rec["method"] == m
                @test rec["ifs"] == ifs_key(ifs)

                pts2, path2, cached2 = cached_points(m, ifs)
                @test cached2
                @test path2 == path
                @test pts2 == pts

                # A different seed or IFS is a different entry.
                _, other, _ = cached_points(Chaos(; npoints = 300, warmup = 10, seed = 8), ifs)
                @test other != path
                _, other_ifs, _ = cached_points(m, IFS(SMALL_EQ .* [1 1 1 1 0.5 1 1]))
                @test other_ifs != path
            end

            @testset "extend_fractal grows the cached chaos game" begin
                m = Chaos(; npoints = 200, warmup = 10, seed = 5)
                old, old_path, _ = cached_points(m, ifs)
                pts, path = extend_fractal(m, ifs, 100; seed = 11)
                @test length(pts) == 300
                @test pts[1:200] == old
                @test path != old_path
                @test isfile(old_path) # original kept
                # The chain continues from the last stored point.
                @test any(f -> f(old[end]) ≈ pts[201], ifs.maps)

                # The extension is the cache entry for the larger Chaos config.
                bigger = Chaos(; npoints = 300, warmup = 10, seed = 5)
                again, again_path, cached = cached_points(bigger, ifs)
                @test cached
                @test again_path == path
                @test again == pts

                # Seeded extensions are reproducible, and an existing entry is overwritten.
                pts_b, path_b = extend_fractal(m, ifs, 100; seed = 11)
                @test path_b == path
                @test pts_b == pts

                # Extending the extension accumulates.
                more, _ = extend_fractal(bigger, ifs, 50; seed = 12)
                @test length(more) == 350
                @test more[1:300] == pts
                @test cached_points(Chaos(; npoints = 350, warmup = 10, seed = 5), ifs)[3]

                # Rendering at the new count reuses the extended cloud.
                before = mtime(path)
                out = render(ifs, bigger; resolution = (16, 16))
                @test mtime(path) == before
                @test size(out.image) == (16, 16)

                @test_throws ArgumentError extend_fractal(m, ifs, 0)
            end

            @testset "fractal_catalog" begin
                render(ifs, Chaos(; npoints = 200, warmup = 10, seed = 5); resolution = (16, 16))
                rows = fractal_catalog()
                nrows = length(rows)
                @test rows isa Vector{<:NamedTuple}
                @test all(r -> isfile(r.path), rows)
                @test issorted([r.path for r in rows])
                @test all(r -> r.gitcommit isa Union{Nothing,String}, rows)
                @test any(r -> r.ifs_key == ifs_key(ifs), rows)

                rows = filter(r -> r.ifs_key == ifs_key(ifs), rows)
                pts_rows = filter(r -> r.kind == :points, rows)
                @test !isempty(pts_rows)
                @test all(r -> r.method == "chaos" && ismissing(r.resolution), pts_rows)
                @test any(r -> r.npoints == 350, pts_rows)

                ren_rows = filter(r -> r.kind == :render, rows)
                @test length(ren_rows) >= 1
                @test all(r -> r.name == "Small Eq" && r.resolution == (16, 16), ren_rows)
                @test any(r -> r.npoints == 200, ren_rows)

                # Unreadable files are skipped, and a missing directory is empty.
                write(joinpath(data_output_dir("points"), "junk.jld2"), "not jld2")
                @test length(fractal_catalog()) == nrows
                @test isempty(fractal_catalog(; dir = joinpath(root, "nope")))
            end
        end
    end
end
