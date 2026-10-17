# --------------------------------
# Precompile workload
# --------------------------------
#
# Exercises the common paths on tiny inputs so the first `render` call (and the test suite) does
# not pay for compiling them: one `render(::Render)` per method type, the parser, and the JLD2
# helpers.
#
# Keep it cheap to precompile:
# - Each `@compile_workload` block below is small and independent. One large block (or a loop
#   over heterogeneous keyword tuples that are splatted into `render`) makes inference of the
#   workload itself dominate the precompile time.
# - `git` is never consulted (`_SKIP_GIT`): shelling out to `gitdescribe` per saved file costs far
#   more than the compilation it would cover.
# - All output goes to a scratch directory that is removed afterwards; nothing is written into the
#   project.
#
# When adding a public render path or a common keyword combination, add a tiny call here in a
# block of its own; check `Pkg.precompile()` time before and after.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    eq = [
        0.5 0.0 0.0 0.5 0.0 0.0 0.6
        -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
    ]
    ifs_text = """
    Workload {
      0.5 0.0 0.0 0.5 0.0 0.0 0.6
     -0.5 0.0 0.0 -0.5 1.0 0.0 0.4
    }
    Workload_Two {
      0.5 -0.5 0.5 0.5 0.0 0.0
     -0.5 -0.5 0.5 -0.5 1.0 0.0
    }
    """
    scratch = mktempdir(; prefix = "fractals_precompile_")
    ifs_path = joinpath(scratch, "workload.ifs")
    write(ifs_path, ifs_text)
    res = (16, 16)
    ifs = IFS(eq)
    pts = chaos_game(ifs, 50; warmup = 2)
    # Run `f` with output redirected into `scratch`.
    inroot(f) =
        with(_SKIP_GIT => true) do
            with_output_root(f, scratch)
        end

    # Parsing and the chaos game.
    @compile_workload begin
        parse_ifs_string(ifs_text)
        parse_ifs_file(ifs_path)
        chaos_game(ifs, 50; warmup = 2)
        chaos_game(ifs, 50; warmup = 2, seed = 1)
        make_image(pts, ifs.limits; resolution = res)
        iterate_image(ifs, make_image(pts, ifs.limits; resolution = res); colors = true, seed = 1)
        random_eq(; rng = MersenneTwister(1))
        AffineMap([0.5 0.0; 0.0 0.5], [0.0, 0.0])
    end

    # One `render(::Render)` per method type; options are fields, so these cover every keyword
    # combination. PNG saving compiles once per pixel type (Gray, GrayA, RGB, RGBA).
    @compile_workload begin
        inroot() do
            render(ifs, Chaos(npoints = 50, warmup = 2); resolution = res)
            render(ifs, Chaos(npoints = 50, warmup = 2); resolution = res, alpha = true)
            render(ifs, Chaos(npoints = 50, warmup = 2); resolution = res, color = true)
            render(
                ifs,
                Chaos(npoints = 50, warmup = 2);
                resolution = res,
                color = true,
                alpha = true,
            )
        end
    end
    @compile_workload begin
        inroot() do
            render(ifs, PointDeterministic(npoints = 8, iterations = 1); resolution = res)
        end
    end
    @compile_workload begin
        inroot() do
            render(ifs, Inverse(iterations = 1); resolution = res)
        end
    end
    @compile_workload begin
        inroot() do
            out = render(ifs, ImageIterate(); resolution = res, overlays = [BasePolygon(), Axes()])
            render(ifs, ImageIterate(FileSource(out.outpath)); resolution = res)
            render(ifs, ImageIterate(Chaos(npoints = 50, warmup = 2)); resolution = res)
            load_render(out.record)
        end
    end
    @compile_workload begin
        inroot() do
            render(ifs, Transformations(); resolution = res, color = true)
            render_transformations_svg(ifs; outpath = "t.svg", width = 16, height = 16)
        end
    end

    # Interpolation frames and animation export (ffmpeg is not required: the missing binary is
    # reported after the export code has been compiled).
    @compile_workload begin
        inroot() do
            left = IFS(eq; name = "left")
            right = IFS(eq .* 0.9; name = "right")
            frames_dir = joinpath(scratch, "frames")
            render_interpolation_frames(
                left,
                right;
                frames = 2,
                outdir = frames_dir,
                resolution = res,
            )
            try
                export_animation(
                    :gif;
                    frames_dir = frames_dir,
                    basename = "frame",
                    ffmpeg_cmd = "definitely_missing_ffmpeg_binary",
                )
            catch
            end
        end
    end

    # JLD2 records.
    @compile_workload begin
        inroot() do
            extend_fractal(Chaos(; npoints = 50, seed = 1), ifs, 10; seed = 2)
            fractal_catalog()
        end
    end

    rm(scratch; recursive = true, force = true)
end
