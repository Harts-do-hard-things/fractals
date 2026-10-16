# --------------------------------
# Precompile workload
# --------------------------------
#
# Exercises the common paths on tiny inputs so the first `render` call (and the test suite) does
# not pay for compiling them. It uses the same shapes of call that users make (explicit `outpath`
# strings, symbol/string/enum methods, `.ifs` file inputs, color/alpha), since each keyword
# combination compiles its own `kwcall` method.
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
    ifs = IFS(eq; npoints = 50)
    # Run `f` with output redirected into `scratch`.
    inroot(f) =
        with(_SKIP_GIT => true) do
            with_output_root(f, scratch)
        end

    # Parsing and the chaos game.
    @compile_workload begin
        parse_ifs_string(ifs_text; npoints = 50)
        parse_ifs_file(ifs_path; npoints = 50)
        iterate!(ifs; warmup = 2)
        iterate!(ifs; warmup = 2, seed = 1)
        make_image(ifs; resolution = res)
        iterate_image(ifs, make_image(ifs; resolution = res); colors = true, seed = 1)
        random_eq(; rng = MersenneTwister(1))
        AffineMap([0.5 0.0; 0.0 0.5], [0.0, 0.0])
    end

    # `render` for matrix, IFS, and file inputs. PNG saving compiles once per pixel type
    # (Gray, GrayA, RGB, RGBA).
    @compile_workload begin
        inroot() do
            render(
                eq;
                method = Chaos,
                resolution = res,
                outpath = "chaos_a.png",
                npoints = 50,
                warmup = 2,
            )
            render(
                eq;
                method = "chaos",
                resolution = res,
                outpath = "chaos_c.png",
                color = true,
                alpha = true,
                npoints = 50,
                warmup = 2,
            )
            render(eq; resolution = res, outpath = "gray.png", npoints = 50, alpha = true)
            render(eq; resolution = res, outpath = "rgb.png", npoints = 50, color = true)
            render(IFS(eq; npoints = 20); method = Chaos, resolution = res, outpath = "ifs.png")
            render(ifs_path; ifs_index = 1, npoints = 50, resolution = res, outpath = "file.png")
            render(
                ifs_path;
                ifs_name = "Workload_Two",
                npoints = 50,
                resolution = res,
                outpath = "file_two.png",
            )
            read_provenance(joinpath(output_dir(), "chaos_a.png"))
        end
    end

    # One render per remaining method, each in its own block.
    @compile_workload begin
        inroot() do
            render(
                eq;
                method = PointDeterministic,
                resolution = res,
                outpath = "pd.png",
                npoints = 8,
                iterations = 1,
            )
        end
    end
    @compile_workload begin
        inroot() do
            render(eq; method = Inverse, resolution = res, outpath = "inverse.png", iterations = 1)
        end
    end
    @compile_workload begin
        inroot() do
            render(
                eq;
                method = ImageIterate,
                resolution = res,
                outpath = "ii.png",
                image_source = :polygon,
                image_iterations = 1,
            )
            render(
                eq;
                method = ImageIterate,
                image_source = :file,
                image_path = joinpath(output_dir(), "chaos_a.png"),
                image_iterations = 1,
                resolution = res,
                outpath = "ii_file.png",
            )
        end
    end
    @compile_workload begin
        inroot() do
            render(
                eq;
                method = RenderTransformations,
                resolution = res,
                outpath = "rt.png",
                npoints = 8,
            )
            render_transformations_svg(
                IFS(eq; npoints = 8);
                outpath = "t.svg",
                width = 16,
                height = 16,
            )
            render_transformations_png(
                IFS(eq; npoints = 8);
                outpath = "t.png",
                width = 16,
                height = 16,
            )
        end
    end

    # Interpolation frames and animation export (ffmpeg is not required: the missing binary is
    # reported after the export code has been compiled).
    @compile_workload begin
        inroot() do
            left = IFS(eq; npoints = 20, name = "left")
            right = IFS(eq .* 0.9; npoints = 20, name = "right")
            render_interpolation_frames(
                left,
                right;
                frames = 2,
                outdir = joinpath(scratch, "frames"),
                resolution = res,
            )
            try
                export_animation(
                    :gif;
                    frames_dir = joinpath(scratch, "frames"),
                    basename = "left_to_right",
                    ffmpeg_cmd = "definitely_missing_ffmpeg_binary",
                )
            catch
            end
        end
    end

    # JLD2 records.
    @compile_workload begin
        inroot() do
            data = fractal_data(eq; npoints = 50, seed = 1, name = "w")
            save_fractal(ifs; image = make_image(ifs; resolution = res), seed = 1)
            load_fractal(data.path)
            render_fractal(data; resolution = res)
            extend_fractal(data, 10; seed = 2)
            fractal_catalog()
        end
    end

    rm(scratch; recursive = true, force = true)
end
