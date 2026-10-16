# --------------------------------
# Precompile workload
# --------------------------------
#
# Exercises the common paths on tiny inputs so the first `render` call (and the test suite)
# does not pay for compiling them. It deliberately uses the same shapes of call that users make
# (explicit `outpath` strings, symbol/string/enum methods, `.ifs` file inputs, color/alpha), since
# each keyword combination compiles its own `kwcall` method. All output goes to a scratch directory
# that is removed afterwards; nothing is written into the project.

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

    @compile_workload begin
        with_output_root(scratch) do
            # Parser and IFS construction.
            parse_ifs_string(ifs_text; npoints = 50)
            parse_ifs_file(ifs_path; npoints = 50)
            ifs = IFS(eq; npoints = 50)
            IFS(eq; npoints = 50, name = "w", docs = "d")
            random_ifs(; npoints = 10)
            iterate!(ifs; warmup = 2)
            iterate!(ifs; warmup = 2, seed = 1)
            make_image(ifs; resolution = res)
            make_image(ifs; resolution = res, backend = :cpu)
            deterministic_iterate(IFS(eq; npoints = 4), 1)
            interpolate_eq_matrix(eq, eq .* 0.9, 0.5)

            # Every render method through the public entrypoint: enum, symbol, and string
            # methods; matrix, IFS, and file inputs; plain, color, alpha; default and explicit
            # output paths.
            extras = (
                (Chaos, (; npoints = 50, warmup = 2)),
                (PointDeterministic, (; npoints = 8, iterations = 1)),
                (Inverse, (; iterations = 1)),
                (ImageIterate, (; image_source = :polygon, image_iterations = 1)),
                (RenderTransformations, (; npoints = 8)),
            )
            for (method, extra) in extras
                name = lowercase(string(method))
                render(eq; method, resolution = res, outpath = "$(name)_a.png", extra...)
                render(
                    eq;
                    method = name,
                    resolution = res,
                    outpath = "$(name)_c.png",
                    color = true,
                    alpha = true,
                    extra...,
                )
            end
            render(ifs_path; ifs_index = 1, npoints = 50, resolution = res, outpath = "file_a.png")
            render(
                ifs_path;
                ifs_name = "Workload_Two",
                npoints = 50,
                resolution = res,
                outpath = "file_b.png",
            )
            render(IFS(eq; npoints = 20); method = Chaos, resolution = res, outpath = "ifs.png")

            # Re-reading a rendered PNG: provenance, source limits, and image_source = :file.
            read_provenance(joinpath(output_dir(), "chaos_a.png"))
            render(
                eq;
                method = ImageIterate,
                image_source = :file,
                image_path = joinpath(output_dir(), "chaos_a.png"),
                image_iterations = 1,
                resolution = res,
                outpath = "from_file.png",
            )

            # SVG / PNG transformation diagrams and interpolation frames.
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
            left = IFS(eq; npoints = 20, name = "left")
            right = IFS(eq .* 0.9; npoints = 20, name = "right")
            render_interpolation_frames(
                left,
                right;
                frames = 2,
                outdir = joinpath(scratch, "frames"),
                resolution = res,
            )
            render_interpolation_frames(
                left,
                right;
                frames = 2,
                outdir = joinpath(scratch, "frames_chaos"),
                render_method = Chaos,
                resolution = res,
                npoints = 20,
                warmup = 2,
            )
        end
    end
    rm(scratch; recursive = true, force = true)
end
