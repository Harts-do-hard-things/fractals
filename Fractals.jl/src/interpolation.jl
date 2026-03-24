# -------------------------------------------------
# IFS interpolation helpers
# -------------------------------------------------

function _normalized_probs(weights)
    probs = Float64.(collect(weights))
    total = sum(probs)
    total > 0 || throw(ArgumentError("IFS weights must have positive total weight"))
    return probs ./ total
end

function _eq_matrix_for_interpolation(eq::AbstractMatrix{<:Real})
    _validate_eq_matrix(eq)
    out = Matrix{Float64}(undef, size(eq, 1), 7)
    out[:, 1:size(eq, 2)] .= Float64.(eq)

    if size(eq, 2) == 6
        _, weights = _build_maps_and_weights(eq)
        out[:, 7] .= _normalized_probs(weights)
    else
        out[:, 7] .= Float64.(eq[:, 7])
        total = sum(out[:, 7])
        out[:, 7] ./= total
    end

    return out
end

function _eq_matrix_for_interpolation(ifs::IFS)
    n = length(ifs.maps)
    eq = Matrix{Float64}(undef, n, 7)
    probs = _normalized_probs(ifs.weights)

    for i in 1:n
        m = ifs.maps[i]
        eq[i, 1] = m.A[1, 1]
        eq[i, 2] = m.A[1, 2]
        eq[i, 3] = m.A[2, 1]
        eq[i, 4] = m.A[2, 2]
        eq[i, 5] = m.b[1]
        eq[i, 6] = m.b[2]
        eq[i, 7] = probs[i]
    end

    return eq
end

function _interpolate_limits(
    left::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}},
    right::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}},
    t::Float64,
)
    return (
        (
            (1 - t) * left[1][1] + t * right[1][1],
            (1 - t) * left[1][2] + t * right[1][2],
        ),
        (
            (1 - t) * left[2][1] + t * right[2][1],
            (1 - t) * left[2][2] + t * right[2][2],
        ),
    )
end

"""
    interpolate_eq_matrix(left, right, t)

Linearly interpolate between two compatible IFS equation matrices and return a
normalized 7-column matrix. Six-column inputs derive a probability column from
their affine determinants before interpolation.
"""
function interpolate_eq_matrix(
    left::AbstractMatrix{<:Real},
    right::AbstractMatrix{<:Real},
    t::Real,
)
    0 <= t <= 1 || throw(ArgumentError("Interpolation parameter t must be in [0, 1], got $t"))
    size(left, 1) == size(right, 1) ||
        throw(ArgumentError("Interpolation requires the same number of transforms, got $(size(left, 1)) and $(size(right, 1))"))

    left_eq = _eq_matrix_for_interpolation(left)
    right_eq = _eq_matrix_for_interpolation(right)
    size(left_eq) == size(right_eq) ||
        throw(ArgumentError("Interpolation requires matching equation matrix shapes, got $(size(left_eq)) and $(size(right_eq))"))

    α = Float64(t)
    out = (1 - α) .* left_eq .+ α .* right_eq
    total = sum(out[:, 7])
    total > 0 || throw(ArgumentError("Interpolated probability column must have positive total weight"))
    out[:, 7] ./= total
    return out
end

"""
    interpolate_ifs(left, right, t; npoints, name, docs, limits_mode)

Build an interpolated IFS suitable for animation or transitional rendering.
Map coefficients, translations, and normalized weights are blended linearly.
By default, viewport limits are also interpolated to avoid frame-to-frame jitter.
"""
function interpolate_ifs(
    left::IFS,
    right::IFS,
    t::Real;
    npoints::Integer=min(length(left.points), length(right.points)),
    name::AbstractString="",
    docs::AbstractString="",
    limits_mode::Symbol=:interpolate,
)
    length(left.maps) == length(right.maps) ||
        throw(ArgumentError("Interpolation requires the same number of transforms, got $(length(left.maps)) and $(length(right.maps))"))
    npoints >= 0 || throw(ArgumentError("npoints must be >= 0, got $npoints"))

    eq = interpolate_eq_matrix(_eq_matrix_for_interpolation(left), _eq_matrix_for_interpolation(right), t)
    maps, weights = _build_maps_and_weights(eq)
    α = Float64(t)

    limits =
        if limits_mode == :interpolate
            _interpolate_limits(left.limits, right.limits, α)
        elseif limits_mode == :left
            left.limits
        elseif limits_mode == :right
            right.limits
        elseif limits_mode == :recompute
            compute_limits(maps, weights)
        else
            throw(ArgumentError("Invalid limits_mode '$limits_mode'. Supported: :interpolate, :left, :right, :recompute"))
        end

    resolved_name =
        isempty(name) ?
        (left.name == right.name ? left.name : string("Interpolated(", left.name, " -> ", right.name, ")")) :
        String(name)
    resolved_docs =
        isempty(docs) ?
        "Interpolated IFS at t=$(round(α; digits=4))" :
        String(docs)

    return IFS(maps, weights; npoints=npoints, name=resolved_name, docs=resolved_docs, limits=limits)
end

function render_interpolation_frames(
    left::IFS,
    right::IFS;
    frames::Integer,
    outdir::AbstractString=joinpath(DEFAULT_MEDIA_DIR, "frames"),
    basename::AbstractString="frame",
    t_start::Real=0.0,
    t_end::Real=1.0,
    render_method::Union{RenderMethod,Symbol,AbstractString}=RenderTransformations,
    limits_mode::Symbol=:interpolate,
    npoints::Integer=min(length(left.points), length(right.points)),
    name::AbstractString="",
    docs::AbstractString="",
    kwargs...,
)
    frames > 0 || throw(ArgumentError("frames must be > 0, got $frames"))
    !isempty(strip(basename)) || throw(ArgumentError("basename must be non-empty"))
    0 <= t_start <= 1 || throw(ArgumentError("t_start must be in [0, 1], got $t_start"))
    0 <= t_end <= 1 || throw(ArgumentError("t_end must be in [0, 1], got $t_end"))

    parsed_method = _parse_render_method(render_method)
    parsed_method == RenderTransformations ||
        throw(ArgumentError("render_interpolation_frames currently supports method=RenderTransformations only"))

    mkpath(outdir)
    ts = collect(Float64, frames == 1 ? [Float64(t_start)] : LinRange(Float64(t_start), Float64(t_end), frames))
    paths = String[]
    sizehint!(paths, frames)

    for (i, t) in enumerate(ts)
        ifs = interpolate_ifs(left, right, t;
                              npoints=npoints,
                              name=name,
                              docs=docs,
                              limits_mode=limits_mode)
        path = joinpath(outdir, @sprintf("%s_%04d.png", basename, i))
        img = _render_transformations_image(ifs;
                                            width=get(kwargs, :resolution, RESOLUTION)[2],
                                            height=get(kwargs, :resolution, RESOLUTION)[1],
                                            show_base=get(kwargs, :show_base, false),
                                            initial_polygon_spec=get(kwargs, :initial_polygon, :default),
                                            color=get(kwargs, :color, false),
                                            axis=get(kwargs, :axis, false),
                                            alpha=get(kwargs, :alpha, false))
        source_limits = _render_transformations_effective_ifs(ifs;
                                                              initial_polygon_spec=get(kwargs, :initial_polygon, :default),
                                                              polygon_limits_iterations=get(kwargs, :polygon_limits_iterations, 1)).limits
        _save_image_with_source_limits(path, img, source_limits)
        push!(paths, path)
    end

    return (
        outdir=String(outdir),
        paths=paths,
        ts=ts,
        render_method=_render_method_symbol(parsed_method),
    )
end

function _animation_frame_pattern(basename::AbstractString)
    stripped = strip(String(basename))
    isempty(stripped) && throw(ArgumentError("basename must be non-empty"))
    return Regex("^" * escape_string(stripped) * "_\\d{4}\\.png\$")
end

function _animation_frame_paths(
    frames_dir::AbstractString,
    basename::AbstractString,
)
    isdir(frames_dir) || throw(ArgumentError("frames_dir '$frames_dir' does not exist"))
    pattern = _animation_frame_pattern(basename)
    matches = sort(filter(name -> occursin(pattern, name), readdir(frames_dir)))
    isempty(matches) &&
        throw(ArgumentError("No PNG frames matching '$(strip(String(basename)))_####.png' were found in '$frames_dir'"))
    first(matches) == "$(strip(String(basename)))_0001.png" ||
        throw(ArgumentError("Frame sequence in '$frames_dir' must start at '$(strip(String(basename)))_0001.png'"))
    return joinpath.(Ref(String(frames_dir)), matches)
end

function _resolve_ffmpeg(ffmpeg_cmd::AbstractString="ffmpeg")
    path = Sys.which(ffmpeg_cmd)
    isnothing(path) && throw(ArgumentError("ffmpeg executable '$ffmpeg_cmd' was not found on PATH"))
    return path
end

function _run_ffmpeg(cmd::Cmd)
    try
        run(cmd)
    catch err
        if err isa ProcessFailedException
            throw(ArgumentError("ffmpeg command failed: $(join(cmd.exec, ' '))"))
        end
        rethrow()
    end
    return nothing
end

"""
    export_animation(format; frames_dir, basename, outpath, fps=12, ffmpeg_cmd="ffmpeg")

Convert a deterministic numbered PNG frame sequence such as `basename_0001.png`
into a GIF or MP4 animation using `ffmpeg`.
"""
function export_animation(
    format::Symbol;
    frames_dir::AbstractString,
    basename::AbstractString,
    outpath::AbstractString,
    fps::Real=12,
    ffmpeg_cmd::AbstractString="ffmpeg",
)
    format in (:gif, :mp4) ||
        throw(ArgumentError("Invalid animation format '$format'. Supported: :gif, :mp4"))
    fps > 0 || throw(ArgumentError("fps must be > 0, got $fps"))

    frame_paths = _animation_frame_paths(frames_dir, basename)
    ffmpeg = _resolve_ffmpeg(ffmpeg_cmd)

    final_outpath = normpath(String(outpath))
    mkpath(dirname(final_outpath))

    input_pattern = joinpath(String(frames_dir), "$(strip(String(basename)))_%04d.png")

    if format == :gif
        palette_path = joinpath(dirname(final_outpath), "$(Base.Filesystem.basename(final_outpath)).palette.png")
        try
            _run_ffmpeg(`$ffmpeg -y -framerate $fps -i $input_pattern -vf palettegen $palette_path`)
            _run_ffmpeg(`$ffmpeg -y -framerate $fps -i $input_pattern -i $palette_path -lavfi paletteuse $final_outpath`)
        finally
            isfile(palette_path) && rm(palette_path; force=true)
        end
    else
        _run_ffmpeg(`$ffmpeg -y -framerate $fps -i $input_pattern -c:v libx264 -pix_fmt yuv420p $final_outpath`)
    end

    return (
        format=format,
        frames_dir=String(frames_dir),
        basename=strip(String(basename)),
        frame_paths=frame_paths,
        fps=Float64(fps),
        outpath=final_outpath,
    )
end
