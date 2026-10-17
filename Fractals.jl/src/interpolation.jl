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

    for i = 1:n
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
        ((1 - t) * left[1][1] + t * right[1][1], (1 - t) * left[1][2] + t * right[1][2]),
        ((1 - t) * left[2][1] + t * right[2][1], (1 - t) * left[2][2] + t * right[2][2]),
    )
end

@inline function _normalize_interpolation_mode(mode::Symbol)
    normalized = Symbol(lowercase(String(mode)))
    normalized in (:rotation_scale, :linear) || throw(
        ArgumentError("Invalid interpolation_mode '$mode'. Supported: :rotation_scale, :linear"),
    )
    return normalized
end

@inline function _normalize_interpolation_mode(mode::AbstractString)
    return _normalize_interpolation_mode(Symbol(strip(mode)))
end

function _linearly_interpolate_eq_matrix(
    left_eq::AbstractMatrix{<:Real},
    right_eq::AbstractMatrix{<:Real},
    α::Float64,
)
    out = (1 - α) .* Float64.(left_eq) .+ α .* Float64.(right_eq)
    total = sum(out[:, 7])
    total > 0 ||
        throw(ArgumentError("Interpolated probability column must have positive total weight"))
    out[:, 7] ./= total
    return out
end

function _rotation_scale_parameters(
    A::AbstractMatrix{<:Real};
    atol::Float64 = 1e-8,
    rtol::Float64 = 1e-6,
)
    c1 = SVector{2,Float64}(A[1, 1], A[2, 1])
    c2 = SVector{2,Float64}(A[1, 2], A[2, 2])
    s1 = norm(c1)
    s2 = norm(c2)
    max_scale = max(s1, s2)
    max_scale > atol || return nothing

    tol = max(atol, rtol * max_scale)
    abs(s1 - s2) <= tol || return nothing
    abs(dot(c1, c2)) <= max(atol, rtol * max_scale^2) || return nothing

    scale = 0.5 * (s1 + s2)
    detA = det(Matrix{Float64}(A))
    detA >= -max(atol, rtol * scale^2) || return nothing

    R = Matrix{Float64}(A) ./ scale
    isapprox(R' * R, Matrix{Float64}(I, 2, 2); atol = 1e-6, rtol = 1e-6) || return nothing
    isapprox(det(R), 1.0; atol = 1e-6, rtol = 1e-6) || return nothing

    angle = atan(R[2, 1], R[1, 1])
    return (scale = scale, angle = angle)
end

@inline function _shortest_angle_delta(left::Float64, right::Float64)
    return mod(right - left + pi, 2pi) - pi
end

function _rotation_scale_matrix(scale::Float64, angle::Float64)
    c = cos(angle)
    s = sin(angle)
    return scale .* [c -s; s c]
end

function _interpolate_linear_block!(
    out::AbstractMatrix{Float64},
    left_eq::AbstractMatrix{Float64},
    right_eq::AbstractMatrix{Float64},
    α::Float64,
)
    for i = 1:size(out, 1)
        leftA = @view left_eq[i, 1:4]
        rightA = @view right_eq[i, 1:4]
        leftM = reshape(collect(leftA), 2, 2)'
        rightM = reshape(collect(rightA), 2, 2)'
        left_params = _rotation_scale_parameters(leftM)
        right_params = _rotation_scale_parameters(rightM)

        if isnothing(left_params) || isnothing(right_params)
            continue
        end

        scale = (1 - α) * left_params.scale + α * right_params.scale
        angle = left_params.angle + α * _shortest_angle_delta(left_params.angle, right_params.angle)
        interp = _rotation_scale_matrix(scale, angle)
        out[i, 1] = interp[1, 1]
        out[i, 2] = interp[1, 2]
        out[i, 3] = interp[2, 1]
        out[i, 4] = interp[2, 2]
    end
    return out
end

"""
    interpolate_eq_matrix(left, right, t; interpolation_mode=:rotation_scale)

Interpolate between two compatible IFS equation matrices and return a normalized
7-column matrix. Six-column inputs derive a probability column from their affine
determinants before interpolation.
"""
function interpolate_eq_matrix(
    left::AbstractMatrix{<:Real},
    right::AbstractMatrix{<:Real},
    t::Real;
    interpolation_mode::Union{Symbol,AbstractString} = :rotation_scale,
)
    0 <= t <= 1 || throw(ArgumentError("Interpolation parameter t must be in [0, 1], got $t"))
    size(left, 1) == size(right, 1) || throw(
        ArgumentError(
            "Interpolation requires the same number of transforms, got $(size(left, 1)) and $(size(right, 1))",
        ),
    )

    left_eq = _eq_matrix_for_interpolation(left)
    right_eq = _eq_matrix_for_interpolation(right)
    size(left_eq) == size(right_eq) || throw(
        ArgumentError(
            "Interpolation requires matching equation matrix shapes, got $(size(left_eq)) and $(size(right_eq))",
        ),
    )

    α = Float64(t)
    mode = _normalize_interpolation_mode(interpolation_mode)
    out = _linearly_interpolate_eq_matrix(left_eq, right_eq, α)
    mode == :rotation_scale && _interpolate_linear_block!(out, left_eq, right_eq, α)
    return out
end

"""
    interpolate_ifs(left, right, t; name, limits_mode, interpolation_mode=:rotation_scale)

Build an interpolated IFS suitable for animation or transitional rendering.
The linear 2x2 map components default to rotation+scale interpolation with
linear fallback; translations and normalized weights remain linearly blended.
By default, viewport limits are also interpolated to avoid frame-to-frame jitter.
"""
function interpolate_ifs(
    left::IFS,
    right::IFS,
    t::Real;
    name::AbstractString = "",
    limits_mode::Symbol = :interpolate,
    interpolation_mode::Union{Symbol,AbstractString} = :rotation_scale,
)
    length(left.maps) == length(right.maps) || throw(
        ArgumentError(
            "Interpolation requires the same number of transforms, got $(length(left.maps)) and $(length(right.maps))",
        ),
    )

    eq = interpolate_eq_matrix(
        _eq_matrix_for_interpolation(left),
        _eq_matrix_for_interpolation(right),
        t;
        interpolation_mode = interpolation_mode,
    )
    maps, weights = _build_maps_and_weights(eq)
    α = Float64(t)

    limits = if limits_mode == :interpolate
        _interpolate_limits(left.limits, right.limits, α)
    elseif limits_mode == :left
        left.limits
    elseif limits_mode == :right
        right.limits
    elseif limits_mode == :recompute
        compute_limits(maps, weights)
    else
        throw(
            ArgumentError(
                "Invalid limits_mode '$limits_mode'. Supported: :interpolate, :left, :right, :recompute",
            ),
        )
    end

    resolved_name =
        isempty(name) ?
        (
            left.name == right.name ? left.name :
            string("Interpolated(", left.name, " -> ", right.name, ")")
        ) : String(name)

    return IFS(maps, weights; name = resolved_name, limits = limits)
end

"""
    render_interpolation_frames(left, right, method = Transformations(); frames, outdir = nothing,
                                basename = "frame", t_start = 0.0, t_end = 1.0,
                                limits_mode = :interpolate, interpolation_mode = :rotation_scale,
                                name = "", resolution = RESOLUTION, color = false, alpha = false,
                                overlays = Overlay[])

Render `frames` evenly spaced interpolation steps from `left` to `right` (see
[`interpolate_ifs`](@ref)) with `method` (any [`RenderMethod`](@ref)), as numbered PNGs
`basename_0001.png`, `basename_0002.png`, ... and return
`(outdir, paths, ts)`. Both systems need the same number of maps. `outdir = nothing` writes to
an auto-named directory under `output_dir("frames")`. Each frame's render record mirrors it under
the data directory (`renders/frames/<sequence>/`, see [`render_record_path`](@ref)). Feed the frames to
[`export_animation`](@ref).
"""
function render_interpolation_frames(
    left::IFS,
    right::IFS,
    method::RenderMethod = Transformations();
    frames::Integer,
    outdir::Union{Nothing,AbstractString} = nothing,
    basename::AbstractString = "frame",
    t_start::Real = 0.0,
    t_end::Real = 1.0,
    limits_mode::Symbol = :interpolate,
    interpolation_mode::Union{Symbol,AbstractString} = :rotation_scale,
    name::AbstractString = "",
    resolution = RESOLUTION,
    color::Bool = false,
    alpha::Bool = false,
    overlays = Overlay[],
)
    frames > 0 || throw(ArgumentError("frames must be > 0, got $frames"))
    !isempty(strip(basename)) || throw(ArgumentError("basename must be non-empty"))
    0 <= t_start <= 1 || throw(ArgumentError("t_start must be in [0, 1], got $t_start"))
    0 <= t_end <= 1 || throw(ArgumentError("t_end must be in [0, 1], got $t_end"))

    sequence_name = left.name == right.name ? left.name : string(left.name, "_to_", right.name)
    outdir = _resolve_output_path(
        outdir,
        joinpath(
            "frames",
            savename(
                _savename_prefix(sequence_name) * "_" * method_tag(method),
                Dict("frames" => frames, "t_start" => t_start, "t_end" => t_end),
            ),
        ),
    )
    mkpath(outdir)
    ts = collect(
        Float64,
        frames == 1 ? [Float64(t_start)] : LinRange(Float64(t_start), Float64(t_end), frames),
    )
    paths = String[]
    for (i, t) in enumerate(ts)
        ifs = interpolate_ifs(left, right, t; name, limits_mode, interpolation_mode)
        r = Render(ifs, method; resolution, color, alpha, overlays)
        path = joinpath(outdir, @sprintf("%s_%04d.png", basename, i))
        push!(paths, _render(r, path, render_record_path(path)).outpath)
    end
    return (outdir = String(outdir), paths = paths, ts = ts)
end

function _animation_frame_pattern(basename::AbstractString)
    stripped = strip(String(basename))
    isempty(stripped) && throw(ArgumentError("basename must be non-empty"))
    return Regex("^" * escape_string(stripped) * "_\\d{4}\\.png\$")
end

function _animation_frame_paths(frames_dir::AbstractString, basename::AbstractString)
    isdir(frames_dir) || throw(ArgumentError("frames_dir '$frames_dir' does not exist"))
    pattern = _animation_frame_pattern(basename)
    matches = sort(filter(name -> occursin(pattern, name), readdir(frames_dir)))
    isempty(matches) && throw(
        ArgumentError(
            "No PNG frames matching '$(strip(String(basename)))_####.png' were found in '$frames_dir'",
        ),
    )
    first(matches) == "$(strip(String(basename)))_0001.png" || throw(
        ArgumentError(
            "Frame sequence in '$frames_dir' must start at '$(strip(String(basename)))_0001.png'",
        ),
    )
    return joinpath.(Ref(String(frames_dir)), matches)
end

function _resolve_ffmpeg(ffmpeg_cmd::AbstractString = "ffmpeg")
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
    export_animation(format; frames_dir, basename, outpath=nothing, fps=12, ffmpeg_cmd="ffmpeg")

Convert a deterministic numbered PNG frame sequence such as `basename_0001.png`
into a GIF or MP4 animation using `ffmpeg`.

Relative `frames_dir`/`outpath` resolve under `output_dir()`; `outpath=nothing` writes
`animations/<basename>.<format>`. An existing animation is overwritten.
"""
function export_animation(
    format::Symbol;
    frames_dir::AbstractString,
    basename::AbstractString,
    outpath::Union{Nothing,AbstractString} = nothing,
    fps::Real = 12,
    ffmpeg_cmd::AbstractString = "ffmpeg",
)
    format in (:gif, :mp4) ||
        throw(ArgumentError("Invalid animation format '$format'. Supported: :gif, :mp4"))
    fps > 0 || throw(ArgumentError("fps must be > 0, got $fps"))

    resolved_frames_dir = _resolve_output_path(frames_dir, "frames")
    frame_paths = _animation_frame_paths(resolved_frames_dir, basename)
    ffmpeg = _resolve_ffmpeg(ffmpeg_cmd)

    final_outpath =
        _resolve_output_path(outpath, joinpath("animations", "$(strip(String(basename))).$format"))
    input_pattern = joinpath(resolved_frames_dir, "$(strip(String(basename)))_%04d.png")

    # ffmpeg writes into a scratch directory; the finished file is then moved into place,
    # replacing any existing animation.
    mktempdir() do scratch
        scratch_outpath = joinpath(scratch, "animation.$format")
        if format == :gif
            palette_path = joinpath(scratch, "palette.png")
            _run_ffmpeg(
                `$ffmpeg -y -framerate $fps -i $input_pattern -vf palettegen -frames:v 1 $palette_path`,
            )
            _run_ffmpeg(
                `$ffmpeg -y -framerate $fps -i $input_pattern -i $palette_path -lavfi paletteuse $scratch_outpath`,
            )
        else
            _run_ffmpeg(
                `$ffmpeg -y -framerate $fps -i $input_pattern -c:v libx264 -pix_fmt yuv420p $scratch_outpath`,
            )
        end
        mkpath(dirname(final_outpath))
        mv(scratch_outpath, final_outpath; force = true)
    end

    return (
        format = format,
        frames_dir = resolved_frames_dir,
        basename = strip(String(basename)),
        frame_paths = frame_paths,
        fps = Float64(fps),
        outpath = final_outpath,
    )
end
