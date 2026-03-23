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
            _get_limits(maps, weights)
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
                                            initial_polygon=get(kwargs, :initial_polygon, :default),
                                            color=get(kwargs, :color, false),
                                            axis=get(kwargs, :axis, false))
        save(path, img)
        push!(paths, path)
    end

    return (
        outdir=String(outdir),
        paths=paths,
        ts=ts,
        render_method=_render_method_symbol(parsed_method),
    )
end
